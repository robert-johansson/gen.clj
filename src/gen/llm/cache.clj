(ns gen.llm.cache
  "The LM protocol's cache over a Branching engine (gen.llm.lm): what the
   engine has read, in a trie of token prefixes (gen.llm.trie), and every
   live branch, each the entry of its context.

   - Content-addressed. A context's log-probabilities are a live branch's
     (a hit), or are read: from the deepest live branch at a prefix of it,
     forked so that it stays, or from nothing. The new branch is the
     context's entry. Every token read keeps its score on the trie's path,
     and a live branch's log-probabilities score the token after it, so
     scoring a continuation along what was read reads nothing.
   - The cache owns the branches; callers hold tokens. A branch is freed by
     dispose!, when the cache evicts it, and never dropped, so the engine's
     GC backstop (gen.llm.slots) never runs on the cache's account.
   - Bounded: at most :max-branches live branches and :max-cells KV cells
     (gen.llm.cells), and the trie holds no more tokens than the cells.
     Eviction is least-recently-used: the entry used longest ago, or, when
     cells are short, the one used longest ago among those whose disposal
     frees cells. A read's sources are never evicted for it; when branches
     are short even so, a read consumes its source instead of forking it.
   - Deterministic: what the cache does is a function of its calls. A call
     answers its contexts in order, reading several tokens from a prefix as
     one extend and gathering the one-token reads into one advance (more
     only when one does not fit). On llama.cpp a read's results depend on
     its batch's shape and where its cells land, so this is what makes a
     program's results repeat bit for bit.

   Not thread-safe: gen.llm.lm serializes the calls."
  (:require [gen.llm.branching :as br]
            [gen.llm.cells :as cells]
            [gen.llm.trie :as trie]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn cache
  "A cache over engine's branches, holding at most :max-branches of them
   (default: every branch the engine has free) and :max-cells KV cells
   (default: the engine's :n-ctx, all of llama.cpp's unified cache)."
  [engine {:keys [max-branches max-cells]}]
  (let [{:keys [n-vocab n-ctx live] engine-branches :max-branches} (br/info engine)
        free (- (long engine-branches) (long live))
        max-branches (or max-branches free)
        max-cells (or max-cells n-ctx)]
    (when-not (and (integer? max-branches) (<= 1 (long max-branches) free))
      (throw (ex-info (str "lm: :max-branches is 1 to the engine's " free " free branches, not " (pr-str max-branches))
                      {:reason :invalid-option :max-branches max-branches :free free})))
    (when-not (pos-int? max-cells)
      (throw (ex-info (str "lm: :max-cells is a positive integer, not " (pr-str max-cells))
                      {:reason :invalid-option :max-cells max-cells})))
    {:engine engine
     :n-vocab (long n-vocab)
     :limit (min (long n-ctx) (long max-cells))
     :max-branches (long max-branches)
     :max-cells (long max-cells)
     :state (volatile! {:trie trie/empty-trie
                        :live {}
                        :cells cells/empty-ledger
                        :tick 0
                        :closed? false
                        :counts {:requests 0 :hits 0 :reads 0 :read-tokens 0 :forks 0 :evictions 0}})}))

;; ---------------------------------------------------------------------------
;; The state: :trie, what was read, whose entries are the ids of the live
;; branches' entries; :live {id {:id :branch :segment (its cells' state)
;; :used (the tick of its last use)}}, an entry's id the tick it was made;
;; :cells the ledger; :tick; :closed?; :counts. Its maps are keyed by ids
;; and tokens, never by branches, which hash by identity: walking them is
;; deterministic.
;; ---------------------------------------------------------------------------

(defn- state [c] @(:state c))

(defn- change!
  ([c f] (vswap! (:state c) f))
  ([c f x] (vswap! (:state c) f x))
  ([c f x y] (vswap! (:state c) f x y)))

(defn- tally [s k n]
  (update s :counts update k + n))

(defn- add-entry
  "s with branch b, which read its tokens from position from on with scores,
   as the entry of its context, its state a new segment of n cells after
   segment parent."
  [s b from scores parent n]
  (let [[ledger segment] (cells/allocate (:cells s) parent n)
        id (:tick s)]
    (-> s
        (assoc :cells ledger :tick (inc (long id)))
        (update :trie trie/insert (br/tokens b) from scores id)
        (assoc-in [:live id] {:id id :branch b :segment segment :used id}))))

(defn- drop-entry
  "s without entry e, whose branch is consumed or disposed."
  [s e]
  (-> s
      (update :trie trie/remove-entry (br/tokens (:branch e)))
      (update :live dissoc (:id e))
      (update :cells cells/release (:segment e))))

(defn- touch
  "s with entry e used now."
  [s e]
  (let [tick (:tick s)]
    (-> s
        (assoc :tick (inc (long tick)))
        (assoc-in [:live (:id e) :used] tick))))

(defn- forget-dead
  "s without the entries whose branches the engine no longer holds (it frees
   those a failed read could not restore, and all when it closes)."
  [s]
  (reduce (fn [s e] (if (br/live? (:branch e)) s (drop-entry s e)))
          s (vals (:live s))))

;; ---------------------------------------------------------------------------
;; Room
;; ---------------------------------------------------------------------------

(defn- free-slots ^long [c s]
  (- (long (:max-branches c)) (count (:live s))))

(defn- free-cells ^long [c s]
  (- (long (:max-cells c)) (cells/used (:cells s))))

(defn- victim
  "The entry to evict: the least recently used one whose branch is not
   pinned, among those whose disposal frees cells when cells?; nil if
   there is none."
  [s pinned cells?]
  (let [ledger (:cells s)]
    (reduce-kv (fn [best _ e]
                 (if (and (not (contains? pinned (:branch e)))
                          (or (nil? best) (< (long (:used e)) (long (:used best))))
                          (or (not cells?) (pos? (cells/freed ledger (:segment e)))))
                   e
                   best))
               nil (:live s))))

(defn- make-room!
  "Evict entries, never those of pinned branches, until slots branches and
   n cells are free or no entry is left to evict, disposing of them in one
   call: the branches still missing. The cells are always had: a context is
   no longer than the cells, and the callers pin only branches at prefixes
   of what they read."
  ^long [c ^long slots ^long n pinned]
  (let [[s victims] (loop [s (state c) victims []]
                      (let [short-cells? (< (free-cells c s) n)
                            e (when (or short-cells? (< (free-slots c s) slots))
                                (victim s pinned short-cells?))]
                        (if e (recur (drop-entry s e) (conj victims e)) [s victims])))]
    (when (< (free-cells c s) n)
      (throw (ex-info (str "lm: no room for " n " KV cells (" (free-cells c s) " of " (:max-cells c)
                           " free at most, the rest held by the read's sources)")
                      {:reason :lm-full :cells n :free (free-cells c s) :max-cells (:max-cells c)})))
    (when (seq victims)
      (br/dispose! (mapv :branch victims))
      (vreset! (:state c) (tally s :evictions (count victims))))
    (max 0 (- slots (free-slots c s)))))

;; ---------------------------------------------------------------------------
;; Reads
;; ---------------------------------------------------------------------------

(defn- reading
  "(f made), an engine call that conj!es onto the volatile made the forks
   it makes (which are not entries). If it throws, the live ones are
   disposed and the exception rethrown."
  [f]
  (let [made (volatile! [])]
    (try
      (f made)
      (catch Throwable t
        (when-let [live (seq (filter br/live? @made))]
          ;; the engine frees them itself if it is closing
          (try (br/dispose! (vec live)) (catch Throwable _)))
        (throw t)))))

(defn- open!
  "A new branch that has read ctx, from nothing: its first token, then the
   others teacher-forced, so that every token after the first is scored."
  [c ctx]
  (let [n (count ctx)]
    (make-room! c 1 n #{})
    (let [b (br/open (:engine c) (subvec ctx 0 1))
          {:keys [branch scores]} (if (== n 1)
                                    {:branch b :scores []}
                                    (reading (fn [made]
                                               (vswap! made conj b)
                                               (br/extend b (subvec ctx 1)))))]
      (change! c #(-> %
                      (add-entry branch 1 scores nil n)
                      (tally :reads (if (== n 1) 1 2))
                      (tally :read-tokens n)))
      branch)))

(defn- extend!
  "A new branch that has read ctx, from entry e at its prefix of length
   depth: a fork of e's branch (e's branch itself when no branch can be
   freed), extended teacher-forced."
  [c ctx e ^long depth]
  (let [source (:branch e)
        n (- (count ctx) depth)
        consume? (pos? (make-room! c 1 n #{source}))
        {:keys [branch scores]} (reading (fn [made]
                                           (let [input (if consume?
                                                         source
                                                         (let [f (br/fork source)] (vswap! made conj f) f))]
                                             (br/extend input (subvec ctx depth)))))]
    (change! c #(cond-> (-> %
                            (add-entry branch depth scores (:segment e) n)
                            (tally :reads 1)
                            (tally :read-tokens n))
                  consume? (drop-entry e)
                  (not consume?) (tally :forks 1)))
    branch))

(defn- inputs!
  "The advance's inputs, one for each of parents, in order: each parent's
   forks, made in one call and conj!ed onto the volatile made, and the
   parent itself last when it is consumed. The maps keyed by branches are
   looked up, never walked."
  [parents consumed made]
  (let [at (reduce (fn [at [i p]] (update at p (fnil conj []) i)) {} (map-indexed vector parents))
        by-index (reduce (fn [by-index p]
                           (let [is (at p)
                                 forks (br/fork p (cond-> (count is) (consumed p) dec))]
                             (vswap! made into forks)
                             (into by-index (map vector is (cond-> forks (consumed p) (conj p))))))
                         {} (distinct parents))]
    (mapv by-index (range (count parents)))))

(defn- step!
  "New branches for queries, [ctx e] pairs where ctx is entry e's context
   and one token more, read in one advance, in order: from forks of the
   entries' branches, except that when branches are short, an entry's
   branch is consumed by its last query instead (the first entries' first)."
  [c queries]
  (let [m (count queries)
        parents (mapv (comp :branch second) queries)
        tokens (mapv (comp peek first) queries)
        consumed (set (take (make-room! c m m (set parents)) (distinct parents)))
        entry-of (into {} (map (fn [[_ e]] [(:branch e) e])) queries)
        read (reading #(br/advance (inputs! parents consumed %) tokens))]
    (change! c (fn [s]
                 (let [s (reduce (fn [s [[ctx e] b t]]
                                   (add-entry s b (dec (count ctx)) [(br/logprob (:branch e) t)] (:segment e) 1))
                                 s (map vector queries read tokens))]
                   (-> (reduce drop-entry s (map entry-of (filter consumed (distinct parents))))
                       (tally :reads 1)
                       (tally :read-tokens m)
                       (tally :forks (- m (count consumed)))))))
    read))

(defn- fits?
  "Can one advance read queries? It needs a branch for each, which it has
   by consuming every entry it reads from if need be (so at most
   :max-branches), and a cell for each, with those entries kept: free now,
   or once every other entry is evicted."
  [c queries]
  (let [s (state c)
        m (count queries)]
    (and (<= m (long (:max-branches c)))
         (or (<= m (free-cells c s))
             (let [parents (into #{} (map (comp :branch second)) queries)
                   ledger (reduce-kv (fn [ledger _ e]
                                       (if (contains? parents (:branch e))
                                         ledger
                                         (cells/release ledger (:segment e))))
                                     (:cells s) (:live s))]
               (<= m (- (long (:max-cells c)) (cells/used ledger))))))))

(defn- fitting
  "The longest prefix of queries, halving, that one advance can read (one
   query always can)."
  [c queries]
  (loop [k (count queries)]
    (let [chunk (subvec queries 0 k)]
      (if (or (== k 1) (fits? c chunk)) chunk (recur (quot k 2))))))

(defn- resolve-in
  "How s answers ctx: [kind entry depth], kind :hit (an entry at ctx),
   :step (one at ctx less its last token), :extend (one at a shorter
   prefix, of length depth) or :open (none)."
  [s ctx]
  (let [{:keys [entry entry-depth]} (trie/match (:trie s) ctx)
        n (count ctx)
        d (long entry-depth)
        e (when entry (get (:live s) entry))]
    (cond (nil? e) [:open nil 0]
          (== d n) [:hit e d]
          (== d (dec n)) [:step e d]
          :else [:extend e d])))

(defn- answer-round!
  "One round of answers for the contexts at indices todo, in order: hits
   and longer reads at once, one-token reads gathered. [out steps read?]:
   out index → branch, steps [index entry] pairs, read? whether a read
   was made (and so may have evicted a step's entry)."
  [c contexts todo out]
  (reduce (fn [[out steps read?] i]
            (let [ctx (contexts i)
                  [kind e d] (resolve-in (state c) ctx)]
              (when e (change! c touch e))
              (case kind
                :hit (do (change! c tally :hits 1) [(assoc out i (:branch e)) steps read?])
                :step [out (conj steps [i e]) read?]
                :extend [(assoc out i (extend! c ctx e d)) steps true]
                :open [(assoc out i (open! c ctx)) steps true])))
          [out [] false] todo))

(defn- still-steps
  "The steps whose contexts are still one token past an entry, with it."
  [c contexts steps]
  (into [] (keep (fn [[i]]
                   (let [[kind e] (resolve-in (state c) (contexts i))]
                     (when (= :step kind) [i e]))))
        steps))

(defn- firsts
  "The first of steps reading each token after each entry, in order."
  [key-of steps]
  (first (reduce (fn [[out seen] q]
                   (let [k (key-of q)]
                     (if (contains? seen k) [out seen] [(conj out q) (conj seen k)])))
                 [[] #{}] steps)))

(defn- answer!
  "Branches with the log-probabilities of contexts, in order. Each round
   answers the contexts left, in order, and reads the one-token reads that
   fit in one advance, each context once (one asked again after its read
   is a hit); a read may evict the entry a one-token read found before it,
   which then waits for the next round."
  [c contexts]
  (loop [todo (vec (range (count contexts))) out {}]
    (if (empty? todo)
      (mapv out (range (count contexts)))
      (let [[out steps read?] (answer-round! c contexts todo out)
            ready (if read? (still-steps c contexts steps) steps)
            key-of (fn [[i e]] [(:id e) (peek (contexts i))])
            distinct-ready (firsts key-of ready)
            chunk (when (seq distinct-ready)
                    (fitting c (mapv (fn [[i e]] [(contexts i) e]) distinct-ready)))
            read (zipmap (map key-of distinct-ready) (when chunk (step! c chunk)))
            answered (filterv #(contains? read (key-of %)) ready)
            out (reduce (fn [out q] (assoc out (first q) (read (key-of q)))) out answered)]
        (change! c tally :hits (- (count answered) (count read)))
        (recur (into [] (comp (map first) (remove #(contains? out %))) steps) out)))))

;; ---------------------------------------------------------------------------
;; The cache's operations (gen.llm.lm checks their arguments' shapes)
;; ---------------------------------------------------------------------------

(defn- check-open [c]
  (when (:closed? (state c))
    (throw (ex-info "lm: the LM is closed" {:reason :closed}))))

(defn- token?
  "Is t a token of a vocabulary of n tokens? (Quick for a long; a big
   integer is out of range, not an error.)"
  [t ^long n]
  (if (instance? Long t)
    (< -1 (.longValue ^Long t) n)
    (and (integer? t) (not (neg? (compare t 0))) (neg? (compare t n)))))

(defn- check-context
  "ctx fits in the cache, and its tokens are tokens of the vocabulary."
  [c ctx]
  (let [n (count ctx)
        limit (long (:limit c))
        n-vocab (long (:n-vocab c))]
    (when (< limit n)
      (throw (ex-info (str "lm: a context of " n " tokens exceeds the " limit " a branch can hold"
                           " (the engine's :n-ctx and the LM's :max-cells)")
                      {:reason :context-full :tokens n :limit limit})))
    (dotimes [i n]
      (let [t (nth ctx i)]
        (when-not (token? t n-vocab)
          (throw (ex-info (str "lm: not a token of the vocabulary (0 to " (dec n-vocab) "): " (pr-str t))
                          {:reason :invalid-token :token t :n-vocab n-vocab})))))))

(defn- guarded
  "(f); if it throws, the entries the engine no longer holds are forgotten
   before the exception goes on."
  [c f]
  (try
    (f)
    (catch Throwable t
      (change! c forget-dead)
      (throw t))))

(defn next-logprobs!
  "The next-token log-probabilities after each of contexts (non-empty token
   vectors), in order: new float arrays."
  [c contexts]
  (check-open c)
  (run! #(check-context c %) contexts)
  (change! c tally :requests (count contexts))
  (mapv br/logprobs (guarded c #(answer! c contexts))))

(defn- known-scores
  "The scores of whole from position from on, if s has them: whole on a
   path, or whole less its last token on a path to an entry, whose
   log-probabilities score that token."
  [s whole from]
  (or (trie/scores (:trie s) whole from)
      (let [ctx (pop whole)
            {:keys [entry entry-depth]} (trie/match (:trie s) ctx)]
        (when (and entry (== (long entry-depth) (count ctx)))
          (conj (trie/scores (:trie s) ctx from)
                (br/logprob (:branch (get (:live s) entry)) (peek whole)))))))

(defn score!
  "The log-probability of each token of continuation given ctx and the
   tokens before it, a vector of doubles: from the trie when it has them,
   else read (and then kept)."
  [c ctx continuation]
  (check-open c)
  (if (empty? continuation)
    []
    (let [whole (into ctx continuation)
          from (count ctx)]
      (check-context c whole)
      (change! c tally :requests 1)
      (if-let [scores (known-scores (state c) whole from)]
        (do (change! c tally :hits 1) scores)
        (do (guarded c #(answer! c [whole]))
            (trie/scores (:trie (state c)) whole from))))))

(defn info
  "Sizes and counts: :live branches, :cells, :tokens in the trie, the
   bounds, and since the cache was made: :requests (contexts asked for),
   :hits (answered with no read), :reads (engine reads: opens, extends,
   advances), :read-tokens, :forks, :evictions."
  [c]
  (let [s (state c)]
    (assoc (:counts s)
           :live (count (:live s))
           :cells (cells/used (:cells s))
           :tokens (trie/size (:trie s))
           :max-branches (:max-branches c)
           :max-cells (:max-cells c)
           :closed? (:closed? s))))

(defn close!
  "Dispose of every branch, and refuse calls from now on."
  [c]
  (when-not (:closed? (state c))
    (let [live (filterv br/live? (map :branch (vals (:live (state c)))))]
      (change! c #(assoc % :closed? true :trie trie/empty-trie :live {} :cells cells/empty-ledger))
      (when (seq live) (br/dispose! live)))))
