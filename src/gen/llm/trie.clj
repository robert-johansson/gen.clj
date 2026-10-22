(ns gen.llm.trie
  "A persistent radix trie of token sequences: the LM protocol's record of
   what its engine has read (gen.llm.cache). Each token on a path carries
   its score, the log-probability it was read with given the tokens before
   it (NaN for a sequence's first token, which has none). A node may hold
   an entry, a value for its context (for the cache, the id of the live
   branch at that context).

     (match trie tokens)                    → {:depth :entry :entry-depth}
     (scores trie tokens from)              → the scores of tokens from position from on, or nil
     (insert trie tokens from scores entry) → trie with tokens's path and an entry at its end
     (remove-entry trie tokens)             → trie without the entry at tokens
     (entries trie), (size trie)

   Paths are compressed: a node stands at each entry and at each token
   where paths part. A path stays only while it leads to an entry, so the
   trie holds no more tokens than its entries' contexts. Tokens compare with
   =, so a context may hold longs or integers. remove-entry takes the
   tokens of an entry the trie holds (:no-entry otherwise).")

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; A node: {:tokens the edge's tokens from the parent (a vector; [] at the
;; root), :scores theirs (doubles), :children {first token of an edge →
;; node}, and :entry when it has one}.

(def empty-trie
  "The trie of no sequences."
  {:tokens [] :scores (vector-of :double) :children {}})

(defn- same-token?
  "(= a b), quicker for the same object and for longs."
  [a b]
  (or (identical? a b)
      (if (and (instance? Long a) (instance? Long b))
        (== (.longValue ^Long a) (.longValue ^Long b))
        (= a b))))

(defn- common
  "How many of edge's first tokens equal those of tokens from position at."
  ^long [edge tokens ^long at]
  (let [n (min (count edge) (- (count tokens) at))]
    (loop [i 0]
      (if (and (< i n) (same-token? (nth edge i) (nth tokens (+ at i))))
        (recur (inc i))
        i))))

(defn match
  "How far tokens follow the trie: {:depth the number of tokens on a path,
   :entry the entry at the longest prefix of tokens that has one (nil if
   none does), :entry-depth that prefix's length (0 if none)}."
  [trie tokens]
  (let [n (count tokens)]
    (loop [node trie depth 0 entry nil entry-depth 0]
      (let [here (:entry node)
            entry (if here here entry)
            entry-depth (if here depth entry-depth)
            child (when (< depth n) (get (:children node) (nth tokens depth)))]
        (if child
          (let [edge (:tokens child)
                k (common edge tokens depth)]
            (if (== k (count edge))
              (recur child (+ depth k) entry entry-depth)
              {:depth (+ depth k) :entry entry :entry-depth entry-depth}))
          {:depth depth :entry entry :entry-depth entry-depth})))))

(defn scores
  "The scores of tokens from position from on, a vector of doubles, when
   tokens lie on a path of the trie; nil otherwise."
  [trie tokens ^long from]
  (let [n (count tokens)]
    (loop [node trie depth 0 out (transient [])]
      (if (== depth n)
        (persistent! out)
        (when-let [child (get (:children node) (nth tokens depth))]
          (let [edge (:tokens child)
                k (common edge tokens depth)
                sc (:scores child)
                out (loop [i (max 0 (- from depth)) out out]
                      (if (< i k) (recur (inc i) (conj! out (nth sc i))) out))]
            (cond (== k (count edge)) (recur child (+ depth k) out)
                  (== (+ depth k) n) (persistent! out))))))))

(defn- edge-scores
  "The scores at positions start to end of a sequence whose positions from
   on scored scores: NaN before from."
  [scores ^long from ^long start ^long end]
  (into (vector-of :double)
        (map (fn [^long p] (if (<= from p) (double (nth scores (- p from))) Double/NaN)))
        (range start end)))

(defn- leaf [tokens depth from scores entry]
  {:tokens (into [] (subvec tokens depth))
   :scores (edge-scores scores from depth (count tokens))
   :children {}
   :entry entry})

(defn- split
  "node with its edge cut after k tokens: a node of the first k, whose one
   child is node with the rest."
  [node ^long k]
  (let [{edge :tokens sc :scores} node]
    {:tokens (subvec edge 0 k)
     :scores (subvec sc 0 k)
     :children {(nth edge k) (assoc node :tokens (subvec edge k) :scores (subvec sc k))}}))

(defn- insert* [node tokens depth from scores entry]
  (if (== (long depth) (count tokens))
    (if (:entry node)
      (throw (ex-info (str "trie: an entry is already at " tokens) {:reason :entry-exists :tokens tokens}))
      (assoc node :entry entry))
    (let [t (nth tokens depth)
          child (get (:children node) t)]
      (assoc-in node [:children t]
                (if child
                  (let [k (common (:tokens child) tokens depth)]
                    (insert* (if (< k (count (:tokens child))) (split child k) child)
                             tokens (+ (long depth) k) from scores entry))
                  (leaf tokens depth from scores entry))))))

(defn insert
  "trie with the path of tokens and entry at its end, which must have none.
   scores are the scores of tokens from position from on; tokens already on
   a path keep the scores they have."
  [trie tokens from scores entry]
  (insert* trie tokens 0 from scores entry))

(defn- tidy
  "A node below the root, after a removal at or under it: nil if it leads
   to no entry, joined to its child if it only passes through to one."
  [node]
  (let [children (:children node)]
    (cond
      (:entry node) node
      (empty? children) nil
      (== 1 (count children)) (let [child (val (first children))]
                                (assoc child
                                       :tokens (into (vec (:tokens node)) (:tokens child))
                                       :scores (into (into (vector-of :double) (:scores node)) (:scores child))))
      :else node)))

(defn- entry-path
  "The nodes from the root down to the entry at tokens, each with the
   token that leads to the next ([node nil] last), or :no-entry when the
   trie holds no entry there (a path it does not hold, one that ends or
   parts inside an edge, a node without an entry)."
  [trie tokens]
  (let [n (count tokens)
        no-entry #(ex-info (str "remove-entry: the trie holds no entry at " (pr-str tokens)) {:reason :no-entry :tokens tokens})]
    (loop [node trie depth 0 path []]
      (if (== depth n)
        (if (contains? node :entry) (conj path [node nil]) (throw (no-entry)))
        (let [t (nth tokens depth)
              child (get (:children node) t)
              end (+ depth (count (:tokens child)))]
          (if (and child (<= end n) (= (:tokens child) (subvec tokens depth end)))
            (recur child end (conj path [node t]))
            (throw (no-entry))))))))

(defn remove-entry
  "trie without the entry at tokens, and without the path to it if that
   leads to no other entry. A loop, not a recursion: a path may pass as
   many nodes as its tokens."
  [trie tokens]
  (let [path (entry-path trie (vec tokens))]
    (reduce (fn [below [node t]]
              (if-let [child (tidy below)]
                (assoc-in node [:children t] child)
                (update node :children dissoc t)))
            (dissoc (first (peek path)) :entry)
            (rseq (pop path)))))

(defn- reduce-entries
  "(f acc entry) over the trie's entries, depth first, as reduce."
  [f init trie]
  (letfn [(walk [acc node]
            (let [acc (if-let [e (:entry node)] (f acc e) acc)]
              (reduce-kv (fn [acc _ child] (walk acc child)) acc (:children node))))]
    (walk init trie)))

(defn entries
  "The trie's entries, depth first: a vector."
  [trie]
  (reduce-entries conj [] trie))

(defn size
  "The number of tokens the trie holds."
  ^long [trie]
  (letfn [(walk [^long n node]
            (reduce-kv (fn [^long n _ child] (walk (+ n (count (:tokens child))) child)) n (:children node)))]
    (walk 0 trie)))
