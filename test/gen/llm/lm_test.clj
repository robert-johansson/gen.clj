(ns gen.llm.lm-test
  "The LM protocol (gen.llm.lm, over gen.llm.cache) on the pure-JVM toy
   engines: gen.llm.toy-engine, and gen.llm.cell-engine, whose KV cells
   are llama.cpp's and whose :noise makes results depend on how a context
   is read, as llama.cpp's do.

   The laws, over random programs (moves over the contexts asked before,
   as inference asks: children alone or together, prefixes, continuations)
   and random bounds:
   - every answer is the model's, exactly, whatever the cache held: a
     cached context's log-probabilities are a fresh read's, a batch's are
     its contexts' one by one, and eviction changes costs, never results;
   - the same calls give the same results, bit for bit, on an engine whose
     results depend on the reads' shapes and the cells' places, from one
     thread or many;
   - the books stay straight: branches and cells within the bounds, the
     LM's counts the engine's (cells counted physically), the trie no
     larger than the cells, and never a refused read or a GC backstop
     collection; also when reads fail;
   - misuse is loud and changes nothing.

   Mutations of gen.llm.cache checked (2026-10-20, each applied, the suite
   run, and reverted), with the tests that caught them:
   - evicting the most recently used, or recording no use for hits:
     eviction-is-least-recently-used;
   - evicting without regard to which branches free cells:
     short-of-cells-the-cache-evicts-what-frees-them;
   - an advance's rows in rotation, or a batch's answers misassigned:
     every-answer-is-the-models, a-batch-answers..., and more;
   - a one-token read's score one position off: a-continuation-on-a-path...,
     the-same-calls...;
   - a one-token read's cell not counted: the books (the engine's cells,
     :kv-full) in every property;
   - a read's source not pinned, a consumed entry kept, victims not
     disposed of, a round's one-token reads not resolved again after a
     read, or not deduplicated: every property (a branch not live, the
     engine's run out, or a second entry at one context);
   - fits? ignoring cells: inference-loops-never-reach-the-backstop;
   - the consumed entries chosen in a set's (identity-hash) order:
     the-same-calls-give-the-same-results (four runs of four; random
     contexts never made it arise, programs of moves over asked contexts
     do);
   - the engine's freed branches not forgotten:
     a-closed-engine-is-loud-and-forgotten;
   - a failed read's forks kept: a-failed-read-leaves-the-books-straight.
   And of 2026-10-21 (review-lm):
   - a held entry's row scoring the token after it before the path's first
     read: a-score-is-the-first-reads (the exact engines cannot tell the
     two apart)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.cache :as cache]
            [gen.llm.cell-engine :as ce]
            [gen.llm.lm :as lm]
            [gen.llm.lm-contract :as contract]
            [gen.llm.toy-engine :as toy]
            [gen.llm.toy-tokenizer :as tt]))

(def ^:private n-vocab 97)

(defn- exact-row ^floats [ctx]
  (toy/toy-logprobs n-vocab ctx))

(defn- exact-scores [ctx cont]
  (mapv (fn [i] (double (aget (exact-row (into ctx (subvec cont 0 i))) (int (cont i))))) (range (count cont))))

(defn- same-row? [^floats a ^floats b]
  (java.util.Arrays/equals a b))

(defn- run-on
  "(f m e) for a new LM m over a new engine e: the cell engine sized to the
   LM's bounds (so a read past them is refused, loudly), or with :toy the
   toy engine."
  [{:keys [engine max-branches max-cells] :as opts} f]
  (with-open [e (if (= engine :toy)
                  (toy/engine {:max-branches max-branches})
                  (ce/engine (merge {:n-vocab n-vocab} (select-keys opts [:max-branches :max-cells :noise :fail-reads]))))
              m (lm/lm e (tt/tokenizer n-vocab) (select-keys opts [:max-branches :max-cells]))]
    (f m e)))

(defn- books
  "What is wrong with the LM's books: its bounds, and its counts against
   the engine's."
  [m e {:keys [engine max-branches max-cells]}]
  (let [{:keys [live cells tokens]} (lm/info m)
        info (br/info e)
        physical (when-not (= engine :toy) (ce/counts e))]
    (cond-> []
      (< (long max-branches) (long live)) (conj [:branches live])
      (< (long max-cells) (long cells)) (conj [:cells cells])
      (< (long cells) (long tokens)) (conj [:tokens tokens :cells cells])
      (not= live (:live info)) (conj [:engine-live live (:live info)])
      (pos? (:collections info)) (conj [:collections (:collections info)])
      (and physical (not= cells (:cells physical))) (conj [:engine-cells cells (:cells physical)])
      (and physical (pos? (:kv-full physical))) (conj [:kv-full (:kv-full physical)]))))

;; ---------------------------------------------------------------------------
;; The contract (gen.llm.lm-contract), exactly
;; ---------------------------------------------------------------------------

(def ^:private contract-opts
  {:prompt [95 40 73 1 2] :continuation [3 0 2 1 1 3] :k 4 :band 0.0})

(defn- toy-setup [engine-opts]
  (let [e (toy/engine (merge {:max-branches 12} engine-opts))]
    {:engine e
     :lm #(lm/lm e (tt/tokenizer n-vocab) {:max-branches 8})
     :fresh (fn [] (let [e (toy/engine (merge {:max-branches 12} engine-opts))]
                     [e (lm/lm e (tt/tokenizer n-vocab) {:max-branches 8})]))}))

(deftest the-contract-holds-exactly
  (doseq [setup [(toy-setup {})
                 (let [e (ce/engine {:max-branches 12 :max-cells 200})]
                   {:engine e
                    :lm #(lm/lm e (tt/tokenizer n-vocab) {:max-branches 8 :max-cells 150})
                    :fresh (fn [] (let [e (ce/engine {:max-branches 12 :max-cells 200 :noise 1e-3})]
                                    [e (lm/lm e (tt/tokenizer n-vocab) {:max-branches 8 :max-cells 150})]))})]]
    (with-open [^java.io.Closeable e (:engine setup)]
      (doseq [[k check] contract/checks]
        (testing k
          (is (= [] (contract/run-check check setup contract-opts)))))
      (is (zero? (:live (br/info e))) "every check closes what it makes"))))

(deftest planted-engine-bugs-fail-their-checks
  ;; the contract's power: each mutation of the toy engine's Slots is caught
  (doseq [[bug caught-by] {:copy-perturbs [:cached-matches-fresh]
                           :step-rotates [:cached-matches-fresh :batch-matches-single]
                           :step-ignores [:cached-matches-fresh :score-matches-steps]
                           :run-shifts [:score-matches-steps]
                           :run-stale [:cached-matches-fresh :score-matches-steps]}]
    (let [setup (toy-setup {:bug bug})]
      (with-open [^java.io.Closeable _ (:engine setup)]
        (doseq [k caught-by]
          (testing [bug k]
            (is (seq (contract/run-check (contract/checks k) setup contract-opts)))))))))

;; ---------------------------------------------------------------------------
;; Random programs
;; ---------------------------------------------------------------------------

(def ^:private gen-context
  ;; a few starts and short tails over four tokens: contexts share prefixes
  (gen/let [start (gen/elements [[95 40 73] [95 40] [95 41 7] [3]])
            tail (gen/vector (gen/choose 0 3) 0 7)]
    (into start tail)))

(def ^:private gen-move
  ;; moves over the contexts asked before (by index, modulo their number),
  ;; as inference asks: children of them alone or together (a token SMC
  ;; step), their prefixes (an MH regeneration), scores of continuations
  (gen/frequency [[2 (gen/fmap #(vector :fresh %) gen-context)]
                  [4 (gen/tuple (gen/return :child) gen/nat (gen/choose 0 3))]
                  [3 (gen/fmap #(vector :children %) (gen/vector (gen/tuple gen/nat (gen/choose 0 3)) 2 6))]
                  [1 (gen/fmap #(vector :batch %) (gen/vector gen-context 1 6))]
                  [2 (gen/tuple (gen/return :back) gen/nat gen/nat)]
                  [2 (gen/tuple (gen/return :score) gen/nat (gen/vector (gen/choose 0 3) 0 4))]]))

(defn- realize
  "The calls of moves: [:next ctx], [:batch ctxs] or [:score ctx cont],
   contexts at most 10 tokens long."
  [moves]
  (let [child (fn [ctx t] (conj (subvec ctx 0 (min 9 (count ctx))) t))
        step (fn [[asked calls] [kind x y]]
               (let [at #(asked (mod % (count asked)))
                     c (case kind
                         :fresh [:next x]
                         :child [:next (child (at x) y)]
                         :children [:batch (mapv (fn [[i t]] (child (at i) t)) x)]
                         :batch [:batch x]
                         :back (let [ctx (at x)] [:next (subvec ctx 0 (inc (mod y (count ctx))))])
                         :score [:score (at x) y])]
                 [(into asked (if (= :batch (first c)) (second c) [(second c)])) (conj calls c)]))]
    (second (reduce step [[[95 40 73]] []] moves))))

(def ^:private gen-calls (gen/fmap realize (gen/vector gen-move 1 30)))

(def ^:private gen-bounds
  ;; the longest sequence is 14 tokens (a 10-token context, 4 scored)
  (gen/hash-map :max-branches (gen/choose 1 6) :max-cells (gen/choose 14 60)))

(defn- call [m [kind x y]]
  (case kind
    :next (lm/next-logprobs m x)
    :batch (lm/batch-next-logprobs m x)
    :score (lm/score m x y)))

(defn- expected [[kind x y]]
  (case kind
    :next (exact-row x)
    :batch (mapv exact-row x)
    :score (exact-scores x y)))

(defn- same-result? [[kind] a b]
  (case kind
    :next (same-row? a b)
    :batch (and (= (count a) (count b)) (every? true? (map same-row? a b)))
    :score (= a b)))

(defn- answers-exactly? [opts calls]
  (run-on opts (fn [m e]
                 (every? (fn [c] (and (same-result? c (call m c) (expected c))
                                      (empty? (books m e opts))))
                         calls))))

(defspec every-answer-is-the-models 120
  (prop/for-all [bounds gen-bounds
                 engine (gen/elements [:cells :toy])
                 calls gen-calls]
    (answers-exactly? (assoc bounds :engine engine) calls)))

(defspec a-batch-answers-as-its-contexts-one-by-one 80
  ;; after the same warm-up, so that some contexts are one token past held ones
  (prop/for-all [bounds gen-bounds
                 warm (gen/vector gen-context 0 4)
                 contexts (gen/vector gen-context 1 12)]
    (let [contexts (into contexts (map #(conj % 1)) warm)
          run (fn [f] (run-on bounds (fn [m _] (run! #(lm/next-logprobs m %) warm) (f m))))]
      (every? true? (map same-row?
                         (run #(lm/batch-next-logprobs % contexts))
                         (run (fn [m] (mapv #(lm/next-logprobs m %) contexts))))))))

(defn- held
  "The contexts the LM holds branches at, by entry."
  [m]
  (->> (.-cache ^gen.llm.lm.CachedLM m) :state deref :live vals (mapv (comp br/tokens :branch))))

(defspec the-same-calls-give-the-same-results 100
  ;; on an engine whose results depend on how a context is read; the
  ;; contexts held after each call are the same too
  (prop/for-all [bounds gen-bounds
                 calls gen-calls]
    (let [run #(run-on (assoc bounds :noise 1e-3) (fn [m _] (mapv (fn [c] [(call m c) (held m)]) calls)))
          a (run)
          b (run)]
      (and (every? true? (map same-result? calls (map first a) (map first b)))
           (= (map second a) (map second b))))))

(defspec a-failed-read-leaves-the-books-straight 80
  (prop/for-all [bounds gen-bounds
                 calls gen-calls
                 fails (gen/set (gen/choose 0 40) {:max-elements 4})]
    (let [opts (assoc bounds :fail-reads fails)]
      (run-on opts (fn [m e]
                     (every? (fn [c]
                               (let [r (try (call m c)
                                            (catch clojure.lang.ExceptionInfo x
                                              (if (= :decode-failed (:reason (ex-data x))) ::failed (throw x))))]
                                 (and (or (= ::failed r) (same-result? c r (expected c)))
                                      (empty? (books m e opts)))))
                             calls))))))

;; ---------------------------------------------------------------------------
;; Threads
;; ---------------------------------------------------------------------------

(def ^:private program
  (gen/generate (gen/fmap realize (gen/vector gen-move 60)) 30 20261020))

(def ^:private noisy {:max-branches 4 :max-cells 30 :noise 1e-3})

(defn- results [m calls] (mapv (partial call m) calls))

(deftest results-do-not-depend-on-the-calling-threads
  (let [alone (run-on noisy (fn [m _] (results m program)))
        pool (java.util.concurrent.Executors/newFixedThreadPool 4)]
    (try
      (testing "each call from another thread"
        (let [hopped (run-on noisy (fn [m _]
                                     (mapv (fn [c] (.get (.submit pool ^Callable #(call m c)))) program)))]
          (is (every? true? (map same-result? program alone hopped)))))
      (testing "two LMs at once, on their own engines and threads"
        (let [other (vec (reverse program))
              other-alone (run-on noisy (fn [m _] (results m other)))
              a (.submit pool ^Callable #(run-on noisy (fn [m _] (results m program))))
              b (.submit pool ^Callable #(run-on noisy (fn [m _] (results m other))))]
          (is (every? true? (map same-result? program alone (.get a))))
          (is (every? true? (map same-result? other other-alone (.get b))))))
      (finally (.shutdown pool)))))

(deftest the-noisy-engine-tells-reads-apart
  ;; what lets the determinism tests fail: a context read in another
  ;; shape, or into other cells, has other log-probabilities
  (let [ctx [95 40 73 1]
        read (fn [& before] (run-on noisy (fn [m _]
                                            (doseq [c before] (lm/next-logprobs m c))
                                            (lm/next-logprobs m ctx))))
        alone (read [95 40 73])
        batched (run-on noisy (fn [m _]
                                (lm/next-logprobs m [95 40 73])
                                (first (lm/batch-next-logprobs m [ctx [95 40 73 2]]))))]
    (is (not (same-row? alone batched)) "one row of two, not alone")
    (is (not (same-row? alone (read [3 3 3] [95 40 73]))) "other cells")
    (is (same-row? alone (read [95 40 73])) "the same read")))

;; ---------------------------------------------------------------------------
;; The cache's choices
;; ---------------------------------------------------------------------------

(def ^:private prompt [95 40 73])

(defn- counts [m] (select-keys (lm/info m) [:requests :hits :reads :read-tokens :forks :evictions :live]))

(deftest a-fresh-context-is-read-from-its-first-token
  (run-on {:max-branches 4 :max-cells 64}
          (fn [m _]
            (let [lp (lm/next-logprobs m prompt)]
              (is (same-row? (exact-row prompt) lp))
              (is (= {:requests 1 :hits 0 :reads 2 :read-tokens 3 :forks 0 :evictions 0 :live 1} (counts m))
                  "its first token, then the rest teacher-forced, so every later token has a score")
              (aset lp 0 (float 1))
              (is (same-row? (exact-row prompt) (lm/next-logprobs m prompt)) "each answer is a new array")
              (is (= {:requests 2 :hits 1 :reads 2} (select-keys (lm/info m) [:requests :hits :reads])))))))

(deftest one-token-reads-share-one-advance
  (run-on {:max-branches 8 :max-cells 64}
          (fn [m _]
            (lm/next-logprobs m prompt)
            (let [kids (mapv #(conj prompt %) [1 2 3 4 5])
                  rows (lm/batch-next-logprobs m kids)]
              (is (every? true? (map same-row? (map exact-row kids) rows)))
              (is (= {:reads 3 :read-tokens 8 :forks 5 :live 6} (select-keys (lm/info m) [:reads :read-tokens :forks :live]))
                  "one advance of five forks, the prompt kept")
              (lm/next-logprobs m prompt)
              (is (= 3 (:reads (lm/info m))) "the prompt is still held")))))

(deftest a-longer-read-extends-the-deepest-held-prefix
  (run-on {:max-branches 8 :max-cells 64}
          (fn [m _]
            (lm/next-logprobs m prompt)
            (let [ctx (into prompt [1 2 3 0])]
              (is (same-row? (exact-row ctx) (lm/next-logprobs m ctx)))
              (is (= {:reads 3 :read-tokens 7 :forks 1 :live 2} (select-keys (lm/info m) [:reads :read-tokens :forks :live]))
                  "a fork of the prompt, extended by four tokens")))))

(deftest a-continuation-on-a-path-is-scored-from-the-trie
  (run-on {:max-branches 8 :max-cells 64}
          (fn [m _]
            (let [path [2 0 3]]
              (doseq [i (range 4)] (lm/next-logprobs m (into prompt (take i path))))
              (let [reads (:reads (lm/info m))]
                (is (= (exact-scores prompt path) (lm/score m prompt path)))
                (is (= (exact-scores (conj prompt 2) [0 3]) (lm/score m (conj prompt 2) [0 3])))
                (is (= (exact-scores prompt (conj path 1)) (lm/score m prompt (conj path 1)))
                    "the last token scored by the held context before it")
                (is (= (exact-scores prompt [1]) (lm/score m prompt [1])))
                (is (= reads (:reads (lm/info m))) "no read: each token kept the score it was read with"))))))

(deftest a-score-is-the-first-reads
  ;; on the noisy engine: [1 2] scored teacher-forced, then prompt+[1] read
  ;; again as a one-token read: the score of 2 after it stays the first
  ;; read's, though prompt+[1]'s row is now the second read's
  (run-on {:max-branches 8 :max-cells 64 :noise 1e-3}
          (fn [m _]
            (lm/next-logprobs m prompt)
            (let [kept (lm/score m prompt [1 2])
                  row (lm/next-logprobs m (conj prompt 1))]
              (is (= kept (lm/score m prompt [1 2])))
              (is (= [(kept 1)] (lm/score m (conj prompt 1) [2])))
              (is (not= (kept 1) (double (aget row 2))) "the second read differs: the test can tell")))))

(deftest a-scored-sequence-is-kept
  (run-on {:max-branches 8 :max-cells 64}
          (fn [m _]
            (is (= (exact-scores prompt [1 2 3]) (lm/score m prompt [1 2 3])))
            (let [reads (:reads (lm/info m))]
              (is (same-row? (exact-row (into prompt [1 2 3])) (lm/next-logprobs m (into prompt [1 2 3]))))
              (is (= reads (:reads (lm/info m))))
              (is (= [] (lm/score m prompt [])))))))

(deftest eviction-is-least-recently-used
  ;; three branches: the prompt and two children; a third child evicts the
  ;; child used longest ago (the prompt, its source, stays)
  (run-on {:max-branches 3 :max-cells 64}
          (fn [m _]
            (let [[c1 c2 c3] (map #(conj prompt %) [1 2 3])]
              (doseq [c [prompt c1 c2 c3]] (lm/next-logprobs m c))
              (let [{:keys [reads evictions]} (lm/info m)]
                (is (= 1 evictions))
                (lm/next-logprobs m c2)
                (lm/next-logprobs m c3)
                (is (= reads (:reads (lm/info m))) "c2 and c3 are held")
                (lm/next-logprobs m c1)
                (is (= (inc reads) (:reads (lm/info m))) "c1 was evicted")))))
  (testing "a hit is a use"
    (run-on {:max-branches 3 :max-cells 64}
            (fn [m _]
              (let [[c1 c2 c3] (map #(conj prompt %) [1 2 3])]
                (doseq [c [prompt c1 c2 c1 c3]] (lm/next-logprobs m c))
                (let [reads (:reads (lm/info m))]
                  (lm/next-logprobs m c1)
                  (is (= reads (:reads (lm/info m))) "c1 is held: c2 was used longest ago")))))))

(deftest short-of-cells-the-cache-evicts-what-frees-them
  ;; the prompt was used longest ago, but its cells are its child's too:
  ;; evicting it frees none, so the child goes, and the prompt stays
  (run-on {:max-branches 8 :max-cells 12}
          (fn [m _]
            (lm/next-logprobs m prompt)
            (lm/next-logprobs m (into prompt [1 2 3 0 1]))
            (lm/next-logprobs m [3 3 3 3 3])
            (is (= 1 (:evictions (lm/info m))))
            (let [reads (:reads (lm/info m))]
              (lm/next-logprobs m prompt)
              (is (= reads (:reads (lm/info m))) "the prompt is held")))))

(deftest short-of-branches-a-read-consumes-its-source
  (run-on {:max-branches 1 :max-cells 64}
          (fn [m e]
            (doseq [i (range 1 6)]
              (let [ctx (into prompt (range i))]
                (is (same-row? (exact-row ctx) (lm/next-logprobs m ctx)))))
            (is (= {:forks 0 :evictions 0 :live 1} (select-keys (lm/info m) [:forks :evictions :live]))
                "each read advanced the one branch")
            (is (empty? (books m e {:max-branches 1 :max-cells 64}))))))

(deftest a-batch-larger-than-the-branches-is-read-in-parts
  (run-on {:max-branches 3 :max-cells 64}
          (fn [m e]
            (lm/next-logprobs m prompt)
            (let [kids (mapv #(conj prompt %) (range 10))]
              (is (every? true? (map same-row? (map exact-row kids) (lm/batch-next-logprobs m kids))))
              (is (empty? (books m e {:max-branches 3 :max-cells 64})))))))

(deftest repeated-contexts-are-read-once
  (run-on {:max-branches 4 :max-cells 64}
          (fn [m _]
            (let [rows (lm/batch-next-logprobs m [prompt prompt (conj prompt 1) prompt])]
              (is (every? true? (map same-row? (map exact-row [prompt prompt (conj prompt 1) prompt]) rows)))
              (is (not (identical? (first rows) (second rows))) "each answer is its own array")
              (is (= {:requests 4 :hits 2 :reads 3} (select-keys (lm/info m) [:requests :hits :reads])))
              (let [kids [(conj prompt 2) (conj prompt 2) (conj prompt 3)]]
                (is (every? true? (map same-row? (map exact-row kids) (lm/batch-next-logprobs m kids))))
                (is (= {:requests 7 :hits 3 :reads 4} (select-keys (lm/info m) [:requests :hits :reads]))
                    "two one-token reads of one context in one advance, once"))))))

;; ---------------------------------------------------------------------------
;; Inference-like use: no backstop, bounded memory
;; ---------------------------------------------------------------------------

(defn- draw
  "A token drawn from row with rng (host sampling, for the tests' loops)."
  [^java.util.Random rng ^floats row]
  (let [u (.nextDouble rng)]
    (loop [t 0 acc 0.0]
      (let [acc (+ acc (Math/exp (aget row t)))]
        (if (or (< u acc) (= t (dec (alength row)))) t (recur (inc t) acc))))))

(defn- smc-like
  "n particles over t steps, read in one batch a step and resampled each
   step: was every read exact?"
  [m n t]
  (let [rng (java.util.Random. 7)]
    (loop [step 0 particles (vec (repeat n prompt)) ok true]
      (if (= step t)
        ok
        (let [rows (lm/batch-next-logprobs m particles)
              ok (and ok (every? true? (map #(same-row? (exact-row %1) %2) particles rows)))
              grown (mapv #(conj %1 (draw rng %2)) particles rows)]
          (recur (inc step) (mapv (fn [_] (grown (.nextInt rng n))) grown) ok))))))

(defn- mh-like
  "A trace of len tokens, regenerated from a random site k times, scored
   whole, accepted at random: every read exact?"
  [m len k]
  (let [rng (java.util.Random. 11)
        extend-from (fn [ctx n] (reduce (fn [ctx _] (conj ctx (draw rng (lm/next-logprobs m ctx)))) ctx (range n)))]
    (loop [i 0 trace (extend-from prompt len) ok true]
      (if (= i k)
        ok
        (let [site (+ (count prompt) (.nextInt rng len))
              proposal (extend-from (subvec trace 0 site) (- (count trace) site))
              cont (subvec proposal (count prompt))
              ok (and ok (= (exact-scores prompt cont) (lm/score m prompt cont)))]
          (recur (inc i) (if (.nextBoolean rng) proposal trace) ok))))))

(deftest inference-loops-never-reach-the-backstop
  (doseq [[what bounds f] [["SMC, 8 particles" {:max-branches 12 :max-cells 120} #(smc-like % 8 20)]
                           ["SMC, short of cells" {:max-branches 20 :max-cells 40} #(smc-like % 8 20)]
                           ["MH over a 15-token trace" {:max-branches 6 :max-cells 60} #(mh-like % 15 25)]]]
    (testing what
      (run-on bounds (fn [m e]
                       (is (f m) "every read exact")
                       (is (empty? (books m e bounds)))
                       (is (pos? (:evictions (lm/info m))) "the bounds were reached"))))))

(deftest the-backstop-count-counts
  ;; so that a zero in the books can be wrong: a dropped branch the engine
  ;; must collect
  (with-open [e (toy/engine {:max-branches 2})]
    (let [b (br/open e [1 2])]
      (count (br/tokens (br/fork b)))
      (let [f (br/fork b)]
        (is (= 1 (:collections (br/info e))))
        (br/dispose! [b f])))))

;; ---------------------------------------------------------------------------
;; Misuse, closing, the vocabulary
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud-and-changes-nothing
  (run-on {:max-branches 4 :max-cells 20}
          (fn [m e]
            (lm/next-logprobs m prompt)
            (let [before (lm/info m)]
              (doseq [[want f] [[:no-tokens #(lm/next-logprobs m [])]
                                [:invalid-token #(lm/next-logprobs m 3)]
                                [:invalid-token #(lm/next-logprobs m (conj prompt n-vocab))]
                                [:invalid-token #(lm/next-logprobs m (conj prompt -1))]
                                [:invalid-token #(lm/next-logprobs m (conj prompt 1.0))]
                                [:invalid-token #(lm/next-logprobs m (conj prompt 100000000000000000000N))]
                                [:context-full #(lm/next-logprobs m (vec (repeat 21 1)))]
                                [:invalid-token #(lm/batch-next-logprobs m [(conj prompt 1) (conj prompt :a)])]
                                [:no-tokens #(lm/batch-next-logprobs m [(conj prompt 1) []])]
                                [:invalid-token #(lm/batch-next-logprobs m (conj prompt 1))]
                                [:no-tokens #(lm/score m [] [1 2])]
                                [:invalid-token #(lm/score m prompt [1 n-vocab])]
                                [:invalid-token #(lm/score m prompt 1)]
                                [:context-full #(lm/score m prompt (vec (repeat 18 1)))]
                                [:invalid-token #(lm/token-bytes m n-vocab)]
                                [:invalid-token #(lm/token-bytes m 100000000000000000000N)]]]
                (is (= want (bc/reason f))))
              (is (= before (lm/info m)) "nothing read, nothing counted")
              (is (empty? (books m e {:max-branches 4 :max-cells 20})))))))

;; One rule and one error for a token of the vocabulary (the post-M5
;; review's I4-03): br/index? and br/invalid-token, which the cache's and
;; token-bytes' checks raise, and the one reader of an :end-tokens option,
;; whose copies in byte-lm and grammar once threw IllegalArgumentException
;; on a big integer. Plants: a copy that coerces with long fails the
;; end-token checks here and in bytes-test and grammar-test.
(deftest one-rule-for-a-token-of-the-vocabulary
  (with-open [m (lm/lm (toy/engine {:max-branches 4 :n-ctx 64}) (tt/tokenizer n-vocab))]
    (let [message #(try (%) nil (catch clojure.lang.ExceptionInfo e (ex-message e)))]
      (is (= "lm: not a token of the vocabulary (0 to 96): 97" (ex-message (br/invalid-token "lm" 97 97))))
      (is (= (ex-message (br/invalid-token "lm" 97 97)) (message #(lm/next-logprobs m [1 97]))) "the cache's")
      (is (= (ex-message (br/invalid-token "token-bytes" 97 97)) (message #(lm/token-bytes m 97))) "token-bytes'")))
  (is (= #{4 7} (lm/end-tokens-option "here" {:end-tokens #{4 (int 7)}} #{1} 10)))
  (is (every? #(instance? Long %) (lm/end-tokens-option "here" {:end-tokens #{4 (int 7)}} #{1} 10)))
  (is (= #{1} (lm/end-tokens-option "here" {} #{1} 10)) "the default")
  (is (= #{} (lm/end-tokens-option "here" {:end-tokens #{}} #{1} 10)))
  (doseq [ts [#{10} #{-1} #{100000000000000000000N} #{1.0} [1] nil]]
    (is (= :invalid-end-tokens (bc/reason #(lm/end-tokens-option "here" {:end-tokens ts} #{1} 10))) (pr-str ts)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"^here: " (lm/end-tokens-option "here" {:end-tokens #{10}} #{} 10))))

;; A read that fails keeps its error, and a failure to dispose of its forks
;; is suppressed onto it, not swallowed (the review's I4-07). Plant: the
;; old (catch Throwable _) fails this test.
(deftest a-failed-read-keeps-the-cleanup-s-failure
  (let [engine (reify br/Branching
                 (-live? [_ _] true)
                 (-dispose [_ _] (throw (ex-info "the dispose failed" {}))))
        fork (gen.llm.branching.Branch. engine 0 [] (float-array 1))
        e (try (#'cache/reading (fn [made] (vswap! made conj fork) (throw (ex-info "the read failed" {}))))
               (catch clojure.lang.ExceptionInfo e e))]
    (is (= "the read failed" (ex-message e)))
    (is (= ["the dispose failed"] (map ex-message (.getSuppressed ^Throwable e))))))

(deftest the-limit-bounds-contexts-and-scored-sequences
  ;; :limit, the lesser of the engine's :n-ctx and :max-cells: a context or
  ;; a scored sequence of that many tokens is read, one more is refused
  (run-on {:max-branches 4 :max-cells 20}
          (fn [m _]
            (let [limit (:limit (lm/info m))
                  room (- limit (count prompt))]
              (is (= 20 limit))
              (is (= n-vocab (alength (lm/next-logprobs m (vec (repeat limit 1))))))
              (is (= :context-full (bc/reason #(lm/next-logprobs m (vec (repeat (inc limit) 1))))))
              (is (= room (count (lm/score m prompt (vec (repeat room 2))))))
              (is (= :context-full (bc/reason #(lm/score m prompt (vec (repeat (inc room) 2))))))))))

(deftest bad-options-and-tokenizers-are-loud
  (with-open [e (ce/engine {:max-branches 4})]
    (let [tok (tt/tokenizer n-vocab)]
      (is (= :invalid-option (bc/reason #(lm/lm e tok {:max-branches 5}))))
      (is (= :invalid-option (bc/reason #(lm/lm e tok {:max-branches 0}))))
      (is (= :invalid-option (bc/reason #(lm/lm e tok {:max-cells 0}))))
      (is (= :invalid-tokenizer (bc/reason #(lm/lm e (tt/tokenizer 98)))))
      (is (= :invalid-tokenizer (bc/reason #(lm/lm e (dissoc tok :decode)))))
      (is (= :invalid-tokenizer (bc/reason #(lm/lm e (assoc tok :eog [96])))))
      (let [b (br/open e [1])]
        (with-open [m (lm/lm e tok)]
          (is (= 3 (:max-branches (lm/info m))) "by default, the branches the engine has free"))
        (br/dispose! b)))))

(deftest closing-frees-every-branch
  (with-open [e (ce/engine {:max-branches 4})]
    (let [m (lm/lm e (tt/tokenizer n-vocab))]
      (lm/batch-next-logprobs m [prompt (conj prompt 1) (conj prompt 2)])
      (is (= 3 (:live (br/info e))))
      (.close m)
      (.close m)
      (is (= 0 (:live (br/info e))))
      (is (:closed? (lm/info m)))
      (is (= :closed (bc/reason #(lm/next-logprobs m prompt))))
      (is (= :closed (bc/reason #(lm/score m prompt [1])))))))

(deftest a-closed-engine-is-loud-and-forgotten
  (let [e (ce/engine {:max-branches 4})
        m (lm/lm e (tt/tokenizer n-vocab))]
    (lm/next-logprobs m prompt)
    (.close e)
    (is (= :closed (bc/reason #(lm/next-logprobs m (conj prompt 1)))))
    (is (= 0 (:live (lm/info m))) "the engine's branches are gone, and so are the entries")
    (.close m)))

(deftest the-vocabulary-is-the-tokenizers
  (run-on {:max-branches 2 :max-cells 64}
          (fn [m _]
            (is (= {:n-vocab 97 :bos 95 :eos 96 :eog #{96} :add-bos? true} (lm/vocab m)))
            (is (= [95 40 73] (lm/encode m "Hi")))
            (is (= [40 73] (lm/encode m "Hi" {:add-special? false})))
            (is (= "<s>Hi" (lm/decode m [95 40 73])))
            (is (= "H" (String. (lm/token-bytes m 40) "UTF-8")))
            (let [bv (lm/byte-vocab m)]
              (is (= 97 (count bv)))
              (is (= "</s>" (String. ^bytes (bv 96) "UTF-8")))
              (aset ^bytes (bv 40) 0 (byte 0))
              (is (= "H" (String. (lm/token-bytes m 40) "UTF-8")) "new arrays")))))
