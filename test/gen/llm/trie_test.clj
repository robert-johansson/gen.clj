(ns gen.llm.trie-test
  "The prefix trie (gen.llm.trie) and the cell ledger (gen.llm.cells)
   against naive models, over random operations.

   The trie's model is a map of contexts to entries and a map of every
   prefix on a path to its score (the first written; a prefix is forgotten
   when no entry's context extends it). The ledger's is physical: each
   branch holds a set of cells, and the cells used are their union.

   Mutations checked (2026-10-20, each applied, the suite run, and
   reverted), with the tests that caught them:
   - match descending an edge it matches only in part: trie-matches-its-model;
   - insert overwriting the scores already on a path: trie-matches-its-model,
     a-path-keeps-its-first-scores;
   - a leaf's scores shifted by one position: the same two;
   - tidy not joining a bare pass-through node: the-trie-stays-compressed;
   - remove-entry keeping a path that leads to no entry: both trie
     properties (size, and the structure);
   - share not counting a holder: ledger-matches-the-cells;
   - freed counting a segment another branch holds: ledger-matches-the-cells."
  (:require [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.llm.cells :as cells]
            [gen.llm.trie :as trie]))

;; ---------------------------------------------------------------------------
;; The trie
;; ---------------------------------------------------------------------------

;; remove-entry is loud off the trie's entries, :no-entry (the post-M5
;; review's I4-07): an absent path recursed until a StackOverflowError, one
;; ending inside an edge threw IndexOutOfBoundsException, and one parting
;; from an edge inside it removed the entry it shares a first token with.
;; Plant: the old recursion fails this test (the absent path, the overflow).
(deftest remove-entry-is-loud-off-the-entries
  (let [t (trie/insert trie/empty-trie [1 2 3] 0 [0.0 0.0 0.0] :x)
        reason #(try (%) :no-error (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))) (catch Throwable e (class e)))]
    (doseq [tokens [[3 4] [1 2] [1 9 3] [1 2 3 4] []]]
      (is (= :no-entry (reason #(trie/remove-entry t tokens))) (pr-str tokens)))
    (is (= trie/empty-trie (trie/remove-entry t [1 2 3])))
    (is (= trie/empty-trie (trie/remove-entry t (list 1 2 3))) "any sequence of the tokens")))

(deftest remove-entry-takes-a-path-of-many-nodes
  ;; an entry at every prefix of 3,000 tokens, a node at each: the deepest
  ;; removed, and then one in the middle (read by match, which loops too)
  (let [n 3000
        tokens (vec (range n))
        t (reduce (fn [t k] (trie/insert t (subvec tokens 0 k) (dec k) [0.0] k)) trie/empty-trie (range 1 (inc n)))
        t1 (trie/remove-entry t tokens)
        t2 (trie/remove-entry t1 (subvec tokens 0 1500))]
    (is (= {:depth n :entry n :entry-depth n} (trie/match t tokens)))
    (is (= {:depth (dec n) :entry (dec n) :entry-depth (dec n)} (trie/match t1 tokens)) "the path to it goes too")
    (is (= {:depth 1500 :entry 1499 :entry-depth 1499} (trie/match t2 (subvec tokens 0 1500))))
    (is (= {:depth (dec n) :entry (dec n) :entry-depth (dec n)} (trie/match t2 tokens)) "the entries past it stay")))

(defn- prefixes [ctx] (map #(subvec ctx 0 %) (range 1 (inc (count ctx)))))

(defn- model-insert [{:keys [entries scores]} ctx from sc entry]
  {:entries (assoc entries ctx entry)
   :scores (reduce (fn [m p]
                     (let [i (dec (count p))]
                       (if (contains? m p) m (assoc m p (if (<= from i) (sc (- i from)) Double/NaN)))))
                   scores (prefixes ctx))})

(defn- model-remove [{:keys [entries scores]} ctx]
  (let [entries (dissoc entries ctx)
        kept (set (mapcat prefixes (keys entries)))]
    {:entries entries :scores (select-keys scores kept)}))

(defn- model-match [{:keys [entries scores]} q]
  (let [on-path (take-while #(contains? scores %) (prefixes q))
        at-entry (filter #(contains? entries %) on-path)]
    {:depth (count on-path)
     :entry (some->> (last at-entry) entries)
     :entry-depth (count (last at-entry))}))

(defn- model-scores [{:keys [scores]} q from]
  (when (every? #(contains? scores %) (prefixes q))
    (mapv #(scores (subvec q 0 (inc %))) (range from (count q)))))

(defn- same-scores? [a b]
  (and (= (count a) (count b))
       (every? true? (map #(or (== %1 %2) (and (Double/isNaN %1) (Double/isNaN %2))) a b))))

(def ^:private gen-context
  (gen/vector (gen/choose 0 2) 1 6))

(def ^:private gen-op
  (gen/one-of [(gen/let [ctx gen-context
                         from (gen/choose 0 6)
                         sc (gen/vector (gen/fmap double gen/small-integer) 6)]
                 [:insert ctx (min from (count ctx)) sc])
               (gen/fmap #(vector :remove %) gen/nat)]))

(defn- apply-op
  "[trie model] after op (inserts at contexts that have an entry, and
   removes of no entry, are skipped)."
  [[t m] [kind x from sc]]
  (let [ks (vec (sort-by pr-str (keys (:entries m))))]
    (case kind
      :insert (if (contains? (:entries m) x)
                [t m]
                (let [entry {:id (count (:entries m)) :at x}
                      sc (subvec sc 0 (- (count x) from))]
                  [(trie/insert t x from sc entry) (model-insert m x from sc entry)]))
      :remove (if (empty? ks)
                [t m]
                (let [ctx (ks (mod x (count ks)))]
                  [(trie/remove-entry t ctx) (model-remove m ctx)])))))

(defn- agrees?
  "Does the trie answer every query as the model does?"
  [t m queries]
  (and (= (set (trie/entries t)) (set (vals (:entries m))))
       (= (trie/size t) (count (:scores m)))
       (every? (fn [[q from]]
                 (let [from (min from (count q))]
                   (and (= (trie/match t q) (model-match m q))
                        (let [a (trie/scores t q from) b (model-scores m q from)]
                          (if (nil? b) (nil? a) (same-scores? a b))))))
               queries)))

(defspec trie-matches-its-model 300
  (prop/for-all [ops (gen/vector gen-op 1 30)
                 queries (gen/vector (gen/tuple gen-context (gen/choose 0 6)) 8)]
    (every? (fn [[t m]] (agrees? t m queries))
            (reductions apply-op [trie/empty-trie {:entries {} :scores {}}] ops))))

(defn- compressed?
  "Every node below the root has an edge, keyed by its first token, with a
   score per token, and has an entry or two children at least."
  [node root?]
  (every? (fn [[k child]]
            (and (seq (:tokens child))
                 (= k (first (:tokens child)))
                 (= (count (:tokens child)) (count (:scores child)))
                 (or (:entry child) (<= 2 (count (:children child))))
                 (compressed? child false)))
          (:children node)))

(defspec the-trie-stays-compressed 300
  (prop/for-all [ops (gen/vector gen-op 1 30)]
    (every? (fn [[t]] (compressed? t true))
            (reductions apply-op [trie/empty-trie {:entries {} :scores {}}] ops))))

(deftest a-second-entry-at-a-context-is-loud
  (let [t (trie/insert trie/empty-trie [1 2] 1 [0.5] {:id 0})]
    (is (= :entry-exists (:reason (ex-data (try (trie/insert t [1 2] 1 [0.5] {:id 1}) (catch Exception x x))))))))

(deftest a-path-keeps-its-first-scores
  ;; a token read again keeps the score it was first read with
  (let [t (-> trie/empty-trie
              (trie/insert [1 2 3] 1 [-1.0 -2.0] {:id 0})
              (trie/insert [1 2 3 4] 1 [-9.0 -9.0 -3.0] {:id 1}))]
    (is (= [-1.0 -2.0 -3.0] (trie/scores t [1 2 3 4] 1)))
    (is (nil? (trie/scores t [1 2 4] 1)))
    (is (= {:depth 3 :entry {:id 0} :entry-depth 3} (trie/match t [1 2 3 7])))))

;; ---------------------------------------------------------------------------
;; The cell ledger
;; ---------------------------------------------------------------------------

(def ^:private gen-ledger-op
  (gen/one-of [(gen/fmap #(vector :open %) (gen/choose 1 4))
               (gen/fmap #(vector :fork %) gen/nat)
               (gen/tuple (gen/return :read) gen/nat (gen/choose 1 3))
               (gen/fmap #(vector :dispose %) gen/nat)]))

(defn- ledger-step
  "{:ledger :branches [{:segment :cells}] :next} after op; a branch's
   :cells are the physical cells it holds."
  [{:keys [ledger branches next] :as w} [kind x k]]
  (let [n (count branches)
        fresh (fn [k] (set (range next (+ next k))))]
    (case kind
      :open (let [[l s] (cells/allocate ledger nil x)]
              {:ledger l :branches (conj branches {:segment s :cells (fresh x)}) :next (+ next x)})
      :fork (if (zero? n) w
                (let [b (branches (mod x n))]
                  (assoc w :ledger (cells/share ledger (:segment b)) :branches (conj branches b))))
      :read (if (zero? n) w
                (let [i (mod x n)
                      b (branches i)
                      [l s] (cells/allocate ledger (:segment b) k)]
                  {:ledger (cells/release l (:segment b))
                   :branches (assoc branches i {:segment s :cells (into (:cells b) (fresh k))})
                   :next (+ next k)}))
      :dispose (if (zero? n) w
                   (let [i (mod x n)]
                     (assoc w
                            :ledger (cells/release ledger (:segment (branches i)))
                            :branches (into (subvec branches 0 i) (subvec branches (inc i)))))))))

(defn- ledger-agrees? [{:keys [ledger branches]}]
  (let [union (fn [bs] (reduce into #{} (map :cells bs)))]
    (and (= (cells/used ledger) (count (union branches)))
         (every? (fn [i]
                   (let [others (union (concat (subvec branches 0 i) (subvec branches (inc i))))]
                     (= (cells/freed ledger (:segment (branches i)))
                        (count (remove others (:cells (branches i)))))))
                 (range (count branches))))))

(defspec ledger-matches-the-cells 300
  (prop/for-all [ops (gen/vector gen-ledger-op 1 40)]
    (every? ledger-agrees?
            (reductions ledger-step {:ledger cells/empty-ledger :branches [] :next 0} ops))))

(deftest an-unknown-segment-is-loud
  (is (= :no-segment (:reason (ex-data (try (cells/release cells/empty-ledger 3) (catch Exception x x)))))))
