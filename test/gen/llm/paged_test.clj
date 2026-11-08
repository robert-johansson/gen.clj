(ns gen.llm.paged-test
  "A paged KV cache's books (gen.llm.paged): the allocator and block
   tables, values on the host.
   1. allocate, retain, release and grow keep every block free or held,
      and refuse what they cannot do, loudly;
   2. prepare takes blocks as positions need them, appends to a shared
      last block filled in part when no holder has written past the
      sequence's end (its written offsets), and copies it otherwise
      (copy-on-write), nothing else: a parent written after its fork
      wrote copies;
   3. a property over random programs of opens, forks, writes and
      disposals, run against a simulated pool (each slot's value, copies
      applied before writes): every live sequence reads back exactly what
      was written to it and its ancestors before the fork, and once all
      are disposed every block is free.
   Planted bugs fail 3 (with-redefs-fn): copy-on-write skipped (a fork's
   write lands in its parent's block), a copy's direction reversed, a
   second writer not copying, the written offsets not advanced, and
   release freeing a block its other holders still read."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.llm.paged :as paged]))

(defn- reason [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest the-allocator-keeps-every-block-free-or-held
  (let [a (paged/allocator 8 2)
        [a ids] (paged/allocate a 3)]
    (is (= [2 3 4] ids) "the reserved blocks never given")
    (is (= 3 (paged/free-count a)))
    (let [a (paged/retain a [3])]
      (is (= 2 (paged/holders a 3)))
      (let [a (paged/release a [3 4])]
        (is (= 1 (paged/holders a 3)) "a shared block held by its other holder")
        (is (zero? (paged/holders a 4)) "a block held once freed")
        (is (= 4 (paged/free-count a)))))
    (is (= :blocks-exhausted (reason #(paged/allocate a 4))))
    (is (= 7 (paged/free-count (paged/grow a 12))))
    (is (= :invalid-args (reason #(paged/release a [6]))) "a free block released")
    (is (= :invalid-args (reason #(paged/retain a [6]))) "a free block shared")
    (is (= :invalid-args (reason #(paged/grow a 4))) "a pool shrunk")
    (is (= :invalid-args (reason #(paged/allocator 4 4))))))

(deftest prepare-takes-blocks-and-copies-a-shared-partial-block
  (let [bs 4
        a (paged/allocator 16 1)
        [a s _ copies] (paged/prepare a paged/empty-sequence 6 bs)]
    (is (= {:blocks [1 2] :len 6} s))
    (is (= [] copies))
    (testing "slots: offset p mod bs of block p/bs"
      (let [[_ _ slots] (paged/prepare a s 3 bs)]
        (is (= [10 11 12] slots) "positions 6, 7 in block 2, then 8 in a new block 3")))
    (testing "a fork shares; the first to write appends in place, the second copies"
      (let [[a f] (paged/fork a s)
            _ (is (= 2 (paged/written a 2)) "block 2's offsets 0 and 1 written")
            [a s' slots copies] (paged/prepare a s 1 bs)]
        (is (= [] copies) "no holder wrote past s's end: in place")
        (is (= {:blocks [1 2] :len 7} s'))
        (is (= [10] slots))
        (is (= 3 (paged/written a 2)) "the block's written offsets advanced")
        (let [[a f' slots copies] (paged/prepare a f 1 bs)]
          (is (= [[2 3]] copies) "s wrote offset 2, which f would write: f copies block 2 to a new block 3")
          (is (= {:blocks [1 3] :len 7} f'))
          (is (= [14] slots))
          (is (= 1 (paged/holders a 2)) "s holds block 2 alone now")
          (is (= 3 (paged/written a 3)) "the copy's written offsets are f's")
          (let [[_ _ _ copies] (paged/prepare a s' 1 bs)]
            (is (= [] copies) "s, alone, appends in place")))))
    (testing "a parent written after its fork wrote copies"
      (let [[a f] (paged/fork a s)
            [a _ _ c1] (paged/prepare a f 1 bs)
            [_ _ _ c2] (paged/prepare a s 1 bs)]
        (is (= [] c1) "the fork goes on in place")
        (is (= [[2 3]] c2) "then the parent, a second writer, copies")))
    (testing "a full shared block is never copied"
      (let [[a s4 _ _] (paged/prepare (paged/allocator 16 1) paged/empty-sequence 4 bs)
            [a _] (paged/fork a s4)
            [_ s5 _ copies] (paged/prepare a s4 1 bs)]
        (is (= [] copies))
        (is (= [1 2] (:blocks s5)))))))

;; ---------------------------------------------------------------------------
;; 3. Random programs against a simulated pool
;; ---------------------------------------------------------------------------

(def ^:private bs 4)

(defn- run-program
  "Runs ops on an allocator and a simulated pool: each live sequence's
   blocks hold its contents. Returns nil when every read matched, else
   the first mismatch."
  [ops]
  (loop [a (paged/allocator 64 1) pool {} seqs {} contents {} next-id 0 [op & more] ops]
    (let [bad (some (fn [[id s]]
                      (let [read (mapv #(get pool (paged/slot s bs %)) (range (:len s)))]
                        (when-not (= read (contents id)) {:seq id :read read :written (contents id)})))
                    seqs)]
      (cond bad bad
            (nil? op) (let [a (reduce paged/dispose a (vals seqs))]
                        (when-not (= 63 (paged/free-count a)) {:leaked (- 63 (paged/free-count a))}))
            :else
            (let [live (vec (sort (keys seqs)))
                  pick #(when (seq live) (live (mod (long %) (count live))))]
              (case (first op)
                :open (recur a pool (assoc seqs next-id paged/empty-sequence) (assoc contents next-id []) (inc next-id) more)
                :fork (if-let [id (pick (second op))]
                        (let [[a s] (paged/fork a (seqs id))]
                          (recur a pool (assoc seqs next-id s) (assoc contents next-id (contents id)) (inc next-id) more))
                        (recur a pool seqs contents next-id more))
                :write (if-let [id (pick (second op))]
                         (let [n (nth op 2)
                               values (mapv #(vector id (+ (count (contents id)) %)) (range n))
                               [a s slots copies] (paged/prepare a (seqs id) n bs)
                               ;; copies first, every block read before any is written
                               pool (let [old pool]
                                      (reduce (fn [p [from to]]
                                                (reduce #(assoc %1 (+ (* to bs) %2) (get old (+ (* from bs) %2))) p (range bs)))
                                              pool copies))
                               pool (reduce (fn [p [sl v]] (assoc p sl v)) pool (map vector slots values))]
                           (recur a pool (assoc seqs id s) (update contents id into values) next-id more))
                         (recur a pool seqs contents next-id more))
                :dispose (if-let [id (pick (second op))]
                           (recur (paged/dispose a (seqs id)) pool (dissoc seqs id) (dissoc contents id) next-id more)
                           (recur a pool seqs contents next-id more))))))))

(def ^:private op-gen
  (gen/one-of [(gen/return [:open])
               (gen/tuple (gen/return :fork) gen/nat)
               (gen/tuple (gen/return :write) gen/nat (gen/choose 1 6))
               (gen/tuple (gen/return :dispose) gen/nat)]))

(def ^:private programs
  (gen/fmap #(into [[:open]] %) (gen/vector op-gen 1 40)))

(defn- holds? [] (:result (tc/quick-check 300 (prop/for-all [ops programs] (nil? (run-program ops))) :seed 20261107)))

(def ^:private planted
  {"copy-on-write skipped (the last block taken to be held alone)"
   (fn [prepare] (fn [a s ^long n ^long bs] (let [id (peek (:blocks s))
                                                  h (paged/holders a id)
                                                  [a' s' slots copies] (prepare (cond-> a (pos? h) (assoc-in [:refs id] 1)) s n bs)]
                                              [(cond-> a' (pos? h) (assoc-in [:refs id] h)) s' slots copies])))
   "a copy's direction reversed"
   (fn [prepare] (fn [a s ^long n ^long bs] (let [[a s' slots copies] (prepare a s n bs)] [a s' slots (mapv (comp vec reverse) copies)])))
   "a second writer not copying (the written offsets taken to be its own)"
   (fn [prepare] (fn [a s ^long n ^long bs] (prepare (cond-> a (seq (:blocks s)) (assoc-in [:written (peek (:blocks s))] (rem (long (:len s)) (long bs))))
                                         s n bs)))
   "the written offsets not advanced"
   (fn [prepare] (fn [a s ^long n ^long bs] (let [[a' s' slots copies] (prepare a s n bs)]
                                  [(assoc a' :written (merge (:written a') (select-keys (:written a) (keys (:written a')))))
                                   s' slots copies])))})

(deftest every-sequence-reads-what-was-written-to-it
  (is (true? (holds?)))
  (testing "planted bugs fail it"
    (doseq [[label wrap] planted]
      (is (not (true? (with-redefs-fn {#'paged/prepare (wrap paged/prepare)} holds?))) label))
    (is (not (true? (with-redefs-fn {#'paged/release (fn [a ids] (-> a (update :refs #(apply dissoc % ids)) (update :free into ids)))}
                      holds?)))
        "release freeing a block its other holders still read")))
