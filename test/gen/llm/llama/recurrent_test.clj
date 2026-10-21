(ns gen.llm.llama.recurrent-test
  "The replay of llama.cpp's recurrent cells (gen.llm.llama.recurrent) on
   cases worked by hand from llama-memory-recurrent.cpp (620d222), its
   invariants under random operations, and its plans for splitting a step.
   That llama.cpp puts the states where the replay says is tested on a
   hybrid model by gen.llm.llama-test (a-scattered-step-is-split), whose
   time depends on it.

   Mutations checked by hand (2026-10-21, each applied, these tests run,
   and reverted), each failing the tests named:
   - the search for an empty cell not wrapping round: the-search-wraps-round,
     a-wrapped-step-is-split, random-operations-keep-the-books;
   - no reset of the head when the cells below it are empty:
     the-head-starts-again-from-the-first-cell;
   - the gather's swaps not moving the tails: the-gather-moves-the-states-between,
     random-operations-keep-the-books;
   - a sequence sharing its cell keeping it (no fork leaves): eight of the
     nine, a-shared-cell-stays-with-its-last-sharer among them;
   - the planner's runs in the order of their cells, not of the search:
     a-leaver-goes-before-the-cell-it-leaves;
   - one run for every cell (no cut): a-wrapped-step-is-split,
     a-leaver-goes-before-the-cell-it-leaves."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.llm.llama.recurrent :as r]))

(def ^:private fixed 6.0)

(defn- step
  "A decode of seqs, each at position pos."
  [st seqs pos]
  (r/decode st (vec seqs) (vec (repeat (count seqs) pos))))

(defn- forks [st src dsts] (reduce #(r/copy %1 src %2) st dsts))

(defn- tails [st seqs] (mapv #(get-in st [:cells % :tail]) seqs))

(defn- extra [st] (- (:n-rs st) (count (:taken st))))

(defn- replay-plan
  "The decodes of groups in turn: the last state and each decode's extra
   states."
  [st seqs pos groups]
  (reduce (fn [[st extras] g]
            (let [st (step st (mapv seqs g) pos)]
              [st (conj extras (extra st))]))
          [st []] groups))

(defn- with-children
  "A replay of size cells: sequence 0 read to position 8 on cell 0, and
   sequences 1 to n forked from it and stepped once, on cells 1 to n."
  [size n]
  (let [st (step (r/empty-cells size) [0] 8)]
    (step (forks st 0 (range 1 (inc n))) (range 1 (inc n)) 9)))

(deftest forks-take-the-next-empty-cells
  (let [st (step (r/empty-cells 16) [0] 8)]
    (is (= [0] (:taken st)))
    (is (= 0 (:head st)))
    (let [st (step (forks st 0 [1 2 3]) [1 2 3] 9)]
      (is (= [1 2 3] (:taken st)) "each fork leaves the shared cell 0 for the next empty one")
      (is (= 3 (:n-rs st)))
      (is (= 1 (:head st)) "the head is the decode's lowest cell")
      (is (= [0 1 2 3] (tails st [0 1 2 3])))
      (is (= #{0} (get-in st [:cells 0 :seq])))
      (is (= 4 (:used st))))))

(deftest the-gather-moves-the-states-between
  (let [st (reduce r/clear (with-children 16 10) [1 7 8 9 10])
        _ (is (= 1 (:head st)))
        st (step (forks st 0 [1 7 8 9 10]) [1 7 8 9 10] 9)]
    (is (= [1 7 8 9 10] (:taken st)))
    (is (= 10 (:n-rs st)) "cells 1 to 10 gathered, five of them for no read")
    (is (= [1 2 3 4 5] (tails st [1 7 8 9 10])) "the decode's sequences end on adjacent cells")
    (is (= [7 8 9 10 6] (tails st [2 3 4 5 6])) "the states they displaced moved to the cells they took")
    (is (every? #(= #{%} (get-in st [:cells (get-in st [:cells % :tail]) :seq])) (range 11)))))

(deftest the-search-wraps-round
  ;; cells 1 to 14 hold sequences 1 to 14; free cell 1, put the head on 14
  ;; with a step of the sequence there, and fork two branches
  (let [st (-> (with-children 16 14) (r/clear 1) (step [14] 10) (forks 0 [15 1]))
        st (step st [15 1] 9)]
    (is (= [15 1] (:taken st)) "the search goes from the head (14) to 15, then round to 1")
    (is (= 15 (:n-rs st)))))

(deftest the-head-starts-again-from-the-first-cell
  ;; two cells used (0 and 10), the head on 10: past used + 2n, the search
  ;; starts again from cell 0
  (let [st (-> (with-children 16 10) (as-> st (reduce r/clear st (range 1 10))) (step [10] 10))]
    (is (= 10 (:head st)))
    (is (= 2 (:used st)))
    (is (= [1] (:taken (step (forks st 0 [1]) [1] 9))))))

(deftest a-shared-cell-stays-with-its-last-sharer
  (let [st (step (forks (step (r/empty-cells 8) [0] 8) 0 [1]) [1] 9)
        st (forks st 1 [2 3])]
    (testing "the shared branch last"
      (let [st (step st [2 3 1] 10)]
        (is (= [2 3 1] (:taken st)) "2 and 3 leave cell 1 for empty ones; 1, then alone there, keeps it")
        (is (= [3 1 2] (tails st [1 2 3])))))
    (testing "the shared branch first"
      (is (= [2 3 1] (:taken (step st [1 2 3] 10))) "1 and 2 leave; 3, the last, keeps cell 1"))))

(deftest a-contiguous-step-is-one-decode
  (let [st (forks (with-children 16 4) 0 [5 6 7])]
    (is (= [[0 1 2]] (r/plan st [5 6 7] [9 9 9] fixed)))))

(deftest a-wrapped-step-is-split
  (let [st (-> (with-children 16 14) (r/clear 1) (step [14] 10) (forks 0 [15 1]))
        groups (r/plan st [15 1] [9 9] fixed)]
    (is (= [[0] [1]] groups) "cell 15 first, as the search finds it, then cell 1")
    (is (= [0 0] (second (replay-plan st [15 1] 9 groups))) "each decode gathers only its own cell")))

(deftest a-leaver-goes-before-the-cell-it-leaves
  ;; 1 is consumed (stepped) with its fork 12, whose empty cell is far from
  ;; 1's: 12 must go first, or 1 would leave cell 1 too
  (let [st (forks (with-children 16 11) 1 [12])
        seqs [12 1 2 3]
        groups (r/plan st seqs [10 10 10 10] fixed)]
    (is (= [12 1 2 3] (:taken (step st seqs 10))))
    (is (= [[0] [1 2 3]] groups))
    (is (= [0 0] (second (replay-plan st seqs 10 groups))))))

;; ---------------------------------------------------------------------------
;; Random operations: open, fork, clear and step, as an engine makes them
;; ---------------------------------------------------------------------------

(defn- books-kept?
  "Every live sequence's tail holds it, every cell's sequences have their
   tails there, and used counts the cells holding any."
  [st live]
  (and (every? (fn [s] (let [t (get-in st [:cells s :tail])]
                         (and (<= 0 t) (contains? (get-in st [:cells t :seq]) s))))
               live)
       (every? (fn [[i c]] (every? #(= i (get-in st [:cells % :tail])) (:seq c)))
               (map-indexed vector (:cells st)))
       (= (set live) (set (mapcat :seq (:cells st))))
       (= (:used st) (count (remove (comp empty? :seq) (:cells st))))))

(defn- run-ops
  "Apply ops (vectors of small integers, read against the live and free
   sequences) to a replay of size cells, each step as the plan splits it:
   failures, or nil."
  [size ops]
  (loop [st (r/empty-cells size) pos {} [[op a b] & more] ops]
    (let [live (vec (sort (keys pos)))
          free (vec (remove pos (range size)))]
      (cond
        (not (books-kept? st live)) [:books st]
        (nil? op) nil
        :else
        (case (long (mod op 4))
          0 (if (seq free) ; open a branch on a free slot
              (let [s (free (mod a (count free)))]
                (recur (r/decode st [s] [(+ 3 (mod b 5))]) (assoc pos s (+ 4 (mod b 5))) more))
              (recur st pos more))
          1 (if (and (seq live) (seq free)) ; fork
              (let [src (live (mod a (count live))) dst (free (mod b (count free)))]
                (recur (r/copy st src dst) (assoc pos dst (pos src)) more))
              (recur st pos more))
          2 (if (seq live) ; dispose of
              (let [s (live (mod a (count live)))]
                (recur (r/clear st s) (dissoc pos s) more))
              (recur st pos more))
          3 (if (seq live) ; step a subset, in a shuffled order
              (let [seqs (vec (sort-by #(hash [a %]) (filter #(odd? (bit-shift-right (long b) (mod % 8))) live)))
                    seqs (if (seq seqs) seqs [(first live)])
                    ps (mapv pos seqs)
                    groups (r/plan st seqs ps fixed)
                    cost (fn [gs] (reduce (fn [[st c] g]
                                            (let [st (r/decode st (mapv seqs g) (mapv ps g))]
                                              [st (+ c fixed (extra st))]))
                                          [st 0.0] gs))
                    [st' c] (cost groups)
                    [_ whole] (cost [(vec (range (count seqs)))])]
                (cond
                  (not= (sort (apply concat groups)) (range (count seqs))) [:not-a-partition groups]
                  (> c whole) [:dearer c whole]
                  :else (recur st' (reduce #(update %1 %2 inc) pos seqs) more)))
              (recur st pos more)))))))

(defspec random-operations-keep-the-books 200
  (prop/for-all [size (gen/choose 4 24)
                 ops (gen/vector (gen/tuple gen/nat gen/nat (gen/choose 0 255)) 0 120)]
    (nil? (run-ops size ops))))
