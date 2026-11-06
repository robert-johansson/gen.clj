(ns gen.partition-test
  "gen.partition on a batch's data directly (genclj-w867; the batched Switch
   tests in batched_test and fuzz_test run it through models): groups,
   gathering particles (values lazily, Parts regrouped, identity kept for
   data with no batched value), gathering into a part's backend, joining
   consecutive parts, merging groups by key, scattering parts back to their
   particles, and the loud error for data no one batch can hold. Planted
   (2026-11-06): groups in key order fails groups-are-in-order-of-first-
   particle; consecutive without the offsets fails parts-join-in-order;
   gen.array.batch/scatter taking by the parts' order in place of its
   inverse fails scatter-puts-each-particle-back; regroup keeping a group
   none is left in fails gathering-parts-regroups-them."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.partition :as part]
            [gen.test-support :as ts]))

(defn- wrap
  "A batched value of the host numbers xs, one per particle."
  [xs]
  (batch/wrap (array/array xs) [(count xs)]))

(defn- particles [x] (ts/host (batch/value x)))

(defn- batch-of [n] (batch/backend array/*backend* [n]))

(deftest groups-are-in-order-of-first-particle
  (is (= [[:b [0 3]] [:a [1 2 4]] [:c [5]]] (part/groups [:b :a :a :b :a :c])))
  (is (= [[0 [0 1 2]]] (part/groups [0 0 0])))
  (is (= [] (part/groups []))))

(deftest gathering-takes-the-particles
  (let [x (wrap [10.0 20.0 30.0])
        gf (gen [] (trace :z (d/normal 0 1)))
        data {:x x :shared 5.0 :gf gf :nested [x {:y x}] :seq (list x 1)}
        g (part/gather data [2 0 0])]
    (is (= [30.0 10.0 10.0] (particles (:x g))))
    (is (= [30.0 10.0 10.0] (particles (get-in g [:nested 1 :y]))))
    (is (= [30.0 10.0 10.0] (particles (first (:seq g)))))
    (is (= 5.0 (:shared g)) "a shared value as it is")
    (is (identical? gf (:gf g)) "a generative function as it is")
    (is (= [3] (array/shape (batch/value (:x g)))) "a batch of the new particles' number"))
  (testing "data holding no batched value keeps its identity"
    (let [plain {:a [1 2] :b #{3} :c (list 4)}]
      (is (identical? plain (part/gather plain [1 0]))))))

(def ^:private parts
  "Two groups of a batch of three: :a the particles 0 and 2, :b particle 1."
  (batch/->Parts [{:key :a :rows [0 2] :data (wrap [100.0 300.0])}
                  {:key :b :rows [1] :data (wrap [200.0])}]))

(deftest gathering-parts-regroups-them
  (let [g (part/gather parts [2 1 2 0])]
    (is (batch/parts? g))
    (is (= [[:a [0 2 3]] [:b [1]]] (mapv (juxt :key :rows) (:groups g))) "the new particles of each group")
    (is (= [300.0 300.0 100.0] (particles (:data (first (:groups g))))) "each group's data, for its new particles")
    (is (= [200.0] (particles (:data (second (:groups g)))))))
  (is (= [[:b [0 1]]] (mapv (juxt :key :rows) (:groups (part/gather parts [1 1])))) "a group none is left in is gone")
  (is (identical? parts (part/gather-into parts nil (batch-of 3))) "all of them, in order: as they are"))

(deftest gathering-into-a-part-s-backend
  (let [b (batch-of 3)
        x (wrap [10.0 20.0 30.0])
        pb (batch/part-backend b [2 0])
        y (part/gather-into {:x x :s 1.5} [2 0] pb)]
    (is (= [30.0 10.0] (particles (:x y))))
    (is (identical? pb (array/backend (:x y))) "a value of the part's backend")
    (is (= 1.5 (:s y)))
    (is (= [10.0 20.0 30.0] (particles (part/gather-into x nil (batch/part-backend b [0 1 2])))) "nil: every particle")))

(deftest parts-join-in-order
  (let [b (batch-of 3)]
    (is (= [1.0 2.0 3.0] (particles (part/join [(wrap [1.0 2.0]) (wrap [3.0])] [2 1] b))))
    (testing "maps and vectors field by field, a value every part shares kept"
      (let [j (part/join [{:x (wrap [1.0 2.0]) :k :same :v [(wrap [5.0 6.0])]}
                          {:x (wrap [3.0]) :k :same :v [(wrap [7.0])]}]
                         [2 1] b)]
        (is (= [1.0 2.0 3.0] (particles (:x j))))
        (is (= [5.0 6.0 7.0] (particles (first (:v j)))))
        (is (= :same (:k j)))))
    (testing "a value of one part shared by its particles is broadcast to them"
      (is (= [1.0 2.0 4.0] (particles (part/join [(wrap [1.0 2.0]) 4.0] [2 1] b)))))
    (testing "choice maps that differ in structure are held by part"
      (let [j (part/join [(cm/choicemap :a 1.0) (cm/choicemap :b 2.0)] [2 1] b)
            ps (cm/get-value j)]
        (is (batch/parts? ps))
        (is (= [[0 1] [2]] (mapv :rows (:groups ps))) "each part a group, its rows in the batch")))
    (testing "consecutive parts' groups, their rows offset to the batch's"
      (is (= [{:key :a :rows [0 1]} {:key :a :rows [2]} {:key :b :rows [3 4]}]
             (vec (part/consecutive [[{:key :a :rows [0 1]}] [{:key :a :rows [0]} {:key :b :rows [1 2]}]] [2 3])))))
    (testing "data no one batch can hold: the batched-branch error"
      (is (:gen.array/batched-branch (ts/thrown-data (part/join [:a :b] [2 1] b))) "values that are no array")
      (is (:gen.array/batched-branch (ts/thrown-data (part/join [{:a 1} {:b 1}] [2 1] b))) "maps of other keys")
      (is (:gen.array/batched-branch (ts/thrown-data (part/join [(wrap [1.0 2.0]) (batch/wrap (array/array [[1.0 2.0]]) [1])] [2 1] b)))
          "arrays of other per-particle shapes"))))

(deftest merging-groups-by-key
  (let [m (part/merge-groups [{:key :a :rows [0] :data (wrap [1.0])}
                              {:key :b :rows [1] :data (wrap [2.0])}
                              {:key :a :rows [2] :data (wrap [3.0])}]
                             (batch-of 3))]
    (is (= [[:a [0 2]] [:b [1]]] (mapv (juxt :key :rows) m)))
    (is (= [1.0 3.0] (particles (:data (first m)))))
    (is (= [2.0] (particles (:data (second m)))))))

(deftest scatter-puts-each-particle-back
  (let [b (batch-of 4)]
    (is (= [10.0 20.0 30.0 40.0] (particles (part/scatter [(wrap [30.0 10.0]) (wrap [40.0 20.0])] [[2 0] [3 1]] b))))
    (is (= {:x [10.0 20.0 30.0 40.0] :k 1}
           (update (part/scatter [{:x (wrap [30.0 10.0]) :k 1} {:x (wrap [40.0 20.0]) :k 1}] [[2 0] [3 1]] b) :x particles))
        "field by field")
    (is (:gen.array/batched-branch (ts/thrown-data (part/scatter [parts parts] [[0 1] [2 3]] b)))
        "parts held by part do not scatter")))

(deftest a-group-s-trace-backend
  (let [b (batch-of 4)]
    (is (identical? (array/backend (wrap [1.0 2.0])) (part/trace-backend {:score (wrap [1.0 2.0])} 2 b))
        "its score's")
    (is (identical? (batch-of 2) (part/trace-backend {:score 0.0} 2 b)) "a shared score: the interned one of m")))
