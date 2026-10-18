(ns gen.choicemap-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as a]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.selection :as sel]
            [gen.trace :as trace]))

;; ---------------------------------------------------------------------------
;; Generators: nested choice maps with scalar and array leaves
;; ---------------------------------------------------------------------------

(def gen-addr (gen/one-of [(gen/elements [:a :b :c :x :y]) (gen/choose 0 3)]))

(def gen-leaf
  (gen/one-of [(gen/double* {:NaN? false :infinite? false})
               (gen/fmap a/array (gen/vector (gen/double* {:NaN? false :infinite? false}) 1 3))]))

(def gen-tree
  (gen/recursive-gen (fn [inner] (gen/map gen-addr inner {:min-elements 1 :max-elements 3}))
                     gen-leaf))

(def gen-choicemap
  (gen/fmap cm/from-map (gen/map gen-addr gen-tree {:max-elements 4})))

(defn- same-leaves?
  "Same addresses, and numerically equal leaves."
  [x y]
  (and (= (cm/addresses x) (cm/addresses y))
       (every? (fn [p] (a/equal? (cm/get-choice x p) (cm/get-choice y p))) (cm/addresses x))))

;; ---------------------------------------------------------------------------
;; Choice maps
;; ---------------------------------------------------------------------------

(deftest construction-and-access
  (let [c (cm/choicemap :slope 2.0 [:y 0] 1.5 [:y 1] -1.0 :params {:a 1 :b 2})]
    (is (= 2.0 (cm/get-choice c :slope)))
    (is (= 1.5 (cm/get-choice c [:y 0])) "a vector is a path")
    (is (= 2 (cm/get-choice c [:params :b])))
    (is (nil? (cm/get-choice c :missing)))
    (is (nil? (cm/get-choice c :y)) "a node has no value")
    (is (cm/has-submap? c :y))
    (is (= {0 1.5 1 -1.0} (cm/to-map (cm/get-submap c :y))))
    (is (= [[:params :a] [:params :b] [:slope] [:y 0] [:y 1]] (cm/addresses c))
        "addresses are in canonical (sorted) order")
    (is (= [1 2 2.0 1.5 -1.0] (vec (cm/values c))))
    (is (cm/empty? cm/EMPTY))
    (is (cm/empty? (cm/from-map {:a {}})))
    (is (not (cm/empty? c)))
    (is (thrown? clojure.lang.ExceptionInfo (cm/choicemap :a))))
  (testing "values are data: equality and printing"
    (is (= (cm/choicemap :a 1.0 :b 2.0) (cm/choicemap :b 2.0 :a 1.0)))
    (is (= "#gen/choicemap {:a 1.0}" (pr-str (cm/choicemap :a 1.0))))))

(deftest functional-update
  (let [c (cm/choicemap :a 1.0)]
    (is (= 2.0 (cm/get-choice (cm/set-choice c [:b :c] 2.0) [:b :c])))
    (is (= 1.0 (cm/get-choice c :a)) "the original is unchanged")
    (is (thrown? clojure.lang.ExceptionInfo (cm/set-choice c [:a :b] 3.0))
        "descending through a leaf throws instead of discarding it"))
  (testing "merge is Gen.jl's disjoint union; override lets the right side win"
    (is (= (cm/choicemap :a 1.0 [:b :c] 2.0)
           (cm/merge (cm/choicemap :a 1.0) (cm/choicemap [:b :c] 2.0))))
    (is (thrown? clojure.lang.ExceptionInfo (cm/merge (cm/choicemap :a 1.0) (cm/choicemap :a 2.0))))
    (is (thrown? clojure.lang.ExceptionInfo (cm/merge (cm/choicemap :a 1.0) (cm/choicemap [:a :b] 2.0))))
    (is (= (cm/choicemap :a 2.0 :b 3.0)
           (cm/override (cm/choicemap :a 1.0 :b 3.0) (cm/choicemap :a 2.0))))))

(deftest arrays-and-batches
  (let [c (cm/choicemap :x (a/array [[1.0 2.0] [3.0 4.0]]) :y 5.0)]
    (is (= [1.0 2.0 3.0 4.0 5.0] (a/->clj (cm/to-array c))))
    (is (same-leaves? c (cm/from-array c (cm/to-array c))))
    (is (thrown? clojure.lang.ExceptionInfo (cm/from-array c (a/zeros [3])))))
  (let [cs [(cm/choicemap :a 1.0 [:b 0] 2.0) (cm/choicemap :a 3.0 [:b 0] 4.0)]
        stacked (cm/stack cs)]
    (is (= [1.0 3.0] (a/->clj (cm/get-choice stacked :a))))
    (is (= cs (cm/unstack stacked 2)))
    (is (thrown? clojure.lang.ExceptionInfo (cm/stack [(cm/choicemap :a 1.0) (cm/choicemap :b 1.0)])))))

(defspec to-array-from-array-round-trips 200
  (prop/for-all [c gen-choicemap]
    (same-leaves? c (cm/from-array c (cm/to-array c)))))

(defspec every-address-holds-a-value 200
  (prop/for-all [c gen-choicemap]
    (every? #(some? (cm/get-choice c %)) (cm/addresses c))))

(defspec from-map-to-map-round-trips 200
  (prop/for-all [c gen-choicemap]
    (= c (cm/from-map (cm/to-map c)))))

;; ---------------------------------------------------------------------------
;; Selections
;; ---------------------------------------------------------------------------

(deftest selection-algebra
  (let [s (sel/select :x [:y 0])]
    (is (sel/selected? s :x))
    (is (not (sel/selected? s :y)) "selecting [:y 0] does not select :y itself")
    (is (sel/selected? (sel/get-subselection s :y) 0))
    (is (not (sel/selected? (sel/get-subselection s :y) 1)))
    (is (sel/selected? (sel/get-subselection s :x) :anything) "selecting :x selects everything under it")
    (is (sel/selected? (sel/complement s) :z))
    (is (= s (sel/complement (sel/complement s))))
    (is (= sel/none (sel/complement sel/all))))
  (let [c (cm/choicemap :x 1.0 [:y 0] 2.0 [:y 1] 3.0 :z 4.0)]
    (is (= (cm/choicemap :x 1.0 [:y 0] 2.0) (sel/get-selected c (sel/select :x [:y 0]))))
    (is (= (cm/choicemap [:y 1] 3.0 :z 4.0)
           (sel/get-selected c (sel/complement (sel/select :x [:y 0])))))
    (is (= c (sel/get-selected c sel/all)))
    (is (cm/empty? (sel/get-selected c sel/none)))
    (is (= c (sel/get-selected c (sel/from-choicemap c))))))

(def gen-choicemap-and-selection
  (gen/let [c gen-choicemap
            picks (gen/vector gen/boolean (count (cm/addresses c)))]
    [c (sel/from-paths (keep (fn [[p pick?]] (when pick? p)) (map vector (cm/addresses c) picks)))]))

(defspec a-selection-and-its-complement-partition-the-choices 300
  (prop/for-all [[c s] gen-choicemap-and-selection]
    (let [in (sel/get-selected c s)
          out (sel/get-selected c (sel/complement s))]
      (and (= (count (cm/addresses c)) (+ (count (cm/addresses in)) (count (cm/addresses out))))
           (same-leaves? c (cm/merge in out))))))

;; ---------------------------------------------------------------------------
;; Traces and diffs
;; ---------------------------------------------------------------------------

(deftest traces
  (let [tr (trace/make-trace {:gen-fn :model :args [1 2] :retval 3.0 :score -1.5
                              :choices (cm/choicemap :slope 2.0 [:y 0] 1.0)})]
    (is (trace/trace? tr))
    (is (= 2.0 (tr :slope)) "calling a trace on an address reads the choice, like trace[addr]")
    (is (= 1.0 (tr [:y 0])))
    (is (= -1.5 (:score tr)))
    (is (= [1 2] (:args tr)))))

(deftest diffs
  (is (diff/no-change? diff/no-change))
  (is (not (diff/changed? diff/no-change)))
  (is (diff/changed? diff/unknown-change))
  (is (= [diff/no-change diff/no-change] (diff/no-changes 2)))
  (let [d (diff/vector-diff 5 3 {1 diff/unknown-change})]
    (is (diff/vector-diff? d))
    (is (not (diff/element-changed? d 0)))
    (is (diff/element-changed? d 1))
    (is (diff/element-changed? d 3) "new elements count as changed")
    (is (diff/element-changed? diff/unknown-change 0))))
