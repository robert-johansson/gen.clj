(ns gen.choicemap-test
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
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
               (gen/fmap array/array (gen/vector (gen/double* {:NaN? false :infinite? false}) 1 3))]))

(def gen-tree
  (gen/recursive-gen (fn [inner] (gen/map gen-addr inner {:min-elements 1 :max-elements 3}))
                     gen-leaf))

(def gen-choicemap
  (gen/fmap cm/from-map (gen/map gen-addr gen-tree {:max-elements 4})))

(defn- same-leaves?
  "Same addresses, and numerically equal leaves."
  [x y]
  (and (= (cm/addresses x) (cm/addresses y))
       ;; equal, or on a float32 backend (where arrays round host doubles) close
       (every? (fn [p] (let [u (cm/get-choice x p) v (cm/get-choice y p)]
                         (if (ts/f32?) (ts/close? u v) (array/equal? u v))))
               (cm/addresses x))))

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

(deftest a-choice-map-value-is-itself
  ;; nodes and values are records, so map? is true of them: choicemap and
  ;; from-map once read their fields as addresses ([:d :m :a :v]), the
  ;; natural port of assignment.jl's StaticChoiceMap with submaps (M5)
  (let [sub (cm/choicemap :a 1.0 :b 2.0)
        expected (cm/choicemap [:d :a] 1.0 [:d :b] 2.0 :c 3.0)]
    (is (= expected (cm/choicemap :c 3.0 :d sub)))
    (is (= expected (cm/from-map {:c 3.0 :d sub})))
    (is (= expected (cm/from-map (cm/from-map {:c 3.0 :d sub}))) "a choice map is its own from-map")
    (is (= [[:c] [:x :d :a] [:x :d :b]] (cm/addresses (cm/from-map {:c 3.0 :x {:d sub}}))) "nested in a plain map")
    (is (= 2.0 (cm/get-choice (cm/choicemap :d (cm/->Value 2.0)) :d)) "a leaf choice map")
    (is (= [[:d]] (cm/addresses (cm/choicemap :d (cm/->Value 2.0)))))))

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
  (let [c (cm/choicemap :x (array/array [[1.0 2.0] [3.0 4.0]]) :y 5.0)]
    (is (= [1.0 2.0 3.0 4.0 5.0] (array/->clj (cm/to-array c))))
    (is (same-leaves? c (cm/from-array c (cm/to-array c))))
    (is (thrown? clojure.lang.ExceptionInfo (cm/from-array c (array/zeros [3]))))
    (testing "leaves come back in the prototype's dtypes"
      (let [mixed (cm/choicemap :b true :k 2 :x 0.5)
            back (cm/from-array mixed (cm/to-array mixed))]
        (is (= (ts/host mixed) (ts/host back)))
        (is (instance? Boolean (ts/host (cm/get-choice back :b))))
        (is (instance? Long (ts/host (cm/get-choice back :k))))
        (is (= [:bool :int] (map #(gen.array.dtype/kind (array/dtype (cm/get-choice back %))) [:b :k]))))))
  (let [cs [(cm/choicemap :a 1.0 [:b 0] 2.0) (cm/choicemap :a 3.0 [:b 0] 4.0)]
        stacked (cm/stack cs)]
    (is (= [1.0 3.0] (array/->clj (cm/get-choice stacked :a))))
    (is (= (ts/host cs) (ts/host (cm/unstack stacked 2))))
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

;; A vector is a path, everywhere (DESIGN.md §9, M5's G9 and F3): every
;; function that takes an address takes a path too, or is loud on one.
(deftest a-vector-is-a-path
  (testing "in selections"
    (let [s (sel/select :x [:y 0])]
      (is (sel/selected? s [:y 0]))
      (is (sel/selected? s [:x :u]) "selecting :x selects every path under it")
      (is (not (sel/selected? s [:y 1])))
      (is (= sel/all (sel/get-subselection s [:y 0])))
      (is (= s (sel/get-subselection s [])) "the empty path is s itself")
      (is (sel/selected? sel/all []) "and the root")
      (is (not (sel/selected? s [])))
      (is (= s (sel/hierarchical :x sel/all [:y 0] sel/all)) "hierarchical nests a path")
      (is (= (sel/select [:y 0] [:y 1]) (sel/set-subselection (sel/select [:y 0]) [:y 1] sel/all)))
      (is (= (sel/select :z) (sel/set-subselection sel/none :z sel/all)))
      (doseq [[why f] [["inside all" #(sel/set-subselection sel/all :x sel/none)]
                       ["inside a selected address" #(sel/set-subselection s [:x :u] sel/none)]
                       ["inside a complement" #(sel/set-subselection (sel/complement s) :z sel/all)]
                       ["inside another kind of selection"
                        #(sel/set-subselection (reify sel/ISelection (-selected? [_ _] false) (-subselection [this _] this))
                                               :z sel/all)]
                       ["not a selection" #(sel/set-subselection sel/none :z true)]
                       ["a vector inside a path" #(sel/select [[:y 0] :z])]
                       ["a vector inside a path" #(sel/hierarchical [[:y] 0] sel/all)]
                       ["a vector inside a path, read" #(sel/selected? s [:y [0]])]
                       ["a vector inside a path, read" #(sel/get-subselection s [:y [0]])]
                       ["a dangling address" #(sel/hierarchical :x)]]]
        (is (thrown? clojure.lang.ExceptionInfo (f)) why))))
  (testing "in choice maps"
    (let [c (cm/choicemap :x 1.0)]
      (is (= (cm/choicemap :x 1.0 [:y 0 :z] 2.0) (cm/set-submap c [:y 0] (cm/choicemap :z 2.0))))
      (is (= (cm/->Value 2.0) (cm/set-choice c [] 2.0)) "the empty path is the root")
      (is (= (cm/choicemap :z 2.0) (cm/set-submap c [] (cm/choicemap :z 2.0))))
      (is (= (cm/choicemap [:a :b] 1 [:a :c] 2 :d 3) (cm/from-map {[:a :b] 1 [:a :c] 2 :d 3}))
          "a map's vector key is a path")
      (is (= (cm/choicemap [:x :y 0] 1) (cm/choicemap :x {[:y 0] 1})))
      (doseq [[why f] [["set-value takes one address" #(cm/set-value c [:y 0] 2.0)]
                       ["remove-submap takes one address" #(cm/remove-submap c [:x])]
                       ["a submap is a choice map" #(cm/set-submap c :y 2.0)]
                       ["a submap under a value" #(cm/set-submap c [:x :y] (cm/choicemap :z 2.0))]
                       ["a submap under a root value" #(cm/set-submap (cm/->Value 1.0) :y (cm/choicemap :z 2.0))]
                       ["a vector inside a path" #(cm/set-choice c [[:y 0] :z] 2.0)]
                       ["a map's paths that overlap" #(cm/from-map {[:a :b] 1 :a 2})]
                       ["a vector inside a path, read" #(cm/get-submap c [[:x] :y])]
                       ["a vector inside a path, read" #(cm/get-choice c [:x [:y]])]]]
        (is (thrown? clojure.lang.ExceptionInfo (f)) why)))))

(deftest merge-takes-any-number
  (is (= cm/EMPTY (cm/merge)))
  (is (= (cm/choicemap :a 1 [:b :c] 2 [:b :d] 3)
         (cm/merge (cm/choicemap :a 1) (cm/choicemap [:b :c] 2) (cm/choicemap [:b :d] 3))))
  (is (thrown? clojure.lang.ExceptionInfo (cm/merge (cm/choicemap :a 1) cm/EMPTY (cm/choicemap :a 2)))))

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

(deftest pair-and-unpair
  (let [a (cm/choicemap :x 1.0 [:y 0] 2.0)
        b (cm/choicemap :z 3.0)
        c (cm/pair a b :left :right)]
    (is (= #{[:left :x] [:left :y 0] [:right :z]} (set (cm/addresses c))))
    (is (= [(cm/to-map a) (cm/to-map b)] (mapv cm/to-map (cm/unpair c :left :right))) "unpair undoes pair")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"more than" (cm/unpair (cm/set-choice c :extra 1.0) :left :right)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must differ" (cm/pair a b :k :k)))))

(deftest pair-takes-single-addresses
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not paths" (cm/pair (cm/choicemap :x 1.0) cm/EMPTY [:a 0] :b)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not paths" (cm/unpair cm/EMPTY [:a 0] :b))))
