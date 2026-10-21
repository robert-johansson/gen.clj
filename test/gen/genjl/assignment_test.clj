(ns gen.genjl.assignment-test
  "Gen.jl 0.4.8's test/assignment.jl on gen.choicemap. There is one kind
   of choice map here: Gen.jl's static ones are built by cm/choicemap from
   their values and submaps, its dynamic ones by functional updates
   (set-choice and set-submap, for set_value! and set_submap!), and the
   testsets they share check the same assertions. A pair path :d => :a is
   the vector [:d :a], c[addr] is (cm/get-choice c addr), a vector-valued
   leaf is an array (a Clojure vector is a product value), and Map's index
   addresses are 0-based (D2). genjl/shallow checks Gen.jl's pairs of
   shallow counts in one assertion. Lookups are loud where Gen.jl throws
   KeyError (D4), with the error's :reason checked, but get_submap at a
   value is that value, a choice map here: D4's one difference
   (DESIGN.md §9)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.genjl.support :as genjl]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(defn- choice "c[addr], read on the host." [c addr] (ts/host (cm/get-choice c addr)))

(defn- to-and-from-array
  "assignment.jl's to_array and from_array assertions on outer: :c 3.0,
   and under each of :d and :e, :a 1.0 and :b [2.0 2.5]."
  [outer]
  (is (= [3.0 1.0 2.0 2.5 1.0 2.0 2.5] (ts/host (cm/to-array outer))))
  (let [choices (cm/from-array outer (array/array [1.0 2.0 3.0 4.0 5.0 6.0 7.0]))]
    (is (= 1.0 (choice choices :c)))
    (is (= 2.0 (choice choices [:d :a])))
    (is (= [3.0 4.0] (choice choices [:d :b])))
    (is (= 5.0 (choice choices [:e :a])))
    (is (= [6.0 7.0] (choice choices [:e :b])))
    (is (= {:values 1 :submaps 2} (genjl/shallow choices)))
    (doseq [addr [:d :e]]
      (testing addr
        (is (= {:values 2 :submaps 0} (genjl/shallow (cm/get-submap choices addr))))))))

;; assignment.jl:1
(deftest static-assignment-to-from-array
  (let [submap (cm/choicemap :a 1.0 :b (array/array [2.0 2.5]))]
    (to-and-from-array (cm/choicemap :c 3.0 :d submap :e submap))))

;; assignment.jl:24
(deftest dynamic-assignment-to-from-array
  (let [submap (-> cm/EMPTY (cm/set-choice :a 1.0) (cm/set-choice :b (array/array [2.0 2.5])))]
    (to-and-from-array (-> cm/EMPTY (cm/set-choice :c 3.0) (cm/set-submap :d submap) (cm/set-submap :e submap)))))

;; assignment.jl:52. A persistent choice map is its own copy.
(deftest dynamic-assignment-copy-constructor
  (let [nested (-> cm/EMPTY (cm/set-choice :z 3) (cm/set-choice :w 4))
        choices (-> cm/EMPTY (cm/set-choice :x 1) (cm/set-choice :y 2) (cm/set-submap :u nested))]
    (is (= 1 (choice choices :x)))
    (is (= 2 (choice choices :y)))
    (is (= 3 (choice choices [:u :z])))
    (is (= 4 (choice choices [:u :w])))))

;; assignment.jl:67. Gen.jl's vectorize_internal is an index-keyed node,
;; as Map's choice maps are.
(deftest internal-vector-assignment-to-from-array
  (let [inner (-> cm/EMPTY (cm/set-choice :a 1.0) (cm/set-choice :b 2.0))
        outer (reduce-kv cm/set-submap cm/EMPTY [inner inner inner])
        choices (cm/from-array outer (array/array [1.0 2.0 3.0 4.0 5.0 6.0]))]
    (is (= [1.0 2.0 1.0 2.0 1.0 2.0] (ts/host (cm/to-array outer))))
    (is (= 1.0 (choice choices [0 :a])))
    (is (= 2.0 (choice choices [0 :b])))
    (is (= 3.0 (choice choices [1 :a])))
    (is (= 4.0 (choice choices [1 :b])))
    (is (= 5.0 (choice choices [2 :a])))
    (is (= 6.0 (choice choices [2 :b])))
    (is (= 3 (count (cm/submaps-shallow choices))))))

(defn- merged
  "assignment.jl's merge assertions on (cm/merge choices1 choices2): :a 1.0
   and :b 2.0 with :c and :shared {:x 1}, and :d 3.0 with :e and :f {:x 1}
   and :shared {:y 4.0}."
  [choices1 choices2]
  (let [choices (cm/merge choices1 choices2)]
    (is (= 1.0 (choice choices :a)))
    (is (= 2.0 (choice choices :b)))
    (is (= 3.0 (choice choices :d)))
    (is (= 1 (choice choices [:c :x])))
    (is (= 1 (choice choices [:e :x])))
    (is (= 1 (choice choices [:f :x])))
    (is (= 1 (choice choices [:shared :x])))
    (is (= 4.0 (choice choices [:shared :y])))
    (is (= {:values 3 :submaps 4} (genjl/shallow choices)))))

;; assignment.jl:86
(deftest dynamic-assignment-merge
  (let [submap (cm/set-choice cm/EMPTY :x 1)
        submap2 (cm/set-choice cm/EMPTY :y 4.0)]
    (merged (-> cm/EMPTY (cm/set-choice :a 1.0) (cm/set-choice :b 2.0)
                (cm/set-submap :c submap) (cm/set-submap :shared submap))
            (-> cm/EMPTY (cm/set-choice :d 3.0) (cm/set-submap :e submap)
                (cm/set-submap :f submap) (cm/set-submap :shared submap2)))))

;; assignment.jl:123
(deftest static-assignment-merge
  (let [submap (cm/choicemap :x 1)
        submap2 (cm/choicemap :y 4.0)]
    (merged (cm/choicemap :a 1.0 :b 2.0 :c submap :shared submap)
            (cm/choicemap :d 3.0 :e submap :f submap :shared submap2))))

;; assignment.jl:114 and :143, whose dynamic and static choice maps are
;; the same here
(deftest variadic-merge
  (let [choices1 (cm/choicemap :a 1)
        choices2 (cm/choicemap :b 2)
        choices3 (cm/choicemap :c 3)]
    (is (= choices1 (cm/merge choices1)))
    (is (= (cm/choicemap :a 1 :b 2 :c 3) (cm/merge choices1 choices2 choices3)))))

(defn- assignment-errors
  "assignment.jl's errors on choice maps that make builds from address-value
   pairs: get_submap through a value and get_value where no value is are
   loud (Gen.jl's KeyError), get_submap where nothing is is empty, and
   get_submap at a value is that value (D4's difference)."
  [make]
  (testing "get_submap at a value: the value, where Gen.jl throws KeyError"
    (is (= (cm/->Value 1) (cm/get-submap (make :x 1) :x))))
  (testing "get_submap through a value"
    (is (= :through-value (:reason (ts/thrown-data (cm/get-submap (make :x 1) [:x :y]))))))
  (testing "get_submap where nothing is"
    (is (cm/empty? (cm/get-submap (make) :x)))
    (is (cm/empty? (cm/get-submap (make) [:x :y]))))
  (testing "get_value at a submap"
    (is (= :node (:reason (ts/thrown-data (cm/get-choice (make [:x :y] 1) :x))))))
  (testing "get_value where nothing is"
    (is (= :absent (:reason (ts/thrown-data (cm/get-choice (make) :x)))))
    (is (= :absent (:reason (ts/thrown-data (cm/get-choice (make) [:x :y])))))))

;; assignment.jl:152. Gen.jl's static_get_submap and static_get_value are n/a
(deftest static-assignment-errors
  (assignment-errors cm/choicemap))

;; assignment.jl:221, the choice maps built by set_value!
(deftest dynamic-assignment-errors
  (assignment-errors (fn [& kvs] (reduce (fn [c [a v]] (cm/set-choice c a v)) cm/EMPTY (partition 2 kvs)))))

;; assignment.jl:259
(deftest dynamic-assignment-overwrite
  (testing "a value with a value"
    (is (= 2 (choice (-> cm/EMPTY (cm/set-choice :x 1) (cm/set-choice :x 2)) :x))))
  (testing "a value with a submap"
    (let [choices (cm/set-choice cm/EMPTY :x 1)]
      (is (genjl/has-value? choices :x))
      (is (not (cm/has-submap? choices :x)))
      (let [choices (cm/set-submap choices :x (cm/set-choice cm/EMPTY :y 2))]
        (is (not (genjl/has-value? choices :x)))
        (is (not (cm/empty? (cm/get-submap choices :x))))
        (is (cm/has-submap? choices :x)))))
  (testing "a submap with a value"
    (let [choices (-> cm/EMPTY (cm/set-choice [:x :y] 1) (cm/set-choice :x 2))]
      (is (not (cm/has-submap? choices :x)))
      (is (= (cm/->Value 2) (cm/get-submap choices :x)) "the value, where Gen.jl throws KeyError (D4)")
      (is (= 2 (choice choices :x)))))
  (testing "a submap with a submap"
    (let [choices (cm/set-choice cm/EMPTY [:x :y] 1)]
      (is (cm/has-submap? choices :x))
      (is (not (cm/has-submap? choices [:x :y])))
      (let [choices (cm/set-submap choices :x (cm/set-choice cm/EMPTY :z 2))]
        (is (not (cm/empty? (cm/get-submap choices :x))))
        (is (not (genjl/has-value? choices [:x :y])))
        (is (= 2 (choice choices [:x :z]))))))
  (testing "a value or a submap set under a value"
    (is (thrown? clojure.lang.ExceptionInfo (cm/set-choice (cm/set-choice cm/EMPTY :x 1) [:x :y] 2)))
    ;; upstream sets :z on choices, not on the submap, which stays empty
    (let [choices (-> cm/EMPTY (cm/set-choice :x 1) (cm/set-choice :z 2))]
      (is (thrown? clojure.lang.ExceptionInfo (cm/set-choice choices [:x :y] cm/EMPTY))))))

;; assignment.jl:315
(deftest dynamic-assignment-constructor
  (let [choices (cm/choicemap :x 1 [:y :a] 2)]
    (is (= 1 (choice choices :x)))
    (is (= 2 (choice choices [:y :a])))))

;; assignment.jl:322. Upstream's (:b => :c => 2) is the pair (:b, :c => 2),
;; a value at :b; the evident intent is the path.
(deftest choice-map-equality
  (let [a (cm/choicemap :a 1 [:b :c] 2 [:d :e :f] 3)
        b (cm/choicemap :a 1 [:d :e :f] 3)
        c (cm/choicemap :a 1 [:b :c] 2 [:d :e :f] 4)]
    (is (= a a))
    (is (= b b))
    (is (not= a b))
    (is (not= b a))
    (is (not= a c))
    (is (not= c a))))

;; assignment.jl:337. Gen.jl's nested_view is cm/to-map.
(deftest choice-map-nested-view
  (let [c (cm/choicemap :a 1 [:b :c] 2)
        cv (cm/to-map c)]
    (is (= (cm/get-choice c :a) (:a cv)))
    (is (= (cm/get-choice c [:b :c]) (-> cv :b :c)))
    (is (= 2 (count cv)))
    (is (= 1 (count (:b cv))))
    (is (= #{:a :b} (set (keys cv))))
    (is (= #{:c} (set (keys (:b cv)))))
    (is (= cv (cm/to-map (cm/choicemap :a 1 [:b :c] 2))))
    (is (not= cv (cm/to-map (cm/choicemap :a 4 [:b :c] 2))))))

;; assignment.jl:358
(deftest filtering-choicemaps-with-selections
  (let [filtered (sel/get-selected (cm/choicemap :a 1 :b 2) (sel/select :a))]
    (is (= 1 (choice filtered :a)))
    (is (not (genjl/has-value? filtered :b))))
  (let [c (cm/choicemap [:x :y] 1 [:x :z] 2)]
    (let [filtered (sel/get-selected c (sel/select :x))]
      (is (= 1 (choice filtered [:x :y])))
      (is (= 2 (choice filtered [:x :z]))))
    (let [filtered (sel/get-selected c (sel/select [:x :y]))]
      (is (= 1 (choice filtered [:x :y])))
      (is (not (genjl/has-value? filtered [:x :z]))))))

;; assignment.jl:377. A malformed pair is an address without a value here.
(deftest invalid-choice-map-constructor
  (is (thrown? clojure.lang.ExceptionInfo (cm/choicemap :a 1 :b))))

;; assignment.jl:383
(deftest choicemap-hashing
  (let [d {(cm/choicemap :x 1) 2
           (cm/choicemap :x 2 :y 3) 5
           (cm/choicemap :x 3 [:y :z] 4) 7
           (cm/choicemap :x 3 [:y :w] 5 [:y :z] 2) 10}]
    (is (not (contains? d (cm/choicemap))))
    (is (not (contains? d cm/EMPTY)))
    (is (not (contains? d (cm/choicemap :a 3))))
    (is (not (contains? d (cm/choicemap :x 2))))
    (is (= 2 (d (cm/choicemap :x 1))))
    (is (= 5 (d (cm/choicemap :x 2 :y 3))))
    (is (= 7 (d (cm/choicemap :x 3 [:y :z] 4))))
    (is (= 10 (d (cm/choicemap :x 3 [:y :w] 5 [:y :z] 2))))
    (testing "in another order"
      (is (= 5 (d (cm/choicemap :y 3 :x 2))))
      (is (= 7 (d (cm/choicemap [:y :z] 4 :x 3))))
      (is (= 10 (d (cm/choicemap [:y :z] 2 [:y :w] 5 :x 3)))))
    (is (= 10 ({cm/EMPTY 10} (cm/choicemap))))
    (testing "no two of d's keys hash alike"
      (is (apply distinct? (map hash (keys d)))))))

;; Not applicable: static_get_submap and static_get_value
;; (assignment.jl:164, 176, 187, 203, 218), and the conversion to a
;; StaticChoiceMap (assignment.jl:404).
