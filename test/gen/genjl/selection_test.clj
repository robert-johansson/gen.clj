(ns gen.genjl.selection-test
  "Gen.jl 0.4.8's test/selection.jl on gen.selection: Gen.jl's addr in s is
   (sel/selected? s addr), s[addr] is (sel/get-subselection s addr), a pair
   path :y => :z is the vector [:y :z], and set_subselection! is the
   persistent sel/set-subselection."
  (:require [clojure.test :refer [deftest is]]
            [gen.selection :as sel]))

;; selection.jl:1
(deftest dynamic-selection
  (let [s (sel/select :x [:y :z] [:y :w])]
    (is (sel/selected? s :x))
    (is (sel/selected? s [:x :u]))
    (is (sel/selected? s [:x :u :v]))
    (is (sel/selected? s [:y :z]))
    (is (sel/selected? s [:y :w]))
    (is (not (sel/selected? s :y)))
    (is (not (sel/selected? s [:y :r])))
    (is (= sel/all (sel/get-subselection s :x)))
    (let [sub (sel/get-subselection s :y)]
      (is (sel/selected? sub :z))
      (is (sel/selected? sub :w)))
    (is (= sel/none (sel/get-subselection s :u)))
    (is (= sel/all (sel/get-subselection s [:y :z])))
    (let [s (sel/set-subselection s :y (sel/select :z))]
      (is (sel/selected? s [:y :z]))
      (is (not (sel/selected? s [:y :w])))))
  (let [selection (sel/select :x)]
    (is (sel/selected? selection :x))
    (let [selection (sel/set-subselection selection :x (sel/select :y))]
      (is (sel/selected? selection [:x :y]))
      (is (not (sel/selected? selection :x))))))

;; selection.jl:36
(deftest all-selection
  (is (sel/selected? sel/all :x))
  (is (sel/selected? sel/all [:x :u]))
  (is (= sel/all (sel/get-subselection sel/all :x)))
  (is (= sel/all (sel/get-subselection sel/all [:x :y]))))

;; selection.jl:49
(deftest complement-selection
  (is (not (sel/selected? (sel/complement sel/all) :x)))
  (is (sel/selected? (sel/complement (sel/select)) :x))
  (is (not (sel/selected? (sel/complement (sel/select :x)) :x)))
  (is (sel/selected? (sel/complement (sel/select :x)) :y))
  (let [xyz (sel/select [:x :y :z])]
    (is (sel/selected? (sel/complement xyz) :x))
    (is (sel/selected? (sel/complement xyz) [:x :y]))
    (is (not (sel/selected? (sel/complement xyz) [:x :y :z])))
    (is (not (sel/selected? (sel/complement (sel/complement xyz)) :x)))
    (is (not (sel/selected? (sel/complement (sel/complement xyz)) [:x :y])))
    (is (sel/selected? (sel/complement (sel/complement xyz)) [:x :y :z]))
    (let [s (sel/get-subselection (sel/complement xyz) :x)]
      (is (not (sel/selected? s [:y :z])))
      (is (sel/selected? s :w))
      (is (sel/selected? s :y)))))

;; Not applicable: selection.jl:17, isa(sub, DynamicSelection) (one kind
;; of hierarchical selection here).
