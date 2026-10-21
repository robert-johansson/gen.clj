(ns gen.genjl.selection-test
  "Gen.jl 0.4.8's test/selection.jl on gen.selection: Gen.jl's addr in s is
   (sel/selected? s addr), s[addr] is (sel/get-subselection s addr), and a
   pair path :y => :z is the vector [:y :z]. The assertions on paths wait
   for G9 (at the end)."
  (:require [clojure.test :refer [deftest is]]
            [gen.selection :as sel]))

;; selection.jl:1
(deftest dynamic-selection
  (let [s (sel/select :x [:y :z] [:y :w])]
    (is (sel/selected? s :x))
    (is (not (sel/selected? s :y)))
    (is (= sel/all (sel/get-subselection s :x)))
    (let [sub (sel/get-subselection s :y)]
      (is (sel/selected? sub :z))
      (is (sel/selected? sub :w)))
    (is (= sel/none (sel/get-subselection s :u))))
  (is (sel/selected? (sel/select :x) :x)))

;; selection.jl:36
(deftest all-selection
  (is (sel/selected? sel/all :x))
  (is (= sel/all (sel/get-subselection sel/all :x))))

;; selection.jl:49
(deftest complement-selection
  (is (not (sel/selected? (sel/complement sel/all) :x)))
  (is (sel/selected? (sel/complement (sel/select)) :x))
  (is (not (sel/selected? (sel/complement (sel/select :x)) :x)))
  (is (sel/selected? (sel/complement (sel/select :x)) :y))
  (let [xyz (sel/select [:x :y :z])]
    (is (sel/selected? (sel/complement xyz) :x))
    (is (not (sel/selected? (sel/complement (sel/complement xyz)) :x)))
    (let [s (sel/get-subselection (sel/complement xyz) :x)]
      (is (sel/selected? s :w))
      (is (sel/selected? s :y)))))

;; Pending G9, a vector is a path everywhere (genclj-cbec): paths in
;; selected? and get-subselection, and set-subselection.
;;   selection.jl:7-10, 12  [:x :u], [:x :u :v], [:y :z] and [:y :w] in s; [:y :r] not
;;   selection.jl:21        s at [:y :z] is all
;;   selection.jl:25-26     after set-subselection of :y to (select :z): [:y :z] in s, [:y :w] not
;;   selection.jl:32-33     after set-subselection of :x to (select :y): [:x :y] in it, :x not
;;   selection.jl:42, 46    [:x :u] in all; all at [:x :y] is all
;;   selection.jl:58-59     [:x :y] in the complement of (select [:x :y :z]); [:x :y :z] not
;;   selection.jl:62-63     the double complement: [:x :y] not in it, [:x :y :z] in it
;;   selection.jl:66        [:y :z] not in the complement's subselection at :x
;; Not applicable: selection.jl:17, isa(sub, DynamicSelection) (one kind
;; of hierarchical selection here).
