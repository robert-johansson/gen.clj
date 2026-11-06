(ns gen.array.tree-test
  "gen.array.tree: a tree's leaves depth first, map-leaves keeping the
   tree's types, skeleton and refill undoing leaves, and leaves-in
   following a skeleton's key order (genclj-w867). Planted (2026-11-06):
   leaves-in taking t's keys for the skeleton's fails
   leaves-follow-the-skeleton-s-key-order; map-leaves building maps with
   into {} fails map-leaves-keeps-the-tree-s-types (the record); refill
   taking the leaves in reverse fails refill-undoes-leaves."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
            [gen.array.tree :as tree]
            [gen.test-support :as ts]))

(defrecord Pair [a b])

(def ^:private trees
  "Trees of nested maps, vectors, lists and records, numbers at the leaves."
  (gen/recursive-gen (fn [inner]
                       (gen/one-of [(gen/vector inner 0 3)
                                    (gen/fmap #(apply list %) (gen/vector inner 0 3))
                                    (gen/map gen/keyword inner {:max-elements 3})
                                    (gen/fmap (fn [[a b]] (->Pair a b)) (gen/tuple inner inner))]))
                     (gen/fmap double gen/small-integer)))

(deftest leaves-are-depth-first
  (is (= [1 2 3 4 5] (tree/leaves (array-map :a [1 2 (list 3 [4])] :b 5))))
  (is (= [7] (tree/leaves 7)) "a leaf is its own tree")
  (is (= [] (tree/leaves {:a [] :b (list)})) "empty branches hold no leaves")
  (testing "an array is a leaf, not a branch"
    (let [x (array/array [1.0 2.0])]
      (is (= [x 3.0] (tree/leaves [x 3.0]))))))

(deftest map-leaves-keeps-the-tree-s-types
  (let [t (array-map :p (->Pair 1 [2 3]) :l (list 4 5) :v [6])
        t' (tree/map-leaves inc t)]
    (is (= {:p (->Pair 2 [3 4]) :l [5 6] :v [7]} t'))
    (is (instance? Pair (:p t')) "a record stays its record")
    (is (vector? (get-in t' [:p :b])))
    (is (seq? (:l t')) "a seq stays a seq")
    (is (= [2 3 4 5 6 7] (tree/leaves t')))))

(deftest skeletons-have-nil-leaves
  (is (= {:a [nil nil] :b (list nil)} (tree/skeleton {:a [1 2] :b (list 3)}))))

(defspec refill-undoes-leaves 100
  (prop/for-all [t trees]
    (and (= t (tree/refill t (tree/leaves t)))
         (= t (tree/refill (tree/skeleton t) (tree/leaves t)))
         (= (tree/map-leaves - t) (tree/refill (tree/skeleton t) (map - (tree/leaves t))))
         (= (tree/leaves t) (tree/leaves-in (tree/skeleton t) t)))))

(deftest leaves-follow-the-skeleton-s-key-order
  ;; equal maps may list their keys in other orders: a cached trace's
  ;; arguments follow its own skeleton, not the call's
  (let [skel (array-map :b nil :a [nil nil])
        t (array-map :a [1 2] :b 3)]
    (is (= skel (tree/skeleton t)) "equal skeletons")
    (is (= [3 1 2] (tree/leaves-in skel t)))
    (is (= [1 2 3] (tree/leaves t)))))

(deftest refill-with-host-values-of-arrays
  ;; compile and the MLX backend take a tree apart and refill it with each
  ;; leaf's evaluated value
  (let [t {:mu (array/array [1.0 2.0]) :rest [(array/array 3.0) 4.0]}
        back (tree/refill t (map ts/host (tree/leaves t)))]
    (is (= {:mu [1.0 2.0] :rest [3.0 4.0]} back))))
