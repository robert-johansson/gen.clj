(ns gen.genjl.diff-test
  "Gen.jl 0.4.8's test/diff.jl on gen.diff's propagation rules: + (on
   clojure.core's and gen.array's), length (count) and getindex (nth and
   get, 0-based: D2). Gen.jl's Diffed, a value with its diff, is a Diffed
   here too; an argument that is not one is unchanged, as in Gen.jl."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.diff :as diff]
            [gen.test-support :as ts]))

(defrecord Diffed [value diff])

(defn- diffed [value d] (->Diffed value d))

(defn- call
  "Gen.jl's f(args...) on Diffed arguments: f's value, with the diff
   gen.diff propagates through f (a var)."
  [f & args]
  (let [args (map #(if (instance? Diffed %) % (diffed % diff/no-change)) args)]
    (diffed (apply f (map :value args))
            (diff/propagate f (mapv :value args) (mapv :diff args)))))

;; diff.jl:3
(deftest diff-arithmetic
  (doseq [plus [#'clojure.core/+ #'array/+]
          [a b d] [[(diffed 1 diff/no-change) (diffed 2 diff/no-change) diff/no-change]
                   [1 (diffed 2 diff/no-change) diff/no-change]
                   [(diffed 1 diff/no-change) 2 diff/no-change]
                   [(diffed 1 diff/unknown-change) (diffed 2 diff/unknown-change) diff/unknown-change]
                   [1 (diffed 2 diff/unknown-change) diff/unknown-change]
                   [(diffed 1 diff/unknown-change) 2 diff/unknown-change]]]
    (testing (pr-str plus a b)
      (let [sum (call plus a b)]
        (is (= d (:diff sum)))
        (is (== 3 (ts/host (:value sum))))))))

;; diff.jl:40
(deftest diff-vectors-length
  (doseq [[d length-diff] [[diff/no-change diff/no-change]
                           [diff/unknown-change diff/unknown-change]
                           [(diff/vector-diff 3 4 {}) (diff/int-diff -1)]]]
    (testing (pr-str d)
      (let [length (call #'count (diffed [1 2 3] d))]
        (is (= 3 (:value length)))
        (is (= length-diff (:diff length)))))))

;; diff.jl:56. Gen.jl's index 2 is 1 here.
(deftest diff-vectors-getindex
  (let [v [1 2 3]
        element-diff (diff/int-diff -3)]
    (doseq [lookup [#'nth #'get]
            [v i d] [[v (diffed 1 diff/unknown-change) diff/unknown-change]
                     [v (diffed 1 diff/no-change) diff/no-change]
                     [v (diffed 1 (diff/int-diff 1)) diff/unknown-change]
                     [(diffed v diff/no-change) 1 diff/no-change]
                     [(diffed v diff/unknown-change) 1 diff/unknown-change]
                     ;; the value at index 1 was not changed
                     [(diffed v (diff/vector-diff 3 4 {})) 1 diff/no-change]
                     ;; the value at index 1 was updated (reduced by 3)
                     [(diffed v (diff/vector-diff 3 4 {1 element-diff})) 1 element-diff]]]
      (testing (pr-str lookup v i)
        (let [element (call lookup v i)]
          (is (= 2 (:value element)))
          (is (= d (:diff element))))))))

;; diff.jl:94 and :97, "diff dictionaries" and "diff sets", are empty upstream.
