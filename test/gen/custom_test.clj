(ns gen.custom-test
  "Custom deterministic generative functions: incremental updates through
   update-with-state, the GFI laws alone and spliced in a model, loud
   errors."
  (:refer-clojure :exclude [+ *])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ *]]
            [gen.choicemap :as cm]
            [gen.custom :as custom]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def ^:private squared
  "How many elements sum-of-squares has squared, to see updates reuse work."
  (atom 0))

(defn- square! [x] (swap! squared inc) (* x x))

(def sum-of-squares
  "Σ x², keeping the squares: an update with a vector-diff squares only the
   changed elements."
  (custom/deterministic
   {:apply-with-state (fn [[xs]] (let [sq (mapv square! xs)] [(reduce + 0.0 sq) sq]))
    :update-with-state (fn [sq [xs] [d]]
                         (let [sq (if (and (diff/vector-diff? d) (= (count sq) (count xs)))
                                    (reduce (fn [sq i] (assoc sq i (square! (xs i)))) sq (keys (:updated d)))
                                    (mapv square! xs))]
                           [sq (reduce + 0.0 sq) diff/unknown-change]))}))

(def spliced
  (gen [n]
    (let [xs (mapv #(trace [:x %] (d/normal 0 1)) (range n))
          s (splice :s sum-of-squares xs)]
      (trace :y (d/normal s 1))
      s)))

(def recomputing (custom/deterministic {:apply-with-state (fn [[a b]] [(+ a b) nil])}))

(def models
  [[:deterministic sum-of-squares [[1.0 2.0 3.0]] [[[1.0 5.0 3.0]] [[2.0]]]]
   [:without-update recomputing [1.0 2.0] [[3.0 4.0]]]
   [:spliced spliced [3] [[2] [4]]]])

(laws/deflaws custom models 100)

(deftest updates-reuse-the-state
  (let [tr (gfi/simulate sum-of-squares [[1.0 2.0 3.0]] (rng/key 1))]
    (is (ts/close? 14.0 (:retval tr)))
    (is (= [] (vec (cm/addresses (:choices tr)))) "no choices")
    (is (ts/close? 0.0 (:score tr)))
    (testing "a vector-diff update squares only the changed element"
      (reset! squared 0)
      (let [{t :trace w :weight} (gfi/update tr [[1.0 5.0 3.0]] [(diff/vector-diff 3 3 {1 diff/unknown-change})]
                                             cm/EMPTY (rng/key 2))]
        (is (= 1 @squared))
        (is (ts/close? 35.0 (:retval t)))
        (is (ts/close? 0.0 w))))
    (testing "an update without changes keeps the trace"
      (let [{t :trace rd :retdiff} (gfi/update tr [[1.0 2.0 3.0]] [diff/no-change] cm/EMPTY (rng/key 3))]
        (is (identical? tr t))
        (is (diff/no-change? rd))))
    (testing "without update-with-state, an update recomputes"
      (is (ts/close? 7.0 (:retval (:trace (gfi/update (gfi/simulate recomputing [1.0 2.0] (rng/key 4))
                                                      [3.0 4.0] [diff/unknown-change diff/unknown-change]
                                                      cm/EMPTY (rng/key 5)))))))))

(deftest it-is-a-generative-function
  (is (ts/close? 14.0 (sum-of-squares [1.0 2.0 3.0])) "an untraced call returns the value")
  (let [tr (gfi/simulate spliced [2] (rng/key 6))]
    (is (ts/close? (:retval tr) (+ (* (tr [:x 0]) (tr [:x 0])) (* (tr [:x 1]) (tr [:x 1]))))
        "spliced, its value is computed from the model's choices")))

(deftest misuse-is-loud
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"makes no choices"
                        (gfi/generate sum-of-squares [[1.0]] (cm/choicemap :x 1.0) (rng/key 1))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"makes no choices"
                        (gfi/assess sum-of-squares [[1.0]] (cm/choicemap :x 1.0))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"apply-with-state" (custom/deterministic {})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"update-with-state"
                        (custom/deterministic {:apply-with-state (fn [_] [0 nil]) :update-with-state 3}))))
