(ns gen.custom-test
  "Custom deterministic generative functions: incremental updates through
   update-with-state, the GFI laws alone and spliced in a model, loud
   errors."
  (:refer-clojure :exclude [+ *])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ *]]
            [gen.array.tape]
            [gen.choicemap :as cm]
            [gen.custom :as custom]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]
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

;; ---------------------------------------------------------------------------
;; Gradients (CustomGradientGF, and CustomDetermGF's gradient_with_state)
;; ---------------------------------------------------------------------------

(def host-tanh
  "tanh by host arithmetic, which the tape cannot differentiate, with its
   gradient given."
  (custom/deterministic {:apply (fn [[x]] (Math/tanh (array/item x)))
                         :gradient (fn [_ retval retgrad] [(* retgrad (- 1.0 (* retval retval)))])}))

(def squashed
  (gen [] (let [x (trace :x (d/normal 0 1))
                y (splice :t host-tanh x)]
            (trace :y (d/normal y 0.5)))))

(def norm
  "The Euclidean norm, kept as the state its gradient uses."
  (custom/deterministic {:apply-with-state (fn [[v]] (let [n (array/sqrt (array/sum (* v v)))] [n n]))
                         :gradient-with-state (fn [n [v] g] [(* g (array// v n))])}))

(deftest gradients-pass-through
  (testing "a given gradient, through code the tape cannot see"
    (let [x 0.4 y 0.9
          tr (:trace (gfi/generate squashed [] (cm/choicemap :x x :y y) (rng/key 7)))
          {:keys [gradients]} (grad/choice-gradients tr (sel/select :x))
          th (Math/tanh x)]
      (is (ts/close? (+ (- x) (* (/ (- y th) 0.25) (- 1 (* th th)))) (cm/get-choice gradients :x)))))
  (testing "with a gradient, the function sees the arguments' values, not the tape's"
    (let [seen (atom nil)
          f (custom/deterministic {:apply (fn [[x]] (reset! seen (gen.array.tape/tracked? x)) (* x x))
                                   :gradient (fn [[x] _ g] [(* 2.0 (* g x))])})]
      (is (ts/close? 6.0 ((array/grad f) 3.0)))
      (is (false? @seen))))
  (testing "a gradient with state"
    (is (ts/close? (array/array [0.6 0.8]) ((array/grad norm) (array/array [3.0 4.0]))))
    (is (ts/close? 5.0 (norm (array/array [3.0 4.0])))))
  (testing "without one, through the function's own code"
    (let [tr (:trace (gfi/generate spliced [2] (cm/choicemap [:x 0] 0.5 [:x 1] -1.0 :y 2.0) (rng/key 8)))
          {:keys [gradients]} (grad/choice-gradients tr (sel/select [:x 0]))]
      ;; d/dx0 of log N(x0) + log N(x1) + log N(y; x0² + x1², 1)
      (is (ts/close? (+ -0.5 (* (- 2.0 1.25) 2 0.5)) (cm/get-choice gradients [:x 0])))))
  (testing "misuse"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not both"
                          (custom/deterministic {:apply identity :gradient identity :gradient-with-state identity})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"returns an array or a number"
                          ((custom/deterministic {:apply (fn [_] [1.0]) :gradient (fn [& _] [nil])}) 1.0)))))
