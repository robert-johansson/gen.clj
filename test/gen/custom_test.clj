(ns gen.custom-test
  "Custom deterministic generative functions: incremental updates through
   update-with-state, the GFI laws alone and spliced in a model, gradients,
   trainable parameters, loud errors."
  (:refer-clojure :exclude [+ *])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ *]]
            [gen.array.tape]
            [gen.choicemap :as cm]
            [gen.custom :as custom]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.laws :as laws]
            [gen.learning :as learn]
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

(def affine
  "w·x + b, w and b trainable parameters, differentiated through the code."
  (custom/deterministic {:params [:w :b] :apply (fn [[x] {:keys [w b]}] (+ (* w x) b))}))

(def line
  "y_i ~ N(affine(x_i), 1): gradients-test's regression, its mean a
   deterministic function reading the model's parameters."
  (gen [xs]
    (doseq [[i x] (map-indexed vector xs)]
      (trace [:y i] (d/normal (splice [:f i] affine x) 1)))))

(def models
  [[:deterministic sum-of-squares [[1.0 2.0 3.0]] [[[1.0 5.0 3.0]] [[2.0]]]]
   [:without-update recomputing [1.0 2.0] [[3.0 4.0]]]
   [:spliced spliced [3] [[2] [4]]]
   [:own-parameters (custom/with-params affine {:w 2.0 :b 1.0}) [3.0] [[4.0]]]
   [:the-model-s-parameters (dyn/with-params line {:w 2.0 :b 1.0}) [[1.0 2.0]] [[[1.0 3.0]] [[1.0]]]]])

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

(deftest gradients-of-trees-and-misuse
  (testing "a gradient for a tree argument reaches its leaves (it was dropped)"
    (let [f (custom/deterministic {:apply (fn [[[a b]]] (* a b)) :gradient (fn [[[a b]] _ g] [[(* g b) (* g a)]])})]
      (is (ts/close? 2.0 ((array/grad (fn [x] (f [x 2.0]))) 3.0)))
      (is (ts/close? 3.0 ((array/grad (fn [x] (f [3.0 x]))) 2.0)))))
  (testing "a cotangent of the wrong shape"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"cotangent of argument 0 has shape"
                          ((array/grad (array/custom-vjp array/sum (fn [_ _ g] [g]))) (array/array [1.0 2.0 3.0])))))
  (testing "f closing over a value being differentiated"
    (when-not (ts/native-grad?)
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"closes over"
                            ((array/value-and-grad (fn [y] ((array/custom-vjp (fn [x] (* x y)) (fn [_ _ g] [(* g 10.0)])) y)))
                             3.0)))))
  (testing "the specification"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs :apply-with-state"
                          (custom/deterministic {:apply identity :gradient-with-state (fn [& _])})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be a function"
                          (custom/deterministic {:apply identity :gradient 3})))
    (is (= 5 ((custom/deterministic {:apply count}) 1 2 3 4 5)) "five arguments")))

;; ---------------------------------------------------------------------------
;; Trainable parameters (Gen.jl's parameters of a CustomDetermGF, G5)
;; ---------------------------------------------------------------------------

(def scaled-sum
  "Σx·w, keeping Σx, with its gradients given (Gen.jl's MyDeterministicGF)."
  (custom/deterministic
   {:params [:w]
    :apply-with-state (fn [[xs] {:keys [w]}] (let [s (reduce + 0.0 xs)] [(* s w) {:sum s :n (count xs) :w w}]))
    :gradient-with-state (fn [{:keys [n w]} _ g] [(vec (repeat n (* g w)))])
    :param-gradient-with-state (fn [{:keys [sum]} _ g] {:w (* g sum)})}))

(def xs [-1.0 0.0 1.0 2.0])
(def ys [-1.1 0.9 3.2 4.8])
(def data (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

(deftest parameters-come-from-the-store-in-effect
  (testing "the function's own store"
    (is (ts/close? 7.0 ((custom/with-params affine {:w 2.0 :b 1.0}) 3.0))))
  (testing "the calling model's"
    (is (ts/close? 7.0 ((gfi/simulate (dyn/with-params line {:w 2.0 :b 1.0}) [[3.0]] (rng/key 1)) [:f 0]))))
  (testing "its own over the model's, as a dynamic function's"
    (let [m (dyn/with-params (gen [x] (splice :f (custom/with-params affine {:w 5.0}) x)) {:w 2.0 :b 1.0})]
      (is (ts/close? 16.0 (:retval (gfi/simulate m [3.0] (rng/key 2)))))))
  (testing "a missing one is loud"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"has no value" (affine 3.0)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"has no value" ((custom/with-params affine {:w 1.0}) 3.0)))))

(deftest parameter-gradients
  (testing "through the code, from the model's trace"
    (let [tr (:trace (gfi/generate (dyn/with-params line {:w 0.5 :b -0.2}) [xs] data (rng/key 3)))
          residuals (map (fn [x y] (- y (+ (* 0.5 x) -0.2))) xs ys)
          grads (grad/param-gradients tr)]
      ;; d/dw Σ log N(y; w·x + b, 1) = Σ r·x, d/db = Σ r
      (is (ts/close? (reduce + (map * residuals xs)) (:w grads)))
      (is (ts/close? (reduce + residuals) (:b grads)))))
  (testing "given, from the function's own trace (custom_determ.jl:122)"
    (let [tr (gfi/simulate (custom/with-params scaled-sum {:w 1.0}) [[1.0 2.0 3.0]] (rng/key 4))]
      (is (ts/close? 12.0 (:w (grad/param-gradients tr 2.0))) "retgrad · Σx")
      (is (ts/close? 36.0 (:w (grad/weighted-sum [3.0] [(grad/param-gradients tr 2.0)])))
          "Gen.jl's scale factor, the caller's weight")))
  (testing "given, in a model"
    (let [m (dyn/with-params (gen [v] (trace :y (d/normal (splice :f scaled-sum v) 1))) {:w 1.5})
          tr (:trace (gfi/generate m [[1.0 2.0]] (cm/choicemap :y 2.0) (rng/key 5)))]
      ;; d/dw log N(2; 3w, 1) = (2 - 3w)·3
      (is (ts/close? (* (- 2.0 4.5) 3.0) (:w (grad/param-gradients tr))))))
  (testing "with a gradient given, the function sees the parameters' values, not the tape's"
    (let [seen (atom nil)
          f (custom/deterministic {:params [:w]
                                   :apply-with-state (fn [[x] {:keys [w]}]
                                                       (reset! seen (gen.array.tape/tracked? w))
                                                       [(* w x) {:x x :w w}])
                                   :gradient-with-state (fn [{:keys [w]} _ g] [(* g w)])
                                   :param-gradient-with-state (fn [{:keys [x]} _ g] {:w (* g x)})})]
      (is (ts/close? 3.0 (:w (grad/param-gradients (gfi/simulate (custom/with-params f {:w 2.0}) [3.0] (rng/key 6)) 1.0))))
      (is (false? @seen)))))

(def remembering
  "x·w, whose update reuses the w it ran with, as Gen.jl's update_with_state
   reads state.my_param."
  (custom/deterministic {:params [:w]
                         :apply-with-state (fn [[x] {:keys [w]}] [(* w x) w])
                         :update-with-state (fn [w [x] _] [w (* w x) diff/unknown-change])}))

(deftest a-changed-store-runs-the-function-again
  ;; the call's trace is marked with the store in effect; under another,
  ;; the function runs again with the new values, not update-with-state
  ;; with the old ones
  (let [model (gen [x] (trace :y (d/normal (splice :f remembering x) 1)))
        tr (:trace (gfi/generate (dyn/with-params model {:w 2.0}) [3.0] (cm/choicemap :y 6.0) (rng/key 7)))
        {t :trace w :weight} (gfi/update (assoc tr :gen-fn (dyn/with-params model {:w 3.0})) [4.0]
                                         [diff/unknown-change] cm/EMPTY (rng/key 8))]
    (is (ts/close? 12.0 (t :f)))
    (is (ts/close? (- (:score t) (:score tr)) w))))

(deftest a-model-trains-the-function-s-parameters
  ;; gen.learning steps the model's store, which the function reads: least
  ;; squares, w = cov(x, y)/var(x) and b = ȳ - w·x̄, as gradients-test's
  ;; learning-finds-the-maximum-likelihood
  (let [mx (/ (reduce + xs) 4) my (/ (reduce + ys) 4)
        w (/ (reduce + (map #(* (- %1 mx) (- %2 my)) xs ys)) (reduce + (map #(* (- % mx) (- % mx)) xs)))
        b (- my (* w mx))
        opt (learn/sgd {:lr 0.05})
        store (reduce (fn [store i]
                        (first (learn/step opt nil store
                                           (grad/param-gradients (:trace (gfi/generate (dyn/with-params line store) [xs] data
                                                                                       (rng/key i)))))))
                      {:w 0.0 :b 0.0}
                      (range 400))]
    (is (< (ts/abs (- (:w store) w)) 1e-3))
    (is (< (ts/abs (- (:b store) b)) 1e-3))))

(deftest parameters-misuse-is-loud
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"a vector of keywords" (custom/deterministic {:apply identity :params :w})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is :gradient-with-state"
                        (custom/deterministic {:params [:w] :apply identity :gradient identity})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs :param-gradient-with-state"
                        (custom/deterministic {:params [:w] :apply-with-state identity :gradient-with-state identity})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs :params and :gradient-with-state"
                        (custom/deterministic {:apply identity :param-gradient-with-state identity})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"names its :params" (custom/with-params sum-of-squares {:w 1.0})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"a map" (custom/with-params affine [1.0]))))
