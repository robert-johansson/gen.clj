(ns gen.array.tape-test
  "Reverse-mode autodiff: every VJP against central finite differences in
   float64, as properties over random inputs (and broadcasting shapes)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
            [gen.array.linalg :as la]
            [gen.dist :as d]
            [gen.dist.core :as dc]))

;; ---------------------------------------------------------------------------
;; Finite differences
;; ---------------------------------------------------------------------------

(defn- flat [x] (vec (flatten (vector (array/->clj x)))))

(defn- from-flat [xs shape]
  (if (empty? shape) (first xs) (array/reshape (array/array (vec xs)) shape)))

;; Finite differences run on the JVM backend in float64 whatever the backend
;; under test, since a float32 backend's differences would be noise
(def ^:private reference (array/backend-named :jvm))

(defn- on-reference
  "x as a float64 array of the reference backend (host scalars as they are)."
  [x]
  (if (array/host? x) (double x) (array/with-backend reference (array/array (array/->clj x) {:dtype :float64}))))

(defn- fd-grad
  "The central-difference gradient of the scalar function f at x, in float64
   on the reference backend (x on it)."
  [f x]
  (array/with-backend reference
    (let [xs (flat x)
          s (array/shape x)
          h 1e-6]
      (from-flat (for [i (range (count xs))]
                   (let [at (fn [d] (double (array/item (f (from-flat (update xs i + d) s)))))]
                     (/ (- (at h) (at (- h))) (* 2 h))))
                 s))))

(defn- close-grads? [g fd]
  (let [g (on-reference g)]
    (and (= (array/shape g) (array/shape fd))
         (array/allclose? g fd (if (ts/f32?) {:rtol 2e-3 :atol 2e-4} {:rtol 1e-5 :atol 1e-6})))))

(defn- grad-matches?
  "Does f's gradient with respect to each argument match finite differences?"
  [f & xs]
  (let [xs (vec xs)
        grads (apply (array/grad f {:argnums (vec (range (count xs)))}) xs)
        ref-xs (mapv on-reference xs)]
    (every? true?
            (for [i (range (count xs))]
              (close-grads? (nth grads i)
                            (fd-grad #(apply f (assoc ref-xs i %)) (nth ref-xs i)))))))

(defn- projected
  "A scalar function of f's arguments: f's output weighted by fixed
   pseudo-random weights, so the check covers the whole VJP."
  [f]
  (fn [& xs]
    (let [out (apply f xs)
          w (from-flat (map #(+ 0.5 (* 0.37 (Math/sin (* 1.7 (inc %))))) (range (array/size out))) (array/shape out))]
      (array/sum (array/* out w)))))

;; ---------------------------------------------------------------------------
;; Elementwise ops, with broadcasting
;; ---------------------------------------------------------------------------

(def shape-pairs [[[] []] [[3] []] [[] [3]] [[2 3] [3]] [[2 1] [1 3]] [[2 3] [2 3]] [[3 1] [3]]])

(def unary-ops
  [[:negative array/negate -3 3] [:abs array/abs 0.1 3] [:exp array/exp -3 3] [:log array/log 0.1 5]
   [:expm1 array/expm1 -2 2] [:log1p array/log1p -0.9 4] [:sqrt array/sqrt 0.1 5] [:sin array/sin -3 3]
   [:cos array/cos -3 3] [:tan array/tan -1.2 1.2] [:tanh array/tanh -3 3] [:erf array/erf -2 2]
   [:erfinv array/erfinv -0.9 0.9] [:square array/square -3 3] [:softplus array/softplus -5 5]
   [:sigmoid array/sigmoid -5 5] [:lgamma array/lgamma 0.2 6] [:lgamma-reflected array/lgamma -3.7 -3.2]
   [:digamma array/digamma 0.3 6]])

(defn- scaled
  "An array of shape from uniforms us in [0, 1], mapped to [lo, hi]."
  [us lo hi shape]
  (from-flat (map #(+ lo (* (- hi lo) %)) (take (reduce * 1 shape) us)) shape))

(def gen-uniforms (gen/vector (gen/double* {:min 0 :max 1 :NaN? false}) 6))

(defspec unary-vjps-match-finite-differences 100
  (prop/for-all [[_ f lo hi] (gen/elements unary-ops)
                 shape (gen/elements [[] [3] [2 2]])
                 us gen-uniforms]
    (grad-matches? (projected f) (scaled us lo hi shape))))

(def binary-ops
  [[:add array/+ [-3 3] [-3 3]] [:subtract array/- [-3 3] [-3 3]] [:multiply array/* [-3 3] [-3 3]]
   [:divide array// [-3 3] [0.5 3]] [:power array/pow [0.2 3] [-2 2]] [:maximum array/maximum [-3 3] [-3 3]]
   [:minimum array/minimum [-3 3] [-3 3]] [:logaddexp array/logaddexp [-3 3] [-3 3]]
   [:xlogy array/xlogy [-3 3] [0.2 3]] [:log-beta array/log-beta [0.3 4] [0.3 4]]])

(defspec binary-vjps-match-finite-differences 100
  (prop/for-all [[label f [xlo xhi] [ylo yhi]] (gen/elements binary-ops)
                 [sx sy] (gen/elements shape-pairs)
                 xs gen-uniforms
                 ys gen-uniforms]
    (let [x (scaled xs xlo xhi sx)
          y (scaled ys ylo yhi sy)]
      ;; maximum/minimum are not differentiable at ties: skip near-ties
      (or (and (#{:maximum :minimum} label) (array/truthy? (array/any (array/< (array/abs (array/- x y)) 1e-3))))
          (grad-matches? (projected f) x y)))))

(deftest where-and-remainder
  (is (grad-matches? (projected (fn [x y] (array/where (array/> x 0) (array/* x y) (array/exp y))))
                     (array/array [-1.0 0.5 2.0]) (array/array [0.3 -0.2 1.1])))
  (is (grad-matches? (projected (fn [x y] (array/mod x y))) (array/array [2.3 -1.7 5.2]) (array/array [1.5 0.9 2.0]))))

;; ---------------------------------------------------------------------------
;; Reductions, scans, shapes, indexing
;; ---------------------------------------------------------------------------

(def m23 (array/array [[0.3 -1.2 2.0] [1.5 0.7 -0.4]]))

(deftest reductions-and-scans
  (doseq [[label f] [[:sum array/sum] [:sum-axis #(array/sum % {:axis 1})] [:sum-keep #(array/sum % {:axis 0 :keep-dims? true})]
                     [:prod array/prod] [:prod-axis #(array/prod % {:axis 1})] [:max array/max] [:min-axis #(array/min % {:axis 0})]
                     [:mean array/mean] [:variance #(array/variance % {:axis 1})] [:std array/std] [:logsumexp array/logsumexp]
                     [:logsumexp-axis #(array/logsumexp % {:axis 1})] [:softmax #(array/softmax % {:axis -1})]
                     [:log-softmax array/log-softmax] [:cumsum #(array/cumsum % {:axis 1})]
                     [:cumsum-rev #(array/cumsum % {:axis 0 :reverse? true})] [:cumprod #(array/cumprod % {:axis 1})]
                     [:logcumsumexp #(array/logcumsumexp % {:axis 1})]]]
    (is (grad-matches? (projected f) m23) (str label)))
  (testing "a product with a zero has the exact gradient (no division)"
    (is (= [0.0 0.0 6.0] (array/->clj ((array/grad array/prod) (array/array [2.0 3.0 0.0]))))))
  (testing "a reduction over no axes is the identity, also backwards (review 16)"
    (is (= [1.0 1.0 1.0] (array/->clj ((array/grad #(array/sum (array/sum % {:axis []}))) (array/array [1.0 2.0 3.0])))))
    (is (= [2.0 3.0 4.0] (array/->clj ((array/grad #(array/sum (array/* % (array/prod % {:axis []})))) (array/array [1.0 1.5 2.0]))))))
  (testing "ties in max split the gradient"
    (is (= [0.5 0.5 0.0] (array/->clj ((array/grad array/max) (array/array [2.0 2.0 1.0])))))))

(deftest shapes-and-indexing
  ;; index arrays are made inside the functions, on the backend they run on
  (let [i #(array/array [2 0 2] {:dtype :int64})]
    (doseq [[label f] [[:reshape #(array/reshape % [3 2])] [:transpose array/transpose] [:flatten array/flatten]
                       [:broadcast #(array/broadcast-to % [4 2 3])] [:concatenate #(array/concatenate [% (array/* 2.0 %)] 1)]
                       [:stack #(array/stack [% %])] [:slice #(array/slice % 1 1 3)] [:slice-step #(array/slice % 1 0 nil 2)]
                       [:slice-neg #(array/slice % 1 nil nil -1)] [:take #(array/take % (i) 1)] [:take-scalar #(array/take % 1 0)]
                       [:take-along #(array/take-along-axis % (array/array [[2 0] [1 1]] {:dtype :int64}) 1)]
                       [:pick #(array/pick % (array/array [1 2] {:dtype :int64}))] [:squeeze #(array/squeeze (array/expand-dims % 0))]
                       [:diag #(array/diag (array/slice % 1 0 2))]
                       [:scatter-add #(array/scatter-add % (array/array [[0 0 2] [1 1 1]] {:dtype :int64}) (array/* 3.0 %) 1)]]]
      (is (grad-matches? (projected f) m23) (str label)))))

;; ---------------------------------------------------------------------------
;; Matrices
;; ---------------------------------------------------------------------------

(def spd (array/array [[4.0 1.0 0.5] [1.0 3.0 0.2] [0.5 0.2 2.0]]))
(def lower (array/array [[2.0 0.0 0.0] [0.5 1.5 0.0] [-0.3 0.4 1.2]]))
(def rhs (array/array [[1.0 -0.5] [0.3 2.0] [-1.2 0.7]]))
(def v3 (array/array [0.4 -1.1 0.9]))

(deftest matrix-vjps
  (is (grad-matches? (projected array/matmul) m23 rhs) "matrix × matrix")
  (is (grad-matches? (projected array/matmul) m23 v3) "matrix × vector")
  (is (grad-matches? (projected array/matmul) (array/array [0.5 -0.2]) m23) "vector × matrix")
  (is (grad-matches? (projected array/matmul) v3 v3) "vector · vector")
  (is (grad-matches? (projected array/matmul) (array/stack [spd lower]) rhs) "batched")
  (is (grad-matches? (projected la/cholesky) spd) "Cholesky (lower triangle read)")
  (is (grad-matches? (projected #(la/solve-triangular %1 %2)) lower rhs))
  (is (grad-matches? (projected #(la/solve-triangular %1 %2 {:transpose? true})) lower v3))
  (is (grad-matches? (projected #(la/solve-triangular %1 %2 {:lower? false})) (array/transpose lower) rhs))
  (is (grad-matches? (projected la/solve) spd rhs))
  (is (grad-matches? (projected la/inv) spd))
  (is (grad-matches? #(second (la/slogdet %)) spd))
  (is (grad-matches? la/det spd))
  (is (grad-matches? #(la/cholesky-logdet (la/cholesky %)) spd)))

;; ---------------------------------------------------------------------------
;; Distributions
;; ---------------------------------------------------------------------------

(deftest log-density-gradients
  (doseq [[label f & xs]
          [[:normal #(dc/log-prob (d/normal %1 %2) %3) 0.3 1.7 -0.4]
           [:gamma #(dc/log-prob (d/gamma %1 %2) %3) 2.5 1.5 2.0]
           [:beta #(dc/log-prob (d/beta %1 %2) %3) 2.5 0.7 0.3]
           [:inv-gamma #(dc/log-prob (d/inv-gamma %1 %2) %3) 3.0 2.0 0.8]
           [:exponential #(dc/log-prob (d/exponential %1) %2) 1.5 0.7]
           [:laplace #(dc/log-prob (d/laplace %1 %2) %3) 0.2 0.8 1.1]
           [:cauchy #(dc/log-prob (d/cauchy %1 %2) %3) 0.2 0.8 1.1]
           [:uniform #(dc/log-prob (d/uniform %1 %2) %3) -1.0 2.0 0.5]
           [:poisson-rate #(dc/log-prob (d/poisson %1) 3) 2.5]
           [:bernoulli-p #(dc/log-prob (d/bernoulli %1) true) 0.3]
           [:categorical #(dc/log-prob (d/categorical-logits %1) 1) (array/array [0.2 -0.5 1.0])]
           [:mvnormal #(dc/log-prob (d/mvnormal %1 %2) %3) (array/array [0.1 -0.3]) (array/array [[2.0 0.3] [0.3 1.0]]) (array/array [0.5 0.2])]
           [:dirichlet #(dc/log-prob (d/dirichlet %1) %2) (array/array [1.5 2.0 0.7]) (array/array [0.2 0.5 0.3])]
           [:normal-mixture #(dc/log-prob ((d/homogeneous-mixture d/normal) [0.3 0.7] %1 %2) %3)
            (array/array [-1.0 1.5]) (array/array [0.6 1.2]) 0.4]
           [:heterogeneous-mixture #(dc/log-prob ((d/heterogeneous-mixture [d/normal d/gamma]) [0.4 0.6] [%1 1.0] [%2 1.0]) %3)
            0.2 2.0 1.3]]]
    (is (apply grad-matches? f xs) (str label))))

;; ---------------------------------------------------------------------------
;; The API
;; ---------------------------------------------------------------------------

(deftest the-api
  (testing "trees keep their structure"
    (let [g ((array/grad (fn [{:keys [w b]}] (array/sum (array/square (array/+ (array/* w 2.0) b))))) {:w (array/array [1.0 2.0]) :b 0.5})]
      (is (= #{:w :b} (set (keys g))))
      (is (= [10.0 18.0] (array/->clj (:w g))))
      (is (= 14.0 (ts/host (:b g))))))
  (testing "value-and-grad, argnums"
    (is (= [7.0 [3.0 2.0]] (ts/host ((array/value-and-grad #(array/+ (array/* %1 %2) 1.0) {:argnums [0 1]}) 2.0 3.0)))))
  (testing "an argument that does not matter has a zero gradient"
    (is (= 0.0 (ts/host ((array/grad (fn [x y] (array/* y 2.0))) 3.0 4.0)))))
  (testing "stop-gradient blocks differentiation"
    (is (= 3.0 (ts/host ((array/grad (fn [x] (array/* (array/stop-gradient x) x))) 3.0)))))
  (testing "branching on a value differentiates the branch taken"
    (is (= -1.0 (ts/host ((array/grad (fn [x] (if (array/truthy? (array/< x 0)) (array/negate x) x))) -2.0)))))
  (testing "a long chain does not overflow the stack"
    (is (array/allclose? 1.0 ((array/grad (fn [x] (reduce (fn [acc _] (array/+ acc 0.0)) x (range 20000)))) 1.0))))
  (testing "errors are loud"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar" ((array/grad (fn [x] (array/* x (array/array [1.0 2.0])))) 1.0)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Cannot differentiate" ((array/grad (fn [x] (array/* x 2))) 3)))
    (if (ts/native-grad?)
      ;; MLX nests gradients: d/dx (d/dy x·y) = d/dx x = 1
      (is (= 1.0 (ts/host ((array/grad (fn [x] ((array/grad (fn [y] (array/* x y))) x))) 1.0))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"inside a differentiated"
                            ((array/grad (fn [x] ((array/grad (fn [y] (array/* x y))) x))) 1.0))))))
