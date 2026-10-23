(ns gen.array.tape-test
  "Reverse-mode autodiff: every VJP against central finite differences in
   float64, as properties over random inputs (and broadcasting shapes); and
   under the batch tracer, every VJP per particle, against one scalar
   gradient per particle (vmap's gradient)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.array.dtype :as dt]
            [gen.array.linalg :as la]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.random :as rng]))

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
  "The central-difference gradient of the scalar function f at x, of step h,
   in float64 on the reference backend (x on it)."
  [f x h]
  (array/with-backend reference
    (let [xs (flat x)
          s (array/shape x)]
      (from-flat (for [i (range (count xs))]
                   (let [at (fn [d] (double (array/item (f (from-flat (update xs i + d) s)))))]
                     (/ (- (at h) (at (- h))) (* 2 h))))
                 s))))

(defn- close-grads? [g fd]
  (let [g (on-reference g)]
    (and (= (array/shape g) (array/shape fd))
         (array/allclose? g fd (if (ts/f32?) {:rtol 2e-3 :atol 2e-4} {:rtol 1e-5 :atol 1e-6})))))

(defn- grad-matches-by?
  "Does f's gradient with respect to each argument match finite differences
   of step h?"
  [h f & xs]
  (let [xs (vec xs)
        grads (apply (array/grad f {:argnums (vec (range (count xs)))}) xs)
        ref-xs (mapv on-reference xs)]
    (every? true?
            (for [i (range (count xs))]
              (close-grads? (nth grads i)
                            (fd-grad #(apply f (assoc ref-xs i %)) (nth ref-xs i) h))))))

(defn- grad-matches?
  "Does f's gradient with respect to each argument match finite differences?"
  [f & xs]
  (apply grad-matches-by? 1e-6 f xs))

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
   [:digamma array/digamma 0.3 6] [:sinh array/sinh -3 3] [:cosh array/cosh -3 3]
   [:arcsin array/asin -0.9 0.9] [:arccos array/acos -0.9 0.9] [:arctan array/atan -3 3]])

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
   [:xlogy array/xlogy [-3 3] [0.2 3]] [:log-beta array/log-beta [0.3 4] [0.3 4]]
   ;; atan2 away from its cut (the negative x axis): the right half-plane, and the upper left
   [:arctan2 array/atan2 [-3 3] [0.5 3]] [:arctan2-left array/atan2 [0.5 3] [-3 -0.5]]])

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

(deftest taped-host-scalars-stay-on-the-host
  ;; Under the tape an op on host scalars computes as it does untaped, on the
  ;; host. On MLX the tape once made them float32 arrays, so x and y below
  ;; (4.5e-8 apart) were equal to some ops and not to others, and
  ;; logaddexp's gradient went entirely to x (found by the property above,
  ;; seed 1792437764135, on Metal)
  (let [x (+ -3.0 (* 6 1.4901161193847656E-8))
        y (+ -3.0 (* 6 7.450580596923828E-9))
        f (fn [x y] (array/logaddexp x y))
        [v [gx gy]] ((array/value-and-grad f {:argnums [0 1]}) x y)]
    (is (ts/close? gx 0.5 {:rtol 1e-6 :atol 1e-6}) "half the gradient to each")
    (is (ts/close? gy 0.5 {:rtol 1e-6 :atol 1e-6}))
    (when-not (ts/native-grad?)
      (is (= (f x y) v) "the taped value is the value"))))

(deftest inverse-trigonometric-and-hyperbolic
  ;; closed forms, so each rule is checked whatever the properties draw
  (doseq [[label f x want] [[:arcsin array/asin 0.6 1.25] [:arccos array/acos 0.6 -1.25]
                            [:arctan array/atan 2.0 0.2] [:sinh array/sinh 1.0 (Math/cosh 1.0)]
                            [:cosh array/cosh 1.0 (Math/sinh 1.0)]]]
    (is (ts/close? want ((array/grad f) x)) (str label)))
  (testing "atan at signed zeros, where the MLX backend returns x itself: 1"
    (is (= [1.0 1.0 0.5] (array/->clj ((array/grad #(array/sum (array/atan %))) (array/array [0.0 -0.0 1.0]))))))
  (testing "atan2(y, x): (x, -y)/(x² + y²), on both sides of the y axis"
    (doseq [[y x want] [[1.0 2.0 [0.4 -0.2]] [1.0 -2.0 [-0.4 -0.2]] [-3.0 -4.0 [-0.16 0.12]]]]
      (is (every? true? (map ts/close? want ((array/grad array/atan2 {:argnums [0 1]}) y x))) (str [y x]))))
  (testing "broadcasting: a shared x's cotangent is summed"
    (is (grad-matches? (projected array/atan2) (array/array [0.3 -1.2 2.0]) (array/array [1.5])))))

(deftest full-of-a-tracked-value
  ;; a fill is a broadcast: the value's gradient is the cotangent's sum (G14),
  ;; under MLX's native gradients too (genclj-hf5z: MLX's Full::vjp scales it
  ;; by the value; with mlx_full for an array value all four fail natively)
  (let [g (array/array [[1.0 2.0] [3.0 4.0]])]
    (is (ts/close? 10.0 ((array/grad #(array/sum (array/* g (array/full [2 2] %)))) 2.3)))
    (is (ts/close? 10.0 ((array/grad #(array/sum (array/* g (array/full-like g %)))) 2.3)) "full-like")
    (is (ts/close? 10.0 ((array/grad #(array/sum (array/* g (array/full [2 2] % {:dtype :float32})))) 2.3))
        "into another dtype")
    (is (ts/close? 6.0 ((array/grad #(array/sum (array/full [2 3] (array/* % %)))) 0.5)) "of a value computed on the tape")))

(deftest where-and-remainder
  (is (grad-matches? (projected (fn [x y] (array/where (array/> x 0) (array/* x y) (array/exp y))))
                     (array/array [-1.0 0.5 2.0]) (array/array [0.3 -0.2 1.1])))
  (is (grad-matches? (projected (fn [x y] (array/mod x y))) (array/array [2.3 -1.7 5.2]) (array/array [1.5 0.9 2.0]))))

;; ---------------------------------------------------------------------------
;; Reductions, scans, shapes, indexing
;; ---------------------------------------------------------------------------

(def m23 (array/array [[0.3 -1.2 2.0] [1.5 0.7 -0.4]]))

(def reduction-ops
  "Reductions and scans of a [2 3] array."
  [[:sum array/sum] [:sum-axis #(array/sum % {:axis 1})] [:sum-keep #(array/sum % {:axis 0 :keep-dims? true})]
   [:prod array/prod] [:prod-axis #(array/prod % {:axis 1})] [:max array/max] [:min-axis #(array/min % {:axis 0})]
   [:mean array/mean] [:variance #(array/variance % {:axis 1})] [:std array/std] [:logsumexp array/logsumexp]
   [:logsumexp-axis #(array/logsumexp % {:axis 1})] [:softmax #(array/softmax % {:axis -1})]
   [:log-softmax array/log-softmax] [:cumsum #(array/cumsum % {:axis 1})]
   [:cumsum-rev #(array/cumsum % {:axis 0 :reverse? true})] [:cumprod #(array/cumprod % {:axis 1})]
   [:logcumsumexp #(array/logcumsumexp % {:axis 1})]])

(deftest reductions-and-scans
  (doseq [[label f] reduction-ops]
    (is (grad-matches? (projected f) m23) (str label)))
  (testing "a product with a zero has the exact gradient (no division)"
    (is (= [0.0 0.0 6.0] (array/->clj ((array/grad array/prod) (array/array [2.0 3.0 0.0]))))))
  (testing "a reduction over no axes is the identity, also backwards (review 16)"
    (is (= [1.0 1.0 1.0] (array/->clj ((array/grad #(array/sum (array/sum % {:axis []}))) (array/array [1.0 2.0 3.0])))))
    (is (= [2.0 3.0 4.0] (array/->clj ((array/grad #(array/sum (array/* % (array/prod % {:axis []})))) (array/array [1.0 1.5 2.0]))))))
  (testing "ties in max split the gradient"
    (is (= [0.5 0.5 0.0] (array/->clj ((array/grad array/max) (array/array [2.0 2.0 1.0])))))))

(def shape-ops
  "Shape and indexing ops of a [2 3] array. Index arrays are made inside the
   functions, on the backend they run on."
  [[:reshape #(array/reshape % [3 2])] [:transpose array/transpose] [:flatten array/flatten]
   [:broadcast #(array/broadcast-to % [4 2 3])] [:concatenate #(array/concatenate [% (array/* 2.0 %)] 1)]
   [:stack #(array/stack [% %])] [:slice #(array/slice % 1 1 3)] [:slice-step #(array/slice % 1 0 nil 2)]
   [:slice-neg #(array/slice % 1 nil nil -1)] [:take #(array/take % (array/array [2 0 2] {:dtype :int64}) 1)]
   [:take-scalar #(array/take % 1 0)]
   [:take-along #(array/take-along-axis % (array/array [[2 0] [1 1]] {:dtype :int64}) 1)]
   [:pick #(array/pick % (array/array [1 2] {:dtype :int64}))] [:squeeze #(array/squeeze (array/expand-dims % 0))]
   [:diag #(array/diag (array/slice % 1 0 2))]
   [:scatter-add #(array/scatter-add % (array/array [[0 0 2] [1 1 1]] {:dtype :int64}) (array/* 3.0 %) 1)]])

(deftest shapes-and-indexing
  (doseq [[label f] shape-ops]
    (is (grad-matches? (projected f) m23) (str label))))

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

(def log-densities
  "Log densities as functions of their parameters and values, at a point."
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
   [:normal-mixture #(dc/log-prob ((d/homogeneous-mixture d/normal) [0.3 0.7] %1 %2) %3)
    (array/array [-1.0 1.5]) (array/array [0.6 1.2]) 0.4]
   [:heterogeneous-mixture #(dc/log-prob ((d/heterogeneous-mixture [d/normal d/gamma]) [0.4 0.6] [%1 1.0] [%2 1.0]) %3)
    0.2 2.0 1.3]])

(deftest log-density-gradients
  (doseq [[label f & xs] log-densities]
    (is (apply grad-matches? f xs) (str label)))
  ;; the value's differences stay inside the support, |sum v - 1| <= 1e-6:
  ;; with a step of 1e-6 two of them left it, and the reference was -Inf and
  ;; +Inf, which allclose? called close to anything (review 7, R7-7)
  (is (grad-matches-by? 1e-8 #(dc/log-prob (d/dirichlet %1) %2) (array/array [1.5 2.0 0.7]) (array/array [0.2 0.5 0.3]))
      ":dirichlet"))

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

(deftest custom-vjps
  (let [square (array/custom-vjp (fn [x] (array/* x x)) (fn [[x] _ g] [(array/* 2.0 (array/* g x))]))
        tripled (array/custom-vjp (fn [x] (array/* x x)) (fn [_ _ g] [(array/* 3.0 g)]))
        scaled (array/custom-vjp (fn [x n] (array/* x n)) (fn [[_ n] _ g] [(array/* g n) nil]))
        v (array/array [1.0 -2.0])]
    (testing "the value is the function's"
      (is (= 4.0 (ts/host (square 2.0))))
      (is (= [1.0 4.0] (ts/host (square v)))))
    (testing "the gradient is the rule's, composed with the rest"
      (is (ts/close? (* 3 (Math/cos 0.5)) ((array/grad #(tripled (array/sin %))) 0.5)))
      (is (ts/close? ((array/grad #(array/sum (array/* % %))) v) ((array/grad #(array/sum (square %))) v))))
    (testing "a nil cotangent: no gradient for that argument"
      (is (= [4.0 4.0] (ts/host ((array/grad #(array/sum (scaled % 4.0))) v))))
      (is (= [0.0 0.0] (ts/host (second ((array/grad #(array/sum (scaled %1 %2)) {:argnums [0 1]}) v (array/array [3.0 3.0])))))))
    (when-not (ts/native-grad?)
      ;; MLX's own transformations run f inside the trace, where no value
      ;; can be read on the host (as in compile)
      (testing "host code the tape cannot see through"
        (let [tanh (array/custom-vjp (fn [x] (Math/tanh (array/item x))) (fn [_ out g] [(array/* g (- 1.0 (* out out)))]))]
          (is (ts/close? (- 1 (Math/pow (Math/tanh 0.3) 2)) ((array/grad #(tanh (array/* 1.0 %))) 0.3)))))
      (testing "a rule with the wrong number of cotangents is loud"
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one cotangent per argument"
                              ((array/grad #(array/sum ((array/custom-vjp (fn [x] x) (fn [_ _ g] [g g])) %))) v)))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not an array" ((array/custom-vjp identity identity) :a)))))

;; ---------------------------------------------------------------------------
;; Under the batch tracer: vmap's gradient
;; ---------------------------------------------------------------------------

(def ^:private n 3)

(def ^:private per-particle-tolerance
  ;; the same ops, batched or one particle at a time: round-off only, a few
  ;; float32 ulps carried through a backward pass; a NaN in the same place
  ;; (cumprod's VJP at a zero)
  (merge {:equal-nan? true} (if (ts/f32?) {:rtol 1e-4 :atol 1e-5} {:rtol 1e-10 :atol 1e-12})))

(defn- particle [x p] (array/take x p 0))

(defn- per-particle?
  "Under the batch tracer, with the arguments at the positions in batched
   holding n particles' values ([n …] arrays) and the others shared, are f's
   value and its gradients with respect to every float argument those of n
   scalar runs, particle by particle? A shared argument's gradient is per
   particle too."
  [f args batched]
  (let [argnums (vec (filter #(= :float (dt/kind (array/dtype (nth args %)))) (range (count args))))
        value-and-grads (fn [args] (apply (array/value-and-grad f {:argnums argnums}) args))
        [v gs] (batch/batched [n] (value-and-grads (map-indexed (fn [i x] (if (batched i) (batch/wrap x [n]) x)) args)))
        runs (mapv (fn [p] (value-and-grads (map-indexed (fn [i x] (if (batched i) (particle x p) x)) args))) (range n))
        same? (fn [got want] (and (= (array/shape want) (array/shape got)) (ts/close? got want per-particle-tolerance)))]
    (and (same? (batch/unwrap v [n]) (array/stack (mapv first runs)))
         (every? #(same? (batch/unwrap (nth gs %) [n]) (array/stack (mapv (fn [[_ g]] (nth g %)) runs))) argnums))))

(defn- particles
  "n particles' values of a host number or array x, each scaled by 1 + p/10:
   inside the domains of the parameters and values below."
  [x]
  (array/stack (mapv #(array/* x (+ 1.0 (* 0.1 %))) (range n))))

(def gen-particle-uniforms (gen/vector (gen/double* {:min 0 :max 1 :NaN? false}) (* n 6)))

(defspec batched-unary-vjps-are-per-particle 100
  (prop/for-all [[_ f lo hi] (gen/elements unary-ops)
                 shape (gen/elements [[] [3] [2 2]])
                 us gen-particle-uniforms]
    (per-particle? (projected f) [(scaled us lo hi (into [n] shape))] #{0})))

(defspec batched-binary-vjps-are-per-particle 100
  ;; ties in maximum and minimum included: both runs split them
  (prop/for-all [[_ f [xlo xhi] [ylo yhi]] (gen/elements binary-ops)
                 [sx sy] (gen/elements shape-pairs)
                 batched (gen/elements [#{0} #{1} #{0 1} #{}])
                 xs gen-particle-uniforms
                 ys gen-particle-uniforms]
    (let [shape (fn [i s] (if (batched i) (into [n] s) s))]
      (per-particle? (projected f) [(scaled xs xlo xhi (shape 0 sx)) (scaled ys ylo yhi (shape 1 sy))] batched))))

(defspec batched-reduction-and-shape-vjps-are-per-particle 100
  (prop/for-all [[_ f] (gen/elements (into reduction-ops shape-ops))
                 us gen-particle-uniforms]
    (per-particle? (projected f) [(scaled us -2 2 [n 2 3])] #{0})))

(deftest batched-indices
  ;; per-particle indices take the tracer's gathers, and their VJPs its
  ;; scatter-adds
  (let [x (scaled (map #(Math/sin (* 1.3 %)) (range 18)) -2 2 [n 2 3])
        ix (array/array [[2 0 2] [1 1 0] [0 2 1]] {:dtype :int64})
        k (array/array [2 0 1] {:dtype :int64})]
    (doseq [[label f args batched]
            [[:take #(array/take %1 %2 1) [x ix] #{0 1}]
             [:take-shared-values #(array/take %1 %2 1) [(particle x 0) ix] #{1}]
             [:pick #(array/pick (array/reshape %1 [6]) %2) [x k] #{0 1}]
             [:take-along #(array/take-along-axis %1 (array/reshape %2 [1 3]) 1) [x ix] #{0 1}]
             [:scatter-add #(array/scatter-add (array/* 0.5 %1) (array/broadcast-to (array/reshape %2 [1 3]) [2 3]) (array/* 3.0 %1) 1)
              [x ix] #{0 1}]]]
      (is (per-particle? (projected f) args batched) (str label)))))

(def ^:private random-spd
  "n random symmetric positive definite 3×3 matrices."
  (let [a (rng/normal (rng/key 11) [n 3 3])]
    (array/+ (array/matmul a (array/transpose a [0 2 1])) (array/* 3.0 (array/eye 3)))))

(def ^:private random-lower
  "n random lower triangular 3×3 matrices, well conditioned."
  (array/+ (array/* (rng/normal (rng/key 12) [n 3 3]) (array/array [[0.0 0.0 0.0] [1.0 0.0 0.0] [1.0 1.0 0.0]]))
           (array/* 2.0 (array/eye 3))))

(deftest batched-matrix-vjps
  (let [bm (rng/normal (rng/key 13) [n 2 3])
        bv (rng/normal (rng/key 14) [n 3])
        brhs (rng/normal (rng/key 15) [n 3 2])]
    (doseq [[label f args batched]
            [["matrix × matrix" array/matmul [bm rhs] #{0}]
             ["matrix × matrix, both" array/matmul [bm brhs] #{0 1}]
             ["matrix × vector" array/matmul [m23 bv] #{1}]
             ["vector × matrix" array/matmul [bv (array/transpose m23)] #{0}]
             ["vector · vector" array/matmul [bv v3] #{0}]
             ["Cholesky" la/cholesky [random-spd] #{0}]
             ["triangular solve" #(la/solve-triangular %1 %2) [random-lower rhs] #{0}]
             ["triangular solve, a per-particle right-hand side" #(la/solve-triangular %1 %2) [lower brhs] #{1}]
             ["transposed triangular solve" #(la/solve-triangular %1 %2 {:transpose? true}) [random-lower bv] #{0 1}]
             ["upper triangular solve" #(la/solve-triangular %1 %2 {:lower? false})
              [(array/transpose random-lower [0 2 1]) rhs] #{0}]
             ["solve" #(la/solve %1 %2) [random-spd rhs] #{0}]
             ["solve, both" #(la/solve %1 %2) [random-spd brhs] #{0 1}]
             ["inverse" la/inv [random-spd] #{0}]
             ["slogdet" #(second (la/slogdet %)) [random-spd] #{0}]
             ["det" la/det [random-spd] #{0}]
             ["cholesky-logdet" #(la/cholesky-logdet (la/cholesky %)) [random-spd] #{0}]]]
      (is (per-particle? (projected f) args batched) label))))

(deftest batched-log-density-gradients
  (doseq [[label f & xs] log-densities]
    (is (per-particle? f (mapv particles xs) (set (range (count xs)))) (str label)))
  (is (per-particle? #(dc/log-prob (d/dirichlet %1) %2) [(particles (array/array [1.5 2.0 0.7])) (array/array [0.2 0.5 0.3])] #{0})
      ":dirichlet"))

(deftest batched-custom-vjps-and-stop-gradient
  (let [square (array/custom-vjp (fn [x] (array/* x x)) (fn [[x] _ g] [(array/* 2.0 (array/* g x))]))
        scaled (array/custom-vjp (fn [x m] (array/* x m)) (fn [[_ m] _ g] [(array/* g m) nil]))
        bx (particles (array/array [1.0 -2.0 0.5]))]
    (testing "the rule runs per particle, on each particle's arguments, value and cotangent"
      (is (per-particle? #(array/sum (square (array/sin %))) [bx] #{0}))
      (is (per-particle? #(array/sum (square %)) [bx] #{}) "shared, in a batched run")
      (is (per-particle? #(array/sum (array/exp (scaled %1 %2))) [bx (array/array [2.0 3.0 0.5])] #{0}))
      (is (per-particle? #(array/sum (array/exp (scaled %1 %2))) [(array/array [2.0 3.0 0.5]) bx] #{1})
          "a nil cotangent for a batched argument"))
    (testing "stop-gradient"
      (is (per-particle? #(array/sum (array/* (array/stop-gradient %) %)) [bx] #{0})))
    (when-not (ts/native-grad?)
      (testing "a rule reading a value on the host cannot serve every particle: the batched-branch error"
        (let [tanh (array/custom-vjp (fn [x] (Math/tanh (array/item x))) (fn [_ out g] [(array/* g (- 1.0 (* out out)))]))]
          (is (:gen.array/batched-branch
               (ts/thrown-data (batch/batched [n] ((array/grad #(tanh (array/* 1.0 %))) (batch/wrap (particles 0.3) [n])))))))))))

(deftest a-shared-argument-s-gradient-is-per-particle
  ;; vmap's: each particle's gradient of its own value, whose sum over the
  ;; particles is the gradient of their total, what a shared parameter's
  ;; learning needs (the bean's brief proposed that sum in place of the
  ;; per-particle gradients: DESIGN §9, PROVISIONAL)
  (let [data (rng/normal (rng/key 16) [n 2])
        f (fn [theta x] (array/sum (array/* (array/square (array/- x theta)) (array/array [1.0 2.0]))))
        theta (array/array [0.5 -0.2])
        per-particle (array/stack (mapv #((array/grad f) theta (particle data %)) (range n)))]
    (is (per-particle? f [theta data] #{1}))
    (testing "differentiating a function that closes over the run's batched values"
      (let [g (batch/batched [n] (let [x (batch/wrap data [n])] ((array/grad #(f % x)) theta)))]
        (is (ts/close? per-particle (batch/unwrap g [n])))))
    (testing "the sum is the gradient of the total"
      (is (ts/close? (array/sum per-particle {:axis 0})
                     ((array/grad (fn [t] (reduce array/+ (map #(f t (particle data %)) (range n))))) theta))))
    (testing "draws inside the function are per particle"
      (let [g (batch/batched [n] ((array/grad (fn [t] (array/* t (rng/normal (rng/key 17))))) 1.0))]
        (is (= [n] (array/shape (batch/unwrap g [n]))))
        (is (= n (count (distinct (ts/host (batch/unwrap g [n]))))))))
    (testing "batched arguments outside a batched run"
      (let [[v g] ((array/value-and-grad #(f theta %)) (batch/wrap data [n]))]
        (is (ts/close? (array/stack (mapv #(f theta (particle data %)) (range n))) (batch/unwrap v [n])))
        (is (ts/close? (array/negate per-particle) (batch/unwrap g [n])))))))

;; The kind of a batched run's gradients: the same whichever differentiates
;; (the tape over the tracer, or MLX's native transformation around it,
;; -Dgen.mlx.grad=native). The value is batched when a batched value
;; reached it, and a gradient when its argument or the value is. Planted
;; (2026-10-24): the tape's gradients as they come fail the unbatched
;; cotangents' cases; the native transformation always broadcasting a
;; shared argument (as it did) failed the first case on Metal

(deftest batched-gradients-in-one-kind
  (let [c (particles 1.0)
        bx (particles (array/array [1.0 -2.0]))
        cases [;; [label f args batched-args value-batched?]
               ["shared, nothing batched reaching it" (fn [_] (fn [t] (array/* t t))) [3.0] #{} false]
               ["shared, a batched value added" (fn [c] (fn [t] (array/+ (array/* t t) c))) [3.0] #{} true]
               ["shared, times a batched value" (fn [c] (fn [t] (array/* t c))) [3.0] #{} true]
               ["batched, a value that ignores it" (fn [_] (fn [_] 2.0)) [bx] #{0} false]
               ["batched, a value through stop-gradient" (fn [_] (fn [x] (array/sum (array/* 0.0 (array/stop-gradient x))))) [bx] #{0} true]
               ["batched, a linear value" (fn [_] (fn [x] (array/sum (array/* 2.0 x)))) [bx] #{0} true]
               ["batched and shared" (fn [_] (fn [x t] (array/sum (array/* x t)))) [bx 1.5] #{0} true]]]
    (doseq [[label make args batched value-batched?] cases]
      (testing label
        (let [argnums (vec (range (count args)))
              [v gs] (batch/batched [n] (apply (array/value-and-grad (make (batch/wrap c [n])) {:argnums argnums})
                                               (map-indexed #(if (batched %1) (batch/wrap %2 [n]) %2) args)))
              runs (mapv (fn [p] (apply (array/value-and-grad (make (particle c p)) {:argnums argnums})
                                        (map-indexed #(if (batched %1) (particle %2 p) %2) args)))
                         (range n))]
          (is (= value-batched? (batch/batched? v)) "the value's kind")
          (is (= (mapv #(or value-batched? (contains? batched %)) argnums) (mapv batch/batched? gs)) "the gradients' kinds")
          (is (ts/close? (array/stack (mapv first runs)) (batch/unwrap v [n])) "the value")
          (doseq [i argnums]
            (is (ts/close? (array/stack (mapv #(nth (second %) i) runs)) (batch/unwrap (nth gs i) [n])) (str "gradient " i))))))))

(deftest batched-gradient-errors
  (let [x (batch/wrap (particles (array/array [1.0 2.0])) [n])]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar" (batch/batched [n] ((array/grad #(array/* % 2.0)) x)))
        "a per-particle vector is no scalar")
    (is (:gen.array/batched-branch (ts/thrown-data (batch/batched [n] ((array/grad #(array/* (array/item (array/sum %)) %)) x))))
        "reading a particle's value on the host")
    (when-not (ts/native-grad?)
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"inside a differentiated"
                            (batch/batched [n] ((array/grad (fn [y] (array/sum ((array/grad (fn [z] (array/sum (array/* y z)))) y)))) x)))))))
