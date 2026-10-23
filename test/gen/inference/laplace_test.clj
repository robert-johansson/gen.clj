(ns gen.inference.laplace-test
  "Hessians, modes, the Laplace approximation and the observed Fisher
   information against closed forms: one-choice families, Gaussian models
   (where the Laplace approximation is exact: two correlated scalars,
   Bayesian linear regression with known noise, a hierarchy inside a Map),
   the Gaussian likelihood's parameters, and a logistic regression, whose
   skewed posterior the Laplace evidence misses and importance sampling with
   the Laplace proposal does not.

   Numeric bands come from the central difference's error (fd-band) and
   the climb's stopping rule (mode-band); the statistical band, from the
   importance weights' variance, computed by quadrature."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as imp]
            [gen.inference.laplace :as lap]
            [gen.inference.proposal :as prop]
            [gen.learning :as learn]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Bands
;; ---------------------------------------------------------------------------

(def ^:private eps
  "The machine epsilon of the backend's float."
  (if (ts/f32?) (double (Math/ulp (float 1.0))) (Math/ulp 1.0)))

(defn- fd-band
  "A bound on the error of a Hessian entry by the namespace's central
   difference of the gradient at a coordinate of value x, step h =
   ε^⅓·max(1, |x|): the truncation h²/6·|f⁗| (f4 a bound on the fourth
   derivatives) and the rounding ε·g/h of two gradients whose terms sum to
   g in size, each doubled for the slack of a leading-order estimate."
  ^double [x f4 g]
  (let [h (* (Math/cbrt eps) (max 1.0 (Math/abs (double x))))]
    (* 2.0 (+ (* h h (/ (Math/abs (double f4)) 6.0)) (/ (* eps (double g)) h)))))

(defn- mode-band
  "A bound on a mode's coordinate of variance v when the climb stops: the
   Newton decrement is then at most the log density's resolution ε·max(1,
   |lp|), half the squared distance to the mode in standard deviations."
  ^double [lp v]
  (Math/sqrt (* 2.0 eps (max 1.0 (ts/abs lp)) (double v))))

(defn- abs-sum ^double [x] (ts/number (array/sum (array/abs x))))

(defn- logdet-band
  "A bound on ½ |Δ log det(-H)| from errors of at most band in H's entries:
   ½ Σᵢⱼ |Σᵢⱼ|·band, Σ = (-H)⁻¹ (to first order)."
  ^double [cov band]
  (* 0.5 band (abs-sum cov)))

(defn- inverse-band
  "A bound on the error of Σ = (-H)⁻¹'s entries from errors of at most band
   in H's: |Σ ΔH Σ| ≤ band·(Σᵢⱼ |Σᵢⱼ|)² (to first order)."
  ^double [cov band]
  (* band (Math/pow (abs-sum cov) 2)))

(defn- within? [x y band] (ts/close? x y {:atol band :rtol 0.0}))

(defn- at [model args choices] (:trace (gfi/generate model args choices (rng/key 0))))

(defn- observed [k ys] (apply cm/choicemap (mapcat (fn [i y] [[k i] y]) (range) ys)))

;; ---------------------------------------------------------------------------
;; Hessians against closed forms
;; ---------------------------------------------------------------------------

(def families
  "Distributions of one choice x at a value, with closed forms of the log
   density's ∂²/∂x² (f2) and ∂⁴/∂x⁴ (f4) there and the size of its
   gradient's terms (g)."
  [(let [mu 0.5 s 1.5 x 1.2]
     {:name "normal" :dist (d/normal mu s) :x x :f2 (/ -1.0 (* s s)) :f4 0.0 :g (/ (Math/abs (- x mu)) (* s s))})
   (let [k 3.0 theta 2.0 x 2.5]
     {:name "gamma" :dist (d/gamma k theta) :x x
      :f2 (/ (- 1.0 k) (* x x)) :f4 (/ (* -6.0 (- k 1.0)) (Math/pow x 4)) :g (+ (/ (- k 1.0) x) (/ 1.0 theta))})
   (let [a 2.5 b 4.0 x 0.3]
     {:name "beta" :dist (d/beta a b) :x x
      :f2 (- (/ (- 1.0 a) (* x x)) (/ (- b 1.0) (Math/pow (- 1.0 x) 2)))
      :f4 (+ (/ (* 6.0 (- a 1.0)) (Math/pow x 4)) (/ (* 6.0 (- b 1.0)) (Math/pow (- 1.0 x) 4)))
      :g (+ (/ (- a 1.0) x) (/ (- b 1.0) (- 1.0 x)))})
   ;; -log(1 + u²), u = (x - x0)/γ: f2 = -2(1 - u²)/(γ²(1 + u²)²), f4 = 12(u⁴ - 6u² + 1)/(γ⁴(1 + u²)⁴)
   (let [x0 0.3 gamma 0.8 x 1.1 u (/ (- x x0) gamma) q (+ 1.0 (* u u))]
     {:name "cauchy" :dist (d/cauchy x0 gamma) :x x
      :f2 (/ (* -2.0 (- 1.0 (* u u))) (* gamma gamma q q))
      :f4 (/ (* 12.0 (+ (Math/pow u 4) (* -6.0 u u) 1.0)) (* (Math/pow gamma 4) (Math/pow q 4)))
      :g (/ (* 2.0 (Math/abs u)) (* gamma q))})])

(deftest hessians-of-one-choice-match-closed-forms
  (doseq [{:keys [name dist x f2 f4 g]} families]
    (let [h (lap/hessian (at (gen [] (trace :x dist)) [] (cm/choicemap :x x)) (sel/select :x))]
      (is (= [1 1] (array/shape h)) name)
      (is (< (ts/abs (- (ts/number h) f2)) (fd-band x f4 g)) name))))

(def cross-y 1.3)

(def cross
  "x ~ N(0, 1), z ~ N(0.7x, 0.5), y ~ N(x + z, 1), y observed: log p is
   quadratic in (x, z), -x²/2 - 2(z - 0.7x)² - (y - x - z)²/2 + c, with
   linear term y(x + z), so the posterior is N((-H)⁻¹[y y], (-H)⁻¹), and
   y ~ N(0, √(1.7² + 0.25 + 1))."
  (gen [] (let [x (trace :x (d/normal 0 1)) z (trace :z (d/normal (* 0.7 x) 0.5))] (trace :y (d/normal (+ x z) 1)))))

(def cross-hessian (array/array [[(- -2.0 1.96) 1.8] [1.8 -5.0]]))

(def cross-g
  "A bound on the size of the gradient's terms at the values the tests
   visit (|x|, |z| ≤ 1)."
  10.0)

(deftest a-hessian-with-cross-terms-matches-its-closed-form
  (let [tr (at cross [] (cm/choicemap :x 0.4 :z -0.3 :y cross-y))
        h (lap/hessian tr (sel/select :x :z))]
    (is (within? h cross-hessian (fd-band 0.0 0.0 cross-g)))
    (testing "rows follow the choices' order (:x before :z), whatever the selection lists"
      (is (ts/close? h (lap/hessian tr (sel/select :z :x)))))))

;; Second differences of the log density, assessed on the JVM backend in
;; float64 whatever the backend under test: an independent Hessian

(def ^:private reference (array/backend-named :jvm))

(def logistic-xs [-0.5 0.5 1.0])
(def logistic-ys [false true true])

(def logistic
  "a, b ~ N(0, 1), yᵢ ~ bernoulli(σ(a + b·xᵢ)): a posterior the likelihood
   skews."
  (gen [xs] (let [a (trace :a (d/normal 0 1)) b (trace :b (d/normal 0 1))]
              (doseq [[i x] (map-indexed vector xs)]
                (trace [:y i] (d/bernoulli (array/sigmoid (+ a (* b x)))))))))

(def logistic-f4
  "A bound on the logistic log density's fourth derivatives in (a, b): each
   observation's log σ(±z) has |∂⁴/∂z⁴| = |σ(1 - σ)(1 - 6σ(1 - σ))| ≤ 1/8,
   times max(1, |x|)⁴."
  (reduce + (map #(/ (Math/pow (max 1.0 (Math/abs (double %))) 4) 8.0) logistic-xs)))

(defn- second-differences
  "The Hessian of the logistic log density in (a, b) at v by second
   differences of step h in float64, with its error bound: truncation
   h²/3·f4 and rounding ε₆₄·S/h², S the size of the log density's terms."
  [[a b] h]
  (array/with-backend reference
    (let [obs (observed :y logistic-ys)
          f (fn [da db] (ts/number (:weight (gfi/assess logistic [logistic-xs] (cm/merge obs (cm/choicemap :a (+ a da) :b (+ b db)))))))
          e [[h 0.0] [0.0 h]]
          entry (fn [i j]
                  (let [[ai bi] (e i) [aj bj] (e j)]
                    (/ (+ (- (f (+ ai aj) (+ bi bj)) (f (- ai aj) (- bi bj)) (f (- aj ai) (- bj bi)))
                          (f (- (+ ai aj)) (- (+ bi bj))))
                       (* 4.0 h h))))]
      [(vec (for [i [0 1]] (vec (for [j [0 1]] (entry i j)))))
       (+ (* h h (/ logistic-f4 3.0)) (/ (* (Math/ulp 1.0) 20.0) (* h h)))])))

(deftest the-hessian-matches-second-differences-of-the-log-density
  (let [v [0.4 -0.3]
        tr (at logistic [logistic-xs] (cm/merge (observed :y logistic-ys) (cm/choicemap :a (v 0) :b (v 1))))
        h (lap/hessian tr (sel/select :a :b))
        [reference-h reference-band] (second-differences v 1e-4)]
    (is (within? h (array/array reference-h) (+ reference-band (fd-band 1.0 logistic-f4 10.0))))
    (testing "symmetric to the bit, though the two differences of a cross term differ"
      (is (= (ts/host h) (ts/host (array/transpose h)))))))

;; ---------------------------------------------------------------------------
;; The Laplace approximation is exact for a Gaussian
;; ---------------------------------------------------------------------------

(defn- gaussian-normal-log-pdf
  "log N(y; 0, cov) of a vector y."
  [cov y]
  (ts/number (dc/log-prob (d/mvnormal (array/zeros [(count y)]) cov) (array/array y))))

(deftest the-laplace-approximation-of-a-gaussian-is-exact
  (let [{:keys [mean covariance hessian log-ml-estimate iterations trace values]}
        (lap/laplace (at cross [] (cm/choicemap :x 0.4 :z -0.3 :y cross-y)) (sel/select :x :z))
        cov (la/inv (array/negate cross-hessian))
        hband (fd-band 0.0 0.0 cross-g)
        evidence (ts/number (dc/log-prob (d/normal 0 (Math/sqrt (+ (* 1.7 1.7) 0.25 1.0))) cross-y))]
    (is (within? hessian cross-hessian hband))
    ;; the decrement bounds the distance to the mode, the Hessian's error
    ;; aside (a factor near 1, taken as 2)
    (is (within? mean (array/matmul cov (array/array [cross-y cross-y]))
                 (* 2.0 (mode-band (:score trace) (ts/number (array/max (la/diagonal cov)))))))
    (is (within? covariance cov (inverse-band cov hband)))
    (is (< (ts/abs (- log-ml-estimate evidence)) (+ (logdet-band cov hband) (* 4.0 eps (Math/abs evidence)))))
    (testing "the trace and the values are at the mean"
      (is (ts/close? (trace :x) (array/take mean 0)))
      (is (= (cm/addresses values) [[:x] [:z]])))
    (when-not (ts/f32?)
      (testing "in one Newton step"
        (is (= 1 iterations))))))

(def blr-xs [-1.0 0.0 0.5 1.5 2.0])
(def blr-ys [-1.1 0.3 0.4 2.2 2.9])

(def blr
  "Bayesian linear regression with known noise: w ~ N(0, 4I), yᵢ ~ N(w₀ +
   w₁xᵢ, 0.5)."
  (gen [xs] (let [w (trace :w (d/mvnormal [0.0 0.0] (* 4.0 (array/eye 2))))]
              (doseq [[i x] (map-indexed vector xs)]
                (trace [:y i] (d/normal (+ (array/take w 0) (* (array/take w 1) x)) 0.5))))))

(deftest bayesian-linear-regression-has-its-closed-form-posterior-and-evidence
  (let [n (count blr-xs)
        X (array/array (mapv #(vector 1.0 %) blr-xs))
        y (array/array blr-ys)
        precision (+ (/ (array/eye 2) 4.0) (/ (array/matmul (array/transpose X) X) 0.25))
        cov (la/inv precision)
        posterior-mean (array/matmul cov (/ (array/matmul (array/transpose X) y) 0.25))
        evidence (gaussian-normal-log-pdf (+ (* 0.25 (array/eye n)) (* 4.0 (array/matmul X (array/transpose X)))) blr-ys)
        {:keys [mean covariance hessian log-ml-estimate trace]}
        (lap/laplace (at blr [blr-xs] (cm/merge (observed :y blr-ys) (cm/choicemap :w (array/array [0.5 -0.5]))))
                     (sel/select :w))
        ;; the gradient's terms: |yᵢ - w·xᵢ|·|xᵢ|/0.25 summed, at |w| ≤ 2
        hband (fd-band 0.0 0.0 (* 4.0 (reduce + (map #(* (+ 2.0 (* 2.0 (Math/abs (double %1))) (Math/abs (double %2)))
                                                         (max 1.0 (Math/abs (double %1))))
                                                      blr-xs blr-ys))))]
    (is (within? hessian (array/negate precision) hband))
    (is (within? mean posterior-mean (* 2.0 (mode-band (:score trace) (ts/number (array/max (la/diagonal cov)))))))
    (is (within? covariance cov (inverse-band cov hband)))
    (is (< (ts/abs (- log-ml-estimate evidence)) (+ (logdet-band cov hband) (* 4.0 eps (Math/abs evidence)))))
    (is (ts/close? (trace :w) mean))))

(def point (gen [mu] (let [x (trace :x (d/normal mu 1))] (trace :y (d/normal x 0.5)))))

(def hierarchy
  "mu ~ N(0, 2), and inside a Map xᵢ ~ N(mu, 1), yᵢ ~ N(xᵢ, 0.5): with the
   y observed, -H over (mu, x₀, x₁, x₂) is [[1/4 + 3, -1 …] [-1, 5, 0, 0] …]
   and y ~ N(0, 4·11ᵀ + 1.25·I)."
  (gen [n] (let [mu (trace :mu (d/normal 0 2))] (splice :obs (comb/map point) (vec (repeat n mu))))))

(def hierarchy-ys [0.8 -0.2 1.5])

(deftest nested-addresses-inside-a-map
  (let [n (count hierarchy-ys)
        obs (apply cm/choicemap (mapcat (fn [i y] [[:obs i :y] y]) (range) hierarchy-ys))
        selection (apply sel/select :mu (map #(vector :obs % :x) (range n)))
        neg-h (array/array [[3.25 -1.0 -1.0 -1.0] [-1.0 5.0 0.0 0.0] [-1.0 0.0 5.0 0.0] [-1.0 0.0 0.0 5.0]])
        cov (la/inv neg-h)
        evidence (gaussian-normal-log-pdf (+ (* 1.25 (array/eye n)) (* 4.0 (array/ones [n n]))) hierarchy-ys)
        ;; the gradient's terms at |mu|, |xᵢ| ≤ 2
        hband (fd-band 0.0 0.0 40.0)
        {:keys [hessian log-ml-estimate values]} (lap/laplace (at hierarchy [n] obs) selection)]
    (is (within? hessian (array/negate neg-h) hband))
    (is (< (ts/abs (- log-ml-estimate evidence)) (+ (logdet-band cov hband) (* 4.0 eps (Math/abs evidence)))))
    (is (= (cm/addresses values) [[:mu] [:obs 0 :x] [:obs 1 :x] [:obs 2 :x]]))))

;; ---------------------------------------------------------------------------
;; The Laplace proposal
;; ---------------------------------------------------------------------------

(def mixed
  "a scalar and a vector latent, a ~ N(0, 1), w ~ N(0, Σ), and y ~ N(a +
   Σw, 0.5): a Gaussian posterior whose covariance couples the two."
  (gen [] (let [a (trace :a (d/normal 0 1)) w (trace :w (d/mvnormal [0.0 0.0] [[1.0 0.3] [0.3 2.0]]))]
            (trace :y (d/normal (+ a (array/sum w)) 0.5)))))

(deftest the-proposal-draws-from-the-gaussian
  (let [approx (lap/laplace (at mixed [] (cm/choicemap :y 1.0)) (sel/select :a :w))
        gaussian (d/mvnormal (:mean approx) (:covariance approx))
        q (lap/proposal approx)]
    (testing "its score is the Gaussian's log density of the values it draws, flattened"
      (doseq [k (range 5)]
        (let [t (gfi/simulate q [] (rng/key k))]
          (is (= [[:a] [:w]] (cm/addresses (:choices t))))
          (is (ts/close? (:score t) (dc/log-prob gaussian (cm/to-array (:choices t))) {:rtol (ts/tol 1e-10) :atol (ts/tol 1e-10)})))))
    (testing "it ignores its arguments"
      (is (= (:choices (gfi/simulate q [] (rng/key 1))) (:choices (gfi/simulate q [1 2] (rng/key 1))))))))

(deftest the-proposal-of-an-exact-posterior-weighs-every-sample-alike
  (let [observations (cm/choicemap :y cross-y)
        approx (lap/laplace (at cross [] (cm/merge observations (cm/choicemap :x 0.4 :z -0.3))) (sel/select :x :z))
        q (lap/proposal approx)
        evidence (ts/number (dc/log-prob (d/normal 0 (Math/sqrt (+ (* 1.7 1.7) 0.25 1.0))) cross-y))
        ;; log w - log p(y) = log p(x, z | y) - log q(x, z): q's error at
        ;; draws within 4 standard deviations, from its log det, its mean
        ;; (the decrement's bound, 4·√(2ε|lp|)) and its precision
        ;; (½ (x - μ)ᵀΔH(x - μ) ≤ 16·band·Σ|Σᵢⱼ| for d = 2), and rounding
        cov (:covariance approx)
        hband (fd-band 0.0 0.0 cross-g)
        band (+ (logdet-band cov hband) (* 4.0 (Math/sqrt (* 2.0 eps (max 1.0 (ts/abs (:score (:trace approx)))))))
                (* 16.0 hband (abs-sum cov)) (* 8.0 eps (Math/abs evidence)))]
    (testing "as importance sampling's proposal"
      (let [{:keys [log-normalized-weights log-ml-estimate]}
            (imp/importance-sampling cross [] observations 10 {:proposal q :proposal-args [] :key (rng/key 3)})]
        (is (every? #(< (ts/abs (- % (- (Math/log 10)))) band) log-normalized-weights))
        (is (< (ts/abs (- log-ml-estimate evidence)) band))))
    (testing "as the internal proposal (with-internal-proposal)"
      (let [model (prop/with-internal-proposal cross q)]
        (doseq [k (range 3)]
          (is (< (ts/abs (- (:weight (gfi/generate model [] observations (rng/key k))) evidence)) band)))))))

(defn- quadrature
  "log ∫∫ exp(f(a, b)) da db on an m×m grid over [-r, r]², by the trapezoid
   rule (exponentially accurate for a smooth integrand that vanishes at the
   edges)."
  [f r m]
  (let [h (/ (* 2.0 r) (dec m))
        vs (for [i (range m) j (range m)] (f (+ (- r) (* i h)) (+ (- r) (* j h))))
        top (reduce max vs)]
    (+ top (Math/log (* h h (reduce + (map #(Math/exp (- % top)) vs)))))))

(defn- sigmoid [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))

(defn- logistic-log-joint
  "log p(a, b, y) of the logistic model, on the host."
  [a b]
  (+ (* -0.5 (+ (* a a) (* b b))) (- (Math/log (* 2.0 Math/PI)))
     (reduce + (map (fn [x y] (let [p (sigmoid (+ a (* b x)))] (Math/log (if y p (- 1.0 p))))) logistic-xs logistic-ys))))

(deftest ^:heavy importance-sampling-with-the-laplace-proposal-corrects-the-approximation
  (let [n 4000
        observations (observed :y logistic-ys)
        approx (lap/laplace (at logistic [logistic-xs] (cm/merge observations (cm/choicemap :a 0.0 :b 0.0)))
                            (sel/select :a :b))
        [ma mb] (ts/host (:mean approx))
        [[caa cab] [_ cbb]] (ts/host (:covariance approx))
        det (- (* caa cbb) (* cab cab))
        log-q (fn [a b] (let [da (- a ma) db (- b mb)]
                          (- (* -0.5 (/ (+ (* cbb da da) (* -2.0 cab da db) (* caa db db)) det))
                             (Math/log (* 2.0 Math/PI (Math/sqrt det))))))
        evidence (quadrature logistic-log-joint 10.0 601)
        ;; Var(w)/E(w)² = E_q[(p/q)²]/p(y)² - 1, also by quadrature
        relative-variance (- (Math/exp (- (quadrature #(- (* 2.0 (logistic-log-joint %1 %2)) (log-q %1 %2)) 10.0 601)
                                          (* 2.0 evidence)))
                             1.0)
        se (Math/sqrt (/ relative-variance n))
        [[p11 p12] [_ p22]] (ts/host (array/negate (:hessian approx)))]
    (testing "the weights' variance is finite: 2I - (-H) is positive definite, as the prior's precision doubled must exceed q's"
      (is (and (< p11 2.0) (pos? (- (* (- 2.0 p11) (- 2.0 p22)) (* p12 p12))))))
    (testing "the posterior is not Gaussian: the Laplace evidence is off by more than the band"
      (is (> (ts/abs (- (:log-ml-estimate approx) evidence)) (* 5 se))))
    (testing "importance sampling with the Laplace proposal estimates the evidence"
      (let [{:keys [log-ml-estimate]} (imp/importance-sampling logistic [logistic-xs] observations n
                                                               {:proposal (lap/proposal approx) :proposal-args [] :key (rng/key 5)})]
        (is (< (ts/abs (- log-ml-estimate evidence)) (* 5 se)))))))

;; ---------------------------------------------------------------------------
;; Parameters
;; ---------------------------------------------------------------------------

(def ys [0.3 1.1 0.8 1.6 -0.2])

(def gaussian-likelihood
  "yᵢ ~ N(mu, e^log-sigma), mu and log-sigma trainable parameters."
  (dyn/with-params (gen [n] (let [s (array/exp (param :log-sigma))]
                              (doseq [i (range n)] (trace [:y i] (d/normal (param :mu) s)))))
                   {:mu 0.2 :log-sigma 0.1}))

(defn- likelihood-trace [stores] (at (gfi/with-stores gaussian-likelihood stores) [(count ys)] (observed :y ys)))

(defn- likelihood-hessian
  "∇² log p(y; mu, log-sigma) in (mu, log-sigma): with s = e^log-sigma and r =
   y - mu, [[-n/s², -2Σr/s²] [-2Σr/s², -2Σr²/s²]]; with its gradient,
   [Σr/s², Σr²/s² - n], and a bound on its fourth derivatives, 8(Σr² + |Σr|
   + n)/s²."
  [mu log-sigma]
  (let [n (count ys) s2 (Math/exp (* 2.0 log-sigma))
        r (map #(- % mu) ys) sr (reduce + r) srr (reduce + (map #(* % %) r))]
    {:hessian [[(- (/ n s2)) (/ (* -2.0 sr) s2)] [(/ (* -2.0 sr) s2) (/ (* -2.0 srr) s2)]]
     :gradient [(/ sr s2) (- (/ srr s2) n)]
     :f4 (/ (* 8.0 (+ srr (Math/abs sr) n)) s2)
     :g (+ (/ (+ srr (reduce + (map #(Math/abs (double %)) r))) s2) n)}))

(deftest the-parameters-hessian-matches-its-closed-form
  (let [tr (likelihood-trace {:mu 0.2 :log-sigma 0.1})
        {:keys [hessian f4 g]} (likelihood-hessian 0.2 0.1)]
    (is (= [:mu :log-sigma] (keys (gfi/stores (:gen-fn tr)))) "rows in the stores' order")
    (is (within? (lap/param-hessian tr) (array/array hessian) (fd-band 1.0 f4 g)))))

(deftest the-maximum-likelihood-its-information-and-bic
  (let [n (count ys)
        mean (/ (reduce + ys) n)
        v (/ (reduce + (map #(* (- % mean) (- % mean)) ys)) n)
        lp (- (* -0.5 n (Math/log (* 2.0 Math/PI v))) (* 0.5 n))
        mle (lap/param-mode (likelihood-trace {:mu 0.2 :log-sigma 0.1}))
        {:keys [mu log-sigma]} (gfi/stores (:gen-fn mle))
        {:keys [f4 g]} (likelihood-hessian mean (* 0.5 (Math/log v)))
        hband (fd-band 1.0 f4 g)
        cov (array/array [[(/ v n) 0.0] [0.0 (/ 1.0 (* 2 n))]])
        dmu (* 2.0 (mode-band lp (/ v n)))
        dls (* 2.0 (mode-band lp (/ 1.0 (* 2 n))))]
    (testing "at the sample mean and the root of the mean squared deviation"
      (is (within? mu mean dmu))
      (is (within? log-sigma (* 0.5 (Math/log v)) dls)))
    (testing "the observed information there, diag(n/v, 2n), and the standard errors it implies"
      (let [{:keys [fisher covariance std-errors]} (lap/observed-fisher mle)
            ;; and the mode's error, through F's derivatives in (mu, log-sigma)
            ;; there: 2n/v and 4n in size
            band (+ hband (* (+ (/ (* 2.0 n) v) (* 4.0 n)) (+ dmu dls)))]
        (is (within? fisher (array/array [[(/ n v) 0.0] [0.0 (* 2.0 n)]]) band))
        (is (within? covariance cov (inverse-band cov band)))
        (is (= #{:mu :log-sigma} (set (keys std-errors))))
        ;; δ√c = δc/2√c
        (is (within? (:mu std-errors) (Math/sqrt (/ v n)) (/ (inverse-band cov band) (* 2.0 (Math/sqrt (/ v n))))))
        (is (within? (:log-sigma std-errors) (Math/sqrt (/ 1.0 (* 2 n)))
                     (/ (inverse-band cov band) (* 2.0 (Math/sqrt (/ 1.0 (* 2 n)))))))))
    (testing "BIC: 2 log n - 2 log p(y; θ̂), within the decrement's ε|lp| and the score's rounding, doubled"
      (is (< (ts/abs (- (lap/bic mle n) (- (* 2.0 (Math/log n)) (* 2.0 lp)))) (* 2.0 2.0 (+ eps (* 4.0 eps)) (Math/abs lp)))))))

(deftest the-natural-gradient-is-the-fisher-preconditioned-gradient
  (let [{:keys [hessian gradient f4 g]} (likelihood-hessian 0.2 0.1)
        expected (la/solve (array/negate (array/array hessian)) (array/array gradient))
        {:keys [mu log-sigma]} (lap/natural-gradient (likelihood-trace {:mu 0.2 :log-sigma 0.1}))
        ;; F⁻¹'s error from F's, through F⁻¹ΔF F⁻¹g
        band (* (fd-band 1.0 f4 g) 4.0 (ts/number (array/sum (array/abs (la/inv (array/array hessian)))))
                (ts/number (array/sum (array/abs expected))))]
    (is (within? (array/stack [mu log-sigma]) expected band))
    (testing "gen.learning's sgd at lr 1 along it takes Newton's steps: near the maximum likelihood, to it"
      (let [sgd (learn/sgd {:lr 1.0})
            stores (reduce (fn [stores _] (first (learn/step sgd nil stores (lap/natural-gradient (likelihood-trace stores)))))
                           {:mu 0.6 :log-sigma -0.3} (range 6))
            n (count ys)
            mean (/ (reduce + ys) n)
            v (/ (reduce + (map #(* (- % mean) (- % mean)) ys)) n)
            ;; where the gradient vanishes to its rounding, ε·g, through
            ;; F⁻¹ = diag(v/n, 1/2n)
            band (* 4.0 eps (:g (likelihood-hessian mean (* 0.5 (Math/log v)))) (+ (/ v n) (/ 1.0 (* 2 n))))]
        (is (within? (:mu stores) mean band))
        (is (within? (:log-sigma stores) (* 0.5 (Math/log v)) band))))))

(def scaled
  "zᵢ ~ N(m·xᵢ, 1) in a Map, m a trainable parameter of its kernel: log p
   has ∂²/∂m² = -Σxᵢ²."
  (comb/map (dyn/with-params (gen [x] (trace :z (d/normal (* (param :m) x) 1))) {:m 0.5})))

(deftest parameters-of-a-combinators-callee
  (let [xs [1.0 2.0 -0.5]
        zs [0.3 1.2 0.1]
        tr (at scaled [xs] (apply cm/choicemap (mapcat (fn [i z] [[i :z] z]) (range) zs)))
        sxx (reduce + (map * xs xs))]
    (is (within? (lap/param-hessian tr) (array/array [[(- sxx)]]) (fd-band 1.0 0.0 (* 2.0 sxx))))
    (testing "and its maximum likelihood, Σxz/Σx²"
      (is (within? (:m (gfi/stores (:gen-fn (lap/param-mode tr)))) (/ (reduce + (map * xs zs)) sxx)
                   (mode-band 5.0 (/ 1.0 sxx)))))))

;; ---------------------------------------------------------------------------
;; Loud errors
;; ---------------------------------------------------------------------------

(def bimodal
  "x ~ N(0, 1), y ~ N(x², 0.1): with y = 1, modes near ±1 and a minimum at
   x = 0, where the gradient vanishes."
  (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (* x x) 0.1)))))

(deftest errors-are-loud
  (let [reason (fn [f] (:reason (ts/thrown-data (f))))]
    (testing "a discrete choice"
      (is (= :discrete (reason #(lap/laplace (at (gen [] (trace :k (d/bernoulli 0.3)) (trace :x (d/normal 0 1))) [] cm/EMPTY)
                                             (sel/select :k :x))))))
    (testing "nothing to differentiate"
      (is (= :empty (reason #(lap/hessian (at cross [] cm/EMPTY) (sel/select :nope)))))
      (is (= :empty (reason #(lap/param-mode (at cross [] cm/EMPTY)))))
      (is (= :empty (reason #(lap/bic (at cross [] cm/EMPTY) 3)))))
    (testing "a discrete parameter"
      (is (= :discrete (reason #(lap/param-hessian (likelihood-trace {:mu 1 :log-sigma 0.1}))))))
    (testing "a log density that is not finite"
      (is (= :non-finite (reason #(lap/laplace (at (gen [] (trace :x (d/gamma 2.0 1.0))) [] (cm/choicemap :x -1.0))
                                               (sel/select :x))))))
    (testing "a stationary point that is no maximum"
      (let [tr (at bimodal [] (cm/choicemap :x 0.0 :y 1.0))]
        (is (pos? (ts/number (lap/hessian tr (sel/select :x)))))
        (is (= :not-a-maximum (reason #(lap/laplace tr (sel/select :x)))))
        (testing "where a climb from beside it finds a mode"
          (is (< 0.9 (ts/number ((lap/choice-mode (at bimodal [] (cm/choicemap :x 0.3 :y 1.0)) (sel/select :x)) :x)) 1.1)))))
    (testing "parameters where -H is not positive definite (all residuals alike)"
      (is (= :not-a-maximum (reason #(lap/observed-fisher (likelihood-trace {:mu -5.0 :log-sigma 0.1}))))))
    (testing "a climb that cannot rise (a kink: the Laplace distribution's log density is linear on either side of it)"
      ;; in float32 the central difference's step, 4.9e-3, straddles the
      ;; kink, which may leave the climb circling it until its last step
      (is (contains? (if (ts/f32?) #{:no-progress :not-converged} #{:no-progress})
                     (reason #(lap/laplace (at (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/laplace x 1.0))))
                                               [] (cm/choicemap :x 1.0 :y 0.3))
                                           (sel/select :y))))))
    (testing "a climb out of steps"
      (is (= :not-converged (reason #(lap/laplace (at cross [] (cm/choicemap :x 0.4 :z -0.3 :y cross-y)) (sel/select :x :z)
                                                  {:max-iterations 0})))))
    (testing "a proposal of a choice with two axes"
      (is (= :shape (reason #(lap/proposal (lap/laplace (at (gen [] (trace :m (d/normal (array/zeros [2 2]) 1.0))) [] cm/EMPTY)
                                                        (sel/select :m)))))))
    (testing "the number of observations"
      (is (= :observations (reason #(lap/bic (likelihood-trace {:mu 0.2 :log-sigma 0.1}) 0)))))))
