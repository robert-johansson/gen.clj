(ns gen.genjl.finite-diff
  "Gen.jl 0.4.8's gradient checks (test/runtests.jl:18-104), for the ports
   of its test suite: finite_diff and its variants, logpdf_grad, and
   isapprox for a gradient.

   The differences are central, of Gen.jl's step dx, and run in float64 on
   the JVM backend whatever the backend under test, since a float32
   backend's would be noise; the gradients they check come from gen.clj on
   the backend under test."
  (:require [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.test-support :as ts]))

(def dx "Gen.jl's step (runtests.jl:104)." 1e-6)

(def probability-dx
  "The step for the elements of a probability vector (a categorical's, a
   mixture's weights): gen.clj accepts one summing to 1 within 1e-6
   (dc/check-probabilities), and Gen.jl's own step sits on that edge, where
   rounding decides."
  1e-7)

(def ^:private reference (array/backend-named :jvm))

(defn- on-reference
  "x as the reference backend holds it: a host number as it is, an array
   as a float64 array, a vector element by element."
  [x]
  (cond (array/host? x) x
        (vector? x) (mapv on-reference x)
        :else (array/with-backend reference (array/array (array/->clj x) {:dtype :float64}))))

(defn finite-diff
  "The central-difference partial derivatives of f, a scalar function of
   args, with respect to argument i: a number for a number argument
   (Gen.jl's finite_diff), an array of the argument's shape for an array
   (finite_diff_arr_fullarg, whose elements are finite_diff_vec's and
   finite_diff_arr's), a vector of partials for a vector of numbers (a
   product distribution's value). With :symmetric?, a matrix's (j, k) and
   (k, j) move together, as a covariance's must (finite_diff_mat_sym).
   :dx changes the step."
  ([f args i] (finite-diff f args i nil))
  ([f args i {:keys [symmetric?] h :dx :or {h dx}}]
   (array/with-backend reference
     (let [args (mapv on-reference args)
           x (nth args i)
           value (fn [arg] (double (array/item (apply f (assoc args i arg)))))
           central (fn [moved n] (/ (- (value (moved h)) (value (moved (- h)))) (* 2 n h)))]
       (cond
         (array/host? x) (central #(+ (double x) %) 1)
         (vector? x) (mapv (fn [j] (central #(update x j + %) 1)) (range (count x)))
         :else
         (let [shape (array/shape x)
               xs (vec (flatten (vector (array/->clj x))))
               cols (peek shape)
               ;; flat index e, and its transpose's when symmetric and off the diagonal
               moves (fn [e] (let [[j k] [(quot e cols) (mod e cols)]]
                               (if (and symmetric? (not= j k)) [e (+ (* k cols) j)] [e])))
               partial (fn [e] (let [es (moves e)]
                                 (central (fn [d] (array/reshape (array/array (reduce #(update %1 %2 + d) xs es)) shape))
                                          (count es))))]
           (array/reshape (array/array (mapv partial (range (count xs)))) shape)))))))

(defn- differentiable?
  "Does Gen.jl give x a gradient (a float, not an integer or a boolean, or
   a vector of floats)?"
  [x]
  (cond (array/host? x) (float? x)
        (vector? x) (every? differentiable? x)
        :else (contains? #{:float16 :bfloat16 :float32 :float64} (array/dtype x))))

(defn logpdf
  "Gen.jl's logpdf(dist, value, args...) as a function of value and args,
   for the distribution (apply make args): its log density at value,
   summed over the elements of an array value, as assess gives it (which
   checks the value's shape)."
  [make]
  (fn [value & args] (:weight (gfi/assess (apply make args) [] (cm/->Value value)))))

(defn logpdf-grad
  "Gen.jl's logpdf_grad: the gradients of (logpdf make) at value and args,
   with respect to the value and each argument, as a vector [value-grad
   arg-grad ...]; nil for an integer or a boolean one, as Gen.jl's
   nothing."
  [make value & args]
  (let [xs (into [value] args)
        wrt (filterv #(differentiable? (xs %)) (range (count xs)))
        grads (when (seq wrt)
                (apply (array/grad (logpdf make) {:argnums wrt}) xs))]
    (reduce (fn [out [k g]] (assoc out k g)) (vec (repeat (count xs) nil)) (map vector wrt grads))))

(def ^:private sqrt-eps
  "isapprox's default rtol for Float64."
  (Math/sqrt (Math/ulp 1.0)))

(defn grad-close?
  "Gen.jl's isapprox(got, want) for a gradient got from gen.clj on the
   backend under test and want from finite-diff: the same shape, within
   rtol (isapprox's default √eps, or the testset's own) and atol (default
   0: a difference's rounding where the log density's own is larger than
   an ulp). On a float32 backend, within the band gen.array.tape-test
   holds VJPs to against float64 differences (rtol 2e-3, atol 2e-4)."
  ([got want] (grad-close? got want nil))
  ([got want {:keys [rtol atol] :as opts}]
   (let [band (if (ts/f32?) {:rtol 2e-3 :atol 2e-4} {:rtol (or rtol sqrt-eps) :atol (or atol 0.0)})
         got (on-reference got)]
     (if (vector? want)
       (and (vector? got) (= (count got) (count want)) (every? true? (map #(grad-close? %1 %2 opts) got want)))
       (and (= (array/shape got) (array/shape want))
            (array/with-backend reference (array/allclose? got want band)))))))

(defn logpdf-grad-matches?
  "Gen.jl's isapprox(logpdf_grad(dist, args...)[i], finite_diff(f, args, i,
   dx)), f its logpdf: the gradient with respect to argument i (0, the
   value), whole, so an array argument's elements are Gen.jl's
   finite_diff_vec assertions. opts are finite-diff's and grad-close?'s."
  ([make args i] (logpdf-grad-matches? make args i nil))
  ([make args i opts]
   (grad-close? (nth (apply logpdf-grad make args) i) (finite-diff (logpdf make) args i opts) opts)))
