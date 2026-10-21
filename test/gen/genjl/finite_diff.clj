(ns gen.genjl.finite-diff
  "Gen.jl 0.4.8's gradient checks (test/runtests.jl:18-104), for the ports
   of its test suite: finite_diff and its variants, logpdf_grad, and
   isapprox for a gradient.

   The differences are central, of Gen.jl's step dx, and run in float64 on
   the JVM backend whatever the backend under test, since a float32
   backend's would be noise; the gradients they check come from gen.clj on
   the backend under test."
  (:require [gen.array :as array]
            [gen.dist.core :as dc]
            [gen.test-support :as ts]))

(def dx "Gen.jl's step (runtests.jl:104)." 1e-6)

(def ^:private reference (array/backend-named :jvm))

(defn- on-reference
  "x as the reference backend holds it: a host number as it is, an array
   as a float64 array."
  [x]
  (if (array/host? x)
    x
    (array/with-backend reference (array/array (array/->clj x) {:dtype :float64}))))

(defn finite-diff
  "The central-difference partial derivatives of f, a scalar function of
   args, with respect to argument i: a number for a number argument
   (Gen.jl's finite_diff), an array of the argument's shape for an array
   (finite_diff_arr_fullarg, whose elements are finite_diff_vec's and
   finite_diff_arr's). With :symmetric?, a matrix's (j, k) and (k, j)
   move together, as a covariance's must (finite_diff_mat_sym)."
  ([f args i] (finite-diff f args i nil))
  ([f args i {:keys [symmetric?]}]
   (array/with-backend reference
     (let [args (mapv on-reference args)
           x (nth args i)
           value (fn [arg] (double (array/item (apply f (assoc args i arg)))))
           central (fn [moved n] (/ (- (value (moved dx)) (value (moved (- dx)))) (* 2 n dx)))]
       (if (array/host? x)
         (central #(+ (double x) %) 1)
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
  "Does Gen.jl give x a gradient (a float, not an integer or a boolean)?"
  [x]
  (if (array/host? x)
    (float? x)
    (contains? #{:float16 :bfloat16 :float32 :float64} (array/dtype x))))

(defn logpdf-grad
  "Gen.jl's logpdf_grad: the gradients of the log density of (apply make
   args) at value, with respect to the value and each argument, as a vector
   [value-grad arg-grad ...]; nil for an integer or a boolean one, as
   Gen.jl's nothing."
  [make value & args]
  (let [xs (into [value] args)
        wrt (filterv #(differentiable? (xs %)) (range (count xs)))
        grads (when (seq wrt)
                (apply (array/grad (fn [v & as] (dc/log-prob (apply make as) v)) {:argnums wrt}) xs))]
    (reduce (fn [out [k g]] (assoc out k g)) (vec (repeat (count xs) nil)) (map vector wrt grads))))

(def ^:private sqrt-eps
  "isapprox's default rtol for Float64."
  (Math/sqrt (Math/ulp 1.0)))

(defn grad-close?
  "Gen.jl's isapprox(got, want) for a gradient got from gen.clj on the
   backend under test and want from finite-diff: the same shape, within
   rtol (isapprox's default √eps, or the testset's own). On a float32
   backend, within the band gen.array.tape-test holds VJPs to against
   float64 differences (rtol 2e-3, atol 2e-4)."
  ([got want] (grad-close? got want nil))
  ([got want {:keys [rtol]}]
   (let [band (if (ts/f32?) {:rtol 2e-3 :atol 2e-4} {:rtol (or rtol sqrt-eps) :atol 0.0})
         got (on-reference got)]
     (and (= (array/shape got) (array/shape want))
          (array/with-backend reference (array/allclose? got want band))))))
