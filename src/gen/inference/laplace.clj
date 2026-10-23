(ns gen.inference.laplace
  "The Laplace approximation and the observed Fisher information, from
   Hessians of a trace's log density: with respect to selected continuous
   choices, or to the trainable parameters its generative function holds
   (gen.gfi/stores).

     (hessian trace selection)       → ∇² log p(τ) in the selected choices, [d d]
     (param-hessian trace)           → ∇²_θ log p(τ; θ) in the parameters, [k k]
     (choice-mode trace selection)   → the trace at the mode of the selected choices
     (param-mode trace)              → the trace at the maximum over the parameters
     (laplace trace selection)       → {:trace :values :mean :covariance :hessian
                                        :log-ml-estimate :iterations}
     (proposal approximation)        → a generative function drawing the selection
                                       from N(mean, covariance)
     (observed-fisher trace)         → {:fisher :covariance :std-errors}
     (natural-gradient trace)        → F⁻¹∇_θ log p(τ; θ), in the stores' shape
     (bic trace n)                   → k log n - 2 log p(τ; θ)

   Coordinates. The selected values are flattened in choice-map order
   (gen.choicemap/to-array), the parameters in the order of the stores'
   leaves (gen.array/tree-leaves), each row-major: row i of a Hessian is
   element i of that flattening.

   Hessians. The tape is first order, so a Hessian is central differences
   of the reverse-mode gradient, 2d gradients for d coordinates: column i
   is (∇f(x + hᵢeᵢ) - ∇f(x - hᵢeᵢ)) / 2hᵢ, the divisor the difference of
   the two points as rounded, and the matrix is symmetrized, (H + Hᵀ)/2.
   The step hᵢ = ε^⅓·max(1, |xᵢ|), ε the machine epsilon of x's dtype
   (6.1e-6 in float64, 4.9e-3 in float32), balances the truncation error,
   O(hᵢ²), against the gradient's rounding, O(ε/hᵢ) (Nocedal and Wright,
   Numerical Optimization, §8.1), on the coordinate's own scale, or 1 near
   0. The option :rel-step replaces ε^⅓.

   Modes. choice-mode, param-mode and laplace climb by Newton's method on
   these Hessians: the step (λI - H)⁻¹∇, with λ = 0 where -H is positive
   definite and elsewhere the least of 10⁻⁸s, 10⁻⁷s, …, 10⁸s (s the
   largest |Hᵢⱼ|) that makes λI - H so (Levenberg's damping), halved until
   the log density rises by 10⁻⁴ of what its slope promises (Armijo). The
   climb stops where λ = 0 and the Newton decrement ½∇ᵀ(-H)⁻¹∇ (the rise a
   step promises, and half its squared length in the standard deviations
   of the Gaussian there) is at most :tol, by default the log density's
   resolution ε·max(1, |log p|). A rise within the log density's rounding
   (√ε of its size) is beyond a line search, and there the full step is
   taken. A Gaussian's mode is one step. Options: :tol, :max-iterations
   (100), :rel-step.

   The Laplace approximation of the selected choices θ at their mode θ̂ is
   N(θ̂, (-H)⁻¹), and log ∫ p(τ) dθ ≈ log p(τ̂) + (d/2) log 2π - ½ log
   det(-H): the log density of the unselected choices with the selected
   ones integrated out, the log marginal likelihood when the unselected
   choices are the observations. Both are exact for a Gaussian.

   Errors are loud, an ex-info with :reason: a discrete choice or parameter
   (:discrete), nothing to differentiate (:empty), a log density, gradient
   or Hessian that is not finite where it is needed (:non-finite), a
   stationary point or parameters where -H is not positive definite
   (:not-a-maximum), and a climb that stops rising (:no-progress) or runs
   out of iterations (:not-converged). The density must be exact
   (gen.gfi/with-exact-density)."
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(def ^:private log-2pi (Math/log (* 2.0 Math/PI)))

(def ^:private machine-epsilon
  "The spacing of each float dtype's values at 1."
  {:float64 (Math/ulp 1.0) :float32 (double (Math/ulp (float 1.0)))
   :float16 (Math/pow 2.0 -10) :bfloat16 (Math/pow 2.0 -7)})

(defn- finite? [x] (array/all-true? (array/isfinite x)))

;; ---------------------------------------------------------------------------
;; Log densities of flat coordinates
;; ---------------------------------------------------------------------------

(defn- log-density
  "assess's weight of gf on args and choices, at a fixed key: a function of
   the choices, as the density is exact."
  [gf args choices]
  (:weight (gfi/with-exact-density "The Laplace approximation" (gfi/assess gf args choices (rng/key 0)))))

(defn- choice-objective
  "The trace's log density in the selected choices, flattened: {:x their
   values, :f x → log p, :at x → the trace there, :values the selected
   choices}."
  [op trace selection]
  (let [{gf :gen-fn args :args choices :choices} trace
        values (sel/get-selected choices selection)
        others (sel/get-selected choices (sel/complement selection))]
    (when (cm/empty? values)
      (throw (ex-info (str op ": the selection selects no choices of the trace") {:reason :empty :selection selection})))
    (doseq [[p v] (cm/leaves values)]
      (when-not (every? grad/continuous? (array/tree-leaves v))
        (throw (ex-info (str op ": the choice at " p " is not continuous") {:reason :discrete :path p :value v}))))
    {:x (cm/to-array values)
     :f (fn [x] (log-density gf args (cm/merge others (cm/from-array values x))))
     :at (fn [x] (:trace (gfi/update trace (cm/from-array values x) (rng/key 0))))
     :values values}))

(defn- refill
  "The tree t with its leaves, in tree-leaves order, replaced by xs."
  [t xs]
  (let [it (volatile! (seq xs))]
    (array/tree-map (fn [_] (let [x (first @it)] (vswap! it next) x)) t)))

(defn- param-objective
  "The trace's log density in its parameters, flattened: {:x their values,
   :f x → log p, :at x → the trace there, :tree x → the stores holding x}."
  [op trace]
  (let [{gf :gen-fn args :args choices :choices} trace
        stores (gfi/stores gf)
        leaves (array/tree-leaves stores)]
    (when (empty? stores)
      (throw (ex-info (str op ": the trace's generative function holds no parameter store") {:reason :empty})))
    (when-let [v (first (remove grad/continuous? leaves))]
      (throw (ex-info (str op ": a parameter is not continuous: " (pr-str v)) {:reason :discrete :value v})))
    (let [[x unflatten] (grad/flattened leaves)
          tree (fn [x] (refill stores (unflatten x)))]
      {:x x
       :f (fn [x] (log-density (gfi/with-stores gf (tree x)) args choices))
       :at (fn [x] (:trace (gfi/generate (gfi/with-stores gf (tree x)) args choices (rng/key 0))))
       :tree tree})))

;; ---------------------------------------------------------------------------
;; Hessians
;; ---------------------------------------------------------------------------

(defn- central-hessian
  "∂²f/∂x² at x from gradient, f's gradient, by central differences,
   symmetrized (the namespace docstring gives the steps)."
  [gradient x {:keys [rel-step]}]
  (let [h (* (or rel-step (Math/cbrt (machine-epsilon (array/dtype x)))) (array/maximum (array/abs x) 1.0))
        steps (array/diag h)
        column (fn [i]
                 (let [up (+ x (array/take steps i))
                       down (- x (array/take steps i))]
                   ;; a column at a time: on a lazy backend the Hessian would
                   ;; otherwise hold all 2d gradients' graphs until it is read
                   (array/eval! (/ (- (gradient up) (gradient down)) (array/take (- up down) i)))))
        hess (array/stack (mapv column (range (array/size x))) 1)]
    (* 0.5 (+ hess (array/transpose hess)))))

(defn- curvature
  "{:value (host) :gradient :hessian} of the objective's f at x, each finite
   (loudly)."
  [op {:keys [f]} x opts]
  (let [[lp g] ((array/value-and-grad f) x)]
    (when-not (and (finite? lp) (finite? g))
      (throw (ex-info (str op ": the log density or its gradient is not finite at " (array/->clj x)
                           " (log density " (array/->clj lp) ")")
                      {:reason :non-finite :x x :value lp :gradient g})))
    (let [h (central-hessian (array/grad f) x opts)]
      (when-not (finite? h)
        (throw (ex-info (str op ": the Hessian is not finite at " (array/->clj x)
                             ": the gradient is not finite a step away (a support's edge?)")
                        {:reason :non-finite :x x :hessian h})))
      {:value (double (array/item lp)) :gradient g :hessian h})))

(defn- check-maximum!
  "l, the Cholesky factor of -H, unless -H is not positive definite (l not
   finite), which is an error: the point is no maximum."
  [op l h x]
  (when-not (finite? l)
    (throw (ex-info (str op ": -H is not positive definite at " (array/->clj x) ", which is no maximum")
                    {:reason :not-a-maximum :x x :hessian h})))
  l)

(defn- inverse
  "(L·Lᵀ)⁻¹ from its Cholesky factor L, as L⁻ᵀL⁻¹ (symmetric to the bit)."
  [l]
  (let [l-inv (la/solve-triangular l (array/eye (first (array/shape l)) {:dtype (array/dtype l)}))]
    (array/matmul (array/transpose l-inv) l-inv)))

(defn hessian
  "∇² log p(τ) with respect to the selected choices, at the trace's values:
   a [d d] array over their d elements, flattened in choice-map order.
   Options: :rel-step."
  ([trace selection] (hessian trace selection nil))
  ([trace selection opts]
   (let [objective (choice-objective "hessian" trace selection)]
     (:hessian (curvature "hessian" objective (:x objective) opts)))))

(defn param-hessian
  "∇²_θ log p(τ; θ) with respect to the trainable parameters the trace's
   generative function holds, at their values: a [k k] array over their k
   elements, flattened in the stores' tree-leaves order. Options:
   :rel-step."
  ([trace] (param-hessian trace nil))
  ([trace opts]
   (let [objective (param-objective "param-hessian" trace)]
     (:hessian (curvature "param-hessian" objective (:x objective) opts)))))

;; ---------------------------------------------------------------------------
;; Newton's climb
;; ---------------------------------------------------------------------------

(defn- ascent
  "Newton's step from the gradient g and Hessian h: {:step (λI - H)⁻¹g,
   :slope gᵀ·step, :damped? λ > 0, :factor λI - H's Cholesky factor}, λ the
   least of 0, 10⁻⁸s, …, 10⁸s (s the largest |Hᵢⱼ|, or 1) that makes λI - H
   positive definite."
  [g h]
  (let [neg-h (array/negate h)
        s (let [m (double (array/item (array/max (array/abs h))))] (if (pos? m) m 1.0))
        eye (array/eye (array/size g) {:dtype (array/dtype h)})
        [lambda l] (or (some (fn [lambda]
                               (let [l (la/cholesky (if (zero? lambda) neg-h (+ neg-h (* lambda eye))))]
                                 (when (finite? l) [lambda l])))
                             (cons 0.0 (map #(* s (Math/pow 10.0 %)) (range -8 9))))
                       (throw (ex-info "No damping makes the Newton step an ascent" {:reason :non-finite :hessian h})))
        y (la/solve-triangular l g)]
    {:step (la/solve-triangular l y {:transpose? true})
     :slope (double (array/item (array/sum (* y y))))
     :damped? (pos? (double lambda))
     :factor l}))

(defn- value-at
  "f at x on the host, -Inf where the density is undefined: NaN, or a
   distribution's parameter failing its check (a scale stepped below 0)."
  [f x]
  (try (let [v (double (array/item (f x)))] (if (Double/isNaN v) ##-Inf v))
       (catch clojure.lang.ExceptionInfo e
         (if (let [{:keys [distribution param]} (ex-data e)] (and distribution param))
           ##-Inf
           (throw e)))))

(defn- line-search
  "x + t·step for the first t of 1, ½, ¼, …, 2⁻⁴⁰ at which f rises from lp by
   at least 10⁻⁴·t·slope (Armijo's condition), or nil."
  [f x lp {:keys [step slope]}]
  (some (fn [t]
          (let [x' (array/eval! (+ x (* t step)))]
            (when (>= (value-at f x') (+ lp (* 1e-4 t slope))) x')))
        (take 41 (iterate #(* 0.5 %) 1.0))))

(defn- climb
  "The maximum of the objective's f near its x by Newton's method (the
   namespace docstring): {:x :value :gradient :hessian :factor (-H's
   Cholesky factor) :iterations}."
  [op objective opts]
  (let [{:keys [tol max-iterations] :or {max-iterations 100}} opts
        f (:f objective)
        eps (machine-epsilon (array/dtype (:x objective)))]
    (loop [x (:x objective) i 0]
      (let [{:keys [value gradient hessian] :as here} (curvature op objective x opts)
            {:keys [slope damped? factor] :as dir} (ascent gradient hessian)
            decrement (* 0.5 slope)
            size (max 1.0 (Math/abs (double value)))
            done #(assoc here :x x :factor factor :iterations i)]
        (cond
          (<= decrement (or tol (* eps size)))
          (if damped?
            (throw (ex-info (str op ": the climb reached a stationary point at " (array/->clj x)
                                 " where -H is not positive definite, which is no maximum")
                            {:reason :not-a-maximum :x x :hessian hessian}))
            (done))

          (>= i max-iterations)
          (throw (ex-info (str op ": no mode within " max-iterations " Newton steps"
                               " (decrement " decrement " at " (array/->clj x) ")")
                          {:reason :not-converged :x x :decrement decrement}))

          :else
          (if-let [x' (or (line-search f x value dir)
                          ;; a rise within the log density's rounding, which
                          ;; no line search sees: Newton's step, unchecked
                          (when (and (not damped?) (<= decrement (* (Math/sqrt eps) size)))
                            (array/eval! (+ x (:step dir)))))]
            (recur x' (inc i))
            (throw (ex-info (str op ": no step from " (array/->clj x) " raises the log density,"
                                 " though Newton's step promises " decrement)
                            {:reason :no-progress :x x :decrement decrement}))))))))

(defn choice-mode
  "The trace with the selected choices at the mode of its log density in
   them, by Newton's method from their values (the namespace docstring;
   options :tol, :max-iterations, :rel-step). A converged mode where -H is
   not positive definite is an error."
  ([trace selection] (choice-mode trace selection nil))
  ([trace selection opts]
   (let [objective (choice-objective "choice-mode" trace selection)]
     ((:at objective) (:x (climb "choice-mode" objective opts))))))

(defn param-mode
  "The trace of its generative function holding the parameters that
   maximize its log density, log p(τ; θ), with its choices held: the
   maximum likelihood estimate when every choice is an observation. By
   Newton's method from the parameters it holds, as choice-mode. The
   stores are (gfi/stores (:gen-fn trace')) of the trace returned."
  ([trace] (param-mode trace nil))
  ([trace opts]
   (let [objective (param-objective "param-mode" trace)]
     ((:at objective) (:x (climb "param-mode" objective opts))))))

;; ---------------------------------------------------------------------------
;; The Laplace approximation
;; ---------------------------------------------------------------------------

(defn laplace
  "The Laplace approximation of the selected choices: their mode θ̂ from the
   trace's values (choice-mode, with its options), and there
   {:trace at the mode, :values the selected choices at the mode, :mean θ̂
   flattened, :covariance (-H)⁻¹, :hessian H, :log-ml-estimate log p(τ̂) +
   (d/2) log 2π - ½ log det(-H) (a host number), :iterations Newton's
   steps}. The estimate is of the log density of the other choices, the
   selected ones integrated out; both it and the Gaussian are exact when
   the log density is quadratic in the selection."
  ([trace selection] (laplace trace selection nil))
  ([trace selection opts]
   (let [objective (choice-objective "laplace" trace selection)
         {:keys [x value hessian factor iterations]} (climb "laplace" objective opts)
         d (array/size x)]
     {:trace ((:at objective) x)
      :values (cm/from-array (:values objective) x)
      :mean x
      :covariance (inverse factor)
      :hessian hessian
      :log-ml-estimate (+ value (* 0.5 d log-2pi) (* -0.5 (double (array/item (la/cholesky-logdet factor)))))
      :iterations iterations})))

(defn proposal
  "A generative function that draws the approximation's selected choices,
   each at its own address, from its Gaussian N(mean, covariance): a choice
   from its conditional given the ones before it (a normal for a 0-d
   choice, an mvnormal for a 1-d one), so its score is the Gaussian's log
   density of them all. It takes any arguments and ignores them, so it
   serves as gen.inference.importance's :proposal (with :proposal-args [])
   and as gen.inference.proposal/with-internal-proposal's q. A selected
   choice of more axes is an error: no distribution draws one with a full
   covariance."
  [{:keys [values mean covariance]}]
  (let [l (la/cholesky covariance)
        leaves (cm/leaves values)
        offsets (vec (reductions clojure.core/+ 0 (map #(array/size (second %)) leaves)))
        rows (fn [m i] (array/slice m 0 (offsets i) (offsets (inc i))))
        block (fn [i j] (array/slice (rows l i) 1 (offsets j) (offsets (inc j))))
        blocks (mapv (fn [i [path v]]
                       (when (< 1 (array/ndim v))
                         (throw (ex-info (str "proposal: the choice at " path " has shape " (array/shape v)
                                              ", and only 0-d and 1-d choices can be drawn with a full covariance")
                                         {:reason :shape :path path :shape (array/shape v)})))
                       (let [scale (block i i)]
                         {:path path
                          :mean (rows mean i)
                          ;; x_i = mean_i + Σ_{j<i} L_ij z_j + L_ii z_i, z standard normal
                          :crosses (mapv #(block i %) (range i))
                          :scale scale
                          :cov (array/matmul scale (array/transpose scale))
                          :scalar? (zero? (array/ndim v))}))
                     (range) leaves)]
    (gen [& _]
      (reduce (fn [zs {:keys [path mean crosses scale cov scalar?]}]
                (let [m (reduce + mean (mapv array/matmul crosses zs))
                      x (trace path (if scalar?
                                      (d/normal (array/reshape m []) (array/reshape scale []))
                                      (d/mvnormal m cov)))]
                  (conj zs (la/solve-triangular scale (- (array/flatten x) m)))))
              [] blocks)
      nil)))

;; ---------------------------------------------------------------------------
;; Fisher information
;; ---------------------------------------------------------------------------

(defn observed-fisher
  "The observed Fisher information of the trace's parameters, F =
   -∇²_θ log p(τ; θ) (param-hessian), with what it implies at a maximum
   (param-mode): {:fisher F, :covariance F⁻¹, :std-errors √diag F⁻¹ in the
   stores' shape}. With every choice an observation this is the observed
   information of the likelihood; with latent choices among them, the
   complete-data information at their values. F not positive definite
   (parameters at no maximum) is an error. Options: :rel-step."
  ([trace] (observed-fisher trace nil))
  ([trace opts]
   (let [{:keys [x tree] :as objective} (param-objective "observed-fisher" trace)
         h (:hessian (curvature "observed-fisher" objective x opts))
         fisher (array/negate h)
         covariance (inverse (check-maximum! "observed-fisher" (la/cholesky fisher) h x))]
     {:fisher fisher
      :covariance covariance
      :std-errors (tree (array/sqrt (la/diagonal covariance)))})))

(defn natural-gradient
  "F⁻¹∇_θ log p(τ; θ) at the trace's parameters, in the stores' shape: the
   gradient preconditioned by the observed Fisher information, damped as
   the Newton steps of the climb are where F is not positive definite
   ((F + λI)⁻¹∇). gen.learning's sgd at :lr 1 along it takes Newton's
   step. Options: :rel-step."
  ([trace] (natural-gradient trace nil))
  ([trace opts]
   (let [{:keys [x tree] :as objective} (param-objective "natural-gradient" trace)
         {:keys [gradient hessian]} (curvature "natural-gradient" objective x opts)]
     (tree (:step (ascent gradient hessian))))))

(defn bic
  "The Bayesian information criterion of the trace's parameters, k log n -
   2 log p(τ; θ), k the number of parameter elements and n the number of
   observations: at the maximum likelihood (param-mode), -2 times an
   asymptotic estimate of the log marginal likelihood. Lower is better."
  [trace n]
  (when-not (pos-int? n)
    (throw (ex-info (str "bic: the number of observations is a positive integer, not " (pr-str n))
                    {:reason :observations :n n})))
  (let [stores (gfi/stores (:gen-fn trace))]
    (when (empty? stores)
      (throw (ex-info "bic: the trace's generative function holds no parameter store" {:reason :empty})))
    (- (* (reduce clojure.core/+ (map array/size (array/tree-leaves stores))) (Math/log n))
       (* 2.0 (double (array/item (:score trace)))))))
