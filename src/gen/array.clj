(ns gen.array
  "The array API every part of gen.clj is written against (docs/array-api.md).

   - Clojure numbers and booleans are 0-d arrays that belong to no backend; an
     op whose operands are all host scalars runs on the host and returns one.
   - Backend arrays are immutable values. An op takes its backend from its array
     operands; creation functions use *backend*.
   - Host numbers are weakly typed: they adopt the dtype of the arrays they meet.

   The arithmetic operators extend Clojure's to arrays and are meant to be
   referred (excluding clojure.core's): on numbers they agree with Clojure math,
   except that / never returns a Ratio and integer overflow wraps."
  {:api :stable}
  (:refer-clojure :exclude [+ - * / < > <= >= == max min mod abs take flatten compile])
  (:require [clojure.core :as c]
            [gen.array.dtype :as dt]
            [gen.array.host :as host]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.array.special :as special]
            [gen.array.tree :as tree]
            [gen.backend.jvm :as jvm]))

(set! *warn-on-reflection* true)

(defn backend-named
  "The backend called k: :jvm (pure Clojure, always available) or :mlx (M2,
   loaded on demand)."
  [k]
  (case k
    :jvm jvm/backend
    :mlx @(requiring-resolve 'gen.backend.mlx/backend)
    (throw (ex-info (str "Unknown backend " k) {:backend k :known #{:jvm :mlx}}))))

(def ^:dynamic *backend*
  "The backend for operations without array operands (creation functions).
   The JVM backend, unless the system property gen.backend names another
   (e.g. -Dgen.backend=mlx)."
  (backend-named (keyword (System/getProperty "gen.backend" "jvm"))))

(defmacro with-backend
  "Evaluate body with b as the backend for creation functions."
  [b & body]
  `(binding [*backend* ~b] ~@body))

;; ---------------------------------------------------------------------------
;; Values and backends
;; ---------------------------------------------------------------------------

(defn host?
  "Is x a host scalar: a Clojure number or boolean (a 0-d array)?"
  [x]
  (or (instance? Number x) (instance? Boolean x)))

(defn backend-array?
  "Is x a backend array (an array value that is not a host scalar)?"
  [x]
  (instance? gen.array.protocols.IArray x))

(defn array?
  "Is x an array value: a host scalar or a backend array?"
  [x]
  (or (host? x) (backend-array? x)))

(defn ^:no-doc not-an-array
  "Throws: op's operand x is not an array (gen.array.transform's too)."
  [op x]
  (throw (ex-info (str (name op) ": not an array: " (pr-str x)
                       " (convert data with gen.array/array)")
                  {:op op :value x})))

(defn- operand-backend
  "The backend of an operand: nil for host scalars."
  [op x]
  (cond (host? x) nil
        (backend-array? x) (p/-backend x)
        :else (not-an-array op x)))

(defn- wraps?
  "Does backend b wrap inner, itself or through the backends it wraps (so
   that mixed operands run on b)? The tape over the batch tracer wraps the
   tracer's backend too, whose arrays are a batched run's shared values."
  [b inner]
  (loop [b b]
    (and (instance? gen.array.protocols.IWrappingBackend b)
         (let [i (p/-inner b)]
           (or (identical? i inner) (recur i))))))

(defn- join-backends [op b1 b2]
  (cond (nil? b1) b2
        (nil? b2) b1
        (identical? b1 b2) b1
        (wraps? b1 b2) b1
        (wraps? b2 b1) b2
        :else (throw (ex-info (str (name op) ": arrays from different backends ("
                                   (p/-name b1) " and " (p/-name b2) ")")
                              {:op op :backends [(p/-name b1) (p/-name b2)]}))))

(defn ^:no-doc backend-of
  "The backend shared by the operands xs, or *backend* when all are host
   scalars; an error names op (gen.array.transform's too)."
  [op xs]
  (or (reduce (fn [b x] (join-backends op b (operand-backend op x))) nil xs)
      *backend*))

(defn backend
  "The backend of x (*backend* for host scalars)."
  [x]
  (or (operand-backend :backend x) *backend*))

(defn common-backend
  "The backend an op on all of xs runs on (*backend* when all are host
   scalars); throws when they come from unrelated backends."
  [xs]
  (backend-of :common-backend xs))

(defn effective-backend
  "b, or *backend* when *backend* wraps b: under a batch tracer
   (gen.array.batch), draws and other new values must be made on the
   wrapping backend even when every operand is unbatched."
  [b]
  (if (wraps? *backend* b) *backend* b))

(defn dtype
  "The dtype keyword of x."
  [x]
  (if (host? x)
    (dt/host-dtype x)
    (p/-dtype (operand-backend :dtype x) x)))

(defn dtype-kind
  "The kind of the dtype dt, a level of the promotion lattice (docs/array-api.md
   §5): :bool, :int (:int32, :int64, :uint32) or :float (:float16, :bfloat16,
   :float32, :float64). It takes a dtype, not an array: an array's or a host
   number's kind is (dtype-kind (dtype x)), so 2's is :int. Anything but a
   dtype keyword is a loud error."
  [dt]
  (if (contains? dt/dtypes dt)
    (dt/kind dt)
    (throw (ex-info (str "dtype-kind: " (pr-str dt) " is not a dtype"
                         (when (array? dt) "; an array x's kind is (dtype-kind (dtype x))"))
                    {:op :dtype-kind :value dt :dtypes dt/dtypes}))))

(defn shape
  "The shape of x as a vector of longs ([] for 0-d)."
  [x]
  (if (host? x)
    []
    (p/-shape (operand-backend :shape x) x)))

(defn ndim "The number of axes of x (0 for a scalar)." [x] (count (shape x)))

(defn concrete?
  "Can x's values be read without forcing a lazy computation? (Host scalars
   and eager backends: yes.) Validation that reads values checks this first."
  [x]
  (or (host? x) (not (p/-lazy? (backend x)))))

(defn size "The number of elements of x." [x] (sh/size (shape x)))

(defn defaults
  "Backend b's default dtypes, {:float … :int …} (docs/array-api.md §5)."
  [b]
  (p/-defaults b))

(defn backend-name
  "The keyword naming backend b, as backend-named takes it: :jvm or :mlx."
  [b]
  (p/-name b))

(defn- common-dtype
  "The dtype the operands xs combine to on backend b (weak host scalars)."
  [b xs]
  (loop [strong nil weak #{} xs (seq xs)]
    (if xs
      (let [x (first xs)]
        (if (host? x)
          (recur strong (conj weak (dt/kind (dt/host-dtype x))) (next xs))
          (let [d (p/-dtype b x)]
            (recur (if strong (dt/promote strong d) d) weak (next xs)))))
      (dt/with-weak strong weak (defaults b)))))

(defn- cast-to [b x dtype]
  (if (host? x)
    (dt/host-cast x dtype)
    (if (= dtype (p/-dtype b x)) x (p/-astype b x dtype))))

;; ---------------------------------------------------------------------------
;; Elementwise dispatch (host scalars' math is gen.array.host's)
;; ---------------------------------------------------------------------------

(defn- ew1 [op x]
  (if (host? x)
    (host/unary op x)
    (let [b (operand-backend op x)
          d (dt/input-dtype op (p/-dtype b x) (defaults b))]
      (p/-elementwise b op [(cast-to b x d)] d))))

(defn- ew2 [op x y]
  (if (and (host? x) (host? y))
    (host/binary op x y)
    (let [b (join-backends op (operand-backend op x) (operand-backend op y))
          common (cond
                   (and (backend-array? x) (backend-array? y))
                   (let [dx (p/-dtype b x) dy (p/-dtype b y)]
                     (if (identical? dx dy) dx (dt/promote dx dy)))
                   (host? y) (dt/with-weak-kind (p/-dtype b x) (dt/kind (dt/host-dtype y)) (defaults b))
                   :else (dt/with-weak-kind (p/-dtype b y) (dt/kind (dt/host-dtype x)) (defaults b)))
          d (dt/input-dtype op common (defaults b))]
      (p/-elementwise b op [(cast-to b x d) (cast-to b y d)] d))))

(defmacro ^:private defunary [fname op doc]
  `(defn ~fname ~doc [~'x] (ew1 ~op ~'x)))

(defmacro ^:private defbinary [fname op doc]
  `(defn ~fname ~doc [~'x ~'y] (ew2 ~op ~'x ~'y)))

;; --- arithmetic --------------------------------------------------------------

(defunary negate :negative "Elementwise negation.")

(defn +
  "Elementwise sum, broadcasting."
  ([] 0)
  ([x] x)
  ([x y] (ew2 :add x y))
  ([x y & more] (reduce + (+ x y) more)))

(defn -
  "Elementwise difference, broadcasting; negation with one argument."
  ([x] (negate x))
  ([x y] (ew2 :subtract x y))
  ([x y & more] (reduce - (- x y) more)))

(defn *
  "Elementwise product, broadcasting."
  ([] 1)
  ([x] x)
  ([x y] (ew2 :multiply x y))
  ([x y & more] (reduce * (* x y) more)))

(defn /
  "Elementwise true division (always floating), broadcasting; the reciprocal
   with one argument."
  ([x] (ew2 :divide 1 x))
  ([x y] (ew2 :divide x y))
  ([x y & more] (reduce / (/ x y) more)))

(defbinary pow :power "Elementwise power.")
(defbinary maximum :maximum "Elementwise maximum (NaN propagates).")
(defbinary minimum :minimum "Elementwise minimum (NaN propagates).")
(defbinary mod :remainder "Elementwise remainder with the sign of the divisor, like clojure.core/mod.")
(defbinary floor-div :floor-divide
  "Elementwise floor division, NumPy's: ints floored (0 for a zero divisor);
   floats so that x = y·q + (mod x y), q an integer, as NumPy rounds it
   ((floor-div 1.0 0.1) is 9.0), and x/y for a zero divisor.")
(defunary abs :abs "Elementwise absolute value.")
(defunary sign :sign "Elementwise sign: -1, 0 or 1 (NaN for NaN).")

;; --- exponentials, logarithms, trigonometry, rounding --------------------------

(defunary exp :exp "Elementwise e^x.")
(defunary log :log "Elementwise natural logarithm.")
(defunary expm1 :expm1 "Elementwise e^x - 1, accurate near 0.")
(defunary log1p :log1p "Elementwise log(1 + x), accurate near 0.")
(defunary sqrt :sqrt "Elementwise square root.")
(defunary sin :sin "Elementwise sine.")
(defunary cos :cos "Elementwise cosine.")
(defunary tan :tan "Elementwise tangent.")
(defunary tanh :tanh "Elementwise hyperbolic tangent.")
(defunary sinh :sinh "Elementwise hyperbolic sine.")
(defunary cosh :cosh "Elementwise hyperbolic cosine.")
(defunary asin :arcsin "Elementwise inverse sine, in [-π/2, π/2] (NaN outside [-1, 1]).")
(defunary acos :arccos "Elementwise inverse cosine, in [0, π] (NaN outside [-1, 1]).")
(defunary atan :arctan "Elementwise inverse tangent, in [-π/2, π/2].")
(defbinary atan2 :arctan2
  "Elementwise angle of the point (x, y) from the positive x axis, in
   [-π, π], broadcasting: (atan2 y x), as Math/atan2 and Julia's atan(y, x).
   Signed zeros choose the side of the cut along the negative x axis.")
(defunary erf :erf "Elementwise error function.")
(defunary erfinv :erfinv "Elementwise inverse error function (±Inf at ±1, NaN outside [-1, 1]).")
(defunary floor :floor "Elementwise floor.")
(defunary ceil :ceil "Elementwise ceiling.")
(defunary round :round "Elementwise rounding to nearest, ties to even.")
(defunary isnan :isnan "Elementwise NaN test.")
(defunary isinf :isinf "Elementwise infinity test.")

;; --- comparison and logic -----------------------------------------------------

(defbinary < :less "Elementwise x < y (bool).")
(defbinary <= :less-equal "Elementwise x <= y (bool).")
(defbinary > :greater "Elementwise x > y (bool).")
(defbinary >= :greater-equal "Elementwise x >= y (bool).")
(defbinary == :equal "Elementwise numeric equality (bool); NaN is never equal.")
(defbinary != :not-equal "Elementwise numeric inequality (bool).")
(defbinary logical-and :logical-and "Elementwise logical and (bool).")
(defbinary logical-or :logical-or "Elementwise logical or (bool).")
(defunary logical-not :logical-not "Elementwise logical not (bool).")

(defn where
  "Elementwise choice: x where condition c is true, y elsewhere, broadcasting."
  [c x y]
  (if (and (host? c) (host? x) (host? y))
    (host/where c x y)
    (let [b (backend-of :where [c x y])
          d (common-dtype b [x y])]
      (p/-elementwise b :where [(cast-to b c :bool) (cast-to b x d) (cast-to b y d)] d))))

(defn stop-gradient
  "x, with differentiation blocked."
  [x]
  (if (host? x) x (ew1 :stop-gradient x)))

(def ^:dynamic ^:private *branch-log*
  "A volatile holding the last batched-branch error raised in the call in
   effect (logging-branches), or nil."
  nil)

(defn batched-branch
  "The batched-branch error, ex-info with ::batched-branch true: one answer
   cannot serve every particle of a batch (gen.array.batch). It is also
   logged for the call in effect (logging-branches), so that code catching
   it cannot hide that the particles differ."
  ([message data] (batched-branch message data nil))
  ([message data cause]
   (let [e (ex-info message (assoc data ::batched-branch true) cause)]
     (when-let [log *branch-log*] (vreset! log e))
     e)))

(defn logging-branches
  "[(f) e]: f's result, and the last batched-branch error raised during the
   call, caught or not (nil if none)."
  [f]
  (binding [*branch-log* (volatile! nil)]
    (let [r (f)]
      [r @*branch-log*])))

(defn- sole
  "The one value of a single-element array's host data."
  [v]
  (if (sequential? v) (recur (first v)) v))

(defn truthy?
  "The Clojure truthiness of x, for branching, the same on every backend:
   a boolean (host or a one-element bool array) by its value; any number,
   and any numeric array, true, as in Clojure; a bool array of several
   elements is ambiguous (an error: use all, any or where); any other value
   by Clojure's rules. A batched bool (gen.array.batch) throws ex-info with
   :gen.array/batched-branch true: one branch cannot serve every particle.
   Code that should stay batched computes both values and picks with `where`;
   a branch in structure (different choices) runs one particle at a time."
  [x]
  (cond (instance? Boolean x) (.booleanValue ^Boolean x)
        (host? x) true
        (instance? gen.array.protocols.IBatchedValue x)
        ;; a batch: numbers are truthy in every particle; booleans may differ
        (if (= :bool (p/-dtype (p/-backend x) x))
          (throw (batched-branch "Cannot branch on a batched value: it differs across particles" {}))
          true)
        (backend-array? x)
        (let [b (p/-backend x)
              s (p/-shape b x)]
          (cond (not= :bool (p/-dtype b x)) true
                (== 1 (sh/size s)) (dt/truthy? (sole (p/-to-host b x)))
                :else (throw (ex-info (str "The truth value of a bool array of shape " s
                                           " is ambiguous: use gen.array/all, any or where")
                                      {:shape s}))))
        :else (boolean x)))

;; --- composites -----------------------------------------------------------------

(defn square "Elementwise x^2." [x] (* x x))

(defn isfinite "Elementwise test for a finite value (bool)." [x]
  (logical-not (logical-or (isnan x) (isinf x))))

(defn softplus
  "Elementwise log(1 + e^x), stable for large |x|: x + log1p(e^-x) above 0,
   log1p(e^x) elsewhere, each branch given a safe input (so that the other's
   cotangent is not NaN). No maximum: its gradient at the tie with 0 would
   be one-sided under MLX's autodiff, where softplus' is σ(0) = 1/2."
  [x]
  (let [pos (> x 0)
        xp (where pos x 0.0)
        xn (where pos 0.0 x)]
    (where pos (+ xp (log1p (exp (negate xp)))) (log1p (exp xn)))))

(defn sigmoid "Elementwise logistic function, stable for large |x|." [x]
  (exp (negate (softplus (negate x)))))

(defn logaddexp
  "Elementwise log(e^x + e^y), stable and exact at infinities: the
   backend's own op where it has one (MLX's, whose gradient is exact at
   ties under its autodiff), this composite otherwise."
  [x y]
  (let [b (when-not (and (host? x) (host? y))
            (join-backends :logaddexp (operand-backend :logaddexp x) (operand-backend :logaddexp y)))]
    (if (and b (contains? (p/-native-ops b) :logaddexp))
      (ew2 :logaddexp x y)
      (let [m (maximum x y)]
        (where (isinf m) m (+ m (log1p (exp (negate (abs (- x y)))))))))))

;; Only the 0·log 0 point is special-cased, and with a safe stand-in input
;; ("double where"): the derivative in x at x = 0 stays log y, and the discarded
;; branch cannot turn into a NaN cotangent.

(defn xlogy "Elementwise x·log(y), with 0·log(y) = 0 (also for y = 0)." [x y]
  (let [both-zero (logical-and (== x 0) (== y 0))]
    (where both-zero 0.0 (* x (log (where both-zero 1.0 y))))))

(defn xlog1py "Elementwise x·log1p(y), with 0·log1p(y) = 0 (also for y = -1)." [x y]
  (let [at-pole (logical-and (== x 0) (== y -1))]
    (where at-pole 0.0 (* x (log1p (where at-pole 0.0 y))))))

(defn clip
  "Elementwise clamp of x to [lo, hi]; either bound may be nil."
  [x lo hi]
  (cond-> x
    (some? lo) (maximum lo)
    (some? hi) (minimum hi)))

;; --- special functions (composites: differentiable and identical everywhere) ---
;; Both use the "double where" trick: the branch a `where` discards is computed
;; on a safe stand-in input, so it never produces infinities that autodiff would
;; turn into NaN gradients.

(def ^:private half-log-2pi 0.9189385332046728)

(defn- exact-double?
  "Is x a host number that its double holds exactly: a Double, or a Long of
   at most 2^53 in magnitude? (The composites' host values are then the
   host paths' in gen.array.special, bit for bit.)"
  [x]
  (or (instance? Double x) (and (instance? Long x) (<= (Math/abs (long x)) 9007199254740992))))

(defn- lgamma-lanczos
  "log Γ(z) for z >= 0.5."
  [z]
  (let [z (- z 1)
        [c0 & cs] special/lanczos-coefficients
        series (reduce + c0 (map-indexed (fn [i c] (/ c (+ z (inc i)))) cs))
        t (+ z 7.5)]
    (+ half-log-2pi (- (* (+ z 0.5) (log t)) t) (log series))))

(defn- lgamma-composite [x]
  (let [reflect? (< x 0.5)
        lg (lgamma-lanczos (where reflect? (- 1 x) x))
        xs (where reflect? x 0.25)
        pole? (logical-and (<= x 0) (== x (floor x)))]
    ;; as SciPy (genclj-9gyi): +Inf at +Inf, where Lanczos gives Inf·Inf - Inf,
    ;; and -Inf at -Inf, which the poles' test takes for one
    (where (== x ##-Inf)
           ##-Inf
           (where (logical-or pole? (== x ##Inf))
                  ##Inf
                  (where reflect?
                         (- (Math/log Math/PI) (log (abs (sin (* Math/PI xs)))) lg)
                         lg)))))

(defn lgamma
  "Elementwise log |Γ(x)| (+Inf at the poles 0, -1, -2, … and at +Inf, -Inf
   at -Inf, as SciPy): Lanczos for x >= 0.5, the reflection formula below. A
   composite, so differentiable and identical on every backend; a host number
   takes gen.array.special's lgamma, the same operations at about 20 ns where
   the composite's dispatch costs 700."
  [x]
  (if (exact-double? x) (special/lgamma (double x)) (lgamma-composite x)))

(defn- digamma-composite [x]
  (let [reflect? (< x 0)
        z (where reflect? (- 1 x) x)
        ;; 1/z first, not 1/(z + 0), which made -0.0 positive: +Inf at -0.0,
        ;; as SciPy (genclj-9gyi)
        shift (reduce + (/ z) (map #(/ (+ z %)) (range 1 10)))
        w (+ z 10)
        w2 (/ (square w))
        series (* w2 (+ (/ 12.0)
                        (* w2 (+ (/ -120.0)
                                 (* w2 (+ (/ 252.0)
                                          (* w2 (+ (/ -240.0)
                                                   (* w2 (+ (/ 132.0)
                                                            (* w2 (/ -691.0 32760.0))))))))))))
        psi (- (log w) (/ 0.5 w) series shift)
        xs (where reflect? x -0.25)
        pole? (logical-and (< x 0) (== x (floor x)))]
    (where pole?
           ##NaN
           (where reflect? (- psi (/ Math/PI (tan (* Math/PI xs)))) psi))))

(defn digamma
  "Elementwise digamma ψ(x) = d/dx log Γ(x): ten recurrence steps
   ψ(x) = ψ(x+1) - 1/x, then the asymptotic series (to x^-12), with the
   reflection formula for x < 0. As SciPy: -Inf at 0.0 and +Inf at -0.0 (the
   limits from either side), NaN at negative integers. A host number takes
   gen.array.special's digamma, as lgamma."
  [x]
  (if (exact-double? x) (special/digamma (double x)) (digamma-composite x)))

(defn log-beta
  "Elementwise log B(a, b) = log Γ(a) + log Γ(b) - log Γ(a + b)."
  [a b]
  (- (+ (lgamma a) (lgamma b)) (lgamma (+ a b))))

;; ---------------------------------------------------------------------------
;; Creation and conversion
;; ---------------------------------------------------------------------------

(defn array
  "An array from host data: a number, a boolean, nested vectors of them, or a
   double[]. Options: :dtype (default: the backend's float if any element is
   a float, :bool if all are booleans, its int otherwise)."
  ([data] (array data nil))
  ([data {:keys [dtype]}]
   (let [b *backend*]
     (if (host? data)
       (p/-from-host b [data] [] (or dtype (dt/weak-host-dtype data (defaults b))))
       (let [[shape values] (host/parse data)]
         (p/-from-host b values shape (or dtype (host/infer-dtype values (defaults b)))))))))

(defn- check-fill
  "A fill value must be a scalar: a host one, or a 0-d (per-particle) array."
  [op value]
  (when-not (or (host? value) (empty? (shape value)))
    (throw (ex-info (str (name op) ": the fill value must be a scalar, not an array of shape " (shape value)
                         " (broadcast-to fills with an array)")
                    {:op op :value-shape (shape value)}))))

(defn full
  "An array of the given shape filled with value: a host scalar or a 0-d
   array. An array runs on its own backend where that wraps *backend*: a
   value being differentiated on the tape, so the fill has its gradient.
   Options: :dtype (default: the backend's default dtype for the kind of
   value)."
  ([shape value] (full shape value nil))
  ([shape value {dt :dtype}]
   (check-fill :full value)
   ;; a negative dimension is loud on every backend, before one sees it
   (sh/size shape)
   (let [b (join-backends :full *backend* (operand-backend :full value))]
     (p/-full b (vec shape) value
              (or dt
                  (if (host? value)
                    (dt/weak-host-dtype value (defaults b))
                    ;; a (per-particle) array fill value keeps its dtype
                    (dtype value)))))))

(defn zeros
  "An array of zeros. Options: :dtype (default: the backend's float)."
  ([shape] (zeros shape nil))
  ([shape {:keys [dtype]}] (full shape 0.0 {:dtype dtype})))

(defn ones
  "An array of ones. Options: :dtype (default: the backend's float)."
  ([shape] (ones shape nil))
  ([shape {:keys [dtype]}] (full shape 1.0 {:dtype dtype})))

(defn weak-dtype
  "The dtype x has on backend b (default *backend*): an array's own; a host
   scalar's is weak, its kind's default dtype there (so float64 only where
   that is the default)."
  ([x] (weak-dtype x *backend*))
  ([x b]
   (if (host? x) (dt/weak-host-dtype x (defaults b)) (dtype x))))

(defn full-like
  "An array with x's shape, backend and (unless given) dtype, filled with
   value, a scalar (on value's backend when that wraps x's, as full). A host
   scalar x is weakly typed (weak-dtype)."
  ([x value] (full-like x value nil))
  ([x value {dt :dtype}]
   (check-fill :full-like value)
   (let [b (join-backends :full-like (backend x) (operand-backend :full-like value))]
     (p/-full b (shape x) value (or dt (weak-dtype x b))))))

(defn zeros-like "Zeros of x's shape, backend and dtype." [x] (full-like x 0))
(defn ones-like "Ones of x's shape, backend and dtype." [x] (full-like x 1))

(defn arange
  "A 1-d array [start, stop) by step, ceil((stop - start) / step) values:
   finite host numbers, step nonzero. Integral arguments give the default
   int dtype, otherwise the default float. Under an int dtype the values are
   NumPy's: the first is start cast to the dtype, the step dtype(start +
   step) - dtype(start) (so a fractional step of an int dtype may repeat
   one value). Options (4-arity): :dtype (not :bool)."
  ([stop] (arange 0 stop 1 nil))
  ([start stop] (arange start stop 1 nil))
  ([start stop step] (arange start stop step nil))
  ([start stop step {:keys [dtype]}]
   (doseq [[what v] [[:start start] [:stop stop] [:step step]]]
     (when-not (and (number? v) (Double/isFinite (double v)))
       (throw (ex-info (str "arange: the " (name what) " must be a finite number, not " (pr-str v))
                       {:reason :invalid-args what v}))))
   (when (c/== 0.0 (double step))
     (throw (ex-info "arange: the step must not be zero" {:reason :invalid-args :step step})))
   (when (= :bool dtype)
     ;; MLX has no bool arange, and NumPy's fails past two elements
     (throw (ex-info "arange: no bool dtype (compare an int arange instead)" {:reason :invalid-dtype :dtype dtype})))
   (let [b *backend*
         d (or dtype
               (if (every? #(= :int (dt/kind (dt/host-dtype %))) [start stop step])
                 (:int (defaults b))
                 (:float (defaults b))))]
     (p/-arange b start stop step d))))

(defn linspace
  "n evenly spaced floats from start to stop inclusive."
  [start stop n]
  (let [n (long n)]
    (if (== n 1)
      (array [(double start)])
      (+ start (* (arange 0 n 1 {:dtype (:float (defaults *backend*))})
                  (c// (c/- (double stop) (double start)) (dec n)))))))

(defn astype
  "x converted to dtype."
  [x dtype]
  (if (host? x)
    (dt/host-cast x dtype)
    (p/-astype (operand-backend :astype x) x dtype)))

(defn ->clj
  "The host value of x: a number or boolean for 0-d, nested vectors otherwise.
   Forces evaluation."
  [x]
  (if (host? x) x (p/-to-host (operand-backend :->clj x) x)))

(defn ^{:api :experimental} ->floats
  "The values of x, row-major, as a new float array (each rounded to
   float32, booleans as 0 and 1): many values read on the host at once,
   without ->clj's nested vectors (a language model's rows of logits).
   Forces evaluation. Reading a value being differentiated is loud, as
   ->clj's. A backend reads it by gen.array.protocols/IHostFloats, or
   through ->clj. Experimental: outside the 1.x promise (DESIGN §9)."
  ^floats [x]
  (if (host? x)
    (float-array [(dt/host-double x)])
    (let [b (operand-backend :->floats x)]
      (if (instance? gen.array.protocols.IHostFloats b)
        (p/-to-floats b x)
        (let [v (p/-to-host b x)]
          (float-array (map dt/host-double (if (sequential? v) (c/flatten v) [v]))))))))

(defn item
  "The host number or boolean of a single-element array."
  [x]
  (if (host? x)
    x
    (let [s (shape x)]
      (if (== 1 (sh/size s))
        (sole (->clj x))
        (throw (ex-info (str "item needs a single-element array, got shape " s)
                        {:shape s}))))))

;; ---------------------------------------------------------------------------
;; Trees and evaluation
;; ---------------------------------------------------------------------------

(def ^{:arglists '([t])
       :doc "The leaves of a tree of nested maps, vectors and seqs, depth first."}
  tree-leaves tree/leaves)

(def ^{:arglists '([f t])
       :doc "The tree t with f applied to every leaf (maps, including records, keep
   their type)."}
  tree-map tree/map-leaves)

(defn eval!
  "Force evaluation of every backend array in the tree t (a hint on eager
   backends: a no-op). Returns t."
  [t]
  ;; no tree walk where only eager backends exist
  (when @p/lazy-backends?
    (doseq [[b xs] (group-by p/-backend (filter backend-array? (tree-leaves t)))]
      (p/-eval! b xs)))
  t)

;; ---------------------------------------------------------------------------
;; Reductions and scans
;; ---------------------------------------------------------------------------

(defn- reduce* [op x {:keys [axis keep-dims?]}]
  (if (host? x)
    (do (sh/axes axis 0)
        (host/reduction op x))
    (let [b (operand-backend op x)
          s (p/-shape b x)
          axes (sh/axes axis (count s))
          d (dt/reduce-input-dtype op (p/-dtype b x) (defaults b))]
      ;; an extremum of nothing is no number: loud on every backend, as
      ;; NumPy and JAX raise (DESIGN §9, 1.2.1's owner calls, item 4)
      (when (and (#{:max :min :argmax :argmin} op) (some #(zero? (long (nth s %))) axes))
        (throw (ex-info (str (name op) " over an empty axis: shape " s ", axes " axes)
                        {:reason :empty-reduction :op op :shapes [s] :axes axes})))
      (p/-reduce b op (cast-to b x d) axes (boolean keep-dims?)))))

(defn sum "Sum over :axis (one or several; all by default). Options: :axis, :keep-dims?."
  ([x] (reduce* :sum x nil)) ([x opts] (reduce* :sum x opts)))
(defn prod "Product over :axis (all by default). Options: :axis, :keep-dims?."
  ([x] (reduce* :prod x nil)) ([x opts] (reduce* :prod x opts)))
(defn max "Maximum over :axis (all by default; NaN propagates). Options: :axis, :keep-dims?."
  ([x] (reduce* :max x nil)) ([x opts] (reduce* :max x opts)))
(defn min "Minimum over :axis (all by default; NaN propagates). Options: :axis, :keep-dims?."
  ([x] (reduce* :min x nil)) ([x opts] (reduce* :min x opts)))
(defn all "Logical and over :axis (all by default). Options: :axis, :keep-dims?."
  ([x] (reduce* :all x nil)) ([x opts] (reduce* :all x opts)))
(defn any "Logical or over :axis (all by default). Options: :axis, :keep-dims?."
  ([x] (reduce* :any x nil)) ([x opts] (reduce* :any x opts)))

(declare flatten reshape squeeze)

(defn- arg-reduce [op x {:keys [axis] :as opts}]
  (cond
    ;; a 0-d value (a host scalar, or a per-particle scalar of a batch) is its
    ;; own extremum
    (zero? (ndim x)) (do (sh/axes axis 0) (full-like x 0 {:dtype (:int (defaults (backend x)))}))
    (and (nil? axis) (c/> (ndim x) 1)) (reduce* op (flatten x) (assoc opts :axis 0))
    :else (reduce* op x (assoc opts :axis (if (nil? axis) 0 axis)))))

(defn argmax "Index of the maximum along :axis (of the flattened array by default); the first NaN wins."
  ([x] (arg-reduce :argmax x nil)) ([x opts] (arg-reduce :argmax x opts)))
(defn argmin "Index of the minimum along :axis (of the flattened array by default); the first NaN wins."
  ([x] (arg-reduce :argmin x nil)) ([x opts] (arg-reduce :argmin x opts)))

(defn- to-float [x]
  (if (= :float (dt/kind (dtype x))) x (astype x (:float (defaults (backend x))))))

(defn- reduced-count ^long [x axis]
  (let [s (shape x)]
    (reduce c/* 1 (map #(nth s %) (sh/axes axis (count s))))))

(defn mean "Mean over :axis (all by default). Options: :axis, :keep-dims?."
  ([x] (mean x nil))
  ([x opts] (/ (sum (to-float x) opts) (reduced-count x (:axis opts)))))

(defn variance
  "Variance over :axis (all by default). Options: :axis, :keep-dims?, :ddof
   (delta degrees of freedom, default 0)."
  ([x] (variance x nil))
  ([x {:keys [ddof] :or {ddof 0} :as opts}]
   (let [m (mean x (assoc opts :keep-dims? true))
         n (reduced-count x (:axis opts))]
     (/ (sum (square (- (to-float x) m)) opts) (c/- n (long ddof))))))

(defn std "Standard deviation over :axis. Options as for variance."
  ([x] (sqrt (variance x nil)))
  ([x opts] (sqrt (variance x opts))))

(defn logsumexp
  "log Σ exp(x) over :axis (all by default), stable, -inf for all -inf.
   Options: :axis, :keep-dims?."
  ([x] (logsumexp x nil))
  ([x {:keys [keep-dims?] :as opts}]
   (if (zero? (reduced-count x (:axis opts)))
     ;; an empty sum, log 0 = -inf, where max of nothing is loud
     (log (sum (exp x) opts))
     (let [kd (assoc opts :keep-dims? true)
           m (stop-gradient (max x kd))
           m (where (isfinite m) m 0.0)
           r (+ (log (sum (exp (- x m)) kd)) m)]
       (if keep-dims? r (squeeze r (sh/axes (:axis opts) (ndim x))))))))

(defn softmax "exp(x) normalized along :axis (default -1)."
  ([x] (softmax x nil))
  ([x {:keys [axis] :or {axis -1}}]
   ;; an empty axis has no max (and no values to shift)
   (let [m (if (zero? (reduced-count x axis)) 0.0 (stop-gradient (max x {:axis axis :keep-dims? true})))
         e (exp (- x (where (isfinite m) m 0.0)))]
     (/ e (sum e {:axis axis :keep-dims? true})))))

(defn log-softmax "log of softmax along :axis (default -1), stable."
  ([x] (log-softmax x nil))
  ([x {:keys [axis] :or {axis -1}}]
   (- x (logsumexp x {:axis axis :keep-dims? true}))))

(defn- scan* [op x {:keys [axis reverse?] :or {axis -1}}]
  (let [b (backend-of op [x])
        x (if (zero? (count (p/-shape b x))) (p/-reshape b x [1]) x)
        d (dt/reduce-input-dtype (if (#{:cumsum :cumprod} op) :sum :max) (p/-dtype b x) (defaults b))]
    (p/-scan b op (cast-to b x d) (sh/axis axis (count (p/-shape b x))) (boolean reverse?))))

(defn cumsum "Inclusive cumulative sum along :axis (default -1). Options: :axis, :reverse?."
  ([x] (scan* :cumsum x nil)) ([x opts] (scan* :cumsum x opts)))
(defn cumprod "Inclusive cumulative product along :axis (default -1). Options: :axis, :reverse?."
  ([x] (scan* :cumprod x nil)) ([x opts] (scan* :cumprod x opts)))

(defn logcumsumexp
  "Inclusive log Σ exp along :axis (default -1), stable. Options: :axis, :reverse?."
  ([x] (logcumsumexp x nil))
  ([x {:keys [axis] :or {axis -1} :as opts}]
   (let [m (stop-gradient (max x {:axis axis :keep-dims? true}))
         m (where (isfinite m) m 0.0)]
     (+ (log (cumsum (exp (- x m)) opts)) m))))

;; ---------------------------------------------------------------------------
;; Shape
;; ---------------------------------------------------------------------------

(defn- reshape-on [b x new-shape]
  (let [target (sh/reshape-target new-shape (size x))]
    (if (and (host? x) (empty? target)) x (p/-reshape b x target))))

(defn reshape
  "x with a new shape of the same size; one dimension may be -1."
  [x new-shape]
  (reshape-on (backend-of :reshape [x]) x new-shape))

(defn flatten "x as a 1-d array." [x] (reshape x [-1]))

(defn transpose
  "x with its axes permuted (reversed by default)."
  ([x] (transpose x (vec (reverse (range (ndim x))))))
  ([x perm]
   (let [r (ndim x)
         perm (mapv #(sh/axis (long %) r) perm)]
     (when-not (= (sort perm) (range r))
       (throw (ex-info (str "transpose: " perm " is not a permutation of the axes")
                       {:perm perm :rank r})))
     (if (= perm (vec (range r)))
       x
       (p/-transpose (backend-of :transpose [x]) x perm)))))

(defn squeeze
  "x without its size-1 axes (or without the given ones, which must be size 1)."
  ([x] (reshape x (vec (remove #(= 1 %) (shape x)))))
  ([x axes]
   (let [s (shape x)
         axes (set (sh/axes axes (count s)))]
     (doseq [a axes]
       (when-not (= 1 (nth s a))
         (throw (ex-info (str "squeeze: axis " a " has size " (nth s a)) {:shape s :axis a}))))
     (reshape x (vec (keep-indexed (fn [i d] (when-not (axes i) d)) s))))))

(defn expand-dims
  "x with new size-1 axes at the given position(s) of the result."
  [x axes]
  (let [s (shape x)
        rank (c/+ (count s) (if (number? axes) 1 (count axes)))
        ;; NumPy's rule: an axis named twice is an error, where a set
        ;; dropped it and inserted one axis too few (genclj-kdau)
        _ (when-not (number? axes)
            (let [normal (mapv #(sh/axis (long %) rank) axes)]
              (when-not (c/== (count (set normal)) (count normal))
                (throw (ex-info (str "expand-dims: an axis is repeated in " (pr-str (vec axes)))
                                {:reason :repeated-axis :axes (vec axes) :shape s})))))
        axes (set (sh/axes axes rank))
        r (c/+ (count s) (count axes))]
    (reshape x (loop [i 0 src s out []]
                 (if (= i r)
                   out
                   (if (axes i)
                     (recur (inc i) src (conj out 1))
                     (recur (inc i) (rest src) (conj out (first src)))))))))

(defn broadcast-to "x broadcast to shape." [x new-shape]
  (let [b (backend-of :broadcast-to [x])
        new-shape (mapv long new-shape)]
    (sh/check-broadcast-to (shape x) new-shape)
    (p/-broadcast-to b x new-shape)))

(defn broadcast-shapes
  "The shape arrays of the given shapes broadcast to, as NumPy's
   broadcast_shapes and as the elementwise ops broadcast: the shapes aligned
   at their last axes, a missing leading axis counting as 1, each axis is
   the one size other than 1 there (1 where every size is 1). No shapes give
   []. Shapes that do not broadcast (two sizes other than 1 on an axis, as
   [3] and [4]) are a loud error naming them all, as is a shape that is no
   vector of non-negative integers."
  [& shapes]
  (doseq [s shapes]
    (when-not (and (sequential? s) (every? #(and (integer? %) (c/<= 0 %)) s))
      (throw (ex-info (str "broadcast-shapes: " (pr-str s) " is not a shape (a vector of non-negative integers)")
                      {:op :broadcast-shapes :shape s}))))
  (let [shapes (mapv #(mapv long %) shapes)]
    (try (sh/broadcast-all shapes)
         (catch clojure.lang.ExceptionInfo e
           (throw (ex-info (str "broadcast-shapes: the shapes " (apply pr-str shapes) " do not broadcast")
                           {:op :broadcast-shapes :shapes shapes}
                           e))))))

(defn concatenate
  "Join arrays along an existing axis (default 0)."
  ([xs] (concatenate xs 0))
  ([xs axis]
   (let [b (backend-of :concatenate xs)
         d (common-dtype b xs)
         axis (sh/axis (long axis) (ndim (first xs)))]
     (sh/concatenate-shape (mapv shape xs) axis)
     (p/-concatenate b (mapv #(cast-to b % d) xs) axis))))

(defn stack
  "Join arrays along a new axis (default 0)."
  ([xs] (stack xs 0))
  ([xs axis]
   (concatenate (mapv #(expand-dims % axis) xs) axis)))

;; ---------------------------------------------------------------------------
;; Indexing
;; ---------------------------------------------------------------------------

(defn- slice-bounds
  "NumPy's slice normalization: [start stop] for a dimension of size d."
  [start stop ^long step ^long d]
  (let [neg (neg? step)
        norm (fn [v dflt]
               (if (nil? v)
                 dflt
                 (let [v (long v)
                       v (if (neg? v) (c/+ v d) v)]
                   (cond (neg? v) (if neg -1 0)
                         (c/>= v d) (if neg (dec d) d)
                         :else v))))]
    [(norm start (if neg (dec d) 0))
     (norm stop (if neg -1 d))]))

(defn slice
  "Elements from start (inclusive) to stop (exclusive) by step along axis. nil
   means the end; negative values count from the end; step may be negative (as
   in NumPy)."
  ([x axis start stop] (slice x axis start stop 1))
  ([x axis start stop step]
   (let [b (backend-of :slice [x])
         s (shape x)
         r (count s)
         ax (sh/axis (long axis) r)
         step (long step)
         _ (when (zero? step) (throw (ex-info "slice step cannot be zero" {:axis axis})))
         [lo hi] (slice-bounds start stop step (long (nth s ax)))]
     (p/-slice b x
               (assoc (vec (repeat r 0)) ax lo)
               (assoc s ax hi)
               (assoc (vec (repeat r 1)) ax step)))))

(defn- check-indices [op indices]
  (when-not (= :int (dt/kind (dtype indices)))
    (throw (ex-info (str (name op) ": indices must be integers, got dtype " (dtype indices))
                    {:op op :dtype (dtype indices)}))))

(defn take
  "Gather along axis (default 0): the axis is replaced by the shape of indices
   (an int array or integer), so a single integer index removes the axis.
   Negative indices count from the end. Out-of-range indices throw on an
   eager backend and for host indices; a lazy backend clamps array indices
   (docs/array-api.md §13)."
  ([x indices] (take x indices 0))
  ([x indices axis]
   (check-indices :take indices)
   (let [b (backend-of :take [x indices])]
     (p/-take b x indices (sh/axis (long axis) (ndim x))))))

(defn take-along-axis
  "Gather along axis with an int array of the same rank as x, broadcast against
   x on every other axis. Negative indices count from the end. Out-of-range
   indices throw on an eager backend; a lazy backend clamps them
   (docs/array-api.md §13)."
  [x indices axis]
  (check-indices :take-along-axis indices)
  (let [b (backend-of :take-along-axis [x indices])
        axis (sh/axis (long axis) (ndim x))]
    (sh/take-along-axis-shape (shape x) (shape indices) axis)
    (p/-take-along-axis b x indices axis)))

(defn scatter-add
  "x with updates added at indices along axis, accumulating repeated indices:
   the transpose of take-along-axis. indices (an int array) and updates have
   one shape, equal to x's on every axis but axis. Indices are bounded as
   take-along-axis's."
  [x indices updates axis]
  (check-indices :scatter-add indices)
  (let [b (backend-of :scatter-add [x updates indices])
        d (common-dtype b [x updates])
        axis (sh/axis (long axis) (ndim x))]
    (sh/check-scatter-add (shape x) (shape indices) (shape updates) axis)
    (p/-scatter-add b (cast-to b x d) indices (cast-to b updates d) axis)))

(defn ^{:api :experimental} put-along-axis
  "x with values written at indices along axis, overwriting what was
   there: the counterpart of take-along-axis that scatter-add, which
   accumulates, is not. indices (an int array) and values have one shape,
   equal to x's on every axis but axis; x and values take their common
   dtype. Indices are bounded as take-along-axis's (negative from the
   end; out of range loud on an eager backend, clamped on a lazy one).
   Along each line of axis (the other coordinates fixed) the indices must
   be distinct: a repeat is loud on the JVM backend (:repeated-indices),
   and on MLX which value lands is unspecified. Its gradient is x's
   cotangent with zeros where values were written, and the values' the
   cotangent taken there. A primitive
   (gen.array.protocols/IOverwriting); staging it is loud. Experimental:
   outside the 1.x promise (DESIGN §9)."
  [x indices values axis]
  (check-indices :put-along-axis indices)
  (let [b (backend-of :put-along-axis [x values indices])
        d (common-dtype b [x values])
        axis (sh/axis (long axis) (ndim x))]
    (sh/check-put-along-axis (shape x) (shape indices) (shape values) axis)
    (p/put-along-axis b (cast-to b x d) indices (cast-to b values d) axis)))

(defn pick
  "Index the last axis of x by the int array idx, broadcasting the remaining
   (batch) axes of x against idx. With log-probabilities x, (pick x v) is the
   categorical log-probability of v, for any batch shapes."
  [x idx]
  (check-indices :pick idx)
  (let [b (backend-of :pick [x idx])
        sx (shape x)
        si (shape idx)
        batch-rank (c/max (dec (count sx)) (count si))
        pad (fn [s n] (into (vec (repeat (c/- n (count s)) 1)) s))
        sx' (pad sx (inc batch-rank))
        si' (conj (pad si batch-rank) 1)
        _ (sh/take-along-axis-shape sx' si' batch-rank)
        x' (reshape-on b x sx')
        i' (reshape-on b idx si')
        out (p/-take-along-axis b x' i' batch-rank)]
    (reshape-on b out (pop (shape out)))))

;; ---------------------------------------------------------------------------
;; Linear algebra (decompositions live in gen.array.linalg)
;; ---------------------------------------------------------------------------

(defn matmul
  "Matrix product with NumPy semantics: 1-d operands are vectors, leading axes
   are batch axes and broadcast."
  [x y]
  (let [b (backend-of :matmul [x y])
        d (dt/input-dtype :multiply (common-dtype b [x y]) (defaults b))]
    (sh/matmul-shape (shape x) (shape y))
    (p/-matmul b (cast-to b x d) (cast-to b y d))))

(defn outer "Outer product of x and y (flattened)." [x y]
  (* (reshape x [-1 1]) (reshape y [1 -1])))

(defn eye
  "The n×n identity matrix. Options: :dtype (default float)."
  ([n] (eye n nil))
  ([n {dt :dtype}]
   (let [i (arange n)]
     (astype (== (reshape i [-1 1]) (reshape i [1 -1]))
             (or dt (:float (defaults *backend*)))))))

(defn diag
  "The diagonal matrix of a vector, or the main diagonal of a matrix."
  [x]
  (case (long (ndim x))
    ;; x where the identity is, not times it: 0·Inf would be NaN
    1 (let [i (with-backend (backend x) (arange (size x)))]
        (where (== (reshape i [-1 1]) (reshape i [1 -1])) x 0))
    2 (let [[n-rows n-cols] (shape x)
            k (c/min (long n-rows) (long n-cols))]
        (reshape (take-along-axis (slice x 0 0 k)
                                  (with-backend (backend x) (reshape (arange k) [k 1]))
                                  1)
                 [k]))
    (throw (ex-info "diag needs a 1-d or 2-d array" {:shape (shape x)}))))

;; ---------------------------------------------------------------------------
;; Differentiation (docs/array-api.md §8)
;; ---------------------------------------------------------------------------

(defn tracked?
  "Is x a value being differentiated: inside a function under value-and-grad
   (or grad), an argument it differentiates or a float computed from one,
   recorded on the tape (gen.array.tape)? Reading one on the host is an
   error, so code that may meet one tests this and reads (stop-gradient x),
   untracked, on purpose. Not tracked: values outside a differentiated
   function, and what has no gradient (bools, ints, comparisons, floor,
   stop-gradient's). Under MLX's native gradients (opt-in,
   -Dgen.mlx.grad=native) no value is: MLX differentiates its own arrays. A
   value can be both tracked and batched (batched?)."
  [x]
  (instance? gen.array.protocols.ITrackedValue x))

(defn value-and-grad
  "A function of the same arguments as f returning [value gradient]: f's
   value (a 0-d array) and its gradient with respect to the argument at
   :argnums (default 0), or a vector of gradients when :argnums is a vector.
   An argument may be a tree of maps and vectors of float arrays and numbers;
   its gradient has the same structure. stop-gradient blocks differentiation.
   First order only: f may not itself differentiate.
   In a batched run (gen.array.batch), or of batched arguments, the value
   and every gradient are per particle, as vmap's (docs/array-api.md §8b):
   a shared argument's gradient too, which sums over the particles to the
   gradient of their total."
  ([f] (value-and-grad f nil))
  ([f {:keys [argnums] :or {argnums 0}}]
   (fn [& args]
     (let [args (vec args)
           positions (if (sequential? argnums) (vec argnums) [argnums])
           _ (doseq [i positions]
               (when-not (and (integer? i) (c/< -1 i (count args)))
                 (throw (ex-info (str "value-and-grad: :argnums " (pr-str argnums) " names no argument of "
                                      (count args) " (0-based)")
                                 {:reason :invalid-option :argnums argnums :arity (count args)}))))
           ;; in a batched run, on the tracer even when every leaf is
           ;; shared: f may reach the run's batched values
           b (effective-backend (backend-of :value-and-grad (mapcat #(tree-leaves (nth args %)) positions)))
           [v grads] (p/-value-and-grad b f args positions)]
       [v (if (sequential? argnums) grads (first grads))]))))

(defn grad
  "A function of the same arguments as f returning its gradient (see
   value-and-grad)."
  ([f] (grad f nil))
  ([f opts] (comp second (value-and-grad f opts))))

(defn custom-vjp
  "f with its gradient given by vjp: for code differentiation cannot see
   through (host arithmetic, a solver's iterations) or should not (a
   cheaper or steadier rule). The result is a function of the same
   arguments, arrays or host numbers, returning (apply f args), a float
   array or number. Under value-and-grad its gradient is
   (vjp args out g): a vector of one cotangent per argument, each of its
   argument's shape (nil for none), from the arguments' values, f's value
   and the cotangent g of the result. Elsewhere it is f."
  [f vjp]
  (let [checked (fn [args out g]
                  (let [cts (vjp args out g)]
                    (when-not (and (sequential? cts) (= (count cts) (count args)))
                      (throw (ex-info (str "custom-vjp: the VJP returned " (pr-str cts) " for " (count args)
                                           " arguments; it returns one cotangent per argument (nil for none)")
                                      {:cotangents cts :arguments (count args)})))
                    (doseq [[i ct arg] (map vector (range) cts args)]
                      (when (and ct (not= (shape ct) (shape arg)))
                        (throw (ex-info (str "custom-vjp: the cotangent of argument " i " has shape " (shape ct)
                                             ", its argument " (shape arg))
                                        {:argument i :cotangent-shape (shape ct) :shape (shape arg)}))))
                    (vec cts)))]
    (fn [& args]
      (let [args (vec args)]
        (doseq [x args] (when-not (array? x) (not-an-array :custom-vjp x)))
        (let [b (backend-of :custom-vjp args)]
          (if (instance? gen.array.protocols.ICustomVjp b)
            (p/-custom-vjp b f checked args)
            (apply f args)))))))

;; ---------------------------------------------------------------------------
;; Loops, compilation, batched values and vmap: gen.array.transform holds
;; their bodies. Written in this namespace's ops, it requires gen.array, so
;; it loads at the first call (docs/array-api.md §8, §8b)
;; ---------------------------------------------------------------------------

(def ^:private transform
  ;; gen.array.transform's vars, resolved once
  (delay (into {} (map (fn [k] [k (requiring-resolve (symbol "gen.array.transform" (name k)))]))
               [:scan :compile :checkpoint :batched? :batch-shape :particle-backend :particle :particles :vmap])))

;; ---------------------------------------------------------------------------
;; Loops (experimental, outside the 1.x promise: DESIGN §9, genclj-yuau)
;; ---------------------------------------------------------------------------

(defn ^{:api :experimental} scan
  "A loop of n steps, lax.scan's: (f carry x) returns [carry' y]; scan
   returns [the last carry, the ys stacked on a new leading axis of n] (a
   tree as y is). x is xs' slice along its leading axis of n (xs a tree of
   arrays, each leaf's leading axis n), or with xs nil the step's index,
   an int32 scalar from 0 to n - 1. carry' must keep carry's structure,
   shapes and dtypes (an ex-info otherwise). With n 0, f runs once on
   zeros of x's shapes, for the shapes of the empty ys.
   On the staging backend (gen.array.stage) the loop is one node, f
   staged once; on every other backend a host loop, f called n times.
   Experimental: outside the 1.x promise (DESIGN §9)."
  [f carry xs n]
  ((:scan @transform) f carry xs n))

;; ---------------------------------------------------------------------------
;; Compilation (docs/array-api.md §8)
;; ---------------------------------------------------------------------------

(defn compile
  "f compiled where the backend can: a function of the same arguments
   (arrays, or trees of them) returning what f returns (an array or a tree
   of arrays).
   - On MLX, f is traced once per input structure, shapes and dtypes into
     one MLX graph that MLX fuses and replays (mlx_compile), so a fixed
     step (a leapfrog, a weight update) costs one call instead of an op
     per node. f must be a pure function of its arguments' values: it may
     not read one on the host (truthy?, item, ->clj), since the trace is
     not run again. Randomness comes from keys closed over or drawn outside:
     a key argument is an error, as the trace would replay its draws.
   - On the JVM backend (and wherever the backend does not compile), f
     itself.
   Options: :shapeless? (MLX: one trace for every shape); :donate
   (experimental, outside the 1.x promise), a set of argument positions:
   the compiled function consumes these arguments, after JAX's
   donate_argnums. On MLX their arrays are donated once a call has
   returned (lazily) its result: MLX may then reuse their buffers for the
   result, so a step updating a large array (a static KV cache) writes in
   place instead of copying it, and any later use of a donated array is
   loud (ex-info, :reason :donated). Where compile is f itself (the JVM
   backend, the tape, the batch tracer, MLX's CPU without a compiler)
   :donate does nothing, and a reuse of a donated argument is NOT caught
   there: never read one again. Its result values are the same either
   way."
  ([f] (compile f nil))
  ([f opts] ((:compile @transform) f opts)))

;; ---------------------------------------------------------------------------
;; Checkpoints (experimental, outside the 1.x promise: DESIGN §9)
;; ---------------------------------------------------------------------------

(defn ^{:api :experimental} checkpoint
  "f, whose gradient recomputes it: a function of the same arguments
   returning what f returns, one float array, with the same values and
   gradients, whose gradient does not keep f's intermediates from the
   forward pass. Under value-and-grad they are made again in the backward
   pass, when the cotangent of f's value reaches it, and dropped after it:
   a training step's memory holds one checkpointed layer's activations at
   a time, not every layer's, for about one more forward pass of compute.
   - The arguments are arrays or trees of them (maps, vectors). Their float
     arrays are the inputs whose gradients flow through; host numbers, int
     arrays and other leaves (a quantized weight's :bits, its uint32 words)
     reach f as they are.
   - f must take every array it differentiates through as an argument:
     closing over a value being differentiated is loud on the tape, and
     its gradient is lost under MLX's native gradients.
   - On the tape (the JVM backend, MLX's default gradients) the backward
     pass runs f again on a tape of its own, seeded with the cotangent.
     Under MLX's native gradients (-Dgen.mlx.grad=native) it is one
     mlx_custom_vjp whose VJP recomputes f (mlx_vjp) on its inputs made to
     depend on its outputs and on the cotangents (mlx_depends), so that a
     stack of checkpointed layers runs its backward layer by layer, each
     recompute after the layer above has its gradient (mlx-node's
     per-layer checkpoint; MLX's own mlx_checkpoint omits the second
     dependency).
   - Where nothing is differentiated, under the batch tracer without the
     tape (MLX's native gradients of a batched run), inside array/compile
     under native gradients, and on backends without
     gen.array.protocols/ICheckpoint (the staging backend), it calls f:
     the same values and gradients, without the memory saved.
   Experimental: outside the 1.x promise (DESIGN §9)."
  [f]
  (fn [& args] ((:checkpoint @transform) f args)))

;; ---------------------------------------------------------------------------
;; The particles of a batched value (docs/array-api.md §8b)
;; ---------------------------------------------------------------------------

(defn batched?
  "Does x hold one value per particle of a batched run (gen.array.batch, the
   batch tracer)? A value of the run that every particle shares (a constant,
   a creation function's, an argument given to all) does not. A value being
   differentiated (tracked?) does when its primal does. To the code it flows
   through, a batched value has its per-particle shape; particle reads one
   particle's value."
  [x]
  ((:batched? @transform) x))

(defn batch-shape
  "The batch axes of a batched value ([n] for a batch of n particles; a
   batch may have several axes), or nil for a value that is not batched."
  [x]
  ((:batch-shape @transform) x))

(defn particle-backend
  "The backend a batched value's particles are arrays of: the one the batch
   tracer wraps, where each particle's own run computes. Code run on
   particles' values apart from the batched run binds it (with-backend), so
   that what it creates or draws is unbatched too. nil for a value that is
   not batched."
  [x]
  ((:particle-backend @transform) x))

(defn particle
  "Particle i's value of x, a value of a batched run (batched?): an
   unbatched array of the backend the batch wraps (particle-backend), of x's
   per-particle shape, the value particle i's own run would hold. The
   particles of a batch of several axes count row-major. x itself when it is
   not batched, which every particle shares (any value, an array or not).
   It gathers, lazily on a lazy backend, and reads nothing on the host:
   reading the particle (->clj) is the caller's, at an inference boundary.
   Loud: i not a host integer in [0, n), and x being differentiated with a
   batched primal, whose particle would carry no gradient (read those of
   (stop-gradient x) to drop it on purpose)."
  [x i]
  ((:particle @transform) x i))

(defn particles
  "The values of x's particles at idx (host integers in [0, n), in their
   order, repeats allowed), stacked: an unbatched array of shape
   (count idx) followed by x's per-particle shape, of the backend the batch
   wraps, particle's value of each from one gather. A value every particle
   shares (not batched) is broadcast to as many rows. Loud as particle, and
   for a shared x that is no array."
  [x idx]
  ((:particles @transform) x idx))

;; ---------------------------------------------------------------------------
;; A function batched over a leading axis (docs/array-api.md §8b)
;; ---------------------------------------------------------------------------

(defn vmap
  "f batched over a leading axis, as JAX's vmap: a function of the same
   arguments that runs f once for the n particles of a batch, under the
   batch tracer (docs/array-api.md §8b).
   - :in-axes has one entry per argument: 0 for an argument batched along
     its leading axis (an array, or maps and vectors of arrays that all
     have it), nil for one every particle shares (any value). By default
     every argument is batched. The batched arrays agree on n, the size of
     their leading axis. Another axis is a transpose away
     (array/transpose), so an in-axis is 0 or nil.
   - Inside f, a batched argument has the shape of one row, and every op
     acts per particle. Shared values meet batched ones by their
     per-particle axes: a per-particle scalar plus a shared vector is the
     vector plus each particle's own value. Draws are per particle, n
     independent draws, even from a shared key. f runs on the tracer over
     the batched arrays' backend, which creation functions use.
   - f's value is an array, a host number, or maps and vectors of them.
     Each leaf comes back with a leading axis of n: the particles' values,
     or a shared value (one no batched value reached) broadcast.
   - Gradients inside f are per particle, as value-and-grad's in a batched
     run: ((vmap (grad g)) xs) is each row's gradient. A shared argument's
     gradient is per particle too; summed over the particles, it is the
     gradient of their total.
   Nothing falls back to one particle at a time, as gen.batched's runs of
   generative functions do. These are loud:
   - f reading a batched value on the host (item, ->clj, truthy? of a
     comparison, =), the batched-branch error (:gen.array/batched-branch);
   - a batched comparison that nothing read: a raw if or when on it took
     one branch for every particle (pick with where);
   - batched arguments without a leading axis or disagreeing on n, and
     in-axes of another length than the arguments;
   - a vmap inside a batched run, since batching does not nest (the
     batched-branch error, so a batched run around it falls back);
   - a vmap of a value being differentiated, a gradient through a vmap
     (differentiate inside it);
   - a value of f that is no tree of arrays.
   Nothing is evaluated (eval!): on a lazy backend the result is lazy."
  ([f] (vmap f nil))
  ([f opts] ((:vmap @transform) f opts)))

;; ---------------------------------------------------------------------------
;; Neural-network operations (experimental, outside the 1.x promise: DESIGN
;; §9; docs/array-api.md §6). gen.array.nn holds their bodies, loaded at
;; the first call, as gen.array.transform's
;; ---------------------------------------------------------------------------

(def ^:private nn
  (delay (into {} (map (fn [k] [k (requiring-resolve (symbol "gen.array.nn" (name k)))]))
               [:rms-norm :rope :sdpa :selective-log-softmax :gated-delta :quantize :dequantize :quantized-matmul :gather-qmm])))

(defn ^{:api :experimental} rms-norm
  "x normalized by its root mean square over the last axis, times weight:
   x·weight / √(mean(x², last axis) + eps). weight is an array of x's
   last axis, or nil for none; eps a non-negative host number. Ints
   promote to the default float; half floats are normalized in float32,
   as MLX does. A composite of gen.array's ops (so differentiated,
   batched and staged as they are), which MLX computes by its fused kernel
   (mlx_fast_rms_norm) on its own arrays in float32 and the half floats.
   Experimental: outside the 1.x promise (DESIGN §9)."
  [x weight eps]
  ((:rms-norm @nn) x weight eps))

(defn ^{:api :experimental} rope
  "Rotary position embedding (RoPE) of the first dims features of x's
   last axis, the position being x's axis -2 (x has 2 axes or more, as
   [B H T D] or [B T D]; dims even, at most D). Position t is
   (offset + t)·scale, its angles θᵢ = position · base^(-2i/dims) for
   i < dims/2 (or position / freqsᵢ), and each pair of features is
   rotated by its angle: NEOX's pairs (i, i + dims/2) by default, or the
   traditional consecutive pairs (2i, 2i + 1). Features past dims pass
   through. Options:
   - :traditional? (false)
   - :base (10000.0), or :freqs, an array of dims/2 frequencies, not both
   - :scale (1.0)
   - :offset (0): a host integer, or an int array of one offset, or of
     one per row of x's axis 0 (x with 3 axes or more): a batch of
     sequences at their own positions, as an LLM's lanes.
   Angles are computed in float32 for the half floats. A composite of
   gen.array's ops, which MLX computes by its fused kernel (mlx_fast_rope,
   mlx_fast_rope_dynamic) on its own arrays in float32 and the half
   floats. Experimental: outside the 1.x promise (DESIGN §9)."
  ([x dims] (rope x dims nil))
  ([x dims opts] ((:rope @nn) x dims opts)))

(defn ^{:api :experimental} sdpa
  "Scaled dot-product attention, softmax(scale·q·kᵀ + mask)·v, of queries
   q [B Hq Tq D] over keys k [B Hk Tk D] and values v [B Hk Tk Dv]. Hq is
   a multiple of Hk, and query head h reads key and value head
   ⌊h / (Hq/Hk)⌋ (grouped-query attention). Options:
   - :scale (1/√D)
   - :mask: nil; :causal, query i seeing key j where j <= i + Tk - Tq
     (the queries the last Tq positions); a bool array, true where a query
     sees a key; or a float array added to the scores. An array mask
     broadcasts to [B Hq Tq Tk].
   A query that sees no key (a bool mask hiding them all, or the first
   Tq - Tk queries of a causal mask) gets zeros: no value it is masked
   from reaches it, and no NaN (a row of -Inf scores in an additive mask
   is the caller's: NaN, and zeros by MLX's Metal kernels). The softmax
   is computed in float32 for the half floats. A composite of gen.array's ops, which MLX computes by
   its fused kernel (mlx_fast_scaled_dot_product_attention) on its own
   arrays in float32 and the half floats. Experimental: outside the 1.x
   promise (DESIGN §9)."
  ([q k v] (sdpa q k v nil))
  ([q k v opts] ((:sdpa @nn) q k v opts)))

(defn ^{:api :experimental} selective-log-softmax
  "log softmax(logits)[index] for each index, without the log-softmax:
   the logit at the index less the logsumexp over the last axis, the
   log-probability of a token under an LLM's logits. The logsumexp is
   taken :chunk entries at a time (logaddexp's identity over the chunks),
   so neither the log-softmax [.. V] nor exp of every logit at once is
   made.
   - logits: a float array [.. V], and indices an int array of its leading
     shape [..]; the result is [..].
   - Or logits a function of a range of entries, (f start stop), giving
     the logits of entries start to stop - 1 as a float array [.. stop -
     start] of the indices' shape, with :vocab V: the logits are made a
     chunk at a time and never whole (an LLM's output projection by rows,
     (fn [s e] (matmul h (transpose (slice w 0 s e))))). With :args, a
     vector of the arrays (or trees) f differentiates through, f is called
     (apply f start stop args) and each chunk is checkpointed over them
     (checkpoint): under value-and-grad its logits are made again in the
     backward pass, so the gradient's memory is one chunk's too. f must
     then take what it differentiates through in :args, not close over it.
   Options: :chunk (entries per chunk; default V, one chunk), :vocab, :args;
   another key is loud (:unknown-options). Indices outside [0, V) are loud
   where they can be read (the JVM backend) and clamped on a lazy backend
   (MLX), as take's. Half floats are reduced in float32, the result's
   dtype then. A composite of gen.array's ops, so differentiated, batched
   and staged as they are; its gradient is the full computation's within
   round-off (the chunks' logsumexps sum in another order).
   Experimental: outside the 1.x promise (DESIGN §9)."
  ([logits indices] (selective-log-softmax logits indices nil))
  ([logits indices opts] ((:selective-log-softmax @nn) logits indices opts)))
(defn ^{:api :experimental} gated-delta
  "The gated delta rule, GatedDeltaNet's recurrence (Yang, Kautz and
   Hatamizadeh, 2025; Qwen3.5's linear attention), over T tokens from a
   state: [y state'], each head's state S a [Dv Dk] matrix that every
   token t decays, corrects and reads,
     S ← exp(gₜ)·S,  S ← S + βₜ·(vₜ - S·kₜ)·kₜᵀ,  yₜ = S·qₜ,
   of q and k [B T H Dk], v [B T H Dv], g [B T H] (log decays, <= 0),
   beta [B T H] (write strengths, in [0, 1]) and state [B H Dv Dk]: y is
   [B T H Dv] and state' the state after the last token. Every head has
   its own keys (a model with fewer key heads repeats them first). The
   float operands take one dtype, computed in float32 for the half floats,
   as MLX's kernels do. Options:
   - :method: :loop, token by token; :chunked, in chunks of :chunk-size
     tokens (the form llama.cpp computes: a chunk's updates as matrix
     products, a triangular inverse by doubling blocks, then the chunks in
     order; T padded to whole chunks by tokens that leave the state as it
     is); or :auto (the default): a backend's fused kernel where it has
     one (MLX's Metal: mlx-lm's gated-delta kernel, gen.backend.mlx.nn),
     else :chunked for T of :chunk-size or more and :loop below. The forms
     are the same function, within float round-off.
   - :chunk-size: a power of 2 (64).
   A composite of gen.array's ops (so differentiated, batched and staged
   as they are). Experimental: outside the 1.x promise (DESIGN §9)."
  ([q k v g beta state] (gated-delta q k v g beta state nil))
  ([q k v g beta state opts] ((:gated-delta @nn) q k v g beta state opts)))

;; Paged KV caches (genclj-1daf): gen.array.paged holds their bodies,
;; loaded at the first call
(def ^:private paged
  (delay (into {} (map (fn [k] [k (requiring-resolve (symbol "gen.array.paged" (name k)))]))
               [:paged-write :copy-blocks :paged-attention])))

(defn ^{:api :experimental} paged-write
  "[k-pool' v-pool']: the keys k and values v [N Hk D] of N positions
   written over their slots (an int array [N], distinct) in a paged KV
   cache's pools, vLLM's layouts: the key pool [NB Hk D/x BS x] and the
   value pool [NB Hk D BS], NB blocks of BS positions, x the values of
   16 bytes in the pools' dtype (4 in float32, 8 in the half floats, 2 in
   float64). Slot s is offset s mod BS of block s/BS. k and v take the
   pools' dtype. Slots are checked (in [0, NB·BS), distinct) where their
   values can be read without forcing a computation (array/concrete?);
   on a lazy backend an out-of-range slot is unspecified. A composite of
   put-along-axis on the pools' elements, its reference; MLX writes by two
   scatters on the pools' block and offset axes (p/IFused's :paged-write,
   gen.backend.mlx.paged), which write a pool in place when nothing else
   holds it (array/compile's :donate) and copy it otherwise. Experimental:
   outside the 1.x promise (DESIGN §9)."
  [k-pool v-pool k v slots]
  ((:paged-write @paged) k-pool v-pool k v slots))

(defn ^{:api :experimental} copy-blocks
  "pool with its blocks dst (an int array [M], distinct) holding what its
   blocks src ([M]) held: a block's copy on axis 0, for copy-on-write
   (every block read before any is written). Indices checked as
   paged-write's slots. A composite of take and put-along-axis.
   Experimental: outside the 1.x promise (DESIGN §9)."
  [pool src dst]
  ((:copy-blocks @paged) pool src dst))

(defn ^{:api :experimental} paged-attention
  "The attention of one query a sequence over its positions in a paged KV
   cache: [B Hq D], of queries q [B Hq D] over the key and value pools
   (paged-write's layouts), sequence b's blocks the row b of tables
   [B MB] (block ids, an int array) and its positions the first lens[b]
   of them (lens an int array [B], each at most MB·BS; 0 gives zeros).
   Query head h reads KV head ⌊h / (Hq/Hk)⌋. q and the pools share one
   float dtype. Options: :scale (1/√D). Table entries past a sequence's
   blocks are never read, but must be blocks of the pool. Tables and lens
   are checked where readable (array/concrete?).
   A composite of gen.array's ops: each sequence's blocks gathered
   (take), then sdpa under a mask of its length, the reference that a
   backend's kernel is checked against. MLX's Metal computes it by
   kernels-community's paged_attention kernels (vLLM's, through
   mlx_fast_metal_kernel; gen.backend.mlx.paged-attention, ported per
   THIRD-PARTY-NOTICES), which read the blocks in place: v1, and past its
   threadgroup memory (tables of about 8,000 positions) v2, by partitions
   of 512 positions then a reduction; they compute the softmax in float32
   and add 1e-6 to its normalizer, as vLLM's do.
   Experimental: outside the 1.x promise (DESIGN §9)."
  ([q k-pool v-pool tables lens] (paged-attention q k-pool v-pool tables lens nil))
  ([q k-pool v-pool tables lens opts] ((:paged-attention @paged) q k-pool v-pool tables lens opts)))

(defn ^{:api :experimental} quantize
  "[wq scales biases]: the float matrices w [.. N K] quantized along
   their last axis, MLX's affine quantization: each group of :group-size
   consecutive values is scale·q + bias, q an unsigned integer of :bits
   bits, packed little-endian in uint32 words, so wq is [.. N K·bits/32]
   and scales and biases (w's dtype) [.. N K/group-size]. MLX's
   algorithm, computed in float32 on every backend: a group's scale
   spans its extrema over 2^bits - 1 steps, and the extremum of the
   larger magnitude is exact. Options: :group-size (64; 32, 64 or 128),
   :bits (4; 2, 3, 4, 5, 6 or 8), :mode (:affine, the one supported; MLX's
   :mxfp4, :nvfp4 and :mxfp8 keep uint8 scales, a dtype gen.array lacks,
   and are loud). Its results carry no gradient: quantizing a value being
   differentiated is loud. A primitive (gen.array.protocols/IQuantized);
   staging it is loud. Experimental: outside the 1.x promise (DESIGN §9)."
  ([w] (quantize w nil))
  ([w opts] ((:quantize @nn) w opts)))

(defn ^{:api :experimental} dequantize
  "The float matrices that quantize's wq, scales and biases stand for:
   scale·q + bias for each value. Options: :group-size, :bits and :mode,
   as quantize's (they must be the ones it used); :dtype (the scales' and
   biases' common float dtype). The weights are frozen: a gradient with
   respect to scales or biases is loud. A primitive (MLX's mlx_dequantize;
   the JVM backend unpacks on the host). Experimental: outside the 1.x
   promise (DESIGN §9)."
  ([wq scales biases] (dequantize wq scales biases nil))
  ([wq scales biases opts] ((:dequantize @nn) wq scales biases opts)))

(defn ^{:api :experimental} quantized-matmul
  "x·wᵀ for the matrix w that quantize's wq [N K·bits/32], scales and
   biases stand for (x [.. K], the result [.. N]), as a linear layer's
   weight is held; or x·w with :transpose? false (x [.. N], the result
   [.. K]). Options as dequantize's, and :transpose? (true). Its dtype is
   x's and the scales' common float dtype. Differentiable with respect to
   x only: the weights are frozen, and a gradient with respect to scales
   or biases is loud (LoRA trains adapters beside them). A primitive
   (MLX's mlx_quantized_matmul; the JVM backend dequantizes and
   multiplies). Experimental: outside the 1.x promise (DESIGN §9)."
  ([x wq scales biases] (quantized-matmul x wq scales biases nil))
  ([x wq scales biases opts] ((:quantized-matmul @nn) x wq scales biases opts)))

(defn ^{:api :experimental} gather-qmm
  "Matrix products of x's matrices with a stack of quantized matrices,
   gathered (a mixture of experts' layers): x [.. M K] (its matrices
   indexed flat over its leading axes), wq, scales and biases a stack
   [E .. N K·bits/32] as quantize gives one (indexed flat over its
   leading axes). Result i is x's matrix lhs[i] times w's matrix rhs[i]
   (transposed, by default), for :lhs-indices and :rhs-indices, int arrays
   that broadcast to the result's leading shape I: the result is
   [I.. M N]. Either defaults to every matrix in order, shaped as the
   array's leading axes. Options as quantized-matmul's, and :sorted?
   (MLX's hint that :rhs-indices are sorted). Out-of-range indices throw
   on the JVM backend and are clamped on MLX, as take's. Differentiable
   with respect to x only, as quantized-matmul. A primitive (MLX's
   mlx_gather_qmm; the JVM backend dequantizes, gathers and multiplies).
   Experimental: outside the 1.x promise (DESIGN §9)."
  ([x wq scales biases] (gather-qmm x wq scales biases nil))
  ([x wq scales biases opts] ((:gather-qmm @nn) x wq scales biases opts)))

(defn ^{:api :experimental} load-gguf
  "The tensors and metadata of a GGUF file (llama.cpp's model format), on
   *backend*: {:metadata :tensors :types :files}.
   - :metadata: the file's keys (strings) to host values: integers as
     longs, floats as doubles, strings, booleans, vectors of these.
   - :tensors: each tensor by its name in the file (llama.cpp's, e.g.
     \"blk.0.attn_q.weight\"), its shape outermost first (a matrix
     weight [out in]). F32 and I32 load as float32 and int32 arrays, F16
     and BF16 as float16 and bfloat16 (float32 on a backend without them,
     the JVM's: the same values). Q8_0, Q4_0,
     Q4_1, Q5_0, Q5_1, Q4_K and Q5_K load as MLX's affine quantization,
     {:wq :scales :biases :bits :group-size}, the arguments and options of
     dequantize and quantized-matmul: groups of 32 (a K-quant's
     sub-block), 8, 4 or 5 bits, the scales and biases float32, so the
     values are the file's exactly (llama.cpp's dequantization, bit for
     bit). Q6_K loads as float32 arrays, its values exactly: its
     sub-blocks of 16 fit no affine group of MLX's.
   - :types: each tensor's GGML type, a keyword (:f32, :q8_0, …).
   - :files: the paths read: path, or every file of a split model
     (<prefix>-00001-of-0000N.gguf, as llama.cpp's gguf-split writes them;
     path must be the first).
   Every other tensor type (Q2_K, Q3_K and the IQ types among them) is a
   loud error (:unsupported), as are a file that is no GGUF of
   version 2 or 3, a split's file missing, and a tensor named twice
   (:invalid-gguf). MLX reads the tensors by its loader (mlx_load_gguf),
   or a file holding a type its loader does not read by gen.array.gguf's
   host reader, as the JVM backend reads every file: the same values, bit
   for bit. A primitive
   (gen.array.protocols/ILoading); a backend without it is loud.
   Experimental: outside the 1.x promise (DESIGN §9)."
  [path]
  ((requiring-resolve 'gen.array.gguf/load-gguf) *backend* path))

(defn ^{:api :experimental} load-safetensors
  "The tensors of a safetensors file or checkpoint (Hugging Face's
   format), on *backend*: {:metadata :tensors :types :files}.
   - path: a .safetensors file; an index (model.safetensors.index.json:
     every shard its weight map names, beside it); or a checkpoint's
     directory, read by its index, or else its model.safetensors.
   - :metadata: the first file's \"__metadata__\" (strings to strings).
   - :tensors: each tensor by its name in the file (e.g.
     \"model.layers.0.self_attn.q_proj.weight\"), its shape as the file
     gives it (a matrix weight [out in]). F32, I32, I64 and BOOL load as
     float32, int32, int64 and bool arrays, F64 as float64 on the JVM
     backend (on MLX, whose loader has none, a file holding F64 is loud);
     F16 and BF16 as
     float16 and bfloat16 (float32 on a backend without them, the JVM's:
     the same values); U32 as uint32 (the packed words of MLX's quantized
     checkpoints, which gen.llm.models' loaders take as quantized
     matrices with their scales and biases).
   - :types: each tensor's dtype in the file, a keyword (:f32, :bf16, …).
   - :files: the paths read.
   Options: :select, a predicate of tensor names: only those it accepts
   load (the others are never read; a checkpoint's vision tower, for its
   language model).
   A tensor of another dtype (I8, I16, U8, U16, U64, the float8 types,
   C64), selected, is loud (:unsupported); so is an int64 value past 2^53
   on the JVM backend, whose doubles do not hold it exactly. A file cut
   short, a header that is no JSON object of tensor entries, tensors whose
   bytes do not cover the data exactly, and a shard missing or holding
   other tensors than the index maps to it are loud (:invalid-safetensors). MLX reads the tensors by its loader
   (mlx_load_safetensors), the JVM backend by gen.array.safetensors's host
   reader: the same values. A primitive (gen.array.protocols/ILoading); a
   backend without it is loud. Experimental: outside the 1.x promise
   (DESIGN §9)."
  ([path] (load-safetensors path nil))
  ([path opts]
   ((requiring-resolve 'gen.array.safetensors/load-safetensors) *backend* path opts)))

;; ---------------------------------------------------------------------------
;; Validation across particles
;; ---------------------------------------------------------------------------

(defn all-particles
  "Every particle of a batched value (gen.array.batch) on a leading axis, for
   a check or a bound over all of them; any other value as it is."
  [x]
  (if (instance? gen.array.protocols.IBatchedValue x) (p/-all-particles x) x))

(defn all-true?
  "Is every element of the bool array x true, across every particle of a
   batched value (gen.array.batch)? For validation, which must hold for all
   particles; branching uses truthy?, which a batch cannot answer."
  [x]
  (truthy? (all (all-particles x))))

(defn any-true?
  "Is some element of the bool array x true, across every particle?"
  [x]
  (truthy? (any (all-particles x))))

(defn equal?
  "Do x and y have the same shape and numerically equal elements? (NaN is
   never equal.)"
  [x y]
  (and (= (shape x) (shape y))
       (truthy? (all (== x y)))))

(defn allclose?
  "Are x and y elementwise within atol + rtol·|y| (broadcasting)? As in
   NumPy, an infinity is close only to itself (the band is infinite there).
   Options: :rtol (1e-5), :atol (1e-8), :equal-nan? (false)."
  ([x y] (allclose? x y nil))
  ([x y {:keys [rtol atol equal-nan?] :or {rtol 1e-5 atol 1e-8}}]
   (let [within (<= (abs (- x y)) (+ atol (* rtol (abs y))))
         close (logical-or (== x y) (logical-and within (logical-and (isfinite x) (isfinite y))))
         close (if equal-nan? (logical-or close (logical-and (isnan x) (isnan y))) close)]
     (truthy? (all close)))))
