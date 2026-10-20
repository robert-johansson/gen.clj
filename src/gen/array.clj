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
  (:refer-clojure :exclude [+ - * / < > <= >= == max min mod abs take flatten compile])
  (:require [clojure.core :as c]
            [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.array.special :as special]
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

(defn- backend-array? [x]
  (instance? gen.array.protocols.IArray x))

(defn array?
  "Is x an array value: a host scalar or a backend array?"
  [x]
  (or (host? x) (backend-array? x)))

(defn- not-an-array [op x]
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
  "Does backend b wrap inner (so that mixed operands run on b)?"
  [b inner]
  (and (instance? gen.array.protocols.IWrappingBackend b) (identical? (p/-inner b) inner)))

(defn- join-backends [op b1 b2]
  (cond (nil? b1) b2
        (nil? b2) b1
        (identical? b1 b2) b1
        (wraps? b1 b2) b1
        (wraps? b2 b1) b2
        :else (throw (ex-info (str (name op) ": arrays from different backends ("
                                   (p/-name b1) " and " (p/-name b2) ")")
                              {:op op :backends [(p/-name b1) (p/-name b2)]}))))

(defn- backend-of
  "The backend shared by the operands xs, or *backend* when all are host scalars."
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

(defn shape
  "The shape of x as a vector of longs ([] for 0-d)."
  [x]
  (if (host? x)
    []
    (p/-shape (operand-backend :shape x) x)))

(defn ndim [x] (count (shape x)))

(defn concrete?
  "Can x's values be read without forcing a lazy computation? (Host scalars
   and eager backends: yes.) Validation that reads values checks this first."
  [x]
  (or (host? x) (not (p/-lazy? (backend x)))))

(defn size [x] (sh/size (shape x)))

(defn- defaults [b] (p/-defaults b))

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
;; Host math: the semantics of every elementwise op on host scalars
;; ---------------------------------------------------------------------------

(def ^:private host-defaults {:float :float64 :int :int64})

(defn- host-dd [op ^double x ^double y]
  (case op
    :add (c/+ x y) :subtract (c/- x y) :multiply (c/* x y) :divide (c// x y)
    :power (Math/pow x y) :maximum (Math/max x y) :minimum (Math/min x y)
    :remainder (dt/floor-mod x y) :floor-divide (Math/floor (c// x y))
    :less (c/< x y) :less-equal (c/<= x y) :greater (c/> x y) :greater-equal (c/>= x y)
    :equal (c/== x y) :not-equal (not (c/== x y))
    :logical-and (and (not (c/== x 0.0)) (not (c/== y 0.0)))
    :logical-or (or (not (c/== x 0.0)) (not (c/== y 0.0)))))

(defn- host-ll [op ^long x ^long y]
  (case op
    :add (unchecked-add x y) :subtract (unchecked-subtract x y)
    :multiply (unchecked-multiply x y)
    :power (unchecked-long (Math/pow (double x) (double y)))
    :maximum (Math/max x y) :minimum (Math/min x y)
    :remainder (if (zero? y) 0 (Math/floorMod x y))
    :floor-divide (if (zero? y) 0 (Math/floorDiv x y))
    :less (c/< x y) :less-equal (c/<= x y) :greater (c/> x y) :greater-equal (c/>= x y)
    :equal (c/== x y) :not-equal (not (c/== x y))))

(defn- host-bb [op x y]
  (case op
    :logical-and (and x y) :logical-or (or x y)
    :equal (= x y) :not-equal (not= x y)
    :maximum (or x y) :minimum (and x y)
    (:less :less-equal :greater :greater-equal) (host-ll op (if x 1 0) (if y 1 0))))

(defn- host-binary [op x y]
  (if (and (instance? Double x) (instance? Double y))
    (host-dd op (.doubleValue ^Double x) (.doubleValue ^Double y))
    (let [in (dt/input-dtype op (dt/promote (dt/host-dtype x) (dt/host-dtype y)) host-defaults)]
      (case (dt/kind in)
        :float (host-dd op (dt/host-double x) (dt/host-double y))
        :int (host-ll op (dt/host-long x) (dt/host-long y))
        :bool (host-bb op (dt/truthy? x) (dt/truthy? y))))))

(defn- host-unary [op x]
  (let [in (dt/input-dtype op (dt/host-dtype x) host-defaults)]
    (case (dt/kind in)
      :float (let [v (dt/host-double x)]
               (case op
                 :negative (c/- v) :abs (Math/abs v) :sign (Math/signum v)
                 :sqrt (Math/sqrt v) :exp (Math/exp v) :log (Math/log v)
                 :expm1 (Math/expm1 v) :log1p (Math/log1p v)
                 :sin (Math/sin v) :cos (Math/cos v) :tan (Math/tan v) :tanh (Math/tanh v)
                 :erf (special/erf v) :erfinv (special/erfinv v)
                 :floor (Math/floor v) :ceil (Math/ceil v) :round (Math/rint v)
                 :isnan (Double/isNaN v) :isinf (Double/isInfinite v)))
      :int (let [v (dt/host-long x)]
             (case op
               :negative (unchecked-negate v) :abs (Math/abs v) :sign (Long/signum v)
               (:floor :ceil :round) v
               (:isnan :isinf) false))
      :bool (let [v (dt/truthy? x)]
              (case op
                :logical-not (not v)
                (:isnan :isinf) false)))))

(defn- host-where [c x y]
  (dt/host-cast (if (dt/truthy? c) x y) (dt/promote (dt/host-dtype x) (dt/host-dtype y))))

(defn host-elementwise
  "The elementwise op on host scalar operands as gen.array computes it for
   them: on the host, whatever the backend. For backends that wrap others
   (the tape), so that a computation on host scalars gives the same values
   wrapped as unwrapped."
  [op args]
  (case op
    :where (apply host-where args)
    :stop-gradient (first args)
    (case (count args)
      1 (host-unary op (first args))
      2 (host-binary op (first args) (second args)))))

;; ---------------------------------------------------------------------------
;; Elementwise dispatch
;; ---------------------------------------------------------------------------

(defn- ew1 [op x]
  (if (host? x)
    (host-unary op x)
    (let [b (operand-backend op x)
          d (dt/input-dtype op (p/-dtype b x) (defaults b))]
      (p/-elementwise b op [(cast-to b x d)] d))))

(defn- ew2 [op x y]
  (if (and (host? x) (host? y))
    (host-binary op x y)
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
(defbinary floor-div :floor-divide "Elementwise floor division.")
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
    (host-where c x y)
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
                (== 1 (sh/size s)) (dt/truthy? (loop [v (p/-to-host b x)] (if (sequential? v) (recur (first v)) v)))
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
  (let [zero? (logical-and (== x 0) (== y 0))]
    (where zero? 0.0 (* x (log (where zero? 1.0 y))))))

(defn xlog1py "Elementwise x·log1p(y), with 0·log1p(y) = 0 (also for y = -1)." [x y]
  (let [zero? (logical-and (== x 0) (== y -1))]
    (where zero? 0.0 (* x (log1p (where zero? 0.0 y))))))

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

(def ^:private lanczos-coefficients
  "Lanczos approximation, g = 7, n = 9 (relative error ~1e-15 for x >= 0.5)."
  [0.99999999999980993 676.5203681218851 -1259.1392167224028 771.32342877765313
   -176.61502916214059 12.507343278686905 -0.13857109526572012
   9.9843695780195716e-6 1.5056327351493116e-7])

(def ^:private half-log-2pi 0.9189385332046728)

(defn- lgamma-lanczos
  "log Γ(z) for z >= 0.5."
  [z]
  (let [z (- z 1)
        [c0 & cs] lanczos-coefficients
        series (reduce + c0 (map-indexed (fn [i c] (/ c (+ z (inc i)))) cs))
        t (+ z 7.5)]
    (+ half-log-2pi (- (* (+ z 0.5) (log t)) t) (log series))))

(defn lgamma
  "Elementwise log |Γ(x)| (+Inf at the poles 0, -1, -2, …, as SciPy): Lanczos
   for x >= 0.5, the reflection formula below."
  [x]
  (let [reflect? (< x 0.5)
        lg (lgamma-lanczos (where reflect? (- 1 x) x))
        xs (where reflect? x 0.25)
        pole? (logical-and (<= x 0) (== x (floor x)))]
    (where pole?
           ##Inf
           (where reflect?
                  (- (Math/log Math/PI) (log (abs (sin (* Math/PI xs)))) lg)
                  lg))))

(defn digamma
  "Elementwise digamma ψ(x) = d/dx log Γ(x): ten recurrence steps
   ψ(x) = ψ(x+1) - 1/x, then the asymptotic series (to x^-12), with the
   reflection formula for x < 0. As SciPy: -Inf at 0, NaN at negative integers."
  [x]
  (let [reflect? (< x 0)
        z (where reflect? (- 1 x) x)
        shift (reduce + (map #(/ (+ z %)) (range 10)))
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

(defn log-beta
  "Elementwise log B(a, b) = log Γ(a) + log Γ(b) - log Γ(a + b)."
  [a b]
  (- (+ (lgamma a) (lgamma b)) (lgamma (+ a b))))

;; ---------------------------------------------------------------------------
;; Creation and conversion
;; ---------------------------------------------------------------------------

(defn array
  "An array from host data: a number, a boolean, nested vectors of them, or a
   double[]. Options: :dtype."
  ([data] (array data nil))
  ([data {:keys [dtype]}]
   (p/-from-host *backend* data dtype)))

(defn full
  "An array of the given shape filled with value. Options: :dtype (default: the
   backend's default dtype for the kind of value)."
  ([shape value] (full shape value nil))
  ([shape value {dt :dtype}]
   (let [b *backend*]
     (p/-full b (vec shape) value
              (or dt
                  (if (host? value)
                    (dt/with-weak nil #{(dt/kind (dt/host-dtype value))} (defaults b))
                    ;; a (per-particle) array fill value keeps its dtype
                    (dtype value)))))))

(defn zeros
  ([shape] (zeros shape nil))
  ([shape {:keys [dtype]}] (full shape 0.0 {:dtype (or dtype (:float (defaults *backend*)))})))

(defn ones
  ([shape] (ones shape nil))
  ([shape {:keys [dtype]}] (full shape 1.0 {:dtype (or dtype (:float (defaults *backend*)))})))

(defn weak-dtype
  "The dtype x has on backend b (default *backend*): an array's own; a host
   scalar's is weak, its kind's default dtype there (so float64 only where
   that is the default)."
  ([x] (weak-dtype x *backend*))
  ([x b]
   (if (host? x) (dt/with-weak nil #{(dt/kind (dt/host-dtype x))} (defaults b)) (dtype x))))

(defn full-like
  "An array with x's shape, backend and (unless given) dtype, filled with
   value. A host scalar x is weakly typed (weak-dtype)."
  ([x value] (full-like x value nil))
  ([x value {dt :dtype}]
   (let [b (backend x)]
     (p/-full b (shape x) value (or dt (weak-dtype x b))))))

(defn zeros-like [x] (full-like x 0))
(defn ones-like [x] (full-like x 1))

(defn arange
  "A 1-d array [start, stop) by step. Integral arguments give the default int
   dtype, otherwise the default float. Options (4-arity): :dtype."
  ([stop] (arange 0 stop 1 nil))
  ([start stop] (arange start stop 1 nil))
  ([start stop step] (arange start stop step nil))
  ([start stop step {:keys [dtype]}]
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

(defn item
  "The host number or boolean of a single-element array."
  [x]
  (if (host? x)
    x
    (let [s (shape x)]
      (if (== 1 (sh/size s))
        (loop [v (->clj x)] (if (sequential? v) (recur (first v)) v))
        (throw (ex-info (str "item needs a single-element array, got shape " s)
                        {:shape s}))))))

;; ---------------------------------------------------------------------------
;; Trees and evaluation
;; ---------------------------------------------------------------------------

(defn tree-leaves
  "The leaves of a tree of nested maps, vectors and seqs, depth first."
  [t]
  (cond (map? t) (mapcat tree-leaves (vals t))
        (or (vector? t) (seq? t)) (mapcat tree-leaves t)
        :else [t]))

(defn tree-map
  "The tree t with f applied to every leaf (maps, including records, keep their
   type)."
  [f t]
  (cond (map? t) (reduce-kv (fn [m k v] (assoc m k (tree-map f v))) t t)
        (vector? t) (mapv #(tree-map f %) t)
        (seq? t) (doall (map #(tree-map f %) t))
        :else (f t)))

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

(defn host-reduce
  "The reduction op of a host scalar as gen.array computes it (see
   host-elementwise)."
  [op x]
  (case op
    (:sum :prod) (if (instance? Boolean x) (if x 1 0) x)
    (:max :min) x
    (:argmax :argmin) 0
    (:all :any) (dt/truthy? x)))

(defn- reduce* [op x {:keys [axis keep-dims?]}]
  (if (host? x)
    (do (sh/axes axis 0)
        (host-reduce op x))
    (let [b (operand-backend op x)
          axes (sh/axes axis (count (p/-shape b x)))
          d (dt/reduce-input-dtype op (p/-dtype b x) (defaults b))]
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
   (let [kd (assoc opts :keep-dims? true)
         m (stop-gradient (max x kd))
         m (where (isfinite m) m 0.0)
         r (+ (log (sum (exp (- x m)) kd)) m)]
     (if keep-dims? r (squeeze r (sh/axes (:axis opts) (ndim x)))))))

(defn softmax "exp(x) normalized along :axis (default -1)."
  ([x] (softmax x nil))
  ([x {:keys [axis] :or {axis -1}}]
   (let [m (stop-gradient (max x {:axis axis :keep-dims? true}))
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
        axes (set (sh/axes axes (c/+ (count s) (if (number? axes) 1 (count axes)))))
        r (c/+ (count s) (count axes))]
    (reshape x (loop [i 0 src s out []]
                 (if (= i r)
                   out
                   (if (axes i)
                     (recur (inc i) src (conj out 1))
                     (recur (inc i) (rest src) (conj out (first src)))))))))

(defn broadcast-to "x broadcast to shape." [x new-shape]
  (p/-broadcast-to (backend-of :broadcast-to [x]) x (mapv long new-shape)))

(defn concatenate
  "Join arrays along an existing axis (default 0)."
  ([xs] (concatenate xs 0))
  ([xs axis]
   (let [b (backend-of :concatenate xs)
         d (common-dtype b xs)]
     (p/-concatenate b (mapv #(cast-to b % d) xs) (sh/axis (long axis) (ndim (first xs)))))))

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
   Negative indices count from the end; out-of-range indices throw."
  ([x indices] (take x indices 0))
  ([x indices axis]
   (check-indices :take indices)
   (let [b (backend-of :take [x indices])]
     (p/-take b x indices (sh/axis (long axis) (ndim x))))))

(defn take-along-axis
  "Gather along axis with an int array of the same rank as x, broadcast against
   x on every other axis."
  [x indices axis]
  (check-indices :take-along-axis indices)
  (let [b (backend-of :take-along-axis [x indices])]
    (p/-take-along-axis b x indices (sh/axis (long axis) (ndim x)))))

(defn scatter-add
  "x with updates added at indices along axis, accumulating repeated indices:
   the transpose of take-along-axis. indices (an int array) and updates have
   one shape, equal to x's on every axis but axis."
  [x indices updates axis]
  (check-indices :scatter-add indices)
  (let [b (backend-of :scatter-add [x updates indices])
        d (common-dtype b [x updates])]
    (p/-scatter-add b (cast-to b x d) indices (cast-to b updates d) (sh/axis (long axis) (ndim x)))))

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
        x' (reshape-on b x (pad sx (inc batch-rank)))
        i' (reshape-on b idx (conj (pad si batch-rank) 1))
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
    1 (* (with-backend (backend x) (eye (size x) {:dtype (dtype x)})) x)
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

(defn value-and-grad
  "A function of the same arguments as f returning [value gradient]: f's
   value (a 0-d array) and its gradient with respect to the argument at
   :argnums (default 0), or a vector of gradients when :argnums is a vector.
   An argument may be a tree of maps and vectors of float arrays and numbers;
   its gradient has the same structure. stop-gradient blocks differentiation.
   First order only: f may not itself differentiate."
  ([f] (value-and-grad f nil))
  ([f {:keys [argnums] :or {argnums 0}}]
   (fn [& args]
     (let [args (vec args)
           positions (if (sequential? argnums) (vec argnums) [argnums])
           b (backend-of :value-and-grad (mapcat #(tree-leaves (nth args %)) positions))
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
;; Compilation (docs/array-api.md §8)
;; ---------------------------------------------------------------------------

(defn- skeleton [t] (tree-map (fn [_] nil) t))

(defn- refill
  "The tree skeleton with its leaves, in tree-leaves order, replaced by xs."
  [skel xs]
  (let [it (volatile! (seq xs))]
    (tree-map (fn [_] (let [x (first @it)] (vswap! it next) x)) skel)))

(defn- leaves-in
  "t's leaves in the order of skel, a skeleton equal to t's. Equal maps may
   list their keys in different orders, so a cached trace's arguments
   follow its own skeleton, not the call's."
  [skel t]
  (cond (map? skel) (mapcat (fn [k] (leaves-in (get skel k) (get t k))) (keys skel))
        (or (vector? skel) (seq? skel)) (mapcat leaves-in skel t)
        :else [t]))

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
   Options: :shapeless? (MLX: one trace for every shape)."
  ([f] (compile f nil))
  ([f opts]
   (let [compiled (atom {})]
     (fn [& args]
       (let [args (vec args)
             leaves (vec (tree-leaves args))]
         (doseq [l leaves]
           (when-not (array? l)
             (throw (ex-info (str "compile: the arguments must be arrays (or trees of them), not "
                                  (pr-str l)
                                  (when (instance? gen.random.threefry.Key l)
                                    ": a key's draws would be replayed by every call; draw outside, or close over the key"))
                             {:value l}))))
         (let [b (backend-of :compile leaves)]
           (if-not (satisfies? p/ICompiling b)
             (apply f args)
             (let [skel (skeleton args)
                   entry (or (get @compiled [b skel])
                             (let [out-skel (atom nil)
                                   g (p/-compile b (fn [ins]
                                                     (let [out (apply f (refill skel ins))]
                                                       (reset! out-skel (skeleton out))
                                                       (vec (tree-leaves out))))
                                                 opts)
                                   e {:g g :out out-skel :in skel}]
                               (swap! compiled assoc [b skel] e)
                               e))
                   ins (with-backend b (mapv #(if (host? %) (array % {:dtype (weak-dtype % b)}) %)
                                             (leaves-in (:in entry) args)))]
               ;; apply first: the first call traces f, which records the output's structure
               (let [outs ((:g entry) ins)]
                 (refill @(:out entry) outs))))))))))

;; ---------------------------------------------------------------------------
;; Comparison helpers (for tests and assertions)
;; ---------------------------------------------------------------------------

(defn all-true?
  "Is every element of the bool array x true, across every particle of a
   batched value (gen.array.batch)? For validation, which must hold for all
   particles; branching uses truthy?, which a batch cannot answer."
  [x]
  (truthy? (all (if (instance? gen.array.protocols.IBatchedValue x) (p/-all-particles x) x))))

(defn any-true?
  "Is some element of the bool array x true, across every particle?"
  [x]
  (truthy? (any (if (instance? gen.array.protocols.IBatchedValue x) (p/-all-particles x) x))))

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
