(ns gen.array.host
  "The host side of gen.array, below the backends:
   - host data, parsed once for every backend: gen.array/array reads nested
     vectors into a shape and row-major values, checked, and gives the
     backend those (docs/array-api.md §10);
   - the math of host scalars, every elementwise op and reduction as
     gen.array computes them on the host, shared with the tape."
  {:api :internal}
  (:require [gen.array.dtype :as dt]
            [gen.array.special :as special]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- scalar? [x] (or (instance? Number x) (instance? Boolean x)))

(defn- nested-shape [data]
  (loop [shape [] d data]
    (if (sequential? d)
      (recur (conj shape (count d)) (first d))
      shape)))

(defn- leaves
  "The leaves of nested data in row-major order, checking that every level is
   rectangular and every leaf is a number or boolean."
  [data shape]
  (let [rank (count shape)
        walk (fn walk [d ^long ax]
               (if (== ax rank)
                 (if (scalar? d)
                   [d]
                   (throw (ex-info (str "Array element is not a number or boolean: " (pr-str d))
                                   {:element d})))
                 (if (and (sequential? d) (== (count d) (long (nth shape ax))))
                   (mapcat #(walk % (inc ax)) d)
                   (throw (ex-info "Nested data is not rectangular" {:shape shape :at-axis ax})))))]
    (walk data 0)))

(defn parse
  "Host data as [shape values], values in row-major order: a double[] is
   [[n] itself], and nested vectors their shape and leaves, checked
   rectangular, every leaf a number or boolean. (A number or boolean needs
   no parsing: gen.array/array makes it a 0-d array as it is.)"
  [data]
  (cond (instance? double/1 data) [[(alength ^doubles data)] data]
        (sequential? data) (let [shape (nested-shape data)] [shape (leaves data shape)])
        :else (throw (ex-info (str "Cannot make an array from " (type data)) {:data data}))))

(defn infer-dtype
  "The dtype of parsed values on a backend with these default dtypes: its
   float when any value is a float (or a ratio or decimal), and for no
   values or a double[]; :bool when every value is a boolean; its int
   otherwise."
  [values defaults]
  (cond (instance? double/1 values) (:float defaults)
        (empty? values) (:float defaults)
        (some #(or (instance? Double %) (instance? Float %) (ratio? %) (decimal? %)) values) (:float defaults)
        (every? #(instance? Boolean %) values) :bool
        :else (:int defaults)))

;; ---------------------------------------------------------------------------
;; Host math: the semantics of every elementwise op on host scalars
;; ---------------------------------------------------------------------------

(def ^:private host-defaults {:float :float64 :int :int64})

(defn- on-doubles [op ^double x ^double y]
  (case op
    :add (+ x y) :subtract (- x y) :multiply (* x y) :divide (/ x y)
    :power (Math/pow x y) :maximum (Math/max x y) :minimum (Math/min x y)
    :remainder (dt/floor-mod x y) :floor-divide (Math/floor (/ x y)) :arctan2 (Math/atan2 x y)
    :less (< x y) :less-equal (<= x y) :greater (> x y) :greater-equal (>= x y)
    :equal (== x y) :not-equal (not (== x y))
    :logical-and (and (not (== x 0.0)) (not (== y 0.0)))
    :logical-or (or (not (== x 0.0)) (not (== y 0.0)))))

(defn- on-longs [op ^long x ^long y]
  (case op
    :add (unchecked-add x y) :subtract (unchecked-subtract x y)
    :multiply (unchecked-multiply x y)
    :power (unchecked-long (Math/pow (double x) (double y)))
    :maximum (Math/max x y) :minimum (Math/min x y)
    :remainder (if (zero? y) 0 (Math/floorMod x y))
    :floor-divide (if (zero? y) 0 (Math/floorDiv x y))
    :less (< x y) :less-equal (<= x y) :greater (> x y) :greater-equal (>= x y)
    :equal (== x y) :not-equal (not (== x y))))

(defn- on-bools [op x y]
  (case op
    :logical-and (and x y) :logical-or (or x y)
    :equal (= x y) :not-equal (not= x y)
    :maximum (or x y) :minimum (and x y)
    (:less :less-equal :greater :greater-equal) (on-longs op (if x 1 0) (if y 1 0))))

(defn binary
  "The binary op on host scalars x and y."
  [op x y]
  (if (and (instance? Double x) (instance? Double y))
    (on-doubles op (.doubleValue ^Double x) (.doubleValue ^Double y))
    (let [in (dt/input-dtype op (dt/promote (dt/host-dtype x) (dt/host-dtype y)) host-defaults)]
      (case (dt/kind in)
        :float (on-doubles op (dt/host-double x) (dt/host-double y))
        :int (on-longs op (dt/host-long x) (dt/host-long y))
        :bool (on-bools op (dt/truthy? x) (dt/truthy? y))))))

(defn unary
  "The unary op on the host scalar x."
  [op x]
  (let [in (dt/input-dtype op (dt/host-dtype x) host-defaults)]
    (case (dt/kind in)
      :float (let [v (dt/host-double x)]
               (case op
                 :negative (- v) :abs (Math/abs v) :sign (Math/signum v)
                 :sqrt (Math/sqrt v) :exp (Math/exp v) :log (Math/log v)
                 :expm1 (Math/expm1 v) :log1p (Math/log1p v)
                 :sin (Math/sin v) :cos (Math/cos v) :tan (Math/tan v) :tanh (Math/tanh v)
                 :arcsin (Math/asin v) :arccos (Math/acos v) :arctan (Math/atan v)
                 :sinh (Math/sinh v) :cosh (Math/cosh v)
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

(defn where
  "x where c is truthy, y elsewhere, in their common dtype: :where on host
   scalars."
  [c x y]
  (dt/host-cast (if (dt/truthy? c) x y) (dt/promote (dt/host-dtype x) (dt/host-dtype y))))

(defn elementwise
  "The elementwise op on host scalar operands as gen.array computes it for
   them: on the host, whatever the backend. For backends that wrap others
   (the tape), so that a computation on host scalars gives the same values
   wrapped as unwrapped."
  [op args]
  (case op
    :where (apply where args)
    :stop-gradient (first args)
    (case (count args)
      1 (unary op (first args))
      2 (binary op (first args) (second args)))))

(defn reduction
  "The reduction op of a host scalar as gen.array computes it (see
   elementwise)."
  [op x]
  (case op
    (:sum :prod) (if (instance? Boolean x) (if x 1 0) x)
    (:max :min) x
    (:argmax :argmin) 0
    (:all :any) (dt/truthy? x)))
