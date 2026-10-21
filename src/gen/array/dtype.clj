(ns gen.array.dtype
  "Dtypes, the promotion lattice, and the dtype signature of every primitive
   op. Defined once here so every backend computes identical result types.

   Lattice: bool < ints < floats. Mixed ints promote to :int64; ints meet floats
   at the float (JAX's rule, kind to float32 on the GPU); differing floats meet
   at :float32 unless one is :float64.

   Host numbers are weakly typed: they adopt the dtype of the arrays they meet
   and only raise the kind (a host float meeting an int array gives the
   backend's default float).")

(def dtypes #{:bool :int32 :int64 :uint32 :float16 :bfloat16 :float32 :float64})

(defn kind
  "The kind of a dtype: :bool, :int or :float."
  [dtype]
  (case dtype
    :bool :bool
    (:int32 :int64 :uint32) :int
    (:float16 :bfloat16 :float32 :float64) :float))

(defn host-dtype
  "The dtype of a host scalar: Boolean is :bool, floating numbers (and ratios
   and decimals) are :float64, every other number is :int64."
  [x]
  (cond (instance? Boolean x) :bool
        (or (instance? Double x) (instance? Float x)
            (ratio? x) (decimal? x)) :float64
        :else :int64))

(defn promote
  "The common dtype of two (strong) dtypes."
  [a b]
  (if (= a b)
    a
    (let [ka (kind a) kb (kind b)]
      (cond (= ka :bool) b
            (= kb :bool) a
            (and (= ka :int) (= kb :int)) :int64
            (= ka :int) b
            (= kb :int) a
            (or (= a :float64) (= b :float64)) :float64
            :else :float32))))

(defn with-weak-kind
  "Combine a strong dtype with the kind of one weak host operand it meets."
  [strong weak-kind defaults]
  (case weak-kind
    :float (if (= :float (kind strong)) strong (:float defaults))
    :int (if (= :bool strong) (:int defaults) strong)
    :bool strong))

(defn with-weak
  "Combine a strong dtype (or nil when there is none) with the set of kinds of
   the weak host operands it meets."
  [strong weak-kinds defaults]
  (if (nil? strong)
    (cond (contains? weak-kinds :float) (:float defaults)
          (contains? weak-kinds :int) (:int defaults)
          :else :bool)
    (let [k (kind strong)]
      (cond (and (contains? weak-kinds :float) (not= k :float)) (:float defaults)
            (and (contains? weak-kinds :int) (= k :bool)) (:int defaults)
            :else strong))))

;; ---------------------------------------------------------------------------
;; Primitive elementwise ops and their dtype classes
;; ---------------------------------------------------------------------------

(def elementwise-ops
  "Every primitive elementwise op, by dtype class:
     :arith   bool operands promote to the default int; result as input
     :float   non-float operands promote to the default float
     :round   bool promotes to int; ints pass through
     :compare result is :bool
     :logic   operands become :bool; result is :bool
     :predicate  result is :bool
     :same    result is the operand dtype"
  {:add :arith :subtract :arith :multiply :arith :power :arith
   :remainder :arith :floor-divide :arith :negative :arith :abs :arith :sign :arith
   :divide :float :exp :float :log :float :expm1 :float :log1p :float :sqrt :float
   :sin :float :cos :float :tan :float :tanh :float :erf :float :erfinv :float :logaddexp :float
   :arcsin :float :arccos :float :arctan :float :arctan2 :float :sinh :float :cosh :float
   :floor :round :ceil :round :round :round
   :less :compare :less-equal :compare :greater :compare :greater-equal :compare
   :equal :compare :not-equal :compare
   :logical-and :logic :logical-or :logic :logical-not :logic
   :isnan :predicate :isinf :predicate
   :maximum :same :minimum :same :where :same :stop-gradient :same})

(defmacro ^:private case-table
  "A fn from op to class, compiled to a case (a hash switch) from the table."
  [table]
  `(fn [op#] (case op# ~@(mapcat identity @(resolve table)) nil)))

(def ^:private op-class (case-table elementwise-ops))

(defn input-dtype
  "The dtype op's operands are converted to, given their common dtype."
  [op common defaults]
  (case (or (op-class op)
            (throw (ex-info (str "Unknown elementwise op " op) {:op op})))
    :arith (if (= common :bool) (:int defaults) common)
    :float (if (= (kind common) :float) common (:float defaults))
    :round (if (= common :bool) (:int defaults) common)
    :logic :bool
    (:compare :predicate :same) common))

(defn result-dtype
  "The dtype of op's result, given the operand dtype."
  [op in]
  (case (op-class op)
    (:compare :logic :predicate) :bool
    in))

(def reduce-ops #{:sum :prod :max :min :argmax :argmin :all :any})

(defn reduce-input-dtype
  "The dtype a reduction's operand is converted to."
  [op dtype defaults]
  (case op
    (:sum :prod) (if (= dtype :bool) (:int defaults) dtype)
    (:all :any) :bool
    dtype))

(defn reduce-result-dtype
  [op in defaults]
  (case op
    (:argmax :argmin) (:int defaults)
    (:all :any) :bool
    in))

;; ---------------------------------------------------------------------------
;; Host scalars
;; ---------------------------------------------------------------------------

(defn floor-mod
  "Remainder with the sign of the divisor (NumPy and Python semantics): NaN for
   a zero divisor, x - y·floor(x/y) otherwise. Built on the exact
   IEEEremainder (clojure.core/rem throws on zero and is inexact for large
   quotients)."
  ^double [^double x ^double y]
  (let [r (Math/IEEEremainder x y)
        ;; C's fmod (sign of x), exact
        m (if (and (not (== r 0.0)) (not= (neg? r) (neg? x))) (+ r (Math/copySign y x)) r)]
    (cond (Double/isNaN m) m
          (== m 0.0) (Math/copySign 0.0 y)
          (not= (neg? m) (neg? y)) (+ m y)
          :else m)))

(defn truthy?
  "Host truthiness: booleans as themselves, numbers when nonzero (NaN is true)."
  [x]
  (if (instance? Boolean x)
    (.booleanValue ^Boolean x)
    (not (== 0.0 (.doubleValue ^Number x)))))

(defn host-double
  ^double [x]
  (if (instance? Boolean x)
    (if (.booleanValue ^Boolean x) 1.0 0.0)
    (.doubleValue ^Number x)))

(defn host-long
  ^long [x]
  (if (instance? Boolean x)
    (if (.booleanValue ^Boolean x) 1 0)
    (.longValue ^Number x)))

(defn float->int
  "x cast to the int dtype as Java casts a double (DESIGN.md §9): truncated
   toward zero, saturating at the dtype's range (the infinities too), NaN
   0. :uint32 saturates at 0 and 2^32 - 1."
  ^long [^double x dtype]
  (case dtype
    :int64 (unchecked-long x)
    :int32 (long (unchecked-int x))
    :uint32 (cond (Double/isNaN x) 0 (<= x 0.0) 0 (>= x 4294967295.0) 4294967295 :else (unchecked-long x))))

(defn int->int
  "x wrapped to the int dtype's width, as Java's narrowing casts."
  ^long [^long x dtype]
  (case dtype
    :int64 x
    :int32 (long (unchecked-int x))
    :uint32 (bit-and x 0xFFFFFFFF)))

(defn host-cast
  "Convert a host scalar to the host representation of dtype: a Double for
   float dtypes (rounded to float32 precision for :float32), a Long for ints
   (a float cast as Java casts, float->int; an integer wrapped, int->int), a
   Boolean for :bool."
  [x dtype]
  (case dtype
    :float64 (host-double x)
    ;; unchecked: a double beyond float32's range rounds to an infinity
    (:float32 :float16 :bfloat16) (double (unchecked-float (host-double x)))
    (:int64 :int32 :uint32) (if (or (instance? Double x) (instance? Float x))
                              (float->int (host-double x) dtype)
                              (int->int (host-long x) dtype))
    :bool (truthy? x)))
