(ns gen.backend.jvm
  "The pure-JVM backend: eager n-d arrays over a double[], row-major and
   contiguous, computed with primitive loops.

   - 0-d arrays are host values (Double, Long, Boolean), so scalar code never
     allocates an array.
   - Every dtype is stored as doubles and the dtype tag carries the meaning:
     ints are exact up to 2^53, bools are 0.0/1.0, and :float32/:int32 results
     are rounded to their type, so the backend can mirror MLX's float32 numerics.
   - Arrays are immutable: no operation writes to an array it did not create,
     which is what lets reshape share data."
  (:require [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.array.special :as special]
            [gen.random.threefry :as threefry]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(declare backend to-host)

(deftype JvmArray [^doubles data shape dtype]
  p/IArray
  (-backend [_] backend)

  ;; immutable values with value equality: the same dtype, shape and
  ;; (bitwise) elements
  Object
  (equals [this o]
    (or (identical? this o)
        (and (instance? JvmArray o)
             (= dtype (.-dtype ^JvmArray o))
             (= shape (.-shape ^JvmArray o))
             (java.util.Arrays/equals data ^doubles (.-data ^JvmArray o)))))
  (hashCode [_] (hash-combine (hash shape) (java.util.Arrays/hashCode data)))

  clojure.lang.IHashEq
  (hasheq [this] (.hashCode this)))

(def supported-dtypes #{:bool :int32 :int64 :float32 :float64})

(defn- check-dtype [dtype]
  (when-not (contains? supported-dtypes dtype)
    (throw (ex-info (str "The JVM backend does not support dtype " dtype)
                    {:dtype dtype :supported supported-dtypes})))
  dtype)

;; ---------------------------------------------------------------------------
;; Operand views: host scalars are 0-d arrays
;; ---------------------------------------------------------------------------

(defn- jvm-array? [x] (instance? JvmArray x))

(defn- data-of ^doubles [x]
  (if (jvm-array? x)
    (.-data ^JvmArray x)
    (doto (double-array 1) (aset 0 (dt/host-double x)))))

(defn- shape-of [x]
  (if (jvm-array? x) (.-shape ^JvmArray x) []))

(defn- dtype-of [x]
  (if (jvm-array? x) (.-dtype ^JvmArray x) (dt/host-dtype x)))

(defn- box
  "The host value of a stored double under dtype."
  [^double v dtype]
  (case (dt/kind dtype)
    :float v
    :int (long v)
    :bool (not (== v 0.0))))

(defn- conform!
  "Round freshly computed data to dtype in place (only for 32-bit dtypes)."
  ^doubles [^doubles a dtype]
  (case dtype
    :float32 (dotimes [i (alength a)] (aset a i (double (float (aget a i)))))
    :int32   (dotimes [i (alength a)] (aset a i (double (unchecked-int (long (aget a i))))))
    nil)
  a)

(defn- make
  "The array (or host scalar, for 0-d) holding freshly computed data."
  [^doubles data shape dtype]
  (let [data (conform! data dtype)]
    (if (zero? (count shape))
      (box (aget data 0) dtype)
      (JvmArray. data shape dtype))))

(defn- gather ^doubles [^doubles src ^longs offs]
  (let [n (alength offs) out (double-array n)]
    (dotimes [i n] (aset out i (aget src (aget offs i))))
    out))

(defn- row-layout
  "How to read an operand of shape s broadcast to os one row (last axis) at a
   time: [row-offsets inner-stride], one offset per row of os."
  [s os]
  (let [r (count os)
        st (sh/broadcast-strides s os)]
    [(sh/strided-offsets (subvec os 0 (dec r)) (java.util.Arrays/copyOf st (int (dec r))) 0)
     (aget st (dec r))]))

;; ---------------------------------------------------------------------------
;; Elementwise kernels. Each op gets its own small loop so the JIT can compile
;; and vectorize it (one giant dispatch method would exceed the JIT's size limit).
;; ---------------------------------------------------------------------------

(defmacro ^:private defunary [fname [x] expr]
  `(defn- ~fname ^"[D" [^"[D" a#]
     (let [n# (alength a#) out# (double-array n#)]
       (dotimes [i# n#]
         (let [~x (aget a# i#)]
           (aset out# i# (double ~expr))))
       out#)))

(defunary u-negative [x] (- x))
(defunary u-abs [x] (Math/abs x))
(defunary u-sign [x] (Math/signum x))
(defunary u-sqrt [x] (Math/sqrt x))
(defunary u-exp [x] (Math/exp x))
(defunary u-log [x] (Math/log x))
(defunary u-expm1 [x] (Math/expm1 x))
(defunary u-log1p [x] (Math/log1p x))
(defunary u-sin [x] (Math/sin x))
(defunary u-cos [x] (Math/cos x))
(defunary u-tan [x] (Math/tan x))
(defunary u-tanh [x] (Math/tanh x))
(defunary u-erf [x] (special/erf x))
(defunary u-erfinv [x] (special/erfinv x))
(defunary u-floor [x] (Math/floor x))
(defunary u-ceil [x] (Math/ceil x))
(defunary u-round [x] (Math/rint x))
(defunary u-isnan [x] (if (Double/isNaN x) 1.0 0.0))
(defunary u-isinf [x] (if (Double/isInfinite x) 1.0 0.0))
(defunary u-not [x] (if (== x 0.0) 1.0 0.0))

(defn- unary ^doubles [op ^doubles a]
  (case op
    :negative (u-negative a) :abs (u-abs a) :sign (u-sign a) :sqrt (u-sqrt a)
    :exp (u-exp a) :log (u-log a) :expm1 (u-expm1 a) :log1p (u-log1p a)
    :sin (u-sin a) :cos (u-cos a) :tan (u-tan a) :tanh (u-tanh a)
    :erf (u-erf a) :erfinv (u-erfinv a)
    :floor (u-floor a) :ceil (u-ceil a) :round (u-round a)
    :isnan (u-isnan a) :isinf (u-isinf a) :logical-not (u-not a)))

(defmacro ^:private defbinary
  "A broadcasting binary kernel. Each loop is its own small fn so the JIT can
   compile and vectorize it: equal shapes, a size-1 operand on either side, and
   the general case (row by row, with per-row base offsets)."
  [fname [x y] expr]
  (let [same (symbol (str fname "-same"))
        right (symbol (str fname "-scalar-right"))
        left (symbol (str fname "-scalar-left"))
        general (symbol (str fname "-general"))]
    `(do
       (defn- ~same ^"[D" [^"[D" a# ^"[D" b# n#]
         (let [n# (long n#) out# (double-array n#)]
           (dotimes [i# n#]
             (let [~x (aget a# i#) ~y (aget b# i#)] (aset out# i# (double ~expr))))
           out#))
       (defn- ~right ^"[D" [^"[D" a# ~y]
         (let [~y (double ~y) n# (alength a#) out# (double-array n#)]
           (dotimes [i# n#] (let [~x (aget a# i#)] (aset out# i# (double ~expr))))
           out#))
       (defn- ~left ^"[D" [~x ^"[D" b#]
         (let [~x (double ~x) n# (alength b#) out# (double-array n#)]
           (dotimes [i# n#] (let [~y (aget b# i#)] (aset out# i# (double ~expr))))
           out#))
       (defn- ~general ^"[D" [^"[D" a# sa# ^"[D" b# sb# os#]
         (let [n# (sh/size os#) out# (double-array n#)
               inner# (long (peek os#))
               la# (row-layout sa# os#) lb# (row-layout sb# os#)
               ^"[J" ra# (nth la# 0) ia# (long (nth la# 1))
               ^"[J" rb# (nth lb# 0) ib# (long (nth lb# 1))]
           (dotimes [o# (alength ra#)]
             (let [ba# (aget ra# o#) bb# (aget rb# o#) bo# (* o# inner#)]
               (dotimes [j# inner#]
                 (let [~x (aget a# (+ ba# (* j# ia#))) ~y (aget b# (+ bb# (* j# ib#)))]
                   (aset out# (+ bo# j#) (double ~expr))))))
           out#))
       (defn- ~fname ^"[D" [^"[D" a# sa# ^"[D" b# sb# os#]
         (cond
           (and (= sa# os#) (= sb# os#)) (~same a# b# (alength a#))
           (and (= sa# os#) (== 1 (alength b#))) (~right a# (aget b# 0))
           (and (= sb# os#) (== 1 (alength a#))) (~left (aget a# 0) b#)
           :else (~general a# sa# b# sb# os#))))))

(defbinary b-add [x y] (+ x y))
(defbinary b-subtract [x y] (- x y))
(defbinary b-multiply [x y] (* x y))
(defbinary b-divide [x y] (/ x y))
(defbinary b-power [x y] (Math/pow x y))
(defbinary b-power-int [x y] (unchecked-long (Math/pow x y)))
(defbinary b-maximum [x y] (Math/max x y))
(defbinary b-minimum [x y] (Math/min x y))
(defbinary b-remainder [x y] (dt/floor-mod x y))
(defbinary b-remainder-int [x y] (if (== y 0.0) 0 (Math/floorMod (long x) (long y))))
(defbinary b-floor-divide [x y] (Math/floor (/ x y)))
(defbinary b-floor-divide-int [x y] (if (== y 0.0) 0 (Math/floorDiv (long x) (long y))))
(defbinary b-less [x y] (if (< x y) 1.0 0.0))
(defbinary b-less-equal [x y] (if (<= x y) 1.0 0.0))
(defbinary b-greater [x y] (if (> x y) 1.0 0.0))
(defbinary b-greater-equal [x y] (if (>= x y) 1.0 0.0))
(defbinary b-equal [x y] (if (== x y) 1.0 0.0))
(defbinary b-not-equal [x y] (if (== x y) 0.0 1.0))
(defbinary b-and [x y] (if (and (not (== x 0.0)) (not (== y 0.0))) 1.0 0.0))
(defbinary b-or [x y] (if (or (not (== x 0.0)) (not (== y 0.0))) 1.0 0.0))

(defn- binary ^doubles [op dtype a sa b sb os]
  (let [int? (= :int (dt/kind dtype))]
    (case op
      :add (b-add a sa b sb os)
      :subtract (b-subtract a sa b sb os)
      :multiply (b-multiply a sa b sb os)
      :divide (b-divide a sa b sb os)
      :power (if int? (b-power-int a sa b sb os) (b-power a sa b sb os))
      :maximum (b-maximum a sa b sb os)
      :minimum (b-minimum a sa b sb os)
      :remainder (if int? (b-remainder-int a sa b sb os) (b-remainder a sa b sb os))
      :floor-divide (if int? (b-floor-divide-int a sa b sb os) (b-floor-divide a sa b sb os))
      :less (b-less a sa b sb os)
      :less-equal (b-less-equal a sa b sb os)
      :greater (b-greater a sa b sb os)
      :greater-equal (b-greater-equal a sa b sb os)
      :equal (b-equal a sa b sb os)
      :not-equal (b-not-equal a sa b sb os)
      :logical-and (b-and a sa b sb os)
      :logical-or (b-or a sa b sb os))))

(defn- where-kernel ^doubles [^doubles c sc ^doubles x sx ^doubles y sy os]
  (let [n (sh/size os) out (double-array n)]
    (if (and (= sc os) (= sx os) (= sy os))
      (dotimes [i n]
        (aset out i (if (== (aget c i) 0.0) (aget y i) (aget x i))))
      (let [inner (long (peek os))
            [^longs rc ic] (row-layout sc os)
            [^longs rx ix] (row-layout sx os)
            [^longs ry iy] (row-layout sy os)
            ic (long ic) ix (long ix) iy (long iy)]
        (dotimes [o (alength rc)]
          (let [bc (aget rc o) bx (aget rx o) by (aget ry o) bo (* o inner)]
            (dotimes [j inner]
              (aset out (+ bo j) (if (== (aget c (+ bc (* j ic))) 0.0)
                                   (aget y (+ by (* j iy)))
                                   (aget x (+ bx (* j ix))))))))))
    out))

(defn- elementwise [op args dtype]
  (let [out-dt (dt/result-dtype op dtype)]
    (case (count args)
      1 (let [x (nth args 0)]
          (if (= op :stop-gradient)
            x
            (make (unary op (data-of x)) (shape-of x) out-dt)))
      2 (let [x (nth args 0) y (nth args 1)
              sx (shape-of x) sy (shape-of y)
              os (sh/broadcast sx sy)]
          (make (binary op dtype (data-of x) sx (data-of y) sy os) os out-dt))
      3 (let [[c x y] args
              sc (shape-of c) sx (shape-of x) sy (shape-of y)
              os (sh/broadcast-all [sc sx sy])]
          (make (where-kernel (data-of c) sc (data-of x) sx (data-of y) sy os) os out-dt)))))

;; ---------------------------------------------------------------------------
;; Reductions and scans
;; ---------------------------------------------------------------------------

(defmacro ^:private defreduce
  "A reduction over the middle axis of src viewed as [outer len inner]: a
   scalar accumulation per output when inner is 1, row accumulation otherwise."
  [fname [acc v] expr]
  `(defn- ~fname ^"[D" [^"[D" src# outer# len# inner# init#]
     (let [outer# (long outer#) len# (long len#) inner# (long inner#) init# (double init#)
           out# (double-array (* outer# inner#))]
       (if (== 1 inner#)
         (dotimes [o# outer#]
           (let [base# (* o# len#)]
             (loop [k# 0 ~acc init#]
               (if (< k# len#)
                 (recur (inc k#) (let [~v (aget src# (+ base# k#))] (double ~expr)))
                 (aset out# o# ~acc)))))
         (dotimes [o# outer#]
           (let [obase# (* o# inner#) sbase# (* o# len# inner#)]
             (dotimes [i# inner#] (aset out# (+ obase# i#) init#))
             (dotimes [k# len#]
               (let [row# (+ sbase# (* k# inner#))]
                 (dotimes [i# inner#]
                   (let [~acc (aget out# (+ obase# i#)) ~v (aget src# (+ row# i#))]
                     (aset out# (+ obase# i#) (double ~expr)))))))))
       out#)))

(defreduce r-sum [acc v] (+ acc v))
(defreduce r-prod [acc v] (* acc v))
(defreduce r-max [acc v] (if (or (Double/isNaN acc) (Double/isNaN v)) Double/NaN (Math/max acc v)))
(defreduce r-min [acc v] (if (or (Double/isNaN acc) (Double/isNaN v)) Double/NaN (Math/min acc v)))
(defreduce r-all [acc v] (if (and (not (== acc 0.0)) (not (== v 0.0))) 1.0 0.0))
(defreduce r-any [acc v] (if (or (not (== acc 0.0)) (not (== v 0.0))) 1.0 0.0))

(defn- arg-reduce
  "argmax (sign 1) or argmin (sign -1) over the middle axis of [outer len
   inner]; the first NaN wins, as in NumPy."
  ^doubles [^doubles src outer len inner sign]
  (let [outer (long outer) len (long len) inner (long inner) sign (long sign)
        out (double-array (* outer inner))]
    (when (zero? len)
      (throw (ex-info "argmax/argmin of an empty axis" {})))
    (dotimes [o outer]
      (dotimes [i inner]
        (let [base (+ (* o len inner) i)]
          (loop [k 1 bk 0 bv (aget src base)]
            (if (< k len)
              (let [v (aget src (+ base (* k inner)))]
                (if (and (not (Double/isNaN bv))
                         (or (Double/isNaN v) (if (pos? sign) (> v bv) (< v bv))))
                  (recur (inc k) k v)
                  (recur (inc k) bk bv)))
              (aset out (+ (* o inner) i) (double bk)))))))
    out))

(declare transpose)

(defn- reduce-layout
  "x's data viewed as [outer len inner] with the reduced axes as the middle
   run, transposing them to the end first when they are not contiguous.
   Returns [src outer len inner]."
  [x axes]
  (let [shape (shape-of x)
        size-of (fn [axs] (sh/size (mapv #(nth shape %) axs)))]
    (cond
      (empty? axes)
      [(data-of x) (sh/size shape) 1 1]

      (= axes (vec (range (first axes) (inc (long (peek axes))))))
      (let [a0 (long (first axes)) a1 (long (peek axes))]
        [(data-of x)
         (sh/size (subvec shape 0 a0))
         (sh/size (subvec shape a0 (inc a1)))
         (sh/size (subvec shape (inc a1)))])

      :else
      (let [kept (vec (remove (set axes) (range (count shape))))]
        [(data-of (transpose x (into kept axes))) (size-of kept) (size-of axes) 1]))))

(defn- reduce-op [op x axes keep-dims?]
  (let [shape (shape-of x)
        axes-set (set axes)
        out-shape (if keep-dims?
                    (vec (map-indexed (fn [i d] (if (axes-set i) 1 d)) shape))
                    (vec (keep-indexed (fn [i d] (when-not (axes-set i) d)) shape)))
        [src outer len inner] (reduce-layout x axes)
        out-dt (dt/reduce-result-dtype op (dtype-of x) (p/-defaults backend))]
    (make (case op
            :sum (r-sum src outer len inner 0.0)
            :prod (r-prod src outer len inner 1.0)
            :max (r-max src outer len inner Double/NEGATIVE_INFINITY)
            :min (r-min src outer len inner Double/POSITIVE_INFINITY)
            :all (r-all src outer len inner 1.0)
            :any (r-any src outer len inner 0.0)
            :argmax (arg-reduce src outer len inner 1)
            :argmin (arg-reduce src outer len inner -1))
          out-shape
          out-dt)))

(defn- scan-op [op x ^long axis reverse?]
  (let [shape (shape-of x)
        ^doubles src (data-of x)
        len (long (nth shape axis))
        inner (sh/size (subvec shape (inc axis)))
        outer (sh/size (subvec shape 0 axis))
        out (double-array (alength src))]
    (dotimes [o outer]
      (dotimes [j inner]
        (let [base (+ (* o len inner) j)]
          (loop [t 0 acc 0.0]
            (when (< t len)
              (let [k (if reverse? (- len 1 t) t)
                    pos (+ base (* k inner))
                    v (aget src pos)
                    acc (if (zero? t)
                          v
                          (case op
                            :cumsum (+ acc v)
                            :cumprod (* acc v)
                            :cummax (if (or (Double/isNaN acc) (Double/isNaN v)) Double/NaN (Math/max acc v))
                            :cummin (if (or (Double/isNaN acc) (Double/isNaN v)) Double/NaN (Math/min acc v))))]
                (aset out pos acc)
                (recur (inc t) acc)))))))
    (make out shape (dtype-of x))))

;; ---------------------------------------------------------------------------
;; Shape and indexing
;; ---------------------------------------------------------------------------

(defn- reshape [x shape]
  (let [shape (vec shape)]
    (cond (zero? (count shape)) (if (jvm-array? x) (box (aget (data-of x) 0) (dtype-of x)) x)
          (jvm-array? x) (JvmArray. (.-data ^JvmArray x) shape (.-dtype ^JvmArray x))
          :else (JvmArray. (data-of x) shape (dtype-of x)))))

(defn- transpose [x perm]
  (let [shape (shape-of x)
        ist (sh/strides shape)
        out-shape (mapv #(nth shape %) perm)
        st (long-array (map #(aget ist (long %)) perm))]
    (make (gather (data-of x) (sh/strided-offsets out-shape st 0)) out-shape (dtype-of x))))

(defn- broadcast-to [x shape]
  (let [shape (vec shape)]
    (when-not (= shape (sh/broadcast (shape-of x) shape))
      (throw (ex-info (str "Cannot broadcast " (shape-of x) " to " shape)
                      {:from (shape-of x) :to shape})))
    (make (gather (data-of x) (sh/broadcast-offsets (shape-of x) shape)) shape (dtype-of x))))

(defn- concatenate [xs ^long axis]
  (let [shapes (map shape-of xs)
        s0 (first shapes)
        _ (doseq [s shapes]
            (when-not (and (= (count s) (count s0))
                           (= (assoc s axis 0) (assoc s0 axis 0)))
              (throw (ex-info "Arrays must agree on every axis but the concatenation axis"
                              {:shapes (vec shapes) :axis axis}))))
        out-shape (assoc s0 axis (reduce + (map #(long (nth % axis)) shapes)))
        outer (sh/size (subvec s0 0 axis))
        out (double-array (sh/size out-shape))]
    (loop [o 0 pos 0]
      (when (< o outer)
        (recur (inc o)
               (long (reduce (fn [^long pos x]
                               (let [^doubles d (data-of x)
                                     chunk (quot (alength d) outer)]
                                 (System/arraycopy d (* o chunk) out pos chunk)
                                 (+ pos chunk)))
                             pos xs)))))
    (make out out-shape (dtype-of (first xs)))))

(defn- slice [x starts stops steps]
  (let [shape (shape-of x)
        ist (sh/strides shape)
        out-shape (mapv (fn [s e st]
                          (let [s (long s) e (long e) st (long st)]
                            (max 0 (quot (+ (- e s) (- st (Long/signum st))) st))))
                        starts stops steps)
        base (reduce + (map-indexed (fn [k s] (* (long s) (aget ist k))) starts))
        st (long-array (map-indexed (fn [k step] (* (long step) (aget ist k))) steps))]
    (make (gather (data-of x) (sh/strided-offsets out-shape st base)) out-shape (dtype-of x))))

(defn- index-value
  "Normalize index v (a stored double) against a dimension of size d."
  ^long [^double v ^long d]
  (let [i (long v)
        i (if (neg? i) (+ i d) i)]
    (if (and (>= i 0) (< i d))
      i
      (throw (ex-info (str "Index " (long v) " is out of bounds for a dimension of size " d)
                      {:index (long v) :size d})))))

(defn- take-op [x indices ^long axis]
  (let [shape (shape-of x)
        ^doubles src (data-of x)
        ^doubles ix (data-of indices)
        d (long (nth shape axis))
        inner (sh/size (subvec shape (inc axis)))
        outer (sh/size (subvec shape 0 axis))
        m (alength ix)
        out-shape (vec (concat (subvec shape 0 axis) (shape-of indices) (subvec shape (inc axis))))
        out (double-array (* outer m inner))]
    (dotimes [o outer]
      (dotimes [j m]
        (let [i (index-value (aget ix j) d)]
          (System/arraycopy src (* (+ (* o d) i) inner) out (* (+ (* o m) j) inner) inner))))
    (make out out-shape (dtype-of x))))

(defn- take-along-axis [x indices ^long axis]
  (let [sx (shape-of x)
        si (shape-of indices)
        _ (when-not (= (count sx) (count si))
            (throw (ex-info "take-along-axis needs indices of the same rank as the array"
                            {:shape sx :indices-shape si})))
        out-shape (mapv (fn [k a b] (if (= k axis) b (first (sh/broadcast [a] [b]))))
                        (range) sx si)
        d (long (nth sx axis))
        ist (sh/strides sx)
        axis-stride (aget ist axis)
        ;; x is read along every axis but `axis` (broadcast where its dim is 1);
        ;; along `axis` the index array picks the position
        x-st (long-array (map-indexed (fn [k dim]
                                        (if (or (= k axis) (== 1 (long dim))) 0 (aget ist k)))
                                      sx))
        ^longs xo (sh/strided-offsets out-shape x-st 0)
        io (sh/broadcast-offsets si out-shape)
        ^doubles src (data-of x)
        ^doubles ix (data-of indices)
        n (alength xo)
        out (double-array n)]
    (dotimes [i n]
      (let [k (index-value (aget ix (aget io i)) d)]
        (aset out i (aget src (+ (aget xo i) (* k axis-stride))))))
    (make out out-shape (dtype-of x))))

(defn- scatter-add [x indices updates ^long axis]
  (let [sx (shape-of x)
        su (shape-of updates)
        _ (when-not (and (= su (shape-of indices)) (= (count sx) (count su))
                         (every? true? (map (fn [k a b] (or (= k axis) (= a b))) (range) sx su)))
            (throw (ex-info "scatter-add needs indices and updates of one shape, equal to the array's but on axis"
                            {:shape sx :indices-shape (shape-of indices) :updates-shape su :axis axis})))
        d (long (nth sx axis))
        st (sh/strides sx)
        axis-stride (aget st axis)
        ^longs xo (sh/strided-offsets su (long-array (map-indexed (fn [k s] (if (= k axis) 0 s)) st)) 0)
        ^doubles src (data-of updates)
        ^doubles ix (data-of indices)
        out (aclone (data-of x))]
    (dotimes [i (alength xo)]
      (let [o (+ (aget xo i) (* (index-value (aget ix i) d) axis-stride))]
        (aset out o (+ (aget out o) (aget src i)))))
    (make out sx (dtype-of x))))

(defn- matmul [x y]
  (let [sx (shape-of x) sy (shape-of y)
        _ (when (or (zero? (count sx)) (zero? (count sy)))
            (throw (ex-info "matmul needs arrays of rank >= 1" {:shapes [sx sy]})))
        x-vec? (= 1 (count sx)) y-vec? (= 1 (count sy))
        sx (if x-vec? [1 (first sx)] sx)
        sy (if y-vec? [(first sy) 1] sy)
        n (long (nth sx (- (count sx) 2))) k (long (peek sx))
        k2 (long (nth sy (- (count sy) 2))) m (long (peek sy))
        _ (when-not (== k k2)
            (throw (ex-info (str "matmul: inner dimensions differ (" k " vs " k2 ")")
                            {:shapes [(shape-of x) (shape-of y)]})))
        bx (subvec sx 0 (- (count sx) 2))
        by (subvec sy 0 (- (count sy) 2))
        batch (sh/broadcast bx by)
        xo (sh/broadcast-offsets bx batch)
        yo (sh/broadcast-offsets by batch)
        nb (sh/size batch)
        ^doubles a (data-of x) ^doubles b (data-of y)
        out (double-array (* nb n m))]
    (dotimes [q nb]
      (let [abase (* (aget xo q) n k)
            bbase (* (aget yo q) k m)
            obase (* q n m)]
        (dotimes [i n]
          (dotimes [p k]
            (let [aip (aget a (+ abase (* i k) p))
                  brow (+ bbase (* p m))
                  orow (+ obase (* i m))]
              (dotimes [j m]
                (aset out (+ orow j) (+ (aget out (+ orow j)) (* aip (aget b (+ brow j)))))))))))
    (make out
          (vec (concat batch (when-not x-vec? [n]) (when-not y-vec? [m])))
          (dtype-of x))))

;; ---------------------------------------------------------------------------
;; Linear algebra: per-matrix loops over the batch axes
;; ---------------------------------------------------------------------------

(defmacro ^:private fold-range
  "Fold expr over k in [start, end) into the double acc (initially init),
   returning a primitive double. The loop runs in statement position: Clojure
   compiles a loop used as a value into a closure, which would box."
  [[k start end] [acc init] expr]
  `(let [cell# (double-array 1)]
     (aset cell# 0 (double ~init))
     (loop [~k (long ~start)]
       (when (< ~k (long ~end))
         (let [~acc (aget cell# 0)] (aset cell# 0 (double ~expr)))
         (recur (inc ~k))))
     (aget cell# 0)))

(defn- check-square ^long [shape op]
  (let [r (count shape)]
    (when (or (< r 2) (not= (nth shape (- r 1)) (nth shape (- r 2))))
      (throw (ex-info (str (name op) " needs square matrices, got shape " shape)
                      {:op op :shape shape})))
    (long (peek shape))))

(defn- cholesky-into!
  "Lower Cholesky factor of the n×n matrix at offset off of a into out (NaN
   everywhere for a matrix that is not positive definite)."
  [^doubles a ^doubles out ^long off ^long n]
  (loop [j 0]
    (when (< j n)
      (let [rj (+ off (* j n))
            s (fold-range [k 0 j] [s (aget a (+ rj j))] (- s (* (aget out (+ rj k)) (aget out (+ rj k)))))]
        (if (not (> s 0.0))
          (java.util.Arrays/fill out (int off) (int (+ off (* n n))) Double/NaN)
          (let [d (Math/sqrt s)]
            (aset out (+ rj j) d)
            (loop [i (inc j)]
              (when (< i n)
                (let [ri (+ off (* i n))]
                  (aset out (+ ri j)
                        (/ (fold-range [k 0 j] [s (aget a (+ ri j))]
                             (- s (* (aget out (+ ri k)) (aget out (+ rj k)))))
                           d)))
                (recur (inc i))))
            (recur (inc j))))))))

(defn- cholesky [x]
  (let [shape (shape-of x)
        n (check-square shape :cholesky)
        ^doubles a (data-of x)
        out (double-array (alength a))]
    (dotimes [q (quot (alength a) (max 1 (* n n)))]
      (cholesky-into! a out (* q n n) n))
    (make out shape (dtype-of x))))

(defn- batch-pairs
  "Matrix batch layout for a [.. n n] against b [.. n k], broadcasting the
   batch axes: [batch-shape a-offsets b-offsets] (offsets in matrices)."
  [sa sb]
  (let [ba (subvec sa 0 (- (count sa) 2))
        bb (subvec sb 0 (- (count sb) 2))
        batch (sh/broadcast ba bb)]
    [batch (sh/broadcast-offsets ba batch) (sh/broadcast-offsets bb batch)]))

(defn- solve-triangular [a b lower? transpose?]
  (let [sa (shape-of a) sb (shape-of b)
        n (check-square sa :solve-triangular)
        k (long (peek sb))
        _ (when-not (== n (long (nth sb (- (count sb) 2))))
            (throw (ex-info "solve-triangular: matrix and right-hand side do not match"
                            {:shapes [sa sb]})))
        [batch ao bo] (batch-pairs sa sb)
        ^longs ao ao ^longs bo bo
        ^doubles ad (data-of a) ^doubles bd (data-of b)
        nb (sh/size batch)
        out (double-array (* nb n k))
        tr? (boolean transpose?)
        ;; solving L x = b (or Uᵀ x = b) runs forward; U x = b (or Lᵀ x = b) backward
        forward? (not= (boolean lower?) tr?)]
    (dotimes [q nb]
      (let [aoff (* (aget ao q) n n) boff (* (aget bo q) n k) ooff (* q n k)]
        (dotimes [c k]
          (dotimes [t n]
            (let [i (if forward? t (- n 1 t))
                  s (fold-range [j (if forward? 0 (inc i)) (if forward? i n)]
                                [s (aget bd (+ boff (* i k) c))]
                      (- s (* (aget ad (+ aoff (if tr? (+ (* j n) i) (+ (* i n) j))))
                              (aget out (+ ooff (* j k) c)))))]
              (aset out (+ ooff (* i k) c) (/ s (aget ad (+ aoff (* i n) i)))))))))
    (make out (into batch [n k]) (dtype-of a))))

(defmacro ^:private at [i j] `(+ ~'off (* ~i ~'n) ~j))

(defn- lu-into!
  "LU-decompose (with partial pivoting) the n×n matrix at offset off of m, in
   place. Fills perm with the row permutation; returns the permutation's sign."
  ^double [^doubles m ^long off ^long n ^longs perm]
  (do
    (dotimes [i n] (aset perm i i))
    (loop [col 0 sign 1.0]
      (if (== col n)
        sign
        (let [p (long (fold-range [r (inc col) n] [best col]
                        (if (> (Math/abs (aget m (at r col))) (Math/abs (aget m (at (long best) col)))) r best)))
              sign (if (== p col)
                     sign
                     (do (dotimes [j n]
                           (let [t (aget m (at col j))]
                             (aset m (at col j) (aget m (at p j)))
                             (aset m (at p j) t)))
                         (let [t (aget perm col)] (aset perm col (aget perm p)) (aset perm p t))
                         (- sign)))
              pivot (aget m (at col col))]
          (when-not (== pivot 0.0)
            (loop [r (inc col)]
              (when (< r n)
                (let [f (/ (aget m (at r col)) pivot)]
                  (aset m (at r col) f)
                  (loop [j (inc col)]
                    (when (< j n)
                      (aset m (at r j) (- (aget m (at r j)) (* f (aget m (at col j)))))
                      (recur (inc j)))))
                (recur (inc r)))))
          (recur (inc col) sign))))))

(defn- solve [a b]
  (let [sa (shape-of a) sb (shape-of b)
        n (check-square sa :solve)
        k (long (peek sb))
        [batch ao bo] (batch-pairs sa sb)
        ^longs ao ao ^longs bo bo
        ^doubles ad (data-of a) ^doubles bd (data-of b)
        nb (sh/size batch)
        out (double-array (* nb n k))
        lu (double-array (* n n))
        perm (long-array n)]
    (dotimes [q nb]
      (System/arraycopy ad (* (aget ao q) n n) lu 0 (* n n))
      (lu-into! lu 0 n perm)
      (let [boff (* (aget bo q) n k) ooff (* q n k)]
        (dotimes [c k]
          ;; forward substitution with the unit lower factor, on the permuted rhs
          (dotimes [i n]
            (aset out (+ ooff (* i k) c)
                  (fold-range [j 0 i] [s (aget bd (+ boff (* (aget perm i) k) c))]
                    (- s (* (aget lu (+ (* i n) j)) (aget out (+ ooff (* j k) c)))))))
          ;; back substitution with the upper factor
          (dotimes [t n]
            (let [i (- n 1 t)
                  s (fold-range [j (inc i) n] [s (aget out (+ ooff (* i k) c))]
                      (- s (* (aget lu (+ (* i n) j)) (aget out (+ ooff (* j k) c)))))]
              (aset out (+ ooff (* i k) c) (/ s (aget lu (+ (* i n) i)))))))))
    (make out (into batch [n k]) (dtype-of a))))

(defn- slogdet [x]
  (let [shape (shape-of x)
        n (check-square shape :slogdet)
        ^doubles a (data-of x)
        nb (sh/size (subvec shape 0 (- (count shape) 2)))
        signs (double-array nb)
        logs (double-array nb)
        lu (double-array (* n n))
        perm (long-array n)]
    (dotimes [q nb]
      (System/arraycopy a (* q n n) lu 0 (* n n))
      (loop [i 0 sign (lu-into! lu 0 n perm) acc 0.0]
        (if (< i n)
          (let [d (aget lu (+ (* i n) i))]
            (recur (inc i) (* sign (Math/signum d)) (+ acc (Math/log (Math/abs d)))))
          (let [singular? (== sign 0.0)]
            (aset signs q (if singular? 0.0 sign))
            (aset logs q (if singular? Double/NEGATIVE_INFINITY acc))))))
    (let [batch (subvec shape 0 (- (count shape) 2))]
      [(make signs batch (dtype-of x)) (make logs batch (dtype-of x))])))

(defn- linalg [op args]
  (case op
    :cholesky (cholesky (nth args 0))
    :solve-triangular (let [[a b {:keys [lower? transpose?]}] args]
                        (solve-triangular a b lower? transpose?))
    :solve (solve (nth args 0) (nth args 1))
    :slogdet (slogdet (nth args 0))))

(defn- random-uniform [key shape dtype]
  (let [n (sh/size shape)
        data (case dtype
               :float64 (threefry/uniform-doubles key n)
               :float32 (threefry/uniform-floats key n)
               (throw (ex-info (str "random-uniform needs a float dtype, got " dtype) {:dtype dtype})))]
    (make data shape dtype)))

;; ---------------------------------------------------------------------------
;; Host boundary
;; ---------------------------------------------------------------------------

(defn- host-shape [data]
  (loop [shape [] d data]
    (if (sequential? d)
      (recur (conj shape (count d)) (first d))
      shape)))

(defn- host-leaves
  "The leaves of nested data in row-major order, checking that every level is
   rectangular and every leaf is a number or boolean."
  [data shape]
  (let [rank (count shape)
        walk (fn walk [d ^long ax]
               (if (== ax rank)
                 (if (or (instance? Number d) (instance? Boolean d))
                   [d]
                   (throw (ex-info (str "Array element is not a number or boolean: " (pr-str d))
                                   {:element d})))
                 (if (and (sequential? d) (== (count d) (long (nth shape ax))))
                   (mapcat #(walk % (inc ax)) d)
                   (throw (ex-info "Nested data is not rectangular" {:shape shape :at-axis ax})))))]
    (walk data 0)))

(defn- infer-dtype [leaves]
  (cond (empty? leaves) :float64
        (some #(or (instance? Double %) (instance? Float %) (ratio? %) (decimal? %)) leaves) :float64
        (every? #(instance? Boolean %) leaves) :bool
        :else :int64))

(defn- from-host [data dtype]
  (cond
    (or (instance? Number data) (instance? Boolean data))
    (dt/host-cast data (check-dtype (or dtype (dt/host-dtype data))))

    (instance? (Class/forName "[D") data)
    (let [^doubles d data dtype (check-dtype (or dtype :float64))]
      (make (aclone d) [(alength d)] dtype))

    (sequential? data)
    (let [shape (host-shape data)
          leaves (host-leaves data shape)
          dtype (check-dtype (or dtype (infer-dtype leaves)))
          out (double-array (count leaves))]
      (loop [i 0 ls (seq leaves)]
        (when ls
          (aset out i (dt/host-double (dt/host-cast (first ls) dtype)))
          (recur (inc i) (next ls))))
      (make out shape dtype))

    :else
    (throw (ex-info (str "Cannot make an array from " (type data)) {:data data}))))

(defn to-host
  "Nested vectors of host values (a host scalar for 0-d)."
  [x]
  (if-not (jvm-array? x)
    x
    (let [^JvmArray x x
          ^doubles d (.-data x)
          dtype (.-dtype x)
          shape (.-shape x)
          st (sh/strides shape)
          build (fn build [^long ax ^long off]
                  (if (= ax (count shape))
                    (box (aget d off) dtype)
                    (mapv #(build (inc ax) (+ off (* (long %) (aget st ax))))
                          (range (nth shape ax)))))]
      (build 0 0))))

(defn- astype [x dtype]
  (check-dtype dtype)
  (cond (not (jvm-array? x)) (dt/host-cast x dtype)
        (= dtype (dtype-of x)) x
        :else
        (let [^doubles src (data-of x)
              n (alength src)
              out (double-array n)]
          (case (dt/kind dtype)
            :float (System/arraycopy src 0 out 0 n)
            :int (dotimes [i n] (aset out i (double (long (aget src i)))))
            :bool (dotimes [i n] (aset out i (if (== (aget src i) 0.0) 0.0 1.0))))
          (make out (shape-of x) dtype))))

;; ---------------------------------------------------------------------------
;; The backend
;; ---------------------------------------------------------------------------

(deftype JvmBackend []
  p/IBackend
  (-name [_] :jvm)
  (-defaults [_] {:float :float64 :int :int64})
  (-lazy? [_] false)
  (-native-ops [_] #{})

  (-from-host [_ data dtype] (from-host data dtype))
  (-to-host [_ x] (to-host x))
  (-eval! [_ _] nil)

  (-shape [_ x] (shape-of x))
  (-dtype [_ x] (dtype-of x))

  (-full [_ shape value dtype]
    (check-dtype dtype)
    (let [v (dt/host-double (dt/host-cast value dtype))]
      (make (doto (double-array (sh/size shape)) (java.util.Arrays/fill v)) shape dtype)))
  (-arange [_ start stop step dtype]
    (check-dtype dtype)
    (let [start (double start) stop (double stop) step (double step)
          n (max 0 (long (Math/ceil (/ (- stop start) step))))
          out (double-array n)]
      (dotimes [i n] (aset out i (+ start (* i step))))
      (make out [n] dtype)))
  (-astype [_ x dtype] (astype x dtype))

  (-elementwise [_ op args dtype] (elementwise op args dtype))
  (-reduce [_ op x axes keep-dims?] (reduce-op op x axes keep-dims?))
  (-scan [_ op x axis reverse?] (scan-op op x axis reverse?))

  (-reshape [_ x shape] (reshape x shape))
  (-transpose [_ x perm] (transpose x perm))
  (-broadcast-to [_ x shape] (broadcast-to x shape))
  (-concatenate [_ xs axis] (concatenate xs axis))
  (-slice [_ x starts stops steps] (slice x starts stops steps))

  (-take [_ x indices axis] (take-op x indices axis))
  (-take-along-axis [_ x indices axis] (take-along-axis x indices axis))
  (-scatter-add [_ x indices updates axis] (scatter-add x indices updates axis))

  (-matmul [_ x y] (matmul x y))
  (-linalg [_ op args] (linalg op args))
  (-random-uniform [_ key shape dtype] (random-uniform key shape dtype))
  ;; reverse mode by the backend-generic tape (loaded on first use, since it
  ;; is written against gen.array, which requires this namespace)
  (-value-and-grad [b f args argnums] ((requiring-resolve 'gen.array.tape/value-and-grad) b f args argnums)))

(def backend (JvmBackend.))

(defmethod print-method JvmArray [^JvmArray x ^java.io.Writer w]
  (.write w "#gen/array[")
  (print-method (.-dtype x) w)
  (.write w " ")
  (print-method (to-host x) w)
  (.write w "]"))
