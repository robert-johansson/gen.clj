(ns gen.backend.mlx
  "The MLX backend (M2): arrays are MLX arrays, computed lazily by MLX on the
   CPU or the GPU (Metal; CUDA on Linux), through mlx-c over FFM. The Java
   binding (src-java/gen/mlx, generated from mlx-c's headers) is compiled with
   `clojure -T:build java`; the native libraries come from native/build-mlx.sh.

   - Arrays are MlxArray values owning an MLX handle (a long). A Cleaner frees
     the handle when the value becomes unreachable; MLX's reference counting
     keeps any graph that still needs it alive. Freeing from the Cleaner's
     thread is safe: it only drops a reference (atomically), and MLX's
     allocators lock.
   - 0-d arrays are MLX arrays too (the JVM backend's are host values), so
     scalar code stays lazy. Reading one on the host (item, truthy?) evaluates.
   - Ops run on the backend's stream, a stream MLX lets any thread use. One
     lock serializes evaluation, so the backend can be used from several
     threads.
   - Decompositions and solves run on the CPU stream (MLX implements them only
     there).
   - Keys are gen.random's (JAX's partitionable Threefry), computed with MLX's
     uint32 ops, so a key draws the same uniforms as on the JVM backend.
   - Gradients use the backend-generic tape (gen.array.tape).

   Configuration (system properties, read when the namespace loads):
     gen.mlx.device  cpu or gpu (default gpu when MLX has one)
     gen.mlx.dtypes  32 (default) or 64: the default dtypes, float32/int32 or
                     float64/int64 (64 needs the CPU device; MLX's GPUs have no
                     float64)"
  (:require [gen.array.dtype :as dt]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.random.threefry])
  (:import [gen.mlx Mlx MlxRaw MlxRuntime MlxException]
           [gen.random.threefry Key]
           [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.lang.ref Cleaner Reference]))

(set! *warn-on-reflection* true)

(declare backend config)

;; ---------------------------------------------------------------------------
;; Dtypes
;; ---------------------------------------------------------------------------

(def ^:private dtype-codes
  {:bool 0 :uint32 3 :int32 7 :int64 8 :float16 9 :float32 10 :float64 11 :bfloat16 12})

(def ^:private code-dtypes (into {} (map (fn [[k v]] [v k])) dtype-codes))

(defn- code ^long [dtype]
  (or (dtype-codes dtype)
      (throw (ex-info (str "The MLX backend does not support dtype " dtype)
                      {:dtype dtype :supported (set (keys dtype-codes))}))))

(defn- dtype-of-code [^long c]
  (or (code-dtypes c)
      (throw (ex-info (str "MLX produced a dtype gen.clj does not use (code " c ")") {:code c}))))

;; ---------------------------------------------------------------------------
;; Arrays
;; ---------------------------------------------------------------------------

(defonce ^:private ^Cleaner cleaner (Cleaner/create))

(declare to-host)

(deftype MlxArray [^long h backend shape dtype]
  p/IArray
  (-backend [_] backend)

  ;; values: equal when dtype, shape and elements are (as the JVM backend's);
  ;; comparing reads both arrays on the host
  Object
  (equals [this o]
    (or (identical? this o)
        (and (instance? MlxArray o)
             (= dtype (.-dtype ^MlxArray o))
             (= shape (.-shape ^MlxArray o))
             (= (to-host this) (to-host o)))))
  (hashCode [this] (hash [shape (to-host this)]))

  clojure.lang.IHashEq
  (hasheq [this] (.hashCode this)))

(defn- mlx-array? [x] (instance? MlxArray x))

(defn- read-shape [^long h]
  (let [n (Mlx/mlx_array_ndim h)]
    (if (zero? n)
      []
      (let [^MemorySegment seg (.reinterpret ^MemorySegment (Mlx/mlx_array_shape h) (* 4 n))]
        (mapv long (.toArray seg ValueLayout/JAVA_INT))))))

(defn- wrap
  "The MlxArray owning the new handle h (freed when it becomes unreachable)."
  [b ^long h]
  (let [x (MlxArray. h b (read-shape h) (dtype-of-code (Mlx/mlx_array_dtype h)))]
    (.register cleaner x (reify Runnable (run [_] (MlxRaw/mlx_array_free h))))
    x))

(defn- fence
  "Keep x reachable (its handle valid) up to this point."
  [x]
  (Reference/reachabilityFence x))

;; ---------------------------------------------------------------------------
;; The backend's state: streams and the evaluation lock
;; ---------------------------------------------------------------------------

(defrecord Config [device ^long stream ^long cpu-stream defaults])

(defn- stream ^long [b] (.-stream ^Config (config b)))
(defn- cpu-stream ^long [b] (.-cpu-stream ^Config (config b)))
(defn- defaults [b] (.-defaults ^Config (config b)))

(defonce ^:private aux-stream
  ;; for conversions made where no backend is at hand
  (Mlx/mlx_stream_new_thread_unsafe (Mlx/mlx_device_new_type (int 0) (int 0))))

(def ^:private eval-lock (Object.))

(defmacro ^:private evaluating
  "Run body (which evaluates MLX arrays) holding the evaluation lock."
  [& body]
  `(locking eval-lock ~@body))

;; ---------------------------------------------------------------------------
;; Operands: MLX arrays and host scalars
;; ---------------------------------------------------------------------------

(defn- host? [x] (or (instance? Number x) (instance? Boolean x)))

(defn- half-scalar-handle
  "A new handle for the host scalar x as :float16 or :bfloat16."
  [x dtype]
  (let [f (MlxRaw/mlx_array_new_float32 (unchecked-float (dt/host-double x)))]
    (try (Mlx/mlx_astype f (int (code dtype)) (long aux-stream))
         (finally (MlxRaw/mlx_array_free f)))))

(defn- scalar-handle
  "A new handle for the host scalar x as dtype."
  ^long [x dtype]
  (case dtype
    :float32 (MlxRaw/mlx_array_new_float32 (unchecked-float (dt/host-double x)))
    :float64 (MlxRaw/mlx_array_new_float64 (dt/host-double x))
    :int32 (MlxRaw/mlx_array_new_int (unchecked-int (dt/host-long x)))
    :bool (MlxRaw/mlx_array_new_bool (dt/truthy? x))
    :int64 (MlxRaw/mlx_array_new_data (.longs (MlxRuntime/scratch) (long-array [(dt/host-long x)]))
                                      MemorySegment/NULL (int 0) (int (code :int64)))
    :uint32 (MlxRaw/mlx_array_new_data (.ints (MlxRuntime/scratch) (int-array [(unchecked-int (dt/host-long x))]))
                                       MemorySegment/NULL (int 0) (int (code :uint32)))
    (:float16 :bfloat16) (long (half-scalar-handle x dtype))))

(defn- host-dtype-on
  "The dtype a host scalar has on b where nothing else gives one: the
   backend's default for its kind (host numbers are weakly typed)."
  [b x]
  (dt/with-weak nil #{(dt/kind (dt/host-dtype x))} (defaults b)))

(defn- operand
  "The handle of operand x: x's own, or a new one for a host scalar (as
   dtype, or b's default for it)."
  ^long [b x dtype]
  (if (mlx-array? x)
    (.-h ^MlxArray x)
    (scalar-handle x (or dtype (host-dtype-on b x)))))

(defn- release
  "Free the handle h of operand x if operand made it for a host scalar."
  [x ^long h]
  (when-not (mlx-array? x) (MlxRaw/mlx_array_free h)))

(defn- shape-of [x] (if (mlx-array? x) (.-shape ^MlxArray x) []))

(defn- dtype-on [b x] (if (mlx-array? x) (.-dtype ^MlxArray x) (host-dtype-on b x)))

(defn- mlx-error
  "ex-info for an error MLX reported in op on args."
  [op args ^MlxException e]
  (ex-info (str (name op) ": " (.getMessage e))
           {:op op :shapes (mapv shape-of args) :mlx-error (.getMessage e)}
           e))

(defmacro ^:private with-operands
  "Bind each handle symbol to the handle of its operand (b, x, dtype), run
   body, then release the handles made for host scalars and keep the operands
   reachable. MLX errors become ex-info naming op."
  [b op bindings & body]
  (let [triples (partition 3 bindings)
        hs (map first triples)
        xs (map second triples)]
    `(let [~@(mapcat (fn [[h x d]] [h `(operand ~b ~x ~d)]) triples)]
       (try
         ~@body
         (catch MlxException e# (throw (mlx-error ~op [~@xs] e#)))
         (finally
           ~@(map (fn [h x] `(release ~x ~h)) hs xs)
           ~@(map (fn [x] `(fence ~x)) xs))))))

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
                 (if (host? d)
                   [d]
                   (throw (ex-info (str "Array element is not a number or boolean: " (pr-str d))
                                   {:element d})))
                 (if (and (sequential? d) (== (count d) (long (nth shape ax))))
                   (mapcat #(walk % (inc ax)) d)
                   (throw (ex-info "Nested data is not rectangular" {:shape shape :at-axis ax})))))]
    (walk data 0)))

(defn- infer-dtype [b leaves]
  (cond (empty? leaves) (:float (defaults b))
        (some #(or (instance? Double %) (instance? Float %) (ratio? %) (decimal? %)) leaves) (:float (defaults b))
        (every? #(instance? Boolean %) leaves) :bool
        :else (:int (defaults b))))

(defn- segment-of
  "Native memory (in arena) holding values (host scalars) as dtype."
  ^MemorySegment [^Arena arena values dtype]
  (let [n (count values)
        vs (map #(dt/host-cast % dtype) values)]
    (case dtype
      :float32 (.allocateFrom arena ValueLayout/JAVA_FLOAT (float-array (map double vs)))
      :float64 (.allocateFrom arena ValueLayout/JAVA_DOUBLE (double-array (map double vs)))
      (:int32 :uint32) (.allocateFrom arena ValueLayout/JAVA_INT (int-array (map #(unchecked-int (long %)) vs)))
      :int64 (.allocateFrom arena ValueLayout/JAVA_LONG (long-array (map long vs)))
      :bool (.allocateFrom arena ValueLayout/JAVA_BYTE (byte-array (map #(if % 1 0) vs)))
      (throw (ex-info (str "Cannot make an MLX array of dtype " dtype " from host data") {:dtype dtype :count n})))))

(defn- from-values
  "An array of shape and dtype from host values in row-major order."
  [b values shape dtype]
  (if (#{:float16 :bfloat16} dtype)
    (let [x (from-values b values shape :float32)]
      (wrap b (Mlx/mlx_astype (.-h ^MlxArray x) (code dtype) (stream b))))
    (with-open [arena (Arena/ofConfined)]
      (let [data (segment-of arena values dtype)
            shp (.allocateFrom arena ValueLayout/JAVA_INT (int-array (map int shape)))]
        (wrap b (MlxRaw/mlx_array_new_data data shp (int (count shape)) (int (code dtype))))))))

(defn- from-host [b data dtype]
  (cond
    (host? data)
    (let [d (or dtype (host-dtype-on b data))]
      (code d)
      (wrap b (scalar-handle data d)))

    (instance? (Class/forName "[D") data)
    (let [^doubles d data]
      (from-values b (vec d) [(alength d)] (or dtype (:float (defaults b)))))

    (sequential? data)
    (let [shape (host-shape data)
          leaves (host-leaves data shape)
          dtype (or dtype (infer-dtype b leaves))]
      (code dtype)
      (from-values b leaves shape dtype))

    :else
    (throw (ex-info (str "Cannot make an array from " (type data)) {:data data}))))

(defn- data-segment
  "The data pointer p of an evaluated array, as a segment of bytes bytes."
  ^MemorySegment [^MemorySegment p bytes]
  (when (= MemorySegment/NULL p)
    (throw (MlxRuntime/error "mlx_array_data")))
  (.reinterpret p (long bytes)))

(defn- read-values
  "The elements of the evaluated, row-contiguous handle h (n of dtype) as a
   vector of host values."
  [^long h ^long n dtype]
  (if (zero? n)
    []
    (case dtype
      :float32 (mapv double (.toArray (data-segment (Mlx/mlx_array_data_float32 h) (* 4 n)) ValueLayout/JAVA_FLOAT))
      :float64 (vec (.toArray (data-segment (Mlx/mlx_array_data_float64 h) (* 8 n)) ValueLayout/JAVA_DOUBLE))
      :int32 (mapv long (.toArray (data-segment (Mlx/mlx_array_data_int32 h) (* 4 n)) ValueLayout/JAVA_INT))
      :uint32 (mapv #(bit-and (long %) 0xFFFFFFFF) (.toArray (data-segment (Mlx/mlx_array_data_uint32 h) (* 4 n)) ValueLayout/JAVA_INT))
      :int64 (vec (.toArray (data-segment (Mlx/mlx_array_data_int64 h) (* 8 n)) ValueLayout/JAVA_LONG))
      :bool (mapv #(not (zero? (long %))) (.toArray (data-segment (Mlx/mlx_array_data_bool h) n) ValueLayout/JAVA_BYTE)))))

(defn- nest
  "Nested vectors of shape from the flat vector vs."
  [vs shape]
  (if (<= (count shape) 1)
    vs
    (let [inner (vec (rest shape))
          step (sh/size inner)]
      (mapv #(nest (subvec vs (* (long %) step) (* (inc (long %)) step)) inner)
            (range (first shape))))))

(defn to-host
  "Nested vectors of host values (a host scalar for 0-d). Evaluates x."
  [x]
  (if-not (mlx-array? x)
    x
    (let [^MlxArray x x
          b (.-backend x)
          shape (.-shape x)
          dtype (.-dtype x)
          read-as (if (#{:float16 :bfloat16} dtype) :float32 dtype)
          n (sh/size shape)
          s (stream b)
          temps (java.util.ArrayList.)]
      (try
        (let [h (.-h x)
              h (if (= read-as dtype) h (let [c (Mlx/mlx_astype h (int (code read-as)) s)] (.add temps c) c))
              ;; data pointers address the buffer as laid out: copy views first
              c (Mlx/mlx_contiguous (long h) false s)
              _ (.add temps c)
              vs (evaluating
                   (Mlx/mlx_array_eval c)
                   (read-values c n read-as))]
          (if (empty? shape) (first vs) (nest vs shape)))
        (catch MlxException e (throw (mlx-error :->clj [x] e)))
        (finally
          (doseq [t temps] (MlxRaw/mlx_array_free (long t)))
          (fence x))))))

(defn- eval-arrays! [xs]
  (let [hs (long-array (map #(.-h ^MlxArray %) xs))
        v (MlxRuntime/vectorOf hs)]
    (try
      (evaluating (Mlx/mlx_eval v))
      (catch MlxException e (throw (mlx-error :eval! xs e)))
      (finally
        (Mlx/mlx_vector_array_free v)
        (fence xs)))
    nil))

;; ---------------------------------------------------------------------------
;; Elementwise ops
;; ---------------------------------------------------------------------------

(defmacro ^:private mlx-fn
  "A fn of handles (and the stream) calling the mlx-c function f."
  [f n]
  (let [args (map #(with-meta % {:tag 'long}) (repeatedly n #(gensym "h")))
        s (with-meta (gensym "s") {:tag 'long})]
    `(fn ~(vec (concat args [s])) (~(symbol "Mlx" (str "mlx_" f)) ~@args ~s))))

(def ^:private unary-fns
  {:negative (mlx-fn negative 1) :abs (mlx-fn abs 1) :sign (mlx-fn sign 1) :sqrt (mlx-fn sqrt 1)
   :exp (mlx-fn exp 1) :log (mlx-fn log 1) :expm1 (mlx-fn expm1 1) :log1p (mlx-fn log1p 1)
   :sin (mlx-fn sin 1) :cos (mlx-fn cos 1) :tan (mlx-fn tan 1) :tanh (mlx-fn tanh 1)
   :erf (mlx-fn erf 1) :erfinv (mlx-fn erfinv 1)
   :floor (mlx-fn floor 1) :ceil (mlx-fn ceil 1) :round (fn [^long h ^long s] (Mlx/mlx_round h (int 0) s))
   :isnan (mlx-fn isnan 1) :isinf (mlx-fn isinf 1) :logical-not (mlx-fn logical_not 1)
   :stop-gradient (mlx-fn stop_gradient 1)})

(def ^:private binary-fns
  {:add (mlx-fn add 2) :subtract (mlx-fn subtract 2) :multiply (mlx-fn multiply 2)
   :divide (mlx-fn divide 2) :power (mlx-fn power 2) :maximum (mlx-fn maximum 2)
   :minimum (mlx-fn minimum 2) :remainder (mlx-fn remainder 2) :floor-divide (mlx-fn floor_divide 2)
   :less (mlx-fn less 2) :less-equal (mlx-fn less_equal 2) :greater (mlx-fn greater 2)
   :greater-equal (mlx-fn greater_equal 2) :equal (mlx-fn equal 2) :not-equal (mlx-fn not_equal 2)
   :logical-and (mlx-fn logical_and 2) :logical-or (mlx-fn logical_or 2)})

(defn- unary [b op x dtype]
  (cond
    ;; MLX's kernels that are not exact take exact routes (gen.backend.mlx.exact,
    ;; loaded on first use: it is written in gen.array, which loads this namespace)
    (and (= dtype :float64) (#{:exp :sin :cos :erf :erfinv} op))
    ((requiring-resolve 'gen.backend.mlx.exact/float64-unary) op (if (mlx-array? x) x (wrap b (operand b x dtype))))

    (and (= op :expm1) (= :gpu (:device (config b))) (#{:float32 :float16 :bfloat16} dtype))
    ((requiring-resolve 'gen.backend.mlx.exact/expm1) (if (mlx-array? x) x (wrap b (operand b x dtype))))

    (and (#{:floor :ceil :round} op) (not= :float (dt/kind dtype)))
    ;; rounding an int is the identity
    (if (mlx-array? x) x (wrap b (operand b x dtype)))

    :else
    (let [f (unary-fns op)
          s (stream b)]
      (with-operands b op [h x dtype]
        (wrap b (if (and (= op :sign) (= :float (dt/kind dtype)))
                  ;; MLX's sign of NaN is 0; NumPy's (and the JVM backend's) NaN
                  (let [sg (f h s)
                        nan (Mlx/mlx_isnan h s)]
                    (try (Mlx/mlx_where nan h sg s)
                         (finally (MlxRaw/mlx_array_free sg) (MlxRaw/mlx_array_free nan))))
                  (f h s)))))))

(defn- int-division
  "remainder or floor-divide of int handles, as the JVM backend (and NumPy):
   0 for a zero divisor, since integer division by zero is undefined in
   MLX's kernels; and floor division rounding down, which MLX's (truncating)
   floor_divide does not: (x - x mod y) / y, an exact division."
  [b op hx hy dtype]
  (let [hx (long hx) hy (long hy)
        s (stream b)
        temps (java.util.ArrayList.)
        t (fn ^long [^long h] (.add temps h) h)
        zero (t (scalar-handle 0 dtype))
        one (t (scalar-handle 1 dtype))]
    (try
      (let [z (t (Mlx/mlx_equal hy zero s))
            safe (t (Mlx/mlx_where z one hy s))
            r (t (Mlx/mlx_remainder hx safe s))
            q (if (= op :remainder)
                r
                (t (Mlx/mlx_floor_divide (t (Mlx/mlx_subtract hx r s)) safe s)))]
        (Mlx/mlx_where z zero q s))
      (finally (doseq [h temps] (MlxRaw/mlx_array_free (long h)))))))

(defn- binary [b op x y dtype]
  (let [sx (shape-of x) sy (shape-of y)]
    (sh/broadcast sx sy)
    (let [s (stream b)]
      (with-operands b op [hx x dtype hy y dtype]
        (wrap b (if (and (#{:remainder :floor-divide} op) (= :int (dt/kind dtype)))
                  (int-division b op hx hy dtype)
                  ((binary-fns op) hx hy s)))))))

(defn- where-op [b c x y dtype]
  (sh/broadcast-all [(shape-of c) (shape-of x) (shape-of y)])
  (let [s (stream b)]
    (with-operands b :where [hc c :bool hx x dtype hy y dtype]
      (wrap b (Mlx/mlx_where hc hx hy s)))))

(defn- elementwise [b op args dtype]
  (case (count args)
    1 (unary b op (nth args 0) dtype)
    2 (binary b op (nth args 0) (nth args 1) dtype)
    3 (where-op b (nth args 0) (nth args 1) (nth args 2) dtype)))


;; ---------------------------------------------------------------------------
;; Reductions and scans
;; ---------------------------------------------------------------------------

(defn- arg-reduce
  "argmax or argmin of handle h along axis, as the default int dtype. As in
   NumPy, the first NaN wins (MLX skips NaNs)."
  [b op h axis kd dtype]
  (let [h (long h)
        s (stream b)
        ax (int axis)
        kd (boolean kd)
        temps (java.util.ArrayList.)
        t (fn ^long [^long x] (.add temps x) x)
        int-code (int (code (:int (defaults b))))]
    (try
      (let [i (t (if (= op :argmax) (Mlx/mlx_argmax_axis h ax kd s) (Mlx/mlx_argmin_axis h ax kd s)))
            i (if (= :float (dt/kind dtype))
                (let [nan (t (Mlx/mlx_isnan h s))
                      first-nan (t (Mlx/mlx_argmax_axis nan ax kd s))
                      any-nan (t (Mlx/mlx_any_axis nan ax kd s))]
                  (t (Mlx/mlx_where any-nan first-nan i s)))
                i)]
        ;; MLX's indices are uint32
        (Mlx/mlx_astype i int-code s))
      (finally (doseq [x temps] (MlxRaw/mlx_array_free (long x)))))))

(defn- reduce-op [b op x axes keep-dims?]
  (let [s (stream b)
        dtype (dtype-on b x)]
    (if (empty? axes)
      ;; over no axis: the identity (argmax/argmin always name one axis)
      (if (mlx-array? x) x (wrap b (operand b x dtype)))
      (let [ax (int-array axes)
            kd (boolean keep-dims?)]
        (with-operands b op [h x dtype]
          (wrap b (case op
                    :sum (Mlx/mlx_sum_axes h ax kd s)
                    :prod (Mlx/mlx_prod_axes h ax kd s)
                    :max (Mlx/mlx_max_axes h ax kd s)
                    :min (Mlx/mlx_min_axes h ax kd s)
                    :all (Mlx/mlx_all_axes h ax kd s)
                    :any (Mlx/mlx_any_axes h ax kd s)
                    (:argmax :argmin) (arg-reduce b op h (first axes) kd dtype))))))))

(defn- scan-op [b op x axis reverse?]
  (let [s (stream b)
        ax (int axis)
        rev (boolean reverse?)]
    (with-operands b op [h x nil]
      (wrap b (case op
                :cumsum (Mlx/mlx_cumsum_axis h ax rev true nil s)
                :cumprod (Mlx/mlx_cumprod_axis h ax rev true nil s)
                :cummax (Mlx/mlx_cummax_axis h ax rev true s)
                :cummin (Mlx/mlx_cummin_axis h ax rev true s))))))

;; ---------------------------------------------------------------------------
;; Shape and indexing
;; ---------------------------------------------------------------------------

(defn- reshape [b x shape]
  (with-operands b :reshape [h x nil]
    (wrap b (Mlx/mlx_reshape h (int-array (map int shape)) (stream b)))))

(defn- transpose [b x perm]
  (with-operands b :transpose [h x nil]
    (wrap b (Mlx/mlx_transpose_axes h (int-array (map int perm)) (stream b)))))

(defn- broadcast-to [b x shape]
  (let [shape (vec shape)]
    (when-not (= shape (sh/broadcast (shape-of x) shape))
      (throw (ex-info (str "Cannot broadcast " (shape-of x) " to " shape)
                      {:from (shape-of x) :to shape})))
    (with-operands b :broadcast-to [h x nil]
      (wrap b (Mlx/mlx_broadcast_to h (int-array (map int shape)) (stream b))))))

(defn- concatenate [b xs axis]
  (let [shapes (mapv shape-of xs)
        s0 (first shapes)]
    (doseq [s shapes]
      (when-not (and (= (count s) (count s0))
                     (= (assoc s axis 0) (assoc s0 axis 0)))
        (throw (ex-info "Arrays must agree on every axis but the concatenation axis"
                        {:shapes shapes :axis axis}))))
    (let [d (dtype-on b (first xs))
          hs (mapv #(operand b % d) xs)
          v (MlxRuntime/vectorOf (long-array hs))]
      (try
        (wrap b (Mlx/mlx_concatenate_axis v (int axis) (stream b)))
        (catch MlxException e (throw (mlx-error :concatenate xs e)))
        (finally
          (Mlx/mlx_vector_array_free v)
          (doseq [[x h] (map vector xs hs)] (release x h))
          (fence xs))))))

(defn- slice [b x starts stops steps]
  (let [shape (shape-of x)
        ;; MLX reads negative starts and stops from the end. With a negative
        ;; step, NumPy's normalized bounds use -1 for "before index 0": a
        ;; start there is an empty slice, and a stop there is -(n+1) to MLX
        neg-empty? (mapv (fn [st step] (and (neg? (long step)) (== -1 (long st)))) starts steps)
        starts (mapv (fn [st e?] (if e? 0 st)) starts neg-empty?)
        stops (mapv (fn [e st n e?]
                      (cond e? 0
                            (and (neg? (long st)) (== -1 (long e))) (- (inc (long n)))
                            :else e))
                    stops steps shape neg-empty?)]
    (with-operands b :slice [h x nil]
      (wrap b (Mlx/mlx_slice h (int-array (map int starts)) (int-array (map int stops))
                             (int-array (map int steps)) (stream b))))))

(defn- host-index
  "The host integer index v checked against a dimension of size d."
  ^long [v ^long d]
  (let [i (long v)
        i (if (neg? i) (+ i d) i)]
    (if (and (>= i 0) (< i d))
      i
      (throw (ex-info (str "Index " v " is out of bounds for a dimension of size " d)
                      {:index v :size d})))))

(defn- safe-indices
  "A handle of the int indices h normalized against a dimension of size d:
   negative ones count from the end, and out-of-range ones are clamped (MLX's
   gathers do not check bounds; a lazy backend cannot check them without
   evaluating)."
  ^long [b ^long h ^long d]
  (let [s (stream b)
        ;; int32 indices, whatever the int dtype
        i32 (Mlx/mlx_astype h (code :int32) s)
        zero (MlxRaw/mlx_array_new_int 0)
        dd (MlxRaw/mlx_array_new_int (int d))
        top (MlxRaw/mlx_array_new_int (int (dec d)))
        neg (Mlx/mlx_less i32 zero s)
        wrapped (Mlx/mlx_add i32 dd s)
        n (Mlx/mlx_where neg wrapped i32 s)]
    (try
      (Mlx/mlx_clip n zero top s)
      (finally
        (doseq [t [i32 zero dd top neg wrapped n]] (MlxRaw/mlx_array_free (long t)))))))

(defn- take-op [b x indices axis]
  (let [d (long (nth (shape-of x) axis))]
    (if (host? indices)
      (let [i (host-index indices d)]
        (with-operands b :take [h x nil hi i :int32]
          (wrap b (Mlx/mlx_take_axis h hi (int axis) (stream b)))))
      (with-operands b :take [h x nil hi indices nil]
        (let [si (safe-indices b hi d)]
          (try (wrap b (Mlx/mlx_take_axis h si (int axis) (stream b)))
               (finally (MlxRaw/mlx_array_free si))))))))

(defn- take-along-axis [b x indices axis]
  (let [sx (shape-of x) si (shape-of indices)]
    (when-not (= (count sx) (count si))
      (throw (ex-info "take-along-axis needs indices of the same rank as the array"
                      {:shape sx :indices-shape si})))
    (doseq [[k dx di] (map vector (range) sx si)]
      (when (not= k axis) (sh/broadcast [dx] [di])))
    (with-operands b :take-along-axis [h x nil hi indices nil]
      (let [si (safe-indices b hi (nth sx axis))]
        (try (wrap b (Mlx/mlx_take_along_axis h si (int axis) (stream b)))
             (finally (MlxRaw/mlx_array_free si)))))))

(defn- scatter-add [b x indices updates axis]
  (let [sx (shape-of x) su (shape-of updates)]
    (when-not (and (= su (shape-of indices)) (= (count sx) (count su))
                   (every? true? (map (fn [k a c] (or (= k axis) (= a c))) (range) sx su)))
      (throw (ex-info "scatter-add needs indices and updates of one shape, equal to the array's but on axis"
                      {:shape sx :indices-shape (shape-of indices) :updates-shape su :axis axis})))
    (let [d (dtype-on b x)]
      (with-operands b :scatter-add [h x d hi indices nil hu updates d]
        (let [si (safe-indices b hi (nth sx axis))]
          (try (wrap b (Mlx/mlx_scatter_add_axis h si hu (int axis) (stream b)))
               (finally (MlxRaw/mlx_array_free si))))))))

;; ---------------------------------------------------------------------------
;; Linear algebra
;; ---------------------------------------------------------------------------

(defn- matmul [b x y]
  (let [sx (shape-of x) sy (shape-of y)]
    (when (or (empty? sx) (empty? sy))
      (throw (ex-info "matmul needs arrays of rank >= 1" {:shapes [sx sy]})))
    (let [kx (peek sx)
          ky (if (= 1 (count sy)) (first sy) (nth sy (- (count sy) 2)))]
      (when-not (= kx ky)
        (throw (ex-info (str "matmul: inner dimensions differ (" kx " vs " ky ")") {:shapes [sx sy]}))))
    (let [d (dtype-on b x)
          s (stream b)]
      (with-operands b :matmul [hx x d hy y d]
        (if (= :float (dt/kind d))
          (wrap b (Mlx/mlx_matmul hx hy s))
          ;; MLX multiplies floats only: ints go through the widest float the
          ;; device has (exact while products stay below 2^24 or 2^53)
          (let [f (code (:float (defaults b)))
                fx (Mlx/mlx_astype hx f s)
                fy (Mlx/mlx_astype hy f s)
                m (Mlx/mlx_matmul fx fy s)]
            (try (wrap b (Mlx/mlx_astype m (code d) s))
                 (finally (doseq [t [fx fy m]] (MlxRaw/mlx_array_free (long t)))))))))))

(defn- check-square [shape op]
  (let [r (count shape)]
    (when (or (< r 2) (not= (nth shape (- r 1)) (nth shape (- r 2))))
      (throw (ex-info (str (name op) " needs square matrices, got shape " shape)
                      {:op op :shape shape})))))

(defn- batch-broadcast
  "Handles of a [.. n n] and rhs [.. n k] broadcast to one batch shape (MLX's
   solves want equal batch axes; gen.array's broadcast them)."
  [b ha sa hr sr temps]
  (let [ha (long ha) hr (long hr)
        ba (subvec sa 0 (- (count sa) 2))
        br (subvec sr 0 (- (count sr) 2))
        batch (sh/broadcast ba br)
        s (cpu-stream b)
        to (fn [^long h shape tail]
             (if (= shape (into batch tail))
               h
               (let [t (Mlx/mlx_broadcast_to h (int-array (map int (into batch tail))) s)]
                 (.add ^java.util.ArrayList temps t)
                 t)))]
    [(to ha sa (subvec sa (- (count sa) 2))) (to hr sr (subvec sr (- (count sr) 2)))]))

(defn- diag-of-temp
  "The diagonals of the matrices h, [.. n], a handle registered with t."
  ^long [t ^long h s]
  (long (t (Mlx/mlx_diagonal h (int 0) (int -2) (int -1) (long s)))))

(defn- linalg [b op args]
  (let [cpu (cpu-stream b)
        temps (java.util.ArrayList.)
        t (fn ^long [^long h] (.add temps h) h)
        nan (fn ^long [dtype] (t (scalar-handle ##NaN dtype)))]
    (try
      (case op
        ;; MLX returns a partial factor for a matrix that is not positive
        ;; definite; LAPACK leaves the failing pivot non-positive there, so the
        ;; contract's NaN is where a diagonal entry is not positive
        :cholesky
        (let [[a] args
              d (dtype-on b a)]
          (check-square (shape-of a) :cholesky)
          (with-operands b :cholesky [h a nil]
            (let [l (t (Mlx/mlx_linalg_cholesky h false cpu))
                  pd (t (Mlx/mlx_all_axis (t (Mlx/mlx_greater (diag-of-temp t l cpu) (t (scalar-handle 0 d)) cpu))
                                          (int -1) true cpu))]
              (wrap b (Mlx/mlx_where (t (Mlx/mlx_expand_dims pd (int -1) cpu)) l (nan d) cpu)))))

        ;; MLX's triangular solve inverts the matrix, which fails on a zero
        ;; diagonal entry (reported at a later evaluation, as for :solve): a NaN
        ;; there instead gives NaN solutions, as a degenerate solve gives
        ;; non-finite ones on the JVM backend (not an error)
        :solve-triangular
        (let [[a rhs {:keys [lower? transpose?]}] args
              d (dtype-on b a)
              sa (shape-of a)]
          (check-square sa :solve-triangular)
          (with-operands b :solve-triangular [ha a nil hr rhs nil]
            (let [[ha hr] (batch-broadcast b ha sa hr (shape-of rhs) temps)
                  n (int (peek sa))
                  eye (t (Mlx/mlx_eye n n (int 0) (int (code :bool)) cpu))
                  zero-diag (t (Mlx/mlx_expand_dims (t (Mlx/mlx_equal (diag-of-temp t ha cpu) (t (scalar-handle 0 d)) cpu))
                                                    (int -2) cpu))
                  safe (t (Mlx/mlx_where (t (Mlx/mlx_logical_and eye zero-diag cpu)) (nan d) ha cpu))]
              ;; Lᵀx = b is solving with the upper triangle of Lᵀ
              (wrap b (if transpose?
                        (Mlx/mlx_linalg_solve_triangular (t (Mlx/mlx_swapaxes safe (int -1) (int -2) cpu)) hr
                                                         (boolean lower?) cpu)
                        (Mlx/mlx_linalg_solve_triangular safe hr (not lower?) cpu))))))

        ;; A singular system fails inside MLX's solve, and MLX reports a CPU
        ;; task's failure at a later evaluation, of other arrays. So singular
        ;; matrices (a zero pivot of the LU factor) are never given to it: the
        ;; identity stands in, and their solutions are NaN
        :solve
        (let [[a rhs] args
              d (dtype-on b a)
              sa (shape-of a)]
          (check-square sa :solve)
          (with-operands b :solve [ha a nil hr rhs nil]
            (let [[ha hr] (batch-broadcast b ha sa hr (shape-of rhs) temps)
                  n (int (peek sa))
                  [lu _] (map t (Mlx/mlx_linalg_lu_factor ha cpu))
                  singular (t (Mlx/mlx_expand_dims
                               (t (Mlx/mlx_any_axis (t (Mlx/mlx_equal (diag-of-temp t lu cpu) (t (scalar-handle 0 d)) cpu))
                                                    (int -1) true cpu))
                               (int -1) cpu))
                  eye (t (Mlx/mlx_eye n n (int 0) (int (code d)) cpu))
                  safe (t (Mlx/mlx_where singular eye ha cpu))
                  x (t (Mlx/mlx_linalg_solve safe hr cpu))]
              (wrap b (Mlx/mlx_where singular (nan d) x cpu)))))

        ;; from the LU factor (MLX's slogdet throws on a singular matrix):
        ;; sign · Π sign(u_ii) · (-1)^swaps and Σ log |u_ii|, so a singular
        ;; matrix gives [0 -Inf], as on the JVM backend
        :slogdet
        (let [[a] args
              d (dtype-on b a)
              sa (shape-of a)]
          (check-square sa :slogdet)
          (with-operands b :slogdet [h a nil]
            (let [[lu piv] (map t (Mlx/mlx_linalg_lu_factor h cpu))
                  u (diag-of-temp t lu cpu)
                  n (peek sa)
                  logabs (Mlx/mlx_sum_axis (t (Mlx/mlx_log (t (Mlx/mlx_abs u cpu)) cpu)) (int -1) false cpu)
                  idx (t (Mlx/mlx_arange 0.0 (double n) 1.0 (int (code :uint32)) cpu))
                  swaps (t (Mlx/mlx_sum_axis (t (Mlx/mlx_astype (t (Mlx/mlx_not_equal piv idx cpu)) (int (code :int32)) cpu))
                                             (int -1) false cpu))
                  odd (t (Mlx/mlx_remainder swaps (t (scalar-handle 2 :int32)) cpu))
                  perm (t (Mlx/mlx_subtract (t (scalar-handle 1 d))
                                            (t (Mlx/mlx_multiply (t (scalar-handle 2 d)) (t (Mlx/mlx_astype odd (int (code d)) cpu)) cpu))
                                            cpu))
                  sign (Mlx/mlx_multiply (t (Mlx/mlx_prod_axis (t (Mlx/mlx_sign u cpu)) (int -1) false cpu)) perm cpu)]
              [(wrap b sign) (wrap b logabs)]))))
      (finally (doseq [h temps] (MlxRaw/mlx_array_free (long h)))))))

;; ---------------------------------------------------------------------------
;; Randomness: JAX's partitionable Threefry-2x32 (gen.random.threefry) in
;; MLX's uint32 ops, so a key draws the same bits as on the JVM backend
;; ---------------------------------------------------------------------------

(defn- threefry-words
  "[w0 w1] handles of Threefry-2x32-20 under key [k0 k1] of the counters
   [c0 c1] (uint32 handles), as gen.random.threefry/hash2x32."
  [b k0 k1 c0 c1]
  (let [s (stream b)
        temps (java.util.ArrayList.)
        t (fn ^long [^long h] (.add temps h) h)
        u (fn ^long [v] (t (scalar-handle v :uint32)))
        add (fn ^long [^long x ^long y] (t (Mlx/mlx_add x y s)))
        rotl (fn ^long [^long x r]
               (t (Mlx/mlx_bitwise_or (t (Mlx/mlx_left_shift x (u r) s))
                                      (t (Mlx/mlx_right_shift x (u (- 32 (long r))) s))
                                      s)))
        hk0 (u k0) hk1 (u k1)
        k2 (bit-xor (long k0) (long k1) 0x1BD11BDA)
        hk2 (u k2)
        ks [hk0 hk1 hk2]
        rounds (fn [[x0 x1] rs]
                 (reduce (fn [[x0 x1] r]
                           (let [x0 (add x0 x1)]
                             [x0 (t (Mlx/mlx_bitwise_xor (rotl x1 r) x0 s))]))
                         [x0 x1] rs))
        inject (fn [[x0 x1] i]
                 ;; after round group i: x0 += ks[i mod 3], x1 += ks[i+1 mod 3] + i
                 [(add x0 (nth ks (mod i 3)))
                  (add x1 (u (bit-and (+ (long (nth [k0 k1 k2] (mod (inc i) 3))) i) 0xFFFFFFFF)))])
        rs-a [13 15 26 6]
        rs-b [17 29 16 24]
        start [(add c0 hk0) (add c1 hk1)]]
    (try
      (let [[w0 w1] (-> start
                        (rounds rs-a) (inject 1)
                        (rounds rs-b) (inject 2)
                        (rounds rs-a) (inject 3)
                        (rounds rs-b) (inject 4)
                        (rounds rs-a) (inject 5))]
        ;; keep the two outputs
        (.remove temps (Long/valueOf (long w0)))
        (.remove temps (Long/valueOf (long w1)))
        [w0 w1])
      (finally (doseq [h temps] (MlxRaw/mlx_array_free (long h)))))))

(defn- random-uniform [b ^Key key shape dtype]
  (when-not (#{:float32 :float64} dtype)
    (throw (ex-info (str "random-uniform needs a float dtype, got " dtype) {:dtype dtype})))
  (let [s (stream b)
        n (sh/size shape)
        _ (when (>= n 0x100000000)
            (throw (ex-info "random-uniform: more than 2^32 elements" {:shape shape})))
        temps (java.util.ArrayList.)
        t (fn ^long [^long h] (.add temps h) h)
        ;; counters [hi32(i) lo32(i)] = [0 i]
        c1 (t (Mlx/mlx_arange 0.0 (double n) 1.0 (int (code :uint32)) s))
        c0 (t (Mlx/mlx_zeros (int-array [n]) (int (code :uint32)) s))]
    (try
      (let [[w0 w1] (mapv t (threefry-words b (.-k0 key) (.-k1 key) c0 c1))
            u (case dtype
                ;; 23 mantissa bits of w0 ^ w1, as gen.random.threefry/uniform-floats
                :float32 (let [bits (t (Mlx/mlx_bitwise_xor w0 w1 s))
                               m (t (Mlx/mlx_bitwise_or (t (Mlx/mlx_right_shift bits (t (scalar-handle 9 :uint32)) s))
                                                        (t (scalar-handle 0x3F800000 :uint32)) s))
                               f (t (Mlx/mlx_view m (int (code :float32)) s))]
                           (t (Mlx/mlx_subtract f (t (scalar-handle 1.0 :float32)) s)))
                ;; 52 mantissa bits of w0 << 32 | w1, as uniform-doubles
                :float64 (let [u64 (int 4)
                               hi (t (Mlx/mlx_astype w0 u64 s))
                               lo (t (Mlx/mlx_astype w1 u64 s))
                               c (fn ^long [v] (t (Mlx/mlx_astype (t (scalar-handle v :int64)) u64 s)))
                               m (t (Mlx/mlx_bitwise_or
                                     (t (Mlx/mlx_bitwise_or (t (Mlx/mlx_left_shift hi (c 20) s))
                                                            (t (Mlx/mlx_right_shift lo (c 12) s)) s))
                                     (c 0x3FF0000000000000) s))
                               f (t (Mlx/mlx_view m (int (code :float64)) s))]
                           (t (Mlx/mlx_subtract f (t (scalar-handle 1.0 :float64)) s))))]
        (wrap b (Mlx/mlx_reshape u (int-array (map int shape)) s)))
      (finally (doseq [h temps] (MlxRaw/mlx_array_free (long h)))))))

;; ---------------------------------------------------------------------------
;; The backend
;; ---------------------------------------------------------------------------

(defmacro ^:private guarded
  "body, with an MLX error raised outside an op's own handling as ex-info
   naming op (gen.array's contract: errors are ex-info with data)."
  [op & body]
  `(try ~@body
        (catch MlxException e# (throw (mlx-error ~op [] e#)))))

(deftype MlxBackend [^Config config]
  p/IBackend
  (-name [_] :mlx)
  (-defaults [_] (.-defaults config))
  (-lazy? [_] true)
  (-native-ops [_] #{})

  (-from-host [b data dtype] (from-host b data dtype))
  (-to-host [_ x] (to-host x))
  (-eval! [_ xs] (eval-arrays! (filter mlx-array? xs)))

  (-shape [_ x] (shape-of x))
  (-dtype [b x] (dtype-on b x))

  (-full [b shape value dtype]
    (code dtype)
    (with-operands b :full [hv value dtype]
      (wrap b (Mlx/mlx_full (int-array (map int shape)) hv (int (code dtype)) (stream b)))))
  (-arange [b start stop step dtype]
    (guarded :arange
      (wrap b (Mlx/mlx_arange (double start) (double stop) (double step) (int (code dtype)) (stream b)))))
  (-astype [b x dtype]
    (with-operands b :astype [h x nil]
      (wrap b (Mlx/mlx_astype h (int (code dtype)) (stream b)))))

  (-elementwise [b op args dtype] (guarded op (elementwise b op args dtype)))
  (-reduce [b op x axes keep-dims?] (guarded op (reduce-op b op x axes keep-dims?)))
  (-scan [b op x axis reverse?] (guarded op (scan-op b op x axis reverse?)))

  (-reshape [b x shape] (reshape b x shape))
  (-transpose [b x perm] (transpose b x perm))
  (-broadcast-to [b x shape] (broadcast-to b x shape))
  (-concatenate [b xs axis] (concatenate b xs axis))
  (-slice [b x starts stops steps] (slice b x starts stops steps))

  (-take [b x indices axis] (guarded :take (take-op b x indices axis)))
  (-take-along-axis [b x indices axis] (guarded :take-along-axis (take-along-axis b x indices axis)))
  (-scatter-add [b x indices updates axis] (guarded :scatter-add (scatter-add b x indices updates axis)))

  (-matmul [b x y] (guarded :matmul (matmul b x y)))
  (-linalg [b op args] (guarded op (linalg b op args)))
  (-random-uniform [b key shape dtype] (guarded :random-uniform (random-uniform b key shape dtype)))
  (-value-and-grad [b f args argnums] ((requiring-resolve 'gen.array.tape/value-and-grad) b f args argnums)))

(defn- config ^Config [b] (.-config ^MlxBackend b))

(defn- device-handle ^long [device]
  (Mlx/mlx_device_new_type (int (case device :cpu 0 :gpu 1)) (int 0)))

(defn gpu?
  "Does MLX have a GPU here (Metal, or CUDA on Linux)?"
  []
  (Mlx/mlx_device_is_available (device-handle :gpu)))

(defn make-backend
  "An MLX backend. Options: :device (:cpu or :gpu, default :gpu when MLX has
   one), :dtypes (32 or 64: the default float and int dtypes; 64 needs the
   CPU)."
  [{:keys [device dtypes] :or {dtypes 32}}]
  (let [device (or device (if (gpu?) :gpu :cpu))]
    (when (and (= device :gpu) (not (gpu?)))
      (throw (ex-info "MLX has no GPU here" {:device device})))
    (when (and (= device :gpu) (= 64 dtypes))
      (throw (ex-info "MLX's GPUs have no float64: use :dtypes 64 with :device :cpu" {:device device :dtypes dtypes})))
    (MlxBackend.
     (->Config device
               (Mlx/mlx_stream_new_thread_unsafe (device-handle device))
               (Mlx/mlx_stream_new_thread_unsafe (device-handle :cpu))
               (case (long dtypes)
                 32 {:float :float32 :int :int32}
                 64 {:float :float64 :int :int64})))))

(def backend
  "The MLX backend configured by the system properties gen.mlx.device and
   gen.mlx.dtypes (see the namespace doc)."
  (make-backend {:device (some-> (System/getProperty "gen.mlx.device") keyword)
                 :dtypes (Long/parseLong (System/getProperty "gen.mlx.dtypes" "32"))}))

(defn device
  "The device of an MLX backend: :cpu or :gpu."
  [b]
  (:device (config b)))

(defn memory
  "MLX's memory use in bytes: :active, :peak and :cache."
  []
  {:active (Mlx/mlx_get_active_memory)
   :peak (Mlx/mlx_get_peak_memory)
   :cache (Mlx/mlx_get_cache_memory)})

(defmethod print-method MlxArray [^MlxArray x ^java.io.Writer w]
  (.write w "#gen/array[")
  (print-method (.-dtype x) w)
  (.write w " ")
  (print-method (to-host x) w)
  (.write w "]"))
