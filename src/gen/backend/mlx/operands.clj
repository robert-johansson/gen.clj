(ns gen.backend.mlx.operands
  "The MLX backend's operands (gen.backend.mlx): handles made for host
   scalars (shared constants), temporaries freed when a site exits (by an
   exception too), an op's operands as handles, kept reachable through its
   call, and arrays from host data and read back."
  {:api :internal}
  (:require [gen.array.dtype :as dt]
            [gen.array.shape :as sh]
            [gen.backend.mlx.array :refer [aux-stream check-dtype code defaults device fence handle mlx-array? stream wrap]]
            [gen.backend.mlx.memory :refer [at-buffer-limit evaluating]])
  (:import [gen.backend.mlx.array MlxArray]
           [gen.mlx Mlx MlxRaw MlxRuntime MlxException]
           [java.lang.foreign Arena MemorySegment ValueLayout]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Operands: MLX arrays and host scalars
;; ---------------------------------------------------------------------------

(defn host? [x] (or (instance? Number x) (instance? Boolean x)))

(defn made
  "The handle (make) returns, from the mlx-c constructor named what: those
   report a failure (out of memory, Metal's buffer limit) only by an empty
   handle, which throws MLX's error, after a retry at the buffer limit
   (at-buffer-limit)."
  ^long [what make]
  (at-buffer-limit what #(let [h (long (make))]
                           (if (zero? h) (throw (MlxRuntime/error what)) h))))

(defn- half-scalar-handle
  "A new handle for the host scalar x as :float16 or :bfloat16."
  [x dtype]
  (let [f (made "mlx_array_new_float32" #(MlxRaw/mlx_array_new_float32 (unchecked-float (dt/host-double x))))]
    (try (Mlx/mlx_astype f (int (code dtype)) (long aux-stream))
         (finally (MlxRaw/mlx_array_free f)))))

(declare scalar-handle*)

(defn scalar-handle
  "A new handle for the host scalar x as dtype."
  ^long [x dtype]
  (made "mlx_array_new" #(scalar-handle* x dtype)))

(defn- scalar-handle*
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

(defn host-dtype-on
  "The dtype a host scalar has on b where nothing else gives one: the
   backend's default for its kind (host numbers are weakly typed)."
  [b x]
  (dt/weak-host-dtype x (defaults b)))

(def ^:private max-constants
  "How many host-scalar constants the backend keeps per dtype
   (gen.mlx.constants; each holds one of Metal's ~500,000 buffers)."
  (Long/parseLong (System/getProperty "gen.mlx.constants" "4096")))

(defonce ^:private constants
  ;; dtype → (value bits → handle). The handles are never freed: graphs
  ;; share them, so a lazy graph pins one buffer per distinct constant
  ;; rather than one per use (Metal allows about 500,000 buffers)
  (java.util.concurrent.ConcurrentHashMap.))

(defn- constant-bits ^long [x dtype]
  (case (dt/kind dtype)
    :float (Double/doubleToRawLongBits (dt/host-double (dt/host-cast x dtype)))
    :int (dt/host-long (dt/host-cast x dtype))
    :bool (if (dt/truthy? x) 1 0)))

(defn constant
  "The shared handle of the host scalar x as dtype, or nil when the cache is
   full."
  [x dtype]
  (let [^java.util.concurrent.ConcurrentHashMap by-value
        (.computeIfAbsent ^java.util.concurrent.ConcurrentHashMap constants dtype
                          (reify java.util.function.Function
                            (apply [_ _] (java.util.concurrent.ConcurrentHashMap.))))
        k (constant-bits x dtype)]
    (or (.get by-value k)
        (when (< (.size by-value) max-constants)
          (let [h (scalar-handle x dtype)
                prev (.putIfAbsent by-value k h)]
            (if prev
              (do (MlxRaw/mlx_array_free h) prev)
              h))))))

;; Temporary handles: every one a site makes is registered as it is made,
;; and all are freed when the site exits, by an exception too (genclj-f0wz)

(defn free-temp
  "Free a temporary: an array's handle (a long), or [kind h] for a closure,
   a custom VJP, a value-and-grad closure or a vector of arrays."
  [x]
  (if (vector? x)
    (let [[kind h] x
          h (long h)]
      (case kind
        :closure (MlxRaw/mlx_closure_free h)
        :custom-vjp (MlxRaw/mlx_closure_custom_free h)
        :value-and-grad (MlxRaw/mlx_closure_value_and_grad_free h)
        :vector (MlxRaw/mlx_vector_array_free h)))
    (MlxRaw/mlx_array_free (long x))))

(def ^:private temp-kinds #{:closure :custom-vjp :value-and-grad :vector})

(defn temp!
  "Register the temporary x (as free-temp takes it) with temps; its handle.
   A kind free-temp does not know fails here, where it is written, not in
   with-temps' finally, where it would mask the body's error."
  ^long [^java.util.ArrayList temps x]
  (when (and (vector? x) (not (temp-kinds (first x))))
    (throw (ex-info (str "Not a kind of temporary: " (pr-str (first x))) {:kind (first x) :kinds temp-kinds})))
  (.add temps x)
  (long (if (vector? x) (second x) x)))

(defmacro with-temps
  "Run body with t a function that registers a new temporary and returns
   its handle: (t h) an array's, (t kind h) another kind's (free-temp).
   Every one registered is freed when body exits, by an exception too, so
   a site whose handles all go through t as it makes them frees those made
   before a failure. (The review's I1-04: written out by hand ten times,
   three copies made handles before their try, and leaked them there.)"
  [[t] & body]
  ;; primitive hints built outside the syntax quote, which would qualify them
  (let [temps (gensym "temps")
        h (with-meta (gensym "h") {:tag 'long})
        kind (gensym "kind")]
    `(let [~temps (java.util.ArrayList.)
           ~t (fn (~(with-meta [h] {:tag 'long}) (temp! ~temps ~h))
                  (~(with-meta [kind h] {:tag 'long}) (temp! ~temps [~kind ~h])))]
       (try
         ~@body
         (finally
           (doseq [x# ~temps] (free-temp x#)))))))

(defn operand
  "The handle of operand x: x's own, or the shared constant for a host scalar
   (as dtype, or b's default for it). When the constant cache is full, a new
   handle, returned negated: release frees it. (Handles are user-space
   addresses, positive as longs.)"
  ^long [b x dtype]
  (if (mlx-array? x)
    (handle x)
    (let [d (or dtype (host-dtype-on b x))]
      (if-let [h (constant x d)]
        (long h)
        (- (scalar-handle x d))))))

(defn as-array
  "x as an MlxArray of b (a host scalar as dtype, or b's default)."
  [b x dtype]
  (if (mlx-array? x) x (wrap b (scalar-handle x (or dtype (host-dtype-on b x))))))

(defn shape-of [x] (if (mlx-array? x) (.-shape ^MlxArray x) []))

(defn dtype-on [b x] (if (mlx-array? x) (.-dtype ^MlxArray x) (host-dtype-on b x)))

(defn mlx-error
  "ex-info for an error MLX reported in op on args."
  [op args ^MlxException e]
  (ex-info (str (name op) ": " (.getMessage e))
           {:op op :shapes (mapv shape-of args) :mlx-error (.getMessage e)}
           e))

(defmacro with-operands
  "Bind each handle symbol to the handle of its operand (b, x, dtype), run
   body, then free the handles made for host scalars (with-temps, so also
   when a later operand's fails to be made) and keep the operands
   reachable. MLX errors become ex-info naming op."
  [b op bindings & body]
  (let [triples (partition 3 bindings)
        ;; each operand form evaluated once, before any handle is made
        xs (repeatedly (count triples) #(gensym "x"))
        t (gensym "t")]
    `(let [~@(mapcat (fn [x [_ form _]] [x form]) xs triples)]
       (with-temps [~t]
         (try
           (let [~@(mapcat (fn [x [h _ d]] [h `(let [raw# (operand ~b ~x ~d)] (if (neg? raw#) (long (~t (- raw#))) raw#))])
                           xs triples)]
             ~@body)
           (catch MlxException e# (throw (mlx-error ~op [~@xs] e#)))
           (finally
             ~@(map (fn [x] `(fence ~x)) xs)))))))

(defn retained
  "A new reference to the handle of x (an MlxArray, or a host scalar as b's
   default dtype) for MLX to own, as a closure's result: x kept reachable
   until the reference is made (a host scalar's array, made for the call,
   would be unreachable as soon as its handle was read). MLX errors name
   op."
  ^long [b op x]
  (long (with-operands b op [h x nil] (MlxRuntime/retain h))))

;; ---------------------------------------------------------------------------
;; Host boundary
;; ---------------------------------------------------------------------------

(defn- segment-of
  "Native memory (in arena) holding values (host scalars, or a double[]) as
   dtype: a double[] copied as it is to float64, rounded element by element
   to float32 (as host-cast rounds), without boxing."
  ^MemorySegment [^Arena arena values dtype]
  (cond
    (and (instance? double/1 values) (= :float64 dtype))
    (.allocateFrom arena ValueLayout/JAVA_DOUBLE ^doubles values)

    (and (instance? double/1 values) (= :float32 dtype))
    (let [^doubles d values
          fs (float-array (alength d))]
      (dotimes [i (alength d)] (aset fs i (unchecked-float (aget d i))))
      (.allocateFrom arena ValueLayout/JAVA_FLOAT fs))

    :else
    (let [n (count values)
          vs (map #(dt/host-cast % dtype) values)]
      (case dtype
        :float32 (.allocateFrom arena ValueLayout/JAVA_FLOAT (float-array (map double vs)))
        :float64 (.allocateFrom arena ValueLayout/JAVA_DOUBLE (double-array (map double vs)))
        (:int32 :uint32) (.allocateFrom arena ValueLayout/JAVA_INT (int-array (map #(unchecked-int (long %)) vs)))
        :int64 (.allocateFrom arena ValueLayout/JAVA_LONG (long-array (map long vs)))
        :bool (.allocateFrom arena ValueLayout/JAVA_BYTE (byte-array (map #(if % 1 0) vs)))
        (throw (ex-info (str "Cannot make an MLX array of dtype " dtype " from host data") {:dtype dtype :count n}))))))

(defn- from-values
  "An array of shape and dtype from host values in row-major order."
  [b values shape dtype]
  (if (#{:float16 :bfloat16} dtype)
    (let [x (from-values b values shape :float32)]
      (with-operands b :astype [h x nil]
        (wrap b (Mlx/mlx_astype h (code dtype) (stream b)))))
    (with-open [arena (Arena/ofConfined)]
      (let [data (segment-of arena values dtype)
            shp (.allocateFrom arena ValueLayout/JAVA_INT (int-array (map int shape)))]
        (wrap b (made "mlx_array_new_data"
                      #(MlxRaw/mlx_array_new_data data shp (int (count shape)) (int (code dtype)))))))))

(defn- gpu-dtype!
  "dtype, unless it is float64 on the GPU device, which MLX's GPUs do not
   have: an array made there from host data would read back but fail at its
   first op, so it fails here."
  [b dtype]
  (when (and (= :float64 dtype) (= :gpu (device b)))
    (throw (ex-info "MLX's GPUs have no float64: make float64 arrays on a CPU-device backend (:dtypes 64)"
                    {:dtype dtype :device :gpu})))
  dtype)

(defn from-host
  "An array of host values (gen.array.host/parse's), a 0-d one through the
   scalar constructor."
  [b values shape dtype]
  (gpu-dtype! b dtype)
  (check-dtype dtype)
  (if (empty? shape)
    (wrap b (scalar-handle (first values) dtype))
    (from-values b values shape dtype)))

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
          s (stream b)]
      (with-temps [t]
        (try
          (let [h (handle x)
                h (if (= read-as dtype) h (t (Mlx/mlx_astype h (int (code read-as)) s)))
                vs (evaluating
                     (Mlx/mlx_array_eval h)
                     ;; data pointers address the buffer as laid out: a
                     ;; row-contiguous array is read in place (its flags are set
                     ;; once evaluated), and only a view is copied first, since a
                     ;; copy costs a device round trip
                     (let [c (if (Mlx/_mlx_array_is_row_contiguous h)
                               h
                               (let [c (t (Mlx/mlx_contiguous (long h) false s))]
                                 (Mlx/mlx_array_eval c)
                                 c))]
                       (read-values c n read-as)))]
            (if (empty? shape) (first vs) (nest vs shape)))
          (catch MlxException e (throw (mlx-error :->clj [x] e)))
          (finally (fence x)))))))

(def ^:private prefetch-bytes
  "The bytes from which a host read on MLX's CUDA moves its pages in one
   go (prefetched): reads of rows of logits; the system property
   gen.mlx.prefetch-bytes (0 turns it off; default 1 MiB)."
  (delay (let [n (Long/getLong "gen.mlx.prefetch-bytes" (* 1024 1024))]
           (when (and (pos? (long n)) (MlxRuntime/canPrefetch)) (long n)))))

(defn- host-segment
  "The data segment (bytes long) of the evaluated handle h, whose pointer
   p MLX gave: on CUDA, at least prefetch-bytes long, its pages moved to
   the host in one transfer first. MLX's CUDA allocator moves a buffer the
   host reads into managed memory, where the host's reads would fault its
   pages over one at a time (genclj-4acj)."
  ^MemorySegment [^MemorySegment p ^long bytes]
  (let [seg (data-segment p bytes)]
    (when-let [from @prefetch-bytes]
      (when (<= (long from) bytes)
        (MlxRuntime/prefetchToHost seg)))
    seg))

(defn- read-floats
  "The n float32 elements of the evaluated, row-contiguous handle h, as a
   float array (MlxRuntime/readFloats: a large read copied in parallel)."
  ^floats [^long h ^long n]
  (if (zero? n)
    (float-array 0)
    (MlxRuntime/readFloats (host-segment (Mlx/mlx_array_data_float32 h) (* 4 n)) (int n))))

(defn- read-halves
  "The n float16 or bfloat16 (dtype) elements of the evaluated,
   row-contiguous handle h, widened to floats on the host, exactly
   (gen.array.host/halves->floats's floats, bit for bit): their bit
   patterns read (mlx-c's uint16 accessor, which any 16-bit dtype's
   buffer answers; its float16 one is absent where its compiler lacks
   _Float16) and widened from the buffer in place by
   MlxRuntime/widenHalves (a large read in parallel; genclj-4acj)."
  ^floats [^long h ^long n dtype]
  (if (zero? n)
    (float-array 0)
    (MlxRuntime/widenHalves (host-segment (Mlx/mlx_array_data_uint16 h) (* 2 n)) (int n) (= :bfloat16 dtype))))

;; A float16 or bfloat16 array is read as its own bits, widened on the
;; host, rather than cast to float32 on the device first (genclj-9y6y). On
;; CUDA, MLX moves a buffer the host reads to unified (managed) memory and
;; later recycles it, by its size, for the device's next allocation of
;; that size: the next read's cast. Freed at once, a cast's buffer was
;; written by the device into pages the host had read and read back from
;; pages the device had written, each a storm of page faults: 26 ms a read
;; of Qwen3-0.6B's 16 rows on fenrir, 4.6 ms its float16 read alone.

(defn to-floats
  "x's values, row-major, as a new float array: a float32 array's buffer
   read in place, a float16 or bfloat16 array's widened on the host,
   another dtype's cast to float32 first. Evaluates x."
  ^floats [x]
  (let [^MlxArray x x
        b (.-backend x)
        dtype (.-dtype x)
        half? (#{:float16 :bfloat16} dtype)
        n (sh/size (.-shape x))
        s (stream b)]
    (with-temps [t]
      (try
        (let [h (handle x)
              h (if (or half? (= :float32 dtype)) h (t (Mlx/mlx_astype h (int (code :float32)) s)))]
          (evaluating
            (Mlx/mlx_array_eval h)
            (let [c (if (Mlx/_mlx_array_is_row_contiguous h)
                      h
                      (let [c (t (Mlx/mlx_contiguous h false s))]
                        (Mlx/mlx_array_eval c)
                        c))]
              (if half? (read-halves c n dtype) (read-floats c n)))))
        (catch MlxException e (throw (mlx-error :->floats [x] e)))
        (finally (fence x))))))

(defn eval-arrays! [xs]
  ;; an evaluation boundary is a good moment to free what became unreachable
  (MlxRuntime/drain)
  (let [hs (long-array (map #(handle %) xs))
        v (MlxRuntime/vectorOf hs)]
    (try
      (evaluating (Mlx/mlx_eval v))
      (catch MlxException e (throw (mlx-error :eval! xs e)))
      (finally
        (Mlx/mlx_vector_array_free v)
        (fence xs)))
    nil))
