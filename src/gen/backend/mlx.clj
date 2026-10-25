(ns gen.backend.mlx
  "The MLX backend (M2): arrays are MLX arrays, computed lazily by MLX on the
   CPU or the GPU (Metal; CUDA on Linux), through mlx-c over FFM. The Java
   binding (src-java/gen/mlx, generated from mlx-c's headers) is compiled with
   `clojure -T:build java`; the native libraries come from native/build-mlx.sh.

   - Arrays are MlxArray values owning an MLX handle (a long), freed once the
     value is unreachable (gen.mlx.MlxRuntime/adopt: phantom references
     drained by the threads making arrays). MLX's reference counting keeps
     any graph that still needs an array alive. The GC does not see MLX's
     memory, so the backend collects under pressure: when 100,000 arrays are
     live (gen.mlx.gc-arrays; Metal allows about 500,000 buffers) or MLX's
     active memory passes an eighth of RAM (gen.mlx.gc-bytes).
   - MLX's own limits are bounded too: its buffer cache (freed buffers kept
     for reuse) to a sixteenth of RAM, and its memory limit, past which it
     frees the cache and waits for running work before allocating, to half
     of RAM. MLX's defaults are all of RAM for both, which let one JVM take
     the whole machine (2026-10-19: a kernel panic from memory exhaustion).
     MLX's limit is no cap, though: past it MLX goes on allocating, without
     an error (genclj-zztc, 2026-10-20).
   - So the backend keeps a budget of its own (budget, set-budget!): MLX's
     active memory may reach the budget it shares with native engines on
     the same machine (an LLM engine's weights and caches, which they
     reserve in gen.array.memory), less those reservations and a headroom.
     After every evaluation the backend checks it; past it, it collects
     unreachable arrays and checks again, and then it is a loud error
     (:reason :mlx-budget-exceeded) with the numbers.
   - 0-d arrays are MLX arrays too (the JVM backend's are host values), so
     scalar code stays lazy. Reading one on the host (item, truthy?) evaluates.
   - Ops run on the backend's stream, a stream MLX lets any thread use. One
     lock serializes evaluation, so the backend can be used from several
     threads.
   - Decompositions and solves run on the CPU stream (MLX implements them only
     there).
   - Keys are gen.random's (JAX's partitionable Threefry), computed with MLX's
     uint32 ops, so a key draws the same uniforms as on the JVM backend.
   - Gradients use the backend-generic tape (gen.array.tape), or MLX's own
     with gen.mlx.grad=native (docs/array-api.md §8).

   Configuration (system properties, read when the namespace loads):
     gen.mlx.device  cpu or gpu (default gpu when MLX has one)
     gen.mlx.dtypes  32 (default) or 64: the default dtypes, float32/int32 or
                     float64/int64 (64 needs the CPU device; MLX's GPUs have no
                     float64)
     gen.mlx.grad    tape (default) or native
     gen.mlx.cache-bytes   MLX's buffer cache (default a sixteenth of RAM)
     gen.mlx.memory-bytes  MLX's memory limit (default half of RAM)
     gen.mlx.budget-bytes  the budget MLX shares with native engines (default
                           half of RAM)
     gen.mlx.budget-headroom-bytes  kept free of both (default a sixteenth
                           of RAM)
     gen.mlx.gc-bytes, gen.mlx.gc-arrays  when the backend collects (above)
     gen.mlx.gc-new-bytes  collect after this many bytes of new arrays
                           (default a sixteenth of RAM)
     gen.mlx.constants     host-scalar constants kept per dtype (default
                           4096; each holds one buffer)
     gen.mlx.cpu-compile   true or false: skip the probe of compile on the
                           CPU device (cpu-compiles?)
     gen.mlx.lib           the directory of libmlxc (default
                           native/build/<platform>/lib; MlxRuntime)"
  {:api :internal}
  (:require [gen.array.dtype :as dt]
            [gen.array.memory :as reservations]
            [gen.array.protocols :as p]
            [gen.array.shape :as sh]
            [gen.array.tree :as tree]
            [gen.random.threefry])
  (:import [gen.mlx Mlx MlxRaw MlxRuntime MlxException]
           [gen.random.threefry Key]
           [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.lang.ref Reference]))

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

(defn- check-dtype
  "Throws unless the backend supports dtype."
  [dtype]
  (code dtype))

(def ^:private ^:const uint64-code
  "MLX's code for uint64, which only the float64 sampler's bit arithmetic
   uses: not a dtype of the backend's (dtype-codes)."
  4)

(defn- dtype-of-code [^long c]
  (or (code-dtypes c)
      (throw (ex-info (str "MLX produced a dtype gen.clj does not use (code " c ")") {:code c}))))

;; ---------------------------------------------------------------------------
;; Arrays
;; ---------------------------------------------------------------------------

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
  "The MlxArray owning the new handle h (freed once it is unreachable:
   MlxRuntime/adopt)."
  [b ^long h]
  (let [x (MlxArray. h b (read-shape h) (dtype-of-code (Mlx/mlx_array_dtype h)))]
    (MlxRuntime/adopt x h)
    x))

(defn- wrap-all
  "The MlxArrays owning the new handles hs (a long[], as
   MlxRuntime/vectorArrays gives them)."
  [b ^longs hs]
  (mapv #(wrap b %) hs))

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

(defn- with-device
  "(f d) for a new handle d of the :cpu or :gpu device, freed after: mlx-c
   allocates each one, and a stream made from it keeps its own copy."
  [device f]
  (let [d (Mlx/mlx_device_new_type (int (case device :cpu 0 :gpu 1)) (int 0))]
    (try (f d) (finally (Mlx/mlx_device_free d)))))

(defonce ^:private aux-stream
  ;; for conversions made where no backend is at hand
  (with-device :cpu #(Mlx/mlx_stream_new_thread_unsafe (long %))))

(def ^:private eval-lock (Object.))

(def ^:private one-eval-thread?
  "Do evaluations need one thread? MLX's CUDA backend fails when a stream
   first evaluated on one thread is evaluated on another (\"invalid resource
   handle\" in cudaGraphAddKernelNode, and the stream stays broken: the
   Jetson Thor, 2026-10-19), so where MLX has a CUDA GPU (Linux with a
   GPU), every evaluation runs on one thread. Metal and the CPU device let
   any thread evaluate (the CPU device on Linux too)."
  (delay (and (not (.contains (.toLowerCase (System/getProperty "os.name")) "mac"))
              (with-device :gpu #(Mlx/mlx_device_is_available (long %))))))

(defonce ^:private eval-thread
  (delay
    (let [t (promise)
          ex (java.util.concurrent.Executors/newSingleThreadExecutor
              (reify java.util.concurrent.ThreadFactory
                (newThread [_ r]
                  (doto (Thread. ^Runnable r "gen.mlx evaluation")
                    (.setDaemon true)
                    (->> (deliver t))))))]
      ;; start it, so its identity is known
      (.get (.submit ex ^Runnable (fn [])))
      {:executor ex :thread @t})))

;; ---------------------------------------------------------------------------
;; The budget (budget and set-budget! below)
;; ---------------------------------------------------------------------------

(defn- physical-memory ^long []
  (.getTotalMemorySize ^com.sun.management.OperatingSystemMXBean
                       (java.lang.management.ManagementFactory/getOperatingSystemMXBean)))

(defonce ^:private budget-limits
  (atom {:bytes (Long/getLong "gen.mlx.budget-bytes" (quot (physical-memory) 2))
         :headroom (Long/getLong "gen.mlx.budget-headroom-bytes" (quot (physical-memory) 16))}))

(defn- mlx-budget
  "The bytes of active memory MLX's arrays may hold: the budget less the
   native engines' reservations and the headroom."
  ^long []
  (let [{:keys [^long bytes ^long headroom]} @budget-limits]
    (max 0 (- bytes (reservations/reserved) headroom))))

(declare collect! budget)

(defn- check-budget!
  "MLX's active memory within its budget, after an evaluation: past it,
   collect the arrays that became unreachable and check again; still past
   it, a loud error."
  []
  (when (> (Mlx/mlx_get_active_memory) (mlx-budget))
    (collect!)
    (let [active (Mlx/mlx_get_active_memory)
          {:keys [mlx] :as b} (budget)]
      (when (> active (long mlx))
        (throw (ex-info (format (str "MLX holds %.1f MB, over its budget of %.1f MB (%.1f MB shared with native"
                                     " engines, less %.1f MB they reserved and %.1f MB of headroom): drop arrays,"
                                     " or raise the budget (set-budget!, gen.mlx.budget-bytes)")
                                (/ active 1e6) (/ (long mlx) 1e6) (/ (long (:bytes b)) 1e6)
                                (/ (long (:reserved b)) 1e6) (/ (long (:headroom b)) 1e6))
                        (assoc b :reason :mlx-budget-exceeded :active active
                               :reservations (reservations/reservations-held))))))))

;; ---------------------------------------------------------------------------
;; Metal's buffer limit (genclj-9tyh, DESIGN §9)
;; ---------------------------------------------------------------------------

(defn- buffer-limit?
  "Is e MLX's error at Metal's buffer limit ([metal::malloc] Resource limit
   exceeded)?"
  [e]
  (and (instance? MlxException e) (.contains (str (ex-message e)) "Resource limit")))

(defn- at-buffer-limit
  "(f), which makes buffers: an array from host data, or an evaluation.
   MLX on macOS makes every buffer through Metal, which allows about
   499,000, and it recovers fully once buffers are freed. So at the limit
   the arrays that became unreachable are collected (whatever the
   thresholds), MLX's buffer cache is cleared, and f runs once more; at the
   limit again, the buffers are held, and the error is loud."
  [what f]
  (let [attempt #(try [(f)]
                      (catch MlxException e (if (buffer-limit? e) e (throw e))))
        r (attempt)
        r (if (vector? r) r (do (MlxRuntime/collectNow) (Mlx/mlx_clear_cache) (attempt)))]
    (if (vector? r)
      (first r)
      (let [live (aget (MlxRuntime/lifetimeStats) 0)]
        (throw (ex-info (str what ": Metal's buffer limit, with " live " MLX arrays live after a collection:"
                             " eval! as you go, and drop graphs and arrays no longer needed")
                        {:reason :mlx-buffer-limit :op what :live-arrays live :cap (MlxRuntime/arrayCap)}
                        r))))))

(defn- evaluating*
  "(f) holding the evaluation lock; where evaluations need one thread
   (one-eval-thread?), on that thread, with the caller's bindings; once
   more after a collection at Metal's buffer limit (at-buffer-limit). Then
   MLX's active memory is checked against the budget (check-budget!)."
  [f]
  (let [run (fn []
              (if (or (not @one-eval-thread?) (identical? (Thread/currentThread) (:thread @eval-thread)))
                (locking eval-lock (f))
                (let [task (.submit ^java.util.concurrent.ExecutorService (:executor @eval-thread)
                                    ^java.util.concurrent.Callable (bound-fn* #(locking eval-lock (f))))]
                  (try (.get task)
                       (catch java.util.concurrent.ExecutionException e (throw (.getCause e)))))))
        v (at-buffer-limit "evaluation" run)]
    (check-budget!)
    v))

(defmacro ^:private evaluating
  "Run body (which evaluates MLX arrays) holding the evaluation lock, on the
   evaluation thread where there is one (evaluating*)."
  [& body]
  `(evaluating* (fn [] ~@body)))

;; ---------------------------------------------------------------------------
;; Operands: MLX arrays and host scalars
;; ---------------------------------------------------------------------------

(defn- host? [x] (or (instance? Number x) (instance? Boolean x)))

(defn- made
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

(defn- scalar-handle
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

(defn- host-dtype-on
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

(defn- constant
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

(defn- free-temp
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

(defn- temp!
  "Register the temporary x (as free-temp takes it) with temps; its handle.
   A kind free-temp does not know fails here, where it is written, not in
   with-temps' finally, where it would mask the body's error."
  ^long [^java.util.ArrayList temps x]
  (when (and (vector? x) (not (temp-kinds (first x))))
    (throw (ex-info (str "Not a kind of temporary: " (pr-str (first x))) {:kind (first x) :kinds temp-kinds})))
  (.add temps x)
  (long (if (vector? x) (second x) x)))

(defmacro ^:private with-temps
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

(defn- operand
  "The handle of operand x: x's own, or the shared constant for a host scalar
   (as dtype, or b's default for it). When the constant cache is full, a new
   handle, returned negated: release frees it. (Handles are user-space
   addresses, positive as longs.)"
  ^long [b x dtype]
  (if (mlx-array? x)
    (.-h ^MlxArray x)
    (let [d (or dtype (host-dtype-on b x))]
      (if-let [h (constant x d)]
        (long h)
        (- (scalar-handle x d))))))

(defn- as-array
  "x as an MlxArray of b (a host scalar as dtype, or b's default)."
  [b x dtype]
  (if (mlx-array? x) x (wrap b (scalar-handle x (or dtype (host-dtype-on b x))))))

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

(defn- retained
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

(declare device)

(defn- gpu-dtype!
  "dtype, unless it is float64 on the GPU device, which MLX's GPUs do not
   have: an array made there from host data would read back but fail at its
   first op, so it fails here."
  [b dtype]
  (when (and (= :float64 dtype) (= :gpu (device b)))
    (throw (ex-info "MLX's GPUs have no float64: make float64 arrays on a CPU-device backend (:dtypes 64)"
                    {:dtype dtype :device :gpu})))
  dtype)

(defn- from-host
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
          (let [h (.-h x)
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

(defn- eval-arrays! [xs]
  ;; an evaluation boundary is a good moment to free what became unreachable
  (MlxRuntime/drain)
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
   :sinh (mlx-fn sinh 1) :cosh (mlx-fn cosh 1)
   :arcsin (mlx-fn arcsin 1) :arccos (mlx-fn arccos 1) :arctan (mlx-fn arctan 1)
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
   :logical-and (mlx-fn logical_and 2) :logical-or (mlx-fn logical_or 2) :logaddexp (mlx-fn logaddexp 2)
   :arctan2 (mlx-fn arctan2 2)})

(declare exact-route broadcast-to)

(defn- unary [b op x dtype]
  (cond
    ;; MLX's kernels that are not exact take exact routes (gen.backend.mlx.exact,
    ;; loaded on first use: it is written in gen.array, which loads this namespace)
    (and (= dtype :float64) (#{:exp :sin :cos :erf :erfinv} op))
    (exact-route b op [(as-array b x dtype)])

    (and (= op :expm1) (= :gpu (device b)) (#{:float32 :float16 :bfloat16} dtype))
    (exact-route b :expm1 [(as-array b x dtype)])

    (and (#{:floor :ceil :round} op) (not= :float (dt/kind dtype)))
    ;; rounding an int is the identity
    (as-array b x dtype)

    :else
    (let [f (unary-fns op)
          s (stream b)]
      (with-operands b op [h x dtype]
        (wrap b (cond
                  (and (= op :sign) (= :float (dt/kind dtype)))
                  ;; MLX's sign of NaN is 0; NumPy's (and the JVM backend's) NaN
                  (with-temps [t]
                    (Mlx/mlx_where (t (Mlx/mlx_isnan h s)) h (t (f h s)) s))

                  (and (= op :arctan) (not= dtype :float64))
                  ;; Metal's float32 atan(-0.0) is +0.0: x itself at zero (where
                  ;; not x), as NumPy and the JVM backend; float64's keeps the sign
                  (with-temps [t]
                    (Mlx/mlx_where (t (Mlx/mlx_logical_not h s)) h (t (f h s)) s))

                  :else (f h s)))))))

(defn- int-division
  "remainder or floor-divide of int handles, as the JVM backend (and NumPy):
   0 for a zero divisor, since integer division by zero is undefined in
   MLX's kernels; and floor division rounding down, which MLX's (truncating)
   floor_divide does not: (x - x mod y) / y, an exact division. A -1
   divisor of a signed dtype never reaches the kernels either: MIN / -1
   overflows, which x86's idiv traps (SIGFPE, a JVM crash on MLX's CPU,
   genclj-kw88). x mod -1 is 0 and x floor-div -1 is -x, wrapping at MIN as
   Java's floorDiv."
  [b op hx hy dtype]
  (let [hx (long hx) hy (long hy)
        s (stream b)
        signed? (not= dtype :uint32)]
    (with-temps [t]
      (let [zero (t (scalar-handle 0 dtype))
            one (t (scalar-handle 1 dtype))
            z (t (Mlx/mlx_equal hy zero s))
            m1 (when signed? (t (Mlx/mlx_equal hy (t (scalar-handle -1 dtype)) s)))
            safe (t (Mlx/mlx_where (if m1 (t (Mlx/mlx_logical_or z m1 s)) z) one hy s))
            r (t (Mlx/mlx_remainder hx safe s))
            q (if (= op :remainder)
                r
                (let [q (t (Mlx/mlx_floor_divide (t (Mlx/mlx_subtract hx r s)) safe s))]
                  (if m1 (t (Mlx/mlx_where m1 (t (Mlx/mlx_negative q s)) q s)) q)))]
        (Mlx/mlx_where z zero q s)))))

(defn- java-int-cast
  "Float handle h (of dtype from) cast to the int dtype as Java casts
   (DESIGN.md §9): truncated toward zero, saturating at the dtype's range
   (the infinities too), NaN 0. MLX's own cast is C++'s, undefined out of
   range: MIN for all of these on x86's CPU, MIN for an int64 NaN on the GPU.
   The bounds are powers of two, exact in every float dtype (or infinite,
   in float16, whose finite values all fit)."
  [b ^long h from dtype]
  (let [s (stream b)
        [lo hi lo-int hi-int] (case dtype
                                :int32 [-2147483648.0 2147483648.0 Integer/MIN_VALUE Integer/MAX_VALUE]
                                :int64 [-9.223372036854775808E18 9.223372036854775808E18 Long/MIN_VALUE Long/MAX_VALUE]
                                :uint32 [0.0 4294967296.0 0 4294967295])]
    (with-temps [t]
      (let [k (fn ^long [v d] (t (scalar-handle v d)))
            over (t (Mlx/mlx_greater_equal h (k hi from) s))
            under (t (Mlx/mlx_less h (k lo from) s))
            bad (t (Mlx/mlx_logical_or (t (Mlx/mlx_logical_or over under s)) (t (Mlx/mlx_isnan h s)) s))
            cast (t (Mlx/mlx_astype (t (Mlx/mlx_where bad (k 0.0 from) h s)) (int (code dtype)) s))]
        (Mlx/mlx_where over (k hi-int dtype) (t (Mlx/mlx_where under (k lo-int dtype) cast s)) s)))))

(defn- int-power
  "x^y of int handles, as the JVM backend (and host math): exact for y >= 0;
   for y < 0 through floats, cast as Java casts (truncated, saturating at
   the ±infinities, 0 for NaN): 1 and -1 give ±1, larger |x| give 0, and 0
   gives the int64 maximum. (MLX's integer power gives 0 for every negative
   exponent, and for every element of an array with one.)"
  [b hx hy dtype]
  (let [hx (long hx) hy (long hy)
        s (stream b)
        fdt (int (code (:float (defaults b))))
        i64 (int (code :int64))]
    (with-temps [t]
      ;; MLX's integer power gives 0 in every element once any exponent is
      ;; negative: the exact path sees only non-negative stand-ins
      (let [k (fn ^long [v d] (t (scalar-handle v d)))
            neg (t (Mlx/mlx_less hy (k 0 dtype) s))
            exact (t (Mlx/mlx_power hx (t (Mlx/mlx_where neg (k 0 dtype) hy s)) s))
            f (t (Mlx/mlx_power (t (Mlx/mlx_astype hx fdt s)) (t (Mlx/mlx_astype hy fdt s)) s))
            fin (t (Mlx/mlx_isinf f s))
            finite (t (Mlx/mlx_where (t (Mlx/mlx_logical_or fin (t (Mlx/mlx_isnan f s)) s))
                                     (k 0.0 (:float (defaults b))) f s))
            cast (t (Mlx/mlx_astype finite i64 s))
            pos-inf (t (Mlx/mlx_logical_and fin (t (Mlx/mlx_greater f (k 0.0 (:float (defaults b))) s)) s))
            neg-inf (t (Mlx/mlx_logical_and fin (t (Mlx/mlx_less f (k 0.0 (:float (defaults b))) s)) s))
            java (t (Mlx/mlx_where pos-inf (k Long/MAX_VALUE :int64)
                                   (t (Mlx/mlx_where neg-inf (k Long/MIN_VALUE :int64) cast s)) s))
            java (if (= dtype :int64) java (t (Mlx/mlx_astype java (int (code dtype)) s)))]
        (Mlx/mlx_where neg java exact s)))))

(defn- binary [b op x y dtype]
  (let [sx (shape-of x) sy (shape-of y)]
    (sh/broadcast sx sy)
    (if (and (= op :logaddexp) (= dtype :float64))
      ;; MLX's float64 exp is not exact (see unary)
      ;; broadcast first: a custom VJP's cotangents have its inputs' shapes
      (let [shape (sh/broadcast sx sy)]
        (exact-route b :logaddexp [(broadcast-to b (as-array b x dtype) shape) (broadcast-to b (as-array b y dtype) shape)]))
    (let [s (stream b)]
      (with-operands b op [hx x dtype hy y dtype]
        (wrap b (cond (and (#{:remainder :floor-divide} op) (= :int (dt/kind dtype)))
                      (int-division b op hx hy dtype)
                      (and (= op :power) (= :int (dt/kind dtype)))
                      (int-power b hx hy dtype)
                      :else ((binary-fns op) hx hy s))))))))

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
        int-code (int (code (:int (defaults b))))]
    (with-temps [t]
      (let [i (t (if (= op :argmax) (Mlx/mlx_argmax_axis h ax kd s) (Mlx/mlx_argmin_axis h ax kd s)))
            i (if (= :float (dt/kind dtype))
                (let [nan (t (Mlx/mlx_isnan h s))
                      first-nan (t (Mlx/mlx_argmax_axis nan ax kd s))
                      any-nan (t (Mlx/mlx_any_axis nan ax kd s))]
                  (t (Mlx/mlx_where any-nan first-nan i s)))
                i)]
        ;; MLX's indices are uint32
        (Mlx/mlx_astype i int-code s)))))

(def ^:private exact-fns
  ;; gen.backend.mlx.exact's routes, resolved on first use, once: they are
  ;; written in gen.array, which loads this namespace
  (delay {:blocked-above (long @(requiring-resolve 'gen.backend.mlx.exact/blocked-above))
          :sum (requiring-resolve 'gen.backend.mlx.exact/sum)
          :half-sum (requiring-resolve 'gen.backend.mlx.exact/half-sum)
          :route (requiring-resolve 'gen.backend.mlx.exact/route)
          :vjp (requiring-resolve 'gen.backend.mlx.exact/vjp)}))

(def ^:private tape-fns
  ;; gen.array.tape's, resolved once, as exact-fns
  (delay {:value-and-grad (requiring-resolve 'gen.array.tape/value-and-grad)
          :linalg-vjp (requiring-resolve 'gen.array.tape/linalg-vjp)}))

(defn- blocked-sum?
  "Does a sum of x over axes take gen.backend.mlx.exact's blocks: float32
   on the CPU, of more terms than exact/blocked-above (genclj-y0mp)?"
  [b x dtype axes]
  (and (= dtype :float32) (= :cpu (device b))
       (> (sh/size (mapv #(nth (shape-of x) %) axes)) (long (:blocked-above @exact-fns)))))

(defn- reduce-op [b op x axes keep-dims?]
  (let [s (stream b)
        dtype (dtype-on b x)]
    (cond
      (empty? axes)
      ;; over no axis: the identity (argmax/argmin always name one axis)
      (as-array b x dtype)

      (and (= op :sum) (blocked-sum? b x dtype axes))
      ;; MLX's CPU accumulates a float32 sum sequentially (DESIGN §9)
      ((:sum @exact-fns) (as-array b x dtype) axes keep-dims?)

      (and (= op :sum) (#{:float16 :bfloat16} dtype) (= :cpu (device b)))
      ;; and a half-float sum in its own precision: in float32 instead
      ((:half-sum @exact-fns) (as-array b x dtype) axes keep-dims?)

      :else
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
  (with-operands b :broadcast-to [h x nil]
    (wrap b (Mlx/mlx_broadcast_to h (int-array (map int shape)) (stream b)))))

(defn- concatenate [b xs axis]
  (let [d (dtype-on b (first xs))]
    (with-temps [t]
      (try
        (let [hs (mapv #(let [raw (operand b % d)] (if (neg? raw) (t (- raw)) raw)) xs)
              v (t :vector (MlxRuntime/vectorOf (long-array hs)))]
          (wrap b (Mlx/mlx_concatenate_axis v (int axis) (stream b))))
        (catch MlxException e (throw (mlx-error :concatenate xs e)))
        (finally (fence xs))))))

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
  (let [s (stream b)]
    (with-temps [t]
      (let [;; int32 indices, whatever the int dtype
            i32 (t (Mlx/mlx_astype h (code :int32) s))
            ;; through made, as every array made from host data: at Metal's
            ;; buffer limit a raw constructor gives an empty handle, which
            ;; the next op rejects (genclj-j9nx)
            zero (t (scalar-handle 0 :int32))
            dd (t (scalar-handle d :int32))
            top (t (scalar-handle (dec d) :int32))
            neg (t (Mlx/mlx_less i32 zero s))
            wrapped (t (Mlx/mlx_add i32 dd s))
            n (t (Mlx/mlx_where neg wrapped i32 s))]
        (Mlx/mlx_clip n zero top s)))))

(defn- take-op [b x indices axis]
  (let [d (long (nth (shape-of x) axis))]
    (if (host? indices)
      (let [i (host-index indices d)]
        (with-operands b :take [h x nil hi i :int32]
          (wrap b (Mlx/mlx_take_axis h hi (int axis) (stream b)))))
      (with-operands b :take [h x nil hi indices nil]
        (with-temps [t]
          (wrap b (Mlx/mlx_take_axis h (t (safe-indices b hi d)) (int axis) (stream b))))))))

(defn- take-along-axis [b x indices axis]
  (with-operands b :take-along-axis [h x nil hi indices nil]
    (with-temps [t]
      (wrap b (Mlx/mlx_take_along_axis h (t (safe-indices b hi (nth (shape-of x) axis))) (int axis) (stream b))))))

(defn- scatter-add [b x indices updates axis]
  (let [d (dtype-on b x)]
    (with-operands b :scatter-add [h x d hi indices nil hu updates d]
      (with-temps [t]
        (wrap b (Mlx/mlx_scatter_add_axis h (t (safe-indices b hi (nth (shape-of x) axis))) hu (int axis) (stream b)))))))

;; ---------------------------------------------------------------------------
;; Linear algebra
;; ---------------------------------------------------------------------------

(defn- matmul [b x y]
  (let [d (dtype-on b x)
        s (stream b)]
    (with-operands b :matmul [hx x d hy y d]
      (if (= :float (dt/kind d))
        (wrap b (Mlx/mlx_matmul hx hy s))
        ;; MLX multiplies floats only: ints go through the widest float the
        ;; device has (exact while products stay below 2^24 or 2^53)
        (with-temps [t]
          (let [f (code (:float (defaults b)))]
            (wrap b (Mlx/mlx_astype (t (Mlx/mlx_matmul (t (Mlx/mlx_astype hx f s)) (t (Mlx/mlx_astype hy f s)) s))
                                    (code d) s))))))))

(defn- batch-broadcast
  "Handles of a [.. n n] and rhs [.. n k] broadcast to one batch shape (MLX's
   solves want equal batch axes; gen.array's broadcast them), new ones
   registered with t (with-temps)."
  [b ha sa hr sr t]
  (let [ha (long ha) hr (long hr)
        ba (subvec sa 0 (- (count sa) 2))
        br (subvec sr 0 (- (count sr) 2))
        batch (sh/broadcast ba br)
        s (cpu-stream b)
        to (fn [^long h shape tail]
             (if (= shape (into batch tail))
               h
               (long (t (Mlx/mlx_broadcast_to h (int-array (map int (into batch tail))) s)))))]
    [(to ha sa (subvec sa (- (count sa) 2))) (to hr sr (subvec sr (- (count sr) 2)))]))

(defn- diag-of-temp
  "The diagonals of the matrices h, [.. n], a handle registered with t."
  ^long [t ^long h s]
  (long (t (Mlx/mlx_diagonal h (int 0) (int -2) (int -1) (long s)))))

(defn- linalg* [b op args]
  (let [cpu (cpu-stream b)]
    (with-temps [t]
      (let [nan (fn ^long [dtype] (t (scalar-handle ##NaN dtype)))]
        (case op
          ;; MLX returns a partial factor for a matrix that is not positive
          ;; definite; LAPACK leaves the failing pivot non-positive there, so the
          ;; contract's NaN is where a diagonal entry is not positive
          :cholesky
          (let [[a] args
                d (dtype-on b a)]
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
            (with-operands b :solve-triangular [ha a nil hr rhs nil]
              (let [[ha hr] (batch-broadcast b ha sa hr (shape-of rhs) t)
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
            (with-operands b :solve [ha a nil hr rhs nil]
              (let [[ha hr] (batch-broadcast b ha sa hr (shape-of rhs) t)
                    n (int (peek sa))
                    [lu _] (mapv t (Mlx/mlx_linalg_lu_factor ha cpu))
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
            (with-operands b :slogdet [h a nil]
              (let [[lu piv] (mapv t (Mlx/mlx_linalg_lu_factor h cpu))
                    u (diag-of-temp t lu cpu)
                    n (peek sa)
                    ;; the results wrapped as they are made: the Cleaner owns
                    ;; logabs while the handles after it are made
                    logabs (wrap b (Mlx/mlx_sum_axis (t (Mlx/mlx_log (t (Mlx/mlx_abs u cpu)) cpu)) (int -1) false cpu))
                    idx (t (Mlx/mlx_arange 0.0 (double n) 1.0 (int (code :uint32)) cpu))
                    swaps (t (Mlx/mlx_sum_axis (t (Mlx/mlx_astype (t (Mlx/mlx_not_equal piv idx cpu)) (int (code :int32)) cpu))
                                               (int -1) false cpu))
                    odd (t (Mlx/mlx_remainder swaps (t (scalar-handle 2 :int32)) cpu))
                    perm (t (Mlx/mlx_subtract (t (scalar-handle 1 d))
                                              (t (Mlx/mlx_multiply (t (scalar-handle 2 d)) (t (Mlx/mlx_astype odd (int (code d)) cpu)) cpu))
                                              cpu))
                    sign (wrap b (Mlx/mlx_multiply (t (Mlx/mlx_prod_axis (t (Mlx/mlx_sign u cpu)) (int -1) false cpu)) perm cpu))]
                [sign logabs]))))))))

;; MLX has no VJP for its Cholesky, LU or inverse (GenMLX's fork patched
;; some), so in native mode each linear-algebra op is one mlx_custom_vjp
;; whose VJP is the tape's rule (gen.array.tape/linalg-vjp): no MLX patch
(declare grad-mode with-default-stream)

(def ^:private ^:dynamic *tracing-compile*
  "True while mlx_compile traces a function (compile-fn). Custom VJPs stay
   out of compiled graphs: MLX keeps those in a thread-local cache, and its
   destruction at a thread's exit releases the custom functions' Java
   closures by an upcall from the exiting thread, which crashes the JVM
   (2026-10-19, SIGILL at exit)."
  false)

(defn- custom-vjps? [] (and (= grad-mode :native) (not *tracing-compile*)))

(def ^:private ^:dynamic *differentiating*
  "True while native-value-and-grad runs its function: a custom VJP left out
   of a compiled graph there would give a wrong gradient."
  false)

(def ^:private default-backend-var (delay (requiring-resolve 'gen.array/*backend*)))

(defn- custom-vjp
  "The outputs of (forward ins), with MLX's gradient through them given by
   (vjp primals cotangents outputs); ins and every result are vectors of
   MlxArrays. Both run with b as gen.array's default backend, as the tape's
   backward pass does: MLX calls them back with whatever its caller had, and
   the rules make arrays (eye, arange)."
  [b forward vjp ins]
  (let [handles (fn [xs] (long-array (map #(retained b :custom-vjp %) xs)))
        on-b (fn [f] (with-bindings {@default-backend-var b} (f)))]
    (with-temps [t]
      (try
        (let [fun (t :closure (MlxRuntime/closure (reify gen.mlx.MlxRuntime$Closure
                                                    (apply [_ hs] (handles (on-b #(forward (wrap-all b hs))))))))
              bwd (t :custom-vjp (MlxRuntime/customVjp (reify gen.mlx.MlxRuntime$CustomVjp
                                                         (apply [_ ps cs os] (handles (on-b #(vjp (wrap-all b ps) (wrap-all b cs) (wrap-all b os))))))))
              cv (t :closure (Mlx/mlx_custom_vjp fun bwd))
              in (t :vector (MlxRuntime/vectorOf (long-array (map #(.-h ^MlxArray %) ins))))
              ;; MLX makes the custom function's node on the default device
              out (t :vector (with-default-stream b #(Mlx/mlx_closure_apply cv in)))]
          (wrap-all b (MlxRuntime/vectorArrays out)))
        (finally (fence ins))))))

(defn- exact-route
  "op of the MlxArrays xs by its exact composite (gen.backend.mlx.exact).
   In native mode it is one custom VJP with the function's derivative, not
   the composite's graph (branches and Newton steps) differentiated."
  [b op xs]
  (let [{:keys [route vjp]} @exact-fns]
    (if-not (custom-vjps?)
      (route op xs)
      (first (custom-vjp b
                         (fn [ys] [(route op ys)])
                         (fn [ps cs os] (vjp op ps (first cs) (first os)))
                         xs)))))

(defn- linalg [b op args]
  (if-not (custom-vjps?)
    (linalg* b op args)
    (let [n-arrays (if (#{:cholesky :slogdet} op) 1 2)
          opts (drop n-arrays args)
          rule (:linalg-vjp @tape-fns)
          outs (custom-vjp b
                           (fn [xs] (let [r (linalg* b op (into xs opts))] (if (vector? r) r [r])))
                           (fn [ps cs os]
                             (let [[out g] (if (= op :slogdet) [(second os) (second cs)] [(first os) (first cs)])]
                               (rule b op (into ps opts) out g (vec (repeat n-arrays true)))))
                           (mapv #(as-array b % nil) (take n-arrays args)))]
      (if (= op :slogdet) outs (first outs)))))

;; ---------------------------------------------------------------------------
;; Randomness: JAX's partitionable Threefry-2x32 (gen.random.threefry) in
;; MLX's uint32 ops, so a key draws the same bits as on the JVM backend
;; ---------------------------------------------------------------------------

(defn- threefry-words
  "[w0 w1] handles of Threefry-2x32-20 of the counters [c0 c1] (uint32
   handles) under the key words k0 and k1 (uint32 handles of shape [1]), as
   gen.random.threefry/hash2x32. Everything else is a shared constant, so a
   pending draw pins only its key's buffer. t registers each handle made
   (with-temps)."
  [b k0 k1 c0 c1 t konst]
  (let [s (stream b)
        u (fn ^long [v] (long (konst v :uint32)))
        add (fn ^long [^long x ^long y] (t (Mlx/mlx_add x y s)))
        rotl (fn ^long [^long x r]
               (t (Mlx/mlx_bitwise_or (t (Mlx/mlx_left_shift x (u r) s))
                                      (t (Mlx/mlx_right_shift x (u (- 32 (long r))) s))
                                      s)))
        k2 (t (Mlx/mlx_bitwise_xor (t (Mlx/mlx_bitwise_xor k0 k1 s)) (u 0x1BD11BDA) s))
        ks [k0 k1 k2]
        rounds (fn [[x0 x1] rs]
                 (reduce (fn [[x0 x1] r]
                           (let [x0 (add x0 x1)]
                             [x0 (t (Mlx/mlx_bitwise_xor (rotl x1 r) x0 s))]))
                         [x0 x1] rs))
        inject (fn [[x0 x1] i]
                 ;; after round group i: x0 += ks[i mod 3], x1 += ks[i+1 mod 3] + i
                 [(add x0 (nth ks (mod i 3)))
                  (add x1 (add (nth ks (mod (inc i) 3)) (u i)))])
        rs-a [13 15 26 6]
        rs-b [17 29 16 24]]
    (-> [(add c0 k0) (add c1 k1)]
        (rounds rs-a) (inject 1)
        (rounds rs-b) (inject 2)
        (rounds rs-a) (inject 3)
        (rounds rs-b) (inject 4)
        (rounds rs-a) (inject 5))))

(defn- random-uniform [b ^Key key shape dtype]
  (when-not (#{:float32 :float64} dtype)
    (throw (ex-info (str "random-uniform needs a float dtype, got " dtype) {:dtype dtype})))
  (let [s (stream b)
        n (sh/size shape)
        _ (when (>= n 0x100000000)
            (throw (ex-info "random-uniform: more than 2^32 elements" {:shape shape})))]
    (with-temps [t]
      (let [;; a shared constant, or a new handle (registered) when the cache is full
            konst (fn ^long [v dtype] (if-let [h (constant v dtype)] (long h) (t (scalar-handle v dtype))))
            ;; the key's two words, one buffer
            kw (t (with-open [arena (Arena/ofConfined)]
                    (let [words (.allocateFrom arena ValueLayout/JAVA_INT (int-array [(unchecked-int (.-k0 key)) (unchecked-int (.-k1 key))]))
                          shape (.allocateFrom arena ValueLayout/JAVA_INT (int-array [2]))]
                      (made "mlx_array_new_data" #(MlxRaw/mlx_array_new_data words shape (int 1) (int (code :uint32)))))))
            k0 (t (Mlx/mlx_slice kw (int-array [0]) (int-array [1]) (int-array [1]) s))
            k1 (t (Mlx/mlx_slice kw (int-array [1]) (int-array [2]) (int-array [1]) s))
            ;; counters [hi32(i) lo32(i)] = [0 i]
            c1 (t (Mlx/mlx_arange 0.0 (double n) 1.0 (int (code :uint32)) s))
            c0 (t (Mlx/mlx_zeros (int-array [n]) (int (code :uint32)) s))
            [w0 w1] (threefry-words b k0 k1 c0 c1 t konst)
            u (case dtype
                ;; 23 mantissa bits of w0 ^ w1, as gen.random.threefry/uniform-floats
                :float32 (let [bits (t (Mlx/mlx_bitwise_xor w0 w1 s))
                               m (t (Mlx/mlx_bitwise_or (t (Mlx/mlx_right_shift bits (konst 9 :uint32) s))
                                                        (konst 0x3F800000 :uint32) s))
                               f (t (Mlx/mlx_view m (int (code :float32)) s))]
                           (t (Mlx/mlx_subtract f (konst 1.0 :float32) s)))
                ;; 52 mantissa bits of w0 << 32 | w1, as uniform-doubles
                :float64 (let [u64 (int uint64-code)
                               hi (t (Mlx/mlx_astype w0 u64 s))
                               lo (t (Mlx/mlx_astype w1 u64 s))
                               c (fn ^long [v] (t (Mlx/mlx_astype (konst v :int64) u64 s)))
                               m (t (Mlx/mlx_bitwise_or
                                     (t (Mlx/mlx_bitwise_or (t (Mlx/mlx_left_shift hi (c 20) s))
                                                            (t (Mlx/mlx_right_shift lo (c 12) s)) s))
                                     (c 0x3FF0000000000000) s))
                               f (t (Mlx/mlx_view m (int (code :float64)) s))]
                           (t (Mlx/mlx_subtract f (konst 1.0 :float64) s))))]
        (wrap b (Mlx/mlx_reshape u (int-array (map int shape)) s))))))

;; ---------------------------------------------------------------------------
;; Gradients: MLX's own (mlx_value_and_grad), with f as an MLX closure of
;; the differentiated leaves
;; ---------------------------------------------------------------------------

(def ^:private grad-mode
  "How the backend differentiates: :tape (gen.array.tape, the default) or
   :native (mlx_value_and_grad); the system property gen.mlx.grad."
  (keyword (System/getProperty "gen.mlx.grad" "tape")))

(defn- with-default-stream
  "(f) with MLX's default device and the thread's default stream set to b's:
   MLX builds a gradient's backward pass there, not on the streams the
   backend passes its ops. The default device is global, so this holds the
   evaluation lock (reentrant: f may read values)."
  [b f]
  (evaluating
    (with-device (device b)
      (fn [dev]
        (let [dev (long dev)
              prev-dev (Mlx/mlx_get_default_device)]
          (try
            (let [prev-stream (Mlx/mlx_get_default_stream dev)]
              (try
                (Mlx/mlx_set_default_device dev)
                (Mlx/mlx_set_default_stream (stream b))
                (f)
                (finally
                  (Mlx/mlx_set_default_stream prev-stream)
                  (Mlx/mlx_set_default_device prev-dev)
                  (Mlx/mlx_stream_free prev-stream))))
            (finally (Mlx/mlx_device_free prev-dev))))))))

(defn- native-value-and-grad
  "[value grads] of f on args with respect to the positions argnums, by
   mlx_value_and_grad: MLX calls f back with its own arrays for the
   differentiated leaves, and differentiates the graph f builds."
  [b f args argnums]
  (let [args (vec args)
        trees (mapv #(nth args %) argnums)
        leaves (vec (tree/leaves trees))
        _ (doseq [l leaves]
            (when-not (and (or (host? l) (mlx-array? l)) (= :float (dt/kind (dtype-on b l))))
              (throw (ex-info (str "Cannot differentiate with respect to " (pr-str l)
                                   " (only float arrays and floating-point numbers)")
                              {:leaf l}))))
        xs (mapv #(as-array b % nil) leaves)
        run (fn [^longs ins]
              (let [ys (mapv #(wrap b %) ins)
                    args' (reduce (fn [as [i t]] (assoc as i t)) args (map vector argnums (tree/refill trees ys)))
                    out (apply f args')]
                (when-not (and (or (host? out) (mlx-array? out)) (= [] (shape-of out)))
                  (throw (ex-info "value-and-grad needs a function returning a scalar"
                                  {:shape (shape-of out) :value out})))
                (long-array [(retained b :value-and-grad out)])))]
    (if (empty? xs)
      ;; nothing to differentiate (MLX's grad needs an argument): the value,
      ;; with the (leafless) trees as gradients, as the tape gives
      [(apply f args) trees]
      (with-temps [t]
        (try
          (let [cl (t :closure (MlxRuntime/closure (reify gen.mlx.MlxRuntime$Closure (apply [_ ins] (run ins)))))
                vag (t :value-and-grad (Mlx/mlx_value_and_grad cl (int-array (range (count xs)))))
                in (t :vector (MlxRuntime/vectorOf (long-array (map #(.-h ^MlxArray %) xs))))
                [vs gs] (mapv #(t :vector %) (with-default-stream b #(Mlx/mlx_closure_value_and_grad_apply vag in)))
                v (first (wrap-all b (MlxRuntime/vectorArrays vs)))
                gs (wrap-all b (MlxRuntime/vectorArrays gs))]
            [v (tree/refill trees gs)])
          (finally (fence xs)))))))

;; ---------------------------------------------------------------------------
;; The backend
;; ---------------------------------------------------------------------------

(defn- compile-fn
  "p/-compile: f (a fn of a vector of MlxArrays to a vector of arrays) as an
   MLX closure, compiled with an id of its own; the returned fn applies it,
   from any thread. MLX traces it once per thread and input signature, and
   its traces are erased from every thread's compile cache when it is freed
   (MlxRuntime.Compiled, genclj-hfwo)."
  [b f {:keys [shapeless?]}]
  (let [run (fn [^longs ins]
              (let [outs (try (binding [*tracing-compile* true] (f (mapv #(wrap b %) ins)))
                              (catch clojure.lang.ExceptionInfo e
                                ;; MLX's words for it: Metal's and the CPU's, then CUDA's
                                (if (re-find #"function transformations|If you are compiling a function" (str (ex-message e)))
                                  (throw (ex-info (str "compile: the function reads an array's value on the host"
                                                       " (truthy?, item, ->clj), which a compiled function cannot:"
                                                       " branch with where, or do not compile it")
                                                  {:op :compile} e))
                                  (throw e))))]
                (doseq [o outs]
                  (when-not (or (mlx-array? o) (host? o))
                    (throw (ex-info (str "compile: the function's results must be arrays (or trees of them), not "
                                         (pr-str o))
                                    {:op :compile :value o}))))
                (long-array (map #(retained b :compile %) outs))))
        cl (MlxRuntime/closure (reify gen.mlx.MlxRuntime$Closure (apply [_ ins] (run ins))))
        ^gen.mlx.MlxRuntime$Compiled c
        (try (gen.mlx.MlxRuntime$Compiled. cl (boolean shapeless?))
             ;; the compiled closure holds the function for retracing
             (finally (MlxRaw/mlx_closure_free cl)))
        compiled (.-handle c)
        g (fn [xs]
            (let [in (MlxRuntime/vectorOf (long-array (map #(.-h ^MlxArray %) xs)))]
              (try
                ;; nothing between enter, the call and exit may block: a
                ;; virtual thread could move to another carrier, whose
                ;; cache is another
                (.enter c)
                (let [out (try (try (Mlx/mlx_closure_apply compiled in)
                                    (finally (.exit c)))
                               (catch MlxException e (throw (mlx-error :compile xs e))))]
                  (try (wrap-all b (MlxRuntime/vectorArrays out))
                       (finally (MlxRaw/mlx_vector_array_free out))))
                (finally
                  (MlxRaw/mlx_vector_array_free in)
                  (fence xs)))))]
    (MlxRuntime/adoptCompiled g c)
    g))

(def ^:private cpu-compile
  "Whether mlx_compile works on the CPU device: nil until the first compile
   there probes it (cpu-compiles?), or -Dgen.mlx.cpu-compile=true|false."
  (atom (some-> (System/getProperty "gen.mlx.cpu-compile") Boolean/parseBoolean)))

(defn- cpu-compiles?
  "Does mlx_compile work on the CPU device here? MLX's CPU compile builds each
   fused kernel at run time with the system's g++, from a header the wheel
   ships, and GCC 13 on Linux rejects that header (it redeclares the built-in
   _Float128: the Jetson Thor, 2026-10-19). The first compile on the CPU
   device compiles and runs a two-op function; where MLX cannot build it,
   compile is the identity on the CPU device (the same values, unfused), and
   a warning says so once."
  [b]
  (or @cpu-compile
      (locking cpu-compile
        (if (some? @cpu-compile)
          @cpu-compile
          (let [probe (compile-fn b (fn [[x]] [(elementwise b :exp [(elementwise b :negative [x] :float32)] :float32)]) nil)
                works? (try (to-host (first (probe [(from-host b [0.5 1.5] [2] :float32)])))
                            true
                            (catch Exception e
                              (if (re-find #"Failed to compile function" (str (ex-message e) " " (some-> (ex-cause e) ex-message)))
                                false
                                (throw e))))]
            (when-not works?
              (binding [*out* *err*]
                (println "gen.backend.mlx: MLX's CPU compile cannot build kernels with this system's g++;"
                         "array/compile runs functions unfused on the CPU device (docs/array-api.md §13)")))
            (reset! cpu-compile works?))))))

(defn compiles?
  "Does array/compile compile on MLX backend b: always on the GPU, and on the
   CPU device where MLX can build its kernels (see cpu-compiles?)?"
  [b]
  (or (not= :cpu (device b)) (cpu-compiles? b)))

(defmacro ^:private guarded
  "body, with an MLX error raised outside an op's own handling as ex-info
   naming op (gen.array's contract: errors are ex-info with data)."
  [op & body]
  `(try ~@body
        (catch MlxException e# (throw (mlx-error ~op [] e#)))))

(deftype MlxBackend [^Config config]
  p/ICustomVjp
  (-custom-vjp [b f vjp args]
    (when (and *tracing-compile* *differentiating*)
      (throw (ex-info (str "custom-vjp inside array/compile under native gradients: MLX's compiled graphs"
                           " leave custom VJPs out, so the gradient would be wrong; differentiate outside"
                           " compile, or use the tape")
                      {:op :custom-vjp})))
    (if-not (custom-vjps?)
      (apply f args)
      (first (custom-vjp b
                         (fn [xs] [(apply f xs)])
                         (fn [ps cs os]
                           (mapv (fn [ct p] (as-array b (or ct (p/-full b (p/-shape b p) 0 (p/-dtype b p))) nil))
                                 (vjp ps (first os) (first cs)) ps))
                         (mapv #(as-array b % nil) args)))))

  p/ICompiling
  (-compile [b f opts] (if (compiles? b) (guarded :compile (compile-fn b f opts)) f))

  p/INativeGrad
  (-native-grad? [_] (= grad-mode :native))

  p/IBackend
  (-name [_] :mlx)
  (-defaults [_] (.-defaults config))
  (-lazy? [_] true)
  (-native-ops [_] #{:logaddexp})

  (-from-host [b values shape dtype] (from-host b values shape dtype))
  (-to-host [_ x] (to-host x))
  (-eval! [_ xs] (eval-arrays! (filter mlx-array? xs)))

  (-shape [_ x] (shape-of x))
  (-dtype [b x] (dtype-on b x))

  (-full [b shape value dtype]
    (check-dtype dtype)
    (if (mlx-array? value)
      ;; an array value cast and broadcast, as the batch tracer fills, not
      ;; mlx_full: MLX's Full::vjp multiplies the cotangent by the value, so
      ;; MLX's own gradients scaled a fill's by it (genclj-hf5z;
      ;; docs/research/mlx-upstream/issue-1-full-vjp.md)
      (with-operands b :full [hv value nil]
        (with-temps [t]
          (let [hd (if (= dtype (dtype-on b value)) hv (t (Mlx/mlx_astype hv (int (code dtype)) (stream b))))]
            (wrap b (Mlx/mlx_broadcast_to hd (int-array (map int shape)) (stream b))))))
      (with-operands b :full [hv value dtype]
        (wrap b (Mlx/mlx_full (int-array (map int shape)) hv (int (code dtype)) (stream b))))))
  (-arange [b start stop step dtype]
    (guarded :arange
      (wrap b (Mlx/mlx_arange (double start) (double stop) (double step) (int (code dtype)) (stream b)))))
  (-astype [b x dtype]
    (with-operands b :astype [h x nil]
      (let [from (dtype-on b x)]
        (wrap b (if (and (= :float (dt/kind from)) (= :int (dt/kind dtype)))
                  (java-int-cast b h from dtype)
                  (Mlx/mlx_astype h (int (code dtype)) (stream b)))))))

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
  (-value-and-grad [b f args argnums]
    (guarded :value-and-grad
      (if (= grad-mode :native)
        (binding [*differentiating* true] (native-value-and-grad b f args argnums))
        ((:value-and-grad @tape-fns) b f args argnums)))))

(defn- config ^Config [b] (.-config ^MlxBackend b))

(defn gpu?
  "Does MLX have a GPU here (Metal, or CUDA on Linux)?"
  []
  (with-device :gpu #(Mlx/mlx_device_is_available (long %))))

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
    (vreset! p/lazy-backends? true)
    (MlxBackend.
     (->Config device
               (with-device device #(Mlx/mlx_stream_new_thread_unsafe (long %)))
               (with-device :cpu #(Mlx/mlx_stream_new_thread_unsafe (long %)))
               (case (long dtypes)
                 32 {:float :float32 :int :int32}
                 64 {:float :float64 :int :int64})))))

(defonce ^:private memory-bounds
  ;; once per process, before any array: set-cache-limit! and
  ;; set-memory-limit! change them later
  (let [ram (physical-memory)]
    (Mlx/mlx_set_cache_limit (long (Long/getLong "gen.mlx.cache-bytes" (quot ram 16))))
    (Mlx/mlx_set_memory_limit (long (Long/getLong "gen.mlx.memory-bytes" (quot ram 2))))
    true))

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
  "MLX's memory use in bytes (:active, :peak, :cache), its :limit, and the
   backend's :live-arrays and :collections (MlxRuntime/adopt)."
  []
  (let [[live collections] (vec (MlxRuntime/lifetimeStats))]
    {:active (Mlx/mlx_get_active_memory)
     :peak (Mlx/mlx_get_peak_memory)
     :cache (Mlx/mlx_get_cache_memory)
     :limit (Mlx/mlx_get_memory_limit)
     :live-arrays live
     :collections collections}))

(defn evaluated?
  "Does MLX hold the MLX array x's data, rather than the graph it is
   computed by? An unevaluated graph holds every array it reads, which
   neither the backend's collections nor MLX's memory counters see
   (DESIGN §9, genclj-9e9e)."
  [x]
  (try (Mlx/_mlx_array_is_available (.-h ^MlxArray x))
       (finally (fence x))))

(defn set-memory-limit!
  "Set MLX's memory limit in bytes; returns the previous limit. It is not a
   cap: near it MLX frees its buffer cache, and allocations past it still
   succeed (measured 2026-10-20 on an M2 Max: 8 GB active against a 4 GB
   limit, no error), so it cannot keep MLX within a memory budget."
  [bytes]
  (Mlx/mlx_set_memory_limit (long bytes)))

(defn set-cache-limit!
  "Set the size of MLX's buffer cache in bytes (0 disables caching); returns
   the previous limit."
  [bytes]
  (Mlx/mlx_set_cache_limit (long bytes)))

(defn clear-cache!
  "Free MLX's cached buffers."
  []
  (Mlx/mlx_clear_cache))

(defn collect!
  "Free every MLX array that is unreachable now, and return memory: the
   collection the backend makes at Metal's buffer limit (MlxRuntime's
   collectNow: a GC, a drain until the queue stays quiet, counted, and the
   thresholds reset), which the budget's check makes too."
  []
  (MlxRuntime/collectNow)
  (memory))

;; ---------------------------------------------------------------------------
;; The budget
;; ---------------------------------------------------------------------------

(defn budget
  "MLX's memory budget: :mlx, the bytes of active memory its arrays may
   hold, which is :bytes (the budget it shares with native engines on this
   machine) less :reserved (what they hold, gen.array.memory) and
   :headroom (kept free of both); and :active, what MLX holds now."
  []
  (let [{:keys [bytes headroom]} @budget-limits]
    {:mlx (mlx-budget)
     :bytes bytes
     :reserved (reservations/reserved)
     :headroom headroom
     :active (Mlx/mlx_get_active_memory)}))

(defn set-budget!
  "Set the budget MLX shares with native engines (:bytes), the headroom
   kept free of both (:headroom), or both: non-negative integers. Returns
   the previous {:bytes :headroom}."
  [limits]
  (doseq [[k v] (select-keys limits [:bytes :headroom])]
    (when-not (and (int? v) (not (neg? v)))
      (throw (ex-info (str "set-budget!: " k " is a non-negative integer, not " (pr-str v))
                      {:reason :invalid-option k v}))))
  (first (swap-vals! budget-limits merge (select-keys limits [:bytes :headroom]))))

(defmethod print-method MlxArray [^MlxArray x ^java.io.Writer w]
  (.write w "#gen/array[")
  (print-method (.-dtype x) w)
  (.write w " ")
  (print-method (to-host x) w)
  (.write w "]"))
