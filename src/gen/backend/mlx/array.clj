(ns gen.backend.mlx.array
  "The MLX backend's arrays (gen.backend.mlx): MLX's dtype codes, the
   MlxArray type (an MLX handle its value owns, freed once the value is
   unreachable), and the backend's configuration: its device, streams and
   default dtypes."
  {:api :internal}
  (:require [gen.array.host :as host]
            [gen.array.protocols :as p])
  (:import [gen.mlx Mlx MlxRuntime]
           [java.lang.foreign MemorySegment ValueLayout]
           [java.lang.ref Reference]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Dtypes
;; ---------------------------------------------------------------------------

(def ^:private dtype-codes
  {:bool 0 :uint32 3 :int32 7 :int64 8 :float16 9 :float32 10 :float64 11 :bfloat16 12})

(def ^:private code-dtypes (into {} (map (fn [[k v]] [v k])) dtype-codes))

(defn code
  "The MLX dtype code of dtype."
  ^long [dtype]
  (or (dtype-codes dtype)
      (throw (ex-info (str "The MLX backend does not support dtype " dtype)
                      {:dtype dtype :supported (set (keys dtype-codes))}))))

(defn check-dtype
  "Throws unless the backend supports dtype."
  [dtype]
  (code dtype))

(def ^:const uint64-code
  "MLX's code for uint64, which only the float64 sampler's bit arithmetic
   uses: not a dtype of the backend's (dtype-codes)."
  4)

(defn- dtype-of-code [^long c]
  (or (code-dtypes c)
      (throw (ex-info (str "MLX produced a dtype gen.clj does not use (code " c ")") {:code c}))))

;; ---------------------------------------------------------------------------
;; Arrays
;; ---------------------------------------------------------------------------

(defn- host-values
  "The MlxArray x's values on the host, by its backend (gen.backend.mlx's
   to-host, which evaluates x)."
  [x]
  (p/-to-host (p/-backend x) x))

(deftype MlxArray [^long h backend shape dtype]
  p/IArray
  (-backend [_] backend)

  ;; values: equal when dtype, shape and elements are (as the JVM backend's);
  ;; comparing reads both arrays on the host
  Object
  ;; a 0-d array and a host number or boolean: the JVM backend's 0-d arrays
  ;; are host values, which = compares, so a silent false here would take
  ;; the other branch of plain Clojure (DESIGN §9, genclj-8rdm)
  (equals [this o]
    (cond (identical? this o) true
          (instance? MlxArray o) (and (= dtype (.-dtype ^MlxArray o))
                                      (= shape (.-shape ^MlxArray o))
                                      (= (host-values this) (host-values o)))
          (and (= [] shape) (or (instance? Number o) (instance? Boolean o)))
          (throw (ex-info (str "= on a 0-d MLX array and the host value " (pr-str o) ": plain Clojure cannot compare them"
                               " (on the JVM backend the array would be a host value). Compare with gen.array"
                               " (array/== x v), or inside a gen body, whose = reads the array.")
                          {:reason :host-comparison :value o :dtype dtype}))
          :else false))
  (hashCode [this] (hash [shape (host-values this)]))

  clojure.lang.IHashEq
  (hasheq [this] (.hashCode this))

  ;; get, nth and count are loud (rule 6; genclj-fzez): an array is no
  ;; Clojure collection, and get would answer nil
  clojure.lang.ILookup
  (valAt [this _] (throw (p/not-a-collection "get" this)))
  (valAt [this _ _] (throw (p/not-a-collection "get" this)))
  clojure.lang.Indexed
  (nth [this _] (throw (p/not-a-collection "nth" this)))
  (nth [this _ _] (throw (p/not-a-collection "nth" this)))
  (count [this] (throw (p/not-a-collection "count" this))))

(defn mlx-array? [x] (instance? MlxArray x))

(defn- read-shape [^long h]
  (let [n (Mlx/mlx_array_ndim h)]
    (if (zero? n)
      []
      (let [^MemorySegment seg (.reinterpret ^MemorySegment (Mlx/mlx_array_shape h) (* 4 n))]
        (mapv long (.toArray seg ValueLayout/JAVA_INT))))))

(defn wrap
  "The MlxArray owning the new handle h (freed once it is unreachable:
   MlxRuntime/adopt)."
  [b ^long h]
  (let [x (MlxArray. h b (read-shape h) (dtype-of-code (Mlx/mlx_array_dtype h)))]
    (MlxRuntime/adopt x h)
    x))

(defn wrap-all
  "The MlxArrays owning the new handles hs (a long[], as
   MlxRuntime/vectorArrays gives them)."
  [b ^longs hs]
  (mapv #(wrap b %) hs))

(defn fence
  "Keep x reachable (its handle valid) up to this point."
  [x]
  (Reference/reachabilityFence x))

;; ---------------------------------------------------------------------------
;; The backend's configuration: its device, streams and default dtypes
;; ---------------------------------------------------------------------------

(defrecord Config [device ^long stream ^long cpu-stream defaults])

(definterface Configured
  (^gen.backend.mlx.array.Config config []))

(defn config
  "Backend b's Config (the MLX backend is Configured)."
  ^Config [b]
  (.config ^Configured b))

(defn stream "Backend b's stream." ^long [b] (.-stream (config b)))
(defn cpu-stream "Backend b's CPU stream." ^long [b] (.-cpu-stream (config b)))
(defn defaults "Backend b's default dtypes." [b] (.-defaults (config b)))

(defn device
  "The device of an MLX backend: :cpu or :gpu."
  [b]
  (:device (config b)))

(defn with-device
  "(f d) for a new handle d of the :cpu or :gpu device, freed after: mlx-c
   allocates each one, and a stream made from it keeps its own copy."
  [device f]
  (let [d (Mlx/mlx_device_new_type (int (case device :cpu 0 :gpu 1)) (int 0))]
    (try (f d) (finally (Mlx/mlx_device_free d)))))

(defonce aux-stream
  ;; for conversions made where no backend is at hand
  (with-device :cpu #(Mlx/mlx_stream_new_thread_unsafe (long %))))

(defmethod print-method MlxArray [^MlxArray x ^java.io.Writer w]
  ;; as gen.serialize writes it, so that the text reads back
  (host/print-array (.-dtype x) (.-shape x) (host-values x) w))
