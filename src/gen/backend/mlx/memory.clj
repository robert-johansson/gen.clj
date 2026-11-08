(ns gen.backend.mlx.memory
  "The memory the MLX backend holds (gen.backend.mlx): MLX's own limits, the
   backend's budget (shared with native engines: gen.array.memory),
   Metal's buffer limit, and evaluation, under one lock (on one thread
   where CUDA needs it), after which the budget is checked."
  {:api :internal}
  (:require [gen.array.memory :as reservations]
            [gen.backend.mlx.array :refer [fence handle with-device]])
  (:import [gen.mlx Mlx MlxException MlxRuntime]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; The evaluation lock and thread
;; ---------------------------------------------------------------------------

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

(defn cuda-gpu?
  "Does MLX compute on a CUDA GPU here (Linux with a GPU: the evaluations
   need one thread)?"
  []
  @one-eval-thread?)

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

(defn at-buffer-limit
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

(defn evaluating*
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

(defmacro evaluating
  "Run body (which evaluates MLX arrays) holding the evaluation lock, on the
   evaluation thread where there is one (evaluating*)."
  [& body]
  `(evaluating* (fn [] ~@body)))

;; ---------------------------------------------------------------------------
;; MLX's limits and memory
;; ---------------------------------------------------------------------------

;; never read: defined for its effect (clj-kondo's unused-private-var does
;; not apply)
#_{:clj-kondo/ignore [:unused-private-var]}
(defonce ^:private memory-bounds
  ;; once per process, before any array: set-cache-limit! and
  ;; set-memory-limit! change them later
  (let [ram (physical-memory)]
    (Mlx/mlx_set_cache_limit (long (Long/getLong "gen.mlx.cache-bytes" (quot ram 16))))
    (Mlx/mlx_set_memory_limit (long (Long/getLong "gen.mlx.memory-bytes" (quot ram 2))))
    true))

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
  (try (Mlx/_mlx_array_is_available (handle x))
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
