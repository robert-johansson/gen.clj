(ns gen.backend.mlx-test
  "The MLX binding and backend beyond the conformance suite: the handle ABI,
   errors, closures, threads and memory. Skipped (with a note) where the
   native library or the compiled binding is missing."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.memory :as reservations]))

(def available?
  (delay
    (try
      (Class/forName "gen.mlx.MlxRuntime")
      true
      (catch Throwable t
        (println "Skipping the MLX tests:" (or (ex-message t) (.getName (class t)))
                 "(build with native/build-mlx.sh and clojure -T:build java)")
        false))))

(defmacro ^:private when-mlx [& body]
  `(when @available? ~@body))

(defn- mlx [] @(requiring-resolve 'gen.backend.mlx/backend))

;; ---------------------------------------------------------------------------
;; The ABI: a struct { void* ctx; } passed and returned by value is a pointer
;; ---------------------------------------------------------------------------

(deftest one-pointer-structs-are-pointers
  (when-mlx
    (let [linker (java.lang.foreign.Linker/nativeLinker)
          lib (gen.mlx.MlxRuntime/library)
          handle (java.lang.foreign.MemoryLayout/structLayout
                  (into-array java.lang.foreign.MemoryLayout [java.lang.foreign.ValueLayout/ADDRESS]))
          find (fn [n] (.orElseThrow (.find ^java.lang.foreign.SymbolLookup lib n)))
          ;; FFM's own by-value struct calls, for comparison with the binding's longs
          new-f32 (.downcallHandle linker (find "mlx_array_new_float32")
                                   (java.lang.foreign.FunctionDescriptor/of handle (into-array java.lang.foreign.MemoryLayout [java.lang.foreign.ValueLayout/JAVA_FLOAT]))
                                   (make-array java.lang.foreign.Linker$Option 0))
          size (.downcallHandle linker (find "mlx_array_size")
                                (java.lang.foreign.FunctionDescriptor/of java.lang.foreign.ValueLayout/JAVA_LONG (into-array java.lang.foreign.MemoryLayout [handle]))
                                (make-array java.lang.foreign.Linker$Option 0))
          arena (java.lang.foreign.Arena/ofConfined)]
      (try
        (let [;; returned as a struct by FFM, read as the binding's long
              ^java.lang.foreign.MemorySegment s (.invokeWithArguments new-f32 (object-array [arena (float 2.5)]))
              h (.get s java.lang.foreign.ValueLayout/JAVA_LONG 0)]
          (is (== 2.5 (gen.mlx.Mlx/mlx_array_item_float32 h)) "a struct return is the handle in the return register")
          (is (== 1 (gen.mlx.Mlx/mlx_array_size h)))
          ;; made by the binding (a long), passed by FFM as a struct
          (let [h2 (gen.mlx.MlxRaw/mlx_array_new_float32 (float 4.0))
                arg (.allocate arena handle)]
            (.set arg java.lang.foreign.ValueLayout/JAVA_LONG 0 h2)
            (is (== 1 (.invokeWithArguments size (object-array [arg]))) "a struct argument is the handle in a register")
            (gen.mlx.MlxRaw/mlx_array_free h2))
          (gen.mlx.MlxRaw/mlx_array_free h))
        (finally (.close arena))))))

;; ---------------------------------------------------------------------------
;; Errors: MLX's are exceptions, and the JVM keeps running
;; ---------------------------------------------------------------------------

(deftest native-linalg-gradients-need-no-default-backend
  ;; in native mode MLX calls a linear-algebra VJP back with whatever
  ;; *backend* its caller had, and the rules make eye and arange, which once
  ;; landed there: "arrays from different backends" (an independent review)
  (when-mlx
    (when (= "native" (System/getProperty "gen.mlx.grad"))
      (let [b (mlx)
            cholesky @(requiring-resolve 'gen.array.linalg/cholesky)
            m (array/with-backend b (array/array [[2.0 0.5] [0.5 1.0]]))
            g (array/with-backend (array/backend-named :jvm)
                ((array/grad (fn [x] (array/sum (cholesky x)))) m))]
        (is (= [2 2] (array/shape g)))))))

(deftest the-array-threshold-stays-above-the-live-count
  ;; with more arrays live than the cap, the threshold was set below the
  ;; live count, so every new array ran a full collection (an independent
  ;; review): past the cap it is now another tenth of the cap away
  (when-mlx
    (let [next-threshold #(gen.mlx.MlxRuntime/nextArrayThreshold %1 100000 300000)]
      (is (= 100000 (next-threshold 10000)) "at least the base")
      (is (= 240000 (next-threshold 120000)) "twice what is live")
      (is (= 300000 (next-threshold 160000)) "the cap")
      (is (= 380000 (next-threshold 350000)) "past the cap, a tenth of it later")
      (is (every? #(> (next-threshold %) %) [10000 160000 299999 350000 1000000])
          "always above what is live"))))

(defn- buffer-limit
  "The GPU's buffer limit (its device info's resource_limit), or nil where
   it reports none or there is no GPU."
  []
  (let [dev (gen.mlx.Mlx/mlx_device_new_type 1 0)]
    (try
      (let [info (gen.mlx.Mlx/mlx_device_info_get dev)]
        (try
          (when (gen.mlx.Mlx/mlx_device_info_has_key info "resource_limit")
            (gen.mlx.Mlx/mlx_device_info_get_size info "resource_limit"))
          (finally (gen.mlx.Mlx/mlx_device_info_free info))))
      (catch gen.mlx.MlxException _ nil)
      (finally (gen.mlx.Mlx/mlx_device_free dev)))))

(deftest the-array-cap-is-metals-buffer-limit
  ;; the cap was read in MlxRuntime's initializer, which the backend's first
  ;; mlx-c call reaches from inside that call's half-initialized binding
  ;; class: with the MLX backend loaded first (-Dgen.backend=mlx) the cap
  ;; was no limit, and the full Metal battery met MLX's buffer limit
  ;; (2026-10-21)
  (when-mlx
    (mlx)
    (if-let [limit (buffer-limit)]
      (is (= (max (Long/getLong "gen.mlx.gc-arrays" 100000) (long (* 0.6 limit)))
             (gen.mlx.MlxRuntime/arrayCap))
          "60% of the buffer limit")
      (is (< (quot Long/MAX_VALUE 8) (gen.mlx.MlxRuntime/arrayCap)) "no limit, no cap"))))

(deftest large-argument-blocks-reuse-scratch
  ;; an argument block over half the scratch buffer (more than 64 handles,
  ;; as eval! of a large trace passes) was allocated afresh on every call
  ;; from the thread's arena, which frees only with the thread: a leak per
  ;; call (an independent review's finding)
  (when-mlx
    (let [address #(.address ^java.lang.foreign.MemorySegment
                             (.longs (gen.mlx.MlxRuntime/scratch) (long-array 200)))]
      (address)
      (is (= (address) (address)) "the same memory, call after call"))))

(deftest mlx-errors-are-exceptions
  (when-mlx
    (let [cpu (gen.mlx.Mlx/mlx_default_cpu_stream_new)
          x (gen.mlx.MlxRaw/mlx_array_new_float32 (float 1.0))]
      (is (thrown-with-msg? gen.mlx.MlxException #"Cannot reshape"
                            (gen.mlx.Mlx/mlx_reshape x (int-array [2 2]) cpu)))
      (testing "and the next call works"
        (is (== 1.0 (gen.mlx.Mlx/mlx_array_item_float32 x))))
      (gen.mlx.MlxRaw/mlx_array_free x))
    (testing "through the backend, as ex-info naming the op"
      (let [b (mlx)]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"do not broadcast"
                              (array/with-backend b (array/+ (array/array [1.0 2.0]) (array/array [1.0 2.0 3.0])))))
        (when (= :gpu ((requiring-resolve 'gen.backend.mlx/device) b))
          (is (thrown? clojure.lang.ExceptionInfo
                       (array/with-backend b (array/->clj (array/array [1.0] {:dtype :float64}))))
              "float64 on the GPU"))))))

;; ---------------------------------------------------------------------------
;; Closures: Java functions MLX calls
;; ---------------------------------------------------------------------------

(defn- closure ^long [f]
  (gen.mlx.MlxRuntime/closure (reify gen.mlx.MlxRuntime$Closure (apply [_ ins] (f ins)))))

(deftest closures
  (when-mlx
    (let [cpu (gen.mlx.Mlx/mlx_default_cpu_stream_new)
          registered #(gen.mlx.MlxRuntime/registeredClosures)
          before (registered)]
      (testing "value_and_grad of a Java function"
        ;; MLX builds the backward pass on its default device: the CPU here,
        ;; since these raw calls evaluate on this thread, and MLX's CUDA
        ;; streams fail (and then hang) when another thread evaluated them
        ;; before (the backend's own evaluations run on one thread there)
        (let [prev (gen.mlx.Mlx/mlx_get_default_device)
              _ (gen.mlx.Mlx/mlx_set_default_device (gen.mlx.Mlx/mlx_device_new_type (int 0) (int 0)))
              f (closure (fn [^longs ins]
                           (let [x (aget ins 0)]
                             (long-array [(gen.mlx.Mlx/mlx_sum (gen.mlx.Mlx/mlx_multiply x x cpu) false cpu)]))))
              vag (gen.mlx.Mlx/mlx_value_and_grad f (int-array [0]))
              x (gen.mlx.MlxRaw/mlx_array_new_float32 (float 3.0))
              xs (gen.mlx.MlxRuntime/vectorOf (long-array [x]))
              [vs gs] (vec (gen.mlx.Mlx/mlx_closure_value_and_grad_apply vag xs))]
          (is (== 9.0 (gen.mlx.Mlx/mlx_array_item_float32 (aget (gen.mlx.MlxRuntime/vectorArrays vs) 0))))
          (is (== 6.0 (gen.mlx.Mlx/mlx_array_item_float32 (aget (gen.mlx.MlxRuntime/vectorArrays gs) 0))))
          (is (= (inc before) (registered)))
          (gen.mlx.MlxRaw/mlx_closure_value_and_grad_free vag)
          (gen.mlx.MlxRaw/mlx_closure_free f)
          (gen.mlx.Mlx/mlx_set_default_device prev)
          (is (= before (registered)) "MLX releases the function with the closure")))
      (testing "an exception in the function comes back as itself"
        (let [f (closure (fn [_] (throw (ex-info "from the closure" {:k 1}))))
              x (gen.mlx.MlxRaw/mlx_array_new_float32 (float 1.0))]
          (is (= {:k 1} (try (gen.mlx.Mlx/mlx_closure_apply f (gen.mlx.MlxRuntime/vectorOf (long-array [x])))
                             nil
                             (catch clojure.lang.ExceptionInfo e (ex-data e)))))
          (gen.mlx.MlxRaw/mlx_closure_free f)))
      (testing "a call inside the function does not clobber the outer call's out-parameter"
        ;; the closure's own calls run on a child scratch
        (let [f (closure (fn [^longs ins]
                           (let [y (gen.mlx.Mlx/mlx_add (aget ins 0) (aget ins 0) cpu)]
                             (long-array [(gen.mlx.Mlx/mlx_add y y cpu)]))))
              x (gen.mlx.MlxRaw/mlx_array_new_float32 (float 1.5))
              out (gen.mlx.Mlx/mlx_closure_apply f (gen.mlx.MlxRuntime/vectorOf (long-array [x])))]
          (is (== 6.0 (gen.mlx.Mlx/mlx_array_item_float32 (aget (gen.mlx.MlxRuntime/vectorArrays out) 0))))
          (is (== 1.5 (gen.mlx.Mlx/mlx_array_item_float32 x)) "the input is untouched")
          (gen.mlx.MlxRaw/mlx_closure_free f))))))

;; ---------------------------------------------------------------------------
;; Threads and memory
;; ---------------------------------------------------------------------------

(deftest arrays-cross-threads
  (when-mlx
    (let [b (mlx)
          x (array/with-backend b (array/+ (array/array [1.0 2.0]) 1.0))]
      (testing "an array made on one thread is evaluated on another"
        (is (= [2.0 3.0] @(future (array/->clj x)))))
      (testing "concurrent evaluation"
        (is (= (repeat 8 6.0)
               (map deref (doall (for [_ (range 8)]
                                   (future (array/item (array/with-backend b (array/sum (array/* (array/array [1.0 2.0 3.0]) 1.0))))))))))))))

(deftest unreachable-arrays-are-freed
  (when-mlx
    (let [b (mlx)
          memory (requiring-resolve 'gen.backend.mlx/memory)
          live (fn [] (System/gc) (Thread/sleep 50) (gen.mlx.MlxRuntime/drain) (:active (memory)))
          base (live)]
      ;; 200 evaluated [100000] float32 arrays: 80 MB if none were freed
      (dotimes [_ 200]
        (array/eval! (array/with-backend b (array/+ (array/zeros [100000]) 1.0))))
      (is (< (- (live) base) 8e6) "unreachable arrays are freed"))))

(deftest array-pressure-collects
  (when-mlx
    (let [b (mlx)
          [live0 collections0] (vec (gen.mlx.MlxRuntime/lifetimeStats))]
      ;; 300,000 unevaluated scalars, kept unreachable: more than the
      ;; collection threshold, and close to Metal's buffer limit if none
      ;; were freed
      (dotimes [_ 300000]
        (array/with-backend b (array/+ (array/array 1.0) 1.0)))
      (let [[live collections] (vec (gen.mlx.MlxRuntime/lifetimeStats))]
        (is (< live 250000) "live arrays stay below the threshold's reach")
        (is (> collections collections0) "pressure collected")))))

(deftest compile-traces-once
  (when-mlx
    (let [b (mlx)
          traces (atom 0)
          f (array/compile (fn [x] (swap! traces inc) (array/sum (array/exp x))))]
      (array/with-backend b
        (let [x (array/array [0.0 1.0 2.0])]
          (if ((requiring-resolve 'gen.backend.mlx/compiles?) b)
            (do
              (dotimes [_ 5] (array/->clj (f x)))
              (is (= 1 @traces) "traced on the first call, replayed after")
              (f (array/array [1.0 2.0]))
              (is (= 2 @traces) "a new shape traces again")
              (is (thrown-with-msg? clojure.lang.ExceptionInfo #"reads an array's value on the host"
                                    ((array/compile (fn [x] (if (array/truthy? (array/> (array/sum x) 0)) x (array/negate x)))) x))
                  "a host read inside is loud"))
            ;; where MLX cannot build CPU kernels (GCC 13 on Linux), compile
            ;; runs the function itself
            (do
              (is (array/allclose? (array/sum (array/exp x)) (f x)))
              (is (= 1 @traces) "called, not traced"))))))))

;; held until the JVM exits, as gen.inference.mcmc's compiled steps are: a
;; compiled function erases its cache entries when it is freed
(def ^:private compiled-at-exit (atom nil))

(deftest compiled-native-gradients-survive-the-exit
  ;; MLX keeps compiled graphs in a thread-local cache. Custom VJPs inside
  ;; one were released, when the main thread's cache was destroyed at the
  ;; JVM's exit, by an upcall into a JVM shutting down: SIGILL (exit 132)
  ;; after every test had passed. This compiles native gradients through
  ;; float64 exact routes and a solve on the test runner's own thread, and
  ;; keeps it: with the bug the JVM crashes at exit, after the summary
  ;; (and its error reporter may then spin instead of exiting)
  (when-mlx
    (when (= "native" (System/getProperty "gen.mlx.grad"))
      (let [b ((requiring-resolve 'gen.backend.mlx/make-backend) {:device :cpu :dtypes 64})
            solve @(requiring-resolve 'gen.array.linalg/solve)]
        (array/with-backend b
          (let [m (array/array [[2.0 0.5] [0.5 1.0]])
                f (array/compile (fn [x] ((array/grad (fn [y] (array/+ (array/sum (array/erf y)) (array/sum (solve m y))))) x)))
                g (array/->clj (f (array/array [0.1 0.5])))
                _ (reset! compiled-at-exit f)
                ;; d/dy: 2/√π e^(-y²) plus the row sums of m⁻ᵀ
                minv-t-sums [(/ (- 1.0 0.5) 1.75) (/ (- 2.0 0.5) 1.75)]]
            ;; the compiled gradient differentiates the composites' own
            ;; graphs, accurate to their approximations
            (is (every? true? (map #(< (Math/abs (- (double %1) (+ (* (/ 2.0 (Math/sqrt Math/PI)) (Math/exp (- (* %2 %2)))) %3))) 1e-9)
                                   g [0.1 0.5] minv-t-sums)))))))))

;; ---------------------------------------------------------------------------
;; Compiled functions called from several threads (genclj-hfwo)
;; ---------------------------------------------------------------------------
;; MLX keeps a compile cache per thread. mlx_compile's closures erased their
;; entry only from the cache of the thread that compiled them, so a call
;; from another thread left a trace there: kept, with every constant it
;; captured, until that thread exited, and keyed by the address of a deleted
;; function, which a later compiled function could be given. Mutation
;; checked on Metal (2026-10-21): the old mlx_compile planted back fails the
;; first test (256 MB kept) and the second (from round 3 on, the other
;; thread returned round 2's x·2 every time).

(defn- on-thread
  "(f) on executor ex's thread: its value, or what it threw."
  [^java.util.concurrent.ExecutorService ex f]
  (try (.get (.submit ex ^java.util.concurrent.Callable f))
       (catch java.util.concurrent.ExecutionException e (throw (.getCause e)))))

(defn- settled-active
  "MLX's active memory once the unreachable arrays and compiled functions
   are freed."
  []
  (let [collect! (requiring-resolve 'gen.backend.mlx/collect!)]
    (dotimes [_ 3] (collect!))
    (:active ((requiring-resolve 'gen.backend.mlx/memory)))))

(deftest a-freed-compiled-function-leaves-no-trace-on-other-threads
  ;; a compiled function closing over a 256 MB constant, called on this
  ;; thread and on another that stays alive, then dropped: once collected,
  ;; MLX's active memory is back where it was
  (when-mlx
    (let [b (mlx)]
      (when ((requiring-resolve 'gen.backend.mlx/compiles?) b)
        (let [ex (java.util.concurrent.Executors/newSingleThreadExecutor)]
          (try
            (let [base (settled-active)
                  [here there] ((fn []
                                  (let [c (array/eval! (array/with-backend b (array/full [(* 64 1024 1024)] 0.5)))
                                        f (array/compile (fn [x] (array/+ x (array/max c))))
                                        x (array/with-backend b (array/array [1.0 2.0]))]
                                    [(array/->clj (f x)) (on-thread ex #(array/->clj (f x)))])))
                  kept (- (settled-active) base)]
              (is (= [1.5 2.5] here there) "both threads ran the compiled function")
              (is (< kept (* 16 1024 1024)) (format "%.0f MB still held" (/ kept 1048576.0))))
            (finally (.shutdown ex))))))))

(deftest ^:slow a-compiled-function-never-replays-another
  ;; rounds of a compiled x·i, each called on this thread and then on
  ;; another, then dropped and collected: every result is its own x·i, never
  ;; an earlier round's trace found under a reused address
  (when-mlx
    (let [b (mlx)]
      (when ((requiring-resolve 'gen.backend.mlx/compiles?) b)
        (let [ex (java.util.concurrent.Executors/newSingleThreadExecutor)
              x (array/with-backend b (array/array [2.0]))]
          (try
            (let [wrong (vec (keep (fn [i]
                                     (let [f (array/compile (fn [y] (array/* y (double i))))
                                           here (array/->clj (f x))
                                           there (on-thread ex #(array/->clj (f x)))]
                                       (settled-active)
                                       (when-not (= [(* 2.0 i)] here there) [i here there])))
                                   (range 1 65)))]
              (is (= [] wrong) "each round's results are its own"))
            (finally (.shutdown ex))))))))

(deftest compiled-functions-run-on-virtual-threads
  ;; a virtual thread's MLX cache is its carrier's, which can change between
  ;; calls: the trace is erased after each call, and the results are right
  (when-mlx
    (let [b (mlx)
          f (array/compile (fn [y] (array/* y 3.0)))
          x (array/with-backend b (array/array [1.0 2.0]))
          results (with-open [ex (java.util.concurrent.Executors/newVirtualThreadPerTaskExecutor)]
                    (mapv #(.get ^java.util.concurrent.Future %)
                          (doall (for [_ (range 8)] (.submit ex ^java.util.concurrent.Callable #(array/->clj (f x)))))))]
      (is (every? #{[3.0 6.0]} results)))))

;; ---------------------------------------------------------------------------
;; The budget: past it an evaluation is a loud error, after a collection
;; ---------------------------------------------------------------------------

(def ^:private mb (* 1024 1024))

(defn- big-array
  "128 MB of float32 ones on backend b, unevaluated."
  [b]
  (array/with-backend b (array/ones [(* 32 mb)])))

(defn- evaluation-error
  "The ex-info an evaluation of x throws, or nil."
  [x]
  (try (array/eval! x) nil (catch clojure.lang.ExceptionInfo e e)))

(defn- with-room*
  "(f) with MLX's budget what MLX and the reservations hold now plus room
   bytes (no headroom), restored after."
  [room f]
  (let [budget (requiring-resolve 'gen.backend.mlx/budget)
        set-budget! (requiring-resolve 'gen.backend.mlx/set-budget!)
        collect! (requiring-resolve 'gen.backend.mlx/collect!)
        _ (collect!)
        old (set-budget! {:bytes (+ (:active (budget)) (reservations/reserved) room) :headroom 0})]
    (try (f)
         (finally (set-budget! old)
                  (collect!)))))

(deftest an-evaluation-past-the-budget-is-loud
  ;; with 64 MB of room, 4 KB fit and 128 MB do not
  (when-mlx
    (let [b (mlx)]
      (with-room* (* 64 mb)
        (fn []
          (is (= 1024.0 (array/item (array/with-backend b (array/sum (array/ones [1024]))))))
          (let [big (big-array b)
                x (evaluation-error big)]
            (is (= :mlx-budget-exceeded (:reason (ex-data x))) (pr-str x))
            (is (< (:mlx (ex-data x)) (:active (ex-data x))) "with the numbers")))))))

(deftest a-collection-comes-before-the-error
  ;; a dead 128 MB array still holds its memory: the next evaluation finds
  ;; MLX past its budget of 64 MB of room, collects, and goes on
  (when-mlx
    (let [b (mlx)
          budget (requiring-resolve 'gen.backend.mlx/budget)
          set-budget! (requiring-resolve 'gen.backend.mlx/set-budget!)]
      (with-room* (* 64 mb)
        (fn []
          (let [tight (set-budget! {:bytes (+ (:bytes (budget)) (* 256 mb))})]
            (array/eval! (big-array b))
            (set-budget! tight)
            (is (< (:mlx (budget)) (:active (budget))) "the dead array still holds its memory")
            (is (= 1024.0 (array/item (array/with-backend b (array/sum (array/ones [1024])))))
                "an evaluation collects it, and goes on")
            (is (<= (:active (budget)) (:mlx (budget))))))))))

(deftest reservations-shrink-the-budget
  ;; what a native engine reserves is not MLX's: 192 MB reserved of 256 MB
  ;; of room leave too little for 128 MB, until it is released
  (when-mlx
    (let [b (mlx)
          budget (requiring-resolve 'gen.backend.mlx/budget)]
      (with-room* (* 256 mb)
        (fn []
          (let [before (:mlx (budget))]
            (reservations/reserve! ::engine (* 192 mb) "a test's engine")
            (try
              (is (= (- before (* 192 mb)) (:mlx (budget))))
              (is (= :mlx-budget-exceeded (:reason (ex-data (evaluation-error (big-array b))))))
              (finally (reservations/release! ::engine)))
            (is (= before (:mlx (budget))))
            (is (nil? (evaluation-error (big-array b))) "released, the same evaluation fits")))))))

(deftest the-default-budget-leaves-room
  ;; half the RAM, less a sixteenth of headroom, less any reservation, and
  ;; what the suites hold stays inside it
  (when-mlx
    (let [{:keys [mlx bytes headroom reserved active]} ((requiring-resolve 'gen.backend.mlx/budget))
          ram (.getTotalMemorySize ^com.sun.management.OperatingSystemMXBean
                                   (java.lang.management.ManagementFactory/getOperatingSystemMXBean))]
      (when-not (or (System/getProperty "gen.mlx.budget-bytes") (System/getProperty "gen.mlx.budget-headroom-bytes"))
        (is (= (quot ram 2) bytes))
        (is (= (quot ram 16) headroom)))
      (is (= mlx (max 0 (- bytes reserved headroom))))
      (is (< active mlx)))))

;; ---------------------------------------------------------------------------
;; Metal's buffer limit: collected and retried once (genclj-9tyh)
;; ---------------------------------------------------------------------------
;; MLX on macOS makes every buffer through Metal, which allows about 499,000,
;; and recovers fully once buffers are freed. Where a buffer is made (an
;; array from host data, an evaluation), the limit runs a full collection
;; and one retry; at the limit again, :mlx-buffer-limit.

(defn- limit-error []
  (gen.mlx.MlxException. "mlx_array_new_data" "[metal::malloc] Resource limit (499000) exceeded."))

(deftest the-buffer-limit-is-retried-once
  ;; the retry itself, with an error made here: fast, on any device
  (when-mlx
    (let [at-limit @(requiring-resolve 'gen.backend.mlx/at-buffer-limit)
          [_ collections0] (vec (gen.mlx.MlxRuntime/lifetimeStats))
          calls (atom 0)]
      (testing "at the limit once: a collection, then the call again"
        (is (= :made (at-limit "x" #(if (= 1 (swap! calls inc)) (throw (limit-error)) :made))))
        (is (= 2 @calls))
        (is (< collections0 (second (gen.mlx.MlxRuntime/lifetimeStats))) "a collection, whatever the thresholds"))
      (testing "at the limit twice: a loud error, MLX's its cause"
        (let [e (try (at-limit "x" #(throw (limit-error))) (catch Exception e e))]
          (is (= :mlx-buffer-limit (:reason (ex-data e))) (pr-str e))
          (is (= "x" (:op (ex-data e))))
          (is (integer? (:live-arrays (ex-data e))))
          (is (= (gen.mlx.MlxRuntime/arrayCap) (:cap (ex-data e))))
          (is (instance? gen.mlx.MlxException (ex-cause e)))))
      (testing "any other MLX error: as it was, without a retry"
        (reset! calls 0)
        (let [e (try (at-limit "x" #(do (swap! calls inc) (throw (gen.mlx.MlxException. "f" "bad shape"))))
                     (catch Exception e e))]
          (is (instance? gen.mlx.MlxException e))
          (is (= 1 @calls)))))))

(defn- source-forms
  "The top-level forms of a source file, read as data."
  [path]
  (with-open [r (java.io.PushbackReader. (clojure.java.io/reader path))]
    (binding [*read-eval* false]
      (doall (take-while #(not= % ::eof) (repeatedly #(read {:eof ::eof} r)))))))

(deftest every-array-from-host-data-is-made-through-made
  ;; at Metal's buffer limit an mlx-c constructor (mlx_array_new_*) returns
  ;; an empty handle; made turns that into MLX's error, collects and
  ;; retries once. A constructor called raw handed the empty handle on, to
  ;; fail in the next op (genclj-j9nx: safe-indices' scalars, found by the
  ;; full Metal battery). Read from the source, so it runs without MLX.
  ;; scalar-handle* is a constructor too, called only through made
  (let [constructor? (fn [x] (and (symbol? x) (#{"Mlx" "MlxRaw"} (namespace x))
                                  (.startsWith (name x) "mlx_array_new")))
        raw (fn raw [form]
              (cond (and (seq? form) (= 'made (first form))) []
                    (and (seq? form) (#{'defn 'defn-} (first form)) (= 'scalar-handle* (second form))) []
                    (and (seq? form) (= 'scalar-handle* (first form))) [form]
                    (constructor? form) [form]
                    (coll? form) (mapcat raw form)
                    :else []))]
    (is (= [] (vec (mapcat raw (source-forms "src/gen/backend/mlx.clj")))))))

(defn- held-until-the-limit
  "One-element arrays made and held until making another fails: how many,
   and the failure. They are unreachable once this returns."
  [b]
  (loop [held (transient [])]
    (let [r (try (array/with-backend b (array/array [1.0])) (catch Exception e e))]
      (if (instance? Throwable r)
        [(count held) r]
        (recur (conj! held r))))))

(deftest ^:slow metals-buffer-limit-is-collected-and-retried
  ;; held arrays reach the limit loudly (:mlx-buffer-limit); dropped, they
  ;; are garbage, so the next array made, and then the next evaluation (a
  ;; graph made beforehand that needs 20,000 buffers at once), meets the
  ;; limit, collects and succeeds. Filling the limit takes a minute or two
  ;; on Metal; skipped where the device reports no limit (CUDA, the CPU on
  ;; Linux). Mutations checked on Metal (2026-10-21): without the retry, the
  ;; array and the evaluation after the drop meet :mlx-buffer-limit (499,000
  ;; arrays live), and the fast test sees one call; with MLX's error in
  ;; place of the ex-info, both tests find no :reason and no cause.
  (when-mlx
    (if (<= (quot Long/MAX_VALUE 8) (gen.mlx.MlxRuntime/arrayCap))
      (println "Skipping metals-buffer-limit-is-collected-and-retried: the device reports no buffer limit")
      (let [b (mlx)
            memory @(requiring-resolve 'gen.backend.mlx/memory)
            k 20000
            g (array/with-backend b
                (let [x (array/array [2.0])]
                  (array/sum (array/concatenate (mapv (fn [_] (array/sin x)) (range k))))))]
        (try
          (testing "held arrays: a loud error"
            (let [[n e] (held-until-the-limit b)]
              (is (= :mlx-buffer-limit (:reason (ex-data e))) (pr-str e))
              (is (< (gen.mlx.MlxRuntime/arrayCap) n) (str n " arrays held"))
              (is (.contains (str (ex-message (ex-cause e))) "Resource limit"))))
          (testing "the same arrays dropped: the next array is made after a collection"
            (let [c0 (:collections (memory))]
              (is (= [2.0] (array/->clj (array/with-backend b (array/array [2.0])))))
              (is (< c0 (:collections (memory))))))
          (testing "filled and dropped again: the graph evaluates after a collection"
            (held-until-the-limit b)
            (let [c0 (:collections (memory))]
              (is (< (Math/abs (- (* k (Math/sin 2.0)) (array/item g))) 0.05))
              (is (< c0 (:collections (memory))))))
          (testing "and take by an array index makes its scalars after a collection (genclj-j9nx)"
            ;; the particle filter's 0-d index, whose bounds safe-indices
            ;; makes from host data
            (let [x (array/with-backend b (array/array [[1.0 2.0] [3.0 4.0]]))
                  i (array/with-backend b (array/array 1))]
              (held-until-the-limit b)
              (let [c0 (:collections (memory))]
                (is (= [3.0 4.0] (array/->clj (array/take x i 0))))
                (is (< c0 (:collections (memory)))))))
          (finally
            ((requiring-resolve 'gen.backend.mlx/collect!))
            ((requiring-resolve 'gen.backend.mlx/clear-cache!))))))))
