(ns gen.backend.mlx-test
  "The MLX binding and backend beyond the conformance suite: the handle ABI,
   errors, closures, threads and memory. Skipped (with a note) where the
   native library or the compiled binding is missing."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.memory :as reservations]
            [gen.array.protocols :as p]))

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
      ;; the protocol called directly: gen.array checks a reshape's size
      ;; before it calls a backend (genclj-q45d), so MLX never sees this one
      (let [b (mlx)
            x (array/with-backend b (array/array [1.0 2.0 3.0]))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"^reshape: .*[Rr]eshape"
                              (p/-reshape b x [2 2])))
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

;; ---------------------------------------------------------------------------
;; Temporary handles: with-temps frees them, failure paths too (genclj-f0wz)
;; ---------------------------------------------------------------------------

(deftest temporaries-are-made-in-with-temps-and-devices-freed
  ;; the review's I1-04: ten hand-written cleanups (a java.util.ArrayList of
  ;; handles and a finally), three of which made handles before their try,
  ;; and device handles never freed. Read from the source, so it runs
  ;; without MLX. Planted (2026-10-22): int-division written by hand again,
  ;; and gpu? taking a device without freeing it, each fail
  (let [forms (source-forms "src/gen/backend/mlx.clj")
        named (fn [form] (when (seq? form) (second form)))
        uses (fn [form pred] (filter pred (tree-seq coll? seq form)))
        array-list? #(= 'java.util.ArrayList. %)
        new-device? #(and (symbol? %) (= "mlx_device_new_type" (name %)))]
    (is (= ['with-temps] (vec (keep #(when (seq (uses % array-list?)) (named %)) forms)))
        "a list of temporaries only in with-temps")
    ;; with-default-stream took its device by hand too, before its try
    ;; (cross-review 3): it goes through with-device now
    (is (= ['with-device] (vec (keep #(when (seq (uses % new-device?)) (named %)) forms)))
        "devices made only by with-device, which frees them")))

(defn- tail-forms
  "The forms in tail position of form, the ones it returns: through let,
   do, if, cond, case, try and fn bodies, and the backend's body macros."
  [form]
  (if-not (seq? form)
    [form]
    (let [[op & args] form]
      (case op
        (let let* loop do when when-not binding fn* with-open with-temps with-operands evaluating with-default-stream)
        (tail-forms (last form))
        (if if-not) (mapcat tail-forms (rest args))
        cond (mapcat tail-forms (take-nth 2 (rest args)))
        case (let [clauses (rest args)]
               (mapcat tail-forms (concat (take-nth 2 (rest clauses)) (when (odd? (count clauses)) [(last clauses)]))))
        try (tail-forms (last (remove #(and (seq? %) (#{'catch 'finally} (first %))) args)))
        [form]))))

(def ^:private handle-makers
  "The backend's makers of new handles besides Mlx's calls: its private fns
   that return one, and the runtime's makers of handles, closures and
   vectors."
  '#{scalar-handle made half-scalar-handle safe-indices int-division java-int-cast int-power
     MlxRuntime/closure MlxRuntime/customVjp MlxRuntime/vectorOf MlxRuntime/vectorArrays MlxRuntime/retain})

(defn- handle-call?
  "Does form make a new handle: an Mlx call (not a free, an evaluation or a
   flag), an MlxRaw constructor, or one of handle-makers?"
  [form]
  (and (seq? form) (symbol? (first form))
       (let [f (first form) n (name f)]
         (case (namespace f)
           "Mlx" (not (or (.endsWith n "_free") (= "mlx_array_eval" n) (.startsWith n "_mlx_array_is")))
           "MlxRaw" (.contains n "_new")
           (contains? handle-makers f)))))

(defn- loose-handles
  "The handle-making calls in the backend's source forms that a failure can
   leak: in a with-temps body, a standalone with-operands body or a helper
   handed t, one neither registered with t, nor wrapped, nor the body's
   result. A nested with-temps is checked as its own body."
  [forms]
  (let [registers? (fn [t f] (or (= t f) (and (seq? f) (= 'fn* (first f)) (seq? (last f)) (= t (first (last f))))))
        held-by (fn [t form]
                  ;; the forms a call hands to t, to wrap (or wrap-all), to
                  ;; a mapv of t, or to made (a constructor's thunk): their tails
                  (when (seq? form)
                    (cond (#{t 'wrap 'wrap-all} (first form)) (tail-forms (last form))
                          (and (= 'mapv (first form)) (registers? t (second form))) (tail-forms (last form))
                          (= 'made (first form)) (tail-forms (last form)))))
        nested-temps? #(and (seq? %) (= 'with-temps (first %)))
        loose (fn [t body]
                (let [inside (tree-seq #(and (coll? %) (not (nested-temps? %))) seq body)
                      ok (java.util.IdentityHashMap.)]
                  (doseq [f (concat (tail-forms (last body)) (mapcat #(held-by t %) inside))]
                    (.put ok f true))
                  (filter #(and (handle-call? %) (not (.containsKey ok %))) inside)))
        bodies (fn [top]
                 (let [all (tree-seq coll? seq top)
                       in-temps (set (mapcat #(when (nested-temps? %) (rest (tree-seq coll? seq %))) all))
                       t-param (when (and (seq? top) (#{'defn 'defn-} (first top)))
                                 (some (fn [v] (when (vector? v) (some '#{t} v))) top))]
                   (concat
                    (for [f all :when (nested-temps? f)] [(first (second f)) (drop 2 f)])
                    (for [f all :when (and (seq? f) (= 'with-operands (first f)) (not (in-temps f)))]
                      ['no-t (drop 4 f)])
                    (when t-param [[t-param (drop-while #(not (vector? %)) top)]]))))]
    (vec (for [top forms [t body] (bodies top) f (loose t body)] f))))

(deftest every-handle-made-is-registered-wrapped-or-returned
  ;; cross-reviews 3 and 5 of genclj-f0wz: each handle a site makes goes to
  ;; t as it is made, or to wrap (the Cleaner then owns it), or is the
  ;; site's result, so a later failure leaks nothing. slogdet held log|det|
  ;; unregistered while six more handles were made, which at Metal's buffer
  ;; limit leaked it and the graph back to its matrix; the failure-path
  ;; test, counting only what t registered, could not see it. The walk
  ;; covers every maker of a handle (not only Mlx's calls), standalone
  ;; with-operands bodies and the helpers handed t. Read from the source,
  ;; so it runs without MLX. Planted (2026-10-22): slogdet's logabs
  ;; unwrapped again fails; the two plants below are cross-review 5's, which
  ;; the narrower walk let through
  (let [path "src/gen/backend/mlx.clj"
        text (slurp path)
        forms-of (fn [s] (binding [*read-eval* false]
                           (let [r (java.io.PushbackReader. (java.io.StringReader. s))]
                             (doall (take-while #(not= % ::eof) (repeatedly #(read {:eof ::eof} r)))))))
        planted (fn [old new]
                  (is (.contains ^String text old) (str "the plant's site is in the source: " old))
                  (loose-handles (forms-of (.replace ^String text ^CharSequence old ^CharSequence new))))]
    (is (= [] (loose-handles (source-forms path))))
    (testing "the walk's power"
      (is (= '[(scalar-handle 0 dtype)]
             (planted "(let [zero (t (scalar-handle 0 dtype))" "(let [zero (scalar-handle 0 dtype)"))
          "int-division's zero, made without its t")
      (is (= '[(Mlx/mlx_contiguous h false (stream b))]
             (planted "    (wrap b (Mlx/mlx_reshape h (int-array (map int shape)) (stream b)))))"
                      "    (let [c (Mlx/mlx_contiguous h false (stream b))] (wrap b (Mlx/mlx_reshape c (int-array (map int shape)) (stream b))))))"))
          "reshape's operand copied before it is wrapped, which leaks on every call"))))

(defn- counting-temporaries
  "[registered freed]: the temporaries with-temps registers and frees while
   (f) runs and is evaluated, the k-th registration failing when fail-at
   is k (the handle then freed, as a constructor failing makes none)."
  [f fail-at]
  (let [registered (atom 0)
        freed (atom 0)
        temp! @(requiring-resolve 'gen.backend.mlx/temp!)
        free-temp @(requiring-resolve 'gen.backend.mlx/free-temp)]
    (with-redefs-fn {(requiring-resolve 'gen.backend.mlx/temp!)
                     (fn ^long [temps x]
                       (if (= fail-at (inc @registered))
                         (do (swap! registered inc) (free-temp x) (throw (ex-info "planted" {::planted true})))
                         (let [h (temp! temps x)] (swap! registered inc) h)))
                     (requiring-resolve 'gen.backend.mlx/free-temp)
                     (fn [x] (swap! freed inc) (free-temp x))}
      #(try (array/eval! (f))
            (catch clojure.lang.ExceptionInfo e (when-not (::planted (ex-data e)) (throw e)))))
    [(if fail-at (dec @registered) @registered) @freed]))

(deftest every-temporary-is-freed-when-a-site-fails
  ;; each site of with-temps, run whole and then failing at each of its
  ;; registrations in turn: what was registered before is freed. Planted
  ;; (2026-10-22): with-temps' finally freeing nothing fails. Run it also
  ;; with -Dgen.mlx.constants=0 (cross-review 3): every host scalar is then
  ;; a new handle, registered by with-operands, concatenate or konst
  (when-mlx
    (array/with-backend (mlx)
      (let [la-ns (requiring-resolve 'gen.array.linalg/cholesky)
            solve (requiring-resolve 'gen.array.linalg/solve)
            solve-triangular (requiring-resolve 'gen.array.linalg/solve-triangular)
            slogdet (requiring-resolve 'gen.array.linalg/slogdet)
            uniform (requiring-resolve 'gen.random/uniform)
            key (requiring-resolve 'gen.random/key)
            ints (array/array [7 -7 5 0] {:dtype :int32})
            divisors (array/array [2 3 0 -1] {:dtype :int32})
            floats (array/array [0.5 ##NaN -2.5 3e10])
            spd (array/array [[4.0 1.0] [1.0 3.0]])
            ;; a batch of two right-hand sides: batch-broadcast broadcasts the matrix
            rhs2 (array/array [[[1.0] [2.0]] [[3.0] [4.0]]])
            sites [["to-host, a half float" #(array/->clj (array/astype floats :float16))]
                   ["to-host, a view" #(array/->clj (array/transpose (array/array [[1.0 2.0] [3.0 4.0]])))]
                   ["unary's sign" #(array/sign floats)]
                   ;; float32: float64's atan has no fix-up, so no temporaries
                   ["unary's atan" #(array/atan (array/astype floats :float32))]
                   ["int-division" #(array/mod ints divisors)]
                   ["floor division" #(array/floor-div ints divisors)]
                   ["java-int-cast" #(array/astype floats :int32)]
                   ["int-power" #(array/pow ints (array/array [2 -1 0 3] {:dtype :int32}))]
                   ["arg-reduce" #(array/argmax floats)]
                   ["safe-indices" #(array/take floats (array/array [0 -1] {:dtype :int32}) 0)]
                   ["take-along-axis" #(array/take-along-axis floats (array/array [3 -1] {:dtype :int32}) 0)]
                   ["scatter-add" #(array/scatter-add floats (array/array [0 2] {:dtype :int32}) (array/array [1.0 2.0]) 0)]
                   ["the integer matmul" #(array/matmul (array/reshape ints [2 2]) (array/reshape ints [2 2]))]
                   ["cholesky" #(la-ns spd)]
                   ["solve" #(solve spd (array/array [1.0 2.0]))]
                   ["solve, a batch of right-hand sides" #(solve spd rhs2)]
                   ["solve-triangular" #(solve-triangular spd (array/array [1.0 2.0]))]
                   ["solve-triangular, a batch of right-hand sides" #(solve-triangular spd rhs2)]
                   ["slogdet" #(slogdet spd)]
                   ["random-uniform" #(uniform (key 3) [5])]
                   ["concatenate" #(array/concatenate [floats (array/array [1.0])] 0)]]
            ;; MLX's own gradients: native value-and-grad, and linalg's custom VJPs
            sites (cond-> sites
                    (= "native" (System/getProperty "gen.mlx.grad"))
                    (conj ["native value-and-grad" #((array/grad (fn [x] (array/sum (array/* x x)))) floats)]
                          ["a custom VJP (cholesky's)" #((array/grad (fn [m] (array/sum (la-ns m)))) spd)]))]
        (doseq [[label f] sites]
          (let [[n freed] (counting-temporaries f nil)]
            (is (pos? n) (str label ": registers its temporaries"))
            (is (= n freed) (str label ": run whole"))
            (doseq [k (range 1 (inc n))]
              (let [[before freed] (counting-temporaries f k)]
                (is (= before freed) (str label ": failing at its temporary " k))))))))))

(deftest a-temporary-of-no-known-kind-fails-where-it-is-registered
  ;; free-temp's case throws on a kind it does not know, which in
  ;; with-temps' finally would mask the body's error (cross-review 3), so
  ;; temp! checks the kind. Planted (2026-10-22): temp! without the check
  ;; fails
  (when-mlx
    (let [temp! @(requiring-resolve 'gen.backend.mlx/temp!)
          temps (java.util.ArrayList.)]
      (is (= :closur (:kind (try (temp! temps [:closur 1]) nil (catch clojure.lang.ExceptionInfo e (ex-data e))))))
      (is (zero? (.size temps)) "and is not registered")
      (is (== 7 (temp! temps [:vector 7])) "a known kind is"))))

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
