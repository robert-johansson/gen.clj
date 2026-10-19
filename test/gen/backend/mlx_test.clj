(ns gen.backend.mlx-test
  "The MLX binding and backend beyond the conformance suite: the handle ABI,
   errors, closures, threads and memory. Skipped (with a note) where the
   native library or the compiled binding is missing."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]))

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
