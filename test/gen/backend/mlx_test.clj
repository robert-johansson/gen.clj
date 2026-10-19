(ns gen.backend.mlx-test
  "The MLX binding and backend beyond the conformance suite: the handle ABI,
   errors, closures, threads and memory. Skipped (with a note) where the
   native library or the compiled binding is missing."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as a]))

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
                              (a/with-backend b (a/+ (a/array [1.0 2.0]) (a/array [1.0 2.0 3.0])))))
        (when (= :gpu ((requiring-resolve 'gen.backend.mlx/device) b))
          (is (thrown? clojure.lang.ExceptionInfo
                       (a/with-backend b (a/->clj (a/array [1.0] {:dtype :float64}))))
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
        (let [f (closure (fn [^longs ins]
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
          x (a/with-backend b (a/+ (a/array [1.0 2.0]) 1.0))]
      (testing "an array made on one thread is evaluated on another"
        (is (= [2.0 3.0] @(future (a/->clj x)))))
      (testing "concurrent evaluation"
        (is (= (repeat 8 6.0)
               (map deref (doall (for [_ (range 8)]
                                   (future (a/item (a/with-backend b (a/sum (a/* (a/array [1.0 2.0 3.0]) 1.0))))))))))))))

(deftest unreachable-arrays-are-freed
  (when-mlx
    (let [b (mlx)
          memory (requiring-resolve 'gen.backend.mlx/memory)
          live (fn [] (System/gc) (Thread/sleep 50) (gen.mlx.MlxRuntime/drain) (:active (memory)))
          base (live)]
      ;; 200 evaluated [100000] float32 arrays: 80 MB if none were freed
      (dotimes [_ 200]
        (a/eval! (a/with-backend b (a/+ (a/zeros [100000]) 1.0))))
      (is (< (- (live) base) 8e6) "unreachable arrays are freed"))))

(deftest array-pressure-collects
  (when-mlx
    (let [b (mlx)
          [live0 collections0] (vec (gen.mlx.MlxRuntime/lifetimeStats))]
      ;; 300,000 unevaluated scalars, kept unreachable: more than the
      ;; collection threshold, and close to Metal's buffer limit if none
      ;; were freed
      (dotimes [_ 300000]
        (a/with-backend b (a/+ (a/array 1.0) 1.0)))
      (let [[live collections] (vec (gen.mlx.MlxRuntime/lifetimeStats))]
        (is (< live 250000) "live arrays stay below the threshold's reach")
        (is (> collections collections0) "pressure collected")))))

(deftest compile-traces-once
  (when-mlx
    (let [b (mlx)
          traces (atom 0)
          f (a/compile (fn [x] (swap! traces inc) (a/sum (a/exp x))))]
      (a/with-backend b
        (let [x (a/array [0.0 1.0 2.0])]
          (dotimes [_ 5] (a/->clj (f x)))
          (is (= 1 @traces) "traced on the first call, replayed after")
          (f (a/array [1.0 2.0]))
          (is (= 2 @traces) "a new shape traces again")
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"reads an array's value on the host"
                                ((a/compile (fn [x] (if (a/truthy? (a/> (a/sum x) 0)) x (a/negate x)))) x))
              "a host read inside is loud"))))))
