(ns gen.backend.mlx.transforms
  "MLX's function transformations for the MLX backend (gen.backend.mlx),
   each on an MLX closure of Clojure code: its own gradients
   (mlx_value_and_grad, gen.mlx.grad=native), custom VJPs (the exact
   routes' and linear algebra's derivatives in native mode, and
   array/custom-vjp's), and compile (mlx_compile)."
  {:api :internal}
  (:require [gen.array.dtype :as dt]
            [gen.array.tree :as tree]
            [gen.backend.mlx.array :refer [consume! device fence handle mlx-array? stream wrap wrap-all with-device]]
            [gen.backend.mlx.memory :refer [evaluating]]
            [gen.backend.mlx.operands :refer [as-array dtype-on host? made mlx-error retained shape-of with-temps]])
  (:import [gen.mlx Mlx MlxRaw MlxRuntime MlxException]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Gradients: MLX's own (mlx_value_and_grad), with f as an MLX closure of
;; the differentiated leaves
;; ---------------------------------------------------------------------------

(def grad-mode
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

(defn native-value-and-grad
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
                in (t :vector (MlxRuntime/vectorOf (long-array (map #(handle %) xs))))
                [vs gs] (mapv #(t :vector %) (with-default-stream b #(Mlx/mlx_closure_value_and_grad_apply vag in)))
                v (first (wrap-all b (MlxRuntime/vectorArrays vs)))
                gs (wrap-all b (MlxRuntime/vectorArrays gs))]
            [v (tree/refill trees gs)])
          (finally (fence xs)))))))

;; ---------------------------------------------------------------------------
;; Custom VJPs: a function and its VJP, each an MLX closure
;; ---------------------------------------------------------------------------

(def ^:dynamic *tracing-compile*
  "True while mlx_compile traces a function (compile-fn). Custom VJPs stay
   out of compiled graphs: MLX keeps those in a thread-local cache, and its
   destruction at a thread's exit releases the custom functions' Java
   closures by an upcall from the exiting thread, which crashes the JVM
   (2026-10-19, SIGILL at exit)."
  false)

(defn custom-vjps? [] (and (= grad-mode :native) (not *tracing-compile*)))

(defn tracing-compile? [] *tracing-compile*)

(def ^:dynamic *differentiating*
  "True while native-value-and-grad runs its function: a custom VJP left out
   of a compiled graph there would give a wrong gradient."
  false)

(def ^:private default-backend-var (delay (requiring-resolve 'gen.array/*backend*)))

(defn custom-vjp
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
              in (t :vector (MlxRuntime/vectorOf (long-array (map #(handle %) ins))))
              ;; MLX makes the custom function's node on the default device
              out (t :vector (with-default-stream b #(Mlx/mlx_closure_apply cv in)))]
          (wrap-all b (MlxRuntime/vectorArrays out)))
        (finally (fence ins))))))

;; ---------------------------------------------------------------------------
;; Compile: mlx_compile of an MLX closure
;; ---------------------------------------------------------------------------

(defonce ^:private husk
  ;; what a donated array's handle holds after: one 0-d array for all
  (delay (made "mlx_array_new_float" #(Mlx/mlx_array_new_float (float 0.0)))))

(defn donate!
  "Consume the MlxArray x, which a compiled function's call read
   (array/compile's :donate): every later read of its handle is loud
   (:donated), and the handle is repointed at a shared 0-d husk, dropping
   its reference to x's data. The graph that reads x then holds that data
   alone, and MLX donates the buffer to the op that consumes it (its
   copy_gpu reuses a buffer only one array references): a scatter update
   of a static cache writes in place. The handle is not freed here: it
   stays valid (holding the husk), and x's phantom reference frees it
   once, as any array's, so nothing is freed twice and no handle MLX
   holds is freed. Returns nil."
  [x]
  (consume! x (fn [^long h] (MlxRuntime/repoint h (long @husk))))
  nil)

(defn compile-fn
  "p/-compile: f (a fn of a vector of MlxArrays to a vector of arrays) as an
   MLX closure, compiled with an id of its own; the returned fn applies it,
   from any thread. MLX traces it once per thread and input signature, and
   its traces are erased from every thread's compile cache when it is freed
   (MlxRuntime.Compiled, genclj-hfwo). The inputs at the positions
   :donate (a set) are donated once a call has returned its outputs
   (donate!): consumed, so that MLX may reuse their buffers in the
   evaluation of the outputs."
  [b f {:keys [shapeless? donate]}]
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
            (let [in (MlxRuntime/vectorOf (long-array (map #(handle %) xs)))
                  outs (try
                         ;; nothing between enter, the call and exit may
                         ;; block: a virtual thread could move to another
                         ;; carrier, whose cache is another
                         (.enter c)
                         (let [out (try (try (Mlx/mlx_closure_apply compiled in)
                                             (finally (.exit c)))
                                        (catch MlxException e (throw (mlx-error :compile xs e))))]
                           (try (wrap-all b (MlxRuntime/vectorArrays out))
                                (finally (MlxRaw/mlx_vector_array_free out))))
                         (finally
                           (MlxRaw/mlx_vector_array_free in)
                           (fence xs)))]
              ;; after the call, which returned: the outputs' graph now
              ;; holds the donated inputs, and the handles here let go
              (doseq [i donate] (donate! (nth xs i)))
              outs))]
    (MlxRuntime/adoptCompiled g c)
    g))
