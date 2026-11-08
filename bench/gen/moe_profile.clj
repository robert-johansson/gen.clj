(ns gen.moe-profile
  "Where a Qwen3.5/3.6 mixture of experts' decode step goes (genclj-p0ad):
   the compiled paged batch step (gen.batch-profile's, the continuous
   mode's, states in place) of a qwen35moe GGUF on MLX, at several row
   counts, and its feed-forward's parts each compiled alone over every
   layer:

     clojure -J-Dgen.backend=mlx -M:bench -m gen.moe-profile <qwen35moe GGUF>

   One GPU JVM at a time. Per row count (gen.bench.rows, 1,8,16): the
   step (host and device, as gen.batch-profile's row), then, ms a call of
   each part compiled alone (40 layers of it), evaluated every 8 calls:
   - moe: the feed-forward (qwen35's);
   - router: the router's product and softmax;
   - routing: the top k of its probabilities and their renormalization;
   - experts: the gathered gate, up and down products (gather-qmm) and
     the weighted sum, for given experts;
   - gate-up, down: the gathered products alone;
   - shared: the shared expert and its gate.
   gen.bench.dtype (float16, the engine bench's). With KPROF_LIB set (the
   path of a kprof library injected by CUDA_INJECTION64_PATH, which
   exports kprof_mark), each timed loop is marked, so its kernels are
   counted and named."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.llm.host :as host]
            [gen.llm.models.common :as common]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.random :as rng])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandle]))

(set! *warn-on-reflection* true)

(defn- prop [k default] (System/getProperty (str "gen.bench." k) (str default)))

(def ^:private settings
  {:rows (mapv parse-long (str/split (prop "rows" "1,8,16") #","))
   :dtype (keyword (prop "dtype" "float16"))
   :steps (parse-long (prop "steps" "48"))
   :parts (set (map keyword (str/split (prop "parts" "moe,router,routing,experts,gate-up,gate-up-4,gate-up-7,down,down-4,shared") #",")))})

;; ---------------------------------------------------------------------------
;; kprof's marks
;; ---------------------------------------------------------------------------

(def ^:private ^MethodHandle mark-handle
  (delay (when-let [lib (System/getenv "KPROF_LIB")]
           (let [f (.find (SymbolLookup/libraryLookup ^String lib (Arena/global)) "kprof_mark")]
             (when (.isPresent f)
               (.downcallHandle (Linker/nativeLinker) (.get f)
                                (FunctionDescriptor/ofVoid (into-array java.lang.foreign.MemoryLayout [ValueLayout/ADDRESS ValueLayout/JAVA_INT]))
                                (make-array java.lang.foreign.Linker$Option 0)))))))

(defn- mark!
  "kprof's totals since the last mark, under label, for calls calls."
  [label ^long calls]
  (when-let [^MethodHandle h @mark-handle]
    (with-open [arena (Arena/ofConfined)]
      (let [^java.util.List args [(.allocateFrom arena ^String label) (int calls)]]
        (.invokeWithArguments h args)))))

;; ---------------------------------------------------------------------------

(def ^:private batch-profile-var (fn [s] @(requiring-resolve (symbol "gen.batch-profile" s))))

(defn- part-ms
  "ms a call of the compiled f of args, evaluated every 8 calls, after 4."
  [label f args]
  (dotimes [_ 4] (array/eval! (apply f args)))
  (mark! "warm" 4)
  (let [n 32
        t0 (System/nanoTime)]
    (loop [i 0 outs []]
      (if (< i n)
        (let [o (apply f args)
              outs (conj outs o)]
          (if (zero? (mod (inc i) 8))
            (do (array/eval! outs) (recur (inc i) []))
            (recur (inc i) outs)))
        (let [ms (/ (- (System/nanoTime) t0) 1e6 n)]
          (mark! label n)
          ms)))))

(defn- chunked
  "gathered's product of the quantized stack w for the pairs of experts
   idx [N k] and x1's matrices lhs, in gather-qmm calls of at most c pairs
   (MLX's CUDA gather_qmm takes its qmv kernel below 8 rows)."
  [x1 w idx lhs c]
  (let [[n k] (array/shape idx)
        b (* (long n) (long k))
        fi (array/reshape idx [b])
        fl (array/reshape (array/broadcast-to lhs [n k]) [b])
        outs (mapv (fn [^long s]
                     (let [e (min b (+ s (long c)))]
                       (array/gather-qmm x1 (:wq w) (:scales w) (:biases w)
                                         (assoc (select-keys w [:bits :group-size])
                                                :lhs-indices (array/slice fl 0 s e) :rhs-indices (array/slice fi 0 s e)))))
                   (range 0 b (long c)))]
    (array/reshape (array/concatenate outs 0) [n k 1 -1])))

(defn- parts
  "The feed-forward's parts at r rows over every layer: {part ms}."
  [{:keys [config weights]} ^long r]
  (let [{:keys [hidden dtype experts-used experts]} config
        es (mapv :moe (:layers weights))
        k (long experts-used)
        moe @#'qwen35/moe
        routing @#'qwen35/routing
        gathered @#'qwen35/gathered
        x (array/astype (rng/normal (rng/key 1) [r 1 hidden]) dtype)
        xs (array/reshape x [r hidden])
        x1 (array/reshape xs [r 1 1 hidden])
        token (array/reshape (array/arange 0 r 1 {:dtype :int32}) [r 1])
        flat (array/reshape (array/arange 0 (* r k) 1 {:dtype :int32}) [r k])
        probs (array/softmax (rng/normal (rng/key 2) [r experts]) {:axis -1})
        [idx p] (routing probs k)
        p (array/astype p dtype)
        act (array/astype (rng/normal (rng/key 3) [r k 1 (:expert-ffn config)]) dtype)
        _ (array/eval! [x probs idx p act])
        fs {:moe [(fn [x] (mapv #(moe config % x) es)) [x]]
            :router [(fn [xs] (mapv #(array/softmax (array/astype (common/linear xs (:router %)) :float32) {:axis -1}) es)) [xs]]
            :routing [(fn [probs] (mapv (fn [_] (routing probs k)) es)) [probs]]
            :experts [(fn [x1 idx p]
                        (mapv (fn [{:keys [gate up down]}]
                                (let [a (* (common/silu (gathered x1 gate idx token)) (gathered x1 up idx token))
                                      y (array/squeeze (gathered a down idx flat) [2])]
                                  (array/sum (* y (array/expand-dims p -1)) {:axis 1})))
                              es))
                      [x1 idx p]]
            :gate-up [(fn [x1 idx] (mapv (fn [{:keys [gate up]}] [(gathered x1 gate idx token) (gathered x1 up idx token)]) es)) [x1 idx]]
            :gate-up-4 [(fn [x1 idx] (mapv (fn [{:keys [gate up]}] [(chunked x1 gate idx token 4) (chunked x1 up idx token 4)]) es)) [x1 idx]]
            :gate-up-7 [(fn [x1 idx] (mapv (fn [{:keys [gate up]}] [(chunked x1 gate idx token 7) (chunked x1 up idx token 7)]) es)) [x1 idx]]
            :down-4 [(fn [act idx] (mapv (fn [{:keys [down]}] (chunked act down idx flat 4)) es)) [act idx]]
            :down [(fn [act idx] (mapv (fn [{:keys [down]}] (gathered act down idx flat)) es)) [act idx]]
            :shared [(fn [xs] (mapv (fn [{:keys [shared shared-gate]}]
                                      (* (common/swiglu shared xs)
                                         (array/sigmoid (common/linear xs (array/reshape shared-gate [1 hidden])))))
                                    es))
                     [xs]]}]
    (into (sorted-map)
          (keep (fn [[part [f args]]]
                  (when ((:parts settings) part)
                    [part (part-ms (str (name part) " " r) (array/compile f) args)])))
          fs)))

(defn- step-row
  "The compiled batch step at r rows (gen.batch-profile's row, lane), then
   its floor, marked."
  [m pool ^long r]
  (let [profile-row (batch-profile-var "profile-row")
        floor-ms (batch-profile-var "floor-ms")
        stepper (batch-profile-var "stepper")
        start (batch-profile-var "start")
        res (profile-row m pool r :lane)
        _ (mark! "warm" 0)
        n (long (:steps settings))
        floor (floor-ms (stepper m :lane) (start m r) r 64 n)]
    (mark! (str "step " r) n)
    (assoc res :floor-marked floor)))

(defn -main [path & _]
  (when-not (= :mlx (array/backend-name array/*backend*))
    (throw (ex-info "Run on MLX: -J-Dgen.backend=mlx" {})))
  (let [{:keys [rows dtype]} settings
        m (qwen35/load path {:dtype dtype})
        _ (array/eval! (:weights m))
        pool (host/pool "profile rows-" (host/default-threads))]
    (println "## A mixture of experts' decode step (gen.moe-profile)\n")
    (println "- model:" path)
    (println "- settings:" (pr-str settings) "kprof:" (some? @mark-handle))
    (println "- top-k by ranks:" (some? (resolve 'gen.llm.models.qwen35/ranked)))
    (println "- os:" (System/getProperty "os.name") (System/getProperty "os.arch") "; java" (System/getProperty "java.version"))
    (mark! "load" 0)
    (when-not (= "false" (prop "step" "true"))
      (let [cols [:rows :step :host :mlx-call :encode :wait :read :normalize :draw :floor :floor-marked :traces :gc :mlx-gc]
            res (mapv #(step-row m pool %) rows)]
        (println "\n### The step (ms)\n")
        (println (str "| " (str/join " | " (map name cols)) " |"))
        (println (str "|" (str/join "|" (repeat (count cols) "---:")) "|"))
        (doseq [r res]
          (println (str "| " (str/join " | " (map #(let [v (% r)] (if (double? v) (format "%.2f" v) (str v))) cols)) " |")))))
    (let [ps (mapv #(parts m %) rows)]
      (println "\n### The feed-forward's parts over 40 layers, each compiled alone (ms a call)\n")
      (println (str "| part | " (str/join " | " (map #(str % " rows") rows)) " |"))
      (println (str "|---|" (str/join "|" (repeat (count rows) "---:")) "|"))
      (doseq [k (keys (first ps))]
        (println (str "| " (name k) " | " (str/join " | " (map #(format "%.2f" (double (get % k))) ps)) " |"))))
    (println "\n- MLX memory:" (pr-str ((requiring-resolve 'gen.backend.mlx.memory/memory))))
    (.shutdown ^java.util.concurrent.ExecutorService pool)
    (shutdown-agents)))
