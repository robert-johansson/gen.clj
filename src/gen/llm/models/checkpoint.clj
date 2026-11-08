(ns gen.llm.models.checkpoint
  "Hugging Face checkpoints for gen.llm.models' families (qwen3/load and
   qwen35/load take one where they take a GGUF file): a checkpoint's
   directory, its config.json and its safetensors (model.safetensors, or
   the shards of model.safetensors.index.json), read by
   gen.array/load-safetensors.

   - config.json is read by gen.array.json; each family maps it to the
     metadata llama.cpp's converter (convert_hf_to_gguf.py) writes, and the
     tensors to the names, layouts and conventions its GGUF holds, so that
     the family's forward reads a checkpoint as it reads that GGUF.
   - MLX's quantized checkpoints (mlx-lm's, mlx-community's): a quantized
     matrix is three tensors, <base>.weight (uint32 words), <base>.scales
     and <base>.biases, MLX's affine quantization, its bits and group size
     config.json's \"quantization\" (or the \"quantization_config\" mlx-lm
     writes beside it), {\"group_size\" \"bits\"}, with an entry of its own
     by <base> for a matrix quantized otherwise (false: not quantized).
     Each is read as gen.array/load-gguf's quantized matrices are, {:wq
     :scales :biases :bits :group-size}, under <base>.weight. Another
     quantization (GPTQ, AWQ, float8 …, a \"quant_method\") is loud
     (:unsupported).

   Transforms of the converter's (the families' docstrings say which they
   take), as gen.array values: shift-norm (Qwen3-Next's RMSNorm weights,
   which the converter adds 1 to), neg-exp (ssm_a = -exp(A_log)) and
   tiled (value heads from grouped to tiled order). Internal: the
   families' namespaces are the API."
  {:api :internal}
  (:refer-clojure :exclude [+ - * / < > <= >= == read])
  (:require [clojure.core :as c]
            [clojure.string :as str]
            [gen.array :as array :refer [+]]
            [gen.array.json :as json]
            [gen.llm.models.common :as common])
  (:import [java.nio.file Files LinkOption Path]))

(set! *warn-on-reflection* true)

(defn- path-of ^Path [path] (Path/of (str path) (make-array String 0)))

(defn checkpoint?
  "Is path a checkpoint's (a directory, a .safetensors file or an index's
   .json) rather than a GGUF file's?"
  [path]
  (let [p (str path)]
    (or (Files/isDirectory (path-of p) (make-array LinkOption 0))
        (str/ends-with? p ".safetensors")
        (str/ends-with? p ".json"))))

(defn- directory
  "The checkpoint's directory: path, or the directory of its file."
  [path]
  (let [p (path-of path)]
    (if (Files/isDirectory p (make-array LinkOption 0))
      (str p)
      (str (or (.getParent (.toAbsolutePath p)) ".")))))

(defn- invalid [who message data]
  (ex-info (str who ": " message) (merge {:reason :invalid-args} data)))

;; ---------------------------------------------------------------------------
;; MLX's quantization
;; ---------------------------------------------------------------------------

(defn- scheme
  "A quantization entry of config.json, {:bits :group-size}, checked:
   MLX's affine mode, the bits and group sizes gen.array/dequantize takes."
  [who k q]
  (let [{bits "bits" gs "group_size" mode "mode"} q]
    (when-not (contains? #{nil "affine"} mode)
      (throw (ex-info (str who ": config.json's quantization " (pr-str mode) (when k (str " of " k))
                           " is not MLX's affine mode, the one gen.array computes")
                      {:reason :unsupported :mode mode})))
    (when-not (and (#{2 3 4 5 6 8} bits) (#{32 64 128} gs))
      (throw (invalid who (str "config.json's quantization" (when k (str " of " k)) " is not {\"bits\" 2-8 \"group_size\" 32, 64"
                               " or 128}: " (pr-str q))
                      {:quantization q})))
    {:bits (long bits) :group-size (long gs)}))

(defn quantization
  "config.json's MLX quantization: nil, or {:bits :group-size :overrides
   {base {:bits :group-size}, or false}}. Loud (:unsupported) for another
   quantization than MLX's affine one."
  [who config]
  (let [q (or (get config "quantization") (get config "quantization_config"))]
    (when q
      (when-not (map? q)
        (throw (invalid who (str "config.json's quantization is not an object: " (pr-str q)) {})))
      (when-let [method (get q "quant_method")]
        (throw (ex-info (str who ": the checkpoint is quantized by " (pr-str method) ", which gen.clj does not read"
                             " (MLX's affine quantization it does)")
                        {:reason :unsupported :quant-method method})))
      (assoc (scheme who nil q)
             :overrides (into {}
                              (keep (fn [[k v]]
                                      (cond (map? v) [k (scheme who k v)]
                                            (false? v) [k false])))
                              q)))))

(defn- quantized-matrix
  "The quantized matrix of base: its words, scales and biases, its bits
   and group size checked against their shapes."
  [who base {:keys [bits group-size]} wq scales biases]
  (let [sw (array/shape wq) ss (array/shape scales)
        in (quot (c/* 32 (long (peek sw))) (long bits))]
    (when-not (= :uint32 (array/dtype wq))
      (throw (invalid who (str base ".weight has scales and biases, but is " (array/dtype wq) ", not uint32") {:tensor base})))
    (when-not (and (seq sw) (= ss (array/shape biases)) (= (pop sw) (pop ss))
                   (zero? (rem (c/* 32 (long (peek sw))) (long bits)))
                   (c/== (long (peek ss)) (quot in (long group-size))) (zero? (rem in (long group-size))))
      (throw (invalid who (str base "'s words " sw ", scales " ss " and biases " (array/shape biases) " are not a matrix of "
                               bits " bits in groups of " group-size)
                      {:tensor base :bits bits :group-size group-size})))
    {:wq wq :scales scales :biases biases :bits bits :group-size group-size}))

(defn- quantized
  "tensors with each quantized matrix's three tensors as one {:wq :scales
   :biases :bits :group-size} under <base>.weight, by the quantization q."
  [who tensors q]
  (let [bases (keep #(when (str/ends-with? % ".scales") (subs % 0 (c/- (count %) (count ".scales")))) (keys tensors))]
    (when (and (seq bases) (nil? q))
      (throw (invalid who (str "the checkpoint holds quantized matrices (" (first bases) ".scales) but config.json no"
                               " quantization")
                      {:tensor (first bases)})))
    (reduce (fn [ts base]
              (let [w (str base ".weight") s (str base ".scales") b (str base ".biases")
                    scheme (get-in q [:overrides base] q)]
                (when-not (and (contains? ts w) (contains? ts b))
                  (throw (invalid who (str s " has no " (if (contains? ts w) b w)) {:tensor base})))
                (when (false? scheme)
                  (throw (invalid who (str "config.json says " base " is not quantized, but it has scales") {:tensor base})))
                (-> ts
                    (dissoc s b)
                    (assoc w (quantized-matrix who base scheme (ts w) (ts s) (ts b))))))
            tensors bases)))

;; ---------------------------------------------------------------------------
;; Reading
;; ---------------------------------------------------------------------------

(defn config
  "config.json of the checkpoint at path (its directory, its .safetensors
   file or its index): its value, strings to values. Loud (:no-file)
   without one beside the tensors."
  [who path]
  (let [config-path (str (.resolve (path-of (directory path)) "config.json"))
        config (json/read-file config-path)]
    (when-not (map? config) (throw (invalid who (str config-path " is no JSON object") {})))
    config))

(defn read
  "The tensors of the checkpoint at path whose config.json is config:
   {:tensors :types :files}, :tensors by name (those select accepts, a
   predicate of names), a quantized matrix as {:wq :scales :biases :bits
   :group-size} under <base>.weight by config's quantization; :types and
   :files gen.array/load-safetensors's."
  [who path config select]
  (let [q (quantization who config)
        {:keys [tensors types files]} (array/load-safetensors path {:select select})]
    {:tensors (quantized who tensors q) :types types :files files}))

;; ---------------------------------------------------------------------------
;; The converter's transforms
;; ---------------------------------------------------------------------------

(defn float32
  "A config.json number as the GGUF's float32 metadata holds it (the
   converter writes eps and RoPE's base so): the double of its float32,
   nil for nil."
  [x]
  (when (some? x) (double (float x))))

(defn shift-norm
  "An RMSNorm weight of Qwen3-Next's form (the norm multiplies by 1 + w)
   as llama.cpp's GGUF holds it: 1 + w in float32, as transformers'
   Qwen3NextRMSNorm computes it (and the converter, which reads a
   checkpoint's bfloat16 as float32)."
  [w]
  (+ (array/astype w :float32) 1.0))

(defn neg-exp
  "ssm_a as llama.cpp's GGUF holds it from a checkpoint's A_log:
   -exp(A_log) in float32."
  [a-log]
  (array/negate (array/exp (array/astype a-log :float32))))

(defn- swap-groups
  "x with its axis (k·r·d long) read as [k r d] reordered to [r k d]:
   value heads grouped by key head (k groups of r heads of d values each,
   Hugging Face's order) to tiled (head j at j mod k's key head, ggml's
   broadcast order), as llama.cpp's converter reorders them
   (_reorder_v_heads)."
  [x axis k r d]
  (let [s (array/shape x)
        axis (long axis)
        split (vec (concat (subvec s 0 axis) [k r d] (subvec s (inc axis))))
        perm (assoc (vec (range (count split))) axis (inc axis) (inc axis) axis)]
    (when-not (c/== (long (nth s axis)) (c/* (long k) (long r) (long d)))
      (throw (ex-info (str "tiled: axis " axis " of " s " is not " k "·" r "·" d) {:reason :invalid-args :shape s})))
    (array/reshape (array/transpose (array/reshape x split) perm) s)))

(defn tiled
  "The weight w (an array, or a quantized matrix's map) with the value
   heads along axis (0, a matrix's rows, or 1, its columns; a vector's
   axis 0) reordered from grouped to tiled: k key heads, r value heads
   each, d values a head. A quantized matrix's rows reorder as its words',
   scales' and biases' rows; its columns, a head's words and groups at
   once, when a head's values fill whole words and groups (else loud,
   :unsupported)."
  [who w axis k r d]
  (cond
    (c/== 1 (long r)) w
    (not (map? w)) (swap-groups w axis k r d)
    (zero? (long axis)) (reduce #(update %1 %2 swap-groups 0 k r d) w [:wq :scales :biases])
    :else (let [{:keys [bits group-size]} w
                bits (long bits) gs (long group-size) d (long d)]
            (when-not (and (zero? (rem (c/* d bits) 32)) (zero? (rem d gs)))
              (throw (ex-info (str who ": value heads of " d " columns do not fill whole words and groups of a quantized"
                                   " matrix of " bits " bits in groups of " gs ", to reorder")
                              {:reason :unsupported :head-dim d :bits bits :group-size gs})))
            (-> w
                (update :wq swap-groups 1 k r (quot (c/* d bits) 32))
                (update :scales swap-groups 1 k r (quot d gs))
                (update :biases swap-groups 1 k r (quot d gs))))))

(defn rows
  "Rows start to stop - 1 of a weight [out in] (dense or quantized)."
  [w start stop]
  (if (map? w)
    (reduce #(update %1 %2 array/slice 0 start stop) w [:wq :scales :biases])
    (array/slice w 0 start stop)))

(defn concat-rows
  "Weights [out in] (dense, or quantized alike) joined by rows."
  [ws]
  (if (map? (first ws))
    (reduce (fn [m k] (assoc m k (array/concatenate (mapv k ws) 0))) (first ws) [:wq :scales :biases])
    (array/concatenate ws 0)))

(defn stack
  "Weights [out in] (dense, or quantized alike: one bits and group size)
   stacked [n out in], a stack of experts."
  [who ws]
  (if (map? (first ws))
    (do (when-not (and (every? map? ws) (apply = (map (juxt :bits :group-size) ws)))
          (throw (invalid who "experts quantized differently, or some not" {})))
        (reduce (fn [m k] (assoc m k (array/stack (mapv k ws) 0))) (first ws) [:wq :scales :biases]))
    (do (when (some map? ws) (throw (invalid who "experts quantized differently, or some not" {})))
        (array/stack ws 0))))

(defn output
  "tensors (llama.cpp's names) with the output projection as the
   checkpoint's config says: a tied output (tie_word_embeddings) drops
   lm_head (transformers ties it to the embedding at load; checkpoints such
   as Qwen3-0.6B's store a copy); an untied one must have it. Without the
   key, the checkpoint's tensors decide. Loud when config.json's
   vocab_size is not the embedding's rows."
  [who tensors tie? vocab-size]
  (let [embed (get tensors "token_embd.weight")]
    (when (and (false? tie?) (not (contains? tensors "output.weight")))
      (throw (ex-info (str who ": config.json says the output is not tied to the embedding, but the checkpoint has no lm_head")
                      {:reason :invalid-args})))
    (when (and vocab-size embed (not= (long vocab-size) (long (first (common/weight-shape embed)))))
      (throw (ex-info (str who ": config.json's vocab_size " vocab-size " is not the embedding's "
                           (first (common/weight-shape embed)) " rows")
                      {:reason :invalid-args :vocab-size vocab-size})))
    (if (true? tie?) (dissoc tensors "output.weight") tensors)))

(defn dense
  "w, checked to be a dense array (not a quantized matrix)."
  [who name w]
  (when (map? w)
    (throw (ex-info (str who ": tensor " name " is quantized; the forward reads it dense") {:reason :unsupported :tensor name})))
  w)
