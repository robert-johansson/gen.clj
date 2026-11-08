(ns gen.llm.convert
  "Language models converted between quantizations, by recipes:

     (convert! in out opts)   → the model of the GGUF file in, written at out as a GGUF file,
                                each tensor at the type its recipe gives
     (plan in opts)           → what convert! writes, each tensor's type, without writing
     (quantize gguf opts)     → gen.array/load-gguf's result with its matrices in MLX's affine
                                quantization, in memory, at any of MLX's bits and group sizes
     file-types               → llama.cpp's file types (llama-quantize's), recipes for convert!
     mlx-recipes              → mlx-lm's recipes (mlx_lm.convert's), for quantize

   The format written is GGUF, llama.cpp's: gen.clj loads it
   (gen.array/load-gguf) and so does llama.cpp, and it holds the
   tokenizer and the architecture beside the weights. Its quantized types
   are llama.cpp's, encoded by llama.cpp's own reference quantizers,
   ported (gen.array.gguf.quants): a file convert! writes is the one
   llama-quantize (v0.9.0, as built on macOS arm64) writes from the same
   source, tensor for tensor, byte for byte (test/gen/llm/convert_test.clj
   and convert_real_test.clj, against llama.cpp's quantizer: Qwen3-0.6B
   and Qwen3.5-0.8B whole). MLX's affine quantization at other bits or
   groups than GGUF's is no GGUF type: quantize holds it in memory, and no
   file does yet (genclj-k8gx: MLX's safetensors layout).

   A recipe (opts :recipe) is
   - a file type of file-types (:q4_k_m, :q8_0, :bf16, …): llama.cpp's
     choice of each tensor's type for it, llama_tensor_get_type's (v0.9.0,
     src/llama-quant.cpp), its rules for the dense and mixture-of-experts
     models gen.clj computes: e.g. Q4_K_M quantizes to Q4_K but the
     output (or a tied embedding) to Q6_K, and attn_v and ffn_down to
     Q6_K in the first eighth of the layers, the last eighth and every
     third between;
   - or a vector of rules [pattern value], tried in order, the first that
     matches a tensor deciding its type, and ending in a file type, which
     decides the tensors no rule matched (as llama-quantize's --tensor-type
     overrides come before its file type), or not, which keeps them.
     - A pattern is a string, a regular expression searched in the
       tensor's name (as --tensor-type's are), a java.util.regex.Pattern,
       or a function of the tensor's description {:name :type :shape
       :layer :layers} (:layer its block's index, nil outside blocks;
       :layers the model's blocks).
     - A value is :keep; a GGUF type (:f32 :f16 :bf16 :q8_0 :q4_0 :q4_1
       :q5_0 :q5_1 :q4_k :q5_k :q6_k); or {:bits :group-size}, MLX's
       affine form, as the GGUF type of that layout: 8 bits in groups of
       32 Q8_0 (its bias fixed at -128·scale), 4 and 5 bits in groups of
       32 Q4_1 and Q5_1 (a scale and a bias each, in float16). Any other
       is MLX-only, loud for a file (:unsupported): quantize holds it.
     E.g. [[\"^(token_embd|output)\\\\.weight$\" :q8_0] :q4_k_m]: the
     embedding and output at 8 bits, the rest as Q4_K_M.
   Whatever the recipe, the tensors llama.cpp never quantizes are kept as
   they are (tensor_allows_quantization: one-axis tensors, norms, a
   mixture's router, convolution kernels, …), and a tensor whose rows
   are no whole number of its type's blocks takes llama.cpp's fallback
   (Q4_K → Q5_0, Q5_K → Q5_1, Q6_K → Q8_0, then F16).

   Experimental: outside the 1.x promise (DESIGN §9)."
  {:api :experimental}
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.array.gguf.quants :as quants])
  (:import [java.nio ByteOrder]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.file Files OpenOption Path StandardOpenOption]))

(set! *warn-on-reflection* true)

(defn- invalid [message data] (ex-info (str "gen.llm.convert: " message) (merge {:reason :invalid-args} data)))
(defn- unsupported [message data] (ex-info (str "gen.llm.convert: " message) (merge {:reason :unsupported} data)))

(defn- check-options [f opts accepted]
  (when-not (or (nil? opts) (map? opts))
    (throw (ex-info (str "gen.llm.convert/" (name f) ": options must be a map, not " (pr-str opts))
                    {:reason :invalid-options :fn f})))
  (when-let [unknown (seq (remove accepted (keys opts)))]
    (throw (ex-info (str "gen.llm.convert/" (name f) ": unknown options " (vec unknown) "; it takes " (vec (sort accepted)))
                    {:reason :unknown-options :fn f :unknown (vec unknown) :accepted accepted}))))

;; ---------------------------------------------------------------------------
;; Tensors llama.cpp quantizes, and their categories (src/llama-quant.cpp)
;; ---------------------------------------------------------------------------

(def ^:private never-quantized
  "Parts of names whose tensors llama.cpp never quantizes
   (tensor_allows_quantization's, every architecture's)."
  ["_norm.weight" "ffn_gate_inp.weight" "ffn_gate_tid2eid.weight" "engram_gate.weight" "altup" "laurel"
   "per_layer_model_proj" "ssm_conv1d" "shortconv.conv.weight" "conv2d_spatial.weight" "conv_dw"
   "indexer.k_proj.weight" "indexer.q_proj.weight" "time_mix_first.weight" "time_mix_w0.weight"
   "time_mix_w1.weight" "time_mix_w2.weight" "time_mix_v0.weight" "time_mix_v1.weight" "time_mix_v2.weight"
   "time_mix_a0.weight" "time_mix_a1.weight" "time_mix_a2.weight" "time_mix_g1.weight" "time_mix_g2.weight"
   "time_mix_decay_w1.weight" "time_mix_decay_w2.weight" "time_mix_lerp_fused.weight" "attn_rel_b.weight"
   ".position_embd" "sam.pos_embd" "sam.neck." "sam.net_" ".rel_pos" ".patch_embd" ".patch_merger"
   "a.rvq.codebook" "mm.a.code_embd"])

(defn quantizable?
  "Does llama.cpp quantize the tensor {:name :shape} (its quantizer's
   tensor_allows_quantization, v0.9.0, the rules of every architecture;
   those of a few others gen.clj does not compute are not ported): a
   tensor of two axes or more named …weight, but norms, a mixture's
   router, positional embeddings, convolution kernels and the like."
  [{:keys [name shape]}]
  (and (>= (count shape) 2)
       (str/ends-with? name "weight")
       (not-any? #(str/includes? name %) never-quantized)
       (not (#{"position_embd.weight" "token_types.weight"} name))
       (not (str/starts-with? name "a.conv1d."))
       (not (str/starts-with? name "a.gen.wav.quant."))))

(defn- category
  "The tensor's category (tensor_get_category)."
  [name]
  (let [has? #(str/includes? name %)]
    (cond (= name "output.weight") :output
          (#{"token_embd.weight" "per_layer_token_embd.weight"} name) :token-embd
          (or (has? "attn_qkv.weight") (has? "attn_qkv_gate.weight")) :attn-qkv
          (has? "attn_kv_b.weight") :attn-kv-b
          (has? "attn_v.weight") :attn-v
          (has? "attn_k.weight") :attn-k
          (has? "attn_q.weight") :attn-q
          (has? "attn_output.weight") :attn-output
          (has? "ffn_up") :ffn-up
          (has? "ffn_gate") :ffn-gate
          (has? "ffn_latent") :ffn-latent
          (has? "attn_gate.weight") :attn-gate
          (has? "ffn_down") :ffn-down
          :else :other)))

(def ^:private attn-v-like #{:attn-v :attn-qkv :attn-kv-b})

(defn- layer-of
  "The block index of a tensor named blk.<i>.…, or nil."
  [name]
  (when-let [[_ i] (re-find #"^blk\.(\d+)\." name)]
    (Long/parseLong i)))

(defn- llama-order
  "Tensors in the order llama.cpp's quantizer reads them (its
   weight_name_comparer): by block, those outside blocks first, then by
   name."
  [tensors]
  (sort-by (juxt #(or (layer-of (:name %)) -1) :name) tensors))

(defn- more-bits?
  "llama.cpp's use_more_bits: the first eighth of n, the last eighth, and
   every third between."
  [^long i ^long n]
  (or (< i (quot n 8)) (>= i (quot (* 7 n) 8)) (== 2 (rem (- i (quot n 8)) 3))))

;; ---------------------------------------------------------------------------
;; File types
;; ---------------------------------------------------------------------------

(def file-types
  "llama.cpp's file types (llama_ftype) that convert! writes, as recipes:
   each the type of its tensors (llama_ftype_get_default_type) and its id
   (general.file_type's value). The K-quant mixtures choose more bits for
   some tensors (llama_tensor_get_type)."
  {:f32 {:type :f32 :id 0}
   :f16 {:type :f16 :id 1}
   :bf16 {:type :bf16 :id 32}
   :q8_0 {:type :q8_0 :id 7}
   :q4_0 {:type :q4_0 :id 2}
   :q4_1 {:type :q4_1 :id 3}
   :q5_0 {:type :q5_0 :id 8}
   :q5_1 {:type :q5_1 :id 9}
   :q4_k_s {:type :q4_k :id 14}
   :q4_k_m {:type :q4_k :id 15}
   :q5_k_s {:type :q5_k :id 16}
   :q5_k_m {:type :q5_k :id 17}
   :q6_k {:type :q6_k :id 18}})

(def ^:private plain-types #{:f32 :f16 :bf16})

(defn- block-of ^long [t] (long (:values (quants/types t))))

(defn- file-type-choice
  "llama_tensor_get_type_impl for file type ft (a key of file-types) and
   the quantizable tensor t: [its type, the counters after it]. The
   counters (state) are llama.cpp's: i-attn-v counts the attn_v-like
   tensors decided so far, i-ffn-down the ffn_down ones. Not ported: the
   rules of the IQ and Q2/Q3 file types, Falcon's, and the 70B models'
   attn_v (a model type llama.cpp knows from its parameters)."
  [ft {:keys [name shape]} {:keys [n-attn-v n-layers n-experts tied?]} state]
  (let [base (:type (file-types ft))
        cat (category name)
        nx (long (peek shape))
        {:keys [i-attn-v i-ffn-down]} state
        n-experts (long n-experts)]
    (cond
      (or (= cat :output) (and tied? (= cat :token-embd)))
      [(cond (not (zero? (rem nx (block-of base)))) :q8_0
             (not= base :q8_0) :q6_k
             :else base)
       state]

      (attn-v-like cat)
      [(let [t (cond (and (#{:q4_k_m :q5_k_m} ft) (more-bits? i-attn-v n-attn-v)) :q6_k
                     (and (= ft :q4_k_s) (< (long i-attn-v) 4)) :q5_k
                     :else base)]
         (if (== 8 n-experts) :q8_0 t))
       (update state :i-attn-v inc)]

      (= cat :attn-k)
      [(if (== 8 n-experts) :q8_0 base) state]

      (= cat :ffn-down)
      (let [layer (long (if (> n-experts 1) (or (layer-of name) 0) i-ffn-down))]
        [(cond (and (#{:q4_k_m :q5_k_m} ft) (more-bits? layer n-layers)) :q6_k
               (and (= ft :q4_k_s) (< layer (quot (long n-layers) 8))) :q5_k
               :else base)
         (update state :i-ffn-down inc)])

      (= cat :attn-output)
      [(if (and (== 8 n-experts) (#{:q4_k_s :q4_k_m} ft)) :q5_k base) state]

      :else [base state])))

(defn- fallback
  "t, or llama.cpp's fallback for a tensor whose rows of nx values are no
   whole number of t's blocks (tensor_type_fallback)."
  [t ^long nx]
  (if (zero? (rem nx (block-of t)))
    t
    (let [f (case t :q4_k :q5_0 :q5_k :q5_1 :q6_k :q8_0 t)]
      (if (zero? (rem nx (block-of f))) f :f16))))

;; ---------------------------------------------------------------------------
;; Recipes
;; ---------------------------------------------------------------------------

(def ^:private gguf-types (set (keys (dissoc quants/types :i32))))

(defn- affine-type
  "The GGUF type of MLX's affine quantization at bits and group-size, or
   loud: MLX-only."
  [{:keys [bits group-size] :as v}]
  (or (get {[8 32] :q8_0 [4 32] :q4_1 [5 32] :q5_1} [bits group-size])
      (throw (unsupported (str "MLX's affine quantization at " bits " bits in groups of " group-size
                               " is no GGUF type (Q8_0, Q4_1 and Q5_1 are 8, 4 and 5 bits in groups of 32,"
                               " and the K-quants 4, 5 and 6 bits in super-blocks of 256): MLX-only, which"
                               " gen.llm.convert/quantize holds in memory")
                          {:value v}))))

(defn- check-value [v where]
  (when-not (or (= :keep v) (gguf-types v)
                (and (map? v) (= #{:bits :group-size} (set (keys v)))
                     (#{2 3 4 5 6 8} (:bits v)) (#{32 64 128} (:group-size v))))
    (throw (invalid (str "a recipe's value is :keep, a GGUF type " (vec (sort gguf-types))
                         " or {:bits :group-size} (MLX's: 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or 128), not "
                         (pr-str v) " (" where ")")
                    {:value v}))))

(defn- matcher
  "A rule's pattern as a predicate of a tensor's description."
  [pattern]
  (cond (string? pattern) (let [re (re-pattern pattern)] #(boolean (re-find re (:name %))))
        (instance? java.util.regex.Pattern pattern) #(boolean (re-find pattern (:name %)))
        (fn? pattern) #(boolean (pattern %))
        :else (throw (invalid (str "a recipe's pattern is a string, a regex or a function, not " (pr-str pattern))
                              {:pattern pattern}))))

(defn- parse-recipe
  "recipe as {:rules [[predicate value] …] :file-type ft-or-nil}."
  [recipe file-type?]
  (let [[rules ft] (cond (keyword? recipe) [[] recipe]
                         (vector? recipe) (if (keyword? (peek recipe)) [(pop recipe) (peek recipe)] [recipe nil])
                         :else (throw (invalid (str "a recipe is a file type or a vector of rules, not " (pr-str recipe))
                                               {:recipe recipe})))]
    (when (and ft (not (file-type? ft)))
      (throw (invalid (str (pr-str ft) " is not a recipe known here") {:recipe ft})))
    {:rules (mapv (fn [rule]
                    (when-not (and (vector? rule) (= 2 (count rule)))
                      (throw (invalid (str "a recipe's rule is [pattern value], not " (pr-str rule)) {:rule rule})))
                    (let [[pattern v] rule]
                      (check-value v (pr-str pattern))
                      [(matcher pattern) v]))
                  rules)
     :file-type ft}))

(defn- described
  "A tensor's description for a rule's pattern."
  [{:keys [name type shape]} layers]
  {:name name :type type :shape shape :layer (layer-of name) :layers layers})

;; ---------------------------------------------------------------------------
;; Converting a GGUF file
;; ---------------------------------------------------------------------------

(defn- model-context
  "What llama.cpp's choices read of the model: its blocks and experts
   (<architecture>.block_count and .expert_count), its attn_v-like
   tensors, whether its output is tied to the embedding."
  [{:keys [metadata]} tensors]
  (let [arch (get metadata "general.architecture")]
    {:n-layers (long (or (get metadata (str arch ".block_count")) 0))
     :n-experts (long (max 1 (long (or (get metadata (str arch ".expert_count")) 0))))
     :n-attn-v (count (filter (comp attn-v-like category :name) tensors))
     :tied? (not-any? #(= "output.weight" (:name %)) tensors)}))

(defn- choose
  "Each tensor's type, by recipe: {name {:type :by}}, :by :never
   (llama.cpp never quantizes it), :rule (the index of the rule that
   matched), :file-type or :unmatched (kept)."
  [header tensors recipe]
  (let [{:keys [rules file-type]} (parse-recipe recipe file-types)
        ctx (model-context header tensors)]
    (first
     (reduce (fn [[out state] {:keys [name type shape] :as t}]
               (let [d (described t (:n-layers ctx))
                     rule (first (keep-indexed (fn [i [p v]] (when (p d) [i v])) rules))
                     quantize? (quantizable? t)
                     [target by state]
                     (cond (not quantize?) [type :never state]
                           rule (let [[i v] rule]
                                  [(cond (= :keep v) type (map? v) (affine-type v) :else v) i state])
                           file-type (let [[target state] (if (plain-types (:type (file-types file-type)))
                                                            [(:type (file-types file-type)) state]
                                                            (file-type-choice file-type t ctx state))]
                                       [target :file-type state])
                           :else [type :unmatched state])
                     target (if (and quantize? (not= target type) (quants/types target))
                              (fallback target (peek shape))
                              target)]
                 [(assoc out name {:type target :by by}) state]))
             [{} {:i-attn-v 0 :i-ffn-down 0}]
             (llama-order tensors)))))

(def ^:private source-floats #{:f32 :f16 :bf16})

(defn plan
  "What convert! writes for the model of the GGUF file in (its first
   file, of a split model) and opts' :recipe, without writing: a vector
   in the file's order of {:name :shape :from :to :by}, :from the source's
   type, :to the type written, :by what decided it (:never, llama.cpp
   never quantizes it; :file-type; :unmatched, kept; or the index of the
   rule that matched). Loud as convert! is, for a recipe or a tensor it
   refuses."
  [in {:keys [recipe allow-requantize?] :as opts}]
  (check-options :plan opts #{:recipe :allow-requantize?})
  (when-not recipe
    (throw (invalid "a :recipe is needed" {})))
  (let [hs (gguf/headers in)
        tensors (vec (mapcat :tensors hs))
        types (choose (first hs) tensors recipe)]
    (mapv (fn [{:keys [name type shape]}]
            (let [{to :type by :by} (types name)]
              (when-not (quants/types type)
                (throw (unsupported (str "tensor " name " is " (clojure.core/name type) ", a type not copied") {:tensor name :type type})))
              (when (and (not= to type) (not (source-floats type)) (not allow-requantize?))
                (throw (ex-info (str "gen.llm.convert: tensor " name " is " (clojure.core/name type)
                                     " already: requantizing it to " (clojure.core/name to)
                                     " loses more than quantizing its float original (:allow-requantize? true to do it,"
                                     " as llama-quantize's --allow-requantize)")
                                {:reason :requantize :tensor name :from type :to to})))
              {:name name :shape shape :from type :to to :by by}))
          tensors)))

(defn- map-file
  "The bytes [offset, offset + n) of the file at path, little-endian."
  [path ^long offset ^long n]
  (with-open [ch (FileChannel/open (Path/of (str path) (make-array String 0)) (into-array OpenOption [StandardOpenOption/READ]))]
    (doto (.map ch FileChannel$MapMode/READ_ONLY offset n)
      (.order ByteOrder/LITTLE_ENDIAN))))

(def ^:private dropped-keys
  #{"split.no" "split.count" "split.tensors.count" "general.file_type" "general.quantization_version"})

(defn- metadata-entries
  "The source's metadata as write-gguf's entries, in its order: a split's
   keys dropped (one file is written), general.quantization_version 2
   (GGML_QNT_VERSION) and general.file_type the file type's id, at the
   end, as llama-quantize sets them; or no general.file_type for a recipe
   of rules alone (llama.cpp guesses it)."
  [{:keys [metadata value-types] order :keys} recipe]
  (let [ft (cond (keyword? recipe) recipe (keyword? (peek recipe)) (peek recipe))]
    (cond-> (into [] (comp (remove dropped-keys) (map (fn [k] [k (value-types k) (metadata k)]))) order)
      true (conj ["general.quantization_version" :uint32 2])
      ft (conj ["general.file_type" :uint32 (:id (file-types ft))]))))

(defn convert!
  "Writes the model of the GGUF file in (its first file, of a split
   model) at out as one GGUF file, each tensor at the type opts' :recipe
   gives it (the namespace's docstring): a tensor kept, or at its own
   type, copied byte for byte; another one's values (gen.array.gguf/
   host-values: a quantized source's dequantized exactly) encoded by
   llama.cpp's reference quantizer for its type (gen.array.gguf.quants),
   on the host, a tensor at a time. The metadata is in's (the tokenizer
   and the architecture's among it), with general.file_type and
   general.quantization_version as llama-quantize sets them. Options:
   :recipe (needed); :allow-requantize? (false), to quantize a tensor
   that is quantized already, which loses more than quantizing its float
   original: else loud (:requantize), as llama-quantize is. Returns
   {:path :tensors :bytes}, :tensors plan's. Needs no backend: the file
   is read, encoded and written on the host."
  [in out opts]
  (let [p (plan in opts)
        hs (gguf/headers in)
        infos (into {} (for [h hs, t (:tensors h)] [(:name t) (assoc t :path (:path h))]))
        tensors (mapv (fn [{:keys [name shape from to]}]
                        (let [{:keys [path offset bytes] :as info} (infos name)]
                          (if (= from to)
                            {:name name :type to :shape shape :bytes #(map-file path offset bytes)}
                            {:name name :type to :shape shape :data #(gguf/host-values path info)})))
                      p)]
    (gguf/write-gguf out {:metadata (metadata-entries (first hs) (:recipe opts)) :tensors tensors})
    {:path (str out) :tensors p :bytes (Files/size (Path/of (str out) (make-array String 0)))}))

;; ---------------------------------------------------------------------------
;; MLX's affine quantization, in memory
;; ---------------------------------------------------------------------------

(def mlx-recipes
  "mlx-lm's recipes (mlx_lm.convert's -q, and its --quant-predicate
   mixed_4_6 and mixed_3_6), for quantize, in groups of 64 as mlx-lm's
   default: each a vector of rules, MLX's affine form at the bits shown.
   - :mlx-4bit and :mlx-8bit: every quantizable matrix at 4 or 8 bits;
   - :mlx-mixed-4-6 and :mlx-mixed-3-6: 6 bits for the output (lm_head)
     and, in the layers use_more_bits picks (as llama.cpp's Q4_K_M does),
     attn_v and ffn_down (v_proj, down_proj); 4 or 3 bits for the rest."
  (let [high? (fn [{:keys [name layer layers]}]
                (and layer (re-find #"attn_v\.weight|ffn_down" name) (more-bits? layer layers)))
        mixed (fn [low] [[#"^output\.weight$" {:bits 6 :group-size 64}]
                         [high? {:bits 6 :group-size 64}]
                         [#"" {:bits low :group-size 64}]])]
    {:mlx-4bit [[#"" {:bits 4 :group-size 64}]]
     :mlx-8bit [[#"" {:bits 8 :group-size 64}]]
     :mlx-mixed-4-6 (mixed 4)
     :mlx-mixed-3-6 (mixed 3)}))

(defn- shape-of
  "The shape a loaded tensor stands for: an array's, or a quantized
   matrix's once dequantized."
  [t]
  (if (map? t)
    (let [s (array/shape (:wq t))]
      (conj (pop s) (quot (* (long (peek s)) 32) (long (:bits t)))))
    (array/shape t)))

(defn- dense
  "A loaded tensor's values, dense in float32."
  [t]
  (if (map? t)
    (array/dequantize (:wq t) (:scales t) (:biases t) (assoc (select-keys t [:bits :group-size]) :dtype :float32))
    (array/astype t :float32)))

(defn quantize
  "gguf (gen.array/load-gguf's result) with the tensors opts' :recipe
   quantizes in MLX's affine layout, in memory (gen.array/quantize): each
   {:wq :scales :biases :bits :group-size}, as load-gguf gives a quantized
   one and the models multiply them (quantized-matmul), its :types entry
   :affine. Any of MLX's layouts, those no GGUF type holds among them.
   :recipe is a key of mlx-recipes or a vector of rules (the namespace's
   docstring) whose values are :keep or {:bits :group-size}; no file
   type. A quantized tensor is dequantized first (to float32) and
   quantized again; the scales and biases are float32. As mlx-lm, a
   matrix whose rows are no whole number of groups is kept, and so are
   the tensors llama.cpp never quantizes (norms and the like)."
  [{:keys [tensors metadata] :as gguf} {:keys [recipe] :as opts}]
  (check-options :quantize opts #{:recipe})
  (when-not recipe (throw (invalid "a :recipe is needed" {})))
  (let [recipe (get mlx-recipes recipe recipe)
        {:keys [rules]} (parse-recipe recipe #{})
        _ (doseq [[_ v] rules]
            (when-not (or (= :keep v) (map? v))
              (throw (invalid (str "quantize's recipe values are :keep or {:bits :group-size}, not " (pr-str v)
                                   " (convert! writes the GGUF types)")
                              {:value v}))))
        arch (get metadata "general.architecture")
        layers (long (or (get metadata (str arch ".block_count")) 0))
        out (into {}
                  (map (fn [[name t]]
                         (let [shape (shape-of t)
                               d (described {:name name :type nil :shape shape} layers)
                               v (some (fn [[p v]] (when (p d) v)) rules)]
                           (if (and (map? v) (quantizable? d) (zero? (rem (long (peek shape)) (long (:group-size v)))))
                             (let [[wq scales biases] (array/quantize (dense t) v)]
                               [name {:wq wq :scales scales :biases biases :bits (:bits v) :group-size (:group-size v)}])
                             [name t]))))
                  tensors)]
    (array/eval! (mapcat (fn [[name t]] (when-not (identical? t (tensors name)) [(:wq t) (:scales t) (:biases t)])) out))
    (assoc gguf
           :tensors out
           :types (into (:types gguf) (keep (fn [[name t]] (when-not (identical? t (tensors name)) [name :affine]))) out))))
