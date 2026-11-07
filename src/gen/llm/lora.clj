(ns gen.llm.lora
  "LoRA (Hu et al. 2021, \"LoRA: Low-Rank Adaptation of Large Language
   Models\"): low-rank adapters trained beside a model's frozen weights,
   for the model families of gen.llm.models (Qwen3's):

     (init m opts key)            → an adapter {:config :params}, A random and B zero
     (adapt m adapter)            → m with the adapter beside its weights
     (adapt m adapter {:key k})   → … its dropout drawn from k, for a training step
     (merge m adapter)            → m with the adapter merged into dense weights
     (save-gguf adapter path)     → llama.cpp's LoRA GGUF, for gen.llm.llama/adapter
     (load-gguf path)             → an adapter from such a file
     (adapted? w), (delta x w)    → an adapted weight and its term, for a family's forward

   A projection's weight w [out in] (y = x·wᵀ) gets an adapter of rank r,
   A [r in] and B [out r], and its adapted product is

     y = x·wᵀ + s·(x·Aᵀ)·Bᵀ,   s = scale · alpha / r,

   llama.cpp's convention (llama-adapter.h's get_scale, llama-graph.cpp's
   build_lora_mm: B (A x), scaled) and PEFT's (lora_alpha / r); scale is
   adapt's :scale (1), as llama.cpp's engines take one per adapter
   (gen.llm.llama/engine's :adapters). w itself is frozen, dense or
   quantized: quantized-matmul differentiates with respect to x only, so
   gradients reach the adapters of the layers below it (QLoRA's).

   An adapter is a value: {:config {:architecture :rank :alpha :dropout}
   :params {layer {target {:a A :b B}}}}, layer a block's index and target
   a projection of targets (:q :k :v :o :gate :up :down). Its :params are
   gen.learning's store, a tree of arrays: a training step takes their
   gradient through the adapted model's loss and steps them (gen.learning
   ascends, so along the loss's negated gradient):

     (let [loss (fn [params k]
                  (qwen3/nll (lora/adapt m (assoc adapter :params params) {:key k})
                             tokens {:mask mask :checkpoint? true}))
           [l g] ((array/value-and-grad loss) (:params adapter) k)
           [params state] (learn/step (learn/adamw {:lr 1e-4}) state (:params adapter)
                                      (array/tree-map array/negate g))]
       (assoc adapter :params params))

   init draws A's entries uniform on [-1/√in, 1/√in] (PEFT's
   kaiming_uniform(a=√5), mlx-lm's) from a key per site, (fold-in
   (fold-in key layer) target's index), and sets B to zeros, so a new
   adapter changes nothing: its model's log-probabilities are the base's
   exactly. Dropout (:dropout p) zeroes each entry of the adapter's input
   with probability p, the rest scaled by 1/(1 - p), only when adapt is
   given a :key (a training step's): the masks are drawn from that key, a
   key per site again, so a checkpointed block recomputes the same masks.
   Without a key the adapter is deterministic, as an engine serves it.

   llama.cpp's LoRA GGUF (convert_lora_to_gguf.py's, gguf-py's bytes):
   metadata general.architecture (the model family's), general.type
   \"adapter\", adapter.type \"lora\" and adapter.lora.alpha (float32);
   tensors blk.<layer>.<name>.weight.lora_a and .lora_b, F32, <name>
   llama.cpp's (attn_q, …, ffn_down), A's GGML dimensions [in r] and B's
   [r out] (gen.array's shapes reversed), as llama.cpp checks them
   against the base's. llama.cpp's rank is B's ne[0], r.

   Experimental: outside the 1.x promise (DESIGN §9)."
  {:api :experimental}
  (:refer-clojure :exclude [+ - * / < > <= >= == merge])
  (:require [clojure.core :as c]
            [gen.array :as array :refer [+ * /]]
            [gen.array.gguf :as gguf]
            [gen.llm.models :as models]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(def targets
  "The projections an adapter may target: a layer's weight key in a model
   (gen.llm.models.qwen3's) and llama.cpp's tensor name for it."
  (array-map :q "attn_q" :k "attn_k" :v "attn_v" :o "attn_output" :gate "ffn_gate" :up "ffn_up" :down "ffn_down"))

(def ^:private target-index (zipmap (keys targets) (range)))

(def ^:private target-of (into {} (map (fn [[k v]] [v k])) targets))

;; ---------------------------------------------------------------------------
;; Checks
;; ---------------------------------------------------------------------------

(defn- invalid
  ([message] (invalid message {}))
  ([message data] (ex-info (str "lora: " message) (c/merge {:reason :invalid-args} data))))

(defn- unsupported
  ([message] (unsupported message {}))
  ([message data] (ex-info (str "lora: " message) (c/merge {:reason :unsupported} data))))

(defn- check-options
  "opts (a map or nil) holding only the keys accepted: else loud, as
   gen.clj's inference refuses options it does not read."
  [f opts accepted]
  (when-not (or (nil? opts) (map? opts))
    (throw (ex-info (str "lora/" (name f) ": options must be a map, not " (pr-str opts))
                    {:reason :invalid-options :fn f})))
  (when-let [unknown (seq (remove accepted (keys opts)))]
    (throw (ex-info (str "lora/" (name f) ": unknown options " (vec unknown) "; it takes " (vec (sort accepted)))
                    {:reason :unknown-options :fn f :unknown (vec unknown) :accepted accepted}))))

(defn- check-config
  "An adapter's :config, checked: a positive integer rank, a positive
   alpha, a dropout in [0, 1)."
  [{:keys [rank alpha dropout] :as config}]
  (when-not (map? config)
    (throw (invalid (str "an adapter is {:config :params}, its :config a map, not " (pr-str config)))))
  (when-not (pos-int? rank)
    (throw (invalid (str ":rank is a positive integer, not " (pr-str rank)) {:rank rank})))
  (when-not (and (number? alpha) (pos? (double alpha)) (Double/isFinite (double alpha)))
    (throw (invalid (str ":alpha is a positive number, not " (pr-str alpha)) {:alpha alpha})))
  (when-not (and (number? dropout) (c/<= 0.0 (double dropout)) (c/< (double dropout) 1.0))
    (throw (invalid (str ":dropout is a number in [0, 1), not " (pr-str dropout)) {:dropout dropout})))
  config)

(defn- check-params
  "An adapter's :params, checked against its rank: {layer {target {:a A
   :b B}}}, A [rank in] and B [out rank] arrays."
  [params rank]
  (when-not (and (map? params) (seq params))
    (throw (invalid (str "an adapter's :params are {layer {target {:a :b}}}, not " (pr-str params)))))
  (doseq [[i ts] params]
    (when-not (and (nat-int? i) (map? ts) (seq ts))
      (throw (invalid (str "an adapter's :params are {layer {target {:a :b}}}: " (pr-str i) " → " (pr-str ts)) {:layer i})))
    (doseq [[t ab] ts
            :let [{:keys [a b]} ab]]
      (when-not (contains? targets t)
        (throw (invalid (str "layer " i "'s target " (pr-str t) " is none of " (vec (keys targets))) {:layer i :target t})))
      (when-not (and (map? ab) (array/array? a) (array/array? b) (= 2 (array/ndim a)) (= 2 (array/ndim b)))
        (throw (invalid (str "layer " i "'s " t " is {:a A :b B}, two matrices") {:layer i :target t})))
      (when-not (and (= rank (first (array/shape a))) (= rank (second (array/shape b))))
        (throw (invalid (str "layer " i "'s " t ": A " (array/shape a) " and B " (array/shape b) " are not of rank " rank
                             " (A [rank in], B [out rank])")
                        {:layer i :target t :a (array/shape a) :b (array/shape b) :rank rank})))))
  params)

(defn- check-adapter [{:keys [config params] :as adapter}]
  (when-not (map? adapter)
    (throw (invalid (str "an adapter is {:config :params}, not " (pr-str adapter)))))
  (check-config config)
  (check-params params (:rank config))
  adapter)

;; ---------------------------------------------------------------------------
;; Weights
;; ---------------------------------------------------------------------------

(defn adapted?
  "Is the weight w one adapt made: {:base w' :lora {…}}?"
  [w]
  (and (map? w) (contains? w :lora)))

(defn- weight-shape
  "The shape [out in] a weight stands for: a dense one's, a quantized
   one's dequantized, an adapted one's base's."
  [w]
  (cond (adapted? w) (weight-shape (:base w))
        (map? w) (let [s (array/shape (:wq w))]
                   (conj (pop s) (quot (c/* (long (peek s)) 32) (long (:bits w)))))
        :else (array/shape w)))

(defn- layer-weight
  "The weight of target t in layer i of model m: loud when m has none."
  [m i t]
  (let [layers (get-in m [:weights :layers])]
    (when-not (and (vector? layers) (c/< -1 (long i) (count layers)))
      (throw (invalid (str "the model has no layer " i " (it has " (count layers) ")") {:layer i :layers (count layers)})))
    (or (get-in layers [i t])
        (throw (invalid (str "the model's layer " i " has no " t " weight") {:layer i :target t})))))

(defn- site-key
  "The key of the adapter on target t of layer i."
  [key i t]
  (rng/fold-in (rng/fold-in key i) (target-index t)))

(defn delta
  "The adapter's term of x through an adapted weight w (adapt's):
   s·(x'·Aᵀ)·Bᵀ, x' x under dropout when adapt drew it (a :key), else x.
   A model family's forward adds it to the base's product: x [.. in], the
   result [.. out]."
  [x w]
  (let [{:keys [a b s p key]} (:lora w)
        x (if key
            (let [keep (c/- 1.0 (double p))]
              (* x (/ (array/astype (rng/bernoulli key keep (array/shape x)) (array/dtype x)) keep)))
            x)]
    (* s (array/matmul (array/matmul x (array/transpose a)) (array/transpose b)))))

;; ---------------------------------------------------------------------------
;; Adapters
;; ---------------------------------------------------------------------------

(def ^:private defaults {:rank 8 :alpha 16.0 :dropout 0.0 :targets [:q :v]})

(defn init
  "A new adapter for model m (a gen.llm.models family's, as qwen3/load
   gives it): {:config {:architecture :rank :alpha :dropout} :params
   {layer {target {:a A :b B}}}}, each A [rank in] uniform on [-1/√in,
   1/√in] from (fold-in (fold-in key layer) target's index), each B [out
   rank] zeros, in the model's dtype. Options:
     :rank     r (default 8)
     :alpha    the scale's numerator, s = alpha / r (default 16)
     :dropout  p, the probability an entry of the adapter's input is
               zeroed in a training step (adapt's :key; default 0)
     :targets  the projections adapted, of targets (default [:q :v])
     :layers   the layers adapted, block indices (default every one; the
               last eight, mlx-lm's custom, are (range (- n 8) n))
   Loud for another key, a target or layer the model lacks, or a value
   out of range (:invalid-args)."
  [m opts key]
  (check-options :init opts #{:rank :alpha :dropout :targets :layers})
  (when-not (rng/key? key)
    (throw (invalid (str "init takes a key (gen.random/key), not " (pr-str key)))))
  (let [{:keys [rank alpha dropout layers] ts :targets} (c/merge defaults opts)
        n (count (get-in m [:weights :layers]))
        layers (if (contains? opts :layers) layers (range n))
        config (check-config {:architecture (models/architecture m) :rank rank
                              :alpha (when (number? alpha) (double alpha)) :dropout (when (number? dropout) (double dropout))})
        dtype (or (get-in m [:config :dtype]) (throw (invalid "the model has no :config :dtype" {})))]
    (when-not (and (sequential? ts) (seq ts) (every? #(contains? targets %) ts) (apply distinct? ts))
      (throw (invalid (str ":targets are distinct projections of " (vec (keys targets)) ", not " (pr-str ts))
                      {:targets ts})))
    (when-not (and (sequential? layers) (seq layers) (every? #(and (integer? %) (c/< -1 (long %) n)) layers)
                   (apply distinct? layers))
      (throw (invalid (str ":layers are distinct block indices below " n ", not " (pr-str layers)) {:layers layers})))
    {:config config
     :params (into {} (for [i layers]
                        [(long i) (into {} (for [t ts
                                                 :let [[out in] (weight-shape (layer-weight m i t))
                                                       bound (c// 1.0 (Math/sqrt (double in)))]]
                                             [t {:a (rng/uniform (site-key key i t) [rank in] {:low (c/- bound) :high bound :dtype dtype})
                                                 :b (array/zeros [out rank] {:dtype dtype})}]))]))}))

(defn adapt
  "Model m with adapter beside the weights it targets: each such weight w
   becomes {:base w :lora {…}}, whose product a family's forward computes
   as w's plus (delta x w), s = scale · alpha / rank. Every function of
   the family takes the result as a model (forward, prefill, step,
   compile-step, log-probs and nll; gen.llm.array-engine/engine's :model),
   and it is differentiable with respect to the adapter's :params, the
   base frozen. Adapting an adapted model adds the second adapter's term
   to the first's (llama.cpp's several adapters). Options:
     :scale  the adapter's strength, as llama.cpp's engines take one
             (default 1; 0 is the base model)
     :key    a training step's key: the adapter's dropout (its :config's
             :dropout, when above 0) is drawn from it; without one there
             is none
   Loud for an adapter that does not fit m: a layer or target it lacks,
   or matrices of other shapes (:invalid-args)."
  ([m adapter] (adapt m adapter nil))
  ([m adapter {:keys [scale key] :or {scale 1.0} :as opts}]
   (check-options :adapt opts #{:scale :key})
   (check-adapter adapter)
   (when-not (and (number? scale) (Double/isFinite (double scale)))
     (throw (invalid (str ":scale is a finite number, not " (pr-str scale)) {:scale scale})))
   (when-not (or (nil? key) (rng/key? key))
     (throw (invalid (str ":key is a gen.random key, not " (pr-str key)))))
   (let [{:keys [rank alpha dropout]} (:config adapter)
         s (c/* (double scale) (c// (double alpha) (double rank)))
         p (double dropout)]
     (reduce-kv (fn [m i ts]
                  (reduce-kv (fn [m t {:keys [a b]}]
                               (let [w (layer-weight m i t)
                                     [out in] (weight-shape w)]
                                 (when-not (and (= [rank in] (array/shape a)) (= [out rank] (array/shape b)))
                                   (throw (invalid (str "layer " i "'s " t " is [" out " " in "]: its A must be [" rank " " in
                                                        "] and its B [" out " " rank "], not " (array/shape a) " and " (array/shape b))
                                                   {:layer i :target t :weight [out in] :a (array/shape a) :b (array/shape b)})))
                                 (assoc-in m [:weights :layers i t]
                                           {:base w
                                            :lora (cond-> {:a a :b b :s s}
                                                    (and key (pos? p)) (assoc :p p :key (site-key key i t)))})))
                             m ts))
                m (:params adapter)))))

(defn- dense
  "The dense matrix [out in] a weight stands for, in dtype: an adapted
   one's base's plus s·B·A."
  [w dtype]
  (let [as (fn [x] (if (= dtype (array/dtype x)) x (array/astype x dtype)))]
    (cond (adapted? w) (let [{:keys [a b s]} (:lora w)]
                         (as (+ (dense (:base w) dtype) (* s (array/matmul b a)))))
          (map? w) (array/dequantize (:wq w) (:scales w) (:biases w) (assoc (select-keys w [:bits :group-size]) :dtype dtype))
          :else (as w))))

(defn merge
  "Model m with adapter merged into the weights it targets: each a dense
   matrix w + s·B·A in the model's dtype (a quantized base dequantized
   first), so the model computes what (adapt m adapter opts) does in one
   product a projection, with no adapter left. Options: :scale, as
   adapt's."
  ([m adapter] (merge m adapter nil))
  ([m adapter opts]
   (check-options :merge opts #{:scale})
   (let [dtype (get-in m [:config :dtype])]
     (update-in (adapt m adapter opts) [:weights :layers]
                (fn [layers]
                  (mapv (fn [layer] (into {} (map (fn [[k w]] [k (if (adapted? w) (dense w dtype) w)])) layer)) layers))))))

;; ---------------------------------------------------------------------------
;; llama.cpp's LoRA GGUF
;; ---------------------------------------------------------------------------

(defn- tensor-base [i t] (str "blk." i "." (targets t) ".weight"))

(defn save-gguf
  "Writes adapter at path as llama.cpp's LoRA GGUF (the namespace's
   docstring), which gen.llm.llama/adapter loads for the base model; an
   engine given it at scale 1 computes what (adapt m adapter) does. Its
   tensors are F32 (an adapter in float64 is rounded), by layer and
   target, in the order and bytes gguf-py writes them. Options: :name,
   the file's general.name. Loud (:invalid-args) for an adapter without
   an :architecture (llama.cpp refuses one of another than its model's).
   The file is complete when it appears (gen.array.gguf/write-gguf).
   Returns path."
  ([adapter path] (save-gguf adapter path nil))
  ([adapter path {:keys [name] :as opts}]
   (check-options :save-gguf opts #{:name})
   (when-not (or (nil? name) (string? name))
     (throw (invalid (str ":name is a string, not " (pr-str name)))))
   (check-adapter adapter)
   (let [{:keys [architecture alpha]} (:config adapter)
         params (:params adapter)]
     (when-not (and (string? architecture) (seq architecture))
       (throw (invalid (str "the adapter's :config has no :architecture (llama.cpp's general.architecture), but "
                            (pr-str architecture))
                       {:architecture architecture})))
     (array/eval! params)
     (gguf/write-gguf path {:metadata (cond-> [["general.architecture" :string architecture]
                                               ["general.type" :string "adapter"]]
                                        name (conj ["general.name" :string name])
                                        true (conj ["adapter.type" :string "lora"]
                                                   ["adapter.lora.alpha" :float32 alpha]))
                            :tensors (for [i (sort (keys params))
                                           :let [ts (params i)]
                                           t (sort-by target-index (keys ts))
                                           [suffix x] [[".lora_a" (:a (ts t))] [".lora_b" (:b (ts t))]]]
                                       {:name (str (tensor-base i t) suffix) :type :f32 :shape (array/shape x)
                                        :data (array/->floats x)})}))))

(def ^:private tensor-name #"^blk\.(\d+)\.([a-z_]+)\.weight\.lora_([ab])$")

(defn load-gguf
  "The adapter in llama.cpp's LoRA GGUF at path (save-gguf's, or
   convert_lora_to_gguf.py's), on gen.array's *backend*: its matrices in
   :dtype (default the backend's default float), its :dropout 0. An
   adapter.lora.alpha of 0 or none is llama.cpp's scale 1: :alpha the
   rank. Loud for a file that is no LoRA adapter (:invalid-args), and
   (:unsupported) for an activated LoRA (aLoRA), a tensor on another than
   targets' projections (llama.cpp's token_embd, output, or the norms it
   skips), quantized matrices, or ranks that differ between projections."
  ([path] (load-gguf path nil))
  ([path {:keys [dtype] :as opts}]
   (check-options :load-gguf opts #{:dtype})
   (let [{:keys [metadata tensors]} (array/load-gguf path)
         dtype (or dtype (:float (array/defaults array/*backend*)))
         as (fn [x] (if (= dtype (array/dtype x)) x (array/astype x dtype)))]
     (when-not (and (= "adapter" (get metadata "general.type")) (= "lora" (get metadata "adapter.type")))
       (throw (invalid (str path " is no LoRA adapter (general.type " (pr-str (get metadata "general.type"))
                            ", adapter.type " (pr-str (get metadata "adapter.type")) ")")
                       {:path path})))
     (when (contains? metadata "adapter.alora.invocation_tokens")
       (throw (unsupported (str path " is an activated LoRA (aLoRA), applied after its invocation tokens only") {:path path})))
     (let [params (reduce-kv (fn [acc nm x]
                               (let [[_ i tn ab] (re-matches tensor-name nm)
                                     t (target-of tn)]
                                 (when-not (and i t)
                                   (throw (unsupported (str path "'s tensor " nm " adapts none of " (vec (vals targets)))
                                                       {:path path :tensor nm})))
                                 (when (map? x)
                                   (throw (unsupported (str path "'s tensor " nm " is quantized") {:path path :tensor nm})))
                                 (assoc-in acc [(parse-long i) t (keyword ab)] (as x))))
                             {} tensors)
           ranks (set (for [[i ts] params [t ab] ts]
                        (do (when-not (and (:a ab) (:b ab))
                              (throw (invalid (str path ": layer " i "'s " t " lacks its lora_" (if (:a ab) "b" "a")) {:path path})))
                            (first (array/shape (:a ab))))))
           _ (when (c/> (count ranks) 1)
               (throw (unsupported (str path ": projections of different ranks " (vec (sort ranks))) {:path path :ranks ranks})))
           rank (first ranks)
           alpha (get metadata "adapter.lora.alpha")]
       (check-adapter {:config {:architecture (get metadata "general.architecture") :rank rank
                                :alpha (if (and (number? alpha) (pos? (double alpha))) (double alpha) (double rank))
                                :dropout 0.0}
                       :params params})))))
