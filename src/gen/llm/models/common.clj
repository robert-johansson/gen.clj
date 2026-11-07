(ns gen.llm.models.common
  "What the forward passes over gen.array share (gen.llm.models.qwen35,
   beside qwen3's own): their loud errors, options and tokens checked, a
   GGUF's weights cast and measured, the linear layers (dense, quantized
   or adapted by gen.llm.lora), training's output projection by chunks and
   masked loss, and a static cache's slots, masks and checks. who, a
   string naming the model family (\"qwen35\"), heads each message.
   Internal: the families' namespaces are the API."
  {:api :internal}
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [clojure.core :as c]
            [gen.array :as array :refer [+ * <=]]
            [gen.llm.lora :as lora]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Loud errors and options
;; ---------------------------------------------------------------------------

(defn unsupported
  "A model the forward does not compute: ex-info, :reason :unsupported."
  ([who message] (unsupported who message {}))
  ([who message data] (ex-info (str who ": " message) (merge {:reason :unsupported} data))))

(defn invalid
  "A bad argument: ex-info, :reason :invalid-args."
  ([who message] (invalid who message {}))
  ([who message data] (ex-info (str who ": " message) (merge {:reason :invalid-args} data))))

(defn check-options
  "opts (a map or nil) holding only the keys accepted: else loud, as
   gen.clj's inference refuses options it does not read."
  [who f opts accepted]
  (when-not (or (nil? opts) (map? opts))
    (throw (ex-info (str who "/" (name f) ": options must be a map, not " (pr-str opts))
                    {:reason :invalid-options :fn f})))
  (when-let [unknown (seq (remove accepted (keys opts)))]
    (throw (ex-info (str who "/" (name f) ": unknown options " (vec unknown) "; it takes " (vec (sort accepted)))
                    {:reason :unknown-options :fn f :unknown (vec unknown) :accepted accepted}))))

(defn check-logits [who logits]
  (when-not (#{:all :last} logits)
    (throw (invalid who (str ":logits must be :all or :last, not " (pr-str logits)) {:logits logits}))))

;; ---------------------------------------------------------------------------
;; Weights
;; ---------------------------------------------------------------------------

(defn float-array?
  "Is x a dense float array (not a quantized matrix)?"
  [x]
  (and (array/array? x) (= :float (array/dtype-kind (array/dtype x)))))

(defn cast-weight
  "w (an array, or a quantized matrix's map) with its floats in dtype."
  [who w dtype]
  (let [cast (fn [x] (if (= dtype (array/dtype x)) x (array/astype x dtype)))]
    (cond (map? w) (-> w (update :scales cast) (update :biases cast))
          (float-array? w) (cast w)
          :else (throw (invalid who (str "a weight is neither a float array nor a quantized matrix: " (pr-str (type w))))))))

(defn weight-shape
  "The shape a weight stands for: an array's, or a quantized matrix's
   once dequantized."
  [w]
  (if (map? w)
    (let [s (array/shape (:wq w))]
      (conj (pop s) (quot (c/* (long (peek s)) 32) (long (:bits w)))))
    (array/shape w)))

(defn check-dtype
  "dtype, a float dtype for a model to compute in."
  [who dtype]
  (when-not (#{:float16 :bfloat16 :float32 :float64} dtype)
    (throw (invalid who (str ":dtype must be a float dtype, not " (pr-str dtype)) {:dtype dtype})))
  dtype)

;; ---------------------------------------------------------------------------
;; Layers
;; ---------------------------------------------------------------------------

(defn linear
  "x·wᵀ for a weight w [out in]: dense, quantized, or adapted
   (gen.llm.lora's: its base's product plus the adapter's term)."
  [x w]
  (cond (lora/adapted? w) (+ (linear x (:base w)) (lora/delta x w))
        (map? w) (array/quantized-matmul x (:wq w) (:scales w) (:biases w) (select-keys w [:bits :group-size]))
        :else (array/matmul x (array/transpose w))))

(defn silu [x] (* x (array/sigmoid x)))

(defn swiglu
  "The gated feed-forward down(silu(gate x) · up x) of x, by the weights
   {:gate :up :down}."
  [{:keys [gate up down]} x]
  (linear (* (silu (linear x gate)) (linear x up)) down))

(defn embed
  "The embeddings [B T hidden] of tokens [B T] (an int array), in dtype."
  [w tokens dtype]
  (if (map? w)
    (array/dequantize (array/take (:wq w) tokens 0) (array/take (:scales w) tokens 0) (array/take (:biases w) tokens 0)
                      (assoc (select-keys w [:bits :group-size]) :dtype dtype))
    (array/take w tokens 0)))

;; ---------------------------------------------------------------------------
;; Training: the output projection by vocabulary chunks, the masked loss
;; ---------------------------------------------------------------------------

(defn rows
  "The rows start to stop - 1 of a weight [out in], dense or quantized."
  [w start stop]
  (if (map? w)
    (reduce #(update %1 %2 array/slice 0 start stop) w [:wq :scales :biases])
    (array/slice w 0 start stop)))

(defn head-logits
  "The logits of the vocabulary entries start to stop - 1 for the normed
   states hn [B T hidden], by those rows of the output weight w: a
   selective-log-softmax chunk's."
  [start stop hn w]
  (linear hn (rows w start stop)))

(defn check-temperature
  "log-probs' :temperature, nil or a positive finite number: else loud."
  [who temperature]
  (when-not (or (nil? temperature) (and (number? temperature) (pos? (double temperature)) (Double/isFinite (double temperature))))
    (throw (invalid who (str "log-probs: :temperature is a positive number, not " (pr-str temperature)) {:temperature temperature}))))

(defn tempered-head
  "head-logits at temperature (log-probs' :temperature): its logits
   divided by it, or head-logits itself at nil or 1 (the model's own, bit
   for bit)."
  [temperature]
  (if (or (nil? temperature) (c/== 1.0 (double temperature)))
    head-logits
    (let [temp (double temperature)]
      (fn [start stop hn w] (array// (head-logits start stop hn w) temp)))))

(defn nll-of
  "-Σ lp (log-probabilities [B T-1]), each times mask's weight where
   given (a float array of lp's shape), a 0-d array: a family's nll."
  [who lp mask]
  (when (and mask (not= (array/shape lp) (array/shape mask)))
    (throw (invalid who (str "nll: the mask's shape " (array/shape mask) " is not the log-probabilities' " (array/shape lp))
                    {:shapes [(array/shape mask) (array/shape lp)]})))
  (array/negate (array/sum (if mask (* mask lp) lp))))

;; ---------------------------------------------------------------------------
;; Tokens
;; ---------------------------------------------------------------------------

(defn token-array
  "tokens as an int array [B T]: host rows checked against the vocabulary,
   or an int array as it is."
  [who tokens vocab]
  (cond
    (and (array/array? tokens) (not (array/host? tokens)))
    (do (when-not (and (= 2 (array/ndim tokens)) (= :int (array/dtype-kind (array/dtype tokens))))
          (throw (invalid who (str "tokens must be an int array [B T], not " (array/dtype tokens) " " (array/shape tokens)))))
        tokens)

    (and (sequential? tokens) (seq tokens) (every? sequential? tokens))
    (let [n (count (first tokens))]
      (when-not (and (pos? n) (every? #(= n (count %)) tokens))
        (throw (invalid who "tokens must be rows of one positive length" {:lengths (mapv count tokens)})))
      (doseq [row tokens t row]
        (when-not (and (integer? t) (c/<= 0 (long t)) (c/< (long t) (long vocab)))
          (throw (ex-info (str who ": token " (pr-str t) " is not in the vocabulary [0, " vocab ")")
                          {:reason :invalid-token :token t :vocab vocab}))))
      (array/array tokens {:dtype :int32}))

    :else (throw (invalid who (str "tokens must be rows of token ids (vectors) or an int array [B T], not " (pr-str tokens))))))

;; ---------------------------------------------------------------------------
;; Static caches: slots written in place of a concatenation
;; ---------------------------------------------------------------------------

(defn chunk-positions
  "The positions of a chunk of t tokens, [B 1 t 1]: row b's i-th at
   positions[b] + i."
  [positions t]
  (+ (array/reshape positions [(first (array/shape positions)) 1 1 1])
     (array/reshape (array/arange 0 t 1 {:dtype :int32}) [1 1 t 1])))

(defn static-mask
  "The slots of a static cache of capacity that a chunk of t tokens' queries
   see, a bool array [B 1 t capacity]: query i of row b, at position
   positions[b] + i, sees the slots up to its own (causal); the slots past
   it hold the chunk's later tokens or padding."
  [positions t capacity]
  (<= (array/arange 0 capacity 1 {:dtype :int32}) (chunk-positions positions t)))

(defn write-slots
  "Where a chunk's keys or values go, as put-along-axis's indices: an int
   array of their shape [B KV-heads t head-dim], positions[b] + i at the
   chunk's i-th position."
  [positions [_ _ t _ :as shape]]
  (array/broadcast-to (chunk-positions positions t) shape))

(defn check-capacity
  "A chunk of t tokens fits in every row of the static cache."
  [who {:keys [capacity offsets]} t]
  (let [last-offset (long (apply max offsets))]
    (when (c/> (c/+ last-offset (long t)) (long capacity))
      (throw (ex-info (str who ": " t " tokens after offset " last-offset " exceed the static cache's capacity of " capacity)
                      {:reason :capacity-exceeded :capacity capacity :offsets offsets :tokens t})))))

(defn check-cache-arg
  "rows and capacity, positive integers, for an empty static cache."
  [who rows capacity]
  (doseq [[what v] [[:rows rows] [:capacity capacity]]]
    (when-not (and (integer? v) (pos? (long v)))
      (throw (invalid who (str "static-cache: " (name what) " must be a positive integer, not " (pr-str v)) {what v})))))

(defn check-row-indices
  "idx, host integers of rows of a cache of n rows, at least one."
  [who idx n]
  (when (empty? idx)
    (throw (invalid who "select-rows: no rows" {:rows idx})))
  (doseq [i idx]
    (when-not (and (integer? i) (c/<= 0 (long i)) (c/< (long i) (long n)))
      (throw (invalid who (str "select-rows: " (pr-str i) " is not a row of a cache of " n) {:row i :rows n})))))

(defn check-grow
  "capacity, an integer of at least old, a static cache's capacity."
  [who old capacity]
  (when-not (and (integer? capacity) (c/<= (long old) (long capacity)))
    (throw (invalid who (str "grow: the capacity is an integer of at least the cache's " old ", not " (pr-str capacity))
                    {:capacity capacity :cache-capacity old}))))

(defn new-offsets
  "offsets (a vector of one per row, or a host integer for every row) for
   rewind, checked against the rows' own os: each from 0 to its row's."
  [who os offsets]
  (let [new (if (integer? offsets) (vec (repeat (count os) offsets)) offsets)]
    (when-not (and (sequential? new) (= (count new) (count os)))
      (throw (invalid who (str "rewind: the offsets are one per row (" (count os) ") or one integer, not " (pr-str offsets))
                      {:offsets offsets :rows (count os)})))
    (doseq [[o n] (map vector os new)]
      (when-not (and (integer? n) (c/<= 0 (long n) (long o)))
        (throw (invalid who (str "rewind: " (pr-str n) " is not an offset from 0 to the row's " o) {:offset n :row-offset o}))))
    (mapv long new)))

(defn check-concat
  "caches, static ones of one capacity and as many layers, at least one."
  [who caches static?]
  (when (empty? caches)
    (throw (invalid who "concat-rows: no caches" {})))
  (doseq [cache caches]
    (when-not (static? cache)
      (throw (invalid who (str "concat-rows takes static caches (static-cache, or prefill's :capacity), not "
                               (if (map? cache) "a concatenating cache" (pr-str cache)))
                      {:fn :concat-rows}))))
  (when-not (apply = (map :capacity caches))
    (throw (invalid who (str "concat-rows: caches of different capacities " (mapv :capacity caches))
                    {:capacities (mapv :capacity caches)})))
  (when-not (apply = (map (comp count :layers) caches))
    (throw (invalid who "concat-rows: caches of different numbers of layers" {:layers (mapv (comp count :layers) caches)}))))
