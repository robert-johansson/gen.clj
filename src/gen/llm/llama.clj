(ns gen.llm.llama
  "The llama.cpp engine (DESIGN.md §6): a Branching engine (gen.llm.branching)
   over one llama.cpp context, through the generated binding of llama.h
   (src-java/gen/llama/ffi, from native/jextract-llama.sh) and the libraries
   of native/build-llama.sh, both pinned to llama.cpp v0.7.0.

     (with-open [e (llama/engine {:model \"Qwen3-0.6B-Q8_0.gguf\"})]
       (let [b (br/open e (llama/tokenize e \"The capital of Sweden is\"))]
         …))

   A branch is a sequence of the context's KV cache, one slot of
   gen.llm.slots, so an engine holds up to :max-branches (llama.cpp allows
   255 with the scratch slot). The cache is unified (kv_unified), so a fork
   is a copy of cell metadata (llama_memory_seq_cp) and forks share their
   prefix's cells: :n-ctx counts cells, shared by all branches, and is
   also the most tokens one branch holds. A batch that does not fit is a
   loud error (:reason :kv-full), as is a decode that fails.

   - Sliding-window models (Gemma) keep a full-size cache (swa_full), so a
     branch keeps every token a fork may need.
   - Recurrent and hybrid models (Mamba, GatedDeltaNet) cannot remove the
     end of a sequence past their checkpoints; this engine never does
     (score reads into a scratch copy), so they need none. Untested: no
     such model was at hand.
   - Encoder-decoder, diffusion and embedding models are refused.

   llama.cpp's log goes to System.err from warnings up (gen.llama.log:
   none, error, warn, info, debug); libllama is found in gen.llama.lib
   (default native/build/llama/<platform>/lib, gen.llama.LlamaRuntime).
   tokenize, token-bytes and detokenize use the GGUF's own vocabulary; they
   are the engine's, for the LM protocol to build on."
  (:require [gen.llm.branching :as br]
            [gen.llm.slots :as slots])
  (:import [gen.llama LlamaRuntime]
           [gen.llama.ffi Llama llama_context_params llama_model_params]
           [java.lang.foreign Arena MemorySegment ValueLayout]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Native calls
;; ---------------------------------------------------------------------------

(defn- null? [^MemorySegment s] (zero? (.address s)))

(defn- llama-error
  "The last error llama.cpp logged on this thread, or nil (and forget it)."
  []
  (let [e (LlamaRuntime/lastError)]
    (LlamaRuntime/clearLastError)
    e))

(defn- decode-error
  "The ex-info for llama_decode's status code."
  [^long code what]
  (let [msg (llama-error)
        [reason text] (case code
                        1 [:kv-full "no room in the KV cache for the batch: the branches' tokens fill its cells (dispose! some, or open the engine with a larger :n-ctx)"]
                        2 [:aborted "the decode was aborted"]
                        -1 [:invalid-batch "llama.cpp refused the batch"]
                        [:decode-failed "llama.cpp's decode failed"])]
    (ex-info (str what ": " text (when msg (str " (llama.cpp: " msg ")")))
             {:reason reason :code code :llama-error msg})))

(defn- restores?
  "Does a failed decode with this code leave the memory as it was? (Per
   llama.h: 1 and -1 do; an abort or a fatal error keeps what it processed.)"
  [^long code]
  (or (== code 1) (== code -1)))

(defn- row
  "The logits of the batch's k-th token."
  ^MemorySegment [^MemorySegment ctx k what]
  (let [r (Llama/llama_get_logits_ith ctx (int k))]
    (when (null? r)
      (throw (ex-info (str what ": llama.cpp has no logits for row " k) {:reason :decode-failed :row k})))
    r))

(defn- log-normalizer
  "log Σ exp(logits) of a row, or a loud error when a logit is NaN or +Inf."
  ^double [^MemorySegment r ^long n-vocab what]
  (let [z (LlamaRuntime/logSumExp r (int n-vocab))]
    (when-not (Double/isFinite z)
      (throw (ex-info (str what ": the model's logits are not finite") {:reason :decode-failed :log-normalizer z})))
    z))

(declare restore)

(deftype LlamaSlots [^MemorySegment model ^MemorySegment ctx ^MemorySegment mem ^MemorySegment vocab
                     ^MemorySegment batch ^Arena arena ^long n-vocab ^long n-batch info]
  slots/Slots
  (slots-info [_] info)

  (slots-run [this slot pos tokens targets]
    (let [toks (int-array tokens)
          n (alength toks)
          tgts (some-> targets int-array)
          scores (when tgts (double-array (dec n)))
          ;; has anything reached the cache? then a failure must remove it
          touched (volatile! false)]
      (try
        (loop [from 0 lp nil]
          (if (< from n)
            (let [len (min n-batch (- n from))
                  last? (= n (+ from len))]
              (LlamaRuntime/setRun batch (int n-batch) toks (int from) (int len) (int (+ (long pos) from))
                                   (int slot) (boolean tgts))
              (LlamaRuntime/clearLastError)
              (let [code (Llama/llama_decode ctx batch)]
                (when-not (restores? code) (vreset! touched true))
                (when-not (zero? code) (throw (decode-error code "read"))))
              (when tgts
                ;; row k scores the token after token from+k (the last row
                ;; of the run is its next-token distribution instead)
                (dotimes [k (if last? (dec len) len)]
                  (let [r (row ctx k "read")
                        g (+ from k)]
                    (aset scores g (- (LlamaRuntime/logit r (int n-vocab) (aget ^ints tgts g))
                                      (log-normalizer r n-vocab "read"))))))
              (recur (+ from len)
                     (when last?
                       (let [r (row ctx (dec len) "read")]
                         (LlamaRuntime/logSoftmax r (int n-vocab) (log-normalizer r n-vocab "read"))))))
            {:logprobs lp :scores (some-> scores vec)}))
        (catch Throwable t
          (throw (if @touched (restore this t [slot] [pos]) t))))))

  (slots-step [this slots positions tokens]
    (LlamaRuntime/setStep batch (int n-batch) (int-array tokens) (int-array positions) (int-array slots))
    (LlamaRuntime/clearLastError)
    (let [code (Llama/llama_decode ctx batch)]
      (when-not (zero? code)
        (let [e (decode-error code "advance")]
          (throw (if (restores? code) e (restore this e slots positions))))))
    (try
      (mapv (fn [k]
              (let [r (row ctx k "advance")]
                (LlamaRuntime/logSoftmax r (int n-vocab) (log-normalizer r n-vocab "advance"))))
            (range (count slots)))
      (catch Throwable t
        (throw (restore this t slots positions)))))

  (slots-copy [_ src dst]
    (Llama/llama_memory_seq_cp mem (int src) (int dst) (int -1) (int -1)))

  (slots-clear [_ slot]
    (Llama/llama_memory_seq_rm mem (int slot) (int -1) (int -1)))

  (slots-close [_]
    (Llama/llama_batch_free batch)
    (Llama/llama_free ctx)
    (Llama/llama_model_free model)
    (.close arena))

  Object
  (toString [_] (str "#<LlamaSlots " (:desc (:model info)) ">")))

(defn- restore
  "After a failed decode or read: remove what it added to each slot (its
   positions from pos on), and return the exception to throw. A slot whose
   removal fails (a recurrent model past its checkpoints) is lost: the
   exception names it (:lost-slots) for the engine to free."
  [^LlamaSlots s ^Throwable t slots positions]
  (let [lost (vec (keep (fn [[slot pos]]
                          (when-not (Llama/llama_memory_seq_rm (.-mem s) (int slot) (int pos) (int -1))
                            slot))
                        (map vector slots positions)))]
    (if (seq lost)
      (ex-info (str (ex-message t) "; the branches on slots " lost " could not be restored and are freed")
               (assoc (ex-data t) :lost-slots lost) t)
      t)))

;; ---------------------------------------------------------------------------
;; Opening
;; ---------------------------------------------------------------------------

(def ^:private max-seq
  ;; LLAMA_MAX_SEQ, llama.cpp's limit on sequences
  256)

(defn- c-string ^MemorySegment [^Arena a ^String s] (.allocateFrom a s))

(defn- model-desc [^MemorySegment model]
  (with-open [a (Arena/ofConfined)]
    (let [buf (.allocate a 256)
          n (Llama/llama_model_desc model buf 256)]
      (if (neg? n) "" (.getString buf 0)))))

(defn- open-slots
  "LlamaSlots for opts, made on the engine's thread."
  [{:keys [model max-branches n-ctx n-batch gpu-layers threads swa-full]
    :or {max-branches 8 n-ctx 4096 n-batch 512 gpu-layers -1 swa-full true}}]
  (when-not (and (string? model) (.isFile (java.io.File. ^String model)))
    (throw (ex-info (str "llama/engine: no model file at " (pr-str model)) {:reason :no-model :model model})))
  (when-not (and (integer? max-branches) (<= 1 max-branches (- max-seq 1)))
    (throw (ex-info (str "llama/engine: :max-branches is 1 to " (- max-seq 1) " (llama.cpp's " max-seq
                         " sequences, one kept for scoring), not " (pr-str max-branches))
                    {:reason :invalid-option :max-branches max-branches})))
  (doseq [[k v ok?] [[:n-ctx n-ctx pos-int?] [:n-batch n-batch pos-int?] [:gpu-layers gpu-layers int?]
                     [:threads threads #(or (nil? %) (pos-int? %))]]]
    (when-not (ok? v)
      (throw (ex-info (str "llama/engine: bad " k ": " (pr-str v)) {:reason :invalid-option k v}))))
  (LlamaRuntime/load)
  (LlamaRuntime/clearLastError)
  (let [n-slots (inc (long max-branches))
        n-batch (max (long n-batch) n-slots)
        arena (Arena/ofConfined)
        ;; what to free, last first, if construction fails
        freeing (volatile! (list #(.close arena)))
        fail (fn [msg data]
               (throw (ex-info (str "llama/engine: " msg) (assoc data :llama-error (llama-error)))))]
    (try
      (let [mp (Llama/llama_model_default_params arena)
            _ (llama_model_params/n_gpu_layers mp (int gpu-layers))
            m (Llama/llama_model_load_from_file (c-string arena model) mp)
            _ (when (null? m) (fail (str "llama.cpp could not load " model) {:reason :model-load-failed :model model}))
            _ (vswap! freeing conj #(Llama/llama_model_free m))
            _ (cond
                (Llama/llama_model_has_encoder m) (fail "encoder-decoder models are not supported" {:reason :unsupported-model})
                (not (Llama/llama_model_has_decoder m)) (fail "the model has no decoder" {:reason :unsupported-model})
                (Llama/llama_model_is_diffusion m) (fail "diffusion models are not supported" {:reason :unsupported-model}))
            cp (Llama/llama_context_default_params arena)
            _ (doto cp
                (llama_context_params/n_ctx (int n-ctx))
                (llama_context_params/n_batch (int n-batch))
                (llama_context_params/n_seq_max (int n-slots))
                (llama_context_params/kv_unified true)
                (llama_context_params/swa_full (boolean swa-full))
                (llama_context_params/no_perf true))
            _ (when threads
                (llama_context_params/n_threads cp (int threads))
                (llama_context_params/n_threads_batch cp (int threads)))
            ctx (Llama/llama_init_from_model m cp)
            _ (when (null? ctx) (fail "llama.cpp could not create a context" {:reason :context-failed :n-ctx n-ctx}))
            _ (vswap! freeing conj #(Llama/llama_free ctx))
            _ (when-not (and (= n-slots (Llama/llama_n_seq_max ctx)) (<= n-slots (Llama/llama_n_batch ctx)))
                (fail "llama.cpp changed the context's sequences or batch"
                      {:reason :context-failed :n-seq-max (Llama/llama_n_seq_max ctx) :n-batch (Llama/llama_n_batch ctx)}))
            mem (Llama/llama_get_memory ctx)
            _ (when (null? mem) (fail "the model keeps no memory (an embedding model?)" {:reason :unsupported-model}))
            vocab (Llama/llama_model_get_vocab m)
            n-vocab (Llama/llama_vocab_n_tokens vocab)
            n-batch (long (Llama/llama_n_batch ctx))
            batch (Llama/llama_batch_init arena (int n-batch) (int 0) (int 1))
            _ (vswap! freeing conj #(Llama/llama_batch_free batch))
            info {:engine :llama.cpp
                  :n-vocab n-vocab
                  :n-slots n-slots
                  :n-ctx (long (Llama/llama_n_ctx_seq ctx))
                  :n-batch n-batch
                  :bos (long (Llama/llama_vocab_bos vocab))
                  :eos (long (Llama/llama_vocab_eos vocab))
                  :eog (into (sorted-set) (filter #(Llama/llama_vocab_is_eog vocab (int %))) (range n-vocab))
                  :add-bos? (Llama/llama_vocab_get_add_bos vocab)
                  :model {:path model
                          :desc (model-desc m)
                          :bytes (Llama/llama_model_size m)
                          :params (Llama/llama_model_n_params m)
                          :n-ctx-train (long (Llama/llama_model_n_ctx_train m))
                          :n-swa (long (Llama/llama_model_n_swa m))
                          :recurrent? (Llama/llama_model_is_recurrent m)
                          :hybrid? (Llama/llama_model_is_hybrid m)}
                  :gpu-offload? (Llama/llama_supports_gpu_offload)}]
        (LlamaSlots. m ctx mem vocab batch arena n-vocab n-batch info))
      (catch Throwable t
        (doseq [f @freeing] (f))
        (throw t)))))

(defn engine
  "A llama.cpp engine (java.io.Closeable; close it to free the model), on a
   thread of its own. opts:
     :model         the GGUF file (required)
     :max-branches  live branches at most (default 8; up to 255)
     :n-ctx         KV cells, shared by every branch, and the most tokens
                    one branch holds (default 4096)
     :n-batch       tokens per decode (default 512, at least
                    :max-branches + 1); longer reads take several
     :gpu-layers    layers offloaded to the GPU (default -1, all)
     :threads       CPU threads (default llama.cpp's)
     :swa-full      a full-size cache for sliding-window layers (default true)"
  ^java.io.Closeable [opts]
  (slots/engine #(open-slots opts) "llama.cpp"))

;; ---------------------------------------------------------------------------
;; The vocabulary (the GGUF's own tokenizer)
;; ---------------------------------------------------------------------------

(defn- vocab-seg ^MemorySegment [^LlamaSlots s] (.-vocab s))

(defn- check-token [e t]
  (let [n (:n-vocab (br/info e))]
    (when-not (and (integer? t) (< -1 (long t) (long n)))
      (throw (ex-info (str "Not a token of the vocabulary (0 to " (dec (long n)) "): " (pr-str t))
                      {:reason :invalid-token :token t :n-vocab n})))))

(defn tokenize
  "The tokens of text, by the model's tokenizer: a vector. opts:
   :add-special? (default true) adds BOS (and EOS) as the model does;
   :parse-special? (default false) reads special tokens written in the
   text (\"<|im_start|>\") as themselves."
  ([e text] (tokenize e text {}))
  ([e ^String text {:keys [add-special? parse-special?] :or {add-special? true parse-special? false}}]
   (let [bytes (.getBytes text "UTF-8")]
     (slots/call e (fn [s]
                     (with-open [a (Arena/ofConfined)]
                       (let [src (.allocateFrom a ValueLayout/JAVA_BYTE bytes)
                             go (fn [^long cap]
                                  (let [out (.allocate a (* 4 (max 1 cap)))
                                        n (Llama/llama_tokenize (vocab-seg s) src (int (alength bytes)) out (int cap)
                                                                (boolean add-special?) (boolean parse-special?))]
                                    [n out]))
                             [n ^MemorySegment out] (let [[n :as r] (go (+ 8 (alength bytes)))]
                                                      (if (neg? (long n)) (go (- (long n))) r))]
                         (when (neg? (long n))
                           (throw (ex-info "tokenize: llama.cpp could not tokenize the text" {:reason :tokenize-failed :n n})))
                         (mapv long (.toArray (.asSlice out 0 (* 4 (long n))) ValueLayout/JAVA_INT)))))))))

(defn token-bytes
  "The bytes of token, raw (a token may hold part of a UTF-8 character;
   special tokens are rendered): a byte array."
  ^bytes [e token]
  (check-token e token)
  (slots/call e (fn [s]
                  (with-open [a (Arena/ofConfined)]
                    (let [go (fn [^long cap]
                               (let [buf (.allocate a (max 1 cap))]
                                 [(Llama/llama_token_to_piece (vocab-seg s) (int token) buf (int cap) (int 0) true) buf]))
                          [n ^MemorySegment buf] (let [[n :as r] (go 64)] (if (neg? (long n)) (go (- (long n))) r))]
                      (.toArray (.asSlice buf 0 (long n)) ValueLayout/JAVA_BYTE))))))

(defn detokenize
  "The text of tokens (special tokens rendered), decoded as UTF-8."
  ^String [e tokens]
  (run! #(check-token e %) tokens)
  (let [toks (int-array tokens)]
    (slots/call e (fn [s]
                    (with-open [a (Arena/ofConfined)]
                      (let [src (.allocateFrom a ValueLayout/JAVA_INT toks)
                            go (fn [^long cap]
                                 (let [buf (.allocate a (max 1 cap))]
                                   [(Llama/llama_detokenize (vocab-seg s) src (int (alength toks)) buf (int cap) false true)
                                    buf]))
                            [n ^MemorySegment buf] (let [[n :as r] (go (* 8 (inc (alength toks))))]
                                                     (if (neg? (long n)) (go (- (long n))) r))]
                        (String. ^bytes (.toArray (.asSlice buf 0 (long n)) ValueLayout/JAVA_BYTE) "UTF-8")))))))

(defn kv-positions
  "Diagnostics: [min max] of the positions each slot holds in the KV cache
   ([-1 -1] when empty), by slot (the last is the scratch slot)."
  [e]
  (slots/call e (fn [^LlamaSlots s]
                  (mapv (fn [slot] [(long (Llama/llama_memory_seq_pos_min (.-mem s) (int slot)))
                                    (long (Llama/llama_memory_seq_pos_max (.-mem s) (int slot)))])
                        (range (:n-slots (.-info s)))))))

(defn system-info
  "llama.cpp's description of its build and the machine (its backends)."
  ^String []
  (LlamaRuntime/load)
  (.getString (Llama/llama_print_system_info) 0))
