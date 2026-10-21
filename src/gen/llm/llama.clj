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
     (score reads into a scratch copy), so they need none (tested with
     Qwen3.5's hybrid; Gemma 3 for the sliding window).
   - A decode of a recurrent or hybrid model copies every state between
     the lowest and the highest cell its sequences land on. The engine
     replays where llama.cpp puts them (gen.llm.llama.recurrent) and splits
     a step whose states would land far apart into decodes over runs of
     adjacent cells, when that costs less: a function of the engine's own
     operations, never of timing. gen.llama.split-recurrent false (default
     true) keeps one decode a step; (info e) says :split-recurrent?, and
     step-splits what was split.
   - Encoder-decoder, diffusion and embedding models are refused.
   - A decode's rows of logits are normalized on a small pool of the
     engine's own, a thread per row up to four, while the engine's thread
     waits: normalizing a vocabulary on the host costs about as much as a
     row's share of the decode.

   llama.cpp's log goes to System.err from warnings up (gen.llama.log:
   none, error, warn, info, debug); libllama is found in gen.llama.lib
   (default native/build/llama/<platform>/lib, gen.llama.LlamaRuntime).
   tokenize, token-bytes and detokenize use the GGUF's own vocabulary; they
   are the engine's, for the LM protocol to build on."
  (:require [gen.llm.branching :as br]
            [gen.llm.llama.recurrent :as recurrent]
            [gen.llm.slots :as slots])
  (:import [gen.llama LlamaRuntime LlamaRuntime$Batch]
           [gen.llama.ffi Llama llama_context_params llama_model_params]
           [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.nio.charset StandardCharsets]
           [java.util Collection]
           [java.util.concurrent Callable ExecutionException ExecutorService Executors Future]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Decoding
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
             (cond-> {:reason reason :code code :llama-error msg}
               ;; gen.llm.slots collects dropped branches, whose cells may
               ;; be what is missing, and retries (unless a slot is lost)
               (== code 1) (assoc :exhausted? true)))))

(defn- decode!
  "llama_decode the batch, or throw its ex-info."
  [^MemorySegment ctx ^LlamaRuntime$Batch batch what]
  (LlamaRuntime/clearLastError)
  (let [code (Llama/llama_decode ctx (.-struct batch))]
    (when-not (zero? code)
      (throw (decode-error code what)))))

(defn- row
  "The logits of the batch's k-th token (on the engine's thread: a native
   call). Valid until the next decode."
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

(defn- row-logprobs
  "A row's log-probabilities."
  ^floats [^MemorySegment r ^long n-vocab what]
  (LlamaRuntime/logSoftmax r (int n-vocab) (log-normalizer r n-vocab what)))

(defn- row-score
  "The log-probability of token in a row."
  ^double [^MemorySegment r ^long n-vocab token what]
  (- (LlamaRuntime/logit r (int n-vocab) (int token)) (log-normalizer r n-vocab what)))

(defn- row-pool
  "The pool that normalizes a decode's rows: up to four platform threads
   (half the cores), daemons."
  ^ExecutorService []
  (Executors/newFixedThreadPool
   (max 1 (min 4 (quot (.availableProcessors (Runtime/getRuntime)) 2)))
   (-> (Thread/ofPlatform) (.name "gen.llm llama.cpp rows-" 0) (.daemon true) .factory)))

(defn- in-parallel
  "(f x) for each x of xs, in order, on pool when there are several (what
   an f threw, rethrown); every call has ended when it returns."
  [^ExecutorService pool f xs]
  (if (next xs)
    (mapv (fn [^Future fut]
            (try (.get fut)
                 (catch ExecutionException x (throw (.getCause x)))))
          (.invokeAll pool ^Collection (mapv (fn [x] ^Callable (fn [] (f x))) xs)))
    (mapv f xs)))

(defn- restore
  "After a failed decode or read: remove what it may have added to each
   slot (its positions from pos on; nothing to remove is a removal that
   succeeds, for recurrent memory too), and return the exception to throw.
   A slot whose removal fails, or leaves it not ending at pos - 1 (a
   recurrent model past its checkpoints), is lost: the exception names it
   (:lost-slots) for the engine to free."
  [^MemorySegment mem ^Throwable t slots positions]
  (let [lost (vec (keep (fn [[slot pos]]
                          (when-not (and (Llama/llama_memory_seq_rm mem (int slot) (int pos) (int -1))
                                         (== (dec (long pos)) (Llama/llama_memory_seq_pos_max mem (int slot))))
                            slot))
                        (map vector slots positions)))]
    (if (seq lost)
      (ex-info (str (ex-message t) "; slots " lost " could not be restored")
               (assoc (ex-data t) :lost-slots lost) t)
      t)))

(def ^:private fixed-states
  "A decode's own cost, in recurrent states copied, for splitting steps
   (recurrent/plan). On Qwen3.5-0.8B on Metal it is about 3 (10 ms a
   decode, 3.5 ms a state), and larger models' decodes cost more against
   their states. The replayed plans cost about the same from 3 to 12; 6
   leans towards fewer decodes."
  6.0)

(defn- split-recurrent?
  "gen.llama.split-recurrent: does a recurrent or hybrid model's engine
   split its steps (default true)?"
  []
  (let [v (System/getProperty "gen.llama.split-recurrent" "true")]
    (case v
      "true" true
      "false" false
      (throw (ex-info (str "llama/engine: gen.llama.split-recurrent is true or false, not " (pr-str v))
                      {:reason :invalid-option :gen.llama.split-recurrent v})))))

;; An engine that splits steps holds a volatile
;;   {:cells   the replay (gen.llm.llama.recurrent), nil once it is off
;;    :off     nil, or why the replay stopped
;;    :steps :split :decodes   counts: steps, those split, their decodes}

(defn- replayed!
  "The replay becomes (f replay args), if it is on."
  [^clojure.lang.Volatile splits f & args]
  (when-let [st (some-> splits deref :cells)]
    (vswap! splits assoc :cells (apply f st args))))

(defn- replay-off!
  "The replay stops: steps are one decode from now on."
  [^clojure.lang.Volatile splits why]
  (vswap! splits assoc :cells nil :off why))

(defn- replay-checked!
  "After an operation on slots: the replay and llama.cpp agree on each
   one's state position (llama_memory_seq_pos_min: a recurrent state's
   position, -1 for none, for a hybrid model too; v0.7.0 tells no cell's
   index). A disagreement stops the replay, loudly on System.err."
  [^clojure.lang.Volatile splits ^MemorySegment mem slots what]
  (when-let [st (some-> splits deref :cells)]
    (when-let [[slot mine theirs]
               (first (keep (fn [slot]
                              (let [cell (long (get-in st [:cells slot :tail]))
                                    mine (if (neg? cell) -1 (long (get-in st [:cells cell :pos])))
                                    theirs (long (Llama/llama_memory_seq_pos_min mem (int slot)))]
                                (when-not (== mine theirs) [slot mine theirs])))
                            slots))]
      (let [why (str what ": the replay of llama.cpp's recurrent cells puts slot " slot "'s state at position "
                     mine ", llama.cpp at " theirs "; steps are one decode from now on")]
        (replay-off! splits why)
        (.println System/err (str "gen.llm.llama: " why))))))

(defn- replay-failed!
  "After a failure, which restore is about to undo: a decode that found no
   room (llama_decode's 1) or refused its batch (-1) changed nothing, and
   a failure after a decode left what the replay already holds, so only
   restore's removals from position 0, of the slots cleared, remain to
   replay. Any other failure of a decode may leave llama.cpp's cells where
   the replay cannot follow: it stops."
  [^clojure.lang.Volatile splits ^Throwable t cleared]
  (when splits
    (if (contains? #{1 -1 nil} (:code (ex-data t)))
      (run! #(replayed! splits recurrent/clear %) cleared)
      (replay-off! splits (str "a decode failed (" (ex-message t) ")")))))

(deftype LlamaSlots [^MemorySegment model ^MemorySegment ctx ^MemorySegment mem ^MemorySegment vocab
                     ^LlamaRuntime$Batch batch ^Arena arena ^ExecutorService pool ^long n-vocab ^long n-batch info
                     ^clojure.lang.Volatile splits]
  slots/Slots
  (slots-info [_] info)

  (slots-run [_ slot pos tokens targets]
    (let [toks (int-array tokens)
          n (alength toks)
          tgts (some-> targets int-array)]
      (try
        (loop [from 0 scores []]
          (let [len (min n-batch (- n from))
                end (+ from len)]
            (.run batch toks (int from) (int len) (int (+ (long pos) from)) (int slot) (some? tgts))
            (decode! ctx batch "read")
            (replayed! splits recurrent/decode [slot] [(+ (long pos) end -1)])
            (replay-checked! splits mem [slot] "read")
            ;; row k scores the token after token from+k; the run's last
            ;; row is its next-token distribution instead
            (let [scores (if tgts
                           (into scores (in-parallel pool
                                                     (fn [[^MemorySegment r k]] (row-score r n-vocab (aget tgts (+ from (long k))) "read"))
                                                     (mapv (fn [k] [(row ctx k "read") k]) (range (min len (- n from 1))))))
                           scores)]
              (if (< end n)
                (recur end scores)
                {:logprobs (row-logprobs (row ctx (dec len) "read") n-vocab "read")
                 :scores (when tgts scores)}))))
        (catch Throwable t
          (replay-failed! splits t (when (zero? (long pos)) [slot]))
          (throw (restore mem t [slot] [pos]))))))

  (slots-step [_ slots positions tokens]
    (let [slots (vec slots) positions (vec positions) tokens (vec tokens)
          ;; one decode, or several over runs of adjacent recurrent cells
          groups (if-let [st (some-> splits deref :cells)]
                   (recurrent/plan st slots positions fixed-states)
                   [(vec (range (count slots)))])
          rows (object-array (count slots))]
      (when splits
        (vswap! splits #(-> % (update :steps inc) (update :decodes + (count groups))
                            (cond-> (next groups) (update :split inc)))))
      (try
        (doseq [g groups]
          (let [ss (mapv slots g)
                ps (mapv positions g)]
            (.step batch (int-array (mapv tokens g)) (int-array ps) (int-array ss))
            (decode! ctx batch "advance")
            (replayed! splits recurrent/decode ss ps)
            (replay-checked! splits mem ss "advance")
            (doseq [[i r] (map vector g (in-parallel pool #(row-logprobs % n-vocab "advance")
                                                     (mapv #(row ctx % "advance") (range (count g)))))]
              (aset rows (int i) r))))
        (vec rows)
        (catch Throwable t
          (replay-failed! splits t nil)
          (throw (restore mem t slots positions))))))

  (slots-copy [_ src dst]
    (Llama/llama_memory_seq_cp mem (int src) (int dst) (int -1) (int -1))
    (replayed! splits recurrent/copy src dst)
    (replay-checked! splits mem [dst] "copy"))

  (slots-clear [_ slot]
    (Llama/llama_memory_seq_rm mem (int slot) (int -1) (int -1))
    (replayed! splits recurrent/clear slot)
    (replay-checked! splits mem [slot] "clear"))

  (slots-close [_]
    (.shutdown pool)
    (Llama/llama_batch_free (.-struct batch))
    (Llama/llama_free ctx)
    (Llama/llama_model_free model)
    (.close arena))

  Object
  (toString [_] (str "#<LlamaSlots " (:desc (:model info)) ">")))

;; ---------------------------------------------------------------------------
;; Opening
;; ---------------------------------------------------------------------------

(def ^:private defaults {:max-branches 8 :n-ctx 4096 :n-batch 512 :gpu-layers -1 :swa-full true})

(def ^:private option-checks
  {:model string? :max-branches pos-int? :n-ctx pos-int? :n-batch pos-int? :gpu-layers int?
   :threads #(or (nil? %) (pos-int? %)) :swa-full boolean?})

(defn- check-options
  "opts with their defaults and :n-slots, checked: a loud error for an
   unknown, missing or bad one."
  [opts]
  (when-let [ks (seq (remove option-checks (keys opts)))]
    (throw (ex-info (str "llama/engine: unknown options " (vec ks)) {:reason :invalid-option :unknown (vec ks)})))
  (let [{:keys [model max-branches] :as opts} (merge defaults opts)
        ;; LLAMA_MAX_SEQ, 256 in v0.7.0
        max-seq (Llama/llama_max_parallel_sequences)]
    (when-not (and (string? model) (.isFile (java.io.File. ^String model)))
      (throw (ex-info (str "llama/engine: no model file at " (pr-str model)) {:reason :no-model :model model})))
    (doseq [[k ok?] option-checks
            :when (not (ok? (get opts k)))]
      (throw (ex-info (str "llama/engine: bad " k ": " (pr-str (get opts k))) {:reason :invalid-option k (get opts k)})))
    (when-not (< (long max-branches) max-seq)
      (throw (ex-info (str "llama/engine: :max-branches is 1 to " (dec max-seq) " (llama.cpp's " max-seq
                           " sequences, one kept for scoring), not " max-branches)
                      {:reason :invalid-option :max-branches max-branches})))
    (assoc opts :n-slots (inc (long max-branches)))))

(defn- llama-failure [msg data]
  (ex-info (str "llama/engine: " msg) (assoc data :llama-error (llama-error))))

(defn- load-model
  "The model at path, layers on the GPU."
  ^MemorySegment [^Arena arena ^String path gpu-layers]
  (let [params (Llama/llama_model_default_params arena)
        _ (llama_model_params/n_gpu_layers params (int gpu-layers))
        m (Llama/llama_model_load_from_file (.allocateFrom arena path) params)]
    (when (null? m)
      (throw (llama-failure (str "llama.cpp could not load " path) {:reason :model-load-failed :model path})))
    m))

(defn- check-decoder-only [^MemorySegment m]
  (cond
    (Llama/llama_model_has_encoder m) (throw (llama-failure "encoder-decoder models are not supported" {:reason :unsupported-model}))
    (not (Llama/llama_model_has_decoder m)) (throw (llama-failure "the model has no decoder" {:reason :unsupported-model}))
    (Llama/llama_model_is_diffusion m) (throw (llama-failure "diffusion models are not supported" {:reason :unsupported-model}))))

(defn- new-context
  "A context of m: n-slots sequences in one unified cache of n-ctx cells."
  ^MemorySegment [^Arena arena ^MemorySegment m {:keys [n-ctx n-batch n-slots swa-full threads]}]
  (let [params (Llama/llama_context_default_params arena)]
    (doto params
      (llama_context_params/n_ctx (int n-ctx))
      (llama_context_params/n_batch (int (max (long n-batch) (long n-slots))))
      (llama_context_params/n_seq_max (int n-slots))
      (llama_context_params/kv_unified true)
      ;; no snapshots of recurrent states, as gen.llm.llama.recurrent replays
      (llama_context_params/n_rs_seq (int 0))
      (llama_context_params/swa_full (boolean swa-full))
      (llama_context_params/no_perf true))
    (when threads
      (llama_context_params/n_threads params (int threads))
      (llama_context_params/n_threads_batch params (int threads)))
    (let [ctx (Llama/llama_init_from_model m params)]
      (when (null? ctx)
        (throw (llama-failure "llama.cpp could not create a context" {:reason :context-failed :n-ctx n-ctx})))
      ctx)))

(defn- check-context
  "ctx has the sequences asked for, and room for a step of all of them."
  [^MemorySegment ctx ^long n-slots]
  (let [n-seq (Llama/llama_n_seq_max ctx)
        n-batch (Llama/llama_n_batch ctx)]
    (when-not (and (== n-slots n-seq) (<= n-slots n-batch))
      (throw (llama-failure "llama.cpp changed the context's sequences or batch"
                            {:reason :context-failed :n-seq-max n-seq :n-batch n-batch})))))

(defn- memory
  "ctx's memory (its KV cache)."
  ^MemorySegment [^MemorySegment ctx]
  (let [mem (Llama/llama_get_memory ctx)]
    (when (null? mem)
      (throw (llama-failure "the model keeps no memory (an embedding model?)" {:reason :unsupported-model})))
    mem))

(defn- model-desc [^MemorySegment model]
  (with-open [a (Arena/ofConfined)]
    (let [buf (.allocate a 256)
          n (Llama/llama_model_desc model buf 256)]
      (if (neg? n) "" (.getString buf 0)))))

(defn- describe
  "The Slots' info: sizes, special tokens, the model."
  [^MemorySegment m ^MemorySegment ctx ^MemorySegment vocab {:keys [model n-slots]}]
  (let [n-vocab (Llama/llama_vocab_n_tokens vocab)]
    {:engine :llama.cpp
     :n-vocab n-vocab
     :n-slots n-slots
     :n-ctx (long (Llama/llama_n_ctx_seq ctx))
     :n-batch (long (Llama/llama_n_batch ctx))
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
     :gpu-offload? (Llama/llama_supports_gpu_offload)}))

(defn- open-slots
  "LlamaSlots for opts, made on the engine's thread. A failure frees what
   was made, last first."
  [opts]
  (LlamaRuntime/load)
  (LlamaRuntime/clearLastError)
  (let [{:keys [model gpu-layers n-slots] :as opts} (check-options opts)
        split? (split-recurrent?)
        arena (Arena/ofConfined)
        undo (volatile! (list #(.close arena)))
        owned (fn [free x] (vswap! undo conj #(free x)) x)]
    (try
      (let [m (owned Llama/llama_model_free (doto (load-model arena model gpu-layers) check-decoder-only))
            ctx (owned Llama/llama_free (new-context arena m opts))
            _ (check-context ctx n-slots)
            n-batch (long (Llama/llama_n_batch ctx))
            batch (owned #(Llama/llama_batch_free (.-struct ^LlamaRuntime$Batch %))
                         (LlamaRuntime$Batch. (Llama/llama_batch_init arena (int n-batch) (int 0) (int 1)) n-batch))
            pool (owned #(.shutdown ^ExecutorService %) (row-pool))
            vocab (Llama/llama_model_get_vocab m)
            described (describe m ctx vocab opts)
            splits (when (and split? (some (:model described) [:recurrent? :hybrid?]))
                     (volatile! {:cells (recurrent/empty-cells n-slots) :off nil :steps 0 :split 0 :decodes 0}))
            info (assoc described :split-recurrent? (some? splits))]
        (LlamaSlots. m ctx (memory ctx) vocab batch arena pool (:n-vocab info) n-batch info splits))
      (catch Throwable t
        (run! #(%) @undo)
        (throw t)))))

(defn engine
  "A llama.cpp engine (java.io.Closeable; close it to free the model), on a
   thread of its own. opts (any other is an error):
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

(defn- on-llama
  "(f slots) on llama.cpp engine e's thread, slots its LlamaSlots."
  [e f]
  (slots/call e (fn [s]
                  (if (instance? LlamaSlots s)
                    (f s)
                    (throw (ex-info "Not a llama.cpp engine" {:reason :not-a-llama-engine :engine e}))))))

(defn- sized
  "[n buf] of (call capacity), a llama.cpp call that returns the size it
   wrote, or minus the size it needs: called again with that size once,
   and a loud error if it still does not fit (or overflows)."
  [what call ^long capacity]
  (let [[n :as r] (call capacity)
        [n :as r] (if (and (neg? (long n)) (not= n Integer/MIN_VALUE)) (call (- (long n))) r)]
    (when (neg? (long n))
      (throw (ex-info (str what ": llama.cpp's result does not fit") {:reason :llama-failed :n n})))
    r))

(defn tokenize
  "The tokens of text, by the model's tokenizer: a vector. opts:
   :add-special? (default true) adds BOS (and EOS) as the model does;
   :parse-special? (default false) reads special tokens written in the
   text (\"<|im_start|>\") as themselves."
  ([e text] (tokenize e text {}))
  ([e ^String text {:keys [add-special? parse-special?] :or {add-special? true parse-special? false}}]
   (let [utf8 (.getBytes text StandardCharsets/UTF_8)]
     (on-llama e (fn [^LlamaSlots s]
                   (with-open [a (Arena/ofConfined)]
                     (let [src (.allocateFrom a ValueLayout/JAVA_BYTE utf8)
                           call (fn [^long cap]
                                  (let [out (.allocate a (* 4 (max 1 cap)))]
                                    [(Llama/llama_tokenize (.-vocab s) src (int (alength utf8)) out (int cap)
                                                           (boolean add-special?) (boolean parse-special?))
                                     out]))
                           [n ^MemorySegment out] (sized "tokenize" call (+ 8 (alength utf8)))]
                       (mapv long (.toArray (.asSlice out 0 (* 4 (long n))) ValueLayout/JAVA_INT)))))))))

(defn token-bytes
  "The bytes of token, raw (a token may hold part of a UTF-8 character;
   special tokens are rendered): a byte array."
  ^bytes [e token]
  (let [[t] (br/check-tokens e [token] "token-bytes")]
    (on-llama e (fn [^LlamaSlots s]
                  (with-open [a (Arena/ofConfined)]
                    (let [call (fn [^long cap]
                                 (let [buf (.allocate a (max 1 cap))]
                                   [(Llama/llama_token_to_piece (.-vocab s) (int t) buf (int cap) (int 0) true) buf]))
                          [n ^MemorySegment buf] (sized "token-bytes" call 64)]
                      (.toArray (.asSlice buf 0 (long n)) ValueLayout/JAVA_BYTE)))))))

(defn detokenize
  "The text of tokens (special tokens rendered), decoded as UTF-8."
  ^String [e tokens]
  (let [toks (int-array (br/check-tokens e tokens "detokenize"))]
    (on-llama e (fn [^LlamaSlots s]
                  (with-open [a (Arena/ofConfined)]
                    (let [src (.allocateFrom a ValueLayout/JAVA_INT toks)
                          call (fn [^long cap]
                                 (let [buf (.allocate a (max 1 cap))]
                                   [(Llama/llama_detokenize (.-vocab s) src (int (alength toks)) buf (int cap) false true)
                                    buf]))
                          [n ^MemorySegment buf] (sized "detokenize" call (* 8 (inc (alength toks))))]
                      (String. ^bytes (.toArray (.asSlice buf 0 (long n)) ValueLayout/JAVA_BYTE)
                               StandardCharsets/UTF_8)))))))

(defn step-splits
  "Diagnostics: how a recurrent or hybrid model's engine split its steps
   ({:steps :split :decodes :off}, :off nil while it replays llama.cpp's
   cells, else why it stopped), or nil for an engine that does not split
   them (an attention model, or gen.llama.split-recurrent false)."
  [e]
  (on-llama e (fn [^LlamaSlots s]
                (some-> ^clojure.lang.Volatile (.-splits s) deref (dissoc :cells)))))

(defn kv-positions
  "Diagnostics: [min max] of the positions each slot holds in the KV cache
   ([-1 -1] when empty), by slot (the last is the scratch slot). A
   recurrent or hybrid model's min is its state's, the last position."
  [e]
  (on-llama e (fn [^LlamaSlots s]
                (mapv (fn [slot] [(long (Llama/llama_memory_seq_pos_min (.-mem s) (int slot)))
                                  (long (Llama/llama_memory_seq_pos_max (.-mem s) (int slot)))])
                      (range (:n-slots (.-info s)))))))

(defn system-info
  "llama.cpp's description of its build and the machine (its backends)."
  ^String []
  (LlamaRuntime/load)
  (.getString (Llama/llama_print_system_info) 0))
