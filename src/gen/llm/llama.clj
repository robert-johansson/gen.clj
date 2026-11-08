(ns gen.llm.llama
  "The llama.cpp engine (DESIGN.md §6): a Branching engine (gen.llm.branching)
   over one llama.cpp context, through the generated binding of llama.h
   (src-java/gen/llama/ffi, from native/jextract-llama.sh) and the libraries
   of native/build-llama.sh, both pinned to llama.cpp v0.9.0 (the binding
   generated from v0.7.0's llama.h, which v0.9.0's changes only by an ftype
   and a #define; its adapter functions from v0.9.0's).

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
     operations, never of timing. It pays on Metal and not on CUDA (where an
     extra decode costs more than the copies it saves), so it is on only
     where the weights are on Metal; gen.llama.split-recurrent true or false
     says otherwise. (info e) says :split-recurrent?, and step-splits what
     was split.
   - Encoder-decoder, diffusion and embedding models are refused.
   - The engine's memory is reserved in gen.array.memory until it closes,
     so the MLX backend's budget leaves it out (:footprint in info: the
     model's weights, its adapters' tensors, and the KV-cache,
     recurrent-state, output and compute buffers llama.cpp reported
     allocating, read from its log as the model, adapters and engine are
     made). The engine reserves its context's buffers; the weights and
     adapters are reserved once, by their model and adapter values.
   - A decode's rows of logits are normalized on a small pool of the
     engine's own (gen.llm.host's: up to four threads, or
     gen.llm.host-threads), while the engine's thread waits: normalizing a
     vocabulary on the host costs about as much as a row's share of the
     decode.
   - Accuracy depends on the device and the weights (DESIGN §6). On Metal
     a token's log-probability agrees with an f32 forward pass within about
     0.01 nats and varies with batch shape by at most about 0.05. With
     quantized weights (Q8_0) the x86 CPU and CUDA builds are about 0.2
     nats a token off that, and on CUDA the result also depends on what
     shares the batch; float weights (F16, BF16) bring both within about
     0.01. For exact scores and weights off Metal, use float weights.

   Models and LoRA adapters. An engine given a path loads a model of its
   own. (model path) loads one that several engines share, each over a
   context of its own, the weights held once; (adapter m path) loads a
   LoRA adapter's GGUF for model m, and an engine over m takes adapters
   with their scales (:adapters {adapter scale}): every branch of the
   engine reads under that set, fixed for the engine's life. Specialised
   critics and proposals from one base model are engines over one model
   with different adapters.

     (with-open [m (llama/model \"Qwen3-0.6B-Q8_0.gguf\")
                 critic (llama/adapter m \"critic-lora.gguf\")
                 base (llama/engine {:model m})
                 tuned (llama/engine {:model m :adapters {critic 1.0}})]
       …)

   Models and adapters are java.io.Closeable values, freed by close, or
   when they become unreachable. A model or an adapter closes only when no
   open engine uses it (a loud error otherwise, :in-use); closing a model
   frees its adapters. engine refuses an adapter of another model, a
   closed one, and an activated LoRA (aLoRA, whose activation after its
   invocation tokens this engine does not implement). Engines over one
   model each keep their own thread; a model's adapters are loaded and
   freed under its lock, which every decode reads under.

   Close engines and models before the JVM exits. llama.cpp's Metal
   backend frees its device as the process exits and asserts that none of
   its buffers is left (ggml-metal's GGML_ASSERT([rsets->data count] ==
   0)), so a JVM exiting with a model or an engine open aborts (status
   134), its work done. A shutdown hook closes what is still open
   (genclj-8dtg): each engine on its own thread, after the operation it
   runs (a thread still calling it then gets :closed), then each model no
   engine uses, under its write lock, after the decodes reading it. It
   waits at most gen.llama.close-at-exit seconds (default 10; 0, no
   hook), so an engine thread that never returns (one that called
   System/exit itself, say) delays the exit by that much, and the JVM then
   exits with the engine open, as without the hook. An engine or a model
   made while the hook runs is left open.

   llama.cpp's log goes to System.err from warnings up (gen.llama.log:
   none, error, warn, info, debug); libllama is found in gen.llama.lib
   (default native/build/llama/<platform>/lib, gen.llama.LlamaRuntime).
   tokenize, token-bytes and detokenize use the GGUF's own vocabulary;
   tokenizer gathers them as the LM protocol's tokenizer (gen.llm.lm).
   Each takes an engine, or a vocabulary loaded alone ((vocabulary path),
   llama.cpp's vocab_only: no weights), the tokenizer of another engine
   over the same GGUF (gen.llm.array-engine)."
  {:api :experimental}
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [gen.array.memory :as reservations]
            [gen.llm.branching :as br]
            [gen.llm.host :as host]
            [gen.llm.llama.recurrent :as recurrent]
            [gen.llm.slots :as slots])
  (:import [gen.llama LlamaRuntime LlamaRuntime$Batch]
           [gen.llama.ffi Llama llama_context_params llama_model_params]
           [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.lang.ref Cleaner Cleaner$Cleanable]
           [java.nio.charset StandardCharsets]
           [java.util.concurrent ExecutorService]
           [java.util.concurrent.locks Lock ReentrantReadWriteLock]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

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

(defn- llama-failure
  "The ex-info for a failure of what (\"llama/engine\"), with llama.cpp's
   last error on this thread."
  [what msg data]
  (ex-info (str what ": " msg) (assoc data :llama-error (llama-error))))

(defn- locked [^Lock l f]
  (.lock l)
  (try (f) (finally (.unlock l))))

;; A model's lock: engines read (make their contexts, decode, free them)
;; under its read lock, while adapters are loaded and freed, and the model
;; freed, under its write lock. llama.cpp keeps a model's adapters in a set
;; that a context reads when it sizes its graphs (at its first decode after
;; its adapters are set) and that loading or freeing an adapter changes.
;; Contexts of one model decoding on threads of their own is a use
;; llama.cpp tests (tests/test-thread-safety.cpp), so the read lock lets
;; them.

(defn- reading [^ReentrantReadWriteLock rw f] (locked (.readLock rw) f))
(defn- writing [^ReentrantReadWriteLock rw f] (locked (.writeLock rw) f))

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
  "llama_decode the batch (under the model's read lock), or throw its
   ex-info."
  [lock ^MemorySegment ctx ^LlamaRuntime$Batch batch what]
  (LlamaRuntime/clearLastError)
  (let [code (long (reading lock #(Llama/llama_decode ctx (.-struct batch))))]
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
  "The pool that normalizes a decode's rows, of the layer's default size
   (gen.llm.host: half the cores, up to four, or gen.llm.host-threads)."
  ^ExecutorService []
  (host/pool "gen.llm llama.cpp rows-" (host/default-threads)))

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

(defn- metal?
  "Are the model's weights on Metal (fp, the engine's footprint: llama.cpp's
   buffer names)?"
  [fp]
  (boolean (some (fn [{:keys [kind buffer]}] (and (= kind :model) (.startsWith (str buffer) "Metal")))
                 (:buffers fp))))

(defn- split-recurrent
  "gen.llama.split-recurrent, read once: does a recurrent or hybrid model's
   engine split its steps? true or false, or nil when it is not set: then
   only on Metal (the owner, DESIGN §9), where a split step of
   Qwen3.5-0.8B cost a quarter to a third of one decode; on CUDA (the RTX-6000 machine,
   2026-10-21) about twice one decode."
  []
  (let [v (System/getProperty "gen.llama.split-recurrent")]
    (case v
      nil nil
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
  [splits f & args]
  (when-let [st (some-> splits deref :cells)]
    (vswap! splits assoc :cells (apply f st args))))

(defn- replay-off!
  "The replay stops: steps are one decode from now on."
  [splits why]
  (vswap! splits assoc :cells nil :off why))

(defn- replay-checked!
  "After an operation on slots: the replay and llama.cpp agree on each
   one's state position (llama_memory_seq_pos_min: a recurrent state's
   position, -1 for none, for a hybrid model too; llama.cpp tells no cell's
   index). A disagreement stops the replay, loudly on System.err."
  [splits ^MemorySegment mem slots what]
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
  [splits ^Throwable t cleared]
  (when splits
    (if (contains? #{1 -1 nil} (:code (ex-data t)))
      (run! #(replayed! splits recurrent/clear %) cleared)
      (replay-off! splits (str "a decode failed (" (ex-message t) ")")))))

;; ---------------------------------------------------------------------------
;; Models and adapters
;; ---------------------------------------------------------------------------

(defonce ^:private cleaner (Cleaner/create))

;; ---------------------------------------------------------------------------
;; Closing at exit (genclj-8dtg)
;; ---------------------------------------------------------------------------

;; What is open, for the exit hook: {token (the action that closes the
;; engine)} and {a model's books [its lock, its Cleanable]}. Neither holds
;; an engine or a model, so neither keeps one reachable; an entry leaves
;; when its value is freed (closed, or by the Cleaner).
(defonce ^:private open-engines (atom {}))
(defonce ^:private open-models (atom {}))

(defn- exit-wait
  "gen.llama.close-at-exit, read once: the seconds the exit hook waits for
   the open engines and models to close (default 10; 0, no hook)."
  ^long []
  (let [v (System/getProperty "gen.llama.close-at-exit" "10")]
    (or (when (re-matches #"\d+" v) (parse-long v))
        (throw (ex-info (str "llama: gen.llama.close-at-exit is a whole number of seconds, not " (pr-str v))
                        {:reason :invalid-option :gen.llama.close-at-exit v})))))

(defn- close-open!
  "Every open engine closed (each on its thread, after the operation it
   runs), then every open model no engine uses (under its write lock,
   after the decodes reading it), what fails written to System.err."
  []
  (doseq [close (vals @open-engines)]
    (try (close)
         (catch Throwable t (.println System/err (str "gen.llm.llama at exit: an engine did not close: " t)))))
  (doseq [[books [lock ^Cleaner$Cleanable c]] @open-models]
    (try (writing lock #(when (empty? (:engines @books)) (.clean c)))
         (catch Throwable t (.println System/err (str "gen.llm.llama at exit: a model did not close: " t))))))

(defn- at-exit
  "The exit hook's thread: close-open! on a daemon thread, waited for at
   most secs seconds, so that an engine thread that never returns (one
   that called System/exit, say) delays the exit, never stops it."
  ^Thread [^long secs]
  (Thread. ^Runnable
   (fn []
     (let [t (doto (Thread. ^Runnable close-open! "gen.llm.llama close-at-exit") (.setDaemon true) (.start))]
       (.join t (* 1000 secs))
       (when (.isAlive t)
         (.println System/err (str "gen.llm.llama at exit: engines or models still open after " secs
                                   " s; exiting with them (llama.cpp's Metal backend may abort)")))))
   "gen.llm.llama exit"))

(defonce ^:private exit-hook
  (delay (let [secs (exit-wait)]
           (when (pos? secs)
             (.addShutdownHook (Runtime/getRuntime) (at-exit secs))))))

(defn- logged
  "[(f), what llama.cpp logged on this thread while f ran]."
  [f]
  (LlamaRuntime/startCapture)
  (let [v (try (f)
               (catch Throwable t
                 (LlamaRuntime/endCapture)
                 (throw t)))]
    [v (LlamaRuntime/endCapture)]))

(def ^:private buffer-line
  ;; llama.cpp (v0.7.0 to v0.9.0) reports each buffer it allocates, at info level:
  ;; "load_tensors:   Metal_Mapped model buffer size =   604.15 MiB",
  ;; "llama_kv_cache:      Metal KV buffer size =   448.00 MiB", the
  ;; recurrent state's (RS), output and compute buffers alike, and an
  ;; adapter's "llama_adapter_lora_init_impl:      Metal LoRA buffer size =     0.16 MiB"
  #"(?m)^\s*\w+:\s+(\S+)\s+(model|KV|RS|output|compute|LoRA) buffer size =\s+([0-9.]+) MiB")

(defn- buffers
  "The buffers llama.cpp reported allocating in log: [{:kind :model|:kv|:rs|
   :output|:compute|:lora, :buffer its backend's name, :bytes}], to 0.01
   MiB."
  [log]
  (mapv (fn [[_ buffer kind mib]]
          {:kind (keyword (.toLowerCase ^String kind))
           :buffer buffer
           :bytes (Math/round (* (Double/parseDouble mib) 1048576.0))})
        (re-seq buffer-line log)))

(defn- load-model
  "The model at path, layers on the GPU."
  ^MemorySegment [^Arena arena ^String path gpu-layers]
  (let [params (Llama/llama_model_default_params arena)
        _ (llama_model_params/n_gpu_layers params (int gpu-layers))
        m (Llama/llama_model_load_from_file (.allocateFrom arena path) params)]
    (when (null? m)
      (throw (llama-failure "llama/model" (str "llama.cpp could not load " path) {:reason :model-load-failed :model path})))
    m))

(defn- check-decoder-only [^MemorySegment m]
  (cond
    (Llama/llama_model_has_encoder m) (throw (llama-failure "llama/model" "encoder-decoder models are not supported" {:reason :unsupported-model}))
    (not (Llama/llama_model_has_decoder m)) (throw (llama-failure "llama/model" "the model has no decoder" {:reason :unsupported-model}))
    (Llama/llama_model_is_diffusion m) (throw (llama-failure "llama/model" "diffusion models are not supported" {:reason :unsupported-model}))))

(defn- c-string
  "The string a llama.cpp call (call buf capacity) writes, which returns
   its full length (snprintf's) or a negative number for none (then nil):
   called again with room for it when it did not fit."
  [call]
  (with-open [a (Arena/ofConfined)]
    (loop [cap 256]
      (let [buf (.allocate a (long cap))
            n (long (call buf cap))]
        (cond (neg? n) nil
              (< n cap) (.getString buf 0)
              :else (recur (inc n)))))))

(defn- model-description
  "The model's info: its path, llama.cpp's description, sizes, kind, and
   its footprint, {:bytes its weights (the model's size), :buffers its
   model buffers, read from log}. The weights count once: the model
   buffers of a mapped file are views of one mapping (Metal's maps the
   whole file, the CPU's the token embeddings again), so they are listed,
   not added."
  [^MemorySegment m path gpu-layers log]
  (let [bufs (filterv #(= :model (:kind %)) (buffers log))]
    (when-not (seq bufs)
      (throw (llama-failure "llama/model" "llama.cpp reported no model buffer, so the model's footprint is unknown"
                            {:reason :no-footprint})))
    {:path path
     :desc (or (c-string (fn [^MemorySegment buf ^long cap] (Llama/llama_model_desc m buf cap))) "")
     :bytes (Llama/llama_model_size m)
     :params (Llama/llama_model_n_params m)
     :n-ctx-train (long (Llama/llama_model_n_ctx_train m))
     :n-swa (long (Llama/llama_model_n_swa m))
     :recurrent? (Llama/llama_model_is_recurrent m)
     :hybrid? (Llama/llama_model_is_hybrid m)
     :gpu-layers gpu-layers
     :footprint {:bytes (Llama/llama_model_size m) :buffers bufs}}))

;; A model's books: an atom, written under its lock,
;;   {:closed? :engines #{a token per open engine}
;;    :adapters #{each loaded adapter's books} :cleanable (frees it, once)}
;; and its reservation's owner. They hold no engine or adapter value, so
;; that the Cleaner's action, which holds them, keeps none reachable.

(deftype LlamaModel [^MemorySegment ptr ^MemorySegment vocab ^ReentrantReadWriteLock lock books info]
  java.io.Closeable
  (close [_]
    (writing lock #(let [{:keys [closed? engines ^Cleaner$Cleanable cleanable]} @books]
                     (when (and (not closed?) (seq engines))
                       (throw (ex-info (str "llama/model: " (count engines) " open engines use " (:path info)
                                            ": close them first")
                                       {:reason :in-use :engines (count engines) :model (:path info)})))
                     (.clean cleanable))))

  Object
  (toString [_] (str "#<LlamaModel " (:desc info) ">")))

;; An adapter's books: {:closed? :users #{a token per open engine using it}
;; :cleanable}, its reservation's owner.

(deftype LlamaAdapter [^MemorySegment ptr ^LlamaModel model books info]
  java.io.Closeable
  (close [_]
    (writing (.-lock model)
             #(let [{:keys [closed? users ^Cleaner$Cleanable cleanable]} @books]
                (when (and (not closed?) (seq users))
                  (throw (ex-info (str "llama/adapter: " (count users) " open engines use " (:path info)
                                       ": close them first")
                                  {:reason :in-use :engines (count users) :adapter (:path info)})))
                (.clean cleanable))))

  Object
  (toString [_] (str "#<LlamaAdapter " (:path info) ">")))

(defn- model-closer
  "The action that frees a model, once: close runs it, and so does the
   Cleaner for a model no one can reach (its engines and adapters, which
   hold it, are closed by then). llama_model_free frees the adapters still
   loaded."
  [^MemorySegment ptr lock books]
  (fn []
    (writing lock #(when-not (:closed? @books)
                     (doseq [ab (:adapters @books)]
                       (swap! ab assoc :closed? true)
                       (reservations/release! ab))
                     (swap! books assoc :closed? true :adapters #{})
                     (Llama/llama_model_free ptr)
                     (swap! open-models dissoc books)
                     (reservations/release! books)))))

(defn- adapter-closer
  "The action that frees an adapter, once (close, or the Cleaner), unless
   its model has freed it."
  [^MemorySegment ptr lock model-books books]
  (fn []
    (writing lock #(when-not (:closed? @books)
                     (Llama/llama_adapter_lora_free ptr)
                     (swap! model-books update :adapters disj books)
                     (swap! books assoc :closed? true)
                     (reservations/release! books)))))

(defn- open-model
  "A LlamaModel of the GGUF at path, loaded on this thread, its weights
   reserved."
  ^LlamaModel [path gpu-layers]
  (LlamaRuntime/load)
  (LlamaRuntime/clearLastError)
  @exit-hook
  (let [[^MemorySegment ptr log] (logged #(with-open [arena (Arena/ofConfined)] (load-model arena path gpu-layers)))]
    (try
      (check-decoder-only ptr)
      (let [info (model-description ptr path gpu-layers log)
            lock (ReentrantReadWriteLock.)
            books (atom {:closed? false :engines #{} :adapters #{}})
            m (LlamaModel. ptr (Llama/llama_model_get_vocab ptr) lock books info)]
        (swap! books assoc :cleanable (.register ^Cleaner cleaner m (model-closer ptr lock books)))
        (reservations/reserve! books (:bytes (:footprint info)) (str "llama.cpp: " (:desc info)))
        ;; closed at exit if still open (genclj-8dtg)
        (swap! open-models assoc books [lock (:cleanable @books)])
        m)
      (catch Throwable t
        (Llama/llama_model_free ptr)
        (throw t)))))

(defn model
  "A llama.cpp model (java.io.Closeable; close it to free its weights), for
   engines (engine's :model) and adapters (adapter) to share. opts (any
   other is an error):
     :gpu-layers   layers offloaded to the GPU (default -1, all)
   An engine given a path instead loads a model of its own."
  (^java.io.Closeable [path] (model path {}))
  (^java.io.Closeable [path opts]
   (when-let [ks (seq (remove #{:gpu-layers} (keys opts)))]
     (throw (ex-info (str "llama/model: unknown options " (vec ks)) {:reason :invalid-option :unknown (vec ks)})))
   (let [{:keys [gpu-layers]} (merge {:gpu-layers -1} opts)]
     (when-not (int? gpu-layers)
       (throw (ex-info (str "llama/model: bad :gpu-layers: " (pr-str gpu-layers)) {:reason :invalid-option :gpu-layers gpu-layers})))
     (when-not (and (string? path) (.isFile (java.io.File. ^String path)))
       (throw (ex-info (str "llama/model: no model file at " (pr-str path)) {:reason :no-model :model path})))
     (open-model path gpu-layers))))

(defn- as-model ^LlamaModel [m what]
  (if (instance? LlamaModel m)
    m
    (throw (ex-info (str what ": not a llama.cpp model (llama/model): " (pr-str m)) {:reason :not-a-model}))))

(defn- as-adapter ^LlamaAdapter [a what]
  (if (instance? LlamaAdapter a)
    a
    (throw (ex-info (str what ": not a llama.cpp adapter (llama/adapter): " (pr-str a)) {:reason :not-an-adapter}))))

(defn- adapter-description
  "An adapter's info: its path and model, its GGUF metadata (scalars, as
   strings), alpha, aLoRA's invocation tokens ([] for a LoRA), and its
   footprint, the LoRA buffers in log."
  [^MemorySegment ptr path ^LlamaModel m log]
  (let [meta (into (sorted-map)
                   (map (fn [i]
                          [(c-string (fn [^MemorySegment buf ^long cap]
                                       (Llama/llama_adapter_meta_key_by_index ptr (int i) buf cap)))
                           (c-string (fn [^MemorySegment buf ^long cap]
                                       (Llama/llama_adapter_meta_val_str_by_index ptr (int i) buf cap)))]))
                   (range (Llama/llama_adapter_meta_count ptr)))
        n (Llama/llama_adapter_get_alora_n_invocation_tokens ptr)
        bufs (filterv #(= :lora (:kind %)) (buffers log))]
    {:path path
     :model (select-keys (.-info m) [:path :desc])
     :alpha (some-> (get meta "adapter.lora.alpha") parse-double)
     :meta meta
     :alora-invocation-tokens (if (zero? n)
                                []
                                (mapv long (.toArray (.reinterpret (Llama/llama_adapter_get_alora_invocation_tokens ptr)
                                                                   (* 4 n))
                                                     ValueLayout/JAVA_INT)))
     :footprint {:bytes (reduce + 0 (map :bytes bufs)) :buffers bufs}}))

(defn adapter
  "A LoRA adapter (java.io.Closeable; close it to free it) for model m (a
   llama/model): the adapter GGUF at path, as llama.cpp's
   convert_lora_to_gguf.py writes one (general.type adapter, the model's
   architecture, the A and B of each weight it adapts). Loaded once, for
   any number of engines over m, each with its own scale (engine's
   :adapters). llama.cpp refuses one of another architecture or shapes (a
   loud error, :adapter-load-failed, with its message)."
  ^java.io.Closeable [m path]
  (let [m (as-model m "llama/adapter")]
    (when-not (and (string? path) (.isFile (java.io.File. ^String path)))
      (throw (ex-info (str "llama/adapter: no adapter file at " (pr-str path)) {:reason :no-adapter :adapter path})))
    (LlamaRuntime/clearLastError)
    (writing (.-lock m)
             (fn []
               (when (:closed? @(.-books m))
                 (throw (ex-info (str "llama/adapter: " m " is closed") {:reason :closed :model (:path (.-info m))})))
               (let [[^MemorySegment ptr log] (logged #(with-open [a (Arena/ofConfined)]
                                                         (Llama/llama_adapter_lora_init (.-ptr m) (.allocateFrom a ^String path))))]
                 (when (null? ptr)
                   (throw (llama-failure "llama/adapter" (str "llama.cpp could not load " path " for " (:path (.-info m)))
                                         {:reason :adapter-load-failed :adapter path :model (:path (.-info m))})))
                 (let [info (try (adapter-description ptr path m log)
                                 (catch Throwable t
                                   (Llama/llama_adapter_lora_free ptr)
                                   (throw t)))
                       books (atom {:closed? false :users #{}})
                       a (LlamaAdapter. ptr m books info)]
                   (swap! books assoc :cleanable
                          (.register ^Cleaner cleaner a (adapter-closer ptr (.-lock m) (.-books m) books)))
                   (swap! (.-books m) update :adapters conj books)
                   (reservations/reserve! books (:bytes (:footprint info)) (str "llama.cpp LoRA: " path))
                   a))))))

(defn model-info
  "A model's description: :path, :desc (llama.cpp's), :bytes, :params,
   :n-ctx-train, :n-swa, :recurrent?, :hybrid?, :gpu-layers, :footprint
   ({:bytes :buffers}), the :engines and :adapters open over it, :closed?."
  [m]
  (let [m (as-model m "llama/model-info")
        {:keys [closed? engines adapters]} @(.-books m)]
    (assoc (.-info m) :closed? (boolean closed?) :engines (count engines) :adapters (count adapters))))

(defn adapter-info
  "An adapter's description: :path, :model ({:path :desc}), :alpha (nil
   when its GGUF has none: then llama.cpp scales B A by an engine's scale
   alone, else by scale × alpha / rank), :meta (its GGUF's scalar
   metadata, as strings), :alora-invocation-tokens ([] but for an aLoRA),
   :footprint ({:bytes :buffers}), the :engines using it, :closed?."
  [a]
  (let [a (as-adapter a "llama/adapter-info")
        {:keys [closed? users]} @(.-books a)]
    (assoc (.-info a) :closed? (boolean closed?) :engines (count users))))

(defn- check-attachable
  "Model m and the adapters can make a new engine's context: m open, each
   adapter open, m's, and a LoRA (an aLoRA's activation is not
   implemented). Under m's lock."
  [^LlamaModel m adapters]
  (when (:closed? @(.-books m))
    (throw (ex-info (str "llama/engine: " m " is closed") {:reason :closed :model (:path (.-info m))})))
  (doseq [^LlamaAdapter a (keys adapters)
          :let [{:keys [path alora-invocation-tokens]} (.-info a)]]
    (cond
      (:closed? @(.-books a))
      (throw (ex-info (str "llama/engine: the adapter " path " is closed") {:reason :closed :adapter path}))

      (not (identical? m (.-model a)))
      (throw (ex-info (str "llama/engine: the adapter " path " was loaded for another model ("
                           (:path (:model (.-info a))) "), not this engine's " m ": load it for this one (llama/adapter)")
                      {:reason :adapter-for-another-model :adapter path
                       :adapter-model (:path (:model (.-info a))) :model (:path (.-info m))}))

      (seq alora-invocation-tokens)
      (throw (ex-info (str "llama/engine: " path " is an activated LoRA (aLoRA), which applies only after its"
                           " invocation tokens; this engine applies an adapter to every token")
                      {:reason :unsupported-adapter :adapter path :alora-invocation-tokens alora-invocation-tokens})))))

(defn- register!
  "An engine (its token) uses m and the adapters."
  [^LlamaModel m token adapters]
  (swap! (.-books m) update :engines conj token)
  (doseq [^LlamaAdapter a (keys adapters)]
    (swap! (.-books a) update :users conj token)))

(defn- deregister!
  "The engine (its token) no longer uses m and the adapters."
  [^LlamaModel m token adapters]
  (doseq [^LlamaAdapter a (keys adapters)]
    (swap! (.-books a) update :users disj token))
  (swap! (.-books m) update :engines disj token))

(defn- set-adapters!
  "Set the adapters, with their scales, on ctx (none: nothing to set)."
  [^MemorySegment ctx adapters]
  (when (seq adapters)
    (with-open [a (Arena/ofConfined)]
      (let [n (count adapters)
            ptrs (.allocate a ValueLayout/ADDRESS (long n))
            scales (.allocateFrom a ValueLayout/JAVA_FLOAT (float-array (map #(.floatValue ^Number %) (vals adapters))))]
        (doseq [[i ^LlamaAdapter ad] (map-indexed vector (keys adapters))]
          (.setAtIndex ptrs ValueLayout/ADDRESS (long i) ^MemorySegment (.-ptr ad)))
        (let [code (Llama/llama_set_adapters_lora ctx ptrs (long n) scales)]
          (when-not (zero? code)
            (throw (llama-failure "llama/engine" (str "llama.cpp could not set the adapters (" code ")")
                                  {:reason :context-failed :code code}))))))))

;; An engine's Slots: model, the LlamaModel it reads (owns-model? when the
;; engine loaded it from a path, to free it on closing), and its adapters
;; ({adapter scale}), all three used under token in their books.

(deftype LlamaSlots [^LlamaModel model ^MemorySegment ctx ^MemorySegment mem ^MemorySegment vocab
                     ^LlamaRuntime$Batch batch ^Arena arena ^ExecutorService pool ^long n-vocab ^long n-batch info
                     ^clojure.lang.Volatile splits token adapters owns-model?]
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
            (decode! (.-lock model) ctx batch "read")
            (replayed! splits recurrent/decode [slot] [(+ (long pos) end -1)])
            (replay-checked! splits mem [slot] "read")
            ;; row k scores the token after token from+k; the run's last
            ;; row is its next-token distribution instead
            (let [scores (if tgts
                           (into scores (host/in-order pool
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
            (decode! (.-lock model) ctx batch "advance")
            (replayed! splits recurrent/decode ss ps)
            (replay-checked! splits mem ss "advance")
            (doseq [[i r] (map vector g (host/in-order pool #(row-logprobs % n-vocab "advance")
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

  (slots-close [this]
    (swap! open-engines dissoc token)
    (reservations/release! this)
    (.shutdown pool)
    (Llama/llama_batch_free (.-struct batch))
    (reading (.-lock model) #(Llama/llama_free ctx))
    (deregister! model token adapters)
    (when owns-model? (.close model))
    (.close arena))

  Object
  (toString [_] (str "#<LlamaSlots " (:desc (:model info)) ">")))

;; ---------------------------------------------------------------------------
;; Opening
;; ---------------------------------------------------------------------------

(def ^:private defaults {:max-branches 8 :n-ctx 4096 :n-batch 512 :swa-full true :adapters {}})

(defn- scale?
  "A finite number a float holds: an adapter's scale."
  [s]
  (and (number? s) (Double/isFinite (double s)) (<= (Math/abs (double s)) (double Float/MAX_VALUE))))

(def ^:private option-checks
  "The options an engine takes, each with its check (a bad :model, no
   model file there, has an error of its own, :no-model)."
  {:model #(or (instance? LlamaModel %) (and (string? %) (.isFile (java.io.File. ^String %))))
   :max-branches pos-int? :n-ctx pos-int? :n-batch pos-int? :gpu-layers int?
   :threads #(or (nil? %) (pos-int? %)) :swa-full boolean?
   :adapters #(and (map? %) (every? (fn [[a s]] (and (instance? LlamaAdapter a) (scale? s))) %))})

(defn- check-options
  "opts with their defaults and :n-slots, checked: a loud error for an
   unknown, missing or bad one."
  [opts]
  (when-let [ks (seq (remove option-checks (keys opts)))]
    (throw (ex-info (str "llama/engine: unknown options " (vec ks)) {:reason :invalid-option :unknown (vec ks)})))
  (let [path? (string? (:model opts))
        {:keys [model max-branches adapters] :as opts} (merge defaults (when path? {:gpu-layers -1}) opts)
        ;; LLAMA_MAX_SEQ, 256 in v0.7.0 and v0.9.0
        max-seq (Llama/llama_max_parallel_sequences)]
    (doseq [[k ok?] option-checks
            :when (and (or (= k :model) (contains? opts k)) (not (ok? (get opts k))))]
      (throw (if (= k :model)
               (ex-info (str "llama/engine: no model file at " (pr-str model)) {:reason :no-model :model model})
               (ex-info (str "llama/engine: bad " k ": " (pr-str (get opts k))) {:reason :invalid-option k (get opts k)}))))
    (when (and (not path?) (contains? opts :gpu-layers))
      (throw (ex-info "llama/engine: :gpu-layers is the model's option (llama/model), not an engine's over it"
                      {:reason :invalid-option :gpu-layers (:gpu-layers opts)})))
    (when (and path? (seq adapters))
      (throw (ex-info (str "llama/engine: an engine given a path loads a model of its own, which no adapter was"
                           " loaded for: load the model (llama/model) and its adapters (llama/adapter), and give"
                           " the engine the model")
                      {:reason :adapter-for-another-model :model model})))
    (when-not (< (long max-branches) max-seq)
      (throw (ex-info (str "llama/engine: :max-branches is 1 to " (dec max-seq) " (llama.cpp's " max-seq
                           " sequences, one kept for scoring), not " max-branches)
                      {:reason :invalid-option :max-branches max-branches})))
    (assoc opts :n-slots (inc (long max-branches)))))

(defn- cpu-list
  "The CPUs a Linux cpulist names (\"0-7,16\" → 0 to 7 and 16)."
  [^String s]
  (into [] (mapcat (fn [part]
                     (let [[a b] (str/split (str/trim part) #"-")]
                       (range (parse-long a) (inc (long (parse-long (or b a))))))))
        (remove str/blank? (str/split (str/trim s) #","))))

(defn ^:no-doc performance-cores
  "On a Linux hybrid CPU (Intel's P- and E-cores), the number of physical
   performance cores: the CPUs /sys/devices/cpu_core/cpus lists, counted by
   their (package, core) pairs, so that hyper-threads count once. nil on
   any other machine. root is sysfs's mount (\"/sys\")."
  ([] (performance-cores "/sys"))
  ([root]
   (let [f (io/file root "devices" "cpu_core" "cpus")]
     (when (.isFile f)
       (let [cpus (cpu-list (slurp f))
             topo (fn [cpu name] (let [g (io/file root "devices" "system" "cpu" (str "cpu" cpu) "topology" name)]
                                   (when (.isFile g) (str/trim (slurp g)))))
             cores (into #{} (map (fn [c] [(topo c "physical_package_id") (or (topo c "core_id") (str "cpu" c))])) cpus)]
         (when (seq cpus) (count cores)))))))

(defn- default-threads
  "llama.cpp's threads when the caller gives none: the physical performance
   cores on a Linux hybrid CPU, whose all-cores default decodes several
   times slower (genclj-xe8c: 8.9 against 38.9 tokens/s on a Core Ultra 9
   285K, 24 cores against its 8 P-cores); nil, llama.cpp's own default,
   elsewhere."
  []
  (try (performance-cores) (catch Exception _ nil)))

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
    (when-let [threads (or threads (default-threads))]
      (llama_context_params/n_threads params (int threads))
      (llama_context_params/n_threads_batch params (int threads)))
    (let [ctx (Llama/llama_init_from_model m params)]
      (when (null? ctx)
        (throw (llama-failure "llama/engine" "llama.cpp could not create a context" {:reason :context-failed :n-ctx n-ctx})))
      ctx)))

(defn- check-context
  "ctx has the sequences asked for, and room for a step of all of them."
  [^MemorySegment ctx ^long n-slots]
  (let [n-seq (Llama/llama_n_seq_max ctx)
        n-batch (Llama/llama_n_batch ctx)]
    (when-not (and (== n-slots n-seq) (<= n-slots n-batch))
      (throw (llama-failure "llama/engine" "llama.cpp changed the context's sequences or batch"
                            {:reason :context-failed :n-seq-max n-seq :n-batch n-batch})))))

(defn- memory
  "ctx's memory (its KV cache)."
  ^MemorySegment [^MemorySegment ctx]
  (let [mem (Llama/llama_get_memory ctx)]
    (when (null? mem)
      (throw (llama-failure "llama/engine" "the model keeps no memory (an embedding model?)" {:reason :unsupported-model})))
    mem))

(defn- footprint
  "What engine uses: :bytes and :buffers of model m's weights, the
   adapters' tensors, and the context's buffers (ctx-buffers: KV cache,
   recurrent state, output and compute, from the log of its making). The
   compute buffer is the one sized as the context was made, before its
   adapters were set; a rank-r adapter adds r values a token to it."
  [^LlamaModel m adapters ctx-buffers]
  (let [parts (concat [(:footprint (.-info m))]
                      (map #(:footprint (.-info ^LlamaAdapter %)) (keys adapters))
                      [{:bytes (reduce + 0 (map :bytes ctx-buffers)) :buffers ctx-buffers}])]
    {:bytes (reduce + 0 (map :bytes parts))
     :buffers (into [] (mapcat :buffers) parts)}))

(defn- vocab-info
  "A vocabulary's size and special tokens (-1 for none)."
  [^MemorySegment vocab]
  (let [n-vocab (Llama/llama_vocab_n_tokens vocab)]
    {:n-vocab n-vocab
     :bos (long (Llama/llama_vocab_bos vocab))
     :eos (long (Llama/llama_vocab_eos vocab))
     :eog (into (sorted-set) (filter #(Llama/llama_vocab_is_eog vocab (int %))) (range n-vocab))
     :add-bos? (Llama/llama_vocab_get_add_bos vocab)}))

(defn- describe
  "The Slots' info: sizes, special tokens, the model, its adapters, the
   footprint."
  [^LlamaModel m ^MemorySegment ctx ^MemorySegment vocab adapters {:keys [n-slots]} footprint]
  (merge
    (vocab-info vocab)
    {:engine :llama.cpp
     :footprint footprint
     :n-slots n-slots
     :n-ctx (long (Llama/llama_n_ctx_seq ctx))
     :n-batch (long (Llama/llama_n_batch ctx))
     :model (select-keys (.-info m) [:path :desc :bytes :params :n-ctx-train :n-swa :recurrent? :hybrid?])
     :adapters (->> adapters
                    (map (fn [[^LlamaAdapter a s]] {:path (:path (.-info a)) :scale (double s) :alpha (:alpha (.-info a))}))
                    (sort-by (juxt :path :scale))
                    vec)
     :gpu-offload? (Llama/llama_supports_gpu_offload)}))

(defn- attach!
  "On the engine's thread, under model m's read lock: check that m and the
   adapters can serve a new engine, record that the engine (its token)
   uses them, make its context and set the adapters on it: [ctx, the log
   of its making]. (undo! f) records how to undo each step (run the last
   first, after a failure)."
  [^LlamaModel m token adapters ^Arena arena opts undo!]
  (reading (.-lock m)
           (fn []
             (check-attachable m adapters)
             (register! m token adapters)
             (undo! #(deregister! m token adapters))
             (let [[ctx log] (logged #(new-context arena (.-ptr m) opts))]
               (undo! #(reading (.-lock m) (fn [] (Llama/llama_free ctx))))
               (set-adapters! ctx adapters)
               [ctx log]))))

(defn- open-slots
  "LlamaSlots for opts, made on the engine's thread (and its model, when
   opts give a path), its context's buffers reserved, token its key in its
   books. A failure frees what was made, last first."
  [opts token]
  (LlamaRuntime/load)
  (LlamaRuntime/clearLastError)
  (let [{:keys [model gpu-layers n-slots adapters] :as opts} (check-options opts)
        split (split-recurrent)
        arena (Arena/ofConfined)
        undo (volatile! (list #(.close arena)))
        undo! (fn [f] (vswap! undo conj f))
        owned (fn [free x] (undo! #(free x)) x)]
    (try
      (let [owns-model? (string? model)
            ^LlamaModel m (if owns-model?
                            (owned #(.close ^java.io.Closeable %) (open-model model gpu-layers))
                            model)
            [^MemorySegment ctx log] (attach! m token adapters arena opts undo!)
            _ (check-context ctx n-slots)
            n-batch (long (Llama/llama_n_batch ctx))
            batch (owned #(Llama/llama_batch_free (.-struct ^LlamaRuntime$Batch %))
                         (LlamaRuntime$Batch. (Llama/llama_batch_init arena (int n-batch) (int 0) (int 1)) n-batch))
            pool (owned #(.shutdown ^ExecutorService %) (row-pool))
            ctx-buffers (filterv #(#{:kv :rs :output :compute} (:kind %)) (buffers log))
            fp (footprint m adapters ctx-buffers)
            described (describe m ctx (.-vocab m) adapters opts fp)
            splits (when (and (if (nil? split) (metal? fp) split) (some (:model described) [:recurrent? :hybrid?]))
                     (volatile! {:cells (recurrent/empty-cells n-slots) :off nil :steps 0 :split 0 :decodes 0}))
            info (assoc described :split-recurrent? (some? splits))
            slots (LlamaSlots. m ctx (memory ctx) (.-vocab m) batch arena pool (:n-vocab info) n-batch info splits
                               token adapters owns-model?)]
        (reservations/reserve! slots (reduce + 0 (map :bytes ctx-buffers)) (str "llama.cpp context: " (:desc (:model info))))
        slots)
      (catch Throwable t
        ;; every undo runs; one that fails is suppressed onto t
        (throw (br/clean-up t @undo))))))

(defn engine
  "A llama.cpp engine (java.io.Closeable; close it to free its context, and
   the model it loaded), on a thread of its own. opts (any other is an
   error):
     :model         the GGUF file, which the engine loads (and frees), or a
                    model (llama/model) it shares (required)
     :adapters      {adapter scale}: LoRA adapters of that model
                    (llama/adapter), each with its scale (a finite number;
                    0 leaves it out), under which every branch reads
                    (default {}; needs a model, not a path)
     :max-branches  live branches at most (default 8; up to 255)
     :n-ctx         KV cells, shared by every branch, and the most tokens
                    one branch holds (default 4096)
     :n-batch       tokens per decode (default 512, at least
                    :max-branches + 1); longer reads take several
     :gpu-layers    layers offloaded to the GPU (default -1, all), with a
                    path (a model has its own, llama/model's option)
     :threads       CPU threads (default: on a Linux hybrid CPU its physical
                    performance cores, else llama.cpp's own)
     :swa-full      a full-size cache for sliding-window layers (default true)"
  ^java.io.Closeable [opts]
  (let [token (Object.)
        e (slots/engine #(open-slots opts token) "llama.cpp")]
    ;; closed at exit if still open (genclj-8dtg)
    (swap! open-engines assoc token (slots/closer-of e))
    e))

;; ---------------------------------------------------------------------------
;; The vocabulary (the GGUF's own tokenizer)
;; ---------------------------------------------------------------------------

;; A vocabulary alone: a model loaded vocab_only (no weights, no context),
;; its books {:closed? :cleanable}. Its calls run on the caller's thread,
;; one at a time (under the value's lock).

(deftype LlamaVocabulary [^MemorySegment model ^MemorySegment vocab books info]
  java.io.Closeable
  (close [_] (.clean ^Cleaner$Cleanable (:cleanable @books)))

  Object
  (toString [_] (str "#<LlamaVocabulary " (:path info) ", " (:n-vocab info) " tokens>")))

(defn vocabulary
  "The vocabulary of the GGUF at path, loaded alone (llama.cpp's
   vocab_only: no weights, no context): a java.io.Closeable value that
   tokenize, token-bytes, detokenize and tokenizer take as they take an
   engine, for an engine of another kind over the same file (gen.llm.
   array-engine). Close it to free it (or let it become unreachable).
   Experimental."
  ^java.io.Closeable [path]
  (when-not (and (string? path) (.isFile (java.io.File. ^String path)))
    (throw (ex-info (str "llama/vocabulary: no model file at " (pr-str path)) {:reason :no-model :model path})))
  (LlamaRuntime/load)
  (LlamaRuntime/clearLastError)
  (let [^MemorySegment m (with-open [arena (Arena/ofConfined)]
                           (let [params (Llama/llama_model_default_params arena)]
                             (llama_model_params/vocab_only params true)
                             (llama_model_params/n_gpu_layers params (int 0))
                             (Llama/llama_model_load_from_file (.allocateFrom arena ^String path) params)))]
    (when (null? m)
      (throw (llama-failure "llama/vocabulary" (str "llama.cpp could not load the vocabulary of " path)
                            {:reason :model-load-failed :model path})))
    (let [vocab (Llama/llama_model_get_vocab m)
          books (atom {:closed? false})
          v (LlamaVocabulary. m vocab books (assoc (vocab-info vocab) :path path))]
      (swap! books assoc :cleanable (.register ^Cleaner cleaner v
                                               (fn [] (locking books
                                                        (when-not (:closed? @books)
                                                          (swap! books assoc :closed? true)
                                                          (Llama/llama_model_free m))))))
      v)))

(defn- on-vocab
  "(f vocab) with x's vocabulary (a segment): on a llama.cpp engine's
   thread, or for a vocabulary under its lock. Loud for anything else, and
   for a closed vocabulary."
  [x what f]
  (if (instance? LlamaVocabulary x)
    (let [^LlamaVocabulary v x
          books (.-books v)]
      (locking books
        (when (:closed? @books)
          (throw (ex-info (str what ": the vocabulary is closed") {:reason :closed})))
        (f (.-vocab v))))
    (slots/call x (fn [s]
                    (if (instance? LlamaSlots s)
                      (f (.-vocab ^LlamaSlots s))
                      (throw (ex-info "Not a llama.cpp engine or vocabulary" {:reason :not-a-llama-engine :engine x})))))))

(defn- vocabulary-info
  "{:n-vocab :bos :eos :eog :add-bos?} of an engine or a vocabulary."
  [x]
  (if (instance? LlamaVocabulary x) (.-info ^LlamaVocabulary x) (br/info x)))

(defn- check-tokens
  "tokens checked against x's vocabulary (br/check-tokens's rule)."
  [x tokens what]
  (if (instance? LlamaVocabulary x)
    (let [n (long (:n-vocab (.-info ^LlamaVocabulary x)))]
      (mapv (fn [t] (if (br/index? t n) (long t) (throw (br/invalid-token what t n)))) tokens))
    (br/check-tokens x tokens what)))

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

(defn- piece
  "token's bytes in vocab (special tokens rendered)."
  ^bytes [^MemorySegment vocab ^long t]
  (with-open [a (Arena/ofConfined)]
    (let [call (fn [^long cap]
                 (let [buf (.allocate a (max 1 cap))]
                   [(Llama/llama_token_to_piece vocab (int t) buf (int cap) (int 0) true) buf]))
          [n ^MemorySegment buf] (sized "token-bytes" call 64)]
      (.toArray (.asSlice buf 0 (long n)) ValueLayout/JAVA_BYTE))))

(defn tokenize
  "The tokens of text, by the model's tokenizer (e an engine or a
   vocabulary): a vector. opts: :add-special? (default true) adds BOS (and
   EOS) as the model does; :parse-special? (default false) reads special
   tokens written in the text (\"<|im_start|>\") as themselves."
  ([e text] (tokenize e text {}))
  ([e ^String text {:keys [add-special? parse-special?] :or {add-special? true parse-special? false}}]
   (let [utf8 (.getBytes text StandardCharsets/UTF_8)]
     (on-vocab e "tokenize"
               (fn [^MemorySegment vocab]
                 (with-open [a (Arena/ofConfined)]
                   (let [src (.allocateFrom a ValueLayout/JAVA_BYTE utf8)
                         call (fn [^long cap]
                                (let [out (.allocate a (* 4 (max 1 cap)))]
                                  [(Llama/llama_tokenize vocab src (int (alength utf8)) out (int cap)
                                                         (boolean add-special?) (boolean parse-special?))
                                   out]))
                         [n ^MemorySegment out] (sized "tokenize" call (+ 8 (alength utf8)))]
                     (mapv long (.toArray (.asSlice out 0 (* 4 (long n))) ValueLayout/JAVA_INT)))))))))

(defn token-bytes
  "The bytes of token, raw (a token may hold part of a UTF-8 character;
   special tokens are rendered): a byte array. e an engine or a
   vocabulary."
  ^bytes [e token]
  (let [[t] (check-tokens e [token] "token-bytes")]
    (on-vocab e "token-bytes" (fn [vocab] (piece vocab t)))))

(defn detokenize
  "The text of tokens (special tokens rendered), decoded as UTF-8. e an
   engine or a vocabulary."
  ^String [e tokens]
  (let [toks (int-array (check-tokens e tokens "detokenize"))]
    (on-vocab e "detokenize"
              (fn [^MemorySegment vocab]
                (with-open [a (Arena/ofConfined)]
                  (let [src (.allocateFrom a ValueLayout/JAVA_INT toks)
                        call (fn [^long cap]
                               (let [buf (.allocate a (max 1 cap))]
                                 [(Llama/llama_detokenize vocab src (int (alength toks)) buf (int cap) false true)
                                  buf]))
                        [n ^MemorySegment buf] (sized "detokenize" call (* 8 (inc (alength toks))))]
                    (String. ^bytes (.toArray (.asSlice buf 0 (long n)) ValueLayout/JAVA_BYTE)
                             StandardCharsets/UTF_8)))))))

(defn tokenizer
  "The model's vocabulary and tokenizer, as the LM protocol takes them
   (gen.llm.lm), of e, an engine or a vocabulary: the size and special
   tokens (:bos and :eos nil when the model has none), every token's
   bytes, read once in one call, and tokenize and detokenize."
  [e]
  (let [{:keys [n-vocab bos eos eog add-bos?]} (vocabulary-info e)
        pieces (on-vocab e "tokenizer" (fn [vocab] (mapv #(piece vocab %) (range n-vocab))))
        special (fn [t] (when-not (neg? (long t)) t))]
    {:n-vocab n-vocab
     :bos (special bos)
     :eos (special eos)
     :eog eog
     :add-bos? add-bos?
     :token-bytes (fn [t] (aclone ^bytes (pieces t)))
     :encode (fn [text opts] (tokenize e text opts))
     :decode (fn [tokens] (detokenize e tokens))}))

(defn- on-llama
  "(f slots) on llama.cpp engine e's thread, slots its LlamaSlots."
  [e f]
  (slots/call e (fn [s]
                  (if (instance? LlamaSlots s)
                    (f s)
                    (throw (ex-info "Not a llama.cpp engine" {:reason :not-a-llama-engine :engine e}))))))

(defn step-splits
  "Diagnostics: how a recurrent or hybrid model's engine split its steps
   ({:steps :split :decodes :off}, :off nil while it replays llama.cpp's
   cells, else why it stopped), or nil for an engine that does not split
   them (an attention model, or gen.llama.split-recurrent false)."
  [e]
  (on-llama e (fn [^LlamaSlots s]
                (some-> (.-splits s) deref (dissoc :cells)))))

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
