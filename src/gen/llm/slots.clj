(ns gen.llm.slots
  "A Branching engine over numbered sequence slots, as llama.cpp's KV cache
   has them: the lifetimes, capacity and threading of gen.llm.branching,
   written once over the Slots operations an engine implements
   (gen.llm.llama for llama.cpp; a pure-JVM toy engine in the tests).

   - One thread per engine. Every Slots call runs on it; the Branching
     operations submit a task and wait (or run at once when called from the
     engine's thread), so callers on any thread are serialized.
   - Each live branch owns one slot. The engine keeps one more, the scratch
     slot, for score, which copies its branch there and reads the
     continuation without touching the branch: no partial removal, which
     recurrent and hybrid models (GatedDeltaNet) cannot do past their few
     checkpoints. max-branches is the slot count less one.
   - advance, extend and resample consume their branches: a successor takes
     over the slot, and the old branch is no longer live. A free slot is
     always empty (cleared when freed), since a copy adds to what its
     destination holds.
   - Lifetime: dispose! frees a slot at once. A live branch that becomes
     unreachable is freed too (a phantom reference the engine thread
     drains before each operation); when slots run short the engine runs
     the GC and drains again before it refuses. A consumed branch owns
     nothing, so collecting it frees nothing.
   - Every check runs before any Slots call, and a failed call leaves the
     slots as they were, so an error never leaves a branch half-advanced:
     llama.cpp aborts the process on some misuse (an out-of-range sequence
     id, a batch over n_batch), so nothing reaches it unchecked."
  (:require [gen.llm.branching :as br])
  (:import [gen.llm.branching Branch]
           [java.lang.ref PhantomReference ReferenceQueue]
           [java.util.concurrent Callable ExecutionException ExecutorService Executors
            RejectedExecutionException ThreadFactory]))

(set! *warn-on-reflection* true)

(defprotocol Slots
  "A slot-based engine's operations, called only on its thread, with
   arguments the caller has checked (slots in range, positions those of the
   slots' contents, tokens in the vocabulary, at most n-slots tokens in a
   step); a run may be of any length within n-ctx, which the engine splits
   into decodes as it needs. Log-probabilities are float arrays of n-vocab
   entries.
   A run or step that throws leaves its slots as they were, unless its
   ex-data names :lost-slots, which the caller then frees."
  (slots-info [s] "{:n-vocab :n-slots :n-ctx …}, fixed for the engine's life")
  (slots-run [s slot pos tokens targets]
    "Read tokens (a vector) into slot, whose contents end at pos:
     {:logprobs after the last token, :scores the log-probability of
     (targets i) after token i, for each i below (count targets)}. targets
     is nil (no scores) or has one entry fewer than tokens.")
  (slots-step [s slots positions tokens]
    "Read one token into each of the distinct slots, in one decode: the
     log-probabilities after each, in order.")
  (slots-copy [s src dst] "Give the empty slot dst src's contents.")
  (slots-clear [s slot] "Empty the slot.")
  (slots-close [s] "Free everything."))

;; ---------------------------------------------------------------------------
;; The engine's thread
;; ---------------------------------------------------------------------------

(defn- closed-error [what]
  (ex-info (str what ": the engine is closed") {:reason :closed}))

(defn- run-on
  "Run (f) on the engine's thread and return its value (or throw what it
   threw), waiting for it."
  [^ExecutorService exec ^Thread thread f what]
  (if (identical? (Thread/currentThread) thread)
    (f)
    (let [^java.util.concurrent.Future fut
          (try
            (.submit exec ^Callable f)
            (catch RejectedExecutionException _
              (throw (closed-error what))))]
      (try
        (.get fut)
        (catch ExecutionException e
          (throw (.getCause e)))))))

;; ---------------------------------------------------------------------------
;; The engine's books (engine thread only)
;; ---------------------------------------------------------------------------

;; state: a volatile holding
;;   :live  {id {:slot s :ref phantom}}   the live branches
;;   :refs  {phantom id}                  every phantom not yet cleared
;;   :free  sorted set of free slots (never the scratch slot)
;;   :next-id, :closed?

(declare engine-info open* fork* advance* extend* score* resample* dispose* close*)

(deftype SlotEngine [spi info ^ExecutorService exec ^Thread thread ^ReferenceQueue queue state]
  br/Branching
  (-info [this] (engine-info this))
  (-open [this tokens] (run-on exec thread #(open* this tokens) "open"))
  (-fork [this b n] (run-on exec thread #(fork* this b n) "fork"))
  (-advance [this bs ts] (run-on exec thread #(advance* this bs ts) "advance"))
  (-extend [this b ts] (run-on exec thread #(extend* this b ts) "extend"))
  (-score [this b ts] (run-on exec thread #(score* this b ts) "score"))
  (-resample [this bs anc] (run-on exec thread #(resample* this bs anc) "resample"))
  (-dispose [this bs] (run-on exec thread #(dispose* this bs) "dispose!"))
  (-live? [_ b]
    (let [st @state]
      (and (not (:closed? st)) (contains? (:live st) (.-id ^Branch b)))))

  java.io.Closeable
  (close [this]
    (try
      (run-on exec thread #(close* this) "close")
      (catch clojure.lang.ExceptionInfo e
        (when-not (= :closed (:reason (ex-data e))) (throw e))))
    (.shutdown exec)))

(defn- engine-info [^SlotEngine e]
  (let [st @(.-state e)]
    (assoc (.-info e)
           :live (count (:live st))
           :free (count (:free st))
           :closed? (boolean (:closed? st)))))

(defn- spi [^SlotEngine e] (.-spi e))
(defn- state [^SlotEngine e] (.-state e))

(defn- check-open [e what]
  (when (:closed? @(state e)) (throw (closed-error what))))

(defn- release-ref!
  "A phantom reference came off the queue: free its branch's slot if the
   branch is still live (it was dropped, not consumed)."
  [e ref]
  (let [st @(state e)
        id (get (:refs st) ref)
        entry (get (:live st) id)]
    (vswap! (state e) update :refs dissoc ref)
    (when entry
      (slots-clear (spi e) (:slot entry))
      (vswap! (state e) #(-> % (update :live dissoc id) (update :free conj (:slot entry)))))))

(defn- drain!
  [^SlotEngine e]
  (loop []
    (when-let [r (.poll ^ReferenceQueue (.-queue e))]
      (release-ref! e r)
      (recur))))

(defn- collect!
  "Run the GC and free the slots of the unreachable branches it finds."
  [^SlotEngine e]
  (System/gc)
  ;; the reference handler enqueues what the GC found after the GC returns
  (loop [wait 100]
    (when-let [r (.remove ^ReferenceQueue (.-queue e) wait)]
      (release-ref! e r)
      (recur 10))))

(defn- ensure-free!
  "At least n free slots, collecting unreachable branches if need be."
  [e n what]
  (drain! e)
  (when (< (count (:free @(state e))) n)
    (collect! e)
    (let [free (count (:free @(state e)))]
      (when (< free n)
        (let [{:keys [max-branches]} (.-info ^SlotEngine e)]
          (throw (ex-info (str what ": " n " more branches would exceed the engine's " max-branches
                               " (" (count (:live @(state e))) " live): dispose! some, or open the engine with more")
                          {:reason :no-free-branches :requested n :free free
                           :live (count (:live @(state e))) :max-branches max-branches})))))))

(defn- take-slots!
  "n free slots, taken (the caller has ensured them)."
  [e n]
  (let [ss (vec (take n (:free @(state e))))]
    (vswap! (state e) update :free #(reduce disj % ss))
    ss))

(defn- free-slot!
  [e slot]
  (slots-clear (spi e) slot)
  (vswap! (state e) update :free conj slot))

(defn- make-branch!
  "A new live branch owning slot."
  [^SlotEngine e tokens logprobs slot]
  (let [id (:next-id @(state e))
        b (Branch. e id tokens logprobs slot)
        ref (PhantomReference. b (.-queue e))]
    (vswap! (state e) #(-> %
                           (update :next-id inc)
                           (assoc-in [:live id] {:slot slot :ref ref})
                           (assoc-in [:refs ref] id)))
    b))

(defn- entry!
  "The live entry of branch b, or a loud error."
  [e ^Branch b what]
  (when-not (identical? e (.-engine b))
    (throw (ex-info (str what ": the branch belongs to another engine") {:reason :engines-differ})))
  (or (get (:live @(state e)) (.-id b))
      (throw (ex-info (str what ": " b " is not live: advance, extend or resample consumed it (use the branch"
                           " it returned), or it was disposed")
                      {:reason :not-live :branch b}))))

(defn- consume!
  "b's successor takes over its slot: b is no longer live, and collecting it
   frees nothing."
  [e ^Branch b]
  (let [{:keys [^PhantomReference ref]} (get (:live @(state e)) (.-id b))]
    (.clear ref)
    (vswap! (state e) #(-> % (update :live dissoc (.-id b)) (update :refs dissoc ref)))))

(defn- check-room
  "n more tokens fit in a branch at position pos?"
  [e pos n what]
  (let [n-ctx (long (:n-ctx (.-info ^SlotEngine e)))]
    (when (< n-ctx (+ (long pos) (long n)))
      (throw (ex-info (str what ": " (+ (long pos) (long n)) " tokens exceed the engine's context of " n-ctx
                           " tokens per branch")
                      {:reason :context-full :position pos :tokens n :n-ctx n-ctx})))))

(defn- lose!
  "After a run or step that left slots it could not restore: free them, and
   the live branches that held them."
  [e ex]
  (doseq [slot (:lost-slots (ex-data ex))]
    (when-let [[id {:keys [^PhantomReference ref]}] (first (filter #(= slot (:slot (val %))) (:live @(state e))))]
      (.clear ref)
      (vswap! (state e) #(-> % (update :live dissoc id) (update :refs dissoc ref))))
    (free-slot! e slot)))

;; ---------------------------------------------------------------------------
;; The operations (engine thread)
;; ---------------------------------------------------------------------------

(defn- logprob ^double [^Branch b t]
  (double (aget ^floats (.-logprobs b) (int t))))

(defn- open* [e tokens]
  (check-open e "open")
  (check-room e 0 (count tokens) "open")
  (ensure-free! e 1 "open")
  (let [[slot] (take-slots! e 1)
        r (try
            (slots-run (spi e) slot 0 tokens nil)
            (catch Throwable t
              (free-slot! e slot)
              (throw t)))]
    (make-branch! e tokens (:logprobs r) slot)))

(defn- fork* [e ^Branch b n]
  (check-open e "fork")
  (let [{:keys [slot]} (entry! e b "fork")]
    (ensure-free! e n "fork")
    (let [ss (take-slots! e n)]
      (try
        (doseq [s ss] (slots-copy (spi e) slot s))
        (catch Throwable t
          (doseq [s ss] (free-slot! e s))
          (throw t)))
      (mapv #(make-branch! e (.-tokens b) (.-logprobs b) %) ss))))

(defn- advance* [e bs ts]
  (check-open e "advance")
  (let [slots (mapv #(:slot (entry! e % "advance")) bs)
        positions (mapv br/position bs)]
    (doseq [p positions] (check-room e p 1 "advance"))
    (let [rows (try
                 (slots-step (spi e) slots positions ts)
                 (catch Throwable t
                   (lose! e t)
                   (throw t)))]
      (run! #(consume! e %) bs)
      (mapv (fn [^Branch b t row slot] (make-branch! e (conj (.-tokens b) t) row slot))
            bs ts rows slots))))

(defn- extend* [e ^Branch b ts]
  (check-open e "extend")
  (let [{:keys [slot]} (entry! e b "extend")
        pos (br/position b)]
    (check-room e pos (count ts) "extend")
    (let [r (try
              (slots-run (spi e) slot pos ts (subvec ts 1))
              (catch Throwable t
                (lose! e t)
                (throw t)))]
      (consume! e b)
      {:branch (make-branch! e (into (.-tokens b) ts) (:logprobs r) slot)
       :scores (into [(logprob b (first ts))] (:scores r))})))

(defn- score* [e ^Branch b ts]
  (check-open e "score")
  (let [{:keys [slot]} (entry! e b "score")
        m (count ts)
        pos (br/position b)
        first-lp (logprob b (first ts))]
    (if (= 1 m)
      [first-lp]
      (let [scratch (:scratch-slot (.-info ^SlotEngine e))]
        ;; the continuation but its last token, read into a copy of the
        ;; branch: its rows score every token after the first
        (check-room e pos (dec m) "score")
        (slots-copy (spi e) slot scratch)
        (try
          (let [r (slots-run (spi e) scratch pos (subvec ts 0 (dec m)) (subvec ts 1 (dec m)))]
            (-> [first-lp]
                (into (:scores r))
                (conj (double (aget ^floats (:logprobs r) (int (peek ts)))))))
          (finally
            (slots-clear (spi e) scratch)))))))

(defn- resample* [e bs ancestors]
  (check-open e "resample")
  (let [slots (mapv #(:slot (entry! e % "resample")) bs)
        children (frequencies ancestors)
        dead (remove children (range (count bs)))
        extra (- (count ancestors) (count children))]
    ;; capacity first, so a refusal changes nothing
    (ensure-free! e (max 0 (- extra (count dead))) "resample")
    (run! #(consume! e %) bs)
    (doseq [i dead] (free-slot! e (slots i)))
    (let [placed (loop [j 0 used #{} out []]
                   (if (= j (count ancestors))
                     out
                     (let [i (ancestors j)]
                       (if (used i)
                         (let [[s] (take-slots! e 1)]
                           (slots-copy (spi e) (slots i) s)
                           (recur (inc j) used (conj out [i s])))
                         (recur (inc j) (conj used i) (conj out [i (slots i)]))))))]
      (mapv (fn [[i s]]
              (let [^Branch p (bs i)]
                (make-branch! e (.-tokens p) (.-logprobs p) s)))
            placed))))

(defn- dispose* [e bs]
  (check-open e "dispose!")
  (let [slots (mapv #(:slot (entry! e % "dispose!")) bs)]
    (run! #(consume! e %) bs)
    (run! #(free-slot! e %) slots)
    nil))

(defn- close* [e]
  (when-not (:closed? @(state e))
    (vswap! (state e) assoc :closed? true :live {} :refs {} :free (sorted-set))
    (slots-close (spi e)))
  nil)

;; ---------------------------------------------------------------------------
;; Construction
;; ---------------------------------------------------------------------------

(defn engine
  "A Branching engine (java.io.Closeable) over the Slots made by (make-spi),
   which runs on the engine's new thread, named name. Its last slot is
   the scratch slot; the others hold branches."
  ^java.io.Closeable [make-spi name]
  (let [thread (promise)
        factory (reify ThreadFactory
                  (newThread [_ r]
                    (let [t (-> (Thread/ofPlatform) (.name (str "gen.llm " name)) (.daemon true) (.unstarted r))]
                      (deliver thread t)
                      t)))
        exec (Executors/newSingleThreadExecutor factory)
        spi (try
              (.get (.submit exec ^Callable make-spi))
              (catch ExecutionException e
                (.shutdown exec)
                (throw (.getCause e))))
        info (slots-info spi)
        n-slots (long (:n-slots info))]
    (when (< n-slots 2)
      (.get (.submit exec ^Callable #(slots-close spi)))
      (.shutdown exec)
      (throw (ex-info "An engine needs at least two slots (one for scoring)" {:n-slots n-slots})))
    (SlotEngine. spi
                 (assoc info :max-branches (dec n-slots) :scratch-slot (dec n-slots))
                 exec @thread (ReferenceQueue.)
                 (volatile! {:live {} :refs {} :free (into (sorted-set) (range (dec n-slots)))
                             :next-id 0 :closed? false}))))

(defn call
  "(f spi) on the engine's thread, for engine-specific operations
   (tokenizing, say): its value, or what it threw. A loud error once the
   engine is closed."
  [^SlotEngine e f]
  (run-on (.-exec e) (.-thread e)
          #(do (check-open e "call") (f (.-spi e)))
          "call"))
