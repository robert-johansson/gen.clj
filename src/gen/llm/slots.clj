(ns gen.llm.slots
  "A Branching engine over numbered sequence slots, as llama.cpp's KV cache
   has them: the lifetimes, capacity and threading of gen.llm.branching,
   written once over the Slots operations an engine implements
   (gen.llm.llama for llama.cpp; a pure-JVM toy engine in the tests).

   - One thread per engine. Every Slots call runs on it; the Branching
     operations submit a task and wait (or run at once when called from the
     engine's thread), so callers on any thread are serialized. A caller
     interrupted while it waits still gets the result (its interrupt flag
     set again), since the operation has consumed its branches; so does one
     interrupted while the engine is made, which would otherwise leak it.
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
     unreachable is freed too: a phantom reference the engine drains at the
     start of each operation (so :live counts dropped branches until then).
     When slots run short, or a read finds the engine's memory full (a full
     KV cache) with its slots restored, the engine runs the GC, frees what
     it found and tries again before it refuses. A consumed branch owns
     nothing, so collecting it frees nothing. An engine that becomes
     unreachable (with its branches) is closed; close it explicitly to free
     its model when you are done.
   - Every check runs before any Slots call, and a failed call leaves the
     slots as they were, or frees them and their branches, loudly, so an
     error never leaves a branch half-advanced: llama.cpp aborts the
     process on some misuse (an out-of-range sequence id, a batch over
     n_batch), so nothing reaches it unchecked."
  {:api :experimental}
  (:require [gen.llm.branching :as br])
  (:import [gen.llm.branching Branch]
           [java.lang.ref Cleaner Cleaner$Cleanable PhantomReference ReferenceQueue]
           [java.util.concurrent CompletableFuture CompletionException ExecutorService Executors
            RejectedExecutionException]
           [java.util.function Supplier]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defprotocol Slots
  "A slot-based engine's operations, called only on its thread, with
   arguments the caller has checked (slots in range, positions those of the
   slots' contents, tokens in the vocabulary, at most n-slots tokens in a
   step); a run may be of any length within n-ctx, which the engine splits
   into decodes as it needs. Log-probabilities are float arrays of n-vocab
   entries.
   A run or step that throws leaves its slots as they were, unless its
   ex-data names :lost-slots, which the caller then frees. One that throws
   with :exhausted? and no :lost-slots (a full KV cache, its slots
   restored) is retried once after the GC has freed dropped branches. copy
   and clear do not fail."
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

(defn- submit
  "(f) on exec, as a CompletableFuture."
  ^CompletableFuture [^ExecutorService exec f what]
  (try
    (CompletableFuture/supplyAsync (reify Supplier (get [_] (f))) exec)
    (catch RejectedExecutionException _
      (throw (closed-error what)))))

(defn- await-value
  "The value of fut, or what its task threw. The wait goes on through
   interrupts and sets the interrupt flag again (CompletableFuture/join):
   the task runs to its end anyway, and its result must not be lost."
  [^CompletableFuture fut]
  (try
    (.join fut)
    (catch CompletionException x
      (throw (.getCause x)))))

(defn- run-on
  "Run (f) on the engine's thread and return its value (or throw what it
   threw), waiting for it."
  [exec thread f what]
  (if (identical? (Thread/currentThread) thread)
    (f)
    (await-value (submit exec f what))))

;; ---------------------------------------------------------------------------
;; The engine's books
;; ---------------------------------------------------------------------------

;; state: a volatile holding the books, an immutable map written only on
;; the engine thread (read from any thread by -live? and info):
;;   :live  {id {:slot s :ref phantom}}   the live branches
;;   :refs  {phantom id}                  every phantom not yet cleared
;;   :free  sorted set of free slots (never the scratch slot)
;;   :next-id, :collections (GC backstops run), :closed?, :cleanable
;;   (closes the engine, once)

(defn- with-branch
  "The books with live branch id on slot, its phantom ref."
  [books id slot ref]
  (-> books
      (update :next-id inc)
      (assoc-in [:live id] {:slot slot :ref ref})
      (assoc-in [:refs ref] id)))

(defn- without-branch
  "The books without live branch id (the slot neither freed nor taken)."
  [books id]
  (let [{:keys [ref]} (get-in books [:live id])]
    (-> books (update :live dissoc id) (update :refs dissoc ref))))

(defn- slot-holder
  "The id of the live branch on slot, or nil."
  [books slot]
  (some (fn [[id entry]] (when (= slot (:slot entry)) id)) (:live books)))

(defn- closed-books [books]
  (assoc books :closed? true :live {} :refs {} :free (sorted-set)))

(declare engine-info open* fork* advance* extend* score* resample* dispose*)

(deftype SlotEngine [impl info ^ExecutorService exec ^Thread thread ^ReferenceQueue queue state]
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
  (close [_]
    (.clean ^Cleaner$Cleanable (:cleanable @state))))

(defn- engine-info [^SlotEngine e]
  (let [st @(.-state e)]
    (-> (.-info e)
        (dissoc :n-slots :scratch-slot)
        (assoc :live (count (:live st))
               :free (count (:free st))
               :collections (:collections st)
               :closed? (boolean (:closed? st))))))

(defn- impl [^SlotEngine e] (.-impl e))
(defn- books-of "The volatile of e's books." [^SlotEngine e] (.-state e))
(defn- scratch-slot ^long [^SlotEngine e] (:scratch-slot (.-info e)))

(defn- check-open [e what]
  (when (:closed? @(books-of e)) (throw (closed-error what))))

(defn- free-slot!
  [e slot]
  (slots-clear (impl e) slot)
  (vswap! (books-of e) update :free conj slot))

(defn- forget!
  "Branch id leaves the books, its phantom cleared (collecting the branch
   then frees nothing): its slot, neither freed nor taken."
  [e id]
  (let [{:keys [slot ^PhantomReference ref]} (get (:live @(books-of e)) id)]
    (.clear ref)
    (vswap! (books-of e) without-branch id)
    slot))

(defn- consume!
  "b's successor takes over its slot: b is no longer live."
  [e ^Branch b]
  (forget! e (.-id b)))

(defn- release-ref!
  "A phantom came off the queue: free its branch's slot if the branch was
   dropped live (not consumed). True if it freed one."
  [e ref]
  (if-let [id (get (:refs @(books-of e)) ref)]
    (do (free-slot! e (forget! e id)) true)
    false))

(defn- drain!
  "Free the slots of the dropped branches the GC has found."
  [^SlotEngine e]
  (loop []
    (when-let [r (.poll ^ReferenceQueue (.-queue e))]
      (release-ref! e r)
      (recur))))

(defn- collect!
  "Run the GC and free the slots of the unreachable branches it finds: how
   many. Counted in (info engine) as :collections."
  ^long [^SlotEngine e]
  (vswap! (books-of e) update :collections inc)
  (System/gc)
  ;; the reference handler enqueues what the GC found after the GC returns
  (loop [wait 100 freed 0]
    (if-let [r (.remove ^ReferenceQueue (.-queue e) wait)]
      (recur 10 (cond-> freed (release-ref! e r) inc))
      freed)))

(defn- begin!
  "Every operation's start: the engine is open, and dropped branches are
   freed."
  [e what]
  (check-open e what)
  (drain! e))

(defn- retrying
  "(f), and once more after a collection if it failed for want of memory
   with its slots restored (:exhausted?, no :lost-slots) and the GC freed a
   dropped branch."
  [e f]
  (try
    (f)
    (catch clojure.lang.ExceptionInfo x
      (let [{:keys [exhausted? lost-slots]} (ex-data x)]
        (if (and exhausted? (empty? lost-slots) (pos? (collect! e)))
          (f)
          (throw x))))))

(defn- ensure-free!
  "At least n free slots, collecting unreachable branches if need be."
  [^SlotEngine e ^long n what]
  (when (< (count (:free @(books-of e))) n)
    (collect! e)
    (let [{:keys [free live]} @(books-of e)
          max-branches (:max-branches (.-info e))]
      (when (< (count free) n)
        (throw (ex-info (str what ": " n " more branches would exceed the engine's " max-branches
                             " (" (count live) " live): dispose! some, or open the engine with more")
                        {:reason :no-free-branches :requested n :free (count free)
                         :live (count live) :max-branches max-branches}))))))

(defn- take-slots!
  "n free slots, taken (the caller has ensured them)."
  [e n]
  (let [ss (vec (take n (:free @(books-of e))))]
    (vswap! (books-of e) update :free #(reduce disj % ss))
    ss))

(defn- make-branch!
  "A new live branch owning slot."
  [^SlotEngine e tokens logprobs slot]
  (let [id (:next-id @(books-of e))
        b (br/->Branch e id tokens logprobs)]
    (vswap! (books-of e) with-branch id slot (PhantomReference. b (.-queue e)))
    b))

(defn- live-entry
  "The live entry of branch b, or a loud error."
  [e ^Branch b what]
  (when-not (identical? e (.-engine b))
    (throw (ex-info (str what ": the branch belongs to another engine") {:reason :engines-differ})))
  (or (get (:live @(books-of e)) (.-id b))
      (throw (ex-info (str what ": " b " is not live: advance, extend or resample consumed it (use the branch"
                           " it returned), or it was disposed")
                      {:reason :not-live :branch b}))))

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
    (when-let [id (slot-holder @(books-of e) slot)]
      (forget! e id))
    (free-slot! e slot)))

(defn child-slots
  "Each child's slot in a resample: its ancestor's own slot for the
   ancestor's first child, the next of fresh for each later one."
  [slots ancestors fresh]
  (loop [[i & is :as anc] ancestors, seen #{}, fresh fresh, out []]
    (cond (empty? anc) out
          (seen i) (recur is seen (rest fresh) (conj out (first fresh)))
          :else (recur is (conj seen i) fresh (conj out (slots i))))))

;; ---------------------------------------------------------------------------
;; The operations (engine thread)
;; ---------------------------------------------------------------------------

(defn- open* [e tokens]
  (begin! e "open")
  (check-room e 0 (count tokens) "open")
  (ensure-free! e 1 "open")
  (let [[slot] (take-slots! e 1)
        r (try
            (retrying e #(slots-run (impl e) slot 0 tokens nil))
            (catch Throwable t
              (free-slot! e slot)
              (throw t)))]
    (make-branch! e tokens (:logprobs r) slot)))

(defn- fork* [e ^Branch b n]
  (begin! e "fork")
  (let [{:keys [slot]} (live-entry e b "fork")]
    (ensure-free! e n "fork")
    (let [ss (take-slots! e n)]
      (try
        (doseq [s ss] (slots-copy (impl e) slot s))
        (catch Throwable t
          (run! #(free-slot! e %) ss)
          (throw t)))
      (mapv #(make-branch! e (.-tokens b) (.-logprobs b) %) ss))))

(defn- advance* [e bs ts]
  (begin! e "advance")
  (let [slots (mapv #(:slot (live-entry e % "advance")) bs)
        positions (mapv br/position bs)]
    (doseq [p positions] (check-room e p 1 "advance"))
    (let [rows (try
                 (retrying e #(slots-step (impl e) slots positions ts))
                 (catch Throwable t
                   (lose! e t)
                   (throw t)))]
      (run! #(consume! e %) bs)
      (mapv (fn [^Branch b t row slot] (make-branch! e (conj (.-tokens b) t) row slot))
            bs ts rows slots))))

(defn- extend* [e ^Branch b ts]
  (begin! e "extend")
  (let [{:keys [slot]} (live-entry e b "extend")
        pos (br/position b)]
    (check-room e pos (count ts) "extend")
    (let [r (try
              (retrying e #(slots-run (impl e) slot pos ts (subvec ts 1)))
              (catch Throwable t
                (lose! e t)
                (throw t)))]
      (consume! e b)
      {:branch (make-branch! e (into (.-tokens b) ts) (:logprobs r) slot)
       :scores (into [(br/logprob b (first ts))] (:scores r))})))

(defn- score* [e ^Branch b ts]
  (begin! e "score")
  (let [{:keys [slot]} (live-entry e b "score")
        m (count ts)
        pos (br/position b)
        first-lp (br/logprob b (first ts))]
    (if (= 1 m)
      [first-lp]
      (let [scratch (scratch-slot e)]
        ;; the continuation but its last token, read into a copy of the
        ;; branch: its rows score every token after the first
        (check-room e pos (dec m) "score")
        (slots-copy (impl e) slot scratch)
        (try
          (let [r (retrying e #(slots-run (impl e) scratch pos (subvec ts 0 (dec m)) (subvec ts 1 (dec m))))]
            (-> [first-lp]
                (into (:scores r))
                (conj (double (aget ^floats (:logprobs r) (int (peek ts)))))))
          (catch clojure.lang.ExceptionInfo x
            ;; a lost scratch slot is no branch's: cleared below
            (throw (ex-info (ex-message x) (dissoc (ex-data x) :lost-slots) x)))
          (finally
            (slots-clear (impl e) scratch)))))))

(defn- resample* [e bs ancestors]
  (begin! e "resample")
  (let [slots (mapv #(:slot (live-entry e % "resample")) bs)
        children (frequencies ancestors)
        dead (remove children (range (count bs)))
        copies (- (count ancestors) (count children))]
    ;; capacity first, so a refusal changes nothing
    (ensure-free! e (max 0 (- copies (count dead))) "resample")
    (run! #(consume! e %) bs)
    (doseq [i dead] (free-slot! e (slots i)))
    (let [dsts (child-slots slots ancestors (take-slots! e copies))]
      (try
        (doseq [[i d] (map vector ancestors dsts)
                :when (not= d (slots i))]
          (slots-copy (impl e) (slots i) d))
        (catch Throwable t
          ;; copies do not fail, by the Slots contract; if one does, its
          ;; slots are no branch's: free them rather than leak them
          (run! #(free-slot! e %) (distinct dsts))
          (throw (ex-info (str "resample failed; its branches are freed: " (ex-message t))
                          {:reason :resample-failed :freed-slots (vec (distinct dsts)) :cause-data (ex-data t)}
                          t))))
      (mapv (fn [i s]
              (let [^Branch p (bs i)]
                (make-branch! e (.-tokens p) (.-logprobs p) s)))
            ancestors dsts))))

(defn- dispose* [e bs]
  (begin! e "dispose!")
  (run! #(live-entry e % "dispose!") bs)
  (run! #(free-slot! e (consume! e %)) bs)
  nil)

;; ---------------------------------------------------------------------------
;; Construction
;; ---------------------------------------------------------------------------

(defonce ^:private cleaner (Cleaner/create))

(defn- closer
  "The action that closes an engine, on its thread, the first time it runs
   (a Cleanable runs it at most once): close calls it, and so does the
   Cleaner for an engine no one can reach any more. It holds what it needs,
   never the engine."
  [^ExecutorService exec thread impl state]
  (fn []
    (try
      (run-on exec thread
              #(when-not (:closed? @state)
                 (vswap! state closed-books)
                 (slots-close impl))
              "close")
      (finally (.shutdown exec)))))

(defn engine
  "A Branching engine (java.io.Closeable) over the Slots made by (make-impl),
   which runs on the engine's new thread, named name. Its last slot is
   the scratch slot; the others hold branches."
  ^java.io.Closeable [make-impl name]
  (let [exec (Executors/newSingleThreadExecutor
              (-> (Thread/ofPlatform) (.name (str "gen.llm " name)) (.daemon true) .factory))
        [thread impl info] (try
                             (await-value (submit exec #(let [s (make-impl)]
                                                          [(Thread/currentThread) s (slots-info s)])
                                                  "engine"))
                             (catch Throwable t
                               (.shutdown exec)
                               (throw t)))
        n-slots (long (:n-slots info))]
    (when (< n-slots 2)
      (run-on exec thread #(slots-close impl) "engine")
      (.shutdown exec)
      (throw (ex-info "An engine needs at least two slots (one for scoring)"
                      {:reason :invalid-option :n-slots n-slots})))
    (let [state (volatile! {:live {} :refs {} :free (into (sorted-set) (range (dec n-slots)))
                            :next-id 0 :collections 0 :closed? false})
          e (SlotEngine. impl
                         (assoc info :max-branches (dec n-slots) :scratch-slot (dec n-slots))
                         exec thread (ReferenceQueue.) state)]
      (vswap! state assoc :cleanable (.register ^Cleaner cleaner e (closer exec thread impl state)))
      e)))

(defn call
  "(f impl) on the engine's thread, for engine-specific operations
   (tokenizing, say), impl its Slots: its value, or what it threw. A loud
   error once the engine is closed."
  [^SlotEngine e f]
  (run-on (.-exec e) (.-thread e)
          #(do (check-open e "call") (f (.-impl e)))
          "call"))

(defn slot-of
  "Diagnostics: the slot holding branch b's state, or nil when b is not
   live."
  [^SlotEngine e ^Branch b]
  (run-on (.-exec e) (.-thread e) #(:slot (get (:live @(books-of e)) (.-id b))) "slot-of"))
