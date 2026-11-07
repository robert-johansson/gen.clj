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
     n_batch), so nothing reaches it unchecked.

   Continuous batching (genclj-b3ow). An engine whose Slots implement
   Batching, and whose slots-info has :batching, reads concurrent callers'
   advances, extends and opens together, in forwards over every read in
   flight; every other engine runs one operation at a time, in order, as
   above, and never waits for anything else.
   - advance, extend and open are requests: checked when they reach the
     engine's thread (the same checks, in the same order), then held in
     flight. Every other operation runs at once. One that needs a branch a
     request in flight holds waits until that request ends, then runs: it
     finds the branch consumed (or, after a failed read, as it was), an
     order its concurrent callers could have had.
   - Between forwards the thread takes every task queued, then runs one
     forward (slots-forward) over the reads in flight: the engine reads
     what it chooses (a long read in chunks over several forwards, other
     requests admitted and finished between them), and the requests it
     finished are answered.
   - The engine never samples, so a caller's next step comes only after
     its last one returned: before a forward the thread waits for the
     callers the last forward answered to come back with their next
     request, for at most (:wait (:batching info)) of that forward's time.
     A lone caller is never kept waiting.
   - Each request's results are those of its reads alone within the
     engine's band (its kernels' rounding depends on a forward's shape,
     as llama.cpp's does). A program calling from one thread is never
     batched with itself, so its results repeat bit for bit; concurrent
     callers' depend on how their calls were gathered.
   - A forward that fails fails every request in it: their branches stay
     as they were, except the slots it names lost and those of reads it
     had begun in an earlier forward, which are freed with their
     branches."
  {:api :experimental}
  (:require [gen.llm.branching :as br])
  (:import [gen.llm.branching Branch]
           [java.lang.ref Cleaner Cleaner$Cleanable PhantomReference ReferenceQueue]
           [java.util.concurrent CompletableFuture CompletionException LinkedBlockingQueue TimeUnit]))

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

(defprotocol Batching
  "Slots that read many requests' tokens in one forward (continuous
   batching), which the engine uses when slots-info has :batching
   {:wait w}: w, a fraction of a forward's time, bounds how long the
   engine waits for its callers' next requests before a forward."
  (slots-forward [s reads]
    "One forward over reads, a vector of {:slot :pos :tokens :targets} of
     distinct slots, each a read of tokens (a vector) into slot, whose
     contents end at pos, scoring (targets i) after token i (targets nil
     for no scores, else one entry fewer than tokens, as slots-run's).
     The engine reads the first k tokens of each, k of its choosing (0 to
     all of them; at least one read gets some), and returns, by read,
     {:read k :scores (the scores of the targets among them, when targets
     are given) :logprobs (after the k-th token, when k is all of them)}.
     A read read in part goes on in a later forward, from where it
     stopped (the next read's :pos and :tokens say so). It throws as
     slots-step does: its slots as they were unless named :lost-slots."))

;; ---------------------------------------------------------------------------
;; The engine's thread
;; ---------------------------------------------------------------------------

(defn- closed-error [what]
  (ex-info (str what ": the engine is closed") {:reason :closed}))

(defn- await-value
  "The value of fut, or what its task threw. The wait goes on through
   interrupts and sets the interrupt flag again (CompletableFuture/join):
   the task runs to its end anyway, and its result must not be lost."
  [^CompletableFuture fut]
  (try
    (.join fut)
    (catch CompletionException x
      (throw (.getCause x)))))

;; A worker is the engine's thread and its queue of tasks:
;;   {:run f :fut cf}                    an operation, (f) run at once
;;   {:admit f :fut cf :thread t
;;    :engine e}                         a request of a Batching engine:
;;                                       (f) admits it, forwards read it
;;   ::stop                              the last task: the thread ends
;; Tasks are submitted under lock while the worker is open, and the stop is
;; the last, so nothing submitted is ever left unrun. An engine without
;; Batching submits only operations: the thread runs them one at a time,
;; in order (run-tasks), as a single-thread executor would.

(deftype Worker [^LinkedBlockingQueue queue ^Thread thread open?])

(defn- enqueue!
  "Put task on w's queue: its future. A loud error once w is shut down."
  ^CompletableFuture [^Worker w task what]
  (locking w
    (when-not @(.-open? w) (throw (closed-error what)))
    ;; offer, never put: an interrupted caller still submits (put waits
    ;; interruptibly for the queue's lock)
    (.offer ^LinkedBlockingQueue (.-queue w) task))
  (:fut task))

(defn- run-on
  "Run (f) on the engine's thread and return its value (or throw what it
   threw), waiting for it."
  [^Worker w f what]
  (if (identical? (Thread/currentThread) (.-thread w))
    (f)
    (await-value (enqueue! w {:run f :fut (CompletableFuture.)} what))))

(defn- shutdown!
  "No task is taken any more; the thread runs those queued and ends."
  [^Worker w]
  (locking w
    (when @(.-open? w)
      (vreset! (.-open? w) false)
      (.offer ^LinkedBlockingQueue (.-queue w) ::stop))))

(declare run-task! schedule! fail-inflight!)

(defn- work
  "The engine thread's loop: each task in order, and while requests are in
   flight (a Batching engine's), the scheduler's forwards between tasks."
  [^LinkedBlockingQueue queue]
  (let [ls (volatile! {:inflight [] :parked [] :expect #{} :deadline 0})]
    (loop []
      (let [task (.take queue)]
        (when-not (= ::stop task)
          (run-task! ls task)
          (when (or (empty? (:inflight @ls))
                    (try (schedule! ls queue)
                         ;; a scheduler's bug: its callers get it, never a
                         ;; dead thread
                         (catch Throwable t (fail-inflight! ls t) true)))
            (recur)))))))

(defn- worker
  "A worker on a new daemon thread named name."
  ^Worker [name]
  (let [q (LinkedBlockingQueue.)
        t (-> (Thread/ofPlatform) (.name (str "gen.llm " name)) (.daemon true) (.unstarted #(work q)))]
    (.start t)
    (Worker. q t (volatile! true))))

;; ---------------------------------------------------------------------------
;; The engine's books
;; ---------------------------------------------------------------------------

;; state: a volatile holding the books, an immutable map written only on
;; the engine thread (read from any thread by -live? and info):
;;   :live  {id {:slot s :ref phantom}}   the live branches
;;   :refs  {phantom id}                  every phantom not yet cleared
;;   :free  sorted set of free slots (never the scratch slot)
;;   :busy  the ids of the branches requests in flight hold (Batching)
;;   :next-id, :collections (GC backstops run), :closed?, :cleanable
;;   (closes the engine, once)
;;   :schedule  the scheduler's counts (schedule)

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
  (assoc books :closed? true :live {} :refs {} :free (sorted-set) :busy #{}))

(declare engine-info operate open-request open* fork* advance-request advance* extend-request extend* score* resample*
         dispose*)

(deftype SlotEngine [impl info ^Worker worker ^ReferenceQueue queue state]
  br/Branching
  (-info [this] (engine-info this))
  (-open [this tokens] (operate this "open" #(open-request this tokens) #(open* this tokens)))
  (-fork [this b n] (run-on worker #(fork* this b n) "fork"))
  (-advance [this bs ts] (operate this "advance" #(advance-request this bs ts) #(advance* this bs ts)))
  (-extend [this b ts] (operate this "extend" #(extend-request this b ts) #(extend* this b ts)))
  (-score [this b ts] (run-on worker #(score* this b ts) "score"))
  (-resample [this bs anc] (run-on worker #(resample* this bs anc) "resample"))
  (-dispose [this bs] (run-on worker #(dispose* this bs) "dispose!"))
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

(defn- busy-error
  "The error of an operation on a branch a request in flight holds, which
   the scheduler catches to run the operation again once the request has
   ended (busy?)."
  [what b]
  (ex-info (str what ": " b " is being read by another caller's request") {:reason :busy ::busy true}))

(defn- busy? [t] (::busy (ex-data t)))

(defn- live-entry
  "The live entry of branch b, or a loud error."
  [e ^Branch b what]
  (when-not (identical? e (.-engine b))
    (throw (ex-info (str what ": the branch belongs to another engine") {:reason :engines-differ})))
  (let [books @(books-of e)]
    (or (when-let [entry (get (:live books) (.-id b))]
          (when (contains? (:busy books) (.-id b))
            (throw (busy-error what b)))
          entry)
        (throw (ex-info (str what ": " b " is not live: advance, extend or resample consumed it (use the branch"
                             " it returned), or it was disposed")
                        {:reason :not-live :branch b})))))

(defn- check-room
  "n more tokens fit in a branch at position pos?"
  [e pos n what]
  (let [n-ctx (long (:n-ctx (.-info ^SlotEngine e)))]
    (when (< n-ctx (+ (long pos) (long n)))
      (throw (ex-info (str what ": " (+ (long pos) (long n)) " tokens exceed the engine's context of " n-ctx
                           " tokens per branch")
                      {:reason :context-full :position pos :tokens n :n-ctx n-ctx})))))

(defn- lose-slots!
  "Free slots, which a failed read left unrestorable, and the live
   branches that held them."
  [e slots]
  (doseq [slot slots]
    (when-let [id (slot-holder @(books-of e) slot)]
      (forget! e id))
    (free-slot! e slot)))

(defn- lose!
  "After a run or step that left slots it could not restore: free them, and
   the live branches that held them."
  [e ex]
  (lose-slots! e (:lost-slots (ex-data ex))))

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

;; advance, extend and open are each a request (…-request: the checks, in
;; the operation's order, then {:reads :busy :finish :fail}) and its read:
;; at once by slots-step or slots-run (…*), or by the scheduler's forwards
;; on a Batching engine. :reads are slots-forward's reads, :busy the ids of
;; the branches it holds, (finish results) its value from the results, by
;; read, {:logprobs :scores}, and (fail lost) undoes it after a failed
;; read, lost the set of slots freed.

(defn- open-request [e tokens]
  (begin! e "open")
  (check-room e 0 (count tokens) "open")
  (ensure-free! e 1 "open")
  (let [[slot] (take-slots! e 1)]
    {:reads [{:slot slot :pos 0 :tokens tokens :targets nil}]
     :busy []
     :finish (fn [[{:keys [logprobs]}]] (make-branch! e tokens logprobs slot))
     :fail (fn [lost] (when-not (contains? lost slot) (free-slot! e slot)))}))

(defn- open* [e tokens]
  (let [{[{:keys [slot]}] :reads :keys [finish]} (open-request e tokens)
        r (try
            (retrying e #(slots-run (impl e) slot 0 tokens nil))
            (catch Throwable t
              (free-slot! e slot)
              (throw t)))]
    (finish [r])))

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

(defn- advance-request [e bs ts]
  (begin! e "advance")
  (let [slots (mapv #(:slot (live-entry e % "advance")) bs)
        positions (mapv br/position bs)]
    (doseq [p positions] (check-room e p 1 "advance"))
    {:reads (mapv (fn [s p t] {:slot s :pos p :tokens [t] :targets nil}) slots positions ts)
     :busy (mapv #(.-id ^Branch %) bs)
     :finish (fn [results]
               (run! #(consume! e %) bs)
               (mapv (fn [^Branch b t {:keys [logprobs]} slot] (make-branch! e (conj (.-tokens b) t) logprobs slot))
                     bs ts results slots))}))

(defn- advance* [e bs ts]
  (let [{:keys [reads finish]} (advance-request e bs ts)
        rows (try
               (retrying e #(slots-step (impl e) (mapv :slot reads) (mapv :pos reads) ts))
               (catch Throwable t
                 (lose! e t)
                 (throw t)))]
    (finish (mapv (fn [row] {:logprobs row}) rows))))

(defn- extend-request [e ^Branch b ts]
  (begin! e "extend")
  (let [{:keys [slot]} (live-entry e b "extend")
        pos (br/position b)]
    (check-room e pos (count ts) "extend")
    {:reads [{:slot slot :pos pos :tokens ts :targets (subvec ts 1)}]
     :busy [(.-id b)]
     :finish (fn [[{:keys [logprobs scores]}]]
               (consume! e b)
               {:branch (make-branch! e (into (.-tokens b) ts) logprobs slot)
                :scores (into [(br/logprob b (first ts))] scores)})}))

(defn- extend* [e b ts]
  (let [{[{:keys [slot pos targets]}] :reads :keys [finish]} (extend-request e b ts)
        r (try
            (retrying e #(slots-run (impl e) slot pos ts targets))
            (catch Throwable t
              (lose! e t)
              (throw t)))]
    (finish [r])))

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
;; The scheduler (Batching engines; genclj-b3ow)
;; ---------------------------------------------------------------------------

;; A request in flight is its request map with :fut, :thread, :engine and
;; its reads' progress: :done (tokens read), :scores and :logprobs, by
;; read. The loop's state ls (the engine thread's alone):
;;   :inflight  requests in flight, in the order they were admitted
;;   :parked    tasks that found a branch busy, run again when a request
;;              ends
;;   :expect    the threads whose requests the last forward finished
;;   :deadline  until when (System/nanoTime) a forward waits for them

(defn- tally! [e k n] (vswap! (books-of e) update-in [:schedule k] (fnil + 0) n))

(defn- started
  "Request r (its request map) in flight from task."
  [r {:keys [fut thread engine]}]
  (let [n (count (:reads r))]
    (assoc r :fut fut :thread thread :engine engine
           :done (vec (repeat n 0)) :scores (mapv #(when (:targets %) []) (:reads r)) :logprobs (vec (repeat n nil)))))

(defn- complete? [r i] (= (long ((:done r) i)) (count (:tokens ((:reads r) i)))))

(defn- finished? [r] (every? #(complete? r %) (range (count (:reads r)))))

(defn- pending
  "The reads of the requests rs left to read, as slots-forward takes them,
   and where each is: [reads refs], refs [request read] index pairs."
  [rs]
  (let [at (for [[j r] (map-indexed vector rs), i (range (count (:reads r))) :when (not (complete? r i))] [j i])]
    [(mapv (fn [[j i]]
             (let [r (rs j)
                   {:keys [slot pos tokens targets]} ((:reads r) i)
                   d (long ((:done r) i))]
               {:slot slot :pos (+ (long pos) d) :tokens (subvec tokens d)
                :targets (when targets (subvec targets (min d (count targets))))}))
           at)
     (vec at)]))

(defn- progressed
  "rs after a forward's results (by read, refs' order)."
  [rs refs results]
  (reduce (fn [rs [[j i] {:keys [read scores logprobs]}]]
            (let [r (rs j)
                  left (- (count (:tokens ((:reads r) i))) (long ((:done r) i)))]
              (when-not (and (int? read) (<= 0 (long read) left) (or (< (long read) left) logprobs))
                (throw (ex-info (str "slots-forward: a read of " left " tokens reported " (pr-str read)
                                     (when-not logprobs " without its log-probabilities")
                                     " (an engine bug)")
                                {:reason :internal :read read :left left})))
              (assoc rs j (-> r
                              (update-in [:done i] + (long read))
                              (update-in [:scores i] #(when % (into % scores)))
                              (assoc-in [:logprobs i] logprobs)))))
          rs (map vector refs results)))

(defn- unbusy! [e r] (vswap! (books-of e) update :busy #(reduce disj % (:busy r))))

(defn- results-of [r]
  (mapv (fn [lp sc] {:logprobs lp :scores sc}) (:logprobs r) (:scores r)))

(defn- forward!
  "One forward of the requests rs in flight (in order), by e's
   slots-forward: [rs' done], rs' the requests it has not finished, with
   their progress, done those it has. If it fails, the slots it names lost
   and those of reads begun in an earlier forward (which no longer end
   where their branches do) are freed with their branches, every request
   of rs is undone (:fail, its branches no longer busy), and the error is
   thrown."
  [e rs]
  (let [[reads refs] (pending rs)]
    (when (empty? reads)
      (throw (ex-info "forward: nothing to read (a scheduler bug)" {:reason :internal})))
    (let [results (try
                    (let [rows (retrying e #(slots-forward (impl e) reads))]
                      (when-not (= (count rows) (count reads))
                        (throw (ex-info (str "slots-forward answered " (count rows) " of " (count reads) " reads (an engine bug)")
                                        {:reason :internal})))
                      (when-not (some #(pos? (long (:read %))) rows)
                        (throw (ex-info "slots-forward read nothing (an engine bug)" {:reason :internal})))
                      rows)
                    (catch Throwable t
                      (let [begun (keep (fn [[j i]] (when (pos? (long (get-in rs [j :done i]))) (get-in rs [j :reads i :slot])))
                                        refs)
                            lost (into (set (:lost-slots (ex-data t))) begun)]
                        (throw (br/clean-up t (concat [#(lose-slots! e (sort lost))]
                                                      (for [r rs] #(do (unbusy! e r) (when-let [f (:fail r)] (f lost))))))))))
          rs (progressed rs refs results)]
      (tally! e :forwards 1)
      [(filterv (complement finished?) rs) (filterv finished? rs)])))

(defn- answer!
  "Request r, finished: its value made (finish, its branches no longer
   busy) and its caller answered."
  [r]
  (let [^CompletableFuture fut (:fut r)]
    (unbusy! (:engine r) r)
    (try (.complete fut ((:finish r) (results-of r)))
         (catch Throwable t (.completeExceptionally fut t)))))

(defn- alone!
  "Request r (a request map) read by forwards of its own: on the engine's
   thread, called from there (an operation's own call)."
  [e r]
  (loop [rs [(started r {:engine e})]]
    (let [[rs done] (forward! e rs)]
      (if (seq done)
        ((:finish (first done)) (results-of (first done)))
        (recur rs)))))

(defn- run-task!
  "Run or admit task; one that finds a branch busy is parked."
  [ls task]
  (let [^CompletableFuture fut (:fut task)]
    (try
      (if-let [admit (:admit task)]
        (let [r (admit)
              e (:engine task)]
          (vswap! (books-of e) update :busy into (:busy r))
          (tally! e :requests 1)
          (vswap! ls update :inflight conj (started r task)))
        (.complete fut ((:run task))))
      (catch Throwable t
        (if (busy? t)
          (vswap! ls update :parked conj task)
          (.completeExceptionally fut t))))))

(defn- run-parked!
  "The parked tasks run again (in order), after a request ended."
  [ls]
  (let [ps (:parked @ls)]
    (vswap! ls assoc :parked [])
    (run! #(run-task! ls %) ps)))

(defn- fail-inflight!
  "Every request in flight (and parked task) failed with t: none of their
   branches busy any more."
  [ls t]
  (let [{:keys [inflight parked]} @ls]
    (vswap! ls assoc :inflight [] :parked [])
    (doseq [r inflight]
      (unbusy! (:engine r) r)
      (.completeExceptionally ^CompletableFuture (:fut r) t))
    (doseq [task parked]
      (.completeExceptionally ^CompletableFuture (:fut task) t))))

(defn- missing?
  "Is a thread the last forward answered not yet back with a request?"
  [{:keys [inflight parked expect]}]
  (let [present (into #{} (map :thread) (concat inflight parked))]
    (boolean (some #(not (contains? present %)) expect))))

(defn- take-queued!
  "Every task queued, run or admitted: false when the stop came (the
   requests in flight then fail, the engine closing)."
  [ls ^LinkedBlockingQueue queue]
  (loop []
    (let [task (.poll queue)]
      (cond (nil? task) true
            (= ::stop task) (do (fail-inflight! ls (closed-error "forward")) false)
            :else (do (run-task! ls task) (recur))))))

(defn- gather!
  "Before a forward: wait for the threads the last forward answered to
   come back with a request (missing?), running what arrives, until the
   deadline. false when the stop came."
  [ls ^LinkedBlockingQueue queue]
  (let [t0 (System/nanoTime)
        e (:engine (first (:inflight @ls)))]
    (loop [waited? false]
      (let [left (- (long (:deadline @ls)) (System/nanoTime))]
        (if (and (missing? @ls) (pos? left))
          (let [task (.poll queue left TimeUnit/NANOSECONDS)]
            (cond (nil? task) (recur true)
                  (= ::stop task) (do (fail-inflight! ls (closed-error "forward")) false)
                  :else (do (run-task! ls task) (recur true))))
          (do (when waited?
                (tally! e :waits 1)
                (tally! e :wait-ns (- (System/nanoTime) t0)))
              true))))))

(defn- schedule!
  "Forwards while requests are in flight, the tasks queued taken between
   them: false when the stop came, true when nothing is in flight."
  [ls ^LinkedBlockingQueue queue]
  (loop []
    (cond
      (empty? (:inflight @ls)) true
      (not (and (take-queued! ls queue) (gather! ls queue))) false
      (empty? (:inflight @ls)) true
      :else
      (let [e (:engine (first (:inflight @ls)))
            t0 (System/nanoTime)]
        (if (:closed? @(books-of e))
          (fail-inflight! ls (closed-error "forward"))
          (try
            (let [[rs done] (forward! e (:inflight @ls))
                  t1 (System/nanoTime)
                  wait (double (:wait (:batching (.-info ^SlotEngine e)) 0.0))]
              (vswap! ls assoc :inflight rs
                      :expect (into #{} (map :thread) done)
                      :deadline (+ t1 (long (* wait (- t1 t0)))))
              (run! answer! done)
              (when (seq done) (run-parked! ls)))
            (catch Throwable t
              ;; forward! undid the requests (or the bug was the
              ;; scheduler's own): their callers get the error
              (vswap! ls assoc :expect #{})
              (fail-inflight! ls t)
              (run-parked! ls))))
        (recur)))))

(defn- operate
  "An advance, extend or open (admit makes its request, run its whole
   operation): on a Batching engine a request, admitted on the engine's
   thread and read by the scheduler; on any other run there, as every
   operation."
  [^SlotEngine e what admit run]
  (let [^Worker w (.-worker e)]
    (cond (not (:batching (.-info e))) (run-on w run what)
          (identical? (Thread/currentThread) (.-thread w)) (alone! e (admit))
          :else (await-value (enqueue! w {:admit admit :fut (CompletableFuture.) :thread (Thread/currentThread) :engine e}
                                       what)))))

;; ---------------------------------------------------------------------------
;; Construction
;; ---------------------------------------------------------------------------

(defonce ^:private cleaner (Cleaner/create))

(defn- closer
  "The action that closes an engine, on its thread, the first time it runs
   (a Cleanable runs it at most once): close calls it, and so does the
   Cleaner for an engine no one can reach any more. It holds what it needs,
   never the engine."
  [^Worker w impl state]
  (fn []
    (try
      (run-on w
              #(when-not (:closed? @state)
                 (vswap! state closed-books)
                 (slots-close impl))
              "close")
      (finally (shutdown! w)))))

(defn engine
  "A Branching engine (java.io.Closeable) over the Slots made by (make-impl),
   which runs on the engine's new thread, named name. Its last slot is
   the scratch slot; the others hold branches. With :batching in its
   slots-info, its Slots implement Batching (continuous batching: see the
   namespace's docstring)."
  ^java.io.Closeable [make-impl name]
  (let [w (worker name)
        [impl info] (try
                      (await-value (enqueue! w {:run #(let [s (make-impl)] [s (slots-info s)]) :fut (CompletableFuture.)}
                                             "engine"))
                      (catch Throwable t
                        (shutdown! w)
                        (throw t)))
        n-slots (long (:n-slots info))
        refuse (fn [msg data]
                 (run-on w #(slots-close impl) "engine")
                 (shutdown! w)
                 (throw (ex-info msg data)))]
    (when (< n-slots 2)
      (refuse "An engine needs at least two slots (one for scoring)" {:reason :invalid-option :n-slots n-slots}))
    (when-let [{:keys [wait]} (:batching info)]
      (when-not (satisfies? Batching impl)
        (refuse "Slots with :batching must implement Batching" {:reason :invalid-option :batching (:batching info)}))
      (when-not (and (number? wait) (<= 0.0 (double wait)))
        (refuse (str ":batching's :wait is a fraction of a forward, not " (pr-str wait))
                {:reason :invalid-option :batching (:batching info)})))
    (let [state (volatile! {:live {} :refs {} :free (into (sorted-set) (range (dec n-slots))) :busy #{}
                            :next-id 0 :collections 0 :closed? false
                            :schedule {:requests 0 :forwards 0 :waits 0 :wait-ns 0}})
          e (SlotEngine. impl
                         (assoc info :max-branches (dec n-slots) :scratch-slot (dec n-slots))
                         w (ReferenceQueue.) state)]
      (vswap! state assoc :cleanable (.register ^Cleaner cleaner e (closer w impl state)))
      e)))

(defn call
  "(f impl) on the engine's thread, for engine-specific operations
   (tokenizing, say), impl its Slots: its value, or what it threw. A loud
   error once the engine is closed."
  [^SlotEngine e f]
  (run-on (.-worker e)
          #(do (check-open e "call") (f (.-impl e)))
          "call"))

(defn slot-of
  "Diagnostics: the slot holding branch b's state, or nil when b is not
   live."
  [^SlotEngine e ^Branch b]
  (run-on (.-worker e) #(:slot (get (:live @(books-of e)) (.-id b))) "slot-of"))

(defn schedule
  "Diagnostics: the scheduler's counts since the engine was made,
   {:requests (advances, extends and opens admitted as requests) :forwards
   (slots-forward calls) :waits (forwards that waited for callers) :wait-ns
   (their time waiting)}, all 0 on an engine without Batching, which runs
   its operations one at a time and never waits."
  [^SlotEngine e]
  (:schedule @(books-of e)))
