(ns gen.llm.array-engine
  "A Branching engine (gen.llm.branching) over gen.array: a model's forward
   passes (gen.llm.models.qwen3, or gen.llm.models.qwen35 for Qwen3.5 and
   Qwen3.6, dense or MoE) on any backend, MLX's Metal and CPU or the JVM's
   for tiny models, as gen.llm.slots's Slots
   (docs/research/llm-mlx-engine.md §3, P3 of §6).

     (with-open [v (llama/vocabulary \"Qwen3-0.6B-Q8_0.gguf\")
                 e (array-engine/engine {:model \"Qwen3-0.6B-Q8_0.gguf\"})
                 m (lm/lm e (llama/tokenizer v))]
       …)

   The model is loaded on gen.array's *backend* when the engine is made
   (or given loaded), and every operation runs on the engine's thread on
   that backend. The tokenizer is not the engine's: llama.cpp's vocabulary
   of the same GGUF, loaded alone (gen.llm.llama/vocabulary), or any
   tokenizer map of the model's vocabulary.

   A branch's state is a row of a static cache (the family's
   static-cache: qwen3's KV cache, or qwen35's, whose GatedDeltaNet
   layers hold a recurrent state a row), a value: a slot holds a group, a
   static cache of one or more rows, and its row there. The family is the
   GGUF's architecture (qwen3, qwen35, qwen35moe), or a loaded model's
   (its config's :arch, qwen3 without one).
   - A copy shares the group (a fork is free), and a clear drops the
     slot's hold on it; a group no slot holds is garbage, freed by the GC
     (the backend's arrays, by their phantom references).
   - A run (open, extend, score's scratch read) reads its tokens into a
     cache of the slot's row alone, in forwards of :n-batch tokens, its
     row copied out of its group first unless the slot holds the group
     alone. With targets, each one's score is read on the device: the
     log-softmax at its position, in the model's dtype, taken at the
     target.
   - A step is one decode of the slots' rows, by the family's compiled step
     over static caches, consuming its cache (compile-step's :donate?): a
     group whose rows are exactly the step's slots, held by them alone (a
     lane group: lockstep particles), is stepped in place, its rows in
     its own order; any other step stacks its rows into a new cache first
     (select-rows of each group, concat-rows), one copy, and steps
     that. The stepped rows form a new group.
   - Capacity grows by blocks (:block positions, at most :n-ctx), with
     zeros (grow): a compiled step traces once per batch size and
     capacity.
   - Logits stay on the device until the step's or run's end, where one
     evaluation reads them (array/->floats): each row is normalized on the
     host in double (log-softmax, loud when a logit is not finite), on a
     small pool of the engine's own and the engine's thread
     (gen.llm.host/log-softmax-rows: the same floats for any number of
     threads), as the llama.cpp engine normalizes its rows.
   - Memory is the only bound on branches (no sequence cap); the arrays
     count against the backend's own accounting (on MLX, its active
     memory, inside the budget gen.array.memory shares with native
     engines). A read or step past MLX's budget is loud (its
     :mlx-budget-exceeded, marked :exhausted? so that the engine collects
     dropped branches and tries once more); a lane group stepped in place
     is consumed by the attempt, so its slots are then lost (:lost-slots),
     freed by the engine with their branches.

   An LM over it (gen.llm.lm) bounds its branches' KV cells as llama.cpp's
   unified cache counts them (:max-cells, by default the engine's
   :n-ctx), which bound nothing here: give it :max-cells for the
   contexts it should hold. What the LM keeps costs this engine memory,
   each step's group a copy of its rows (2 × layers × KV-heads ×
   head-dim floats a position of their capacity, and a Qwen3.5's
   recurrent states), so keep its :max-branches to what a run needs.

   The paged mode (:paged? true, Qwen3 and Qwen3.5; genclj-1daf,
   genclj-6qb6) keeps every branch in one pool of blocks instead
   (gen.llm.models.qwen3/paged-pools, gen.array's paged operations), its
   books values on the host (gen.llm.paged): a slot holds a block table.
   Qwen3.5's attention layers are paged; its GatedDeltaNet layers' states
   are a row each, in groups as the static mode's: a fork shares its
   row, a step of a group's rows (a lane) steps it in place, and any
   other step stacks its rows' states first (a copy, :stacked-steps).
   - A copy (fork) shares the table and copies nothing; a step that writes
     a block shared and filled in part copies that block first when
     another holder wrote there (copy-on-write, below); full blocks stay
     shared.
   - A step is one compiled decode of the slots' rows by paged attention
     over their tables (qwen3/compile-paged-step, consuming the pools):
     no row is stacked, whatever groups the slots came from, so token
     SMC's resampled particles step without a copy. Rows are padded to a
     power of 2 (at least :min-rows) and tables to a power of 2 of
     blocks, so a step traces once per such shape; padding rows write
     the pool's scratch blocks, and no row reads another's.
   - A run reads its tokens after the slot's positions gathered from its
     blocks (a concatenating cache, qwen3/paged-cache), eagerly, then
     writes their keys and values into new blocks (compiled, consuming
     the pools).
   - The pool (:pool-blocks, of :block-size positions) doubles when its
     free blocks run out (a copy, and new traces). A donated write's
     cost does not grow with the pool (measured on Metal, 2026-11-08:
     0.3-0.7 ms into one pool pair of 0.14 to 3 GB; genclj-1ovd).
   - A shared block filled in part is written in place by the first of
     its holders to go on (gen.llm.paged/prepare: no holder has written
     past its end), as an LM's entry and the fork that goes on from it
     share it, and copied only for a second writer (copy-on-write: token
     SMC's duplicated particles).
   - A copy-on-write is written within the step (or a run's write) that
     needs it, before its keys and values (qwen3/paged-step's :copy). Its
     blocks are gathered and evaluated first, a call of their own: a
     read of the pools' blocks in the call that writes them makes MLX
     copy each pool, the read still holding it.
   - A paged step, run or copy that fails has consumed the pools: every
     branch is lost (:lost-slots), loudly, and the pools are made again.

   The continuous mode (:paged? and :continuous? true; genclj-b3ow)
   batches concurrent callers' reads (gen.llm.slots's Batching: its
   scheduler gathers every advance, extend and open in flight between
   forwards, see that namespace). A forward is one compiled call
   (qwen3/compile-paged-batch-step, consuming the pools) of rows of one
   token:
   - every one-token read in flight (an advance's rows, a one-token open
     or extend, a long read's last token): decode rows, attending by
     paged attention over their tables, padded to a power of 2 (at least
     :min-rows);
   - then a chunk of the oldest longer read (open, extend), as many of
     its tokens as the budget (:n-batch tokens a forward) leaves after
     the decode rows, at least a block: its rows written into its blocks
     and attending over them gathered (sdpa, each row to its position),
     padded to a power of 2 within the budget. Longer reads wait their
     turn, first come first served; their chunks go on in the next
     forwards, between which other requests come and go.
   The logits are computed for the rows read only (each decode row, a
   read's last token, the scored rows of an extend), the scores on the
   device. A read alone (a score's, or a program's from one thread) is
   forwards of its own, so its results repeat bit for bit; a read
   batched with others' is the same within float round-off (kernels
   differ by shape).

   Results depend on batch shapes and capacities as llama.cpp's do (float
   round-off: MLX's kernels differ by shape); the same read from the same
   state gives the same rows, bit for bit, and what is stacked or reused is
   a function of the engine's operations alone. On the JVM backend (float64)
   a context's rows are the same however it is read, but for rounding to
   float.

   Experimental: outside the 1.x promise (DESIGN §9)."
  {:api :experimental}
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [clojure.core :as c]
            [gen.array :as array :refer [-]]
            [gen.llm.host :as host]
            [gen.array.gguf :as gguf]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.llm.paged :as paged]
            [gen.llm.slots :as slots])
  (:import [java.util.concurrent ExecutorService]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; Rows on the host
;; ---------------------------------------------------------------------------

(defn- host-rows
  "The rows of logits [B 1 vocab] (or [B vocab]), read at once and
   normalized on pool and the calling thread (gen.llm.host's
   log-softmax-rows): a vector of B float arrays."
  [pool logits what]
  (host/log-softmax-rows pool (array/->floats logits) (peek (array/shape logits)) what))

;; ---------------------------------------------------------------------------
;; The model families
;; ---------------------------------------------------------------------------

(def ^:private families
  "The functions of each model family the engine runs (their vars), by a
   GGUF's architecture."
  (let [qwen3 {:load #'qwen3/load :static-cache #'qwen3/static-cache :step #'qwen3/step :compile-step #'qwen3/compile-step
               :offsets #'qwen3/offsets :select-rows #'qwen3/select-rows :concat-rows #'qwen3/concat-rows :grow #'qwen3/grow
               ;; the paged mode's
               :paged-pools #'qwen3/paged-pools :paged-cache #'qwen3/paged-cache :paged-entries #'qwen3/paged-entries
               :paged-blocks #'qwen3/paged-blocks :paged-write-entries #'qwen3/paged-write-entries
               :compile-paged-step #'qwen3/compile-paged-step :compile-paged-batch-step #'qwen3/compile-paged-batch-step}
        qwen35 {:load #'qwen35/load :static-cache #'qwen35/static-cache :step #'qwen35/step :compile-step #'qwen35/compile-step
                :offsets #'qwen35/offsets :select-rows #'qwen35/select-rows :concat-rows #'qwen35/concat-rows :grow #'qwen35/grow
                ;; the paged mode's: the attention layers paged, the
                ;; GatedDeltaNet layers' states a row each (:paged-states)
                :paged-pools #'qwen35/paged-pools :paged-cache #'qwen35/paged-cache :paged-entries #'qwen35/paged-entries
                :paged-blocks #'qwen35/paged-blocks :paged-write-entries #'qwen35/paged-write-entries
                :compile-paged-step #'qwen35/compile-paged-step :paged-states #'qwen35/paged-states}]
    {"qwen3" qwen3 "qwen35" qwen35 "qwen35moe" qwen35}))

(defn- family
  "The family of the architecture arch (a string), loud for another."
  [arch]
  (or (families arch)
      (throw (ex-info (str "array-engine/engine: no forward for the architecture " (pr-str arch)
                           "; it runs " (vec (sort (keys families))))
                      {:reason :unsupported :architecture arch}))))

(defn- arch-of
  "The architecture of model: a GGUF file's (its header), or a loaded
   model's config's :arch (qwen3's has none)."
  [model]
  (if (string? model)
    (get-in (gguf/header model) [:metadata "general.architecture"])
    (get-in model [:config :arch] "qwen3")))

;; ---------------------------------------------------------------------------
;; The books: a slot's {:group g :row r}, a group {:id :cache}
;; ---------------------------------------------------------------------------

;; state, a volatile written on the engine's thread:
;;   :table   a vector, by slot, of nil (empty) or {:group {:id :cache} :row r}
;;   :next    the next group's id
;;   :stats   {:runs :steps :lane-steps :stacked-steps :stacked-rows :grows}
;;   :family  the model family's functions (families), set once

(defn- call
  "The model family's function f (a key of families) on args."
  [state f & args]
  (apply (get-in @state [:family f]) args))

(defn- entry [state slot] (nth (:table @state) slot))

(defn- holders
  "The slots holding group id."
  [state id]
  (into #{} (keep-indexed (fn [slot e] (when (and e (c/== (long id) (long (:id (:group e))))) slot)))
        (:table @state)))

(defn- tally! [state k n] (vswap! state update-in [:stats k] c/+ n))

(defn- new-group!
  "A new group of cache, its id never used before. (Not inside a vswap! of
   state: the id it takes would be lost.)"
  [state cache]
  (let [id (:next @state)]
    (vswap! state update :next inc)
    {:id id :cache cache}))

(defn- position-of
  "The positions slot's row holds."
  ^long [state slot]
  (if-let [{:keys [group row]} (entry state slot)]
    (long (nth (call state :offsets (:cache group)) row))
    0))

(defn- check-position
  "The Slots contract's invariant: slot holds pos positions (a loud
   internal error otherwise, as a bug of this engine or gen.llm.slots)."
  [state slot pos what]
  (let [have (position-of state slot)]
    (when-not (c/== have (long pos))
      (throw (ex-info (str what ": slot " slot " holds " have " positions, not " pos " (an engine bug)")
                      {:reason :internal :slot slot :position pos :holds have})))))

(defn- exhausted
  "e with :exhausted? when it is MLX's budget refusing an evaluation (so
   that gen.llm.slots collects dropped branches and tries once more), and
   with :lost-slots when lost (slots whose state the attempt consumed)."
  [^Throwable e lost]
  (let [data (ex-data e)
        budget? (= :mlx-budget-exceeded (:reason data))]
    (if (or budget? (seq lost))
      (ex-info (ex-message e)
               (cond-> (or data {})
                 budget? (assoc :exhausted? true)
                 (seq lost) (assoc :lost-slots (vec lost)))
               e)
      e)))

;; ---------------------------------------------------------------------------
;; The Slots
;; ---------------------------------------------------------------------------

(defn- capacity-for
  "The capacity of a row of n positions: n rounded up to a block, at most
   n-ctx."
  ^long [^long n ^long block ^long n-ctx]
  (min n-ctx (c/* block (quot (c/+ n (dec block)) block))))

(defn- row-cache
  "A static cache of slot's row alone that the caller may step: its
   group's when the slot holds the group's one row alone, else the row
   copied out (select-rows)."
  [state slot]
  (let [{:keys [group row]} (entry state slot)
        cache (:cache group)]
    (if (and (c/== 1 (count (call state :offsets cache))) (= #{slot} (holders state (:id group))))
      cache
      (call state :select-rows cache [row]))))

(defn- run-scores
  "The log-probabilities of targets (a vector of s tokens) at the first s
   positions of logits [1 T vocab], on the device in its dtype: [1 s 1]."
  [logits targets]
  (let [s (count targets)
        x (array/slice logits 1 0 s)
        idx (array/reshape (array/array targets {:dtype :int32}) [1 s 1])]
    (- (array/take-along-axis x idx 2) (array/logsumexp x {:axis -1 :keep-dims? true}))))

(defn- read-run
  "slots-run's body: tokens read into slot's row (holding pos positions) in
   forwards of n-batch tokens, eagerly; each target's score from the
   forward that read its position; the next-token row after the last."
  [{:keys [model state pool n-batch block n-ctx]} slot pos tokens targets]
  (check-position state slot pos "read")
  (let [pos (long pos)
        n (count tokens)
        k (count targets)
        cap (capacity-for (c/+ pos n) block n-ctx)
        start (if (zero? pos)
                (call state :static-cache model 1 cap)
                (let [c (row-cache state slot)]
                  (when (c/< (long (:capacity c)) cap) (tally! state :grows 1))
                  (call state :grow c (max cap (long (:capacity c))))))
        step-size (long n-batch)]
    (loop [from 0 cache start scores []]
      (let [end (min n (c/+ from step-size))
            last? (c/== end n)
            scored (max 0 (c/- (min end k) from))
            [logits cache] (call state :step model cache [(subvec tokens from end)] {:logits (if (pos? scored) :all :last)})
            s (when (pos? scored) (run-scores logits (subvec targets from (c/+ from scored))))
            row (when last? (array/slice logits 1 (dec (long (second (array/shape logits)))) nil))]
        (array/eval! [s row cache])
        (let [scores (if s (into scores (map double) (array/->floats s)) scores)]
          (if last?
            (let [g (new-group! state cache)]
              (vswap! state assoc-in [:table slot] {:group g :row 0})
              (tally! state :runs 1)
              {:logprobs (first (host-rows pool row "read"))
               :scores (when targets scores)})
            (recur end cache scores)))))))

(defn- lane?
  "Is the group id, whose rows (all n of them) the step's slots hold one
   each, held by those slots alone, so that the step may consume it?"
  [state id step-slots]
  (= (holders state id) (set step-slots)))

(defn- plan
  "How a step of slots stacks its rows: [parts order], parts a vector of
   {:cache :lane?} (lane? when it is a group held by exactly these slots,
   all of its rows, used in place), order the step's index of each row of
   the stacked cache, in order."
  [state slots]
  (let [es (mapv #(entry state %) slots)
        ids (distinct (map (comp :id :group) es))]
    (reduce (fn [[parts order] id]
              (let [is (filterv #(c/== (long id) (long (:id (:group (es %))))) (range (count slots)))
                    {:keys [cache]} (:group (es (first is)))
                    rows (mapv #(:row (es %)) is)
                    n (count (call state :offsets cache))]
                (if (and (c/== n (count is)) (apply distinct? rows) (lane? state id (map slots is)))
                  ;; the group itself: its rows in its order
                  (let [by-row (zipmap rows is)]
                    [(conj parts {:cache cache :lane? true}) (into order (map by-row) (range n))])
                  [(conj parts {:cache (call state :select-rows cache rows) :lane? false}) (into order is)])))
            [[] []] ids)))

(defn- decode-step
  "slots-step's body: one compiled decode of the slots' rows (see the
   namespace's docstring), [rows in the slots' order]."
  [{:keys [state pool block n-ctx step-fn]} slots positions tokens]
  (doseq [[slot pos] (map vector slots positions)] (check-position state slot pos "advance"))
  (let [[parts order] (plan state slots)
        lane (when (and (c/== 1 (count parts)) (:lane? (first parts))) (vec slots))
        need (capacity-for (inc (long (reduce max 0 positions))) block n-ctx)
        cap (long (reduce max need (map #(long (:capacity (:cache %))) parts)))
        grown (mapv (fn [{:keys [cache]}]
                      (when (c/< (long (:capacity cache)) cap) (tally! state :grows 1))
                      (call state :grow cache cap))
                    parts)
        cache (if (c/== 1 (count grown)) (first grown) (call state :concat-rows grown))
        ;; a lane group not grown is the step's cache itself, which the
        ;; step consumes (donates)
        consumed (when (and lane (identical? cache (:cache (first parts)))) lane)
        row-tokens (mapv #(vector (tokens %)) order)]
    (try
      (let [[logits cache'] (step-fn cache row-tokens)
            _ (array/eval! [logits cache'])
            rows (host-rows pool logits "advance")
            g (new-group! state cache')]
        (doseq [[r i] (map-indexed vector order)]
          (vswap! state assoc-in [:table (slots i)] {:group g :row r}))
        (tally! state :steps 1)
        (if lane
          (tally! state :lane-steps 1)
          (do (tally! state :stacked-steps 1) (tally! state :stacked-rows (count slots))))
        (reduce (fn [out [r i]] (assoc out i (rows r))) (vec (repeat (count slots) nil)) (map-indexed vector order)))
      (catch Throwable t
        ;; a lane group was consumed by the step (donated): its slots hold
        ;; nothing any more; a stacked copy's slots kept their groups
        (throw (exhausted t consumed))))))

(deftype ArraySlots [model backend step-fn ^ExecutorService pool info state n-batch block n-ctx]
  slots/Slots
  (slots-info [_] info)

  (slots-run [_ slot pos tokens targets]
    (array/with-backend backend
      (try
        (read-run {:model model :state state :pool pool :n-batch n-batch :block block :n-ctx n-ctx}
                  slot pos tokens targets)
        (catch Throwable t (throw (exhausted t nil))))))

  (slots-step [_ slots positions tokens]
    (array/with-backend backend
      (decode-step {:state state :pool pool :block block :n-ctx n-ctx :step-fn step-fn}
                   (vec slots) (vec positions) (vec tokens))))

  (slots-copy [_ src dst]
    (vswap! state assoc-in [:table dst] (entry state src)))

  (slots-clear [_ slot]
    (vswap! state assoc-in [:table slot] nil))

  (slots-close [_]
    (vswap! state assoc :table (vec (repeat (count (:table @state)) nil)))
    (.shutdown pool))

  Object
  (toString [_] (str "#<ArraySlots " (:path (:model info)) ">")))

;; ---------------------------------------------------------------------------
;; The paged mode: one pool of blocks, a slot a block table
;; ---------------------------------------------------------------------------

;; state, a volatile written on the engine's thread:
;;   :table   a vector, by slot, of nil (empty) or a gen.llm.paged sequence
;;   :alloc   the pool's allocator (gen.llm.paged), its first :scratch
;;            blocks reserved for a step's padding rows
;;   :pools   the model's pools (the family's paged-pools), by layer
;;   :rows    (a family with recurrent layers, Qwen3.5) a vector, by slot,
;;            of nil or {:group {:id :states :n} :row r}: its recurrent
;;            layers' states, row r of a group's (:states by layer, n
;;            rows), shared by forks as the static mode's groups are
;;   :next    the next group's id
;;   :stats   {:runs :steps :rows :padded-rows :copied-blocks
;;            :pool-grows (:stacked-steps with :rows)}

(defn- next-pow2 ^long [^long n] (loop [p 1] (if (c/< p n) (recur (c/* 2 p)) p)))

(defn- pools? [entry] (contains? entry :k))

(defn- merged
  "A step's layers: the pools by layer, each merged with the rows'
   states (by layer) when there are any."
  [pools states]
  (if states (mapv merge pools states) pools))

(defn- split-layers
  "[pools states] of a step's layers (states nil without :rows)."
  [state layers]
  (if (contains? @state :rows)
    [(mapv #(select-keys % [:k :v]) layers) (mapv #(select-keys % [:conv :state]) layers)]
    [layers nil]))

(defn- take-rows
  "states (by layer) at their rows idx (host integers): a copy."
  [states idx]
  (let [ix (array/array (vec idx) {:dtype :int32})]
    (mapv (fn [e] (update-vals e #(array/take % ix 0))) states)))

(defn- cat-rows
  "The rows of the states (each by layer) in order: a copy."
  [parts]
  (apply mapv (fn [& es] (into {} (map (fn [k] [k (array/concatenate (mapv k es) 0)])) (keys (first es)))) parts))

(defn- new-states-group!
  "A new group of states (by layer, n rows), its id never used before."
  [state states n]
  (let [id (:next @state)]
    (vswap! state update :next inc)
    {:id id :states states :n (long n)}))

(defn- row-states
  "The states of slot's row alone (by layer, one row): its group's when
   it is one row, else that row taken (a copy)."
  [state slot]
  (let [{:keys [group row]} (nth (:rows @state) slot)]
    (if (c/== 1 (long (:n group))) (:states group) (take-rows (:states group) [row]))))

(defn- step-states
  "[states order lane] of a step of slots padded to r rows: the step's
   row i is (slots (order i)), and states its rows' states (by layer, r
   rows). A group of r rows whose first ones are exactly the slots',
   held by them alone (a lane: the rows a step gave them, its padding
   rows after), is used as it is, in its order (lane: its slots, which
   the step consumes); otherwise the rows are taken from their groups
   and stacked (a copy), as the static mode's steps, zeros after them."
  [state slots r]
  (let [n (count slots)
        r (long r)
        rs (mapv #(nth (:rows @state) %) slots)
        {:keys [id states] :as g} (:group (first rs))
        rows (mapv :row rs)
        holders (into #{} (keep-indexed (fn [s e] (when (and e (c/== (long id) (long (:id (:group e))))) s))) (:rows @state))]
    (if (and (every? #(c/== (long id) (long (:id (:group %)))) rs)
             (c/== r (long (:n g)))
             (= (set rows) (set (range n)))
             (= holders (set slots)))
      [states (mapv (zipmap rows (range n)) (range n)) (vec slots)]
      (let [ids (distinct (map (comp :id :group) rs))
            [parts order] (reduce (fn [[parts order] id]
                                    (let [is (filterv #(c/== (long id) (long (:id (:group (rs %))))) (range n))]
                                      [(conj parts (take-rows (:states (:group (rs (first is)))) (mapv rows is)))
                                       (into order is)]))
                                  [[] []] ids)
            stacked (if (c/== 1 (count parts)) (first parts) (cat-rows parts))]
        [(if (c/== n r)
           stacked
           (mapv (fn [e] (update-vals e #(array/concatenate [% (array/zeros (assoc (array/shape %) 0 (c/- r n)) {:dtype (array/dtype %)})]
                                                            0)))
                 stacked))
         order nil]))))

(defn- paged-position ^long [state slot]
  (long (or (:len (entry state slot)) 0)))

(defn- check-paged-position [state slot pos what]
  (let [have (paged-position state slot)]
    (when-not (c/== have (long pos))
      (throw (ex-info (str what ": slot " slot " holds " have " positions, not " pos " (an engine bug)")
                      {:reason :internal :slot slot :position pos :holds have})))))

(defn- live-slots [state]
  (into [] (keep-indexed (fn [slot e] (when e slot))) (:table @state)))

(defn- lose-all!
  "After a donating call failed: the pools it consumed are gone, and with
   them every slot's contents. The pools made again (empty, the same
   size), every slot emptied: the lost slots, for :lost-slots."
  [{:keys [model state]}]
  (let [lost (live-slots state)
        {:keys [size]} (:alloc @state)]
    (vswap! state (fn [s] (cond-> (assoc s
                                         :table (vec (repeat (count (:table s)) nil))
                                         :alloc (paged/allocator size (:scratch s))
                                         :pools (call state :paged-pools model size (:block-size s)))
                            (contains? s :rows) (assoc :rows (vec (repeat (count (:table s)) nil))))))
    lost))

(defn- donating
  "(f) of a call that consumes the pools: on failure every slot is lost
   (lose-all!), named in the error's :lost-slots."
  [ctx f]
  (try (f)
       (catch Throwable t
         (throw (exhausted t (lose-all! ctx))))))

(defn- prepared
  "[alloc seqs slots copies]: each of seqs (blocks of bs) made ready for
   its count of ns positions more (gen.llm.paged/prepare), in order, the
   pools grown first (doubled, at least to what they need) when their
   free blocks do not suffice."
  [{:keys [model state] :as ctx} seqs ns]
  (let [bs (long (:block-size @state))
        go #(reduce (fn [[a ss slots copies] [s n]]
                      (let [[a s' sl cs] (paged/prepare a s n bs)]
                        [a (conj ss s') (conj slots sl) (into copies cs)]))
                    [% [] [] []] (map vector seqs ns))]
    (try (go (:alloc @state))
         (catch clojure.lang.ExceptionInfo e
           (if-not (= :blocks-exhausted (:reason (ex-data e)))
             (throw e)
             (let [{:keys [size] :as a} (:alloc @state)
                   ;; the most a reading of these could take: each a block
                   ;; copied and its new positions' blocks
                   need (long (reduce c/+ 0 (map (fn [s n] (c/+ 1 (paged/blocks-for (c/+ (rem (long (:len s)) bs) (long n)) bs)))
                                                 seqs ns)))
                   to (max (c/* 2 (long size)) (c/+ (long size) need))
                   grown (call state :paged-pools model (c/- to (long size)) bs)
                   pools (mapv (fn [old new] (if (pools? old)
                                               {:k (array/concatenate [(:k old) (:k new)] 0)
                                                :v (array/concatenate [(:v old) (:v new)] 0)}
                                               old))
                               (:pools @state) grown)]
               (array/eval! pools)
               (vswap! state assoc :alloc (paged/grow a to) :pools pools)
               (tally! state :pool-grows 1)
               (prepared ctx seqs ns)))))))

(defn- i32 [xs] (array/array (vec xs) {:dtype :int32}))

(defn- padded
  "copies padded to a power of 2 of pairs with self-copies of blocks
   whose contents mean nothing (free in allocator a, or the scratch
   blocks), as many as there are, so that a step with copies traces once
   per power of 2."
  [a scratch copies]
  (let [want (next-pow2 (count copies))
        dsts (set (map second copies))]
    (into copies (map (fn [b] [b b])) (take (c/- want (count copies)) (remove dsts (concat (:free a) (range scratch)))))))

(defn- copy-input
  "A step's :copy (qwen3/paged-step's) of the block copies [[from to] …]
   (copy-on-write), a the allocator after them, or nil for none: the
   copies padded to a power of 2 (padded: a padding copy writes a block
   whose contents mean nothing), their blocks gathered from the pools and
   evaluated, a call before the step that writes them. In the call that
   writes the pools, a read of their blocks makes MLX copy each pool
   instead of writing it in place, the read still holding it (measured on
   Metal, 2026-11-07: 14 to 120 ms a copy as the pools grew from 0.6 to
   3.8 GB)."
  [{:keys [state gather-fn]} a copies]
  (when (seq copies)
    (let [all (padded a (:scratch @state) copies)]
      (tally! state :copied-blocks (count copies))
      {:blocks (doto (gather-fn (:pools @state) (i32 (map first all))) array/eval!)
       :dst (i32 (map second all))})))

(defn- paged-read-run
  "slots-run's body in the paged mode: slot's sequence (pos positions)
   read as a concatenating cache (its blocks gathered), tokens read after
   it in forwards of n-batch tokens, eagerly; then their keys and values
   written into the pools at the sequence's new slots, after its copy
   (copy-on-write), in one call."
  [{:keys [model state pool n-batch write-fn] :as ctx} slot pos tokens targets]
  (check-paged-position state slot pos "read")
  (let [s (or (entry state slot) paged/empty-sequence)
        pos (long pos)
        n (count tokens)
        k (count targets)
        bs (long (:block-size @state))
        states? (contains? @state :rows)
        start (when (pos? pos)
                (call state :paged-cache model (merged (:pools @state) (when states? (row-states state slot))) (:blocks s) pos bs))]
    (loop [from 0 cache start scores []]
      (let [end (min n (c/+ from (long n-batch)))
            last? (c/== end n)
            scored (max 0 (c/- (min end k) from))
            [logits cache] (call state :step model cache [(subvec tokens from end)] {:logits (if (pos? scored) :all :last)})
            sc (when (pos? scored) (run-scores logits (subvec targets from (c/+ from scored))))
            row (when last? (array/slice logits 1 (dec (long (second (array/shape logits)))) nil))]
        (array/eval! [sc row cache])
        (let [scores (if sc (into scores (map double) (array/->floats sc)) scores)]
          (if-not last?
            (recur end cache scores)
            (let [[a [s'] [slots] copies] (prepared ctx [s] [n])
                  entries (call state :paged-entries cache pos)
                  copy (copy-input ctx a copies)
                  pools (donating ctx #(doto (write-fn (:pools @state) entries (i32 slots) (if copy {:copy copy} {}))
                                         array/eval!))]
              (when states?
                ;; the run's recurrent states, a group of one row
                (let [g (new-states-group! state (call state :paged-states model cache) 1)]
                  (vswap! state assoc-in [:rows slot] {:group g :row 0})))
              (vswap! state #(-> % (assoc :pools pools :alloc a) (assoc-in [:table slot] s')))
              (tally! state :runs 1)
              {:logprobs (first (host-rows pool row "read"))
               :scores (when targets scores)})))))))

(declare paged-rows!)

(defn- paged-decode-step
  "slots-step's body in the paged mode: each slot's sequence made ready
   for a position, then one compiled decode of the rows, padded to a
   power of 2 rows (at least :min-rows), over block tables of a power of
   2 width: the step's shapes, and so its traces, are few. A block a
   second writer shares is copied first, within the step (copy-on-write:
   its :copy, copy-input's). A family with recurrent layers (:rows) steps
   its rows' states too (step-states), its rows in their order."
  [{:keys [state min-rows] :as ctx} slots positions tokens]
  (doseq [[slot pos] (map vector slots positions)] (check-paged-position state slot pos "advance"))
  (let [n (count slots)
        r (max (long min-rows) (next-pow2 n))]
    (if-not (contains? @state :rows)
      (paged-rows! ctx slots tokens r nil)
      (let [[states order lane] (step-states state slots r)
            ;; a lane's group is consumed by the step (its states donated)
            rows (paged-rows! ctx (mapv slots order) (mapv tokens order) r states)]
        (tally! state :stacked-steps (if lane 0 1))
        (reduce (fn [out [i k]] (assoc out k (rows i))) (vec (repeat n nil)) (map-indexed vector order))))))

(defn- paged-rows!
  "paged-decode-step's step of slots (in the step's row order), padded
   to r rows, with the rows' states (by layer, r rows; nil for a family
   without recurrent layers), which the step consumes: the rows, in
   order."
  [{:keys [state pool step-fn] :as ctx} slots tokens r states]
  (let [n (count slots)
        r (long r)
        bs (long (:block-size @state))
        seqs (mapv #(or (entry state %) paged/empty-sequence) slots)
        [a seqs' write-slots copies] (prepared ctx seqs (repeat n 1))
        copy (copy-input ctx a copies)
        width (next-pow2 (reduce max 1 (map (comp count :blocks) seqs')))
        ;; padding row i writes slot i of the scratch blocks, and reads it
        pad-row (fn [^long i] (vec (cons (quot i bs) (repeat (dec width) 0))))
        tables (i32 (into (mapv #(paged/table-row % width 0) seqs') (map pad-row) (range n r)))
        pos (i32 (into (mapv :len seqs) (repeat (c/- r n) 0)))
        all-slots (into (mapv first write-slots) (range n r))
        sl (i32 all-slots)
        toks (into (mapv #(vector (tokens %)) (range n)) (repeat (c/- r n) [0]))
        [logits layers] (donating ctx #(let [out (step-fn (merged (:pools @state) states) pos tables sl toks {:copy copy})]
                                         (array/eval! out)
                                         out))
        [pools states'] (split-layers state layers)
        g (when states (new-states-group! state states' r))
        rows (host-rows pool (array/slice logits 0 0 n) "advance")]
    (vswap! state (fn [st] (cond-> (-> st
                                       (assoc :pools pools :alloc a)
                                       (update :table #(reduce (fn [t [slot s']] (assoc t slot s')) % (map vector slots seqs'))))
                             g (update :rows #(reduce (fn [t [i slot]] (assoc t slot {:group g :row i})) % (map-indexed vector slots))))))
    (tally! state :steps 1)
    (tally! state :rows n)
    (tally! state :padded-rows (c/- r n))
    rows))

(defn- paged-copy!
  "slots-copy in the paged modes: dst shares src's blocks (and its row
   of states)."
  [state src dst]
  (let [[a s] (paged/fork (:alloc @state) (entry state src))]
    (vswap! state #(cond-> (-> % (assoc :alloc a) (assoc-in [:table dst] s))
                     (contains? % :rows) (assoc-in [:rows dst] (get-in % [:rows src]))))))

(defn- paged-clear!
  "slots-clear in the paged modes: slot's blocks released (and its row
   of states dropped)."
  [state slot]
  (when-let [s (entry state slot)]
    (vswap! state #(cond-> (-> % (assoc :alloc (paged/dispose (:alloc %) s)) (assoc-in [:table slot] nil))
                     (contains? % :rows) (assoc-in [:rows slot] nil)))))

(defn- paged-close!
  "slots-close in the paged modes."
  [state ^ExecutorService pool]
  (vswap! state #(cond-> (assoc % :table (vec (repeat (count (:table %)) nil)) :pools nil)
                   (contains? % :rows) (assoc :rows (vec (repeat (count (:table %)) nil)))))
  (.shutdown pool))

(deftype PagedSlots [model backend ctx ^ExecutorService pool info state]
  slots/Slots
  (slots-info [_] info)

  (slots-run [_ slot pos tokens targets]
    (array/with-backend backend
      (try
        (paged-read-run ctx slot pos tokens targets)
        (catch Throwable t (throw (exhausted t nil))))))

  (slots-step [_ slots positions tokens]
    (array/with-backend backend
      (try
        (paged-decode-step ctx (vec slots) (vec positions) (vec tokens))
        (catch Throwable t (throw (exhausted t nil))))))

  (slots-copy [_ src dst] (paged-copy! state src dst))
  (slots-clear [_ slot] (paged-clear! state slot))
  (slots-close [_] (paged-close! state pool))

  Object
  (toString [_] (str "#<PagedSlots " (:path (:model info)) ">")))

;; ---------------------------------------------------------------------------
;; The continuous mode: reads of many requests in one forward (genclj-b3ow)
;; ---------------------------------------------------------------------------

;; The paged mode's books, with :stats {:forwards :decode-rows :chunk-rows
;; :padded-rows :mixed-forwards :copied-blocks :pool-grows}, and its
;; scratch blocks enough for the padding rows of a forward's decode rows
;; and of its chunk.

(defn- one-token? [r] (c/== 1 (count (:tokens r))))

(defn- forward-plan
  "Which of reads (gen.llm.slots/slots-forward's) a forward reads, and
   its rows: every one-token read (decode rows, in order, padded to a
   power of 2, at least min-rows), and the first longer read's next chunk
   of what the budget (n-batch tokens) leaves, at least a block (bs), its
   rows padded to a power of 2 within the budget. {:decode [indices]
   :d-rows :chunk index-or-nil :c (its tokens) :c-rows}."
  [reads min-rows n-batch bs]
  (let [decode (filterv #(one-token? (reads %)) (range (count reads)))
        d (count decode)
        d-rows (if (pos? d) (max (long min-rows) (next-pow2 d)) 0)
        ci (first (remove #(one-token? (reads %)) (range (count reads))))
        left (max (long bs) (c/- (long n-batch) d-rows))
        c (when ci (min (count (:tokens (reads ci))) left))]
    {:decode decode :d-rows d-rows :chunk ci :c c
     :c-rows (when c (min (next-pow2 c) (max (long c) left)))}))

(defn- batch-forward!
  "slots-forward's body in the continuous mode (see the namespace's
   docstring): the plan's reads made ready (copy-on-write), one compiled
   forward of their rows (qwen3/compile-paged-batch-step, consuming the
   pools), the books written, then the rows read and normalized on the
   host."
  [{:keys [state pool batch-fn min-rows n-batch] :as ctx} reads]
  (let [bs (long (:block-size @state))
        {:keys [decode d-rows chunk c c-rows]} (forward-plan reads min-rows n-batch bs)
        chosen (cond-> decode chunk (conj chunk))
        _ (doseq [i chosen] (check-paged-position state (:slot (reads i)) (:pos (reads i)) "forward"))
        seqs (mapv #(or (entry state (:slot (reads %))) paged/empty-sequence) chosen)
        [a seqs' write-slots copies] (prepared ctx seqs (mapv #(if (= % chunk) c 1) chosen))
        copy (copy-input ctx a copies)
        d (count decode)
        d-rows (long d-rows)
        c (long (or c 0))
        c-rows (long (or c-rows 0))
        r (c/+ d-rows c-rows)
        dseqs (subvec seqs' 0 d)
        width (next-pow2 (reduce max 1 (map (comp count :blocks) dseqs)))
        ;; a padding row writes the scratch slot of its row's index, and a
        ;; padding decode row reads that slot's block
        pad-table (fn [^long i] (vec (cons (quot i bs) (repeat (dec width) 0))))
        {cpos :pos ctoks :tokens :as cread} (when chunk (reads chunk))
        cpos (long (or cpos 0))
        tokens (-> []
                   (into (map #(vector (first (:tokens (reads %))))) decode)
                   (into (repeat (c/- d-rows d) [0]))
                   (into (map vector) (when chunk (subvec ctoks 0 c)))
                   (into (repeat (c/- c-rows c) [0])))
        positions (-> []
                      (into (map #(:pos (reads %))) decode)
                      (into (repeat (c/- d-rows d) 0))
                      (into (range cpos (c/+ cpos c-rows))))
        slots (-> []
                  (into (map first) (subvec write-slots 0 d))
                  (into (range d d-rows))
                  (into (when chunk (peek write-slots)))
                  (into (range (c/+ d-rows c) r)))
        ;; the rows read: each decode row, the chunk's last when it ends
        ;; its read, then the chunk's scored rows
        ends? (and chunk (c/== c (count ctoks)))
        lp-rows (cond-> (vec (range d)) ends? (conj (c/+ d-rows (dec c))))
        ;; the chunk's scored rows: those of its targets among its c tokens
        k (if (:targets cread) (min c (count (:targets cread))) 0)
        out (into lp-rows (map #(c/+ d-rows (long %))) (range k))
        o (next-pow2 (count out))
        batch (cond-> {:tokens (i32 tokens) :positions (i32 positions) :slots (i32 slots)
                       :out (i32 (into out (repeat (c/- o (count out)) 0)))}
                copy (assoc :copy copy)
                (pos? d) (assoc :tables (i32 (into (mapv #(paged/table-row % width 0) dseqs) (map pad-table) (range d d-rows))))
                chunk (assoc :chunk (i32 [(paged/table-row (peek seqs') (next-pow2 (count (:blocks (peek seqs')))) 0)]))
                (pos? k) (assoc :targets (i32 (-> (vec (repeat (count lp-rows) 0))
                                                  (into (subvec (:targets cread) 0 k))
                                                  (into (repeat (c/- o (count out)) 0))))))
        {:keys [logits scores pools]} (donating ctx #(let [res (batch-fn (:pools @state) batch)]
                                                       (array/eval! res)
                                                       res))]
    ;; the books first: the pools are the step's now, whatever the reading
    ;; of its rows does
    (vswap! state (fn [st] (-> st
                               (assoc :pools pools :alloc a)
                               (update :table #(reduce (fn [t [i s']] (assoc t (:slot (reads i)) s')) % (map vector chosen seqs'))))))
    (tally! state :forwards 1)
    (tally! state :decode-rows d)
    (tally! state :chunk-rows c)
    (tally! state :padded-rows (c/- r d c))
    (when (and (pos? d) chunk) (tally! state :mixed-forwards 1))
    (let [[rows sc] (try
                      [(when (seq lp-rows)
                         (host-rows pool (if (c/== o (count lp-rows)) logits (array/slice logits 0 0 (count lp-rows))) "advance"))
                       (when (pos? k) (mapv double (array/->floats scores)))]
                      (catch Throwable t
                        ;; their sequences hold the step's positions now
                        (throw (ex-info (ex-message t) (assoc (ex-data t) :lost-slots (mapv #(:slot (reads %)) chosen)) t))))
          by (zipmap decode (range))]
      (mapv (fn [i]
              (let [{:keys [targets]} (reads i)]
                (cond (contains? by i) {:read 1 :scores (when targets []) :logprobs (rows (by i))}
                      (= i chunk) {:read c
                                   :scores (when targets (subvec sc (count lp-rows) (c/+ (count lp-rows) k)))
                                   :logprobs (when ends? (rows (dec (count lp-rows))))}
                      :else {:read 0 :scores (when targets [])})))
            (range (count reads))))))

(defn- continuous-run
  "slots-run's body in the continuous mode: the run's forwards alone. A
   run that fails after its first forward names its slot lost (it holds
   the tokens read so far)."
  [ctx slot pos tokens targets]
  (loop [pos (long pos) tokens tokens targets targets scores (when targets []) begun? false]
    (let [[{:keys [read logprobs] :as res}]
          (try (batch-forward! ctx [{:slot slot :pos pos :tokens tokens :targets targets}])
               (catch Throwable t
                 (throw (if begun?
                          (ex-info (ex-message t) (update (or (ex-data t) {}) :lost-slots #(vec (distinct (conj (vec %) slot)))) t)
                          t))))
          read (long read)
          scores (when scores (into scores (:scores res)))]
      (if (c/== read (count tokens))
        {:logprobs logprobs :scores scores}
        (recur (c/+ pos read) (subvec tokens read) (when targets (subvec targets (min read (count targets)))) scores true)))))

(deftype ContinuousSlots [model backend ctx ^ExecutorService pool info state]
  slots/Slots
  (slots-info [_] info)

  (slots-run [_ slot pos tokens targets]
    (array/with-backend backend
      (try
        (continuous-run ctx slot pos tokens targets)
        (catch Throwable t (throw (exhausted t nil))))))

  (slots-step [_ slots positions tokens]
    (array/with-backend backend
      (try
        (mapv :logprobs (batch-forward! ctx (mapv (fn [s p t] {:slot s :pos p :tokens [t] :targets nil}) slots positions tokens)))
        (catch Throwable t (throw (exhausted t nil))))))

  (slots-copy [_ src dst] (paged-copy! state src dst))
  (slots-clear [_ slot] (paged-clear! state slot))
  (slots-close [_] (paged-close! state pool))

  slots/Batching
  (slots-forward [_ reads]
    (array/with-backend backend
      (try
        (batch-forward! ctx (vec reads))
        (catch Throwable t (throw (exhausted t nil))))))

  Object
  (toString [_] (str "#<ContinuousSlots " (:path (:model info)) ">")))

;; ---------------------------------------------------------------------------
;; Opening
;; ---------------------------------------------------------------------------

(def ^:private defaults {:max-branches 32 :n-ctx 4096 :n-batch 256 :block 128
                         :paged? false :block-size 16 :pool-blocks 128 :min-rows 1
                         :continuous? false :wait 0.25})

(def ^:private option-checks
  {:model #(or (and (string? %) (.isFile (java.io.File. ^String %)))
               (and (map? %) (contains? % :config) (contains? % :weights)))
   :dtype #{:float16 :bfloat16 :float32 :float64}
   :max-branches pos-int? :n-ctx pos-int? :n-batch pos-int? :block pos-int?
   :paged? boolean? :block-size #{8 16 32} :pool-blocks pos-int? :min-rows pos-int?
   :continuous? boolean? :wait #(and (number? %) (c/<= 0.0 (double %) 10.0))})

(defn- check-options
  "opts with their defaults, checked: a loud error for an unknown, missing
   or bad one."
  [opts]
  (when-not (map? opts)
    (throw (ex-info (str "array-engine/engine: options are a map, not " (pr-str opts)) {:reason :invalid-option})))
  (when-let [ks (seq (remove option-checks (keys opts)))]
    (throw (ex-info (str "array-engine/engine: unknown options " (vec ks) "; it takes " (vec (sort (keys option-checks))))
                    {:reason :invalid-option :unknown (vec ks)})))
  (let [{:keys [model] :as opts} (merge defaults opts)]
    (doseq [[k ok?] option-checks
            :when (and (or (= k :model) (contains? opts k)) (not (ok? (get opts k))))]
      (throw (if (= k :model)
               (ex-info (str "array-engine/engine: :model is a GGUF file or a model (qwen3/load's, qwen35/load's), not " (pr-str model))
                        {:reason :no-model :model (if (string? model) model (type model))})
               (ex-info (str "array-engine/engine: bad " k ": " (pr-str (get opts k))) {:reason :invalid-option k (get opts k)}))))
    (when (and (map? model) (contains? opts :dtype))
      (throw (ex-info "array-engine/engine: :dtype is for a GGUF file the engine loads; a model computes in its own"
                      {:reason :invalid-option :dtype (:dtype opts)})))
    (when (and (:continuous? opts) (not (:paged? opts)))
      (throw (ex-info "array-engine/engine: :continuous? batches over the paged mode: give :paged? true too"
                      {:reason :invalid-option :continuous? true})))
    opts))

(defn- model-backend
  "The backend m's weights live on."
  [m]
  (array/backend (first (filter array/backend-array? (array/tree-leaves (:weights m))))))

(defn- open-paged-slots
  "PagedSlots (with :continuous?, ContinuousSlots) for checked opts
   (:paged?), the model m loaded on backend b, its family's functions
   fam: the pools made, the step compiled."
  [{:keys [model max-branches n-ctx n-batch block-size pool-blocks min-rows continuous? wait]} m b fam]
  (when-not (:paged-pools fam)
    (throw (ex-info (str "array-engine/engine: the paged mode runs Qwen3 and Qwen3.5, not " (pr-str (arch-of model)))
                    {:reason :unsupported :architecture (arch-of model)})))
  (when (and continuous? (not (:compile-paged-batch-step fam)))
    (throw (ex-info (str "array-engine/engine: the continuous mode runs Qwen3, not " (pr-str (arch-of model)))
                    {:reason :unsupported :architecture (arch-of model)})))
  (let [{:keys [vocab context] :as cfg} (:config m)
        n-ctx (long (if context (min (long n-ctx) (long context)) n-ctx))
        n-slots (inc (long max-branches))
        ;; scratch blocks: a slot each for the most padding rows a step has
        ;; (in the continuous mode, a forward: its decode rows' and its
        ;; chunk's, which is at most the budget, or a block)
        scratch (paged/blocks-for (c/+ (next-pow2 (max (long min-rows) n-slots))
                                       (if continuous? (max (long n-batch) (long block-size)) 0))
                                  block-size)
        size (c/+ scratch (long pool-blocks))
        step-fn (if continuous?
                  ((:compile-paged-batch-step fam) m {:donate? true})
                  ((:compile-paged-step fam) m {:donate? true}))
        ;; a copy's blocks read alone, before the step that writes them
        gather-fn (array/compile (:paged-blocks fam))
        ;; a run's keys and values written, after its copy
        write-fn (array/compile (:paged-write-entries fam) {:donate #{0}})
        info {:engine :gen.array
              :backend (array/backend-name b)
              :n-vocab vocab
              :n-slots n-slots
              :n-ctx n-ctx
              :n-batch n-batch
              :paged (cond-> {:block-size block-size :pool-blocks pool-blocks :min-rows min-rows}
                       continuous? (assoc :continuous? true))
              :model (merge {:path (when (string? model) model) :arch (arch-of model)}
                            (select-keys cfg [:layers :hidden :heads :kv-heads :head-dim :dtype :context]))}
        state (volatile! (cond-> {:table (vec (repeat n-slots nil)) :family fam :block-size block-size :scratch scratch
                                  :alloc (paged/allocator size scratch) :pools ((:paged-pools fam) m size block-size)
                                  :stats (if continuous?
                                           {:forwards 0 :decode-rows 0 :chunk-rows 0 :padded-rows 0 :mixed-forwards 0
                                            :copied-blocks 0 :pool-grows 0}
                                           (cond-> {:runs 0 :steps 0 :rows 0 :padded-rows 0 :copied-blocks 0
                                                    :pool-grows 0}
                                             (:paged-states fam) (assoc :stacked-steps 0)))}
                           ;; recurrent layers' states, a row each (Qwen3.5)
                           (:paged-states fam) (assoc :rows (vec (repeat n-slots nil)) :next 0)))
        pool (host/pool "gen.llm array rows-" (host/default-threads))
        ctx {:model m :state state :pool pool :n-batch n-batch :min-rows min-rows :gather-fn gather-fn}]
    (if continuous?
      (ContinuousSlots. m b (assoc ctx :batch-fn step-fn) pool (assoc info :batching {:wait wait}) state)
      (PagedSlots. m b (assoc ctx :step-fn step-fn :write-fn write-fn) pool info state))))

(defn- open-slots
  "ArraySlots (or with :paged?, PagedSlots) for checked opts, on the
   engine's thread, on backend b: the model loaded (a path), its step
   compiled."
  [{:keys [model dtype max-branches n-ctx n-batch block paged?] :as opts} b]
  (array/with-backend b
    (if paged?
      (let [fam (family (arch-of model))
            m (if (string? model) ((:load fam) model (when dtype {:dtype dtype})) model)
            mb (model-backend m)]
        (array/with-backend mb (open-paged-slots opts m mb fam)))
    (let [fam (family (arch-of model))
          m (if (string? model) ((:load fam) model (when dtype {:dtype dtype})) model)
          b (model-backend m)
          {:keys [vocab context] :as cfg} (:config m)
          n-ctx (long (if context (min (long n-ctx) (long context)) n-ctx))
          step-fn (array/with-backend b ((:compile-step fam) m {:logits :last :donate? true}))
          n-slots (inc (long max-branches))
          info {:engine :gen.array
                :backend (array/backend-name b)
                :n-vocab vocab
                :n-slots n-slots
                :n-ctx n-ctx
                :n-batch n-batch
                :block block
                :model (merge {:path (when (string? model) model) :arch (arch-of model)}
                              (select-keys cfg [:layers :hidden :heads :kv-heads :head-dim :dtype :context]))}]
      (ArraySlots. m b step-fn (host/pool "gen.llm array rows-" (host/default-threads)) info
                   (volatile! {:table (vec (repeat n-slots nil)) :next 0 :family fam
                               :stats {:runs 0 :steps 0 :lane-steps 0 :stacked-steps 0 :stacked-rows 0 :grows 0}})
                   (long n-batch) (long block) n-ctx)))))

(defn engine
  "A Branching engine over gen.array (java.io.Closeable; close it to free
   its model when it loaded one), on a thread of its own, its model on
   gen.array's *backend* (as it is when engine is called). opts (any other
   is an error):
     :model         a Qwen3, Qwen3.5 or Qwen3.6 GGUF file (architecture
                    qwen3, qwen35 or qwen35moe), which the engine loads
                    (gen.llm.models.qwen3/load, gen.llm.models.qwen35/load),
                    or a model loaded already, a Qwen3 model with a LoRA
                    adapter among them (gen.llm.lora/adapt; its dropout off,
                    without a :key) (required)
     :dtype         with a file: the float dtype it computes in (default
                    the backend's default float); its KV caches hold it too
     :max-branches  live branches at most (default 32): memory is the
                    bound, a branch's row costing 2 × layers × KV-heads ×
                    head-dim floats a position of its capacity (Qwen3-0.6B
                    in float32: 112 KiB; Qwen3.5-0.8B, whose attention
                    layers are a quarter: 24 KiB, and 19 MiB of
                    GatedDeltaNet states whatever the capacity)
     :n-ctx         the most tokens a branch holds (default 4096, at most
                    the model's context)
     :n-batch       tokens a forward of a read (default 256)
     :block         positions a branch's capacity grows by (default 128)
     :paged?        true: the paged mode (see the namespace's docstring;
                    Qwen3 and Qwen3.5), with
       :block-size  positions a block (8, 16 or 32; default 16)
       :pool-blocks the pool's blocks at first (default 128), doubled
                    when they run out
       :min-rows    the fewest rows a step computes (default 1): rows
                    are padded to a power of 2 at least this
       :continuous? true: concurrent callers' reads batched (continuous
                    batching; see the namespace's docstring; Qwen3 only), :n-batch
                    then the tokens a forward reads at most (a long
                    read's chunk), with
         :wait      how long a forward waits for the callers the last
                    one answered, at most, as a fraction of its time
                    (default 0.25; 0 never waits)"
  ^java.io.Closeable [opts]
  (let [opts (check-options opts)
        b array/*backend*]
    (slots/engine #(open-slots opts b) "gen.array")))

(defn stats
  "Diagnostics: the engine's reads and steps since it was made, {:runs
   :steps :lane-steps (a lane group stepped in place) :stacked-steps
   (rows stacked into a new cache) :stacked-rows :grows (capacities
   grown)}; in the paged mode {:runs :steps :rows :padded-rows
   :copied-blocks (copy-on-write) :pool-grows}, and for Qwen3.5
   :stacked-steps (steps whose rows' recurrent states were stacked, a
   copy); in the continuous mode {:forwards :decode-rows :chunk-rows (a
   prefill chunk's) :padded-rows :mixed-forwards (forwards of decode rows
   and a chunk) :copied-blocks :pool-grows}, and gen.llm.slots/schedule
   the scheduler's counts."
  [e]
  (slots/call e (fn [s]
                  (cond (instance? ArraySlots s) (:stats @(.-state ^ArraySlots s))
                        (instance? PagedSlots s) (:stats @(.-state ^PagedSlots s))
                        (instance? ContinuousSlots s) (:stats @(.-state ^ContinuousSlots s))
                        :else (throw (ex-info "Not a gen.array engine" {:reason :not-an-array-engine :engine e}))))))

(defn blocks
  "Diagnostics of the paged mode (:paged?): {:free the pool's free blocks
   :size its blocks :table by slot, nil (empty) or [block-ids length]}:
   which slots share blocks."
  [e]
  (slots/call e (fn [s]
                  (if (or (instance? PagedSlots s) (instance? ContinuousSlots s))
                    (let [{:keys [alloc table]} @(if (instance? PagedSlots s) (.-state ^PagedSlots s) (.-state ^ContinuousSlots s))]
                      {:free (paged/free-count alloc) :size (:size alloc)
                       :table (mapv #(when % [(:blocks %) (:len %)]) table)})
                    (throw (ex-info "Not a paged gen.array engine (:paged?)" {:reason :not-paged :engine e}))))))

(defn groups
  "Diagnostics: by slot, nil (empty) or [group-id row position]: which
   slots share a group's cache, and their rows."
  [e]
  (slots/call e (fn [s]
                  (if (instance? ArraySlots s)
                    (let [st (.-state ^ArraySlots s)]
                      (mapv (fn [slot] (when-let [{:keys [group row]} (entry st slot)]
                                         [(:id group) row (position-of st slot)]))
                            (range (count (:table @st)))))
                    (throw (ex-info "Not a gen.array engine" {:reason :not-an-array-engine :engine e}))))))
