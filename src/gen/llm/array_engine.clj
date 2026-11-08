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
   tokenizer map of the model's vocabulary. The family is the GGUF's
   architecture (qwen3, qwen35, qwen35moe), or a loaded model's (its
   config's :arch, qwen3 without one).

   It runs in one of three modes (engine's options):
   - paged, the default (genclj-17it): every branch's keys and values in
     one pool of blocks, a branch a block table, so forks share their
     blocks and a step stacks no rows (below);
   - continuous (:continuous? true, over the paged mode): concurrent
     callers' reads gathered into shared forwards. Serving several
     clients at once needs it: the paged mode, as the static one, runs
     one caller's read at a time. 8 clients, tokens/s (2026-11-08, Q8_0,
     float16): Qwen3.5-0.8B on Metal, paged 74, continuous 233,
     llama.cpp 69; Qwen3-0.6B on CUDA, paged 219, continuous 555,
     llama.cpp 503. It stays opt-in, since concurrent callers' results
     then depend on how their reads were gathered, within the engine's
     band;
   - static (:paged? false): every branch a row of a static cache, as
     below.
   The paged mode's results equal the static one's within float round-off
   (kernels differ by shape), not bit for bit.

   The static mode. A branch's state is a row of a static cache (the
   family's static-cache: qwen3's KV cache, or qwen35's, whose
   GatedDeltaNet layers hold a recurrent state a row), a value: a slot
   holds a group, a static cache of one or more rows, and its row
   there.
   - A copy shares the group (a fork is free), and a clear drops the
     slot's hold on it; a group no slot holds is released at once
     (array/release!: on MLX its buffers go back for the next step's
     arrays, rather than when the GC finds them).
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
     threads), as the llama.cpp engine normalizes its rows. A run whose
     row fails so leaves its slot as it was: its books are written after.
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

   The paged mode (the default, Qwen3 and Qwen3.5; genclj-1daf,
   genclj-6qb6, genclj-17it) keeps every branch in one pool of blocks
   (gen.llm.models.qwen3/paged-pools, gen.array's paged operations), its
   books values on the host (gen.llm.paged): a slot holds a block table.
   Qwen3.5's attention layers are paged; its GatedDeltaNet layers' states
   are a row each, in groups as the static mode's (a step's rows a new
   group), and a fork shares its row. A step reads its rows' states
   (step-states):
   - a group's rows, all of them, held by the step's slots alone (a
     lane): stepped in place, consumed;
   - a group's consecutive rows that others hold too (forks: an LM's
     entries and the branches it goes on from): read in place, never
     consumed (:held-steps);
   - otherwise stacked into a new array, one copy (:stacked-steps): one
     group's rows taken, or rows of several groups (and zeros for a slot
     with none yet) concatenated.
   Measured on Metal (2026-11-08, genclj-6qb6; bench/gen/
   gdn_states_bench.clj, Qwen3.5-0.8B's states, 16 rows, ms a step with
   the GC): a lane 6.7, read in place 12.5 (the kernel's fresh states
   are new bytes, collected), stacked from one group 17.7, from two
   groups with padding 23.9 (slices, one concatenate; 42 by the three
   copies before); a pool of every slot's states costs more (18-26:
   the kernel cannot
   write a pool in place, so its states are written back, a copy, and
   in one call MLX may copy the whole pool, 32 ms at 8 rows).
   - A copy (fork) shares the table and copies nothing; a step that writes
     a block shared and filled in part copies that block first when
     another holder wrote there (copy-on-write, below); full blocks stay
     shared.
   - A step is one compiled decode of the slots' rows by paged attention
     over their tables (qwen3/compile-paged-step, consuming the pools):
     no row is stacked, whatever groups the slots came from, so token
     SMC's resampled particles step without a copy. Its shape is
     bucketed (Shapes, below): padding rows write the pool's scratch
     blocks, and no row reads another's.
   - A run reads its tokens in forwards of :n-batch tokens, each one
     compiled call of the family's batch step (compile-paged-batch-step,
     consuming the pools) whose rows are a prefill chunk: written into
     the slot's blocks, then attending over them (gen.array/
     paged-attention of a run), its rows bucketed, as the continuous
     mode reads a lone caller's (genclj-6qb6; it was read eagerly over a
     concatenating cache before).
   - The pool (:pool-blocks, of :block-size positions) doubles when its
     free blocks run out (a copy, and new traces). A donated write's
     cost does not grow with the pool (measured on Metal, 2026-11-08:
     0.3-0.7 ms into one pool pair of 0.14 to 3 GB; genclj-1ovd).
   - A shared block filled in part is written in place by the first of
     its holders to go on (gen.llm.paged/prepare: no holder has written
     past its end), as an LM's entry and the fork that goes on from it
     share it, and copied only for a second writer (copy-on-write: token
     SMC's duplicated particles).
   - A copy-on-write is two calls before the step (or a run's write):
     the blocks gathered and evaluated, then written, consuming the pools
     (in one call MLX copies each pool, its read still holding it). In
     the step itself it was no faster: token SMC copies little
     (measured on Metal, 2026-11-08; genclj-6qb6).
   - A paged step, run or copy that fails has consumed the pools: every
     branch is lost (:lost-slots), loudly, and the pools are made again (a
     score's read too: gen.llm.slots frees every branch it names).
   - A step or a run's forward whose rows fail to normalize, loudly,
     leaves its slots as they were: it keeps the pools it wrote (those it
     consumed gone), its books unchanged, what it wrote lying past each
     sequence's end or in blocks the books hold free. A lane's slots,
     whose states the step consumed, are lost (:lost-slots), and so is a
     run's slot when a forward after its first fails (it holds the
     tokens read so far).

   The continuous mode (:continuous? true, over the paged mode;
   genclj-b3ow, for Qwen3.5 genclj-6qb6) batches concurrent callers'
   reads (gen.llm.slots's Batching: its scheduler gathers every advance,
   extend and open in flight between forwards, see that namespace). A
   forward is the compiled calls (the family's compile-paged-batch-step,
   consuming the pools) of its decode rows and of its chunks, evaluated
   together (Shapes, below), of rows of one token:
   - every one-token read in flight (an advance's rows, a one-token open
     or extend, a long read's last token): decode rows, attending by
     paged attention over their tables, their count bucketed;
   - then a chunk of the oldest longer read (open, extend), as many of
     its tokens as the budget (:n-batch tokens a forward) leaves after
     the decode rows, at least a block, padded to t rows (its tokens
     bucketed); and chunks of the next longer
     reads, at most t tokens each, padded to t rows, while the budget
     holds them (at most :max-chunks in all, and a power of 2 of them,
     so that forwards trace for few shapes; genclj-6qb6). Each chunk's
     rows are written into its blocks and attend over them
     (gen.array/paged-attention of a run: its blocks gathered once, each
     row to its own position); a chunk's padding rows sit at its last
     position. Longer reads past the budget wait their turn, first come
     first served; their chunks go on in the next forwards, between
     which other requests come and go.
   Under Qwen3.5 the GatedDeltaNet layers read a forward's decode rows as
   a paged step does (each its own sequence's next token, their states
   read as step-states says: in place, a lane, or stacked) and each
   chunk as one sequence, in order, from the read's states (one chunk:
   zeros at its start, else its row's, read in place; several: their
   rows stacked, a copy), its padding rows past the chunk's tokens
   leaving the states as they were (qwen35/compile-paged-batch-step's
   :chunk-len). Its states after a forward are two new groups: the
   decode rows' and the chunks'.
   The logits are computed for the rows read only (each decode row, a
   read's last token, the scored rows of an extend), the scores on the
   device. A read alone (a score's, or a program's from one thread) is
   forwards of its own, so its results repeat bit for bit; a read
   batched with others' is the same within float round-off (kernels
   differ by shape). A forward fails as a paged step does: one whose call
   fails has consumed the pools, every branch lost (:lost-slots), and one
   whose rows fail to normalize leaves its slots as they were (a read
   begun in an earlier forward is then lost, by gen.llm.slots's rule).

   Shapes (genclj-i3fp). A compiled call traces once for each shape of
   its operands (on Metal 10-160 ms for the 0.6B and 0.8B, and about 230
   ms more for a table width new to the process: one paged-attention
   kernel a width; 200-430 ms on the MoE), so the paged modes bucket
   what varies: rows (a step's, a forward's decode rows; at least
   :min-rows) and chunks' lengths (at least a block) round up to powers
   of 2, block tables' widths too, at least 256 positions' (contexts up
   to 256 share one), and the rows whose logits a forward computes
   follow from its other shapes. The padding is masked or ignored:
   padding rows write the scratch blocks, chunk padding rows sit at the
   chunk's last position (and under Qwen3.5 past its :chunk-len), a
   table's columns past a sequence's blocks are never read (each row
   reads its own positions). A forward's decode rows and its chunks are
   two compiled calls, evaluated together, so that their shapes do not
   multiply (on Metal, 2026-11-08: 3-7 ms more a mixed forward than one
   call, the weights read twice; 8 clients of Qwen3.5-0.8B lost 1-2% of
   their tokens/s and 4% of their time to first token).
   An engine compiles the common shapes when it opens (:precompile?,
   precompiled): those short contexts take, a decode of up to 16 rows,
   a lone read's chunk of each length, several prompts' chunks of up to
   64 tokens (continuous), and copy-on-write's copies; its :traces stat
   counts the calls of any other shape (shapes lists them). A pool that
   grows makes every shape new. Then it gives back the memory its backend
   keeps for later arrays (array/release-cache!): on MLX the warm calls'
   buffers would fill most of the buffer cache, leaving the steps' freed
   buffers too little of it (genclj-v1kg).

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
            [gen.array.paged :as gpaged]
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
               :paged-pools #'qwen3/paged-pools
               :compile-paged-step #'qwen3/compile-paged-step :compile-paged-batch-step #'qwen3/compile-paged-batch-step}
        qwen35 {:load #'qwen35/load :static-cache #'qwen35/static-cache :step #'qwen35/step :compile-step #'qwen35/compile-step
                :offsets #'qwen35/offsets :select-rows #'qwen35/select-rows :concat-rows #'qwen35/concat-rows :grow #'qwen35/grow
                ;; the paged mode's: the attention layers paged, the
                ;; GatedDeltaNet layers' states a row each (:paged-states)
                :paged-pools #'qwen35/paged-pools
                :compile-paged-step #'qwen35/compile-paged-step :paged-states #'qwen35/paged-states
                :compile-paged-batch-step #'qwen35/compile-paged-batch-step}]
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

(defn- release-dropped!
  "Release what each of groups holds (its payload, :cache or :states)
   once no slot's entry in the books (:table or :rows) is in it any more
   (array/release!), after the books dropped them: on MLX its buffers go
   back to its cache for the next step at once, not when the GC finds
   them. A step that reads its rows in place (forks still hold the group)
   or stacks them makes new arrays, and with the groups no branch holds
   left to the GC, Qwen3.5-0.8B's GatedDeltaNet states (18 MiB a row) ran
   a collection every 13 steps at 8 rows and every 7 at 16, and new
   buffers between (Metal, 2026-11-08, genclj-qu77: 23.1 against 20.2 ms
   a step at 8 rows, 39.1 against 32.8 at 16, float16). A group a step
   consumed (a lane) is released already."
  [state books payload groups]
  (when (seq groups)
    (let [held (into #{} (keep #(some-> % :group :id)) (books @state))]
      (doseq [g (vals (into {} (keep (fn [g] (when g [(:id g) g]))) groups))
              :when (not (held (:id g)))]
        (array/release! (payload g))))))

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
            ;; the row normalized before the books are written: a failure
            ;; leaves the slot as it was
            (let [lp (first (host-rows pool row "read"))
                  g (new-group! state cache)
                  old [(:group (entry state slot))]]
              (vswap! state assoc-in [:table slot] {:group g :row 0})
              (release-dropped! state :table :cache old)
              (tally! state :runs 1)
              {:logprobs lp :scores (when targets scores)})
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
            g (new-group! state cache')
            old (mapv #(:group (entry state %)) slots)]
        (doseq [[r i] (map-indexed vector order)]
          (vswap! state assoc-in [:table (slots i)] {:group g :row r}))
        (release-dropped! state :table :cache old)
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
    (let [old [(:group (entry state dst))]]
      (vswap! state assoc-in [:table dst] (entry state src))
      (release-dropped! state :table :cache old)))

  (slots-clear [_ slot]
    (let [old [(:group (entry state slot))]]
      (vswap! state assoc-in [:table slot] nil)
      (release-dropped! state :table :cache old)))

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
;;            :pool-grows (:stacked-steps with :rows), and the runs'
;;            forwards' (the continuous mode's)}

(defn- next-pow2 ^long [^long n] (loop [p 1] (if (c/< p n) (recur (c/* 2 p)) p)))

;; Shapes (genclj-i3fp; the namespace's docstring, Shapes): what varies in
;; a compiled call's operands is bucketed, so that its traces are few, and
;; the common shapes are compiled when the engine opens.

(defn- bucket
  "n rounded up to a power of 2, at least floor: a compiled call's sizes
   (its rows, its block tables' widths, its chunks' lengths), so that its
   shapes, and so its traces, are few."
  ^long [^long n ^long floor]
  (max floor (next-pow2 n)))

(def ^:private min-span
  "The positions a block table spans at the least (its width in blocks a
   power of 2): contexts up to it share one width. The paged kernels read
   each decode row's own positions only, whatever the width (measured on
   Metal, 2026-11-08: a step of 8 rows of Qwen3-0.6B 11.3 ms at 16
   blocks, 11.7 at 128; v1 holds 4 bytes a position of the width in
   threadgroup memory, 1 KiB here). A chunk's composite gathers and
   attends over the width's blocks: four chunks of 64 tokens of
   Qwen3-0.6B take 112 ms at 4 blocks and 120 at 16 in float16, 132 and
   151 in float32 (Qwen3.5-0.8B, a quarter of whose layers attend, 163
   and 165, 197 and 196). Unfloored, a chunk's width would follow its
   context (a prompt's first chunk, a split prompt's next, an extend),
   multiplying its shapes."
  256)

(defn- table-width
  "The width of the block tables of seqs (blocks of bs): their most
   blocks, bucketed, at least min-span positions' worth."
  ^long [seqs ^long bs]
  (bucket (reduce max 1 (map (comp count :blocks) seqs)) (paged/blocks-for min-span bs)))

(defn- chunk-floor
  "The fewest rows a chunk has: a block (bs), within a forward's budget of
   n-batch tokens' bucket."
  ^long [^long bs ^long n-batch]
  (min bs (next-pow2 n-batch)))

(defn- shape!
  "A call of the compiled shape sig counted (the engine's :shapes, by
   signature: the structure and shapes that key a compiled call's
   traces), and as a trace (:traces) when the engine had not made one of
   that shape before: on MLX, mlx_compile traces it then."
  [state sig]
  (when-not (contains? (:shapes @state) sig) (tally! state :traces 1))
  (vswap! state update-in [:shapes sig] (fnil inc 0)))

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

(defn- slice-rows
  "states (by layer) at their rows from to - 1: views, no copy."
  [states ^long from ^long to]
  (mapv (fn [e] (update-vals e #(array/slice % 0 from to))) states))

(defn- new-states-group!
  "A new group of states (by layer, n rows), its id never used before."
  [state states n]
  (let [id (:next @state)]
    (vswap! state update :next inc)
    {:id id :states states :n (long n)}))

(defn- groups-of
  "The groups of states slots' rows are in (nil for a slot without
   states)."
  [state slots]
  (mapv #(:group (nth (:rows @state) %)) slots))

(defn- row-states
  "The states of slot's row alone (by layer, one row): its group's when
   it is one row, else that row's view (both read, never consumed)."
  [state slot]
  (let [{:keys [group row]} (nth (:rows @state) slot)
        row (long row)]
    (if (c/== 1 (long (:n group))) (:states group) (slice-rows (:states group) row (inc row)))))

(defn- zero-states
  "The states (by layer) of n rows at the start: zeros."
  [{:keys [model state]} n]
  (call state :paged-states model n))

(defn- stacked
  "The states (by layer) of a step's rows srcs (each [group row], or nil
   for zeros: no states yet), padded to r rows: one copy. One group's
   rows alone are taken (padded by its first); otherwise runs of
   consecutive rows are sliced (views) and concatenated with zeros for
   the rest."
  [ctx srcs r]
  (let [n (count srcs)
        r (long r)
        ;; groups by id: a group's hash would read its arrays
        ids (distinct (map #(some-> % first :id) srcs))]
    (if (and (c/== 1 (count ids)) (first ids))
      (let [rows (mapv second srcs)]
        (take-rows (:states (ffirst srcs)) (into rows (repeat (c/- r n) (first rows)))))
      ;; runs [group from len], a group nil for zeros, padding last
      (let [runs (-> (reduce (fn [runs [g row]]
                               (let [[pg from len] (peek runs)]
                                 (if (and (seq runs) (= (:id g) (:id pg))
                                          (or (nil? g) (c/== (long row) (c/+ (long from) (long len)))))
                                   (conj (pop runs) [pg from (inc (long len))])
                                   (conj runs [g row 1]))))
                             [] srcs)
                     (cond-> (c/< n r) (conj [nil 0 (c/- r n)])))
            zeros (when (some (comp nil? first) runs) (zero-states ctx 1))
            parts (mapv (fn [[g from len]]
                          (if g
                            (slice-rows (:states g) from (c/+ (long from) (long len)))
                            (mapv (fn [e] (update-vals e #(array/broadcast-to % (assoc (array/shape %) 0 len)))) zeros)))
                        runs)]
        (apply mapv (fn [& es] (into {} (map (fn [k] [k (array/concatenate (mapv k es) 0)])) (keys (first es)))) parts)))))

(defn- step-states
  "How a step of slots (whose family has recurrent layers) reads their
   rows' states, padded to r rows: {:order :fresh :held :lane}, the
   step's row i being (slots (order i)) (the step chooses its rows'
   order); fresh the states (by layer) it consumes, held those it reads
   only (a vector by layer, empty for none); lane the slots whose group
   it consumes. A slot without states (none read yet) starts from zeros.
   - A group of r rows, exactly the slots' (their first, padding after)
     and held by them alone, is stepped in place (a lane: consumed).
   - A group's consecutive rows, the slots' first, are read in place
     (held: forks of their branches still read them; its rows after
     them the step's padding).
   - Otherwise the rows are stacked (stacked: a copy, consumed)."
  [{:keys [state] :as ctx} slots r]
  (let [n (count slots)
        r (long r)
        rs (mapv #(nth (:rows @state) %) slots)
        g (:group (first rs))
        rows (mapv #(some-> % :row long) rs)
        one-group? (and g (every? #(and % (c/== (long (:id g)) (long (:id (:group %))))) rs))
        lo (when one-group? (long (reduce min rows)))
        run? (and one-group? (apply distinct? rows) (c/== (long (reduce max rows)) (c/+ (long lo) (dec n)))
                  (c/<= (c/+ (long lo) r) (long (:n g))))
        by-row (fn [] (mapv (zipmap rows (range n)) (range lo (c/+ (long lo) n))))]
    (cond
      (and run? (zero? (long lo)) (c/== r (long (:n g)))
           (= (set slots) (into #{} (keep-indexed (fn [s e] (when (and e (c/== (long (:id g)) (long (:id (:group e))))) s)))
                                (:rows @state))))
      {:order (by-row) :fresh (:states g) :held [] :lane (vec slots)}

      run?
      {:order (by-row) :fresh nil
       :held (if (and (zero? (long lo)) (c/== r (long (:n g)))) (:states g) (slice-rows (:states g) lo (c/+ (long lo) r)))}

      :else
      (let [order (vec (sort-by (fn [i] (if-let [e (rs i)] [0 (:id (:group e)) (:row e)] [1 0 0])) (range n)))]
        {:order order :fresh (stacked ctx (mapv #(when-let [e (rs %)] [(:group e) (:row e)]) order) r) :held []}))))

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
  (let [want (bucket (count copies) 1)
        dsts (set (map second copies))]
    (into copies (map (fn [b] [b b])) (take (c/- want (count copies)) (remove dsts (concat (:free a) (range scratch)))))))

(defn- copy-call!
  "copy-blocks!'s two calls, of the block pairs all."
  [{:keys [state gather-fn put-fn] :as ctx} all]
  (shape! state [:copy (count all) (:size (:alloc @state))])
  (let [blocks (doto (gather-fn (:pools @state) (i32 (map first all))) array/eval!)
        pools (donating ctx #(doto (put-fn (:pools @state) blocks (i32 (map second all))) array/eval!))]
    (vswap! state assoc :pools pools)))

(defn- copy-blocks!
  "The block copies [[from to] …] made in the pools (copy-on-write), a
   the allocator after them: in two calls, the blocks read and evaluated
   first (gather-fn), then written (put-fn, consuming the pools), padded
   to a power of 2 (padded: a padding copy writes a block whose contents
   mean nothing). In one call MLX copies each pool instead of writing it
   in place, its read still holding it (measured on Metal, 2026-11-07: 14
   to 120 ms a copy as the pools grew from 0.6 to 3.8 GB)."
  [{:keys [state] :as ctx} a copies]
  (when (seq copies)
    (copy-call! ctx (padded a (:scratch @state) copies))
    (tally! state :copied-blocks (count copies))))

(declare paged-rows! continuous-run)

(defn- paged-decode-step
  "slots-step's body in the paged mode: each slot's sequence made ready
   for a position, then one compiled decode of the rows, padded to a
   power of 2 rows (at least :min-rows), over block tables of a power of
   2 width: the step's shapes, and so its traces, are few. A block a
   second writer shares is copied first (copy-on-write, copy-blocks!). A
   family with recurrent layers (:rows) steps its rows' states too
   (step-states), its rows in their order."
  [{:keys [state min-rows] :as ctx} slots positions tokens]
  (doseq [[slot pos] (map vector slots positions)] (check-paged-position state slot pos "advance"))
  (let [n (count slots)
        r (bucket n min-rows)]
    (if-not (contains? @state :rows)
      (paged-rows! ctx slots tokens r nil)
      (let [{:keys [order fresh] :as states} (step-states ctx slots r)
            rows (paged-rows! ctx (mapv slots order) (mapv tokens order) r states)]
        (cond (:lane states) nil
              fresh (tally! state :stacked-steps 1)
              :else (tally! state :held-steps 1))
        (reduce (fn [out [i k]] (assoc out k (rows i))) (vec (repeat n nil)) (map-indexed vector order))))))

(defn- normalized
  "The rows of logits (a vector of arrays, nil for none), in order,
   normalized on the host (host-rows), where a failure after a step that
   consumed a lane's states (lane, its slots) loses those slots: named in
   the error's :lost-slots."
  [pool logits lane]
  (try (into [] (mapcat #(host-rows pool % "advance")) (remove nil? logits))
       (catch Throwable t
         (throw (if (seq lane) (exhausted t lane) t)))))

(defn- pad-slot
  "The slot padding row i of a call writes: slot i of the scratch blocks
   (the first :scratch blocks, which no sequence holds)."
  ^long [^long i]
  i)

(defn- pad-table
  "Padding row i's block table, of width: it writes pad-slot's slot i
   (its position 0) and reads that slot's block."
  [^long i ^long bs ^long width]
  (vec (cons (quot i bs) (repeat (dec width) 0))))

(defn- step-call!
  "One compiled decode step (the family's compile-paged-step, consuming
   the pools and the fresh states) of r rows, width wide: host vectors
   of the rows' positions, block tables, write slots and tokens, and
   their states ({:fresh :held}, nil without recurrent layers); its
   shape counted (shape!). [logits states'], the pools it wrote stored
   at once (it consumed those it was given)."
  [{:keys [state step-fn] :as ctx} r width positions tables slots tokens {:keys [fresh held] :as states}]
  (shape! state [:step r width (cond fresh :fresh (seq held) :held) (:size (:alloc @state))])
  (let [pos (i32 positions)
        tables (i32 tables)
        sl (i32 slots)
        [logits layers] (donating ctx #(let [out (if states
                                                   (step-fn (merged (:pools @state) fresh) pos tables sl tokens held)
                                                   (step-fn (:pools @state) pos tables sl tokens))]
                                         (array/eval! out)
                                         out))
        [pools states'] (split-layers state layers)]
    (vswap! state assoc :pools pools)
    [logits states']))

(defn- paged-rows!
  "paged-decode-step's step of slots (in the step's row order), padded
   to r rows, with the rows' states (step-states' {:fresh :held :lane}
   of r rows; nil for a family without recurrent layers), the fresh ones
   consumed by the step: the rows, in order."
  [{:keys [state pool] :as ctx} slots tokens r {:keys [lane] :as states}]
  (let [n (count slots)
        r (long r)
        bs (long (:block-size @state))
        seqs (mapv #(or (entry state %) paged/empty-sequence) slots)
        [a seqs' write-slots copies] (prepared ctx seqs (repeat n 1))
        _ (copy-blocks! ctx a copies)
        width (table-width seqs' bs)
        ;; the step's pools are stored at once (step-call!), the rest of
        ;; the books once its rows are normalized: a failure then leaves
        ;; every slot as it was, since what the step wrote lies past each
        ;; sequence's end or in blocks the books still hold free (but a
        ;; lane's, whose states it consumed: lost)
        [logits states'] (step-call! ctx r width
                                     (into (mapv :len seqs) (repeat (c/- r n) 0))
                                     (into (mapv #(paged/table-row % width 0) seqs') (map #(pad-table % bs width)) (range n r))
                                     (into (mapv first write-slots) (map pad-slot) (range n r))
                                     (into (mapv #(vector (tokens %)) (range n)) (repeat (c/- r n) [0]))
                                     states)
        g (when states (new-states-group! state states' r))
        rows (normalized pool [(array/slice logits 0 0 n)] lane)
        old (when g (groups-of state slots))]
    (vswap! state (fn [st] (cond-> (-> st
                                       (assoc :alloc a)
                                       (update :table #(reduce (fn [t [slot s']] (assoc t slot s')) % (map vector slots seqs'))))
                             g (update :rows #(reduce (fn [t [i slot]] (assoc t slot {:group g :row i})) % (map-indexed vector slots))))))
    (release-dropped! state :rows :states old)
    (tally! state :steps 1)
    (tally! state :rows n)
    (tally! state :padded-rows (c/- r n))
    rows))

(defn- paged-copy!
  "slots-copy in the paged modes: dst shares src's blocks (and its row
   of states)."
  [state src dst]
  (let [[a s] (paged/fork (:alloc @state) (entry state src))
        old (when (contains? @state :rows) (groups-of state [dst]))]
    (vswap! state #(cond-> (-> % (assoc :alloc a) (assoc-in [:table dst] s))
                     (contains? % :rows) (assoc-in [:rows dst] (get-in % [:rows src]))))
    (release-dropped! state :rows :states old)))

(defn- paged-clear!
  "slots-clear in the paged modes: slot's blocks released (and its row
   of states dropped)."
  [state slot]
  (when-let [s (entry state slot)]
    (let [old (when (contains? @state :rows) (groups-of state [slot]))]
      (vswap! state #(cond-> (-> % (assoc :alloc (paged/dispose (:alloc %) s)) (assoc-in [:table slot] nil))
                       (contains? % :rows) (assoc-in [:rows slot] nil)))
      (release-dropped! state :rows :states old))))

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
        (let [r (continuous-run ctx slot pos tokens targets)]
          (tally! state :runs 1)
          r)
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
;; :chunks :padded-rows :mixed-forwards :copied-blocks :pool-grows}, and
;; its scratch blocks enough for the padding rows of a forward's decode
;; rows and of its chunks (at most the budget, or a block).

(defn- one-token? [r] (c/== 1 (count (:tokens r))))

(defn- forward-plan
  "Which of reads (gen.llm.slots/slots-forward's) a forward reads, and
   its rows: every one-token read (decode rows, in order, their count
   bucketed, at least min-rows), and chunks of the longer reads, first
   come first served: the first one's next chunk of what the budget
   (n-batch tokens) leaves, at least a block (bs), its rows t bucketed
   (genclj-i3fp: at least chunk-floor, and up, past what the budget
   leaves when its tokens do); then the next longer reads' chunks of at
   most t tokens each, t rows each, while the budget holds them, at most
   max-chunks in all, and a power of 2 of them (the rest wait a forward;
   genclj-6qb6). {:decode [indices] :d-rows :chunks [indices] :cs [their
   tokens] :t}."
  [reads min-rows n-batch bs max-chunks]
  (let [decode (filterv #(one-token? (reads %)) (range (count reads)))
        d (count decode)
        d-rows (if (pos? d) (bucket d min-rows) 0)
        longer (remove #(one-token? (reads %)) (range (count reads)))
        n-of #(count (:tokens (reads %)))
        left (max (long bs) (c/- (long n-batch) d-rows))
        c (when (seq longer) (min (long (n-of (first longer))) left))
        t (when c (bucket c (chunk-floor bs n-batch)))
        k (when t (min (long max-chunks) (max 1 (quot left (long t))) (count longer)))
        chunks (when t (vec (take (Long/highestOneBit k) longer)))]
    {:decode decode :d-rows d-rows :chunks (or chunks []) :t t
     :cs (into (if c [c] []) (map #(min (long (n-of %)) (long t))) (rest chunks))}))

(defn- chunked
  "States (by layer) as a chunk's: a GatedDeltaNet layer's {:chunk-conv
   :chunk-state}."
  [states]
  (mapv #(if (contains? % :state) {:chunk-conv (:conv %) :chunk-state (:state %)} {}) states))

(defn- chunk-states
  "{:fresh :held} the states (by layer, a row a chunk, a GatedDeltaNet
   layer's {:chunk-conv :chunk-state}) the chunks read into slots start
   from. One chunk: its row's, read in place, or zeros when it has none
   yet (held: one structure, so one trace a shape). Several: their rows
   stacked (stacked: a copy, zeros for a slot with none yet; fresh)."
  [{:keys [state] :as ctx} slots]
  (cond
      (c/< 1 (count slots))
      {:fresh (chunked (stacked ctx (mapv #(when-let [e (nth (:rows @state) %)] [(:group e) (:row e)]) slots) (count slots)))}
      (nth (:rows @state) (first slots))
      {:held (chunked (row-states state (first slots)))}
      :else
      {:held (chunked (zero-states ctx 1))}))

(defn- batch-layers
  "A forward's [pools decode-states chunk-states] of its layers (the
   family's batch step's :layers, or qwen3's :pools): the states by
   layer, a GatedDeltaNet layer's {:conv :state}, nil without recurrent
   layers."
  [state {:keys [pools layers]}]
  (if-not (contains? @state :rows)
    [pools nil nil]
    [(mapv #(select-keys % [:k :v]) layers)
     (mapv #(select-keys % [:conv :state]) layers)
     (mapv #(if (contains? % :chunk-state) {:conv (:chunk-conv %) :state (:chunk-state %)} {}) layers)]))

(defn- states-kind
  "How a call is given states ({:fresh :held}): :fresh, :held, or nil
   (none): what its structure, and so its traces, depend on."
  [{:keys [fresh held]}]
  (cond fresh :fresh (seq held) :held))

(defn- forward-call!
  "One compiled forward (the family's compile-paged-batch-step, consuming
   the pools and the fresh states) of batch's host vectors (:tokens, a
   vector of [id] a row, :positions :slots :out, and when present
   :tables, :chunk, :targets and :chunk-len, a vector a row each) and
   the decode rows' and chunks' states ({:fresh :held}; nil for none),
   its shape counted (shape!), not evaluated: [result decode-states'
   chunk-states'], the pools it wrote stored at once (it consumed those
   it was given), for the next call."
  [{:keys [state batch-fn]} batch dstates cstates]
  (let [{:keys [tokens out tables chunk targets]} batch
        _ (shape! state [:forward (count tokens) (count out) (count tables) (count (first tables)) (count chunk)
                         (count (first chunk)) (some? targets) (states-kind dstates) (states-kind cstates)
                         (:size (:alloc @state))])
        batch (update-vals batch i32)
        res (if (contains? @state :rows)
              (batch-fn (reduce merged (:pools @state) (keep :fresh [dstates cstates])) batch
                        (reduce (fn [h s] (if (seq s) (if (seq h) (mapv merge h s) s) h)) [] (map :held [dstates cstates])))
              (batch-fn (:pools @state) batch))
        [pools dstates' cstates'] (batch-layers state res)]
    (vswap! state assoc :pools pools)
    [res dstates' cstates']))

(defn- forward-calls!
  "The compiled forwards of calls ([batch dstates cstates] each,
   forward-call!'s), in order, each given the pools the last wrote,
   evaluated together: their [result decode-states' chunk-states'], the
   pools the last wrote stored. On failure every slot is lost
   (donating)."
  [ctx calls]
  (donating ctx #(let [outs (mapv (fn [[batch ds cs]] (forward-call! ctx batch ds cs)) calls)]
                   ;; the pools each call but the last wrote, the next
                   ;; consumed: not evaluated (read) here
                   (array/eval! (conj (mapv (fn [[res ds cs]] [(select-keys res [:logits :scores]) ds cs]) (pop outs))
                                      (peek outs)))
                   outs)))

(defn- batch-forward!
  "slots-forward's body in the continuous mode (see the namespace's
   docstring): the plan's reads made ready (copy-on-write), the compiled
   forwards of their rows (the family's compile-paged-batch-step,
   consuming the pools, and the states they were given fresh): one of
   the decode rows, one of the chunks, evaluated together, so that their
   shapes are the decode rows' and the chunks' apart, not every pair
   (genclj-i3fp). Their pools kept, their rows read and normalized on the
   host, then the rest of the books written: a forward whose rows fail
   to normalize leaves its slots as they were (but a lane's, whose
   states it consumed: lost). A family with recurrent layers (Qwen3.5)
   reads its decode rows' states as a paged step does (step-states, in
   their order) and the chunks' from their rows (chunk-states), each
   chunk's padding rows past its :chunk-len."
  [{:keys [state pool min-rows n-batch max-chunks] :as ctx} reads]
  (let [bs (long (:block-size @state))
        {:keys [decode d-rows chunks cs t]} (forward-plan reads min-rows n-batch bs (or max-chunks 1))
        rows? (contains? @state :rows)
        dstates (when (and rows? (seq decode)) (step-states ctx (mapv #(:slot (reads %)) decode) d-rows))
        decode (if dstates (mapv decode (:order dstates)) decode)
        cstates (when (and rows? (seq chunks)) (chunk-states ctx (mapv #(:slot (reads %)) chunks)))
        chosen (into decode chunks)
        _ (doseq [i chosen] (check-paged-position state (:slot (reads i)) (:pos (reads i)) "forward"))
        seqs (mapv #(or (entry state (:slot (reads %))) paged/empty-sequence) chosen)
        by-chunk (zipmap chunks cs)
        [a seqs' write-slots copies] (prepared ctx seqs (mapv #(get by-chunk % 1) chosen))
        _ (copy-blocks! ctx a copies)
        d (count decode)
        d-rows (long d-rows)
        t (long (or t 0))
        nk (count chunks)
        read-c (long (reduce c/+ 0 cs))
        dseqs (subvec seqs' 0 d)
        cseqs (subvec seqs' d)
        ;; the decode rows' call: a token a row, padding rows after them
        ;; (row i writing the scratch slot i), the logits of every row
        dbatch (when (pos? d)
                 (let [width (table-width dseqs bs)]
                   {:tokens (into (mapv #(vector (first (:tokens (reads %)))) decode) (repeat (c/- d-rows d) [0]))
                    :positions (into (mapv #(:pos (reads %)) decode) (repeat (c/- d-rows d) 0))
                    :slots (into (mapv first (subvec write-slots 0 d)) (map pad-slot) (range d d-rows))
                    :out (vec (range d-rows))
                    :tables (into (mapv #(paged/table-row % width 0) dseqs) (map #(pad-table % bs width)) (range d d-rows))}))
        ;; the chunks' call: chunk j's rows (row i of it at (at j i)), its
        ;; c tokens, then padding to t rows at its last position (seeing
        ;; what that sees: within its table), writing the scratch slots
        ;; after the decode rows'
        at (fn [j i] (c/+ (c/* (long j) t) (long i)))
        per-chunk (fn [f] (into [] (mapcat f) (map vector (range) chunks cs)))
        ;; the chunks' rows read: each chunk's last when it ends its read,
        ;; then their scored rows (those of its targets among its tokens)
        ended (filterv (fn [j] (c/== (long (cs j)) (count (:tokens (reads (chunks j)))))) (range nk))
        ks (mapv (fn [i c] (if-let [tg (:targets (reads i))] (min (long c) (count tg)) 0)) chunks cs)
        scored? (some pos? ks)
        cbatch (when (pos? nk)
                 (let [cwidth (table-width cseqs bs)
                       out (-> (mapv #(at % (dec (long (cs %)))) ended)
                               (into (mapcat (fn [j k] (map #(at j (long %)) (range k))) (range) ks)))
                       ;; the rows whose logits are computed, as many as
                       ;; its other shapes allow at most: with scores,
                       ;; each chunk's rows and its last; else its last
                       o (long (if scored? (c/* nk (inc t)) nk))]
                   (cond-> {:tokens (per-chunk (fn [[_ i c]]
                                                 (concat (map vector (subvec (:tokens (reads i)) 0 c)) (repeat (c/- t (long c)) [0]))))
                            :positions (per-chunk (fn [[_ i c]]
                                                    (let [p (long (:pos (reads i))) c (long c)]
                                                      (concat (range p (c/+ p c)) (repeat (c/- t c) (c/+ p (dec c)))))))
                            :slots (vec (mapcat (fn [j sl c] (concat sl (map pad-slot (range (c/+ d-rows (long (at j c))) (c/+ d-rows (long (at j t)))))))
                                                (range) (subvec write-slots d) cs))
                            :out (into out (repeat (c/- o (count out)) 0))
                            :chunk (mapv #(paged/table-row % cwidth 0) cseqs)}
                     scored? (assoc :targets (-> (vec (repeat (count ended) 0))
                                                 (into (mapcat (fn [i k] (when (pos? (long k)) (subvec (:targets (reads i)) 0 k))) chunks ks))
                                                 (into (repeat (c/- o (count out)) 0))))
                     rows? (assoc :chunk-len cs))))
        ;; the forwards' pools are stored at once (forward-calls!), the
        ;; rest of the books once their rows are read and normalized: a
        ;; failure then leaves every slot as it was, as a paged step's
        ;; does, since what the forwards wrote lies past each sequence's
        ;; end or in blocks the books still hold free (but a lane's, whose
        ;; states it consumed: lost)
        outs (forward-calls! ctx (cond-> []
                                   dbatch (conj [dbatch dstates nil])
                                   cbatch (conj [cbatch nil cstates])))
        [dres dstates'] (when dbatch (first outs))
        [cres _ cstates'] (when cbatch (peek outs))
        gd (when (and dstates (pos? d)) (new-states-group! state dstates' d-rows))
        gc (when (and rows? (pos? nk)) (new-states-group! state cstates' nk))
        rows (normalized pool [(when dres (array/slice (:logits dres) 0 0 d))
                               (when (seq ended) (array/slice (:logits cres) 0 0 (count ended)))]
                         (:lane dstates))
        sc (when scored? (mapv double (array/->floats (:scores cres))))
        old (when rows? (groups-of state (mapv #(:slot (reads %)) chosen)))]
    (vswap! state (fn [st] (cond-> (-> st
                                       (assoc :alloc a)
                                       (update :table #(reduce (fn [t [i s']] (assoc t (:slot (reads i)) s')) % (map vector chosen seqs'))))
                             gd (update :rows #(reduce (fn [t [j i]] (assoc t (:slot (reads i)) {:group gd :row j})) % (map-indexed vector decode)))
                             gc (update :rows #(reduce (fn [t [j i]] (assoc t (:slot (reads i)) {:group gc :row j})) % (map-indexed vector chunks))))))
    (release-dropped! state :rows :states old)
    (when dstates
      (tally! state (cond (:lane dstates) :lane-forwards (:fresh dstates) :stacked-forwards :else :held-forwards) 1))
    (tally! state :forwards 1)
    (tally! state :decode-rows d)
    (tally! state :chunk-rows read-c)
    (tally! state :chunks nk)
    (tally! state :padded-rows (c/- (c/+ d-rows (c/* nk t)) d read-c))
    (when (and (pos? d) (pos? nk)) (tally! state :mixed-forwards 1))
    (let [by (zipmap decode (range))
          cj (zipmap chunks (range))
          ;; chunk j's row among the rows read (when it ends its read),
          ;; and its scores' offset in sc
          lp-of (zipmap ended (range d (c/+ d (count ended))))
          sc-of (vec (reductions c/+ (count ended) ks))]
      (mapv (fn [i]
              (let [{:keys [targets]} (reads i)]
                (if-let [j (cj i)]
                  (let [k (long (ks j))
                        from (long (sc-of j))]
                    {:read (cs j)
                     :scores (when targets (if (pos? k) (subvec sc from (c/+ from k)) []))
                     :logprobs (when-let [row (lp-of j)] (rows row))})
                  (if (contains? by i)
                    {:read 1 :scores (when targets []) :logprobs (rows (by i))}
                    {:read 0 :scores (when targets [])}))))
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
;; Shapes compiled when the engine opens (genclj-i3fp)
;; ---------------------------------------------------------------------------

(defn- warm!
  "A compiled call of shape (shape! 's signature) made once, every row
   padding: rows at position 0 writing the scratch blocks (row i its slot
   i) and reading them, tables of block 0 and the scratch blocks, tokens
   and targets 0, states zeros; its outputs released, the pools it wrote
   kept. On MLX it traces the call and makes its kernels."
  [{:keys [state] :as ctx} [kind & dims]]
  (let [bs (long (:block-size @state))
        z (fn [n] (zero-states ctx n))]
    (case kind
      :step (let [[r width skind] dims
                  r (long r)
                  states (case skind :fresh {:fresh (z r) :held []} :held {:held (z r)} nil)]
              (array/release!
               [(step-call! ctx r width (vec (repeat r 0)) (mapv #(pad-table % bs width) (range r)) (mapv pad-slot (range r))
                            (vec (repeat r [0])) states)
                (:held states)]))
      :forward (let [[r o d width k cwidth targets? dkind ckind] dims
                     r (long r)
                     d (long d)
                     k (long k)
                     dstates (case dkind :fresh {:fresh (z d) :held []} :held {:held (z d)} nil)
                     cstates (case ckind :fresh {:fresh (chunked (z k))} :held {:held (chunked (z k))} nil)
                     [[res ds cs]] (forward-calls!
                                    ctx
                                    [[(cond-> {:tokens (vec (repeat r [0])) :positions (vec (repeat r 0))
                                               :slots (mapv pad-slot (range r)) :out (vec (repeat o 0))}
                                        (pos? d) (assoc :tables (mapv #(pad-table % bs width) (range d)))
                                        (pos? k) (assoc :chunk (vec (repeat k (vec (repeat cwidth 0)))))
                                        targets? (assoc :targets (vec (repeat o 0)))
                                        (and (contains? @state :rows) (pos? k)) (assoc :chunk-len (vec (repeat k (quot (c/- r d) k)))))
                                      dstates cstates]])]
                 ;; the pools it wrote are the engine's (in :pools, and in
                 ;; a Qwen3.5's :layers): not released
                 (array/release! [(select-keys res [:logits :scores]) ds cs (:held dstates) (:held cstates)]))
      ;; self-copies of blocks whose contents mean nothing (padded's)
      :copy (let [[n] dims] (copy-call! ctx (mapv (fn [b] [b b]) (take n (concat (range (:scratch @state)) (:free (:alloc @state))))))))
    nil))

(defn- forward-shape
  "A forward's shape (forward-call!'s signature but the pools' size): d
   decode rows over tables of width, k chunks of t rows over tables of
   cwidth, its logits' rows following (batch-forward!), with targets or
   not, its decode rows' and chunks' states given as dkind and ckind."
  [d width k t cwidth targets? dkind ckind]
  (let [d (long d) k (long k) t (long t)]
    [:forward (c/+ d (c/* k t)) (if targets? (c/+ d (c/* k (inc t))) (c/+ d k)) d (if (pos? d) width 0) k (if (pos? k) cwidth 0)
     (boolean targets?) dkind ckind]))

(defn- pow2s
  "The powers of 2 from lo to hi."
  [^long lo ^long hi]
  (take-while #(c/<= (long %) hi) (iterate #(c/* 2 (long %)) lo)))

(defn- precompiled
  "The shapes an engine compiles when it opens (:precompile?), as
   forward-shape's, [:step rows width states] and [:copy pairs]: the
   shapes contexts up to min-span positions take (tables of its width),
   most common first: a decode of each row count up to 16 (or the
   slots'); a lone read's chunk of each length up to the budget, with
   scores (an LM's extend) and without (an open); in the continuous
   mode, several prompts' chunks of up to 64 tokens, with scores; and
   copy-on-write's first pair counts. A forward's decode rows and chunks
   are separate calls (batch-forward!), so these are its shapes."
  [{:keys [continuous? min-rows n-batch max-chunks]} rows? ^long bs ^long n-slots]
  (let [w (paged/blocks-for min-span bs)
        n-batch (long n-batch)
        rs (pow2s (bucket 1 min-rows) (min 16 (bucket n-slots min-rows)))
        ts (pow2s (chunk-floor bs n-batch) (next-pow2 n-batch))
        dkinds (if rows? [:fresh :held] [nil])]
    (vec (concat (if continuous?
                   (for [r rs, k dkinds] (forward-shape r w 0 0 0 false k nil))
                   (concat (for [r rs, k dkinds] [:step r w k])
                           ;; a run of one token (an LM's first)
                           (for [k dkinds] (forward-shape 1 w 0 0 0 false k nil))))
                 (for [t ts, targets? [false true]] (forward-shape 0 0 1 t w targets? nil (when rows? :held)))
                 (when continuous?
                   (for [t ts
                         :while (c/<= (long t) 64)
                         k (pow2s 2 (min (long max-chunks) (quot n-batch (long t))))]
                     (forward-shape 0 0 k t w true nil (when rows? :fresh))))
                 (for [n (pow2s 1 16)] [:copy n])))))

(defn- precompile!
  "shapes (precompiled's) compiled, each by one call (warm!), on backend
   b: the engine's :shapes then hold them, counting no call (nor
   :traces). Then the memory b keeps for later arrays is given back
   (array/release-cache!): on MLX the warm calls' buffers filled most of
   its buffer cache (1.9 of 2.1 GB on a 32 GB Mac, Qwen3.5-0.8B), so the
   steps' freed buffers found it full, were freed and made anew, and
   token SMC lost 2-3% (genclj-v1kg). [shapes ms]."
  [{:keys [state] :as ctx} b shapes]
  (let [t0 (System/nanoTime)]
    (run! #(warm! ctx %) shapes)
    (array/release-cache! b)
    (vswap! state (fn [s] (-> s
                              (update :shapes update-vals (constantly 0))
                              (assoc-in [:stats :traces] 0))))
    [(count (:shapes @state)) (c// (double (c/- (System/nanoTime) t0)) 1e6)]))

;; ---------------------------------------------------------------------------
;; Opening
;; ---------------------------------------------------------------------------

(def ^:private defaults {:max-branches 32 :n-ctx 4096 :n-batch 256 :block 128
                         :paged? true :block-size 16 :pool-blocks 128 :min-rows 1
                         :continuous? false :wait 0.25 :max-chunks 16 :precompile? true})

(def ^:private option-checks
  {:model #(or (and (string? %) (.isFile (java.io.File. ^String %)))
               (and (map? %) (contains? % :config) (contains? % :weights)))
   :dtype #{:float16 :bfloat16 :float32 :float64}
   :max-branches pos-int? :n-ctx pos-int? :n-batch pos-int? :block pos-int?
   :paged? boolean? :block-size #{8 16 32} :pool-blocks pos-int? :min-rows pos-int?
   :continuous? boolean? :wait #(and (number? %) (c/<= 0.0 (double %) 10.0)) :max-chunks pos-int?
   :precompile? boolean?})

(def ^:private mode-options
  "The options only some modes read, by mode: one given to a mode that
   does not read it is loud (rule 6), as an unknown one is."
  {:static #{:block}
   :paged #{:block-size :pool-blocks :min-rows :precompile?}
   :continuous #{:block-size :pool-blocks :min-rows :precompile? :wait :max-chunks}})

(def ^:private mode-only
  "Every option some mode does not read."
  (into [] (sort (distinct (mapcat val mode-options)))))

(defn- check-options
  "opts with their defaults, checked: a loud error for an unknown, missing
   or bad one."
  [opts]
  (when-not (map? opts)
    (throw (ex-info (str "array-engine/engine: options are a map, not " (pr-str opts)) {:reason :invalid-option})))
  (when-let [ks (seq (remove option-checks (keys opts)))]
    (throw (ex-info (str "array-engine/engine: unknown options " (vec ks) "; it takes " (vec (sort (keys option-checks))))
                    {:reason :invalid-option :unknown (vec ks)})))
  (let [given opts
        {:keys [model] :as opts} (merge defaults opts)]
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
      (throw (ex-info "array-engine/engine: :continuous? batches over the paged mode, which :paged? false turns off"
                      {:reason :invalid-option :continuous? true :paged? false})))
    (let [mode (cond (not (:paged? opts)) :static (:continuous? opts) :continuous :else :paged)
          unread (filter (fn [k] (and (contains? given k) (not (contains? (mode-options mode) k)))) mode-only)]
      (when (seq unread)
        (throw (ex-info (str "array-engine/engine: the " (name mode) " mode does not read " (vec unread))
                        {:reason :invalid-option :mode mode :unread (vec unread)}))))
    opts))

(defn- quantized-file?
  "Does the GGUF model at path (its files, a split's too) hold quantized
   weights: Q8_0, Q4_0/1, Q5_0/1 or K-quants?"
  [path]
  (boolean (some #(or (gguf/quantized? %) (= :q6_k %)) (map :type (mapcat :tensors (gguf/headers path))))))

(defn- experts-file?
  "Is the GGUF model at path a mixture of experts (its architecture's
   expert_count positive)?"
  [path]
  (let [m (:metadata (gguf/header path))]
    (pos? (long (or (get m (str (get m "general.architecture") ".expert_count")) 0)))))

(defn- default-dtype
  "Checked opts, with :dtype :float16 where no :dtype is given and the
   engine loads a GGUF file of quantized weights:
   - on MLX's CUDA (backend b), every such file (the owner's call,
     2026-11-08: MLX's fast quantized products on CUDA take half floats
     only, and every lead over llama.cpp there was measured in float16),
     a mixture of experts included (after genclj-fxag: in float16 its
     distance from the float32 forward is its router's flips at near-ties,
     forced to the same experts a median of 0.015 nats, and its NLL is
     float32's, -0.004 +- 0.002 nats a token over 4,080);
   - on Metal, a mixture of experts only (the owner, 2026-11-08, after
     genclj-0nxn): it now loads at a lower peak in float16 than in float32
     (Qwen3.6-35B-A3B 23.1 against 28.4 GB) and runs faster (token SMC
     about 13%), its NLL float32's. Dense models stay float32 there, where
     their bands against llama.cpp were measured.
   As given elsewhere (MLX's CPU, the JVM backend, a model passed loaded)."
  [{:keys [model dtype] :as opts} b]
  (let [device (when (and (nil? dtype) (string? model)) (array/device b))]
    (if (and (or (= :cuda device) (and (= :metal device) (experts-file? model)))
             (quantized-file? model))
      (assoc opts :dtype :float16)
      opts)))

(defn- model-backend
  "The backend m's weights live on."
  [m]
  (array/backend (first (filter array/backend-array? (array/tree-leaves (:weights m))))))

(defn- open-paged-slots
  "PagedSlots (with :continuous?, ContinuousSlots) for checked opts
   (:paged?), the model m loaded on backend b, its family's functions
   fam: the pools made, the step compiled."
  [{:keys [model max-branches n-ctx n-batch block-size pool-blocks min-rows continuous? wait max-chunks precompile?] :as opts} m b fam]
  (when-not (:paged-pools fam)
    (throw (ex-info (str "array-engine/engine: the paged mode runs Qwen3 and Qwen3.5, not " (pr-str (arch-of model)))
                    {:reason :unsupported :architecture (arch-of model)})))
  (let [{:keys [vocab context] :as cfg} (:config m)
        n-ctx (long (if context (min (long n-ctx) (long context)) n-ctx))
        n-slots (inc (long max-branches))
        ;; scratch blocks: a slot each for the most rows a step or forward
        ;; has: its decode rows' and its chunks', which are at most the
        ;; budget's bucket, or a block
        scratch (paged/blocks-for (c/+ (bucket n-slots min-rows) (max (next-pow2 n-batch) (long block-size)))
                                  block-size)
        size (c/+ scratch (long pool-blocks))
        ;; a forward of decode rows and chunks (the continuous mode's, and a
        ;; run's in the paged mode), and the paged mode's decode step
        batch-fn ((:compile-paged-batch-step fam) m {:donate? true})
        step-fn (when-not continuous? ((:compile-paged-step fam) m {:donate? true}))
        ;; copy-blocks in two calls (copy-blocks!), each layer with pools
        ;; (a Qwen3.5's GatedDeltaNet layers have none: {})
        gather-fn (array/compile (fn [pools src] (mapv #(if (pools? %) {:k (array/take (:k %) src 0) :v (array/take (:v %) src 0)} {})
                                                       pools)))
        put-fn (array/compile (fn [pools blocks dst]
                                (mapv (fn [{:keys [k v] :as e} b]
                                        (if (pools? e) {:k (gpaged/put-blocks k (:k b) dst) :v (gpaged/put-blocks v (:v b) dst)} e))
                                      pools blocks))
                              {:donate #{0}})
        state (volatile! (cond-> {:table (vec (repeat n-slots nil)) :family fam :block-size block-size :scratch scratch
                                  :alloc (paged/allocator size scratch) :pools ((:paged-pools fam) m size block-size)
                                  :shapes {}
                                  :stats (if continuous?
                                           (cond-> {:forwards 0 :decode-rows 0 :chunk-rows 0 :chunks 0 :padded-rows 0
                                                    :mixed-forwards 0 :copied-blocks 0 :pool-grows 0 :traces 0}
                                             (:paged-states fam) (assoc :lane-forwards 0 :held-forwards 0 :stacked-forwards 0))
                                           (cond-> {:runs 0 :steps 0 :rows 0 :padded-rows 0 :copied-blocks 0
                                                    :pool-grows 0 :forwards 0 :decode-rows 0 :chunk-rows 0 :chunks 0
                                                    :mixed-forwards 0 :traces 0}
                                             (:paged-states fam) (assoc :stacked-steps 0 :held-steps 0
                                                                        :lane-forwards 0 :held-forwards 0 :stacked-forwards 0)))}
                           ;; recurrent layers' states, a row each (Qwen3.5)
                           (:paged-states fam) (assoc :rows (vec (repeat n-slots nil)) :next 0)))
        pool (host/pool "gen.llm array rows-" (host/default-threads))
        ctx (cond-> {:model m :state state :pool pool :n-batch n-batch :min-rows min-rows :gather-fn gather-fn :put-fn put-fn
                     :batch-fn batch-fn :max-chunks max-chunks}
              (not continuous?) (assoc :step-fn step-fn))
        [n ms] (when precompile?
                 (try (precompile! ctx b (precompiled opts (contains? @state :rows) block-size n-slots))
                      (catch Throwable t
                        (.shutdown ^ExecutorService pool)
                        (throw t))))
        info {:engine :gen.array
              :backend (array/backend-name b)
              :n-vocab vocab
              :n-slots n-slots
              :n-ctx n-ctx
              :n-batch n-batch
              :paged (cond-> {:block-size block-size :pool-blocks pool-blocks :min-rows min-rows}
                       continuous? (assoc :continuous? true :max-chunks max-chunks)
                       precompile? (assoc :precompiled n :precompile-ms ms))
              :model (merge {:path (when (string? model) model) :arch (arch-of model)}
                            (select-keys cfg [:layers :hidden :heads :kv-heads :head-dim :dtype :context]))}]
    (if continuous?
      (ContinuousSlots. m b ctx pool (assoc info :batching {:wait wait}) state)
      (PagedSlots. m b ctx pool info state))))

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
     :dtype         with a file: the float dtype it computes in; its KV
                    caches hold it too. Default: the backend's default
                    float, but float16 for a file of quantized weights
                    (Q8_0, Q4_0/1, Q5_0/1, K-quants) on MLX's CUDA
                    (gen.array/device :cuda), whose fast quantized
                    products take half floats only. A loaded model
                    computes in its own dtype (gen.llm.lora, sft and grpo
                    train in float32)
     :max-branches  live branches at most (default 32): memory is the
                    bound, a branch's row costing 2 × layers × KV-heads ×
                    head-dim floats a position of its capacity (Qwen3-0.6B
                    in float32: 112 KiB; Qwen3.5-0.8B, whose attention
                    layers are a quarter: 24 KiB, and 19 MiB of
                    GatedDeltaNet states whatever the capacity)
     :n-ctx         the most tokens a branch holds (default 4096, at most
                    the model's context)
     :n-batch       tokens a forward of a read (default 256)
     :paged?        the mode (see the namespace's docstring): true, the
                    default, the paged mode; false, the static mode,
                    with
       :block       positions a branch's capacity grows by (default 128)
     the paged mode's (Qwen3 and Qwen3.5):
     :block-size    positions a block (8, 16 or 32; default 16)
     :pool-blocks   the pool's blocks at first (default 128), doubled
                    when they run out
     :min-rows      the fewest rows a step computes (default 1): rows
                    are bucketed, at least this
     :precompile?   true (the default): the common shapes of its
                    compiled calls compiled when it opens (Shapes, in
                    the namespace's docstring; on Metal 1.2-2.2 s for
                    the 0.6B and 0.8B in the paged mode, 2-4 s in the
                    continuous mode, float32 the longer), so that serving
                    does not stall on their traces, then the backend's
                    cache of freed buffers given back
                    (array/release-cache!); false for none
     :continuous?   true: concurrent callers' reads batched (continuous
                    batching, over the paged mode; see the namespace's
                    docstring), which serving several clients at once
                    needs; :n-batch then the tokens a forward reads at
                    most (its long reads' chunks), with
       :wait        how long a forward waits for the callers the last
                    one answered, at most, as a fraction of its time
                    (default 0.25; 0 never waits)
       :max-chunks  the most longer reads (prompts) a forward reads a
                    chunk of (default 16; 1 reads one at a time), within
                    :n-batch
   :continuous? with :paged? false is an error, and so is an option the
   chosen mode does not read (:block in the paged mode, :wait in the
   static one)."
  ^java.io.Closeable [opts]
  (let [b array/*backend*
        opts (default-dtype (check-options opts) b)]
    (slots/engine #(open-slots opts b) "gen.array")))

(defn stats
  "Diagnostics: the engine's reads and steps since it was made, {:runs
   :steps :lane-steps (a lane group stepped in place) :stacked-steps
   (rows stacked into a new cache) :stacked-rows :grows (capacities
   grown)}; in the paged mode {:runs :steps :rows :padded-rows
   :copied-blocks (copy-on-write) :pool-grows :traces (calls of a shape
   the engine had not called or compiled when it opened: on MLX, each a
   trace; shapes has them)}, its runs' forwards
   counted as the continuous mode's (their padding in :padded-rows
   too), and for Qwen3.5
   :stacked-steps (steps whose rows' recurrent states were stacked, a
   copy) and :held-steps (read in place from a group others hold); in
   the continuous mode {:forwards :decode-rows :chunk-rows (prefill
   chunks' tokens) :chunks (prefill chunks) :padded-rows :mixed-forwards
   (forwards of decode rows and chunks) :copied-blocks :pool-grows
   :traces},
   for Qwen3.5 with :lane-forwards :held-forwards :stacked-forwards (how
   the forwards with decode rows
   read their states), and gen.llm.slots/schedule the scheduler's
   counts."
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

(defn shapes
  "Diagnostics of the paged modes (:paged?): the shapes of the engine's
   compiled calls, {signature calls}, those compiled when it opened
   (:precompile?) among them, each counting the calls since. A
   signature is what keys a compiled call's traces (on MLX, mlx_compile
   traces each once): [:forward rows logits-rows decode-rows width chunks
   chunk-width targets? decode-states chunk-states pool-blocks] (the
   continuous mode's forwards, the paged mode's runs), [:step rows width
   states pool-blocks] (the paged mode's steps) or [:copy pairs
   pool-blocks] (copy-on-write), the states :fresh (consumed), :held
   (read in place) or nil (none). Its :traces stat counts the calls of a
   shape the engine had not made."
  [e]
  (slots/call e (fn [s]
                  (if (or (instance? PagedSlots s) (instance? ContinuousSlots s))
                    (:shapes @(if (instance? PagedSlots s) (.-state ^PagedSlots s) (.-state ^ContinuousSlots s)))
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
