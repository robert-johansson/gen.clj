(ns gen.llm.array-engine
  "A Branching engine (gen.llm.branching) over gen.array: a model's forward
   passes (gen.llm.models.qwen3) on any backend, MLX's Metal and CPU or
   the JVM's for tiny models, as gen.llm.slots's Slots
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

   A branch's state is a row of a static KV cache (qwen3/static-cache), a
   value: a slot holds a group, a static cache of one or more rows, and
   its row there.
   - A copy shares the group (a fork is free), and a clear drops the
     slot's hold on it; a group no slot holds is garbage, freed by the GC
     (the backend's arrays, by their phantom references).
   - A run (open, extend, score's scratch read) reads its tokens into a
     cache of the slot's row alone, in forwards of :n-batch tokens, its
     row copied out of its group first unless the slot holds the group
     alone. With targets, each one's score is read on the device: the
     log-softmax at its position, in the model's dtype, taken at the
     target.
   - A step is one decode of the slots' rows, by qwen3's compiled step
     over static caches, consuming its cache (compile-step's :donate?): a
     group whose rows are exactly the step's slots, held by them alone (a
     lane group: lockstep particles), is stepped in place, its rows in
     its own order; any other step stacks its rows into a new cache first
     (qwen3/select-rows of each group, concat-rows), one copy, and steps
     that. The stepped rows form a new group.
   - Capacity grows by blocks (:block positions, at most :n-ctx), with
     zeros (qwen3/grow): a compiled step traces once per batch size and
     capacity.
   - Logits stay on the device until the step's or run's end, where one
     evaluation reads them (array/->floats): each row is normalized on the
     host in double (log-softmax, loud when a logit is not finite), on a
     small pool of the engine's own (gen.llm.host's), as the llama.cpp
     engine normalizes its rows.
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
   head-dim floats a position of their capacity), so keep its
   :max-branches to what a run needs.

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
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.slots :as slots])
  (:import [java.util.concurrent ExecutorService]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; Rows on the host
;; ---------------------------------------------------------------------------

(defn- row-max
  "The largest of the n floats of xs from off."
  ^double [^floats xs ^long off ^long n]
  (loop [j 0 m Double/NEGATIVE_INFINITY]
    (if (c/< j n) (recur (inc j) (Math/max m (double (aget xs (c/+ off j))))) m)))

(defn- row-sum-exp
  "Σ exp(x - m) over the n floats of xs from off, in double."
  ^double [^floats xs ^long off ^long n ^double m]
  (loop [j 0 s 0.0]
    (if (c/< j n) (recur (inc j) (c/+ s (Math/exp (c/- (double (aget xs (c/+ off j))) m)))) s)))

(defn- normalized
  "Row i of logits (n floats each, row after row) as log-probabilities
   normalized in double, a new float array; loud when a logit is NaN or
   +Inf."
  ^floats [^floats logits ^long i ^long n what]
  (let [off (c/* i n)
        m (row-max logits off n)
        z (if (Double/isFinite m) (c/+ m (Math/log (row-sum-exp logits off n m))) m)]
    (when-not (Double/isFinite z)
      (throw (ex-info (str what ": the model's logits are not finite") {:reason :decode-failed :log-normalizer z})))
    (let [out (float-array n)]
      (dotimes [j n] (aset out j (float (c/- (double (aget logits (c/+ off j))) z))))
      out)))

(defn- host-rows
  "The rows of logits [B 1 vocab] (or [B vocab]), read at once and each
   normalized on pool: a vector of B float arrays."
  [pool logits what]
  (let [s (array/shape logits)
        n (long (peek s))
        xs (array/->floats logits)
        b (quot (alength xs) n)]
    (host/in-order pool #(normalized xs % n what) (vec (range b)))))

;; ---------------------------------------------------------------------------
;; The books: a slot's {:group g :row r}, a group {:id :cache}
;; ---------------------------------------------------------------------------

;; state, a volatile written on the engine's thread:
;;   :table  a vector, by slot, of nil (empty) or {:group {:id :cache} :row r}
;;   :next   the next group's id
;;   :stats  {:runs :steps :lane-steps :stacked-steps :stacked-rows :grows}

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
    (long (nth (qwen3/offsets (:cache group)) row))
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
    (if (and (c/== 1 (count (qwen3/offsets cache))) (= #{slot} (holders state (:id group))))
      cache
      (qwen3/select-rows cache [row]))))

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
                (qwen3/static-cache model 1 cap)
                (let [c (row-cache state slot)]
                  (when (c/< (long (:capacity c)) cap) (tally! state :grows 1))
                  (qwen3/grow c (max cap (long (:capacity c))))))
        step-size (long n-batch)]
    (loop [from 0 cache start scores []]
      (let [end (min n (c/+ from step-size))
            last? (c/== end n)
            scored (max 0 (c/- (min end k) from))
            [logits cache] (qwen3/step model cache [(subvec tokens from end)] {:logits (if (pos? scored) :all :last)})
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
                    n (count (qwen3/offsets cache))]
                (if (and (c/== n (count is)) (apply distinct? rows) (lane? state id (map slots is)))
                  ;; the group itself: its rows in its order
                  (let [by-row (zipmap rows is)]
                    [(conj parts {:cache cache :lane? true}) (into order (map by-row) (range n))])
                  [(conj parts {:cache (qwen3/select-rows cache rows) :lane? false}) (into order is)])))
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
                      (qwen3/grow cache cap))
                    parts)
        cache (if (c/== 1 (count grown)) (first grown) (qwen3/concat-rows grown))
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
;; Opening
;; ---------------------------------------------------------------------------

(def ^:private defaults {:max-branches 32 :n-ctx 4096 :n-batch 256 :block 128})

(def ^:private option-checks
  {:model #(or (and (string? %) (.isFile (java.io.File. ^String %)))
               (and (map? %) (contains? % :config) (contains? % :weights)))
   :dtype #{:float16 :bfloat16 :float32 :float64}
   :max-branches pos-int? :n-ctx pos-int? :n-batch pos-int? :block pos-int?})

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
               (ex-info (str "array-engine/engine: :model is a GGUF file or a model (qwen3/load's), not " (pr-str model))
                        {:reason :no-model :model (if (string? model) model (type model))})
               (ex-info (str "array-engine/engine: bad " k ": " (pr-str (get opts k))) {:reason :invalid-option k (get opts k)}))))
    (when (and (map? model) (contains? opts :dtype))
      (throw (ex-info "array-engine/engine: :dtype is for a GGUF file the engine loads; a model computes in its own"
                      {:reason :invalid-option :dtype (:dtype opts)})))
    opts))

(defn- model-backend
  "The backend m's weights live on."
  [m]
  (array/backend (first (filter array/backend-array? (array/tree-leaves (:weights m))))))

(defn- open-slots
  "ArraySlots for checked opts, on the engine's thread, on backend b: the
   model loaded (a path), its step compiled."
  [{:keys [model dtype max-branches n-ctx n-batch block]} b]
  (array/with-backend b
    (let [m (if (string? model) (qwen3/load model (when dtype {:dtype dtype})) model)
          b (model-backend m)
          {:keys [vocab context] :as cfg} (:config m)
          n-ctx (long (if context (min (long n-ctx) (long context)) n-ctx))
          step-fn (array/with-backend b (qwen3/compile-step m {:logits :last :donate? true}))
          n-slots (inc (long max-branches))
          info {:engine :gen.array
                :backend (array/backend-name b)
                :n-vocab vocab
                :n-slots n-slots
                :n-ctx n-ctx
                :n-batch n-batch
                :block block
                :model (merge {:path (when (string? model) model)} (select-keys cfg [:layers :hidden :heads :kv-heads :head-dim :dtype :context]))}]
      (ArraySlots. m b step-fn (host/pool "gen.llm array rows-" (host/default-threads)) info
                   (volatile! {:table (vec (repeat n-slots nil)) :next 0
                               :stats {:runs 0 :steps 0 :lane-steps 0 :stacked-steps 0 :stacked-rows 0 :grows 0}})
                   (long n-batch) (long block) n-ctx))))

(defn engine
  "A Branching engine over gen.array (java.io.Closeable; close it to free
   its model when it loaded one), on a thread of its own, its model on
   gen.array's *backend* (as it is when engine is called). opts (any other
   is an error):
     :model         a Qwen3 GGUF file, which the engine loads
                    (gen.llm.models.qwen3/load), or a model loaded already,
                    a LoRA adapter's among them (gen.llm.lora/adapt; its
                    dropout off, without a :key) (required)
     :dtype         with a file: the float dtype it computes in (default
                    the backend's default float); its KV caches hold it too
     :max-branches  live branches at most (default 32): memory is the
                    bound, a branch's row costing 2 × layers × KV-heads ×
                    head-dim floats a position of its capacity (Qwen3-0.6B
                    in float32: 112 KiB)
     :n-ctx         the most tokens a branch holds (default 4096, at most
                    the model's context)
     :n-batch       tokens a forward of a read (default 256)
     :block         positions a branch's capacity grows by (default 128)"
  ^java.io.Closeable [opts]
  (let [opts (check-options opts)
        b array/*backend*]
    (slots/engine #(open-slots opts b) "gen.array")))

(defn stats
  "Diagnostics: the engine's reads and steps since it was made, {:runs
   :steps :lane-steps (a lane group stepped in place) :stacked-steps
   (rows stacked into a new cache) :stacked-rows :grows (capacities
   grown)}."
  [e]
  (slots/call e (fn [s]
                  (if (instance? ArraySlots s)
                    (:stats @(.-state ^ArraySlots s))
                    (throw (ex-info "Not a gen.array engine" {:reason :not-an-array-engine :engine e}))))))

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
