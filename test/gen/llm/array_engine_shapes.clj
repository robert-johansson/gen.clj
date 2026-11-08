(ns gen.llm.array-engine-shapes
  "The array engine's bucketed shapes (gen.llm.array-engine, genclj-i3fp),
   checked for a model family by its continuous suite: a workload of
   reads on an engine's slots, its forwards mixing decode rows and
   several prompts' chunks (slots-forward, so that which reads share a
   forward is fixed), whose every row and score must be the model's
   forward's and those of the same reads with nothing bucketed (each
   call's rows, tables' widths and chunks' lengths exactly what its reads
   take), within the suite's band; and planted bugs of the padding, each
   failing it."
  (:require [gen.array :as array]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching-contract :as bc]
            [gen.llm.slots :as slots]))

(def ^:private prompts
  [[95 51 72 69 0] [67 65 80 73 84 65 76] [0 73 83] [13 7 99 42 5 61 88 33 70 79 82 87]
   (vec (take 20 (cycle [65 82 68 0 95 51 72])))])

(defn- drive
  "reads (slots-forward's) read to their ends by slots-forward on slots
   s, forward after forward: by read, {:logprobs (after its last token)
   :scores (every target's)}."
  [s reads]
  (loop [todo (vec (map-indexed (fn [i r] (assoc r :i i)) reads))
         out (vec (repeat (count reads) {:scores []}))]
    (if (empty? todo)
      out
      (let [res (slots/slots-forward s (mapv #(dissoc % :i) todo))
            out (reduce (fn [out [{:keys [i]} {:keys [logprobs scores]}]]
                          (cond-> (update-in out [i :scores] into scores)
                            logprobs (assoc-in [i :logprobs] logprobs)))
                        out (map vector todo res))
            todo (into [] (keep (fn [[{:keys [tokens targets pos] :as r} {:keys [read]}]]
                                  (let [n (long read)]
                                    (when (< n (count tokens))
                                      (assoc r :pos (+ (long pos) n) :tokens (subvec tokens n)
                                             :targets (when targets (subvec targets (min n (count targets)))))))))
                       (map vector todo res))]
        (recur todo out)))))

(defn workload
  "The workload on e's slots 10 to 14 (cleared after): three prompts read
   together (one with its targets); then two decode rows beside a new
   prompt's chunk and an extend's (with targets); then four decode rows
   beside a prompt of 20 tokens. {:seen [[context logprobs] …] :scores
   [[context token score] …]}."
  [e]
  (slots/call e
              (fn [s]
                (let [ctx (volatile! {})
                      seen (volatile! [])
                      scores (volatile! [])
                      read! (fn [reads]
                              (doseq [[{:keys [slot tokens targets]} {:keys [logprobs] sc :scores}] (map vector reads (drive s reads))]
                                (let [before (get @ctx slot [])
                                      after (into before tokens)]
                                  (vswap! ctx assoc slot after)
                                  (vswap! seen conj [after logprobs])
                                  (doseq [[j score] (map-indexed vector sc)]
                                    (vswap! scores conj [(into before (subvec tokens 0 (inc j))) (targets j) score])))))
                      at (fn [slot tokens targets?]
                           {:slot slot :pos (count (get @ctx slot [])) :tokens tokens :targets (when targets? (subvec tokens 1))})]
                  (read! [(at 10 (prompts 0) false) (at 11 (prompts 1) true) (at 12 (prompts 2) false)])
                  (read! [(at 10 [13] false) (at 11 [7] false) (at 13 (prompts 3) true) (at 12 [33 70 79 82 87 65] true)])
                  (read! [(at 10 [99] false) (at 11 [42] false) (at 12 [5] false) (at 13 [61] false) (at 14 (prompts 4) false)])
                  (doseq [slot (range 10 15)] (slots/slots-clear s slot))
                  {:seen @seen :scores @scores}))))

(def exact
  "Nothing bucketed: each call's sizes exactly what its reads take."
  {#'ae/bucket (fn ^long [^long n ^long _floor] (max 1 n))})

(defn failures
  "The failures of the workload's results r: against forward-row (a
   function of a context: the model's log-probabilities after it, as
   floats) and against the results with nothing bucketed (exact), each
   within band."
  [r exact-r forward-row band]
  (let [off (fn [what d] (when-not (<= d band) (format "%s: |Δ| %.3g" what d)))]
    (-> []
        (into (keep (fn [[ctx lp]] (off (str "row after " (count ctx) " tokens") (bc/max-diff (forward-row ctx) lp 64))))
              (:seen r))
        (into (keep (fn [[ctx t s]] (off (str "score after " (count ctx) " tokens")
                                         (Math/abs (- (double (aget ^floats (forward-row ctx) (int t))) (double s))))))
              (:scores r))
        (into (keep (fn [[[c1 l1] [c2 l2]]] (if (= c1 c2)
                                              (off (str "row after " (count c1) " tokens, against exact sizes") (bc/max-diff l2 l1 64))
                                              "a different context")))
              (map vector (:seen r) (:seen exact-r)))
        (into (keep (fn [[[_ _ s1] [_ _ s2]]] (off "a score against exact sizes" (Math/abs (- (double s1) (double s2))))))
              (map vector (:scores r) (:scores exact-r)))
        (cond-> (not= (count (:seen r)) (count (:seen exact-r))) (conj "not as many rows as with exact sizes")))))

(defn- full-lens
  "paged-attention reading every column of its tables, the padding's too:
   each length the table's width in positions."
  [f]
  (let [full (fn [tables lens v] (array/full-like lens (* (long (second (array/shape tables))) (long (nth (array/shape v) 3)))))]
    (fn
      ([q k v tables lens] (f q k v tables (full tables lens v)))
      ([q k v tables lens opts] (f q k v tables (full tables lens v) opts)))))

(defn planted
  "Mutations of the padding and the buckets, each failing failures:
   {name (fn [scratch-slots] redefs)}, scratch-slots the engine's scratch
   blocks' positions (the first real slot)."
  []
  {"the buckets rounded down"
   (fn [_] {#'ae/bucket (fn ^long [^long n ^long floor] (max floor (Long/highestOneBit (max 1 n))))})
   "a padding row writing a real block's slot"
   (fn [^long scratch-slots] {#'ae/pad-slot (fn ^long [^long i] (+ i scratch-slots))})
   "a row reading its table's padding columns"
   (fn [_] {#'array/paged-attention (full-lens array/paged-attention)})})
