(ns gen.llm.lm-contract
  "The LM protocol's contract over a Branching engine, as checks any engine
   must pass (the toy engines exactly, llama.cpp within its float noise).
   Each check takes a setup and options and returns a vector of failures
   (strings), empty when it passes; one that throws fails with its message.

   The setup: {:engine a shared engine, :lm (fn [] a new LM over it, which
   the check closes), :fresh (fn [] [engine lm], both new, which the check
   closes)}. :lm's LMs must leave the engine a few branches free (the checks
   read fresh branches beside them). Options: :prompt, :continuation (token
   vectors), :k (one-token reads in a batch), :band (the largest |Δ log p|
   allowed between two computations of one distribution, over the
   reference's :top most likely tokens)."
  (:require [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lm :as lm]))

(set! *warn-on-reflection* true)

(defn run-check
  "The failures of (check setup opts), an exception counted as one."
  [check setup opts]
  (try
    (vec (check setup opts))
    (catch Throwable t
      [(str "threw " (.getName (class t)) ": " (ex-message t) " " (pr-str (ex-data t)))])))

(defn- within [failures what d band]
  (cond-> failures
    (not (<= d band)) (conj (format "%s: |Δ log p| = %.3g > band %.3g" what d (double band)))))

(defn- fresh-read
  "The log-probabilities after tokens by a fresh branch of the engine."
  [engine tokens]
  (let [b (br/open engine tokens)
        lp (br/logprobs b)]
    (br/dispose! b)
    lp))

(defn- same-floats? [^floats a ^floats b] (java.util.Arrays/equals a b))

(defn- new-lm ^java.io.Closeable [setup] ((:lm setup)))

(defn check-a-hit-is-the-read
  "A held context's log-probabilities are those it was read with, bit for
   bit, as a new array each time; a held context costs no read."
  [setup {:keys [prompt]}]
  (with-open [m (new-lm setup)]
    (let [a (lm/next-logprobs m prompt)
          reads (:reads (lm/info m))
          b (lm/next-logprobs m prompt)]
      (cond-> []
        (not (same-floats? a b)) (conj "a hit differs from the read")
        (identical? a b) (conj "a hit is the same array")
        (not= reads (:reads (lm/info m))) (conj "a hit read")))))

(defn check-cached-matches-fresh
  "Contexts one token past a held one, read together, have the
   distributions of fresh reads."
  [{:keys [engine] :as setup} {:keys [prompt k band top] :or {top 64}}]
  (with-open [m (new-lm setup)]
    (let [ts (bc/top-tokens (lm/next-logprobs m prompt) k)
          ctxs (mapv #(conj prompt %) ts)
          rows (lm/batch-next-logprobs m ctxs)]
      (reduce (fn [fs [ctx row]]
                (within fs (str "cached " (peek ctx) " vs fresh") (bc/max-diff (fresh-read engine ctx) row top) band))
              [] (map vector ctxs rows)))))

(defn check-batch-matches-single
  "A batch's log-probabilities are its contexts' read one by one (each by
   a new LM)."
  [setup {:keys [prompt k band top] :or {top 64}}]
  (let [ctxs (with-open [m (new-lm setup)] (mapv #(conj prompt %) (bc/top-tokens (lm/next-logprobs m prompt) k)))
        batched (with-open [m (new-lm setup)] (lm/next-logprobs m prompt) (lm/batch-next-logprobs m ctxs))]
    (reduce (fn [fs [ctx row]]
              (let [single (with-open [m (new-lm setup)] (lm/next-logprobs m prompt) (lm/next-logprobs m ctx))]
                (within fs (str "batched " (peek ctx) " vs alone") (bc/max-diff single row top) band)))
            [] (map vector ctxs batched))))

(defn check-score-matches-steps
  "A continuation's scores are its tokens' log-probabilities read step by
   step (within the band; by another LM); scored after the steps, they are
   the steps' own, bit for bit, read from the trie."
  [setup {:keys [prompt continuation band]}]
  (let [scored (with-open [m (new-lm setup)] (lm/score m prompt continuation))]
    (with-open [m (new-lm setup)]
      (let [steps (mapv (fn [i] (double (aget (lm/next-logprobs m (into prompt (subvec continuation 0 i)))
                                              (int (continuation i)))))
                        (range (count continuation)))
            reads (:reads (lm/info m))
            again (lm/score m prompt continuation)]
        (cond-> (reduce (fn [fs [i s step]] (within fs (str "score vs step, token " i) (Math/abs (- (double s) (double step))) band))
                        [] (map vector (range) scored steps))
          (not= steps again) (conj "the scores along a read path are not the steps'")
          (not= reads (:reads (lm/info m))) (conj "scoring a read path read"))))))

(defn check-books
  "A token-SMC-like loop under tight bounds: the LM stays inside them, its
   branches are the engine's, and the engine never runs its GC backstop."
  [{:keys [engine] :as setup} {:keys [prompt]}]
  (with-open [m (new-lm setup)]
    (let [{:keys [max-branches max-cells]} (lm/info m)
          rng (java.util.Random. 3)
          before (:collections (br/info engine))]
      (loop [step 0 particles (vec (repeat 6 prompt)) failures []]
        (if (= step 12)
          (cond-> failures
            (not= before (:collections (br/info engine))) (conj "the engine ran its GC backstop"))
          (let [rows (lm/batch-next-logprobs m particles)
                grown (mapv (fn [p row] (conj p ((bc/top-tokens row 3) (.nextInt rng 3)))) particles rows)
                {:keys [live cells]} (lm/info m)
                failures (cond-> failures
                           (< (long max-branches) (long live)) (conj (str "step " step ": " live " branches"))
                           (< (long max-cells) (long cells)) (conj (str "step " step ": " cells " cells"))
                           (not= live (:live (br/info engine))) (conj (str "step " step ": the engine's branches")))]
            (recur (inc step) (mapv (fn [_] (grown (.nextInt rng 6))) grown) failures)))))))

(defn check-deterministic
  "The same calls on a new engine and LM give the same results, bit for
   bit: one-token reads in batches, longer reads, and scores."
  [{:keys [fresh]} {:keys [prompt continuation]}]
  (let [run (fn []
              (let [[^java.io.Closeable e ^java.io.Closeable m] (fresh)]
                (try
                  (let [a (lm/next-logprobs m prompt)
                        ts (bc/top-tokens a 4)
                        b (lm/batch-next-logprobs m (mapv #(conj prompt %) ts))
                        c (lm/score m (conj prompt (first ts)) continuation)
                        d (lm/next-logprobs m (into prompt (take 3 continuation)))]
                    (concat [a] b [d c]))
                  (finally (.close m) (.close e)))))
        x (run)
        y (run)]
    (cond-> []
      (not (every? true? (map #(if (vector? %1) (= %1 %2) (same-floats? %1 %2)) x y)))
      (conj "two runs differ"))))

(def checks
  "Every check, by name."
  {:a-hit-is-the-read check-a-hit-is-the-read
   :cached-matches-fresh check-cached-matches-fresh
   :batch-matches-single check-batch-matches-single
   :score-matches-steps check-score-matches-steps
   :books check-books
   :deterministic check-deterministic})
