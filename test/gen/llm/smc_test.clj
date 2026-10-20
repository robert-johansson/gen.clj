(ns gen.llm.smc-test
  "Token SMC (gen.llm.smc) on the toy engine with a vocabulary of five
   tokens, token 4 ending a completion, so that every completion of up to
   four tokens can be enumerated (625 at most) and Z = Σ p(x)·exp(complete(x))
   computed exactly.

   - Unbiasedness in exp space: the mean of 40 runs' Ẑ is within 5
     standard errors of Z, for each proposal (the LM's rows, the masked
     rows, a caller's uniform), a constraint and a soft twist, resampling at
     every step, adaptively or never; and so is the unnormalized estimate of
     a test function's integral.
   - Exact identities: no potential gives Ẑ = 1 (the toy's ledger is its
     rows), max-tokens 0 gives exp(complete([])), a masked constraint's
     particles all satisfy it.
   - One read a step; particles convert to the completion's traces, whose
     scores are the particles' log p; runs are pure functions of their key;
     no GC backstop; misuse is loud.

   Mutations of gen.llm.smc checked (2026-10-21, each applied, the suite
   run, and reverted): see docs/research/llm-token-smc.md, Tests."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng]))

(def ^:private n-vocab 5)
(def ^:private end 4)

(def ^:private tokenizer
  {:n-vocab n-vocab :bos nil :eos end :eog #{end} :add-bos? false
   :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))})

(def ^:private engine (toy/engine {:n-vocab n-vocab :max-branches 128 :n-ctx 64}))
(def ^:private m (lm/lm engine tokenizer {:max-cells 8192}))
(def ^:private completion (token/completion m))
(def ^:private prompt [0 1])
(def ^:private max-tokens 4)
(def ^:private args [prompt max-tokens])

(defn- row ^floats [ctx] (toy/toy-logprobs n-vocab ctx))

(defn- completions
  "Every completion of [prompt max-tokens] with its log p: [tokens log-p]."
  []
  (letfn [(walk [tokens ^double lp]
            (if (or (= max-tokens (count tokens)) (and (seq tokens) (= end (peek tokens))))
              [[tokens lp]]
              (let [^floats r (row (into prompt tokens))]
                (mapcat (fn [t] (walk (conj tokens t) (+ lp (aget r (int t))))) (range n-vocab)))))]
    (walk [] 0.0)))

(defn- body
  "The tokens a potential weighs as a whole sequence: those before an end."
  [tokens]
  (if (and (seq tokens) (= end (peek tokens))) (pop tokens) tokens))

(defn- exact-z
  "Σ p(x)·exp(complete(x))·f(x), by enumeration."
  ([p] (exact-z p (constantly 1.0)))
  ([p f] (reduce + (for [[x lp] (completions)]
                     (* (Math/exp (+ lp (potential/complete p (body x)))) (double (f x)))))))

(def ^:private no-repeats
  "A constraint: no token the same as the one before, no 3 at all, and at
   least two tokens before the end (so that ending and drawing token 4 as a
   token weigh differently)."
  (let [ok? (fn [ctx] (and (not-any? #{3} ctx) (not-any? (fn [[a b]] (= a b)) (partition 2 1 ctx))))]
    (potential/potential n-vocab {:prefix #(if (ok? %) 0.0 ##-Inf)
                                  :complete #(if (and (ok? %) (<= 2 (count %))) 0.0 ##-Inf)})))

(def ^:private soft
  "A soft twist: each 1 costs 0.7; a whole sequence costs 0.5 unless its
   last token is 2 (so that a sequence stopped at max-tokens weighs other
   than its prefix); and 0.3 for any sequence (so that the empty context's
   prefix weight counts)."
  (let [ones #(+ 0.3 (* -0.7 (count (filter #{1} %))))]
    (potential/potential n-vocab {:prefix ones :complete #(- (ones %) (if (= 2 (peek %)) 0.0 0.5))})))

(def ^:private nothing
  "A potential no sequence satisfies."
  (potential/potential n-vocab {:prefix (fn [c] (if (seq c) ##-Inf 0.0)) :complete (constantly ##-Inf)}))

(defn- uniform-proposal [_context ^floats row]
  (float-array (alength row) (float (- (Math/log (alength row))))))

(defn- mean ^double [xs] (/ (reduce + 0.0 xs) (count xs)))
(defn- sd ^double [xs] (let [mu (mean xs)] (Math/sqrt (/ (reduce + (map #(Math/pow (- % mu) 2) xs)) (dec (count xs))))))

(defn- within-5-se? [zs ^double z]
  (<= (Math/abs (- (mean zs) z)) (* 5 (/ (sd zs) (Math/sqrt (count zs))))))

(defn- estimates
  "exp(log Ẑ) of runs runs of n particles."
  [opts n runs]
  (mapv #(Math/exp (smc/log-ml-estimate (smc/run completion args n (assoc opts :key (rng/key %))))) (range runs)))

;; ---------------------------------------------------------------------------
;; Unbiasedness
;; ---------------------------------------------------------------------------

(deftest the-estimate-is-unbiased-in-exp-space
  (doseq [[pname p] [[:constraint no-repeats] [:soft soft]]
          proposal [:lm :masked uniform-proposal]
          [rname threshold] [[:always 51.0] [:adaptive nil] [:never 0.0]]]
    (testing [pname (if (fn? proposal) :uniform proposal) rname]
      (let [zs (estimates {:potential p :proposal proposal :ess-threshold threshold} 50 40)]
        (is (within-5-se? zs (exact-z p)) (str "mean " (mean zs) " exact " (exact-z p)))))))

(deftest a-test-functions-integral-is-unbiased
  ;; exp(log-ml) · (1/n) Σ exp(w_i) f(x_i) estimates Σ p(x)·exp(complete(x))·f(x)
  (let [f #(if (some #{2} %) 1.0 0.0)
        zs (mapv (fn [seed]
                   (let [s (smc/run completion args 50 {:potential soft :key (rng/key seed)})]
                     (* (Math/exp (:log-ml s))
                        (mean (map (fn [x lw] (* (Math/exp lw) (f x))) (smc/particles s) (:log-weights s))))))
                 (range 40))]
    (is (within-5-se? zs (exact-z soft f)))))

(deftest particles-draw-independently
  ;; the first tokens of 400 particles, by the prompt's row within 5 standard errors
  (let [k 400
        counts (frequencies (map first (smc/particles (smc/step (smc/initialize completion args k {:key (rng/key 9)})))))
        ^floats r (row prompt)]
    (doseq [t (range n-vocab)]
      (let [p (Math/exp (aget r t))]
        (is (<= (Math/abs (- (get counts t 0) (* k p))) (* 5 (Math/sqrt (* k p (- 1 p))))) (str "token " t))))))

;; ---------------------------------------------------------------------------
;; Exact identities
;; ---------------------------------------------------------------------------

(deftest without-a-potential-the-bootstrap-weighs-nothing
  ;; log p - log q is 0 when the LM's ledger is the row drawn from (the toy)
  (let [s (smc/run completion args 20 {:key (rng/key 1)})]
    (is (every? zero? (:log-weights s)))
    (is (= 0.0 (smc/log-ml-estimate s)))))

(deftest no-tokens-weigh-the-empty-sequence
  (let [s (smc/run completion [prompt 0] 5 {:potential soft :key (rng/key 1)})]
    (is (every? #(= [] %) (smc/particles s)))
    (is (laws/close? (potential/complete soft []) (smc/log-ml-estimate s)))))

(deftest a-masked-constraint-is-always-satisfied
  (let [opts {:potential no-repeats :proposal :masked :key (rng/key 2)}
        s (smc/run completion args 40 opts)]
    (is (every? #(Double/isFinite (potential/complete no-repeats (body %))) (smc/particles s)))
    (is (every? Double/isFinite (:log-weights s)))
    (is (every? Double/isFinite (:log-weights (smc/step (smc/initialize completion args 200 opts))))
        "no particle ends at the first step, which the constraint forbids")))

;; ---------------------------------------------------------------------------
;; Reads, traces, keys
;; ---------------------------------------------------------------------------

(deftest a-step-is-one-read
  ;; on an LM of its own: the first step reads the prompt (its first token,
  ;; then the rest), each later one the particles' new contexts in one advance
  (with-open [^java.io.Closeable e (toy/engine {:n-vocab n-vocab :max-branches 64 :n-ctx 64})
              fresh (lm/lm e tokenizer)]
    (let [gf (token/completion fresh)
          reads #(:reads (lm/info fresh))
          s1 (smc/step (smc/initialize gf [prompt 6] 30 {:potential soft :key (rng/key 3)}))
          _ (is (= 2 (reads)))
          s2 (smc/step (smc/resample s1))
          _ (is (= 3 (reads)))
          s3 (smc/step s2)]
      (is (= 4 (reads)) "each step one read")
      (is (some #(not (:done? %)) (:particles s3)) "the particles still grow"))))

(deftest particles-convert-to-the-completions-traces
  (let [s (smc/run completion args 20 {:potential soft :key (rng/key 4)})
        trs (smc/traces s)]
    (is (= (smc/particles s) (mapv :retval trs)))
    (is (= (mapv :log-p (:particles s)) (mapv :score trs)) "the particles' log p are the traces' scores")
    (is (every? #(laws/close? (:score %) (:weight (gfi/assess completion args (:choices %)))) trs))
    (is (= :no-reason (try (smc/traces (smc/step (smc/initialize completion args 3 {:key (rng/key 5)}))) :no-error
                          (catch clojure.lang.ExceptionInfo _ :no-reason)))
        "unfinished particles have no traces yet")))

(deftest runs-are-pure-functions-of-the-key
  (let [run #(smc/run completion args 20 {:potential no-repeats :key (rng/key 6)})]
    (is (= (smc/particles (run)) (smc/particles (run))))
    (is (= (:log-weights (run)) (:log-weights (run))))))

(deftest inference-never-reaches-the-backstop
  (dotimes [seed 5] (smc/run completion [prompt 8] 60 {:potential soft :key (rng/key seed)}))
  (is (zero? (:collections (br/info engine))))
  (is (<= (:live (lm/info m)) (:max-branches (lm/info m)))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (doseq [[what f] [["no completion" #(smc/initialize m args 4)]
                    ["no particles" #(smc/initialize completion args 0)]
                    ["bad arguments" #(smc/initialize completion [prompt] 4)]
                    ["an empty prompt" #(smc/initialize completion [[] 3] 4)]
                    ["a masked proposal without a potential" #(smc/initialize completion args 4 {:proposal :masked})]
                    ["an unknown proposal" #(smc/initialize completion args 4 {:proposal :best})]
                    ["a potential of another vocabulary"
                     #(smc/step (smc/initialize completion args 4 {:potential (potential/potential 3 {:prefix (constantly 0.0)
                                                                                                      :complete (constantly 0.0)})}))]
                    ["a proposal that returns no row" #(smc/step (smc/initialize completion args 4 {:proposal (constantly [0.0])}))]
                    ["resampling particles all of weight 0"
                     #(smc/resample (smc/step (smc/initialize completion args 4 {:potential nothing})))]]]
    (is (thrown? clojure.lang.ExceptionInfo (f)) what)))

(deftest particles-all-of-weight-0-estimate-z-as-0
  ;; the estimator's value, not an error: refusing it would bias Ẑ upwards
  (let [s (smc/run completion args 4 {:potential nothing :key (rng/key 1)})]
    (is (smc/done? s))
    (is (= ##-Inf (smc/log-ml-estimate s)))
    (is (= 0.0 (smc/ess s)))
    (is (every? nil? (smc/traces s)) "dead particles have no traces")
    (is (= :impossible (bc/reason #(smc/resample s))) "and cannot be resampled")))
