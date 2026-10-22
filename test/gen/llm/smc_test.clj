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
     no GC backstop; misuse is loud: the arguments by the completion's own
     check, for every key, and a constrained completion refused, its target
     reached by SMC's :potential and :masked.
   - The host pool: runs on pools of 1, 2 and 8 threads are the calling
     thread's bit for bit; no LM, potential or proposal-function call runs
     on the pool; a task's error and the caller's bindings reach it as on
     the calling thread; the default pool is shared.

   Mutations of gen.llm.smc checked (2026-10-21, each applied, the suite
   run, and reverted): see docs/research/llm-token-smc.md, Tests."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.constrained :as constrained]
            [gen.llm.grammar :as grammar]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng])
  (:import [java.util.concurrent ExecutorService ThreadPoolExecutor]))

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
    (is (= :not-done (bc/reason #(smc/traces (smc/step (smc/initialize completion args 3 {:key (rng/key 5)})))))
        "unfinished particles have no traces yet")))

(deftest runs-are-pure-functions-of-the-key
  (let [run #(smc/run completion args 20 {:potential no-repeats :key (rng/key 6)})]
    (is (= (smc/particles (run)) (smc/particles (run))))
    (is (= (:log-weights (run)) (:log-weights (run))))))

;; ---------------------------------------------------------------------------
;; The host pool
;; ---------------------------------------------------------------------------

(deftest the-host-pool-changes-nothing
  ;; the proposal rows and draws on pools of 1, 2 and 8 threads give the
  ;; calling thread's run bit for bit: the particles, the weights, the
  ;; estimate; for each kind of proposal
  (doseq [opts [{:potential soft} {:potential soft :proposal :masked} {:potential no-repeats :proposal :masked}
                {:potential soft :proposal uniform-proposal}]]
    (let [run (fn [pool] (smc/run completion [prompt 8] 40 (assoc opts :key (rng/key 7) :host-pool pool)))
          alone (run false)]
      (doseq [n [1 2 8]]
        (with-open [^ExecutorService pool (smc/host-pool n)]
          (let [r (run pool)]
            (is (= (smc/particles alone) (smc/particles r)) (str n " threads"))
            (is (= (:log-weights alone) (:log-weights r)) (str n " threads"))
            (is (= (smc/log-ml-estimate alone) (smc/log-ml-estimate r)) (str n " threads"))
            (is (pos? (.getCompletedTaskCount ^ThreadPoolExecutor pool)) "the pool ran the draws")))))))

(deftest no-lm-or-potential-call-runs-on-the-pool
  ;; a recording LM and potential, and a recording proposal function: every
  ;; call on the calling thread
  (let [caller (Thread/currentThread)
        threads (atom #{})
        note! (fn [] (swap! threads conj (Thread/currentThread)))
        recording (reify lm/LM
                    (-tokenizer [_] (lm/-tokenizer m))
                    (-next-logprobs [_ contexts] (note!) (lm/-next-logprobs m contexts))
                    (-score [_ context continuation] (note!) (lm/-score m context continuation))
                    (-info [_] (lm/-info m)))
        noted (potential/potential n-vocab {:prefix #(do (note!) (potential/prefix soft %))
                                            :complete #(do (note!) (potential/complete soft %))
                                            :logw-next #(do (note!) (potential/logw-next soft %))})]
    (with-open [^ExecutorService pool (smc/host-pool 4)]
      (doseq [proposal [:masked :lm (fn [context row] (note!) (uniform-proposal context row))]]
        (smc/run (token/completion recording) [prompt 8] 40
                 {:potential noted :proposal proposal :key (rng/key 8) :host-pool pool}))
      (is (pos? (.getCompletedTaskCount ^ThreadPoolExecutor pool))))
    (is (= #{caller} @threads))))

(deftest a-pool-task-s-error-is-raised-as-itself
  ;; a twist of NaN makes the masked rows NaN, which the draw, on the pool,
  ;; finds loud
  (let [nan (potential/potential n-vocab {:prefix (constantly 0.0) :complete (constantly 0.0)
                                          :logw-next (fn [_] (float-array (inc n-vocab) Float/NaN))})]
    (with-open [^ExecutorService pool (smc/host-pool 2)]
      (is (= :invalid-row (bc/reason #(smc/run completion args 8 {:potential nan :proposal :masked :host-pool pool
                                                                  :key (rng/key 9)})))))))

(deftest a-pool-task-sees-the-calling-thread-s-bindings
  ;; under a batch tracer a token draw is the batched-branch error (one
  ;; particle at a time), on the pool as on the calling thread
  (let [s (smc/initialize completion args 8 {:key (rng/key 10) :host-pool false})
        branch? (fn [f] (try (f) false
                             (catch clojure.lang.ExceptionInfo e (boolean (::array/batched-branch (ex-data e))))))]
    (is (branch? #(batch/batched [2] (smc/step s))))
    (with-open [^ExecutorService pool (smc/host-pool 2)]
      (is (branch? #(batch/batched [2] (smc/step (assoc s :host-pool pool))))))))

(deftest the-pool-s-size-is-a-positive-integer-or-loud
  (let [threads #'smc/host-threads]
    (is (= 4 (threads nil 10)) "half the processors")
    (is (= 4 (threads nil 32)) "up to 4")
    (is (= 1 (threads nil 1)))
    (is (= 3 (threads "3" 10)))
    (is (= 1 (threads "1" 10)) "the off switch")
    (doseq [bad ["four" "4 " "" "0" "-2" "2.5"]]
      (is (= :invalid-option (bc/reason #(threads bad 10))) (pr-str bad)))))

(deftest the-default-pool-is-shared
  (let [pools (map #(:host-pool (smc/initialize completion args 4 {:key (rng/key %)})) (range 2))]
    (is (or (every? false? pools) (and (instance? ExecutorService (first pools)) (apply identical? pools)))
        "one pool for every run, or the calling thread on one processor")))

(deftest inference-never-reaches-the-backstop
  (dotimes [seed 5] (smc/run completion [prompt 8] 60 {:potential soft :key (rng/key seed)}))
  (is (zero? (:collections (br/info engine))))
  (is (<= (:live (lm/info m)) (:max-branches (lm/info m)))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (doseq [[what f reason]
          [["no completion" #(smc/initialize m args 4) :invalid-model]
           ["no particles" #(smc/initialize completion args 0) :invalid-args]
           ["bad arguments" #(smc/initialize completion [prompt] 4) :invalid-args]
           ["an empty prompt" #(smc/initialize completion [[] 3] 4) :no-tokens]
           ["a masked proposal without a potential" #(smc/initialize completion args 4 {:proposal :masked}) :invalid-option]
           ["an unknown proposal" #(smc/initialize completion args 4 {:proposal :best}) :invalid-option]
           ["a potential of another vocabulary"
            #(smc/step (smc/initialize completion args 4 {:potential (potential/potential 3 {:prefix (constantly 0.0)
                                                                                             :complete (constantly 0.0)})}))
            :invalid-potential]
           ["a proposal that returns no row" #(smc/step (smc/initialize completion args 4 {:proposal (constantly [0.0])}))
            :invalid-row]
           ["a pool of no threads" #(smc/host-pool 0) :invalid-args]
           ["a host pool that is no executor" #(smc/initialize completion args 4 {:host-pool 3}) :invalid-option]
           ["resampling particles all of weight 0"
            #(smc/resample (smc/step (smc/initialize completion args 4 {:potential nothing}))) :impossible]]]
    (is (= reason (bc/reason f)) what)))

;; The completion's arguments are checked as the completion checks them
;; (token/check-args), before any draw: the same error for every key. With
;; a copy of its own, SMC once accepted a prompt and max-tokens past the
;; LM's :limit, which then failed or not by the key (the post-M5 review's
;; P4-05), and a prompt token outside the vocabulary.
(deftest the-arguments-are-checked-as-the-completion-checks-them
  (let [small (token/completion (lm/lm (toy/engine {:n-vocab n-vocab :max-branches 64 :n-ctx 64}) tokenizer {:max-cells 8}))]
    (is (= 8 (:limit small)))
    (doseq [k (range 12)]
      (is (= :context-full (bc/reason #(smc/run small [prompt 10] 4 {:key (rng/key k)}))) (str "past the limit, key " k))
      (is (= :invalid-token (bc/reason #(smc/run completion [[0 7] 3] 4 {:key (rng/key k)})))
          (str "a token outside the vocabulary, key " k))))
  (is (= :invalid-token (bc/reason #(smc/initialize completion [[0 :a] 3] 4))))
  (is (= :invalid-token (bc/reason #(smc/initialize completion [[0 -1] 3] 4)))))

;; A completion with a potential of its own is refused (DESIGN §9, "A
;; constrained completion is refused"; the review's P4-01): SMC draws and
;; weighs by the LM's rows, so its particles broke the potential and
;; smc/traces scored them -Inf. The target such a pairing means, the LM's
;; completions held to the potential, is the constrained completion's
;; conditional weights' (gen.llm.constrained) and SMC's with the potential
;; as :potential and the :masked proposal: both estimate its Z.
(deftest a-constrained-completion-is-refused
  (let [constrained (token/completion m {:potential no-repeats})]
    (doseq [f [#(smc/initialize constrained args 4) #(smc/run constrained args 4 {:key (rng/key 1)})]]
      (is (= :constrained-completion (bc/reason f))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"SMC's :potential.*:masked" (smc/initialize constrained args 4))
        "naming the way that is")
    (let [z (exact-z no-repeats)
          weights (mapv #(Math/exp (constrained/conditional-weight (gfi/simulate constrained args (rng/key %)))) (range 2000))
          zs (estimates {:potential no-repeats :proposal :masked} 50 40)]
      (is (within-5-se? weights z) (str "the conditional weights' mean " (mean weights) ", Z " z))
      (is (within-5-se? zs z) (str "SMC's mean " (mean zs) ", Z " z)))))

;; SMC's :potential must read contexts with the completion's end tokens, if
;; it declares any (DESIGN §9, "A potential declares its end tokens"; the
;; review's P4-02 and I4-02). DESIGN's example: a newline that ends the
;; completion, and a grammar that needs one in its text. SMC once took the
;; pair and returned log Ẑ = -Inf for every run, silently; the matched
;; grammar, whose text stops where the newline ends it, estimates Z.
(deftest the-potential-s-end-tokens-are-the-completion-s
  (let [nl (bt/tokenizer ["a" "b" "\n"])
        m (lm/lm (toy/engine {:n-vocab (:n-vocab nl) :max-branches 64 :n-ctx 128}) nl {:max-cells 5000})
        lines (token/completion m {:end-tokens #{2}})
        args [[(:bos nl)] 6]
        g (grammar/grammar m "[ab]{3}\n" {:end-tokens #{}})]
    (doseq [p [g (potential/product g)]]
      (is (= :mismatched-end-tokens (bc/reason #(smc/initialize lines args 8 {:potential p})))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"^smc: " (smc/initialize lines args 8 {:potential g})))
    (let [s (smc/run lines args 32 {:potential (grammar/grammar m "[ab]{3}" {:end-tokens #{2}}) :proposal :masked
                                    :key (rng/key 1)})]
      (is (Double/isFinite (smc/log-ml-estimate s)))
      (is (every? #(and (= 4 (count %)) (= 2 (peek %)) (every? #{0 1} (pop %))) (smc/particles s))
          "three letters, then the end"))))

(deftest particles-all-of-weight-0-estimate-z-as-0
  ;; the estimator's value, not an error: refusing it would bias Ẑ upwards
  (let [s (smc/run completion args 4 {:potential nothing :key (rng/key 1)})]
    (is (smc/done? s))
    (is (= ##-Inf (smc/log-ml-estimate s)))
    (is (= 0.0 (smc/ess s)))
    (is (every? nil? (smc/traces s)) "dead particles have no traces")
    (is (= :impossible (bc/reason #(smc/resample s))) "and cannot be resampled")))
