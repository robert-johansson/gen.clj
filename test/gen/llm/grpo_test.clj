(ns gen.llm.grpo-test
  "GRPO and its variants (gen.llm.grpo, experimental) on the tiny random
   Qwen3 of test/fixtures/gguf (vocabulary 128, hidden 64, 3 blocks), F32,
   on the backend under test and its gradients, rollouts drawn by
   gen.llm.array-engine; the tokenizer a byte tokenizer of the chat
   template's pieces and printable ASCII, padded to the model's 128
   tokens, <|im_end|> its end of generation:
   1. rollouts: each completion's old log-probabilities are the LM's
      scores of its tokens (the end token's last), at the sampling
      temperature; a function of the key;
   2. rewards: the built-in checkers, and a reward that is not finite is
      loud;
   3. advantages by hand, a flat group's exactly 0;
   4. batches: a row is the prompt, the completion, then padding, the
      mask exactly the completion's tokens, the old log-probabilities
      where they were drawn; weights and divisors by loss type;
   5. the loss of each loss type, against a computation by hand from the
      forward's log-softmax, the rollouts drawn by another adapter (so the
      ratios move and clip), β > 0 against the base;
   6. at a round's first step every ratio is 1 but for float round-off,
      at temperature 1 and 0.7;
   7. a step moves the policy toward the completions of positive
      advantage, and toward the reference when β > 0 and the advantages
      are 0;
   8. accumulated microbatches step as one batch (grpo, dapo, dr-grpo);
      bnpo is dapo with one microbatch;
   9. training raises a reward the policy can reach;
   10. a run saved at round 2 and resumed is the run to round 4, bit for
       bit;
   11. a loss that is not finite is loud; misuse is loud.
   Planted bugs (2026-11-07, the JVM backend, with-redefs-fn), each
   failing the tests named, in place: the ratio against the new
   log-probabilities, always 1 (5); the advantages' sign flipped (7, 9);
   dapo's divisor the completions, mlx-node's (5); the KL's sign (5, 7);
   the mask a position late (4, 5, 6)."
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.learning :as learn]
            [gen.llm.array-engine :as ae]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.grpo :as grpo]
            [gen.llm.lm :as lm]
            [gen.llm.lora :as lora]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.sft :as sft]
            [gen.random :as rng]
            [gen.test-support :as ts])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private model (delay (qwen3/load "test/fixtures/gguf/qwen3-tiny-f32.gguf")))

(def ^:private tokenizer
  "The chat template's pieces, printable ASCII and 26 fillers: tokens 0
   to 125, BOS 126, EOS 127; <|im_end|> (1) ends generation."
  (assoc (bt/tokenizer (concat ["<|im_start|>" "<|im_end|>" "<think>" "</think>" "\n"] (map #(str (char %)) (range 32 127))
                               (map #(str "<x" % ">") (range 26))))
         :eog #{1}))

(def ^:private prompts (mapv (fn [i] {:prompt (str "Say " i) :answer (str i)}) (range 16)))

(defn- lower
  "A reward the tiny model can reach: the fraction of a completion's
   characters that are lower-case letters."
  [_ c]
  (if (empty? c) 0.0 (/ (double (count (filter #(Character/isLowerCase (char %)) c))) (count c))))

(defn- adapter
  "An adapter of the tiny model, rank 4 on q, v and down of every layer,
   its B drawn (scaled by s) so that it changes the model."
  [seed s]
  (let [ad (lora/init @model {:rank 4 :alpha 8 :targets [:q :v :down]} (rng/key 7))]
    (update ad :params
            (fn [params]
              (into {} (for [[i ts] params]
                         [i (into {} (for [[t {:keys [a b]}] ts]
                                       [t {:a a :b (array/* s (rng/normal (rng/fold-in (rng/key seed) (+ (* 10 i) (count (name t))))
                                                                          (array/shape b)))}]))]))))))

(def ^:private ratio-band
  "|ρ - 1| at a round's first step: the engine's row of each token is
   normalized in double and rounded to float (2^-24 relative, |log p|
   below 8: 5e-7 measured on the JVM backend, band 2e-6, the array
   engine's own); on MLX float32's round-off through 3 blocks, the
   engine's band 2e-5 twice."
  (if (ts/jvm?) 2e-6 4e-5))

(defn- draw
  "Rollouts of prompts by the tiny model adapted by ad (nil: the base),
   K completions each, from key."
  ([ad ps k key] (draw ad ps k key nil))
  ([ad ps k key opts]
   (with-open [e (ae/engine {:model (if ad (lora/adapt @model ad) @model) :max-branches 48})
               m (lm/lm e tokenizer {:max-branches 48 :max-cells 4096})]
     (grpo/rollouts m ps (merge {:group-size k :max-tokens 6 :key key} opts)))))

(defn- temp-dir []
  (str (Files/createTempDirectory "grpo-test" (make-array FileAttribute 0))))

(defn- delete-tree [path]
  (doseq [f (reverse (file-seq (io/file path)))] (io/delete-file f true)))

;; ---------------------------------------------------------------------------
;; 1. Rollouts
;; ---------------------------------------------------------------------------

(defn- log-softmax-rows
  "The log-softmax of m's forward of tokens at temperature temp, on the
   host: [T vocab]."
  [m tokens temp]
  (first (ts/host (array/log-softmax (array// (qwen3/forward m [tokens]) temp)))))

(deftest rollouts-record-each-token-s-log-probability
  (doseq [temp [1.0 0.7]]
    (let [ad (adapter 3 0.1)
          ro (draw ad (subvec prompts 0 2) 3 (rng/key 4) {:sampling {:temp temp :top-k nil :top-p 1.0}})
          m (lora/adapt @model ad)]
      (is (= 2 (count ro)))
      (doseq [{:keys [context completions]} ro
              {:keys [tokens stop end-token logprobs text]} completions]
        (let [own (if (= :end stop) (conj tokens end-token) tokens)
              lp (log-softmax-rows m (into context own) temp)]
          (is (= (count own) (count logprobs)) "a log-probability a token, the end token's too")
          (is (= text ((:decode tokenizer) tokens)))
          (is (every? true? (map-indexed (fn [k l] (< (Math/abs (- (double l) (double (get-in lp [(+ (count context) k -1) (own k)]))))
                                                      ratio-band))
                                         logprobs))
              (str "temperature " temp ": each token's log-probability where it was drawn")))))
    (testing "a function of the key"
      (is (= (draw nil (subvec prompts 0 2) 3 (rng/key 4)) (draw nil (subvec prompts 0 2) 3 (rng/key 4))))
      (is (not= (draw nil (subvec prompts 0 2) 3 (rng/key 4)) (draw nil (subvec prompts 0 2) 3 (rng/key 5)))))))

;; ---------------------------------------------------------------------------
;; 2. Rewards
;; ---------------------------------------------------------------------------

(deftest the-built-in-rewards
  (let [exact (grpo/exact-match)
        extracted (grpo/exact-match {:extract #"answer is (-?\d+)"})
        bare (grpo/exact-match {:extract #"-?\d+" :key :n})
        fmt (grpo/format-match #"\d+")]
    (is (= 1.0 (exact {:answer 42} " 42\n")))
    (is (= 0.0 (exact {:answer 42} "42.")))
    (is (= 1.0 (extracted {:answer "7"} "first 3, so the answer is 7")))
    (is (= 0.0 (extracted {:answer "7"} "the answer is 3, not 7")))
    (is (= 0.0 (extracted {:answer "7"} "7")) "nothing extracted")
    (is (= 1.0 (bare {:n 9} "3 then 9")) "the last match")
    (is (= [1.0 0.0] [(fmt nil " 123 ") (fmt nil "12a")]))
    (is (= :invalid-args (:reason (ts/thrown-data (exact {:question "?"} "1")))) "a prompt without its answer")))

(deftest a-reward-that-is-not-finite-is-loud
  (let [ro (draw nil (subvec prompts 0 2) 2 (rng/key 1))]
    (doseq [bad [##NaN ##Inf nil "1"]]
      (let [data (ts/thrown-data (grpo/rewards (fn [p _] (if (= "Say 1" (:prompt p)) bad 0.0)) ro))]
        (is (= {:reason :non-finite-reward :prompt 1 :completion 0} (select-keys data [:reason :prompt :completion]))
            (pr-str bad))))))

;; ---------------------------------------------------------------------------
;; 3. Advantages
;; ---------------------------------------------------------------------------

(deftest advantages-are-relative-to-the-group
  (let [rs [[1.0 0.0 0.0 1.0] [0.5 0.5 0.5] [2.0 0.0 1.0]]
        [a b c] (grpo/advantages rs)
        std (Math/sqrt (/ 2.0 3.0))]
    (is (= [(/ 0.5 (+ 0.5 1e-4)) (/ -0.5 (+ 0.5 1e-4)) (/ -0.5 (+ 0.5 1e-4)) (/ 0.5 (+ 0.5 1e-4))] a) "the population's std")
    (is (= [0.0 0.0 0.0] b) "a flat group, exactly 0")
    (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-15) c [(/ 1.0 (+ std 1e-4)) (/ -1.0 (+ std 1e-4)) 0.0])))
    (is (= [[0.5 -0.5 -0.5 0.5] [0.0 0.0 0.0] [1.0 -1.0 0.0]] (grpo/advantages rs {:scale-rewards? false})) "Dr. GRPO's")
    (is (= [true false] (mapv grpo/flat? [[0.1 0.1] [0.1 0.2]])))))

;; ---------------------------------------------------------------------------
;; 4. Batches
;; ---------------------------------------------------------------------------

(defn- batch-holds-the-completions?
  "Does each row of the one batch of ro hold its prompt, then its
   completion, then padding, its mask exactly the completion's targets and
   its old log-probabilities there?"
  [ro b]
  (let [toks (ts/host (:tokens b))
        mask (ts/host (:mask b))
        old (ts/host (:old b))
        rows (for [{:keys [context completions]} ro c completions] [context c])]
    (every? true?
            (map (fn [[context {:keys [tokens stop end-token logprobs]}] row mrow orow]
                   (let [own (if (= :end stop) (conj tokens end-token) tokens)
                         n (+ (count context) (count own))
                         targets (keep-indexed (fn [t w] (when (pos? w) (row (inc t)))) mrow)]
                     (and (= (into context own) (subvec row 0 n))
                          (every? zero? (subvec row n))
                          (= own (vec targets))
                          (= (map double logprobs) (keep-indexed (fn [t w] (when (pos? w) (orow t))) mrow)))))
                 rows toks mask old))))

(deftest batches-are-prompt-completion-padding
  (let [ro (draw nil (subvec prompts 0 3) 2 (rng/key 2) {:max-tokens 5})
        adv (grpo/advantages (grpo/rewards lower ro))
        [b :as bs] (grpo/batches ro adv nil)
        lengths (for [{:keys [completions]} ro {:keys [tokens stop]} completions] (+ (count tokens) (if (= :end stop) 1 0)))]
    (is (= 1 (count bs)))
    (is (batch-holds-the-completions? ro b))
    (is (= (ts/host (:advantages b)) (mapv vector (apply concat adv))))
    (testing "weights and divisors by loss type"
      (let [by (fn [lt] (first (grpo/batches ro adv {:loss-type lt :max-tokens 5})))
            row-weights (fn [b] (mapv #(reduce max 0.0 %) (ts/host (:weights b))))
            tokens (reduce + lengths)]
        (is (= [6 tokens 30 1] (mapv (comp :count by) [:grpo :dapo :dr-grpo :bnpo])))
        (is (= (mapv #(/ 1.0 %) lengths) (row-weights (by :grpo))))
        (is (= (repeat 6 1.0) (row-weights (by :dapo)) (row-weights (by :dr-grpo))))
        (is (= (repeat 6 (/ 1.0 tokens)) (row-weights (by :bnpo))))
        (is (= (ts/host (:mask b)) (ts/host (array/where (array/> (:weights (by :bnpo)) 0.0) 1.0 0.0))))))
    (testing "microbatches"
      (let [bs (grpo/batches ro adv {:batch-size 4})]
        (is (= [4 2] (mapv :completions bs)))
        (is (= [4 2] (mapv :count bs)))))
    (let [row-of @#'grpo/row-of]
      (is (not (with-redefs-fn {#'grpo/row-of (fn [& args] (update (apply row-of args) :start inc))}
                 #(batch-holds-the-completions? ro (first (grpo/batches ro adv nil)))))
          "planted: the mask a position late"))))

;; ---------------------------------------------------------------------------
;; 5. The loss, by hand
;; ---------------------------------------------------------------------------

(defn- k3 [d] (- (Math/exp d) d 1.0))

(defn- loss-by-hand
  "The step's loss of loss type lt on ro (advantages adv) under m, the
   reference the base, by the forward's log-softmax read on the host,
   microbatches of size rows."
  [m ro adv {:keys [loss-type eps-low eps-high beta size max-tokens]}]
  (let [rows (vec (for [[{:keys [context completions]} as] (map vector ro adv)
                        [{:keys [tokens stop end-token logprobs]} a] (map vector completions as)]
                    (let [own (if (= :end stop) (conj tokens end-token) tokens)
                          lp (log-softmax-rows m (into context own) 1.0)
                          ref (log-softmax-rows @model (into context own) 1.0)
                          at (fn [rows k] (double (get-in rows [(+ (count context) k -1) (own k)])))]
                      (vec (for [k (range (count own))]
                             (let [new (at lp k)
                                   rho (Math/exp (- new (double (logprobs k))))
                                   clipped (min (max rho (- 1.0 eps-low)) (+ 1.0 eps-high))]
                               (+ (- (min (* rho a) (* clipped a))) (* beta (k3 (- (at ref k) new))))))))))
        mbs (partition-all size rows)
        sum (fn [xs] (reduce + 0.0 xs))]
    (case loss-type
      :grpo (/ (sum (map #(/ (sum %) (count %)) rows)) (count rows))
      :dapo (/ (sum (map sum rows)) (sum (map count rows)))
      :dr-grpo (/ (sum (map sum rows)) (* (count rows) max-tokens))
      :bnpo (/ (sum (map (fn [mb] (/ (sum (map sum mb)) (sum (map count mb)))) mbs)) (count mbs)))))

(defn- losses-are-by-hand?
  "Is each loss type's step loss its computation by hand, rollouts drawn
   by one adapter and the loss read under another, β 0.05, microbatches
   of 4? The completions end at any of 20 tokens, so that their lengths
   differ (and the loss types' values), and the adapters differ enough
   for the clip to act (measured: max |ρ - 1| 5.2, half the tokens
   clipped)."
  []
  (let [old (adapter 3 0.1)
        new (adapter 5 0.3)
        ro (draw old (subvec prompts 0 3) 3 (rng/key 6) {:end-tokens (set (range 40 60))})
        adv (grpo/advantages (grpo/rewards lower ro))
        m (lora/adapt @model new)
        tr (grpo/trainer new (learn/sgd {:lr 0.0}) (rng/key 1))
        results (for [lt [:grpo :dapo :dr-grpo :bnpo]]
                  (let [o {:loss-type lt :eps-low 0.2 :eps-high 0.28 :beta 0.05 :size 4 :max-tokens 6}
                        bs (grpo/with-reference @model (grpo/batches ro adv {:loss-type lt :batch-size 4 :max-tokens 6}) {})
                        [_ info] (sft/step @model tr bs {:loss (fn [m' b opts] (grpo/loss m' b (merge {:eps-low 0.2 :eps-high 0.28 :beta 0.05} opts)))})
                        want (loss-by-hand m ro adv o)]
                    {:lt lt :got (:loss info) :want want
                     :ok? (< (Math/abs (- (:loss info) want)) (* (ts/tol 1e-10) (max 1.0 (Math/abs want))))}))
        lengths (for [{:keys [completions]} ro {:keys [tokens stop]} completions] (+ (count tokens) (if (= :end stop) 1 0)))]
    (and (< 2 (count (distinct lengths)))
         (apply distinct? (map :want results))
         (every? :ok? results))))

(deftest each-loss-is-its-paper-s
  (is (losses-are-by-hand?))
  (testing "planted bugs"
    (is (not (with-redefs-fn {#'grpo/ratio-of (fn [lp _ on] (array/exp (array/where on (array/- lp (array/stop-gradient lp)) 0.0)))}
               losses-are-by-hand?))
        "the ratio against the new log-probabilities")
    (let [divisor @#'grpo/divisor]
      (is (not (with-redefs-fn {#'grpo/divisor (fn [lt n tokens budget] (if (= :dapo lt) n (divisor lt n tokens budget)))}
                 losses-are-by-hand?))
          "dapo's divisor the completions (mlx-node's)"))
    (let [k3 @#'grpo/k3]
      (is (not (with-redefs-fn {#'grpo/k3 (fn [d] (array/negate (k3 d)))} losses-are-by-hand?)) "the KL's sign"))
    (let [row-of @#'grpo/row-of]
      (is (not (with-redefs-fn {#'grpo/row-of (fn [& args] (update (apply row-of args) :start inc))} losses-are-by-hand?))
          "the mask a position late"))))

;; ---------------------------------------------------------------------------
;; 6. The ratios at a round's first step
;; ---------------------------------------------------------------------------

(defn- first-ratios
  "max |ρ - 1| at the first step of a round of the tiny model under
   adapter ad, at temperature temp."
  [ad temp]
  (let [tr (grpo/trainer ad (learn/sgd {:lr 0.1}) (rng/key 3))
        [_ info] (grpo/round @model tokenizer tr (subvec prompts 0 3) lower
                             {:group-size 4 :max-tokens 6 :iterations 2 :sampling {:temp temp :top-k nil :top-p 1.0}})]
    (mapv :ratio-max (:iterations info))))

(deftest ratios-are-1-at-a-round-s-first-step
  (doseq [[ad temp] [[(lora/init @model {:rank 4 :targets [:q :v]} (rng/key 1)) 1.0] [(adapter 3 0.1) 1.0] [(adapter 3 0.1) 0.7]]]
    (let [[first-step second-step] (first-ratios ad temp)]
      (is (< first-step ratio-band) (str "temperature " temp ": " first-step))
      (is (> second-step 1e-3) "the policy has moved at the second")))
  (let [row-of @#'grpo/row-of]
    (is (not (< (first (with-redefs-fn {#'grpo/row-of (fn [& args] (update (apply row-of args) :start inc))}
                         #(first-ratios (adapter 3 0.1) 1.0)))
                ratio-band))
        "planted: the mask a position late")))

;; ---------------------------------------------------------------------------
;; 7. The direction of a step
;; ---------------------------------------------------------------------------

(defn- completion-log-probs
  "Each completion's log-probability under m (by the forward), in order."
  [m ro]
  (vec (for [{:keys [context completions]} ro
             {:keys [tokens stop end-token]} completions]
         (let [own (if (= :end stop) (conj tokens end-token) tokens)
               lp (log-softmax-rows m (into context own) 1.0)]
           (reduce + (map-indexed (fn [k t] (double (get-in lp [(+ (count context) k -1) t]))) own))))))

(defn- moves-toward-the-advantage?
  "Does one SGD step of each loss type, β 0, raise Σ Â·log π of the
   rollouts it is taken on?"
  []
  (let [ad (adapter 3 0.1)
        ro (draw ad (subvec prompts 0 3) 4 (rng/key 8))
        adv (grpo/advantages (grpo/rewards lower ro))
        flat (vec (apply concat adv))
        before (completion-log-probs (lora/adapt @model ad) ro)]
    (every? (fn [lt]
              (let [tr (grpo/trainer ad (learn/sgd {:lr 0.05}) (rng/key 1))
                    bs (grpo/batches ro (grpo/advantages (grpo/rewards lower ro) {:scale-rewards? (not= lt :dr-grpo)})
                                     {:loss-type lt :max-tokens 6})
                    [tr'] (sft/step @model tr bs {:loss (fn [m' b opts] (grpo/loss m' b (merge {:eps-low 0.2 :eps-high 0.2 :beta 0.0} opts)))})
                    after (completion-log-probs (lora/adapt @model (:adapter tr')) ro)]
                (pos? (reduce + (map (fn [a x y] (* a (- y x))) flat before after)))))
            [:grpo :dapo :dr-grpo :bnpo])))

(defn- moves-toward-the-reference?
  "With every reward the same (advantages 0) and β > 0, do a round's two
   steps lower the KL to the base?"
  []
  (let [tr (grpo/trainer (adapter 3 0.3) (learn/sgd {:lr 0.005}) (rng/key 1))
        [_ info] (grpo/round @model tokenizer tr (subvec prompts 0 3) (constantly 1.0)
                             {:group-size 4 :max-tokens 6 :iterations 2 :beta 0.5})
        [k0 k1] (map :kl (:iterations info))]
    (and (= 3 (:flat-groups info)) (pos? k0) (< k1 k0))))

(deftest a-step-moves-the-policy-toward-the-advantage
  (is (moves-toward-the-advantage?))
  (is (moves-toward-the-reference?))
  (testing "planted bugs"
    (let [advantages grpo/advantages]
      (is (not (with-redefs-fn {#'grpo/advantages (fn [& args] (mapv #(mapv - %) (apply advantages args)))}
                 moves-toward-the-advantage?))
          "the advantages' sign flipped"))
    (let [k3 @#'grpo/k3]
      (is (not (with-redefs-fn {#'grpo/k3 (fn [d] (array/negate (k3 d)))} moves-toward-the-reference?)) "the KL's sign"))))

;; ---------------------------------------------------------------------------
;; 8. Accumulation
;; ---------------------------------------------------------------------------

(deftest accumulated-microbatches-step-as-one-batch
  (let [ad (adapter 3 0.1)
        ro (draw ad (subvec prompts 0 3) 4 (rng/key 9))
        adv (grpo/advantages (grpo/rewards lower ro))
        flat #(array/concatenate (mapv (fn [x] (array/reshape x [-1])) (array/tree-leaves (get-in % [:adapter :params]))))
        step (fn [lt size]
               (let [tr (grpo/trainer (adapter 5 0.2) (learn/sgd {:lr 0.5}) (rng/key 1))
                     bs (grpo/with-reference @model (grpo/batches ro adv {:loss-type lt :batch-size size :max-tokens 6}) {})]
                 (sft/step @model tr bs {:loss (fn [m' b opts] (grpo/loss m' b (merge {:eps-low 0.2 :eps-high 0.28 :beta 0.04} opts)))})))
        same? (fn [[t1 i1] [t2 i2]]
                (and (ts/close? (:loss i1) (:loss i2) {:rtol (ts/tol 1e-10) :atol 0.0})
                     (ts/close? (flat t1) (flat t2) {:rtol 0.0 :atol (ts/tol 1e-10)})))]
    (doseq [lt [:grpo :dapo :dr-grpo]]
      (is (same? (step lt 5) (step lt 12)) (str lt ": microbatches of 5, 5 and 2 as one of 12")))
    (is (same? (step :bnpo 12) (step :dapo 12)) "bnpo with one microbatch is dapo")
    (is (not (same? (step :bnpo 5) (step :bnpo 12))) "bnpo normalizes each microbatch")))

;; ---------------------------------------------------------------------------
;; 9. Training raises a reward
;; ---------------------------------------------------------------------------

(defn- rewards-by-round
  "The mean reward of each of 5 rounds of the tiny model (4 prompts of 8
   completions, AdamW at 3e-2) on the lower-case reward."
  [lt]
  (let [tr (grpo/trainer (lora/init @model {:rank 4 :targets [:q :v :down]} (rng/key 1)) (learn/adamw {:lr 3e-2 :weight-decay 0.0})
                         (rng/key 2))]
    (mapv :reward-mean (:history (grpo/train @model tokenizer tr prompts lower
                                             {:rounds 5 :prompts-per-round 4 :group-size 8 :max-tokens 8 :loss-type lt})))))

(deftest training-raises-the-reward
  ;; measured 2026-11-07 (the JVM backend), grpo: 0.24, 0.53, 0.73, 0.98,
  ;; 1.00; dapo 0.24, 0.53, 0.86, 0.99, 1.00
  (doseq [lt [:grpo :dapo :dr-grpo :bnpo]]
    (let [rs (rewards-by-round lt)]
      (is (< (first rs) 0.3) (pr-str lt rs))
      (is (> (last rs) 0.8) (pr-str lt rs))))
  (let [advantages grpo/advantages
        rs (with-redefs-fn {#'grpo/advantages (fn [& args] (mapv #(mapv - %) (apply advantages args)))}
             #(rewards-by-round :grpo))]
    (is (not (> (last rs) 0.8)) (str "planted: the advantages' sign flipped " (pr-str rs)))))

;; ---------------------------------------------------------------------------
;; 10. Resuming from a checkpoint
;; ---------------------------------------------------------------------------

(defn- file-bytes [path] (Files/readAllBytes (.toPath (io/file path))))

(deftest a-checkpoint-resumes-exactly
  (let [opt (learn/warmup-cosine (learn/clip-norm (learn/adamw {:lr 1e-2}) {:max-norm 1.0}) {:warmup 1 :steps 8})
        tr (grpo/trainer (adapter 3 0.1) opt (rng/key 5))
        opts {:rounds 4 :prompts-per-round 3 :group-size 3 :max-tokens 5 :iterations 2 :loss-type :dapo :batch-size 4}
        a (temp-dir)
        b (temp-dir)
        same (fn [h] (mapv #(dissoc % :seconds) h))]
    (try
      (let [whole (grpo/train @model tokenizer tr prompts lower (assoc opts :save-dir a :save-every 2))
            loaded (sft/load-checkpoint (str a "/step-00000004"))
            resumed (grpo/train @model tokenizer loaded prompts lower (assoc opts :save-dir b))
            at-8 (fn [dir f] (file-bytes (str dir "/step-00000008/" f)))]
        (is (= 2 (:round loaded)))
        (is (= 4 (count (:history whole))))
        (is (= (same (drop 2 (:history whole))) (same (:history resumed))))
        (is (java.util.Arrays/equals ^bytes (at-8 a "trainer.edn") ^bytes (at-8 b "trainer.edn")))
        (is (java.util.Arrays/equals ^bytes (at-8 a "adapter.gguf") ^bytes (at-8 b "adapter.gguf"))))
      (finally (delete-tree a) (delete-tree b)))))

;; ---------------------------------------------------------------------------
;; 11. Loud errors
;; ---------------------------------------------------------------------------

(deftest a-loss-that-is-not-finite-is-loud
  (let [ad (assoc-in (adapter 3 0.1) [:params 1 :v :b] (array/full [32 4] ##NaN))
        data (ts/thrown-data (grpo/round @model tokenizer (grpo/trainer ad (learn/adamw {}) (rng/key 1)) (subvec prompts 0 2)
                                         lower {:group-size 2 :max-tokens 3}))]
    (is (contains? #{:non-finite :decode-failed} (:reason data)) (pr-str data))))

(deftest misuse-is-loud
  (let [reason (fn [f] (:reason (ts/thrown-data (f))))
        tr (grpo/trainer (adapter 3 0.1) (learn/sgd {:lr 0.1}) (rng/key 1))
        round #(grpo/round @model tokenizer tr (subvec prompts 0 2) lower (merge {:group-size 2 :max-tokens 3} %))
        ro (draw nil (subvec prompts 0 2) 2 (rng/key 1))
        adv (grpo/advantages (grpo/rewards lower ro))]
    (testing "a round's options"
      (is (= :invalid-args (reason #(round {:loss-type :ppo}))))
      (is (= :invalid-args (reason #(round {:group-size 1}))) "a group of one has no relative advantage")
      (is (= :invalid-args (reason #(round {:eps-low 1.5}))))
      (is (= :invalid-args (reason #(round {:beta -0.1}))))
      (is (= :invalid-args (reason #(round {:iterations 0}))))
      (is (= :unknown-options (reason #(round {:mu 2})))))
    (testing "rollouts, batches and the loss"
      (is (= :invalid-args (reason #(draw nil [{:text "hi"}] 2 (rng/key 1)))))
      (is (= :invalid-args (reason #(with-open [e (ae/engine {:model @model})
                                                m (lm/lm e tokenizer)]
                                      (grpo/rollouts m (subvec prompts 0 1) {:group-size 2})))))
      (is (= :invalid-args (reason #(grpo/batches ro adv {:loss-type :dr-grpo}))) ":dr-grpo needs the budget")
      (is (= :invalid-args (reason #(grpo/batches ro (pop adv) nil))))
      (is (= :invalid-args (reason #(grpo/loss @model (first (grpo/batches ro adv nil)) {:eps-low 0.2 :eps-high 0.2 :beta 0.1})))
          "β > 0 without the reference"))
    (testing "training"
      (is (= :invalid-args (reason #(grpo/trainer (adapter 3 0.1) (learn/sgd {:lr 0.1}) 7))) "a key, not a seed")
      (is (= :invalid-args (reason #(grpo/train @model tokenizer (sft/trainer (adapter 3 0.1) (learn/sgd {:lr 0.1}) (rng/key 1))
                                                prompts lower {:rounds 1})))
          "an SFT trainer, without its round")
      (is (= :invalid-args (reason #(grpo/train @model tokenizer (assoc tr :round 3) prompts lower {:rounds 2}))))
      (is (= :unknown-options (reason #(grpo/train @model tokenizer tr prompts lower {:rounds 1 :epochs 2})))))))

(deftest the-log-of-a-round
  (let [tr (grpo/trainer (adapter 3 0.1) (learn/sgd {:lr 0.1}) (rng/key 3))
        [tr' info] (grpo/round @model tokenizer tr (subvec prompts 0 3) lower {:group-size 4 :max-tokens 6 :iterations 2})]
    (is (= 1 (:round tr')))
    (is (= 2 (:step tr') (:step info)))
    (is (= {:round 0 :groups 3 :completions 12 :loss-type :grpo}
           (select-keys info [:round :groups :completions :loss-type])))
    (is (every? #(contains? info %) [:reward-mean :reward-std :flat-groups :tokens :length-mean :truncated :entropy :seconds]))
    (is (= [0 1] (mapv :step (:iterations info))))
    (is (every? #(every? (partial contains? %) [:loss :grad-norm :ratio-max :clip-fraction :kl]) (:iterations info)))
    (is (pos? (:entropy info)))))
