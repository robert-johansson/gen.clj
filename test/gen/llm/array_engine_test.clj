(ns gen.llm.array-engine-test
  "gen.array's engine (gen.llm.array-engine) on the backend under test,
   over the tiny random Qwen3 of test/fixtures/gguf (vocabulary 128, 3
   blocks; gen.llm.models.qwen3-test) as F32 and Q8_0, with the toy
   tokenizer's 128 tokens (printable ASCII, BOS 95, EOS 96):
   - the Branching contract (gen.llm.branching-contract) within the
     engine's band, the LM contract (gen.llm.lm-contract) within it, and
     the token generative function's and token SMC's checks
     (gen.llm.token-contract), which hold exactly within one LM;
   - a branch's rows are qwen3's forward's (log-softmax of its logits);
   - lane groups: a step of exactly a group's rows reuses it in place,
     every other step stacks its rows, with the same results; capacities
     grow by blocks;
   - misuse is loud. (The shared memory budget with the engine open is
     gen.llm.array-engine-llama-test's: a tiny model's arrays are too
     small for MLX's active memory to tell.)

   The band. On the JVM backend (float64) a context's logits are the same
   however it is read (batched, stacked, in chunks, at any capacity: a
   masked slot's weight is an exact 0, summed after the others;
   gen.llm.models.qwen3-static-test), and a row is normalized in double
   from them rounded to float: rows agree exactly (measured: 0 over every
   check), and a score, read on the device in float64, differs from the
   row's float by its rounding, at most half an ulp of a float below 32
   in magnitude, 2^-19 (measured 4.8e-7 = 2^-21): band 2e-6. On MLX
   (float32) the reads differ by float32's round-off through 3 blocks,
   with shapes and capacities (gen.llm.models.qwen3-test's band of 4e-5
   in a logit); a row's log-probabilities move with its logits and their
   normalizer: band 2e-5, twice the largest |Δ log p| measured (6.2e-6,
   rounded up; the docstring of band).

   Planted mutations, run here as negative controls (with-redefs-fn): a
   stacked step's rows assigned to the slots in reverse, a run's scores
   read one target late, a capacity grown with its cache dropped, the
   rows left unnormalized; each fails the contract checks named in
   planted-bugs-fail-the-contract. A decode of fewer rows than the host
   pool's threads normalizes each row in parts across them, the same
   floats bit for bit (rows-normalized-in-parts-are-normalized-whole)."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array :as array]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lm :as lm]
            [gen.llm.lm-contract :as lc]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.token :as token]
            [gen.llm.token-contract :as tc]
            [gen.llm.toy-engine :as toy]
            [gen.llm.toy-tokenizer :as tt]
            [gen.test-support :as ts]))

(def ^:private files
  {:f32 "test/fixtures/gguf/qwen3-tiny-f32.gguf"
   :q8_0 "test/fixtures/gguf/qwen3-tiny-q8_0.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen3/load f)])) files)))

(def ^:private band
  "The largest |Δ log p| between two computations of one distribution
   (the namespace's docstring): 2e-6 on the JVM backend (float rounding);
   on MLX 2e-5, twice the largest the contract's comparisons measured
   over three prompts and continuations, F32 and Q8_0 (Mac mini M4,
   2026-11-07): 6.2e-6 on Metal, 3.3e-6 on MLX's CPU."
  (if (ts/jvm?) 2e-6 2e-5))

(def ^:private tokenizer (tt/tokenizer 128))

(def ^:private engines (atom {}))

(defn- engine
  "The shared engine of model k (opened once): small blocks and reads, so
   that capacities grow and reads take several forwards."
  [k]
  (or (@engines k)
      (get (swap! engines assoc k (ae/engine {:model (@models k) :max-branches 80 :block 16 :n-batch 8})) k)))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (doseq [[_ ^java.io.Closeable e] @engines] (.close e))
                                    (reset! engines {})))))

(def ^:private contract-opts
  {:prompt [95 51 72 69 0 67 65 80 73 84 65 76 0 73 83] :continuation [0 83 84 79 67 75 72 79 76 77 12 0 73]
   :k 4 :band band :top 64})

;; ---------------------------------------------------------------------------
;; The contracts
;; ---------------------------------------------------------------------------

(deftest the-branching-contract-holds-within-the-band
  (doseq [k (keys files)
          :let [e (engine k)]]
    (doseq [[c check] bc/checks]
      (testing [k c]
        (is (= [] (bc/run-check check e contract-opts)))))
    (is (zero? (:live (br/info e))) "every check disposes what it makes")))

(defn- lm-setup [k]
  (let [e (engine k)]
    {:engine e
     :lm #(lm/lm e tokenizer {:max-branches 12})
     :fresh (fn [] (let [e (ae/engine {:model (@models k) :max-branches 16 :block 16 :n-batch 8})]
                     [e (lm/lm e tokenizer {:max-branches 12})]))}))

(deftest the-lm-contract-holds-within-the-band
  (doseq [k (keys files)]
    (doseq [[c check] lc/checks]
      (testing [k c]
        (is (= [] (lc/run-check check (lm-setup k) contract-opts)))))
    (is (zero? (:live (br/info (engine k)))) "every check closes what it makes")))

(deftest the-token-and-smc-checks-hold
  ;; each check on an LM of its own, with room enough, in cells and
  ;; branches, that the LM evicts nothing SMC read: every score is then the
  ;; one its token was read with (score once), the particles' and their
  ;; traces' alike. (After an eviction a token is read again, by a run,
  ;; whose score differs from the step's row by float rounding: on this
  ;; tiny model many particles end early, and their last rows would be
  ;; evicted with 255 branches.) The LM's cells are no bound of this
  ;; engine, whose branches share no pool of cells
  (with-open [e (ae/engine {:model (:f32 @models) :max-branches 512 :block 16})]
    (let [{:keys [encode]} tokenizer
          opts {:prompt (encode "The capital of Sweden is" {}) :forced (encode " S" {:add-special? false})
                :min-weight -40.0 :sea-prompt (encode "Write about the sea:" {})
                :date-prompt (encode "Today's date:" {}) :particles 32}]
      (doseq [[c check] tc/checks]
        (with-open [m (lm/lm e tokenizer {:max-cells 8192})]
          (testing c
            (is (= [] (tc/run-check check {:engine e :lm m :completion (token/completion m)} opts)))))))))

;; ---------------------------------------------------------------------------
;; The engine's own
;; ---------------------------------------------------------------------------

(defn- forward-row
  "The log-probabilities after ctx by qwen3's forward of model k, its
   logits normalized on the host in double, as floats."
  ^floats [k ctx]
  (let [[logits] (qwen3/prefill (@models k) [ctx] {:logits :last})
        xs (mapv double (array/->floats logits))
        m (reduce max xs)
        z (+ m (Math/log (reduce + (map #(Math/exp (- % m)) xs))))]
    (float-array (map #(- % z) xs))))

(deftest a-branch-s-rows-are-the-forward-s
  (doseq [k (keys files)
          :let [e (engine k)
                b (br/open e (:prompt contract-opts))]]
    (is (<= (bc/max-diff (forward-row k (:prompt contract-opts)) (br/logprobs b) 128) band) (name k))
    (let [{:keys [branch]} (br/extend b (:continuation contract-opts))]
      (is (<= (bc/max-diff (forward-row k (into (:prompt contract-opts) (:continuation contract-opts))) (br/logprobs branch) 128)
              band)
          (str (name k) ", extended past a block"))
      (br/dispose! branch))))

(defn- sum-by-parts
  "The terms exp(x - m) of the n floats of xs from off summed part by part
   (k parts), then the parts' sums: the same terms in another order."
  [^floats xs off n k]
  (let [m (double (reduce max (map #(aget xs (+ off %)) (range n))))
        part (fn [p] (reduce + 0.0 (map #(Math/exp (- (double (aget xs (+ off %))) m))
                                        (range (quot (* p n) k) (quot (* (inc p) n) k)))))]
    (+ m (Math/log (reduce + 0.0 (map part (range k)))))))

(deftest rows-normalized-in-parts-are-normalized-whole
  ;; a decode of fewer rows than the host pool's threads normalizes each
  ;; row in parts across them (genclj-c7uf): its log-normalizer and its
  ;; floats are the whole row's, bit for bit, over random rows (Qwen3's
  ;; 151,936 logits among them) in any number of parts, a -Inf logit is a
  ;; probability 0, and NaN and +Inf are loud alike. Power: the same terms
  ;; summed part by part give other log-normalizers on these rows, so a
  ;; change of the sum's order fails the first check
  (let [normalized @#'ae/normalized
        in-parts @#'ae/normalized-in-parts
        whole-z @#'ae/row-normalizer
        parts-z @#'ae/row-normalizer-in-parts
        bits #(Double/doubleToRawLongBits %)
        pool ((requiring-resolve 'gen.llm.host/pool) "test rows-" 3)
        r (java.util.Random. 11)
        rows (fn [b n scale] (float-array (repeatedly (* b n) #(* scale (.nextGaussian r)))))]
    (try
      (doseq [n [1 2 7 128 1000 151936] k [2 5 16] scale [1.0 8.0 30.0]
              :let [xs (rows 2 n scale)]
              i [0 1]]
        (is (= (bits (whole-z xs (* i n) n)) (bits (parts-z pool k xs (* i n) n))) (pr-str [n k scale i]))
        (is (java.util.Arrays/equals ^floats (normalized xs i n "t") ^floats (in-parts pool k xs i n "t")) (pr-str [n k scale i])))
      (let [xs (rows 1 1000 8.0)]
        (aset xs 17 Float/NEGATIVE_INFINITY)
        (is (java.util.Arrays/equals ^floats (normalized xs 0 1000 "t") ^floats (in-parts pool 16 xs 0 1000 "t")))
        (is (= Float/NEGATIVE_INFINITY (aget ^floats (in-parts pool 16 xs 0 1000 "t") 17))))
      (doseq [bad [Float/NaN Float/POSITIVE_INFINITY]
              :let [xs (rows 1 1000 8.0)]]
        (aset xs 500 (float bad))
        (is (= :decode-failed (:reason (ts/thrown-data (normalized xs 0 1000 "t")))))
        (is (= :decode-failed (:reason (ts/thrown-data (in-parts pool 16 xs 0 1000 "t"))))))
      (is (= :decode-failed (:reason (ts/thrown-data (in-parts pool 16 (float-array (repeat 64 Float/NEGATIVE_INFINITY)) 0 64 "t"))))
          "every logit -Inf")
      (testing "the power: another order of the same sum moves the log-normalizer"
        (let [xs (rows 6 151936 8.0)]
          (is (some #(not= (bits (whole-z xs (* % 151936) 151936)) (bits (sum-by-parts xs (* % 151936) 151936 16)))
                    (range 6)))))
      (finally (.shutdown ^java.util.concurrent.ExecutorService pool)))))

(defn- stats-delta
  "The engine's stats that (f) changes, and f's value."
  [e f]
  (let [before (ae/stats e)
        v (f)]
    [(into {} (keep (fn [[k n]] (let [d (- (long n) (long (before k)))] (when-not (zero? d) [k d])))) (ae/stats e)) v]))

(deftest a-lane-group-steps-in-place
  ;; four forks of one branch advanced together are stacked (they share
  ;; one row); their successors, a group held by them alone, step in
  ;; place, as lockstep particles do; a fork of one breaks the lane (the
  ;; group is shared), and a resample's repeated rows are stacked again.
  ;; The rows are the same either way.
  (let [e (engine :f32)
        p (:prompt contract-opts)
        root (br/open e p)
        ts [10 20 30 40]
        kids (br/fork root 4)
        _ (br/dispose! root)
        [d1 s1] (stats-delta e #(br/advance kids ts))
        [d2 s2] (stats-delta e #(br/advance s1 [1 2 3 4]))
        groups (ae/groups e)
        fresh (fn [ctx] (let [b (br/open e ctx) lp (br/logprobs b)] (br/dispose! b) lp))]
    (is (= {:steps 1 :stacked-steps 1 :stacked-rows 4} (dissoc d1 :grows)))
    (is (= {:steps 1 :lane-steps 1} (dissoc d2 :grows)))
    (is (apply = (keep #(some-> % first) groups)) "one group")
    (is (= #{0 1 2 3} (set (keep second groups))) "its rows")
    (doseq [[b t u] (map vector s2 ts [1 2 3 4])]
      (is (<= (bc/max-diff (fresh (conj p t u)) (br/logprobs b) 64) band)))
    (let [f (br/fork (first s2))
          [d3 s3] (stats-delta e #(br/advance s2 [5 6 7 8]))
          kids (br/resample s3 [0 0 1 3])
          [d4 s4] (stats-delta e #(br/advance kids [9 9 9 9]))]
      (is (= {:steps 1 :stacked-steps 1 :stacked-rows 4} (dissoc d3 :grows)) "a fork shares the group")
      (is (= {:steps 1 :stacked-steps 1 :stacked-rows 4} (dissoc d4 :grows)) "repeated rows")
      (doseq [[b t] (map vector s4 [1 1 2 4])]
        (is (<= (bc/max-diff (fresh (conj p (ts (dec t)) t (+ 4 t) 9)) (br/logprobs b) 64) band)))
      (br/dispose! (conj s4 f)))))

(deftest a-branch-alone-in-its-group-steps-in-place
  ;; a read's group and a step's are new groups: a fork of a branch,
  ;; advanced, holds a group of its own, which its next step reuses while
  ;; the branch it was forked from stays as it was. (Planted, 2026-11-07,
  ;; found so: a read's new group's id lost inside a vswap!, so that the
  ;; step's group took the read's id, and the step was stacked)
  (let [e (engine :f32)
        p (:prompt contract-opts)
        b (br/open e p)
        [s] (br/advance [(br/fork b)] [7])
        ids (keep first (ae/groups e))
        [d s2] (stats-delta e #(br/advance [s] [8]))]
    (is (apply distinct? ids) "one group each")
    (is (= {:steps 1 :lane-steps 1} (dissoc d :grows)))
    (is (<= (bc/max-diff (forward-row :f32 (conj p 7 8)) (br/logprobs (first s2)) 128) band))
    (let [[t] (br/advance [(br/fork b)] [9])]
      (is (<= (bc/max-diff (forward-row :f32 (conj p 9)) (br/logprobs t) 128) band) "the branch forked from")
      (br/dispose! (into [b t] s2)))))

(deftest capacities-grow-by-blocks
  ;; blocks of 16: a branch read to 15 positions, then advanced to 40 one
  ;; token at a time, grows twice and keeps the forward's rows
  (let [e (engine :f32)
        ctx (vec (take 40 (cycle [33 70 79 82 87 65 82 68 0])))
        [d b] (stats-delta e #(loop [b (br/open e (subvec ctx 0 15)) i 15]
                                (if (= i 40) b (recur (first (br/advance [b] [(ctx i)])) (inc i)))))]
    (is (= 2 (:grows d)))
    (is (<= (bc/max-diff (forward-row :f32 ctx) (br/logprobs b) 128) band))
    (br/dispose! b)))

;; ---------------------------------------------------------------------------
;; Planted bugs
;; ---------------------------------------------------------------------------

(def ^:private planted
  "Mutations of the engine, each with the contract checks it fails."
  {"a step's rows given to its slots in reverse"
   [{#'ae/host-rows (let [rows @#'ae/host-rows] (fn [pool logits what] (vec (rseq (rows pool logits what)))))}
    [:fork-matches-fresh :batch-matches-single :resample]]
   "a run's scores read one target late"
   [{#'ae/run-scores (let [scores @#'ae/run-scores]
                       (fn [logits targets] (scores logits (into [(first targets)] (butlast targets)))))}
    [:score-matches-steps]]
   "a capacity grown with its cache dropped"
   [{#'qwen3/grow (fn [cache capacity]
                    (if (= capacity (:capacity cache))
                      cache
                      (assoc (qwen3/static-cache {:config (:config (:f32 @models))} (count (:offsets cache)) capacity)
                             :offsets (:offsets cache) :positions (:positions cache))))}
    [:score-matches-steps]]
   "rows left unnormalized"
   [{#'ae/log-normalizer (fn [m _] m)}
    [:normalized]]})

(defn- shared-group-check
  "A group of two rows held by its branches and a fork of the first: the
   branches step together (stacked: the fork shares the group), and the
   fork then reads its own state."
  [e {:keys [prompt band]}]
  (let [b (br/open e prompt)
        [x y] (br/advance (br/fork b 2) [1 2])
        f (br/fork x)
        stepped (br/advance [x y] [3 4])
        [g] (br/advance [f] [5])
        fresh (let [c (br/open e (conj prompt 1 5)) lp (br/logprobs c)] (br/dispose! c) lp)
        d (bc/max-diff fresh (br/logprobs g) 128)]
    (br/dispose! (into [b g] stepped))
    (cond-> [] (not (<= d band)) (conj (str "the fork's read: |Δ log p| = " d)))))

(deftest planted-bugs-fail-the-contract
  (doseq [[what [redefs caught-by]] planted]
    (with-open [e (ae/engine {:model (:f32 @models) :max-branches 12 :block 16 :n-batch 8})]
      (doseq [c caught-by]
        (testing [what c]
          (is (seq (with-redefs-fn redefs #(bc/run-check (bc/checks c) e contract-opts))))))))
  (testing "a shared group stepped in place (a fork of one of its rows ignored): on MLX, whose step consumes it"
    ;; on the JVM backend a step consumes nothing (compile is the step
    ;; itself), so the values stay right there and the plant is invisible
    (with-open [e (ae/engine {:model (:f32 @models) :max-branches 12 :block 16 :n-batch 8})]
      (is (= [] (bc/run-check shared-group-check e contract-opts)) "unplanted")
      (when-not (ts/jvm?)
        (is (seq (with-redefs-fn {#'ae/lane? (constantly true)} #(bc/run-check shared-group-check e contract-opts))))))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [reason #(:reason (ts/thrown-data (%)))]
    (is (= :invalid-option (reason #(ae/engine {:model (:f32 files) :max-branch 3}))) "an unknown option")
    (is (= :invalid-option (reason #(ae/engine {:model (:f32 files) :n-ctx 0}))))
    (is (= :invalid-option (reason #(ae/engine {:model (:f32 files) :dtype :int32}))))
    (is (= :invalid-option (reason #(ae/engine {:model (:f32 @models) :dtype :float32}))) ":dtype is a file's")
    (is (= :no-model (reason #(ae/engine {:model "no/such.gguf"}))))
    (is (= :no-model (reason #(ae/engine {}))))
    (with-open [t (toy/engine {})]
      (is (= :not-an-array-engine (reason #(ae/stats t)))))
    (let [e (engine :f32)
          {:keys [n-ctx max-branches]} (br/info e)]
      (is (= 256 n-ctx) "the model's context")
      (is (= 80 max-branches))
      (is (= :context-full (reason #(br/open e (vec (repeat (inc n-ctx) 3)))))))))
