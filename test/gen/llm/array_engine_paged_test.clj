(ns gen.llm.array-engine-paged-test
  "gen.array's engine in its paged mode (gen.llm.array-engine's :paged?,
   experimental; genclj-1daf) on the backend under test, over the tiny
   random Qwen3 of test/fixtures/gguf (vocabulary 128, 3 blocks; F32 and
   Q8_0), with the toy tokenizer's 128 tokens:
   - the Branching contract (gen.llm.branching-contract), the LM contract
     (gen.llm.lm-contract) and the token generative function's and token
     SMC's checks (gen.llm.token-contract), within the engine's band
     (gen.llm.array-engine-test's: 2e-6 on the JVM backend, 2e-5 on MLX);
   - a branch's rows are qwen3's forward's, through steps that cross
     blocks;
   - forks share their blocks: a fork copies nothing, a step copies a
     shared block filled in part (copy-on-write) and nothing else, and a
     disposed branch's blocks are free again;
   - a step pads its rows to a power of 2 (:min-rows at least), and the
     padding changes no row;
   - the pool grows when its blocks run out, keeping every branch;
   - misuse is loud.
   Planted mutations fail the contract checks named in
   planted-bugs-fail-the-contract: copy-on-write skipped (a fork's step
   writes into its parent's block), a step's rows given to its slots in
   reverse, and the attention blind to each row's last position."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array :as array]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lm :as lm]
            [gen.llm.lm-contract :as lc]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.paged :as paged]
            [gen.llm.token :as token]
            [gen.llm.token-contract :as tc]
            [gen.llm.toy-tokenizer :as tt]
            [gen.test-support :as ts]))

(def ^:private files
  {:f32 "test/fixtures/gguf/qwen3-tiny-f32.gguf"
   :q8_0 "test/fixtures/gguf/qwen3-tiny-q8_0.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen3/load f)])) files)))

(def ^:private band (if (ts/jvm?) 2e-6 2e-5))

(def ^:private tokenizer (tt/tokenizer 128))

(def ^:private engines (atom {}))

(def ^:private paged-opts
  "Small blocks and pools, so that steps cross blocks, forks share partial
   blocks, and the pool grows."
  {:paged? true :block-size 8 :pool-blocks 16 :n-batch 8})

(defn- engine
  "The shared paged engine of model k (opened once)."
  [k]
  (or (@engines k)
      (get (swap! engines assoc k (ae/engine (merge paged-opts {:model (@models k) :max-branches 80}))) k)))

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

(deftest the-lm-contract-holds-within-the-band
  (doseq [k (keys files)]
    (doseq [[c check] lc/checks]
      (testing [k c]
        (is (= [] (lc/run-check check {:engine (engine k)
                                       :lm #(lm/lm (engine k) tokenizer {:max-branches 12})
                                       :fresh (fn [] (let [e (ae/engine (merge paged-opts {:model (@models k) :max-branches 16}))]
                                                       [e (lm/lm e tokenizer {:max-branches 12})]))}
                                contract-opts)))))))

(deftest the-token-and-smc-checks-hold
  (with-open [e (ae/engine (merge paged-opts {:model (:f32 @models) :max-branches 512}))]
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
  "The log-probabilities after ctx by qwen3's forward of model k,
   normalized on the host in double, as floats."
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
    (let [ctx (into (:prompt contract-opts) (:continuation contract-opts))
          stepped (reduce (fn [b t] (first (br/advance [b] [t]))) (br/fork b) (:continuation contract-opts))]
      (is (<= (bc/max-diff (forward-row k ctx) (br/logprobs stepped) 128) band) (str (name k) ", stepped across blocks"))
      (br/dispose! [b stepped]))))

(defn- stats-delta [e f]
  (let [before (ae/stats e)
        v (f)]
    [(into {} (keep (fn [[k n]] (let [d (- (long n) (long (before k)))] (when-not (zero? d) [k d])))) (ae/stats e)) v]))

(deftest forks-share-blocks-copy-on-write
  (with-open [e (ae/engine (merge paged-opts {:model (:f32 @models) :max-branches 12}))]
    (let [p (vec (take 12 (cycle (:prompt contract-opts))))  ; a block and a half of 8
          free0 (:free (ae/blocks e))
          root (br/open e p)
          kids (br/fork root 3)
          held #(- (long free0) (long (:free (ae/blocks e))))]
      (is (= 2 (held)) "the root's two blocks; three forks share them")
      (let [[d stepped] (stats-delta e #(br/advance kids [1 2 3]))]
        (is (= {:steps 1 :rows 3 :padded-rows 1 :copied-blocks 3 :copy-reads 1} d)
            (str "three rows padded to 4; each fork copies the shared partial block (the root still holds it),"
                 " read by a call of its own: a run wrote it"))
        (is (= 5 (held)) "the root's two, three copies")
        (doseq [[b t] (map vector stepped [1 2 3])]
          (is (<= (bc/max-diff (forward-row :f32 (conj p t)) (br/logprobs b) 128) band)))
        (br/dispose! root)
        (is (= 4 (held)) "the root disposed: its partial block freed, the full one shared by the steps")
        (let [[d2 s2] (stats-delta e #(br/advance stepped [4 5 6]))]
          (is (zero? (:copied-blocks d2 0)) "each holds its partial block alone: written in place")
          (br/dispose! s2))
        (is (= free0 (:free (ae/blocks e))) "every block free again")))))

(defn- tails-scenario
  "Steps whose copies read the last steps' tails, on a fresh paged engine
   (opts over paged-opts): {:gaps {what max |Δ log p| against the
   forward} :copies [[copied-blocks copy-reads] by step]}."
  [opts]
  (with-open [e (ae/engine (merge paged-opts {:model (:f32 @models) :max-branches 16} opts))]
    (let [p (vec (take 10 (cycle (:prompt contract-opts))))  ; a block and a quarter of 8
          gap (fn [bs ctxs] (reduce max 0.0 (map #(bc/max-diff (forward-row :f32 %2) (br/logprobs %1) 128) bs ctxs)))
          copies (fn [d] [(:copied-blocks d 0) (:copy-reads d 0)])
          root (br/open e p)
          [d1 s1] (stats-delta e #(br/advance (br/fork root 3) [1 2 3]))
          ;; the stepped branches forked: the blocks they share, the last
          ;; step wrote (token SMC's resampled particles)
          [d2 s2] (stats-delta e #(br/advance (mapv br/fork s1) [4 5 6]))
          ;; the parents, sharing nothing now, step in place, their blocks
          ;; unchanged by their forks' copies; then the forks' forks copy
          ;; from the tails two steps back, the blocks the parents' step
          ;; wrote being others
          [d3 s3] (stats-delta e #(br/advance s1 [7 8 9]))
          [d4 s4] (stats-delta e #(br/advance (mapv br/fork s2) [10 11 12]))
          ;; a run writes a block of the tails in place: a step's copy of
          ;; it after reads the pools, never the stale tails (nor the
          ;; others', read with it)
          x (:branch (br/extend (s4 0) [13]))
          [d5 [y1 y2]] (stats-delta e #(br/advance [(br/fork x) (br/fork (s4 1))] [15 16]))
          ;; a step without copies writes a block of the tails in place:
          ;; the same
          [d6 [y3]] (stats-delta e #(br/advance [y2] [17]))
          [d7 [y4]] (stats-delta e #(br/advance [(br/fork y3)] [18]))
          out {:gaps {:forks (gap s2 (map #(conj p %1 %2) [1 2 3] [4 5 6]))
                      :parents (gap s3 (map #(conj p %1 %2) [1 2 3] [7 8 9]))
                      :forks-of-forks (gap s4 (map #(conj p %1 %2 %3) [1 2 3] [4 5 6] [10 11 12]))
                      :after-a-run (gap [y1 y2] [(conj p 1 4 10 13 15) (conj p 2 5 11 16)])
                      :after-a-step (gap [y3 y4] [(conj p 2 5 11 16 17) (conj p 2 5 11 16 17 18)])}
               :copies (mapv copies [d1 d2 d3 d4 d5 d6 d7])}]
      (br/dispose! (concat [root x y1 y3 y4] s2 s3 (subvec s4 1)))
      out)))

(def ^:private tails-planted
  "Mutations of the tails' books, each failing tails-scenario's gaps."
  {"the tails' rows shifted by one"
   {#'ae/tails-index (let [f @#'ae/tails-index] (fn [slots n bs] (update-vals (f slots n bs) #(mod (inc (long %)) (long n)))))}
   "a run's written blocks left in the tails"
   {#'ae/tails-after (fn [last own _slots _copies _bs] (or own last))}
   "a step's written blocks left in the last tails"
   {#'ae/tails-after (let [f @#'ae/tails-after] (fn [last own _slots _copies bs] (f last own [] [] bs)))}})

(deftest a-step-s-copies-read-the-last-step-s-tails
  (doseq [opts [nil {:continuous? true}]]
    (testing (pr-str opts)
      (let [{:keys [gaps copies]} (tails-scenario opts)]
        (doseq [[what g] gaps]
          (is (<= g band) (name what)))
        (is (= [[3 1] [3 0] [0 0] [3 0] [2 1] [0 0] [1 1]] copies)
            (str "the run's blocks read alone, then the tails (kept by a step without copies that wrote other blocks),"
                 " read alone again once a run or a step wrote them")))))
  (testing "planted bugs fail it"
    (doseq [[what redefs] tails-planted]
      (is (< 0.01 (reduce max (vals (:gaps (with-redefs-fn redefs #(tails-scenario nil)))))) what))))

(deftest a-chunk-s-last-block-is-in-the-tails
  (with-open [e (ae/engine (merge paged-opts {:model (:f32 @models) :max-branches 12 :continuous? true}))]
    (let [p (vec (take 10 (cycle (:prompt contract-opts))))
          root (br/open e p)
          ;; a chunk of seven after the root's shared partial block (a
          ;; copy), into the next block
          [d1 {:keys [branch]}] (stats-delta e #(br/extend (br/fork root) [1 2 3 4 5 6 7]))
          [d2 [b]] (stats-delta e #(br/advance [(br/fork branch)] [8]))]
      (is (= [1 1] [(:copied-blocks d1 0) (:copy-reads d1 0)]) "the root's block, read alone")
      (is (= [1 0] [(:copied-blocks d2 0) (:copy-reads d2 0)]) "the chunk's last block, from its forward's tails")
      (is (<= (bc/max-diff (forward-row :f32 (into p [1 2 3 4 5 6 7 8])) (br/logprobs b) 128) band))
      (br/dispose! [root branch b]))))

(deftest padding-changes-no-row
  (doseq [min-rows [1 4 16]]
    (with-open [e (ae/engine (merge paged-opts {:model (:f32 @models) :max-branches 12 :min-rows min-rows}))]
      (let [p (:prompt contract-opts)
            b (br/open e p)
            [d [s]] (stats-delta e #(br/advance [(br/fork b)] [7]))]
        (is (= (max 1 min-rows) (+ (:rows d) (:padded-rows d 0))) (str "padded to " min-rows))
        (is (<= (bc/max-diff (forward-row :f32 (conj p 7)) (br/logprobs s) 128) band))
        (br/dispose! [b s])))))

(deftest the-pool-grows-when-its-blocks-run-out
  (with-open [e (ae/engine (merge paged-opts {:model (:f32 @models) :max-branches 12 :pool-blocks 2}))]
    (let [ctx (vec (take 40 (cycle [33 70 79 82 87 65 82 68 0])))
          [d b] (stats-delta e #(loop [b (br/open e (subvec ctx 0 15)) i 15]
                                  (if (= i 40) b (recur (first (br/advance [b] [(ctx i)])) (inc i)))))]
      (is (pos? (:pool-grows d 0)))
      (is (<= (bc/max-diff (forward-row :f32 ctx) (br/logprobs b) 128) band))
      (br/dispose! b))))

;; ---------------------------------------------------------------------------
;; Planted bugs
;; ---------------------------------------------------------------------------

(def ^:private planted
  "Mutations of the paged mode, each with the contract checks it fails."
  {"copy-on-write skipped (a fork writes its parent's block)"
   [{#'paged/prepare (let [prepare paged/prepare]
                       (fn [a s n bs]
                         (let [[a' s' slots copies] (prepare a s n bs)]
                           (if (seq copies)
                             ;; the shared block kept, the copy's block given back
                             (let [[from to] (first copies)]
                               [(paged/release (paged/retain a' [from]) [to]) (update s' :blocks #(conj (pop %) from))
                                (mapv #(paged/slot (update s' :blocks (fn [b] (conj (pop b) from))) bs %) (range (:len s) (+ (long (:len s)) (long n))))
                                []])
                             [a' s' slots copies]))))}
    [:fork-matches-fresh]]
   "a step's rows given to its slots in reverse"
   [{#'ae/host-rows (let [rows @#'ae/host-rows] (fn [pool logits what] (vec (rseq (rows pool logits what)))))}
    [:fork-matches-fresh :batch-matches-single]]
   "the attention blind to each row's last position"
   [{#'array/paged-attention (let [pa array/paged-attention]
                               (fn [q kp vp tables lens & more] (apply pa q kp vp tables (array/maximum (array/- lens 1) 1) more)))}
    [:fork-matches-fresh]]})

(deftest planted-bugs-fail-the-contract
  (doseq [[what [redefs caught-by]] planted]
    (with-open [e (ae/engine (merge paged-opts {:model (:f32 @models) :max-branches 12}))]
      (doseq [c caught-by]
        (testing [what c]
          (is (seq (with-redefs-fn redefs #(bc/run-check (bc/checks c) e contract-opts)))))))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [reason #(:reason (ts/thrown-data (%)))]
    (is (= :invalid-option (reason #(ae/engine (merge paged-opts {:model (:f32 files) :block-size 12})))))
    (is (= :invalid-option (reason #(ae/engine (merge paged-opts {:model (:f32 files) :paged? 1})))))
    (is (= :unsupported (reason #(ae/engine {:model "test/fixtures/gguf/qwen35-tiny-f32.gguf" :paged? true}))) "Qwen3.5")
    (with-open [e (ae/engine {:model (:f32 @models)})]
      (is (= :not-paged (reason #(ae/blocks e)))))))
