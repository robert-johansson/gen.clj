(ns gen.llm.array-engine-qwen35-paged-test
  "gen.array's engine in its paged mode (:paged?, genclj-6qb6) running
   Qwen3.5's family: its gated attention layers' keys and values in the
   pool of blocks, its GatedDeltaNet layers' states a row each, shared by
   forks as the static mode's groups are. On the tiny random Qwen3.5 and
   Qwen3.5-MoE of test/fixtures/gguf (3 GatedDeltaNet blocks, then a
   gated attention block; F32), with the toy tokenizer's 128 tokens, on
   the backend under test:
   - the Branching contract (gen.llm.branching-contract), the LM contract
     (gen.llm.lm-contract) and the token generative function's and token
     SMC's checks (gen.llm.token-contract), within gen.llm.array-engine-
     test's band;
   - a branch's rows are qwen35's forward's, through steps that cross
     blocks, stacked steps, lane steps and steps reading a shared group
     in place, forks appending in place and copying (a second writer);
   - qwen35/paged-step's rows are the concatenating cache's (the model
     alone).
   Planted bugs fail them (with-redefs-fn): a stacked step's states all
   the first row's, a lane's rows given in reverse, a shared group's
   rows read in reverse, a fork's copy skipped, and a run's states
   zeroed."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array :as array]
            [gen.array.paged :as gpaged]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lm :as lm]
            [gen.llm.lm-contract :as lc]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.llm.paged :as paged]
            [gen.llm.token :as token]
            [gen.llm.token-contract :as tc]
            [gen.llm.toy-tokenizer :as tt]
            [gen.test-support :as ts]))

(def ^:private files
  {:dense "test/fixtures/gguf/qwen35-tiny-f32.gguf"
   :moe "test/fixtures/gguf/qwen35moe-tiny-f32.gguf"})

(def ^:private models (delay (into {} (map (fn [[k f]] [k (qwen35/load f)])) files)))

(def ^:private band (if (ts/jvm?) 2e-6 2e-5))

(def ^:private tokenizer (tt/tokenizer 128))

(def ^:private paged-opts
  "Small blocks and pools, so that steps cross blocks, forks share partial
   blocks, and the pool grows."
  {:paged? true :block-size 8 :pool-blocks 16 :n-batch 8})

(def ^:private engines (atom {}))

(defn- engine [k]
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
    (is (= {:arch (if (= k :moe) "qwen35moe" "qwen35")} (select-keys (:model (br/info e)) [:arch])))
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

(deftest ^:slow the-token-and-smc-checks-hold
  (with-open [e (ae/engine (merge paged-opts {:model (:dense @models) :max-branches 512}))]
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
  "The log-probabilities after ctx by qwen35's forward of model k,
   normalized on the host in double, as floats."
  ^floats [k ctx]
  (let [[logits] (qwen35/prefill (@models k) [ctx] {:logits :last})
        xs (mapv double (array/->floats logits))
        mx (reduce max xs)
        z (+ mx (Math/log (reduce + (map #(Math/exp (- % mx)) xs))))]
    (float-array (map #(- % z) xs))))

(defn- stats-delta [e f]
  (let [before (ae/stats e)
        v (f)]
    [(into {} (keep (fn [[k n]] (let [d (- (long n) (long (before k)))] (when-not (zero? d) [k d])))) (ae/stats e)) v]))

(defn- scenario
  "Steps of forks, stacked and lane, on a fresh paged engine of model k:
   {:gaps {what max |Δ log p| against the forward} :stats [by step]}."
  [k]
  (with-open [e (ae/engine (merge paged-opts {:model (@models k) :max-branches 16}))]
    (let [p (vec (take 10 (cycle (:prompt contract-opts))))  ; a block and a quarter of 8
          gap (fn [bs ctxs] (reduce max 0.0 (map #(bc/max-diff (forward-row k %2) (br/logprobs %1) 128) bs ctxs)))
          root (br/open e p)
          ;; forks of one row, stacked (its group shared with the root)
          [d1 s1] (stats-delta e #(br/advance (br/fork root 3) [1 2 3]))
          ;; the three, a lane group, in place
          [d2 s2] (stats-delta e #(br/advance s1 [4 5 6]))
          ;; resampled: the second twice, the first once, reversed
          [d3 s3] (stats-delta e #(br/advance [(br/fork (s2 1)) (br/fork (s2 0)) (br/fork (s2 1))] [7 8 9]))
          ;; an extend from a row of a group of three
          x (:branch (br/extend (br/fork (s3 2)) [10 11]))
          [d4 [y]] (stats-delta e #(br/advance [(br/fork x)] [12]))
          ;; forks of the lane's three, its group shared: read in place
          [d5 s5] (stats-delta e #(br/advance (mapv br/fork s2) [13 14 15]))
          out {:gaps {:stacked (gap s1 (map #(conj p %) [1 2 3]))
                      :lane (gap s2 (map #(conj p %1 %2) [1 2 3] [4 5 6]))
                      :resampled (gap s3 [(conj p 2 5 7) (conj p 1 4 8) (conj p 2 5 9)])
                      :after-a-run (gap [x y] [(conj p 2 5 9 10 11) (conj p 2 5 9 10 11 12)])
                      :held (gap s5 (map #(conj p %1 %2 %3) [1 2 3] [4 5 6] [13 14 15]))}
               :stats (mapv #(select-keys % [:stacked-steps :held-steps :copied-blocks]) [d1 d2 d3 d4 d5])}]
      (br/dispose! (concat [root x y] s2 s3 s5))
      out)))

(deftest a-branch-s-rows-are-the-forward-s
  (doseq [k (keys files)]
    (testing (name k)
      (let [{:keys [gaps stats]} (scenario k)]
        (doseq [[what g] gaps]
          (is (<= g band) (name what)))
        (is (= [{:stacked-steps 1 :copied-blocks 2}
                {}
                {:stacked-steps 1 :copied-blocks 1}
                {:held-steps 1}
                {:held-steps 1 :copied-blocks 2}]
               stats)
            (str "three forks stacked (the first appends to the root's block, the others copy it), a lane in place"
                 " (its padding row after its rows), resampled forks stacked (a duplicate copies), the fork of"
                 " a run's branch, appending in place, its row read in place (shared with the run's), and forks"
                 " of the lane's three, their group shared, read in place (two blocks copied: the resample's"
                 " forks of two of the three wrote past them)"))))))

(def ^:private planted
  "Mutations of the paged mode's states, each failing scenario's gaps."
  {"a stacked step's states all the first row's"
   {#'ae/take-rows (let [f @#'ae/take-rows] (fn [states idx] (f states (repeat (count idx) (first idx)))))}
   "a lane's rows in reverse"
   {#'ae/step-states (let [f @#'ae/step-states] (fn [ctx slots r] (let [s (f ctx slots r)]
                                                                     (if (:lane s) (update s :order (comp vec rseq)) s))))}
   "a shared group's rows read in place in reverse"
   {#'ae/step-states (let [f @#'ae/step-states] (fn [ctx slots r] (let [s (f ctx slots r)]
                                                                     (if (seq (:held s)) (update s :order (comp vec rseq)) s))))}
   "a fork's copy skipped"
   {#'gpaged/put-blocks (fn [pool _ _] pool)}
   "a run's states zeroed"
   {#'ae/new-states-group! (let [f @#'ae/new-states-group!]
                             (fn [state states n] (if (== 1 (long n)) (f state (mapv #(update-vals % array/zeros-like) states) n) (f state states n))))}})

(deftest planted-bugs-fail-it
  (doseq [[what redefs] planted]
    (is (< 0.01 (reduce max (vals (:gaps (with-redefs-fn redefs #(scenario :dense)))))) what)))

(deftest a-paged-step-is-the-concatenating-cache-s
  (doseq [[k m] @models]
    (testing (name k)
      (let [bs 8
            ps [(vec (take 3 (cycle [5 9 13]))) (vec (take 11 (cycle [7 3 1 2]))) (vec (take 6 (cycle [4 8])))]
            dense (mapv #(qwen35/prefill m [%] {:logits :last}) ps)
            ;; each prompt's keys and values written into the pools, its
            ;; states a row of the step's
            [a seqs pools] (reduce (fn [[a ss pools] [p [_ cache]]]
                                     (let [[a s slots] (paged/prepare a paged/empty-sequence (count p) bs)
                                           sl (array/array slots {:dtype :int32})
                                           pools (mapv (fn [e [kn vn]] (if (contains? e :k) (zipmap [:k :v] (array/paged-write (:k e) (:v e) kn vn sl)) e))
                                                       pools (qwen35/paged-entries cache 0))]
                                       [a (conj ss s) pools]))
                                   [(paged/allocator 32 1) [] (qwen35/paged-pools m 32 bs)] (map vector ps dense))
            states (apply mapv (fn [& es] (into {} (map (fn [kk] [kk (array/concatenate (mapv kk es) 0)])) (keys (first es))))
                          (map (fn [[_ cache]] (qwen35/paged-states m cache)) dense))
            [_ seqs' slots] (reduce (fn [[a ss sl] s] (let [[a s' [x]] (paged/prepare a s 1 bs)] [a (conj ss s') (conj sl x)]))
                                    [a [] []] seqs)
            i32 #(array/array % {:dtype :int32})
            toks [[21] [22] [23]]
            [logits] (qwen35/paged-step m (mapv merge pools states) (i32 (mapv :len seqs))
                                        (i32 (mapv #(paged/table-row % 2 0) seqs')) (i32 slots) toks)
            refs (mapv (fn [[_ cache] t] (first (qwen35/step m cache [t] {:logits :last}))) dense toks)]
        (is (<= (reduce max (map-indexed (fn [i ref] (double (array/item (array/max (array/abs (array/- (array/slice logits 0 i (inc i)) ref))))))
                                         refs))
                (if (ts/jvm?) 1e-12 2e-4)))))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [reason #(:reason (ts/thrown-data (%)))
        m (:dense @models)
        pools (qwen35/paged-pools m 4 8)
        i32 #(array/array % {:dtype :int32})]
    (is (= :invalid-args (reason #(qwen35/paged-step m pools (i32 [0]) (i32 [[1]]) (i32 [4]) [[7]]))) "no states")
    (is (= :invalid-args (reason #(qwen35/paged-step m (mapv merge pools (qwen35/paged-states m 2)) (i32 [0]) (i32 [[1]]) (i32 [4]) [[7]])))
        "states of two rows for one")))
