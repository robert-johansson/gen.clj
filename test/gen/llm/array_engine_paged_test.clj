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
   - forks share their blocks: a fork copies nothing; the first holder to
     write a shared block filled in part appends in place (an LM's kept
     entry and the fork that goes on), a second writer copies it
     (copy-on-write, within its step), and nothing else is copied; the
     parent, the root and a resampled duplicate written after their forks
     each read their own history; a disposed branch's blocks are free
     again (planted bugs: a second writer not copying, the written
     offsets not advanced);
   - a step pads its rows to a power of 2 (:min-rows at least), and the
     padding changes no row;
   - the pool grows when its blocks run out, keeping every branch;
   - a score whose read fails in a call consuming the pools loses every
     branch, loudly, in the paged and the continuous modes; a step or
     read whose rows fail to normalize leaves its branch and its blocks
     as they were, and the next step is given the pools the failed one
     wrote (steps consuming their pools as MLX's do, on any backend:
     donation-checked). Each failed before its fix (genclj-5t5l): the
     scored branches stayed live over emptied slots, the next step was
     given the pools the failed one had consumed, a read left its slot
     ahead of its branch;
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

(deftest forks-share-blocks-and-a-second-writer-copies
  (with-open [e (ae/engine (merge paged-opts {:model (:f32 @models) :max-branches 12}))]
    (let [p (vec (take 12 (cycle (:prompt contract-opts))))  ; a block and a half of 8
          free0 (:free (ae/blocks e))
          root (br/open e p)
          kids (br/fork root 3)
          held #(- (long free0) (long (:free (ae/blocks e))))]
      (is (= 2 (held)) "the root's two blocks; three forks share them")
      (let [[d stepped] (stats-delta e #(br/advance kids [1 2 3]))]
        (is (= {:steps 1 :rows 3 :padded-rows 1 :copied-blocks 2} d)
            (str "three rows padded to 4: the first fork appends to the shared partial block in place (the root"
                 " never wrote past its end), the other two copy it"))
        (is (= 4 (held)) "the root's two, two copies")
        (doseq [[b t] (map vector stepped [1 2 3])]
          (is (<= (bc/max-diff (forward-row :f32 (conj p t)) (br/logprobs b) 128) band)))
        (is (<= (bc/max-diff (forward-row :f32 p) (br/logprobs root) 128) band) "the root's own row, read before")
        (br/dispose! root)
        (is (= 4 (held)) "the root disposed: its blocks the first fork's and shared")
        (let [[d2 s2] (stats-delta e #(br/advance stepped [4 5 6]))]
          (is (zero? (:copied-blocks d2 0)) "each holds its partial block alone: written in place")
          (br/dispose! s2))
        (is (= free0 (:free (ae/blocks e))) "every block free again")))))

(defn- writers-scenario
  "Forks and their parents written in turn, on a fresh paged engine (opts
   over paged-opts): {:gaps {what max |Δ log p| against the forward}
   :copies [copied-blocks by step]}."
  [opts]
  (with-open [e (ae/engine (merge paged-opts {:model (:f32 @models) :max-branches 16} opts))]
    (let [p (vec (take 10 (cycle (:prompt contract-opts))))  ; a block and a quarter of 8
          gap (fn [bs ctxs] (reduce max 0.0 (map #(bc/max-diff (forward-row :f32 %2) (br/logprobs %1) 128) bs ctxs)))
          copies #(:copied-blocks % 0)
          root (br/open e p)
          ;; an LM's pattern: an entry kept, a fork of it going on; the
          ;; fork appends in place
          [d1 [a]] (stats-delta e #(br/advance [(br/fork root)] [1]))
          [d2 [b]] (stats-delta e #(br/advance [(br/fork a)] [2]))
          ;; the parent written after its fork: a second writer, it copies
          [d3 [a']] (stats-delta e #(br/advance [a] [3]))
          ;; the root itself, two writers back
          [d4 [r']] (stats-delta e #(br/advance [root] [4]))
          ;; resampled: two forks of b in one step, the second copies
          [d5 [c1 c2]] (stats-delta e #(br/advance [(br/fork b) (br/fork b)] [5 6]))
          ;; b's extend after them, a third writer
          x (:branch (br/extend b [7 8]))
          out {:gaps {:a-fork (gap [b] [(conj p 1 2)])
                      :its-parent (gap [a'] [(conj p 1 3)])
                      :the-root (gap [r'] [(conj p 4)])
                      :resampled (gap [c1 c2] [(conj p 1 2 5) (conj p 1 2 6)])
                      :a-run (gap [x] [(conj p 1 2 7 8)])}
               :copies (mapv copies [d1 d2 d3 d4 d5])}]
      (br/dispose! [a' r' c1 c2 x])
      out)))

(def ^:private writers-planted
  "Mutations of the books' written offsets, each failing writers-scenario's
   gaps."
  {"a second writer not copying (the written offsets taken to be its own)"
   {#'paged/prepare (let [prepare paged/prepare]
                      (fn [a s ^long n ^long bs] (prepare (cond-> a (seq (:blocks s)) (assoc-in [:written (peek (:blocks s))] (rem (long (:len s)) (long bs))))
                                              s n bs)))}
   "the written offsets not advanced"
   {#'paged/prepare (let [prepare paged/prepare]
                      (fn [a s ^long n ^long bs] (let [[a' s' slots copies] (prepare a s n bs)]
                                       [(assoc a' :written (merge (:written a') (select-keys (:written a) (keys (:written a')))))
                                        s' slots copies])))}})

(deftest a-second-writer-copies-and-every-history-holds
  (doseq [opts [nil {:continuous? true}]]
    (testing (pr-str opts)
      (let [{:keys [gaps copies]} (writers-scenario opts)]
        (doseq [[what g] gaps]
          (is (<= g band) (name what)))
        (is (= [0 0 1 1 1] copies)
            "forks of kept entries in place; the parent, the root and a resampled duplicate copy"))))
  (testing "planted bugs fail it"
    (doseq [[what redefs] writers-planted]
      ;; loud counts as failing: two rows of a step written at one slot
      (is (< 0.01 (try (reduce max (vals (:gaps (with-redefs-fn redefs #(writers-scenario nil)))))
                       (catch clojure.lang.ExceptionInfo _ ##Inf)))
          what))))

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
;; Failures (genclj-5t5l)
;; ---------------------------------------------------------------------------

(def ^:private failing-donations
  "Every call that consumes the pools fails (lose-all! then runs, as after
   a real failure): the redefs."
  {#'ae/donating (let [donating @#'ae/donating]
                   (fn [ctx _] (donating ctx #(throw (ex-info "planted" {:reason :planted})))))})

(deftest a-failed-score-loses-every-branch-the-pools-held
  ;; a score's read failing in a call that consumes the pools, in the
  ;; paged mode and its continuous mode: every branch is lost, loudly
  ;; (:lost-slots, the branches freed), and the engine reads on from new
  ;; pools
  (doseq [opts [paged-opts (assoc paged-opts :continuous? true)]]
    (with-open [e (ae/engine (merge opts {:model (:f32 @models) :max-branches 12}))]
      (testing (if (:continuous? opts) "continuous" "paged")
        (let [p (:prompt contract-opts)
              b (br/open e p)
              other (br/open e [1 2 3])
              data (with-redefs-fn failing-donations #(ts/thrown-data (br/score b [4 5 6])))]
          (is (= :planted (:reason data)))
          (is (seq (:lost-slots data)) "named lost")
          (is (not (br/live? b)) "the scored branch is lost")
          (is (not (br/live? other)) "as is every other")
          (is (= :not-live (bc/reason #(br/advance [b] [4]))))
          (is (zero? (:live (br/info e))))
          (let [c (br/open e p)]
            (is (<= (bc/max-diff (forward-row :f32 p) (br/logprobs c) 128) band) "a new branch, from new pools")
            (br/dispose! c)))))))

(defn- donation-checked
  "compile-paged-step whose steps consume their pools as MLX's donation
   does, on any backend: a step given pools an earlier step consumed is
   loud (:donated)."
  [compile-step]
  (fn [m opts]
    (let [step (compile-step m opts)
          consumed (java.util.Collections/newSetFromMap (java.util.IdentityHashMap.))]
      (fn [pools & args]
        (let [leaves (array/tree-leaves pools)]
          (when (some #(.contains consumed %) leaves)
            (throw (ex-info "a step given consumed pools" {:reason :donated})))
          (.addAll consumed leaves)
          (apply step pools args))))))

(def ^:private failing-rows
  "The rows' normalization fails: the redefs."
  {#'ae/host-rows (fn [& _] (throw (ex-info "planted" {:reason :planted})))})

(deftest a-failed-normalization-leaves-the-slots-as-they-were
  ;; a step's or read's rows fail to normalize after it has written the
  ;; pools (consuming those it was given): loud, the branch and its blocks
  ;; as they were, and the next step given the pools written, not those
  ;; consumed
  (with-open [e (with-redefs-fn {#'qwen3/compile-paged-step (donation-checked qwen3/compile-paged-step)}
                  #(ae/engine (merge paged-opts {:model (:f32 @models) :max-branches 12})))]
    (let [p (:prompt contract-opts)
          more [5 6 7 8 9 10 11 12]
          failed (fn [f]
                   (let [b (br/open e p)
                         free (:free (ae/blocks e))
                         data (with-redefs-fn failing-rows #(ts/thrown-data (f b)))]
                     (is (= :planted (:reason data)))
                     (is (nil? (:lost-slots data)))
                     (is (br/live? b) "the branch as it was")
                     (is (= free (:free (ae/blocks e))) "its blocks as they were")
                     b))]
      (testing "a step"
        (let [[s] (br/advance [(failed #(br/advance [%] [4]))] [4])]
          (is (<= (bc/max-diff (forward-row :f32 (conj p 4)) (br/logprobs s) 128) band) "it steps on")
          (br/dispose! s)))
      (testing "a read"
        (let [{x :branch} (br/extend (failed #(br/extend % more)) more)
              [s] (br/advance [x] [4])]
          (is (<= (bc/max-diff (forward-row :f32 (into p more)) (br/logprobs x) 128) band) "it reads on")
          (is (<= (bc/max-diff (forward-row :f32 (conj (into p more) 4)) (br/logprobs s) 128) band) "and steps on")
          (br/dispose! s))))))

;; ---------------------------------------------------------------------------
;; Planted bugs
;; ---------------------------------------------------------------------------

(def ^:private planted
  "Mutations of the paged mode, each with the contract checks it fails."
  {"copy-on-write skipped (a fork writes its parent's block)"
   [{#'paged/prepare (let [prepare paged/prepare]
                       (fn [a s ^long n ^long bs]
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
    (is (= :unsupported (reason #(ae/engine {:model "test/fixtures/gguf/qwen35-tiny-f32.gguf" :paged? true :continuous? true})))
        "Qwen3.5 continuously")
    (with-open [e (ae/engine {:model (:f32 @models)})]
      (is (= :not-paged (reason #(ae/blocks e)))))))
