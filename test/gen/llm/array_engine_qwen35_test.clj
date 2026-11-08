(ns gen.llm.array-engine-qwen35-test
  "gen.array's engine (gen.llm.array-engine) in its static mode
   (:paged? false; the paged mode, the default, is
   gen.llm.array-engine-qwen35-paged-test's) running Qwen3.5's family
   (gen.llm.models.qwen35): the tiny random Qwen3.5 and Qwen3.5-MoE of
   test/fixtures/gguf (gen.llm.models.qwen35-test) as F32, loaded by the
   engine from the GGUF (its architecture read from the header), with the
   toy tokenizer's 128 tokens, on the backend under test:
   - the Branching contract (gen.llm.branching-contract) within
     gen.llm.array-engine-test's band: its forks share a group's
     recurrent states as they share its slots, and a stacked step copies
     them;
   - a branch's rows are qwen35's forward's (log-softmax of its logits),
     read in forwards of 8 tokens, the GatedDeltaNet states carried from
     one to the next.
   A planted bug fails the contract: a stacked step's recurrent states
   read from the first row (every row's state the first's), failing the
   checks named in planted-bugs-fail-the-contract."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.array :as array]
            [gen.llm.array-engine :as ae]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.test-support :as ts]))

(def ^:private files
  {:dense "test/fixtures/gguf/qwen35-tiny-f32.gguf"
   :moe "test/fixtures/gguf/qwen35moe-tiny-f32.gguf"})

(def ^:private band
  "gen.llm.array-engine-test's: 2e-6 on the JVM backend (float rounding),
   2e-5 on MLX."
  (if (ts/jvm?) 2e-6 2e-5))

(def ^:private engines (atom {}))

(defn- engine
  "The shared engine of model k (opened once, from its file): small
   blocks and reads, so that capacities grow and reads take several
   forwards."
  [k]
  (or (@engines k)
      (get (swap! engines assoc k (ae/engine {:model (files k) :max-branches 80 :block 16 :n-batch 8 :paged? false})) k)))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (doseq [[_ ^java.io.Closeable e] @engines] (.close e))
                                    (reset! engines {})))))

(def ^:private contract-opts
  {:prompt [95 51 72 69 0 67 65 80 73 84 65 76 0 73 83] :continuation [0 83 84 79 67 75 72 79 76 77 12 0 73]
   :k 4 :band band :top 64})

(deftest the-branching-contract-holds-within-the-band
  (doseq [k (keys files)
          :let [e (engine k)]]
    (is (= {:arch (if (= k :moe) "qwen35moe" "qwen35") :layers 4} (select-keys (:model (br/info e)) [:arch :layers])))
    (doseq [[c check] bc/checks]
      (testing [k c]
        (is (= [] (bc/run-check check e contract-opts)))))
    (is (zero? (:live (br/info e))) "every check disposes what it makes")))

(defn- forward-row
  "The log-probabilities after ctx by qwen35's forward of model k, its
   logits normalized on the host in double, as floats."
  ^floats [m ctx]
  (let [[logits] (qwen35/prefill m [ctx] {:logits :last})
        xs (mapv double (array/->floats logits))
        mx (reduce max xs)
        z (+ mx (Math/log (reduce + (map #(Math/exp (- % mx)) xs))))]
    (float-array (map #(- % z) xs))))

(deftest a-branch-s-rows-are-the-forward-s
  (doseq [k (keys files)
          :let [e (engine k)
                m (qwen35/load (files k))
                prompt (:prompt contract-opts)
                root (br/open e prompt)]]
    (try
      (let [[b] (br/advance [(br/fork root)] [(first (:continuation contract-opts))])]
        (try
          (is (<= (bc/max-diff (forward-row m prompt) (br/logprobs root) 128) band) (str (name k) ": a read of 15 in forwards of 8"))
          (is (<= (bc/max-diff (forward-row m (conj prompt (first (:continuation contract-opts)))) (br/logprobs b) 128) band)
              (str (name k) ": a step"))
          (finally (br/dispose! b))))
      (finally (br/dispose! root)))))

(deftest planted-bugs-fail-the-contract
  (let [select-rows qwen35/select-rows
        e (engine :dense)
        fails (fn [redefs checks]
                (with-redefs-fn redefs
                  #(some (fn [c] (seq (bc/run-check (bc/checks c) e contract-opts))) checks)))]
    (testing "a stacked step's recurrent states all the first row's"
      (is (fails {#'qwen35/select-rows (fn [cache idx]
                                         (let [c (select-rows cache idx)
                                               first-row (fn [x] (array/broadcast-to (array/slice x 0 0 1) (array/shape x)))]
                                           (update c :layers (fn [ls] (mapv #(if (:state %) (update-vals % first-row) %) ls)))))}
                 (keys bc/checks))))))
