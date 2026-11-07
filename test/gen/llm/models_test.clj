(ns gen.llm.models-test
  "gen.llm.models (experimental): the families' functions by a model's
   architecture, on the tiny random Qwen3, Qwen3.5 and Qwen3.6 MoE of
   test/fixtures/gguf."
  (:require [clojure.test :refer [deftest is]]
            [gen.array :as array]
            [gen.llm.models :as models]
            [gen.llm.models.qwen3 :as qwen3]
            [gen.llm.models.qwen35 :as qwen35]
            [gen.test-support :as ts]))

(def ^:private model (delay (qwen3/load "test/fixtures/gguf/qwen3-tiny-f32.gguf")))

(def ^:private tokens [[5 17 42 3 88 11 60 2 9]])

(deftest a-model-s-family-by-its-architecture
  (let [m @model
        mask (array/array [[0.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0]])]
    (is (= "qwen3" (models/architecture m)))
    (is (= "qwen35" (models/architecture (assoc-in m [:config :arch] "qwen35"))) ":arch first")
    (is (= (ts/host (qwen3/log-probs m tokens {:chunk 48})) (ts/host (models/log-probs m tokens {:chunk 48}))))
    (is (= (ts/host (qwen3/nll m tokens {:mask mask})) (ts/host (models/nll m tokens {:mask mask}))))
    (is (= (ts/host (qwen3/nll m tokens)) (ts/host (models/nll m tokens))))
    (is (= {:reason :unsupported :fn 'nll :architecture "llama"}
           (ts/thrown-data (models/nll (assoc-in m [:config :architecture] "llama") tokens))))
    (is (= :unsupported (:reason (ts/thrown-data (models/log-probs {:config {}} tokens)))))))

(deftest qwen35-s-family-by-its-architectures
  (doseq [[arch file] [["qwen35" "test/fixtures/gguf/qwen35-tiny-f32.gguf"] ["qwen35moe" "test/fixtures/gguf/qwen35moe-tiny-q8_0.gguf"]]
          :let [m (qwen35/load file)
                mask (array/array [[0.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0]])]]
    (is (= arch (models/architecture m)))
    (is (= (ts/host (qwen35/log-probs m tokens {:chunk 48})) (ts/host (models/log-probs m tokens {:chunk 48}))) arch)
    (is (= (ts/host (qwen35/nll m tokens {:mask mask :checkpoint? true})) (ts/host (models/nll m tokens {:mask mask :checkpoint? true})))
        arch)))
