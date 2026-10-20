(ns gen.llm.llama-lm-test
  "The LM protocol over the llama.cpp engine: the contract
   (gen.llm.lm-contract) within llama.cpp's float noise, the band of
   gen.llm.llama-test (0.1 nats, twice the largest difference
   bench/gen/llama_noise.clj measured between two computations of one
   distribution on Metal), and bit for bit where a computation repeats; and
   the GGUF's tokenizer as the LM's.

   Needs llama.cpp's libraries (native/build-llama.sh), the compiled
   binding (clojure -T:build java) and a model, given as
   -J-Dgen.llama.test-model=<a GGUF file> (or GEN_LLAMA_TEST_MODEL), as
   gen.llm.llama-test does; skipped with a note otherwise."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as bc]
            [gen.llm.lm :as lm]
            [gen.llm.lm-contract :as contract]
            [gen.llm.toy-engine :as toy]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(defn- llama
  "A function of gen.llm.llama, loaded only when the tests run (it imports
   the compiled binding)."
  [sym]
  @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the llama.cpp LM tests:" why) false)]
      (cond
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(def ^:private engine-opts {:max-branches 16 :n-ctx 2048})

(def ^:private lm-opts {:max-branches 12})

(defn- new-engine ^java.io.Closeable [] ((llama 'engine) (assoc engine-opts :model model-path)))

(def ^:private shared (atom nil))

(defn- engine []
  (or @shared (reset! shared (new-engine))))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [^java.io.Closeable e @shared]
                                      (.close e)
                                      (reset! shared nil))))))

(defn- setup []
  (let [e (engine)
        tok ((llama 'tokenizer) e)]
    {:engine e
     :lm #(lm/lm e tok lm-opts)
     :fresh (fn [] (let [e (new-engine)] [e (lm/lm e ((llama 'tokenizer) e) lm-opts)]))}))

(defn- opts [e]
  {:prompt ((llama 'tokenize) e "The capital of Sweden is a city called")
   :continuation ((llama 'tokenize) e " Stockholm, which is known for its many islands and" {:add-special? false})
   :k 4
   :band 0.1})

(deftest the-contract-holds-within-the-noise
  (when @available?
    (let [{:keys [engine] :as s} (setup)]
      (doseq [[k check] contract/checks]
        (testing k
          (is (= [] (contract/run-check check s (opts engine))))))
      (is (zero? (:live (br/info engine))) "every check closes what it makes"))))

(deftest the-gguf-tokenizer-is-the-lms
  (when @available?
    (let [e (engine)
          t0 (System/nanoTime)
          tok ((llama 'tokenizer) e)
          ms (/ (- (System/nanoTime) t0) 1e6)]
      (println (format "llama tokenizer: %d tokens' bytes in %.0f ms" (:n-vocab tok) ms))
      (with-open [m (lm/lm e tok lm-opts)]
        (let [text "Stockholm är Sveriges huvudstad: 1 000 000 människor, ℝ³ → ∞ 🙂"
              toks (lm/encode m text {:add-special? false})
              bv (lm/byte-vocab m)]
          (is (= (:n-vocab (br/info e)) (count bv)))
          (is (= (seq (.getBytes text "UTF-8")) (seq (byte-array (mapcat bv toks)))) "the tokens' bytes are the text's")
          (is (= text (lm/decode m toks)))
          (is (contains? (:eog (lm/vocab m)) (:eos (lm/vocab m))))))
      (with-open [toy (toy/engine {})]
        (is (= :not-a-llama-engine (bc/reason #((llama 'tokenizer) toy))))))))
