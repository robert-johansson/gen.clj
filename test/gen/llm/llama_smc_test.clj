(ns gen.llm.llama-smc-test
  "Token SMC (gen.llm.smc) over llama.cpp, with a lipogram as the potential
   (no token whose bytes hold the letter e: a grammar's mask in miniature),
   by the masked proposal and by the bootstrap, and with a grammar
   (gen.llm.grammar, a date) by the masked proposal: every particle obeys
   it, its trace's score is its log p (score once), a step after the first
   is one read, and the engine's GC backstop never runs
   (gen.llm.token-contract's checks, which gen.llm.array-engine-test runs
   on gen.array's engine).

   Needs llama.cpp's libraries (native/build-llama.sh), the compiled
   binding (clojure -T:build java) and a model, given as
   -J-Dgen.llama.test-model=<a GGUF file> (or GEN_LLAMA_TEST_MODEL), as
   gen.llm.llama-test does; skipped with a note otherwise."
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [gen.llm.lm :as lm]
            [gen.llm.token :as token]
            [gen.llm.token-contract :as contract]
            [gen.test-support :as ts]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(defn- llama [sym] @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the llama.cpp token SMC tests:" why) (ts/native-unavailable! :llama why) false)]
      (cond
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(def ^:private shared (atom nil))

(defn- setup
  "{:engine :lm :completion}, made once."
  []
  (or @shared
      (reset! shared (let [e ((llama 'engine) {:model model-path :max-branches 255 :n-ctx 8192})
                           m (lm/lm e ((llama 'tokenizer) e))]
                       {:engine e :lm m :completion (token/completion m)}))))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [{:keys [^java.io.Closeable engine lm]} @shared]
                                      (.close ^java.io.Closeable lm)
                                      (.close engine)
                                      (reset! shared nil))))))

(defn- opts []
  (let [e (:engine (setup))]
    {:sea-prompt ((llama 'tokenize) e "Write one short sentence about the sea:")
     :date-prompt ((llama 'tokenize) e "Today's date, written as YYYY-MM-DD, is")
     :particles 32}))

(deftest a-lipogram-holds-for-every-particle
  (ts/when-native :llama available?
    (is (= [] (contract/run-check contract/check-lipogram (setup) (opts))))))

(deftest a-grammar-holds-for-every-particle
  (ts/when-native :llama available?
    (is (= [] (contract/run-check contract/check-grammar (setup) (opts))))))

(deftest a-step-after-the-first-is-one-read
  (ts/when-native :llama available?
    (is (= [] (contract/run-check contract/check-one-read (setup) (opts))))))
