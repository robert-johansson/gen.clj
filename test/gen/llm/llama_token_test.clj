(ns gen.llm.llama-token-test
  "The token generative function (gen.llm.token) over llama.cpp: the GFI
   laws within one LM, where score once makes them exact although
   llama.cpp's reads differ with their shapes; forced text; completions
   that decode to text; no GC backstop.

   Needs llama.cpp's libraries (native/build-llama.sh), the compiled
   binding (clojure -T:build java) and a model, given as
   -J-Dgen.llama.test-model=<a GGUF file> (or GEN_LLAMA_TEST_MODEL), as
   gen.llm.llama-test does; skipped with a note otherwise."
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [clojure.test.check :as tc]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.llm.branching :as br]
            [gen.llm.lm :as lm]
            [gen.llm.token :as token]
            [gen.random :as rng]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(defn- llama [sym] @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the llama.cpp token tests:" why) false)]
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

(defn- prompt [] ((llama 'tokenize) (:engine (setup)) "The capital of Sweden is"))

(deftest the-laws-hold-within-one-lm
  (when @available?
    (let [{:keys [completion]} (setup)
          p (prompt)
          cases (laws/cases [[:llama completion [p 4] [[p 2] [p 6]]]])]
      (doseq [[k law] (merge laws/all laws/implementation)]
        (let [r (tc/quick-check 10 (law cases) :seed 20261021)]
          (is (:pass? r) (str k " " (pr-str (select-keys r [:fail :shrunk])))))))))

(deftest forced-text-and-decoded-completions
  (when @available?
    (let [{:keys [lm completion engine]} (setup)
          p (prompt)
          stockholm (lm/encode lm " Stockholm" {:add-special? false})
          {tr :trace w :weight} (gfi/generate completion [p 12] (token/forced stockholm) (rng/key 1))
          text (lm/decode lm (:retval tr))]
      (is (= stockholm (subvec (:retval tr) 0 (count stockholm))))
      (is (< -20.0 w 0.0) "a likely continuation, scored by the model")
      (is (.startsWith ^String text " Stockholm"))
      (is (= (:score tr) (:weight (gfi/assess completion [p 12] (:choices tr)))) "score once, bit for bit")
      (println "llama completion:" (pr-str text))
      (is (zero? (:collections (br/info engine))) "no GC backstop"))))
