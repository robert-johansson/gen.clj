(ns gen.llm.llama-grammar-test
  "Grammars over llama.cpp (gen.llm.grammar, gen.llm.constrained,
   gen.llm.bytes): a real vocabulary's masks, completions under grammars
   that stay in their languages and decode to their strings, the GFI laws
   within one LM, the conditional weights, and a text's probability over
   its tokenizations against its canonical tokenization's.

   Needs llama.cpp's libraries (native/build-llama.sh), the compiled
   binding (clojure -T:build java) and a model, given as
   -J-Dgen.llama.test-model=<a GGUF file> (or GEN_LLAMA_TEST_MODEL), as
   gen.llm.llama-token-test does; skipped with a note otherwise."
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [clojure.test.check :as tc]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.llm.branching :as br]
            [gen.llm.bytes :as bytes]
            [gen.llm.constrained :as constrained]
            [gen.llm.grammar :as grammar]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.token :as token]
            [gen.random :as rng]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(defn- llama [sym] @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the llama.cpp grammar tests:" why) false)]
      (cond
        (nil? model-path) (skip "no model (-J-Dgen.llama.test-model=<a GGUF file>, or GEN_LLAMA_TEST_MODEL)")
        (not (.isFile (java.io.File. ^String model-path))) (skip (str "no model file at " model-path))
        :else (try ((llama 'system-info)) true
                   (catch Throwable t
                     (skip (str (or (ex-message t) (.getName (class t)))
                                " (build with native/build-llama.sh and clojure -T:build java)"))))))))

(def ^:private shared (atom nil))

(defn- setup
  "{:engine :lm :grammars}, made once."
  []
  (or @shared
      (reset! shared (let [e ((llama 'engine) {:model model-path :max-branches 255 :n-ctx 8192})
                           m (lm/lm e ((llama 'tokenizer) e))
                           made (fn [re] (let [t0 (System/nanoTime)
                                               g (grammar/grammar m re)]
                                           (println "llama grammar" (pr-str re) (grammar/info g)
                                                    (format "(%.0f ms)" (/ (- (System/nanoTime) t0) 1e6)))
                                           g))]
                       {:engine e :lm m
                        :phone (made " ?[0-9]{3}-[0-9]{4}")
                        :answer (made " ?(?:yes|no)")
                        :json (made " ?\\{\"city\": \"[A-Z][a-z]+\", \"population\": [1-9][0-9]{0,7}\\}")}))))

(use-fixtures :once (fn [f]
                      (try (f)
                           (finally (when-let [{:keys [^java.io.Closeable engine lm]} @shared]
                                      (.close ^java.io.Closeable lm)
                                      (.close engine)
                                      (reset! shared nil))))))

(defn- prompt [text] ((llama 'tokenize) (:engine (setup)) text))

(defn- pop-end
  "A completion's tokens before its end token."
  [gf tokens]
  (if (and (seq tokens) (contains? (:end-tokens gf) (peek tokens))) (pop tokens) tokens))

(defn- text-of [gf tokens] (lm/decode (:lm gf) (pop-end gf tokens)))

(deftest completions-stay-in-their-languages
  (when @available?
    (let [{:keys [lm phone answer json engine]} (setup)]
      (doseq [[g p n re] [[phone (prompt "Call me at") 12 #" ?[0-9]{3}-[0-9]{4}"]
                          [answer (prompt "Is Stockholm the capital of Sweden? Answer:") 4 #" ?(yes|no)"]
                          [json (prompt "Stockholm as JSON:") 40
                           #" ?\{\"city\": \"[A-Z][a-z]+\", \"population\": [1-9][0-9]{0,7}\}"]]]
        (let [gf (token/completion lm {:potential g})
              trs (mapv #(gfi/simulate gf [p n] (rng/key %)) (range 5))]
          (doseq [tr trs]
            (is (zero? (potential/prefix g (pop-end gf (:retval tr)))) "a prefix of the language")
            (when (contains? (:end-tokens gf) (peek (:retval tr)))
              (is (re-matches re (text-of gf (:retval tr))) "an ended text is one of its strings"))
            (is (= (:score tr) (:weight (gfi/assess gf [p n] (:choices tr)))) "within one LM, bit for bit")
            (is (< Double/NEGATIVE_INFINITY (constrained/conditional-weight tr) Double/POSITIVE_INFINITY)))
          (println "llama under a grammar:" (mapv #(pr-str (text-of gf (:retval %))) trs))))
      (is (zero? (:collections (br/info engine))) "no GC backstop"))))

(deftest the-laws-hold-within-one-lm
  (when @available?
    (let [{:keys [lm phone]} (setup)
          p (prompt "Call me at")
          gf (token/completion lm {:potential phone})
          cases (laws/cases [[:llama gf [p 4] [[p 2] [p 6]]]])]
      (doseq [[k law] (merge laws/all laws/implementation)]
        (let [r (tc/quick-check 10 (law cases) :seed 20261021)]
          (is (:pass? r) (str k " " (pr-str (select-keys r [:fail :shrunk])))))))))

(def ^:private band
  "How far one token's log-probability may differ between two reads of it
   in different batch shapes on llama.cpp (Metal): 0.1 nats, twice the
   largest difference measured on short prompts (5.2e-2 on the M2 Max,
   4.3e-2 on the M4; docs/research/llm-branching.md, Numerics). CPU builds
   need their own."
  0.1)

(deftest a-text-is-at-least-as-likely-as-its-tokenization
  (when @available?
    (let [{:keys [lm]} (setup)
          p (prompt "The capital of Sweden is")
          stockholm (lm/encode lm " Stockholm" {:add-special? false})
          canonical (reduce + (lm/score lm p stockholm))
          b (bytes/byte-lm lm {:beam 8})
          marginal (bytes/prefix-logprob b p " Stockholm")]
      ;; the text's prefix probability counts every tokenization and every
      ;; token after it that begins with the rest: at least the canonical
      ;; tokenization's, read by rows where the score is teacher-forced, so
      ;; within the band for each of its tokens
      (println "llama: log P(\" Stockholm\" ...) over tokenizations" marginal ", canonical" canonical
               "(" (count stockholm) "tokens)")
      (is (<= (- canonical (* (count stockholm) band)) marginal 0.0)))))
