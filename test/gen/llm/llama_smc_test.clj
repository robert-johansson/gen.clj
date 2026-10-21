(ns gen.llm.llama-smc-test
  "Token SMC (gen.llm.smc) over llama.cpp, with a lipogram as the potential
   (no token whose bytes hold the letter e: a grammar's mask in miniature),
   by the masked proposal and by the bootstrap: every particle obeys it,
   its trace's score is its log p (score once), a step after the first is
   one read, and the engine's GC backstop never runs.

   Needs llama.cpp's libraries (native/build-llama.sh), the compiled
   binding (clojure -T:build java) and a model, given as
   -J-Dgen.llama.test-model=<a GGUF file> (or GEN_LLAMA_TEST_MODEL), as
   gen.llm.llama-test does; skipped with a note otherwise."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [gen.gfi :as gfi]
            [gen.llm.branching :as br]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.random :as rng]))

(def ^:private model-path
  (or (System/getProperty "gen.llama.test-model") (System/getenv "GEN_LLAMA_TEST_MODEL")))

(defn- llama [sym] @(requiring-resolve (symbol "gen.llm.llama" (name sym))))

(def ^:private available?
  (delay
    (let [skip (fn [why] (println "Skipping the llama.cpp token SMC tests:" why) false)]
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

(defn- prompt [] ((llama 'tokenize) (:engine (setup)) "Write one short sentence about the sea:"))

(defn- lipogram
  "The potential that forbids every token whose bytes hold an e: 0 or
   -Inf, a constraint. The end is allowed."
  [m]
  (let [n (long (:n-vocab (lm/vocab m)))
        banned (boolean-array (map (fn [^bytes b] (boolean (some #(== 0x65 (long %)) b))) (lm/byte-vocab m)))
        allowed? (fn [ctx] (not-any? #(aget banned (int %)) ctx))
        row (let [w (float-array (inc n))]
              (dotimes [t n] (when (aget banned t) (aset w t (unchecked-float ##-Inf))))
              w)]
    (potential/potential n {:prefix #(if (allowed? %) 0.0 ##-Inf)
                            :complete #(if (allowed? %) 0.0 ##-Inf)
                            :logw-next (fn [_] (aclone row))})))

(defn- text-without-e?
  "Does tokens's text, the end tokens aside, hold no e?"
  [m tokens]
  (not (.contains ^String (lm/decode m (vec (remove (:eog (lm/vocab m)) tokens))) "e")))

(deftest a-lipogram-holds-for-every-particle
  (when @available?
    (let [{:keys [lm completion engine]} (setup)
          args [(prompt) 12]
          pot (lipogram lm)]
      (doseq [proposal [:masked :lm]]
        (testing (name proposal)
          (let [s (smc/run completion args 32 {:potential pot :proposal proposal :key (rng/key 1)})
                live (remove :dead? (:particles s))
                trs (remove nil? (smc/traces s))]
            (is (seq live) "particles survive")
            (when (= :masked proposal) (is (= 32 (count live)) "the masked proposal draws only allowed tokens"))
            (is (every? #(text-without-e? lm (:tokens %)) live))
            (is (Double/isFinite (smc/log-ml-estimate s)))
            (is (<= (smc/log-ml-estimate s) 0.0) "the probability of a lipogram, estimated")
            (is (= (mapv :log-p live) (mapv :score trs)) "the particles' log p are the traces' scores")
            (is (every? #(= (:score %) (:weight (gfi/assess completion args (:choices %)))) trs)
                "score once, bit for bit")
            (println (str "llama SMC (" (name proposal) "): " (pr-str (lm/decode lm (:tokens (first live)))))))))
      (is (zero? (:collections (br/info engine))) "no GC backstop"))))

(deftest a-step-after-the-first-is-one-read
  (when @available?
    (let [{:keys [lm completion]} (setup)
          reads #(:reads (lm/info lm))
          s1 (smc/step (smc/initialize completion [(prompt) 8] 32 {:potential (lipogram lm) :proposal :masked
                                                                    :key (rng/key 2)}))
          per-step (loop [s s1 out []]
                     (if (or (smc/done? s) (= 6 (count out)))
                       out
                       (let [r0 (reads)
                             s (smc/step (smc/maybe-resample s {}))]
                         (recur s (conj out (- (reads) r0))))))]
      (is (seq per-step))
      (is (every? #(= 1 %) per-step) (str "reads a step: " per-step)))))
