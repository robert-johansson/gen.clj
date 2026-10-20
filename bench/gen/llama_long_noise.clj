(ns gen.llama-long-noise
  "The float noise of llama.cpp's distributions for branches longer than a
   sliding window, for the band of gen.llm.llama-test's long-branch test:
   the Branching contract's own checks (gen.llm.branching-contract), run
   with no band, report every |Δ log p| they compare (over the reference's
   64 most likely tokens), for prompts of four long texts at positions past
   512 tokens (Gemma 3's window) and continuations crossing it.

   Run (with native/build-llama.sh's libraries; the contract is test code):
     clojure -M:test:bench -m gen.llama-long-noise model.gguf"
  (:require [clojure.string :as str]
            [gen.llm.branching :as br]
            [gen.llm.branching-contract :as contract]
            [gen.llm.llama :as llama]
            [gen.llm.long-texts :refer [texts]]))

(defn- summary [xs]
  (let [v (vec (sort xs)) n (count v)]
    (format "n %4d  median %.2e  p99 %.2e  max %.2e" n (v (quot n 2)) (v (min (dec n) (long (* 0.99 n)))) (peek v))))

(defn -main [model & _]
  (with-open [e (llama/engine {:model model :max-branches 8 :n-ctx 4096})] ; as gen.llm.llama-test's
    (println (:desc (:model (br/info e))) "window" (:n-swa (:model (br/info e))))
    (let [seen (atom [])
          here (atom nil)
          checks (select-keys contract/checks [:fork-matches-fresh :batch-matches-single
                                               :score-matches-steps :resample])]
      (binding [contract/*observe* (fn [what d]
                                     (swap! seen conj [(first (str/split what #",| token| by| \d")) d (conj @here what)]))]
        (doseq [[label text] texts
                :let [all (llama/tokenize e text)]
                [p c] (concat (for [len [600 800 1000 1400]] [(subvec all 0 len) (subvec all len (+ len 40))])
                              ;; a continuation crossing the window
                              [[(subvec all 0 472) (subvec all 472 552)]])
                [k check] checks]
          (reset! here [label (count p) (count c)])
          (let [failures (contract/run-check check e {:prompt p :continuation c :k 4 :band Double/MAX_VALUE})]
            (when (seq failures) (println "  " label (count p) k failures)))))
      (println "\n|Δ log p| by comparison (64 most likely tokens):")
      (doseq [[what xs] (sort-by key (group-by first @seen))]
        (println (format "  %-40s %s" what (summary (map second xs)))))
      (println (format "overall max: %.3e" (reduce max (map second @seen))))
      (println "the largest, with their text, prompt and continuation lengths:")
      (doseq [[_ d where] (take 5 (sort-by (comp - second) @seen))]
        (println (format "  %.2e  %s" d (pr-str where)))))))
