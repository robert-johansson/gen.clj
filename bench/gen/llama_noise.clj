(ns gen.llama-noise
  "The float noise of llama.cpp's distributions, for the band of the
   Branching contract's llama.cpp tests (gen.llm.llama-test): every pair of
   computations the contract compares (a fork advanced against a fresh read,
   a batched advance against a lone one, score and extend against steps, a
   repeated computation), over several prompts and batch sizes. The
   statistic is the contract's: max |Δ log p| over the reference's 64 most
   likely tokens (and, for comparison, over the whole vocabulary).
   Also the costs of the operations.

   Run (with native/build-llama.sh's libraries):
     clojure -M:bench -m gen.llama-noise model.gguf"
  (:require [gen.llm.branching :as br]
            [gen.llm.llama :as llama]))

(set! *warn-on-reflection* true)

(def prompts
  ["The capital of Sweden is a city called"
   "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
   "In probabilistic programming, a generative function is"
   "Once upon a time, in a small village by the sea, there lived an old fisherman who"
   "1, 1, 2, 3, 5, 8, 13,"
   "Translate to French: The weather is lovely today. →"
   "The mitochondria is the powerhouse of"
   "SELECT name, age FROM users WHERE"
   "Dear Sir or Madam, I am writing to"
   "The quick brown fox jumps over the"
   "In 1969, Neil Armstrong became the first person to"
   "To be, or not to be, that is the"
   "import numpy as np\nx = np."
   "Stockholm är Sveriges huvudstad och"
   "The derivative of sin(x) is"
   "Bayes' theorem states that the posterior is proportional to"
   "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
   "A list of prime numbers: 2, 3, 5, 7,"
   "The recipe calls for two cups of flour, one egg, and"
   "Error: undefined is not a function at"])

(defn- top-tokens [^floats lp k]
  (->> (range (alength lp)) (sort-by #(- (aget lp (int %)))) (take k) vec))

(defn- max-diff
  "max |ref_j - got_j| over the top most likely tokens j of ref (the
   contract's statistic, gen.llm.branching-contract)."
  ^double [^floats ref ^floats got top]
  (reduce max 0.0 (map #(Math/abs (- (aget ref (int %)) (aget got (int %)))) (top-tokens ref top))))

(defn- all-diff ^double [^floats a ^floats b]
  (areduce a j m 0.0 (Math/max m (Math/abs (- (aget a j) (aget b j))))))

(defn- fresh [e tokens]
  (let [b (br/open e tokens) lp (br/logprobs b)] (br/dispose! b) lp))

(defn- summary [xs]
  (let [v (vec (sort xs)) n (count v)]
    (format "n %4d  median %.2e  p99 %.2e  max %.2e" n (v (quot n 2)) (v (min (dec n) (long (* 0.99 n)))) (peek v))))

(defn- ms [t0] (/ (- (System/nanoTime) t0) 1e6))

(defn -main [model & _]
  (with-open [e (llama/engine {:model model :max-branches 16 :n-ctx 4096})]
    (println (llama/system-info))
    (println (dissoc (br/info e) :eog))
    (let [acc (atom {})
          worst (atom {})
          current (atom nil)
          note! (fn [k top all]
                  (swap! acc update k (fnil conj []) [top all])
                  (swap! worst update @current (fnil max 0.0) top))]
      (doseq [text prompts
              :let [_ (reset! current text)
                    p (llama/tokenize e text)
                    root (br/open e p)
                    lp0 (br/logprobs root)]]
        ;; the same tokens read again while the first read is live (other
        ;; cells of the unified cache), and the same read twice from the
        ;; same state (determinism)
        (let [again (fresh e p)]
          (note! :reread-elsewhere (max-diff lp0 again 64) (all-diff lp0 again)))
        (let [a (fresh e (conj p 13)) b (fresh e (conj p 13))]
          (note! :repeated-same-state (max-diff a b 64) (all-diff a b)))
        (doseq [k [1 2 4 8 13]]
          (let [ts (top-tokens lp0 k)
                batched (br/advance (br/fork root k) ts)]
            (doseq [[t a] (map vector ts batched)]
              (let [ref (fresh e (conj p t))
                    alone (let [[s] (br/advance [(br/fork root)] [t])] (br/logprobs (doto s br/dispose!)))]
                (note! :fork-vs-fresh (max-diff ref (br/logprobs a) 64) (all-diff ref (br/logprobs a)))
                (note! :batched-vs-alone (max-diff alone (br/logprobs a) 64) (all-diff alone (br/logprobs a)))))
            (br/dispose! batched)))
        ;; score and extend against steps, over a greedy continuation
        (let [cont (loop [b (br/fork root) acc []]
                     (if (= 24 (count acc))
                       (do (br/dispose! b) acc)
                       (let [t (first (top-tokens (br/logprobs b) 1)) [b'] (br/advance [b] [t])]
                         (recur b' (conj acc t)))))
              scores (br/score root cont)
              {:keys [branch] ext :scores} (br/extend (br/fork root) cont)
              steps (loop [b (br/fork root) cs cont acc []]
                      (if-let [t (first cs)]
                        (let [lp (br/logprob b t) [b'] (br/advance [b] [t])] (recur b' (rest cs) (conj acc lp)))
                        (do (br/dispose! b) acc)))
              ref (fresh e (into p cont))]
          (doseq [[s x st] (map vector scores ext steps)]
            (note! :score-vs-step (Math/abs (- (double s) (double st))) nil)
            (note! :extend-vs-step (Math/abs (- (double x) (double st))) nil))
          (note! :extend-vs-fresh (max-diff ref (br/logprobs branch) 64) (all-diff ref (br/logprobs branch)))
          (br/dispose! branch))
        (br/dispose! root))
      (println "\n|Δ log p| over the reference's top 64 tokens (and over the whole vocabulary):")
      (doseq [[k xs] (sort-by key @acc)]
        (println (format "  %-19s top-64 %s" (name k) (summary (map first xs))))
        (when (second (first xs))
          (println (format "  %-19s all    %s" "" (summary (map second xs))))))
      (println (format "overall max (top 64): %.3e" (reduce max (map first (mapcat val @acc)))))
      (println "worst |Δ| (top 64) by prompt:")
      (doseq [[text w] (sort-by (comp - val) @worst)]
        (println (format "  %.2e  %s" w (pr-str (subs text 0 (min 50 (count text))))))))
    ;; costs
    (let [p (llama/tokenize e (first prompts))
          root (br/open e p)
          time (fn [label n f] (f) (let [t0 (System/nanoTime)] (dotimes [_ n] (f)) (println (format "  %-38s %8.3f ms" label (/ (ms t0) n)))))]
      (println "\ncosts (mean):")
      (time "open (prefill of 8 tokens)" 20 #(br/dispose! (br/open e p)))
      (time "fork + dispose" 200 #(br/dispose! (br/fork root)))
      (doseq [k [1 4 8 15]]
        (time (str "advance " k " branches (one decode)") 20
              #(br/dispose! (br/advance (br/fork root k) (repeat k 13)))))
      (time "score 16 tokens" 20 #(br/score root (vec (range 100 116))))
      (br/dispose! root))))
