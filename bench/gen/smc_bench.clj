(ns gen.smc-bench
  "Token SMC's costs (gen.llm.smc) on the toy engine: a step of n particles
   against the batched LM call under it, with no potential, a cheap one (a
   mask, as a grammar's) and the masked proposal; and at Qwen3's
   vocabulary (151,936 tokens) the part of a particle's step that is the
   SMC's own, by the host pool's size; with a GGUF model, the same over
   llama.cpp, and under a grammar (gen.llm.grammar).

   Run: clojure -M:bench -m gen.smc-bench
        clojure -M:bench -m gen.smc-bench host     (the host pool alone)
        clojure -J-Dgen.llama.lib=<lib> -M:bench -m gen.smc-bench model.gguf"
  (:require [clojure.string :as str]
            [gen.llm.branching :as br]
            [gen.llm.grammar :as grammar]
            [gen.llm.lm :as lm]
            [gen.llm.potential :as potential]
            [gen.llm.smc :as smc]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.llm.toy-tokenizer :as tt]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- ms ^double [f]
  (f) ;; warm
  (let [t0 (System/nanoTime)]
    (dotimes [_ 5] (f))
    (/ (- (System/nanoTime) t0) 5e6)))

(defn- mask
  "A potential that forbids the tokens 0 to 9 (a cheap logw-next, as a
   grammar's mask)."
  [n-vocab]
  (let [allowed? #(not-any? (fn [t] (< (long t) 10)) %)]
    (potential/potential n-vocab
                         {:prefix #(if (allowed? %) 0.0 ##-Inf)
                          :complete #(if (allowed? %) 0.0 ##-Inf)
                          :logw-next (fn [_] (let [w (float-array (inc n-vocab))]
                                               (dotimes [t 10] (aset w t (unchecked-float ##-Inf)))
                                               w))})))

(defn- step-row
  "A table row for n particles of steps steps over the LM m from prompt:
   ms a step, for the LM's batch alone and for SMC; with a grammar, also
   for SMC under it by the masked proposal."
  [m prompt n steps grammar]
  (let [n-vocab (long (:n-vocab (lm/vocab m)))
        gf (token/completion m {:end-tokens #{}})
        seed (volatile! 0)
        run (fn [opts] (fn [] (smc/run gf [prompt steps] n (assoc opts :key (rng/key (vswap! seed inc))))))
        raw (fn [] (loop [ctxs (vec (for [i (range n)] (conj prompt (mod (+ i (vswap! seed inc)) 90)))) t 0]
                     (when (< t steps)
                       (lm/batch-next-logprobs m ctxs)
                       (recur (mapv #(conj % 1) ctxs) (inc t)))))]
    (str (format "| %d | %.2f | %.2f | %.2f | %.2f |" n
                 (/ (ms raw) steps) (/ (ms (run {})) steps)
                 (/ (ms (run {:potential (mask n-vocab)})) steps)
                 (/ (ms (run {:potential (mask n-vocab) :proposal :masked})) steps))
         (when grammar
           (format " %.2f |" (/ (ms (run {:potential grammar :proposal :masked})) steps))))))

(def ^:private header
  ["| particles | the LM's batch alone | SMC, no potential | a mask, bootstrap | a mask, masked proposal |"
   "|---|---|---|---|---|"])

(def ^:private grammar-language
  "The llama bench's grammar: digits, commas and spaces, any number, so
   that every particle can go on for the bench's 16 tokens."
  "[0-9, ]*")

(defn- per-step [n-vocab]
  (println (format "\n## A step of n particles over 32 steps, toy engine (%d tokens), ms a step (resampling at ESS < n/2)" n-vocab))
  (run! println header)
  (doseq [n [8 32 64]]
    (with-open [^java.io.Closeable e (toy/engine {:n-vocab n-vocab :max-branches 160 :n-ctx 128})
                m (lm/lm e (tt/tokenizer n-vocab) {:max-cells 1000000})]
      (println (step-row m [95 40 73] n 32 nil)))))

(defn- with-pool
  "(f pool) for a host pool of threads threads, or (f false) for none."
  [threads f]
  (if threads
    (with-open [^java.util.concurrent.ExecutorService pool (smc/host-pool threads)] (f pool))
    (f false)))

(defn- host-costs
  "The host pool's effect at Qwen3's vocabulary on the toy engine: a step
   of n particles over 16 steps, never resampling, so that every step reads
   n contexts, as the LM's batch alone does; ms a step."
  []
  (let [n-vocab 151936
        steps 16]
    (println "\n## The host pool: a step of n particles over 16 steps, toy engine (151,936 tokens), no resampling, ms a step")
    (println "| particles | proposal | the LM's batch alone | calling thread | 2 threads | 4 threads | 8 threads |")
    (println "|---|---|---|---|---|---|---|")
    (doseq [n [8 32 64]]
      (with-open [^java.io.Closeable e (toy/engine {:n-vocab n-vocab :max-branches 160 :n-ctx 128})
                  m (lm/lm e (tt/tokenizer n-vocab) {:max-cells 1000000})]
        (let [gf (token/completion m {:end-tokens #{}})
              prompt [95 40 73]
              seed (volatile! 0)
              raw (fn [] (loop [ctxs (vec (for [i (range n)] (conj prompt (mod (+ i (vswap! seed inc)) 90)))) t 0]
                           (when (< t steps)
                             (lm/batch-next-logprobs m ctxs)
                             (recur (mapv #(conj % 1) ctxs) (inc t)))))
              batch (/ (ms raw) steps)]
          (doseq [[label opts] [["bootstrap" {}] ["masked, a mask" {:potential (mask n-vocab) :proposal :masked}]]]
            (println (format "| %d | %s | %.2f | %s |" n label batch
                             (str/join " | "
                                       (for [threads [false 2 4 8]]
                                         (with-pool threads
                                           (fn [pool]
                                             (format "%.2f" (/ (ms #(smc/run gf [prompt steps] n
                                                                             (assoc opts :key (rng/key (vswap! seed inc))
                                                                                    :host-pool pool :ess-threshold 0)))
                                                               steps))))))))))))))

(defn- llama-per-step
  "The same over llama.cpp with model, 16 steps."
  [model]
  (let [engine @(requiring-resolve 'gen.llm.llama/engine)
        tokenizer @(requiring-resolve 'gen.llm.llama/tokenizer)
        tokenize @(requiring-resolve 'gen.llm.llama/tokenize)]
    (with-open [^java.io.Closeable e (engine {:model model :max-branches 160 :n-ctx 8192})
                m (lm/lm e (tokenizer e))]
      (println (format "\n## A step of n particles over 16 steps, llama.cpp: %s (%d tokens), ms a step"
                       (:desc (:model (br/info e))) (:n-vocab (lm/vocab m))))
      (let [t0 (System/nanoTime)
            g (grammar/grammar m grammar-language {:end-tokens #{}})]
        (println (format "The grammar %s: %s, made in %.0f ms" (pr-str grammar-language) (grammar/info g)
                         (/ (- (System/nanoTime) t0) 1e6)))
        (println (str (first header) " a grammar, masked proposal |"))
        (println (str (second header) "---|"))
        (doseq [n [8 32 64]]
          (println (step-row m (tokenize e "Once upon a time") n 16 g))))
      (println (format "\nBackstop collections: %d" (:collections (br/info e)))))))

(defn -main
  "The toy engine's costs; with host, the host pool's alone; with a GGUF
   file, llama.cpp's."
  [& [arg]]
  (case arg
    nil (do (per-step 97) (per-step 151936) (host-costs))
    "host" (host-costs)
    (llama-per-step arg))
  (shutdown-agents))
