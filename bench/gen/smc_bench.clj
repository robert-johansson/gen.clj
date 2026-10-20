(ns gen.smc-bench
  "Token SMC's costs (gen.llm.smc) on the toy engine: a step of n particles
   against the batched LM call under it, with no potential, a cheap one (a
   mask, as a grammar's) and the masked proposal; and at Qwen3's
   vocabulary (151,936 tokens) the part of a particle's step that is the
   SMC's own.

   Run: clojure -M:bench -m gen.smc-bench"
  (:require [gen.llm.lm :as lm]
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

(defn- per-step [n-vocab]
  (println (format "\n## A step of n particles over 32 steps, toy engine (%d tokens), ms a step (resampling at ESS < n/2)" n-vocab))
  (println "| particles | the LM's batch alone | SMC, no potential | a mask, bootstrap | a mask, masked proposal |")
  (println "|---|---|---|---|---|")
  (doseq [n [8 32 64]]
    (with-open [^java.io.Closeable e (toy/engine {:n-vocab n-vocab :max-branches 160 :n-ctx 128})
                m (lm/lm e (tt/tokenizer n-vocab) {:max-cells 1000000})]
      (let [gf (token/completion m {:end-tokens #{}})
            prompt [95 40 73]
            seed (volatile! 0)
            run (fn [opts] (fn [] (smc/run gf [prompt 32] n (assoc opts :key (rng/key (vswap! seed inc))))))
            raw (fn [] (loop [ctxs (vec (for [i (range n)] (conj prompt (mod (+ i (vswap! seed inc)) 90)))) t 0]
                         (when (< t 32)
                           (lm/batch-next-logprobs m ctxs)
                           (recur (mapv #(conj % 1) ctxs) (inc t)))))]
        (println (format "| %d | %.2f | %.2f | %.2f | %.2f |" n
                         (/ (ms raw) 32) (/ (ms (run {})) 32)
                         (/ (ms (run {:potential (mask n-vocab)})) 32)
                         (/ (ms (run {:potential (mask n-vocab) :proposal :masked})) 32)))))))

(defn -main [& _]
  (per-step 97)
  (per-step 151936)
  (shutdown-agents))
