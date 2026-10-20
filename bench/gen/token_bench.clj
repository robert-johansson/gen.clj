(ns gen.token-bench
  "The token generative function's costs (gen.llm.token) on the toy engine:
   per token, what a completion adds to the LM calls under it; each
   operation's cost on a 50-token completion; and at Qwen3's vocabulary
   (151,936 tokens) the parts of a token's cost that are the
   completion's own (the row's copy, the host draw, the score's lookup).

   Run: clojure -M:bench -m gen.token-bench"
  (:require [criterium.core :as crit]
            [gen.choicemap :as cm]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.llm.lm :as lm]
            [gen.llm.token :as token]
            [gen.llm.toy-engine :as toy]
            [gen.llm.toy-tokenizer :as tt]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- us [f] (* 1e6 (first (:mean (crit/quick-benchmark* f {})))))

(def ^:private prompt [95 40 73])

(defn- argmax ^long [^floats row]
  (loop [t 0 best 0]
    (if (< t (alength row)) (recur (inc t) (if (> (aget row t) (aget row best)) t best)) best)))

(defn- per-token []
  (println "\n## Per token, toy engine (97 tokens), µs: a 50-token completion against the LM calls alone")
  (with-open [^java.io.Closeable e (toy/engine {:max-branches 64 :n-ctx 1024})
              m (lm/lm e (tt/tokenizer 97) {:max-cells 65536})]
    (let [gf (token/completion m {:end-tokens #{}})
          seed (volatile! 0)
          ;; each run new tokens, so that every token is a one-token read
          sim (us #(gfi/simulate gf [prompt 50] (rng/key (vswap! seed inc))))
          raw (us #(loop [ctx (conj prompt (mod (vswap! seed inc) 90)) i 0]
                     (when (< i 50)
                       (let [t (argmax (lm/next-logprobs m ctx))] (recur (conj ctx t) (inc i))))))
          tr (gfi/simulate gf [prompt 50] (rng/key 1))
          held-assess (us #(gfi/assess gf [prompt 50] (:choices tr)))
          update-25 (us #(gfi/update tr (cm/choicemap 25 (mod (vswap! seed inc) 90)) (rng/key 2)))
          regen-25 (us #(gfi/regenerate tr (token/from 25) (rng/key (vswap! seed inc))))]
      (println "| | µs |")
      (println "|---|---|")
      (println (format "| simulate, per token | %.1f |" (/ sim 50)))
      (println (format "| the LM calls alone (next-logprobs and an argmax), per token | %.1f |" (/ raw 50)))
      (println (format "| assess of a held 50-token completion (scores from the trie) | %.1f |" held-assess))
      (println (format "| update of token 25 (the 24 after it rescored in one read) | %.1f |" update-25))
      (println (format "| regenerate from token 25 (25 tokens drawn) | %.1f |" regen-25)))))

(defn- at-qwen-size []
  (println "\n## At Qwen3's vocabulary (151,936 tokens), µs per token")
  (with-open [^java.io.Closeable e (toy/engine {:n-vocab 151936 :max-branches 16 :n-ctx 256})
              m (lm/lm e (tt/tokenizer 151936))]
    (let [ctx (conj prompt 7)
          _ (lm/next-logprobs m ctx)
          row (lm/next-logprobs m ctx)
          dist (token/dist row)
          key (rng/key 1)]
      (println "| | µs |")
      (println "|---|---|")
      (doseq [[what f] [["the toy engine's row (its \"decode\")" #(toy/toy-logprobs 151936 ctx)]
                        ["next-logprobs of a held context (the row's copy)" #(lm/next-logprobs m ctx)]
                        ["the host draw (inverse CDF of one uniform)" #(dc/sample dist key)]
                        ["the score, from the LM's ledger" #(lm/score m ctx [17])]]]
        (println (format "| %s | %.0f |" what (us f)))))))

(defn -main [& _]
  (per-token)
  (at-qwen-size)
  (shutdown-agents))
