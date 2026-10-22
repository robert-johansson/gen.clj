;; P4-01: token SMC ignores the completion's own :potential.
;; Run from the snapshot worktree:
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/P4-01.clj
(require '[gen.gfi :as gfi]
         '[gen.llm.lm :as lm]
         '[gen.llm.potential :as potential]
         '[gen.llm.smc :as smc]
         '[gen.llm.token :as token]
         '[gen.llm.constrained :as constrained]
         '[gen.llm.toy-engine :as toy]
         '[gen.random :as rng])

(def n-vocab 5)
(def end 4)
(def tokenizer
  {:n-vocab n-vocab :bos nil :eos end :eog #{end} :add-bos? false
   :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))})
(def engine (toy/engine {:n-vocab n-vocab :max-branches 128 :n-ctx 64}))
(def m (lm/lm engine tokenizer {:max-cells 8192}))
(def prompt [0 1])
(def max-tokens 4)
(def args [prompt max-tokens])

;; a constraint: every token is 2 (an end token ends it), so the LM satisfies it rarely
(def only2? (fn [ctx] (every? #{2} ctx)))
(def no3 (potential/potential n-vocab {:prefix #(if (only2? %) 0.0 ##-Inf)
                                       :complete #(if (only2? %) 0.0 ##-Inf)}))

(def plain (token/completion m))
(def constrained-c (token/completion m {:potential no3}))

(defn row ^floats [ctx] (toy/toy-logprobs n-vocab ctx))

;; every completion of the LM, with its log p
(defn completions []
  (letfn [(walk [tokens ^double lp]
            (if (or (= max-tokens (count tokens)) (and (seq tokens) (= end (peek tokens))))
              [[tokens lp]]
              (let [^floats r (row (into prompt tokens))]
                (mapcat (fn [t] (walk (conj tokens t) (+ lp (aget r (int t))))) (range n-vocab)))))]
    (walk [] 0.0)))

(defn body [tokens] (if (and (seq tokens) (= end (peek tokens))) (pop tokens) tokens))

(println "--- 1. the constrained completion's own draws all satisfy it (tokens 2, then perhaps the end):")
(let [trs (mapv #(gfi/simulate constrained-c args (rng/key %)) (range 2000))]
  (println "simulate: fraction violating it =" (/ (count (remove #(only2? (body (:retval %))) trs)) 2000.0)))

(println "--- 2. SMC over the constrained completion, no SMC potential (target per smc's docstring: the completion's distribution):")
(let [s (smc/run constrained-c args 2000 {:key (rng/key 7)})
      ps (smc/particles s)
      with3 (count (remove #(only2? (body %)) ps))]
  (println "particles violating the completion's constraint:" with3 "of" (count ps) " log-ml-estimate:" (smc/log-ml-estimate s))
  (let [trs (smc/traces s)
        mismatched (count (remove true? (map #(= (:log-p %1) (:score %2)) (:particles s) trs)))
        neg-inf (count (filter #(= ##-Inf (:score %)) trs))]
    (println "traces whose score differs from the particle's :log-p:" mismatched
             "; traces of score -Inf (impossible under the completion):" neg-inf)))

(println "--- 3. SMC over the constrained completion twisted by the same constraint, :masked:")
(let [z-lm (reduce + (for [[x lp] (completions)] (Math/exp (+ lp (potential/complete no3 (body x))))))
      ;; the documented target's Z: sum over x of q(x) exp(complete(x)), q the
      ;; constrained completion's density, from assess
      z-doc (reduce + (for [[x _] (completions)
                            :let [w (:weight (gfi/assess constrained-c args (token/forced x)))]
                            :when (Double/isFinite w)]
                        (Math/exp (+ w (potential/complete no3 (body x))))))
      zs (mapv #(Math/exp (smc/log-ml-estimate (smc/run constrained-c args 50 {:potential no3 :proposal :masked
                                                                               :key (rng/key %)})))
               (range 40))
      mean (/ (reduce + zs) (count zs))]
  (println "Z under the LM, sum p(x) exp(complete(x))      =" z-lm)
  (println "Z under the completion, sum q(x) exp(complete(x)) =" z-doc)
  (println "mean of 40 runs' Z-hat (50 particles)            =" mean))

(println "--- 4. the plain completion, for comparison (the documented use):")
(let [zs (mapv #(Math/exp (smc/log-ml-estimate (smc/run plain args 50 {:potential no3 :proposal :masked
                                                                       :key (rng/key %)})))
               (range 40))]
  (println "mean Z-hat =" (/ (reduce + zs) (count zs))))
(shutdown-agents)
(System/exit 0)
