;; V-P4-01 (verifier): which target does token SMC have over a completion
;; with its own :potential? Three readings, each Z by enumeration:
;;   A  the completion's density q_g (smc's docstring: "p the completion's density")
;;   B  the LM's rows, the completion's potential ignored
;;   C  genlm-control's: the LM times every potential, p·exp(g + h)
;; with g a constraint on the completion and h a soft SMC twist, so all three differ.
;; Run from the snapshot worktree:
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/V-P4-01.clj
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
(def engine (toy/engine {:n-vocab n-vocab :max-branches 256 :n-ctx 64}))
(def m (lm/lm engine tokenizer {:max-cells 8192}))
(def prompt [3 0])
(def max-tokens 3)
(def args [prompt max-tokens])

;; g: no token 1 anywhere (a constraint on the completion)
(def ok? (fn [ctx] (not-any? #{1} ctx)))
(def g (potential/potential n-vocab {:prefix #(if (ok? %) 0.0 ##-Inf) :complete #(if (ok? %) 0.0 ##-Inf)}))
;; h: soft, each 2 costs 0.9, a whole sequence gains 0.4 if it ends in 0
(def h (let [twos #(* -0.9 (count (filter #{2} %)))]
         (potential/potential n-vocab {:prefix twos :complete #(+ (twos %) (if (= 0 (peek %)) 0.4 0.0))})))

(def plain (token/completion m))
(def cg (token/completion m {:potential g}))

(defn row ^floats [ctx] (toy/toy-logprobs n-vocab ctx))
(defn body [tokens] (if (and (seq tokens) (= end (peek tokens))) (pop tokens) tokens))
(defn completions []
  (letfn [(walk [tokens ^double lp]
            (if (or (= max-tokens (count tokens)) (and (seq tokens) (= end (peek tokens))))
              [[tokens lp]]
              (let [^floats r (row (into prompt tokens))]
                (mapcat (fn [t] (walk (conj tokens t) (+ lp (aget r (int t))))) (range n-vocab)))))]
    (walk [] 0.0)))

(def xs (completions))
(def z-a (reduce + (for [[x _] xs :let [w (:weight (gfi/assess cg args (token/forced x)))] :when (Double/isFinite w)]
                     (Math/exp (+ w (potential/complete h (body x)))))))
(def z-b (reduce + (for [[x lp] xs] (Math/exp (+ lp (potential/complete h (body x)))))))
(def z-c (reduce + (for [[x lp] xs] (Math/exp (+ lp (potential/complete g (body x)) (potential/complete h (body x)))))))
(println "completions enumerated:" (count xs) "; sum of q_g over them:"
         (reduce + (for [[x _] xs :let [w (:weight (gfi/assess cg args (token/forced x)))]] (Math/exp w))))
(println "Z_A (q_g·exp h) =" z-a)
(println "Z_B (p·exp h)   =" z-b)
(println "Z_C (p·exp(g+h))=" z-c)

(defn mean ^double [v] (/ (reduce + 0.0 v) (count v)))
(defn se ^double [v] (let [mu (mean v)] (Math/sqrt (/ (/ (reduce + (map #(Math/pow (- % mu) 2) v)) (dec (count v))) (count v)))))

(doseq [proposal [:lm :masked]]
  (let [zs (mapv #(Math/exp (smc/log-ml-estimate (smc/run cg args 50 {:potential h :proposal proposal :key (rng/key (+ 100 %))})))
                 (range 60))
        zp (mapv #(Math/exp (smc/log-ml-estimate (smc/run plain args 50 {:potential h :proposal proposal :key (rng/key (+ 100 %))})))
                 (range 60))]
    (println proposal "over the constrained completion: mean Z-hat" (mean zs) "± 5 SE" (* 5 (se zs))
             "| z-scores A/B/C:" (mapv #(/ (- (mean zs) %) (se zs)) [z-a z-b z-c]))
    (println proposal "same keys over the plain completion: bit-identical estimates?" (= zs zp))))

(println "--- first token of 4000 particles over the constrained completion, no SMC potential:")
(let [k 4000
      firsts (frequencies (map first (smc/particles (smc/step (smc/initialize cg args k {:key (rng/key 5)})))))
      ^floats r (row prompt)
      [^floats q] (constrained/weighted #{end} r (potential/logw-next g []))]
  (doseq [t (range n-vocab)]
    (println "token" t ": observed" (/ (get firsts t 0) (double k)) " p_LM" (Math/exp (aget r t)) " q_g" (Math/exp (aget q t)))))

(println "--- traces of a finished run over the constrained completion, no SMC potential:")
(let [s (smc/run cg args 500 {:key (rng/key 6)})
      trs (smc/traces s)
      ps (:particles s)]
  (println "particles with a 1 (outside g):" (count (remove #(ok? (body (:tokens %))) ps))
           "; traces of score -Inf:" (count (filter #(= ##-Inf (:score %)) trs))
           "; scores = :log-p:" (count (filter true? (map #(== (:log-p %1) (:score %2)) ps trs))) "of" (count ps)))
(let [s (smc/run plain args 500 {:key (rng/key 6)})]
  (println "plain completion, scores = :log-p:" (count (filter true? (map #(== (:log-p %1) (:score %2)) (:particles s) (smc/traces s))))
           "of" (count (:particles s))))
(shutdown-agents)
(System/exit 0)
