;; P4-04 (verified sound; the suite never runs it): a completion under a soft potential and under a
;; product, the paths the suite never runs (it tests grammars only).
;; Densities by enumeration, conditional-weight's mean exactly, and every law.
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/P4-04.clj
(require (quote gen.combinators))
(require '[clojure.test.check :as tc]
         '[gen.gfi :as gfi]
         '[gen.laws :as laws]
         '[gen.llm.constrained :as constrained]
         '[gen.llm.lm :as lm]
         '[gen.llm.potential :as potential]
         '[gen.llm.token :as token]
         '[gen.llm.toy-engine :as toy]
         '[gen.random :as rng])

(def n-vocab 5)
(def end 4)
(def tokenizer
  {:n-vocab n-vocab :bos nil :eos end :eog #{end} :add-bos? false
   :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))})
(def m (lm/lm (toy/engine {:n-vocab n-vocab :max-branches 128 :n-ctx 64}) tokenizer {:max-cells 8192}))
(def prompt [0 1])
(def max-tokens 4)
(def args [prompt max-tokens])

(def soft
  (let [ones #(+ 0.3 (* -0.7 (count (filter #{1} %))))]
    (potential/potential n-vocab {:prefix ones :complete #(- (ones %) (if (= 2 (peek %)) 0.0 0.5))})))
(def no3 (potential/potential n-vocab {:prefix #(if (some #{3} %) ##-Inf 0.0) :complete #(if (some #{3} %) ##-Inf 0.0)}))

(defn row ^floats [ctx] (toy/toy-logprobs n-vocab ctx))
(defn completions []
  (letfn [(walk [tokens ^double lp]
            (if (or (= max-tokens (count tokens)) (and (seq tokens) (= end (peek tokens))))
              [[tokens lp]]
              (let [^floats r (row (into prompt tokens))]
                (mapcat (fn [t] (walk (conj tokens t) (+ lp (aget r (int t))))) (range n-vocab)))))]
    (walk [] 0.0)))
(defn body [tokens] (if (and (seq tokens) (= end (peek tokens))) (pop tokens) tokens))

(doseq [[label p] [[:soft soft] [:product (potential/product soft no3)]]]
  (let [gf (token/completion m {:potential p})
        xs (completions)
        q (for [[x _] xs] (:weight (gfi/assess gf args (token/forced x))))
        z (reduce + (for [[x lp] xs] (Math/exp (+ lp (potential/complete p (body x))))))
        ;; E_q[exp(conditional-weight)], exactly: each completion's trace by generate
        e (reduce + (for [[x _] xs
                          :let [tr (:trace (gfi/generate gf args (token/forced x)))]
                          :when (Double/isFinite (:score tr))]
                      (Math/exp (+ (:score tr) (constrained/conditional-weight tr)))))]
    (println label "sum of q over every completion =" (reduce + (map #(Math/exp %) q))
             "; Z =" z "; E_q[exp(cw)] =" e)
    (let [models [[label gf args [[prompt 2] [prompt 6] [[3] 4]]]
                  [(keyword (str (name label) "-map")) (gen.combinators/map gf) [[prompt [3]] [3 2]] [[[prompt [3]] [1 4]]]]]]
      (doseq [[law f] (sort-by key laws/all)]
        (let [r (tc/quick-check 100 (f (laws/cases models)) :seed 11)]
          (when-not (:pass? r) (println "  FAILED" law (select-keys r [:fail :shrunk]))))))
    (println label "laws: done")))
(shutdown-agents)
(System/exit 0)
