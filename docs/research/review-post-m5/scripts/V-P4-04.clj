;; V-P4-04 (verifier): would a regression on the soft-potential path pass the
;; suite? Plant, at run time only (alter-var-root; no file changes), a
;; conditional-weight that reads a finite whole-sequence weight as 0 (right
;; for every grammar, wrong for any soft potential), and run the only tests
;; that call conditional-weight (constrained_test's local-is-not-global and
;; an-unconstrained-completion-has-weight-0; grep finds no other caller). A
;; control mutant (no whole-sequence weight at all, the mutation the grammar
;; doc lists as caught) shows the method can fail.
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/V-P4-04.clj
(require '[clojure.test :as t]
         '[gen.gfi :as gfi]
         '[gen.llm.constrained :as constrained]
         '[gen.llm.lm :as lm]
         '[gen.llm.potential :as potential]
         '[gen.llm.token :as token]
         '[gen.llm.toy-engine :as toy]
         '[gen.llm.constrained-test])

(def original constrained/conditional-weight)

(defn mutant [clamp]
  (fn ^double [tr]
    (let [{:keys [lm potential] :as gf} (:gen-fn tr)
          [prompt] (:args tr)
          tokens (:retval tr)
          score (double (:score tr))]
      (cond (nil? potential) 0.0
            (== Double/NEGATIVE_INFINITY score) Double/NEGATIVE_INFINITY
            :else (let [text (if (and (seq tokens) (contains? (:end-tokens gf) (peek tokens))) (pop tokens) tokens)
                        log-p (double (if (seq tokens) (reduce + 0.0 (lm/score lm prompt tokens)) 0.0))]
                    (- (+ log-p (clamp (potential/complete potential text))) score))))))

(def clamped (mutant (fn [^double w] (if (Double/isInfinite w) w 0.0))))
(def dropped (mutant (fn [_] 0.0)))

(defn run-cw-tests []
  (let [counters (atom nil)]
    (binding [t/*report-counters* (ref t/*initial-report-counters*)
              t/*test-out* (java.io.StringWriter.)]
      (t/test-vars [#'gen.llm.constrained-test/local-is-not-global
                    #'gen.llm.constrained-test/an-unconstrained-completion-has-weight-0])
      (reset! counters @t/*report-counters*))
    (select-keys @counters [:pass :fail :error])))

(println "original:          " (run-cw-tests))
(alter-var-root #'constrained/conditional-weight (constantly clamped))
(println "clamped mutant:    " (run-cw-tests))
(alter-var-root #'constrained/conditional-weight (constantly dropped))
(println "dropped (control): " (run-cw-tests))
(alter-var-root #'constrained/conditional-weight (constantly original))

;; the clamped mutant is wrong under a soft potential: E_q[exp(cw)] should be Z
(def n-vocab 5)
(def tokenizer {:n-vocab n-vocab :bos nil :eos 4 :eog #{4} :add-bos? false
                :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))})
(def m (lm/lm (toy/engine {:n-vocab n-vocab :max-branches 128 :n-ctx 64}) tokenizer {:max-cells 8192}))
(def soft (potential/potential n-vocab {:prefix #(* -0.6 (count (filter #{0} %))) :complete #(* -0.6 (count (filter #{0} %)))}))
(def gf (token/completion m {:potential soft}))
(def args [[2 3] 3])
(defn completions []
  (letfn [(walk [tokens ^double lp]
            (if (or (= 3 (count tokens)) (and (seq tokens) (= 4 (peek tokens))))
              [[tokens lp]]
              (let [^floats r (toy/toy-logprobs n-vocab (into [2 3] tokens))]
                (mapcat (fn [t] (walk (conj tokens t) (+ lp (aget r (int t))))) (range n-vocab)))))]
    (walk [] 0.0)))
(defn body [x] (if (and (seq x) (= 4 (peek x))) (pop x) x))
(let [xs (completions)
      z (reduce + (for [[x lp] xs] (Math/exp (+ lp (potential/complete soft (body x))))))
      e (fn [cw] (reduce + (for [[x _] xs :let [tr (:trace (gfi/generate gf args (token/forced x)))]]
                             (Math/exp (+ (:score tr) (cw tr))))))]
  (println "soft potential: Z =" z "| E_q[exp cw], original =" (e original) "| clamped mutant =" (e clamped)))
(shutdown-agents)
(System/exit 0)
