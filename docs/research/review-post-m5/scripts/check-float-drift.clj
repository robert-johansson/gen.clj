;; A check (not a finding): a token SMC particle's final log weight is complete(x) only up to the
;; float32 rounding of every step's twist (logw-next is a float row), so the
;; estimate is unbiased for a perturbed Z; the bootstrap on the toy engine
;; (log p - log q = 0 exactly) shows the drift directly.
;;   nice -n 10 clojure -J-Xmx1g -M:test -i OUT/scripts/check-float-drift.clj
(require '[gen.llm.lm :as lm]
         '[gen.llm.potential :as potential]
         '[gen.llm.smc :as smc]
         '[gen.llm.token :as token]
         '[gen.llm.toy-engine :as toy]
         '[gen.random :as rng])

(def n-vocab 5)
(def tokenizer {:n-vocab n-vocab :bos nil :eos 4 :eog #{4} :add-bos? false
                :token-bytes (fn [t] (byte-array [(byte t)])) :encode (fn [_ _] []) :decode (fn [ts] (apply str ts))})
(def m (lm/lm (toy/engine {:n-vocab n-vocab :max-branches 128 :n-ctx 256}) tokenizer {:max-cells 1000000}))
(def c (token/completion m {:end-tokens #{}}))

;; a soft potential with large weights: each token t costs 1000.1 * (t + 1) / 3
(defn w ^double [ctx] (reduce + 0.0 (map #(/ (* 1000.1 (inc (long %))) 3.0) ctx)))
(def heavy (potential/potential n-vocab {:prefix #(- (w %)) :complete #(- (w %))}))

(doseq [steps [10 100]]
  (let [s (smc/run c [[0 1] steps] 20 {:potential heavy :key (rng/key 1) :ess-threshold 0.0})
        drift (map (fn [x lw] (Math/abs (- (double lw) (potential/complete heavy x))))
                   (smc/particles s) (:log-weights s))]
    (println steps "tokens: |log weight - complete(x)|, max over 20 particles =" (apply max drift)
             "(complete(x) about" (potential/complete heavy (first (smc/particles s))) ")")))
(shutdown-agents)
(System/exit 0)
