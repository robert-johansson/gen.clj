(ns hooks.gen.llm
  "gen.llm.constrained's fold-weights: a loop over the tokens t with the
   accumulator, w bound to each token's weight.")

(defmacro fold-weights [[t w ws ends] [acc init] & body]
  `(do ~ws ~ends
       (loop [~t 0 ~acc ~init]
         (if (< ~t 1)
           (let [~w (double ~t)] (recur (inc ~t) (do ~@body)))
           ~acc))))
