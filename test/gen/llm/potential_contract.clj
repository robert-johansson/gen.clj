(ns gen.llm.potential-contract
  "The potential contract (gen.llm.potential) as a check any potential must
   pass: over contexts of finite prefix weight, logw-next holds the
   differences of prefix and complete, within tol (float rounding: the
   weights are stored as floats). Returns a vector of failures, empty when
   it passes."
  (:require [gen.llm.potential :as potential]))

(set! *warn-on-reflection* true)

(defn- off? [^double want ^double got ^double tol]
  (not (or (= want got) (<= (Math/abs (- want got)) tol))))

(defn check-consistent
  "The failures of p's consistency over contexts: logw-next's length,
   each token's entry against prefix(ctx ++ [t]) - prefix(ctx), and the
   end's against complete(ctx) - prefix(ctx)."
  [p n-vocab contexts tol]
  (vec (for [ctx contexts
             :let [base (potential/prefix p ctx)]
             :when (not (Double/isInfinite base))
             :let [^floats w (potential/logw-next p ctx)]
             failure (if (not= (alength w) (inc (long n-vocab)))
                       [(str ctx ": logw-next has " (alength w) " entries, not " (inc (long n-vocab)))]
                       (concat
                        (for [t (range n-vocab)
                              :let [want (- (potential/prefix p (conj (vec ctx) t)) base)
                                    got (double (aget w (int t)))]
                              :when (off? want got tol)]
                          (str ctx ", token " t ": logw-next " got ", the prefixes' difference " want))
                        (let [want (- (potential/complete p ctx) base)
                              got (double (aget w (int n-vocab)))]
                          (when (off? want got tol)
                            [(str ctx ", the end: logw-next " got ", complete less prefix " want)]))))]
         failure)))
