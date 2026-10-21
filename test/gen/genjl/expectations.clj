(ns gen.genjl.expectations
  "Gen.jl's assertions of a result written as data, for the ports whose
   testsets repeat the same kinds of assertion case after case: an
   expectation map, a key per kind, checked by a table from key to a
   function of the result and the expected value. An unknown key is an
   error, so a misspelt one cannot pass silently."
  (:require [gen.diff :as diff]))

(defn check
  "Assert each entry of expected with checks' function for its key, on
   what the result shows."
  [checks shown expected]
  (doseq [[k want] expected]
    (if-let [f (checks k)]
      (f shown want)
      (throw (ex-info (str "check: no assertion is named " k) {:key k :known (sort (keys checks))})))))

(defn diff-kind
  "A diff's kind, as the expectations name it: :no-change,
   :unknown-change or :vector-diff."
  [d]
  (cond (diff/no-change? d) :no-change
        (diff/unknown-change? d) :unknown-change
        (diff/vector-diff? d) :vector-diff
        :else d))
