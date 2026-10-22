(ns gen.inference.importance
  "Importance sampling and importance resampling, as Gen.jl's
   importance_sampling and importance_resampling.

     (importance-sampling model args observations n)
       → {:traces :log-normalized-weights :log-ml-estimate}
     (importance-resampling model args observations n)
       → {:trace :log-ml-estimate}

   The proposal is the model's internal proposal by default. With
   {:proposal q :proposal-args xs} each sample instead draws choices from q,
   generates the model constrained by them and the observations, and is
   weighted log p(t) - log q(choices) - log q_internal(the rest): choices q
   leaves out are sampled by the model's internal proposal, whose weight
   generate accounts for. q must not propose an observed address.

   With {:batched? true}, the n samples run as one batched execution
   (gen.batched), and the result holds :particles, a particle collection, in
   place of :traces; (traces result) gives the scalar traces either way. A
   custom proposal runs one sample at a time, ignoring :batched?.

   The options map also takes :key; omitting it draws one from the implicit
   key source."
  (:refer-clojure :exclude [-])
  (:require [gen.array :as array :refer [-]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.inference.weights :as w]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn weighted-sample
  "One trace of model on args with its log importance weight ({:trace
   :weight}): generated with the observations, and with the choices of
   :proposal on :proposal-args, less their log density, when given. A
   proposal's choice at an observed address is a loud error."
  [model args observations {:keys [proposal proposal-args]} key]
  (if proposal
    (let [[k-propose k-generate] (rng/split key)
          {:keys [choices weight]} (gfi/propose proposal (vec proposal-args) k-propose)
          overlap (filter (set (cm/addresses observations)) (cm/addresses choices))
          _ (when (seq overlap)
              (throw (ex-info (str "The proposal's choices overlap the observations at " (vec overlap)
                                   ": a proposal must leave the observations alone")
                              {:addresses (vec overlap)})))
          result (gfi/generate model args (cm/merge observations choices) k-generate)]
      (update result :weight - weight))
    (gfi/generate model args observations key)))

(defn importance-sampling
  "n traces of model on args, constrained by observations, with their
   normalized log weights and the log marginal likelihood estimate: the log
   of the mean importance weight, unbiased for p(observations) in exp space."
  ([model args observations n] (importance-sampling model args observations n {}))
  ([model args observations n opts]
   (let [key (or (:key opts) (gfi/next-key))]
     (if (and (:batched? opts) (not (:proposal opts)))
       (let [{ps :particles ws :weights} (batched/generate model args observations n key)
             log-weights (array/->clj ws)]
         {:particles ps
          :log-normalized-weights (:log-weights (w/normalize log-weights))
          :log-ml-estimate (w/log-mean-exp log-weights)})
       (let [results (mapv (fn [k]
                             ;; one particle's graph at a time on a lazy backend
                             (let [r (weighted-sample model args observations opts k)]
                               (gfi/eval-trace! (:trace r) (:weight r))
                               r))
                           (rng/split key n))
             log-weights (mapv :weight results)]
         {:traces (mapv :trace results)
          :log-normalized-weights (:log-weights (w/normalize log-weights))
          :log-ml-estimate (w/log-mean-exp log-weights)})))))

(defn traces
  "The scalar traces of an importance-sampling result, batched or not."
  [result]
  (if-let [ps (:particles result)] (batched/traces ps) (:traces result)))

(defn importance-resampling
  "One trace drawn in proportion to the weights of n importance samples
   (sampling importance resampling), with the log marginal likelihood
   estimate. As n grows, the trace's distribution approaches the posterior."
  ([model args observations n] (importance-resampling model args observations n {}))
  ([model args observations n opts]
   (let [[k-samples k-resample] (rng/split (or (:key opts) (gfi/next-key)))
         {scalar-traces :traces :keys [particles log-normalized-weights log-ml-estimate]}
         (importance-sampling model args observations n (assoc opts :key k-samples))
         i (first (w/multinomial log-normalized-weights 1 k-resample))]
     {:trace (if particles (first (batched/traces (batched/take particles [i]))) (scalar-traces i))
      :log-ml-estimate log-ml-estimate})))
