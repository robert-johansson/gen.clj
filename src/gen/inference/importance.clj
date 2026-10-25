(ns gen.inference.importance
  "Importance sampling and importance resampling, as Gen.jl's
   importance_sampling and importance_resampling.

     (importance-sampling model args observations n)
       → {:traces :log-weights :log-normalized-weights :log-ml-estimate :ess}
     (importance-resampling model args observations n)
       → {:trace :log-ml-estimate}
     (estimate f model args observations n)
       → {:estimate :log-ml-estimate :ess}

   Weights are host numbers, vectors of doubles: they are read here, at
   the inference loop, one per sample ((array/array w) makes an array of
   them). importance-resampling and estimate keep no traces but the one
   they return, so n can be large on any backend: one sample at a time,
   resampling keeps a sample with probability its share of the weight so
   far (Gen.jl's), and estimate keeps (f trace) and the weight.

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

(defn- sample-at
  "weighted-sample, evaluated (one particle's graph at a time on a lazy
   backend), its weight a host double."
  [model args observations opts key]
  (let [{t :trace wt :weight} (weighted-sample model args observations opts key)]
    (gfi/eval-trace! t wt)
    {:trace t :weight (double (array/item wt))}))

(defn- weights-summary
  "The log weights' raw and normalized vectors, the log marginal likelihood
   estimate and the effective sample size."
  [log-weights]
  {:log-weights log-weights
   :log-normalized-weights (:log-weights (w/normalize log-weights))
   :log-ml-estimate (w/log-mean-exp log-weights)
   :ess (w/ess log-weights)})

(defn importance-sampling
  "n traces of model on args, constrained by observations, with their log
   weights (raw, and normalized), the log marginal likelihood estimate (the
   log of the mean importance weight, unbiased for p(observations) in exp
   space) and the effective sample size."
  ([model args observations n] (importance-sampling model args observations n {}))
  ([model args observations n opts]
   (let [key (or (:key opts) (gfi/next-key))]
     (if (and (:batched? opts) (not (:proposal opts)))
       (let [{ps :particles ws :weights} (batched/generate model args observations n key)]
         (assoc (weights-summary (mapv double (array/->clj ws))) :particles ps))
       (let [results (mapv #(sample-at model args observations opts %) (rng/split key n))]
         (assoc (weights-summary (mapv :weight results)) :traces (mapv :trace results)))))))

(defn traces
  "The scalar traces of an importance-sampling result, batched or not."
  [result]
  (if-let [ps (:particles result)] (batched/traces ps) (:traces result)))

(defn- impossible! [n]
  (throw (ex-info (str "Every one of the " n " importance weights is -Inf: no sample is consistent with the"
                       " observations")
                  {:reason :impossible :n n})))

(defn importance-resampling
  "One trace drawn in proportion to the weights of n importance samples
   (sampling importance resampling), with the log marginal likelihood
   estimate. As n grows, the trace's distribution approaches the posterior.
   One sample at a time, each replaces the one kept with probability its
   share of the weight so far, which draws the kept one in proportion to
   the weights with no more than one trace held (Gen.jl's); batched, the
   samples run as one batched execution and one is drawn from them."
  ([model args observations n] (importance-resampling model args observations n {}))
  ([model args observations n opts]
   (let [[k-samples k-resample] (rng/split (or (:key opts) (gfi/next-key)))]
     (if (and (:batched? opts) (not (:proposal opts)))
       (let [{:keys [particles log-normalized-weights log-ml-estimate]}
             (importance-sampling model args observations n (assoc opts :key k-samples))
             i (first (w/multinomial log-normalized-weights 1 k-resample))]
         {:trace (first (batched/traces (batched/take particles [i])))
          :log-ml-estimate log-ml-estimate})
       (let [{:keys [trace log-total]}
             (reduce (fn [{:keys [log-total] :as kept} [k k-u]]
                       (let [{t :trace lw :weight} (sample-at model args observations opts k)
                             total (w/log-sum-exp [log-total lw])]
                         (if (and (> lw ##-Inf)
                                  (< (Math/log (double (array/item (rng/uniform k-u)))) (- lw total)))
                           {:trace t :log-total total}
                           (assoc kept :log-total total))))
                     {:trace nil :log-total ##-Inf}
                     (map vector (rng/split k-samples n) (rng/split k-resample n)))]
         (when (== ##-Inf (double log-total)) (impossible! n))
         {:trace trace :log-ml-estimate (- (double log-total) (Math/log n))})))))

(defn estimate
  "The self-normalized importance sampling estimate of the expected value of
   (f trace) given the observations, Σ p_i f(t_i) over n samples with
   normalized weights p_i, with the log marginal likelihood estimate and the
   effective sample size. f returns a host number. No trace is kept: f is
   applied to each sample as it is drawn (batched, to each particle's
   rebuilt trace in turn), but not to a sample of weight -Inf, outside the
   support, whose trace holds its choices only up to the one outside it
   (DESIGN.md §9): it counts nothing."
  ([f model args observations n] (estimate f model args observations n {}))
  ([f model args observations n opts]
   (let [key (or (:key opts) (gfi/next-key))
         ;; (f (trace)) of a sample of log weight lw
         value (fn [trace lw] (if (gfi/impossible? lw) 0.0 (double (f (trace)))))
         [values log-weights]
         (if (and (:batched? opts) (not (:proposal opts)))
           (let [{ps :particles ws :weights} (batched/generate model args observations n key)
                 lws (mapv double (array/->clj ws))]
             [(mapv (fn [i] (value #(first (batched/traces (batched/take ps [i]))) (lws i))) (range n))
              lws])
           (let [samples (map #(let [{t :trace lw :weight} (sample-at model args observations opts %)]
                                 [(value (constantly t) lw) lw])
                              (rng/split key n))]
             [(mapv first samples) (mapv second samples)]))
         {:keys [log-normalized-weights log-ml-estimate ess]} (weights-summary log-weights)]
     {:estimate (reduce + 0.0 (map #(* %1 (Math/exp %2)) values log-normalized-weights))
      :log-ml-estimate log-ml-estimate
      :ess ess})))
