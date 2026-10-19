(ns gen.inference.particle-filter
  "Particle filtering, as Gen.jl's initialize_particle_filter,
   particle_filter_step!, maybe_resample! and log_ml_estimate, with an
   immutable state: each step returns a new state.

     (initialize model args observations n)       → state
     (step state args argdiffs observations)       → state
     (maybe-resample state)                        → state
     (log-ml-estimate state)                       → number

   A state holds :traces, their unnormalized :log-weights, and :log-ml, the
   log marginal likelihood accumulated at earlier resamplings. A step extends
   every particle's trace with update (typically an Unfold growing by one
   step) and adds the update weight to its log weight. A step may only add
   choices: one that changes or removes a choice (a non-empty discard) is a
   loud error, as in Gen.jl.

   Options maps take :proposal and :proposal-args, a custom proposal
   generative function called on (trace, proposal-args...) for steps (and on
   proposal-args alone for initialize), and :key. maybe-resample takes
   :ess-threshold (n/2 by default), :method (:multinomial, as Gen.jl, or
   :systematic, :stratified or :residual) and :key.

   With {:batched? true} at initialize, the particles run as one batched
   execution (gen.batched): the state holds :particles, a particle
   collection, in place of :traces, and every later step stays batched
   (falling back to one particle at a time if the model branches on particle
   values). A custom proposal runs one particle at a time: at initialize it
   ignores :batched?, and a step with one turns a batched state into
   per-particle traces. traces returns the scalar traces either way."
  (:refer-clojure :exclude [+ -])
  (:require [gen.array :as a :refer [+ -]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.inference.weights :as w]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- particle-keys [opts n] (rng/split (or (:key opts) (gfi/next-key)) n))

(defn initialize
  "n particles of model on args, generated with the observations (from the
   internal proposal, or from :proposal)."
  ([model args observations n] (initialize model args observations n {}))
  ([model args observations n {:keys [proposal proposal-args batched?] :as opts}]
   (if (and batched? (not proposal))
     (let [{ps :particles ws :weights} (batched/generate model args observations n (or (:key opts) (gfi/next-key)))]
       {:particles ps :log-weights (vec (a/->clj ws)) :log-ml 0.0})
   (let [particle (fn [key]
                    (if proposal
                      (let [[k-propose k-generate] (rng/split key)
                            {:keys [choices weight]} (gfi/propose proposal (vec proposal-args) k-propose)
                            r (gfi/generate model args (cm/merge observations choices) k-generate)]
                        (update r :weight - weight))
                      (gfi/generate model args observations key)))
         results (mapv particle (particle-keys opts n))]
     {:traces (mapv :trace results)
      :log-weights (mapv :weight results)
      :log-ml 0.0}))))

(defn- check-discard [discard i]
  (when-not (cm/empty? discard)
    (throw (ex-info (str "A particle filter step changed or removed choices of particle " i
                         "; a step may only add choices")
                    {:particle i :discard discard}))))

(defn step
  "Every particle's trace updated to args with the observations (and choices
   from :proposal), each log weight increased by its incremental weight."
  ([state args argdiffs observations] (step state args argdiffs observations {}))
  ([state args argdiffs observations {:keys [proposal proposal-args] :as opts}]
   (if-let [ps (and (not proposal) (:particles state))]
     (let [{ps' :particles ws :weights ds :discards}
           (batched/update ps args argdiffs observations (or (:key opts) (gfi/next-key)))]
       (doseq [[i d] (map-indexed vector ds)] (check-discard d i))
       (assoc state :particles ps' :log-weights (mapv + (:log-weights state) (a/->clj ws))))
   (let [state (if-let [ps (:particles state)]
                 (-> state (dissoc :particles) (assoc :traces (batched/traces ps)))
                 state)
         n (count (:traces state))
         move (fn [i tr key]
                (if proposal
                  (let [[k-propose k-update] (rng/split key)
                        {:keys [choices weight]} (gfi/propose proposal (into [tr] proposal-args) k-propose)
                        r (gfi/update tr args argdiffs (cm/merge observations choices) k-update)]
                    (check-discard (:discard r) i)
                    (update r :weight - weight))
                  (let [r (gfi/update tr args argdiffs observations key)]
                    (check-discard (:discard r) i)
                    r)))
         results (mapv move (range n) (:traces state) (particle-keys opts n))]
     (assoc state
            :traces (mapv :trace results)
            :log-weights (mapv + (:log-weights state) (mapv :weight results)))))))

(defn ess
  "The effective sample size of the particles' weights."
  [state]
  (w/ess (:log-weights state)))

(defn resample
  "n particles drawn in proportion to their weights; the weights become
   equal, and the log marginal likelihood so far moves into :log-ml."
  ([state] (resample state {}))
  ([state {:keys [method key] :or {method :multinomial}}]
   (let [n (count (:log-weights state))
         {:keys [log-total]} (w/normalize (:log-weights state))
         parents (w/resample method (:log-weights state) n (or key (gfi/next-key)))]
     (assoc (if-let [ps (:particles state)]
              (assoc state :particles (batched/take ps parents))
              (assoc state :traces (mapv (:traces state) parents)))
            :log-weights (vec (repeat n 0.0))
            :log-ml (+ (:log-ml state) (- log-total (Math/log n)))
            :parents parents))))

(defn maybe-resample
  "Resample when the effective sample size is below :ess-threshold (n/2 by
   default). The returned state's :resampled? says whether it did."
  ([state] (maybe-resample state {}))
  ([state {:keys [ess-threshold] :as opts}]
   (let [n (count (:log-weights state))
         threshold (or ess-threshold (/ n 2.0))]
     (if (< (ess state) threshold)
       (assoc (resample state opts) :resampled? true)
       (assoc state :resampled? false)))))

(defn log-ml-estimate
  "The estimate of log p(observations so far): unbiased in exp space. (A
   batched state is spot-checked first: gen.batched/check.)"
  [state]
  (some-> (:particles state) batched/check)
  (+ (:log-ml state) (w/log-mean-exp (:log-weights state))))

(defn traces
  "The particles' traces (scalar traces, also for a batched filter)."
  [state]
  (if-let [ps (:particles state)] (batched/traces ps) (:traces state)))

(defn log-weights [state] (:log-weights state))

(defn sample-unweighted-traces
  "n traces drawn in proportion to the particles' weights."
  ([state n] (sample-unweighted-traces state n (gfi/next-key)))
  ([state n key]
   (let [idx (w/multinomial (:log-weights state) n key)]
     (if-let [ps (:particles state)]
       (batched/traces (batched/take ps idx))
       (mapv (:traces state) idx)))))
