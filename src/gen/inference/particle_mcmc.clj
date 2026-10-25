(ns gen.inference.particle-mcmc
  "Particle MCMC (Andrieu, Doucet and Holenstein 2010, \"Particle Markov
   chain Monte Carlo methods\", JRSS B): conditional SMC and particle
   Gibbs, over the particle-filter API (gen.inference.particle-filter).

     (conditional-smc trace selection steps n)   → a filter's final state
     (particle-gibbs selection steps n)          → a kernel (gen.inference.kernel)

   Conditional SMC is a particle filter conditioned on a trace. The trace's
   selected choices (the latent states) are the trajectory of one particle,
   the reference, particle 0; its other choices (the observations, and any
   parameters) are constrained in every particle. The other n - 1
   particles are drawn as a filter draws them. When the particles resample
   (their effective sample size below :ess-threshold, n/2 by default; ##Inf
   resamples at every step, as Andrieu et al.), those n - 1 are independent
   multinomial draws from all n, and the reference keeps its place and its
   lineage. A particle drawn from the final weights is then a Gibbs move on
   the selected choices, for any n ≥ 2: it leaves their distribution given
   the rest invariant. It is also reversible, since under Andrieu et al.'s
   extended target the reference and the drawn particle are independent
   draws from the final weights given the particles. particle-gibbs is that
   move as a kernel, which composes with the others. For a model
   (gen [T] (let [a (trace :a …)] (splice :xs (comb/unfold step) T 0.0 a)))
   whose steps trace :x and :y:

     (kernel/cycle [(particle-gibbs (sel/from-paths (for [t (range T)] [:xs t :x]))
                                    (mapv (fn [t] {:args [(inc t)]}) (range T))
                                    10)
                    (kernel/mh (sel/select :a))])

   steps are the filter's steps, maps as the particle filter takes them:
   the first, {:args :proposal :proposal-args}, initialize's; each later
   one, {:args :argdiffs :proposal :proposal-args}, a step's. A :proposal is
   optional, and :argdiffs default to diff/compute's from the step before's
   :args. The last step's :args are the trace's own. A step's observations
   are the trace's unselected choices that the step adds, and the
   reference's latents at the step its selected ones: the trace is taken
   back through the steps by updates to the step before's arguments, whose
   discards are what each step added. So a step may only add choices, as
   in the particle filter.

   The reference's weight at a step is the weight a particle drawn at its
   values would get: the update weight with its latents constrained, less
   their density under the step's proposal. That is :proposal's (assess;
   it must propose every latent the step adds), or else the internal
   proposal's, which draws a new choice from the model, so that its density
   is the model's (project): true of gen bodies and the combinators.

   Encapsulated randomness (thesis §4.5) is a loud error
   (gfi/with-exact-density): the reference's would be drawn afresh, which
   conditions on its states but not on its randomness. PMMH needs nothing
   of its own: it is MH on a model whose likelihood is a particle filter's
   estimate, gen.custom/estimated with an :estimate that runs one, as
   gen.inference.particle-mcmc-test shows.

   Not here yet: ancestor sampling (Lindsten, Jordan and Schön 2014), and
   conditional forms of the other resampling schemes."
  {:api :experimental}
  (:refer-clojure :exclude [+ -])
  (:require [gen.array :as array :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as kernel]
            [gen.inference.particle-filter :as pf]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defn- check-arguments! [trace selection steps n]
  (when-not (trace/trace? trace)
    (throw (ex-info (str "conditional-smc: the reference is a trace, not a " (pr-str (type trace))) {:reference trace})))
  (when-not (sel/selection? selection)
    (throw (ex-info (str "conditional-smc: the latents are a selection, not " (pr-str selection)) {:selection selection})))
  (when-not (and (vector? steps) (seq steps) (every? #(and (map? %) (sequential? (:args %))) steps))
    (throw (ex-info "conditional-smc: steps is a non-empty vector of maps, each with its :args" {:steps steps})))
  (when-not (and (integer? n) (>= n 2))
    (throw (ex-info (str "conditional-smc needs two particles at least, the reference and another, not " n) {:n n})))
  (when-not (= (:args trace) (vec (:args (peek steps))))
    (throw (ex-info "conditional-smc: the last step's :args must be the reference trace's own"
                    {:args (:args (peek steps)) :reference-args (:args trace)}))))

(defn- argdiffs-between
  "Argdiffs from the arguments from to the arguments to, diff/compute's:
   unknown-change throughout when there are not as many."
  [from to]
  (if (= (count from) (count to))
    (mapv diff/compute from to)
    (vec (repeat (count to) diff/unknown-change))))

(defn- additions
  "The choices of the trace that each step adds, [D_0 … D_T]: taken back
   to each step before's arguments, an update with no constraints, the
   trace discards what the step added; D_0 is what is left."
  [trace steps key]
  (loop [t (dec (count steps)) tr trace added ()]
    (if (zero? t)
      (vec (cons (:choices tr) added))
      (let [args (vec (:args (steps (dec t))))
            {tr' :trace discard :discard} (gfi/update tr args (argdiffs-between (:args tr) args) cm/EMPTY
                                                      (rng/fold-in key t))]
        (recur (dec t) (gfi/eval-trace! tr') (cons discard added))))))

(defn- reference-weight
  "The weight of the reference's move, w (its latents constrained), less
   their density under the step's proposal, on proposal-args (assess), or
   under the internal proposal, the model's in the new trace tr (project)."
  [w tr latents {:keys [proposal]} proposal-args key]
  (- w (if proposal
         (:weight (gfi/assess proposal proposal-args latents key))
         (gfi/project tr (sel/from-choicemap latents)))))

(defn- initialize
  "The filter's first step: n - 1 particles from pf/initialize, and the
   reference generated at its first choices, as particle 0."
  [model {:keys [args proposal-args] :as spec} observations added latents n key]
  (let [[k-free k-reference k-assess] (rng/split key 3)
        args (vec args)
        free (pf/initialize model args observations (dec n) (assoc (select-keys spec [:proposal :proposal-args]) :key k-free))
        {tr :trace w :weight} (gfi/generate model args added k-reference)
        w (reference-weight w tr latents spec (vec proposal-args) k-assess)]
    (assoc free
           :traces (into [(gfi/eval-trace! tr w)] (:traces free))
           :log-weights (into [w] (:log-weights free)))))

(defn- resample
  "The particles resampled when their effective sample size is below
   threshold: n independent multinomial draws, of which the reference
   keeps its own place and lineage, so that the others are n - 1
   independent draws from all n."
  [state threshold key]
  (if (< (pf/ess state) threshold)
    (-> (pf/resample state {:method :multinomial :key key})
        (assoc-in [:traces 0] (first (:traces state)))
        (assoc-in [:parents 0] 0))
    state))

(defn- step
  "A later step of the filter, the t-th: particles 1 … n - 1 stepped by
   pf/step, the reference by the update that adds its choices."
  [state {:keys [args proposal-args] :as spec} prev-args observations added latents t key]
  (let [[k-free k-reference k-assess] (rng/split key 3)
        args (vec args)
        argdiffs (or (:argdiffs spec) (argdiffs-between prev-args args))
        state (pf/update-view state (range 1 (count (:log-weights state))) pf/step args argdiffs observations
                              (assoc (select-keys spec [:proposal :proposal-args]) :key k-free))
        prev (first (:traces state))
        {tr :trace w :weight discard :discard} (gfi/update prev args argdiffs added k-reference)
        _ (when-not (cm/empty? discard)
            (throw (ex-info (str "conditional-smc: step " t " changed or removed choices of the reference;"
                                 " a step may only add choices")
                            {:step t :discard discard})))
        w (+ (first (:log-weights state)) (reference-weight w tr latents spec (into [prev] proposal-args) k-assess))]
    (assoc state
           :traces (assoc (:traces state) 0 (gfi/eval-trace! tr w))
           :log-weights (assoc (:log-weights state) 0 w))))

(defn conditional-smc
  "A particle filter of n particles over steps, conditioned on the trace:
   its selected choices are the trajectory of the reference, particle 0,
   which every resampling keeps, and its other choices are observed in
   every particle (see the namespace doc). Returns the filter's final
   state (gen.inference.particle-filter's), in which a particle drawn from
   the weights (pf/sample-unweighted-traces) is a Gibbs move on the
   selected choices. Its log-ml-estimate is conditioned on the reference,
   so not unbiased. Options: :ess-threshold (n/2), :key."
  ([trace selection steps n] (conditional-smc trace selection steps n {}))
  ([trace selection steps n {:keys [ess-threshold key]}]
   (check-arguments! trace selection steps n)
   (gfi/with-exact-density "conditional-smc"
     (let [[k-additions k-filter] (rng/split (or key (gfi/next-key)))
           added (additions trace steps k-additions)
           observed (sel/complement selection)
           latents (mapv #(sel/get-selected % selection) added)
           observations (mapv #(sel/get-selected % observed) added)
           threshold (or ess-threshold (/ n 2.0))
           ks (rng/split k-filter (count steps))]
       (reduce (fn [state t]
                 (let [[k-resample k-step] (rng/split (ks t))]
                   (-> state
                       (resample threshold k-resample)
                       (step (steps t) (:args (steps (dec t))) (observations t) (added t) (latents t) t k-step))))
               (initialize (:gen-fn trace) (first steps) (first observations) (first added) (first latents) n (first ks))
               (range 1 (count steps)))))))

(defn particle-gibbs
  "The particle Gibbs kernel of the selected choices (Andrieu et al.'s
   conditional SMC update): a conditional-smc of n particles conditioned
   on the trace, and a particle drawn from its final weights. steps are
   conditional-smc's, or a function of the trace giving them. The kernel
   is its own reversal, so it composes with the MH kernels in every
   combinator of gen.inference.kernel. Options as conditional-smc's."
  ([selection steps n] (particle-gibbs selection steps n {}))
  ([selection steps n opts]
   (kernel/symmetric
    (fn [trace key]
      (let [[k-filter k-draw] (rng/split key)
            state (conditional-smc trace selection (if (fn? steps) (steps trace) steps) n (assoc opts :key k-filter))]
        {:trace (first (pf/sample-unweighted-traces state 1 k-draw))})))))
