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
   per-particle traces. traces returns the scalar traces either way.

   From GenParticleFilters: stratified initialization (:strata, with
   choice-product), priority resampling and resize, rejuvenation by
   move-accept (rejuvenate) and move-reweight (move-reweight, with the
   kernels of reweighting), replicate and dereplicate, coalesce, introduce,
   and the weighted statistics mean, variance and proportions."
  (:refer-clojure :exclude [+ - * replicate])
  (:require [gen.array :as array :refer [+ - *]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.gfi :as gfi]
            [gen.inference.mh :as mh]
            [gen.inference.weights :as w]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- particle-keys [opts n] (rng/split (or (:key opts) (gfi/next-key)) n))

(defn- slots
  "The group of each of groups × size slots: in contiguous blocks (group g
   at g·size … g·size + size - 1) or interleaved (group g at g, g + groups,
   …)."
  [layout groups size]
  (let [group-of (case layout
                   :contiguous #(quot % size)
                   :interleaved #(mod % groups)
                   (throw (ex-info (str "The layout is :contiguous or :interleaved, not " (pr-str layout))
                                   {:layout layout})))]
    (mapv group-of (range (* groups size)))))

(defn- strata-of
  "The stratum of each of n particles over k strata (GenParticleFilters'
   stratified_map!): ⌊n/k⌋ particles each, in contiguous blocks or
   interleaved, then the remaining particles' strata drawn uniformly."
  [n k layout key]
  (when-not (pos? k)
    (throw (ex-info "Stratified initialization needs at least one stratum" {:strata k})))
  (let [b (quot n k)]
    (into (slots layout k b) (w/multinomial (repeat k 0.0) (- n (* b k)) key))))

(defn initialize
  "n particles of model on args, generated with the observations (from the
   internal proposal, or from :proposal). With :strata, a seq of choice maps
   (e.g. from choice-product), the initialization is stratified: each
   particle is constrained to its stratum as well, ⌊n/k⌋ particles to each
   of the k strata (:layout :contiguous, the default, or :interleaved), the
   rest to strata drawn uniformly, and every weight gains log k. As with a
   proposal, a stratified initialization runs one particle at a time and
   ignores :batched?."
  ([model args observations n] (initialize model args observations n {}))
  ([model args observations n {:keys [proposal proposal-args batched? strata layout] :or {layout :contiguous} :as opts}]
   (if (and batched? (not proposal) (not strata))
     (let [{ps :particles ws :weights} (batched/generate model args observations n (or (:key opts) (gfi/next-key)))]
       {:particles ps :log-weights (vec (array/->clj ws)) :log-ml 0.0})
   (let [particle (fn [constraints key]
                    (if proposal
                      (let [[k-propose k-generate] (rng/split key)
                            {:keys [choices weight]} (gfi/propose proposal (vec proposal-args) k-propose)
                            r (gfi/generate model args (cm/merge constraints choices) k-generate)]
                        (update r :weight - weight))
                      (gfi/generate model args constraints key)))
         results (if strata
                   (let [strata (vec strata)
                         k (count strata)
                         [k-particles k-strata] (rng/split (or (:key opts) (gfi/next-key)))]
                     (mapv (fn [s key] (update (particle (cm/merge (strata s) observations) key) :weight + (Math/log k)))
                           (strata-of n k layout k-strata) (rng/split k-particles n)))
                   (mapv #(particle observations %) (particle-keys opts n)))]
     {:traces (mapv :trace results)
      :log-weights (mapv :weight results)
      :log-ml 0.0}))))

(defn choice-product
  "A choice map for every combination of values (GenParticleFilters'
   choiceproduct): (choice-product [[:a [1 2]] [:b [true false]]]) gives
   four choice maps, the first address varying slowest. A map of addresses
   to values works too, in the map's own order."
  [address-values]
  (reduce (fn [cms [addr vs]] (for [c cms v vs] (cm/set-choice c addr v)))
          [(cm/choicemap)]
          address-values))

(defn- check-discard [discard i]
  (when-not (cm/empty? discard)
    (throw (ex-info (str "A particle filter step changed or removed choices of particle " i
                         "; a step may only add choices")
                    {:particle i :discard discard}))))

(defn- unbatched
  "The state with per-particle traces in place of a batched collection."
  [state]
  (if-let [ps (:particles state)]
    (-> state (dissoc :particles) (assoc :traces (batched/traces ps)))
    state))

(defn traces
  "The particles' traces (scalar traces, also for a batched filter)."
  [state]
  (if-let [ps (:particles state)] (batched/traces ps) (:traces state)))

(defn- select-particles
  "The state's particles at the indices idx (batched or not), with the log
   weights lw."
  [state idx lw]
  (assoc (if-let [ps (:particles state)]
           (assoc state :particles (batched/take ps idx))
           (assoc state :traces (mapv (:traces state) idx)))
         :log-weights lw
         :parents idx))

(defn step
  "Every particle's trace updated to args with the observations (and choices
   from :proposal), each log weight increased by its incremental weight.
   With :backward and :backward-args, a backward proposal (GenParticleFilters'
   pf_update! with one), the proposal may change existing choices too: the
   discard is assessed under (backward trace' backward-args…) and its log
   density added to the weight."
  ([state args argdiffs observations] (step state args argdiffs observations {}))
  ([state args argdiffs observations {:keys [proposal proposal-args backward backward-args] :as opts}]
   (when (and backward (not proposal))
     (throw (ex-info "A step's :backward proposal weighs a forward :proposal's changes; there is no :proposal" {})))
   (if-let [ps (and (not proposal) (:particles state))]
     (let [{ps' :particles ws :weights ds :discards}
           (batched/update ps args argdiffs observations (or (:key opts) (gfi/next-key)))]
       (doseq [[i d] (map-indexed vector ds)] (check-discard d i))
       (array/eval! (:trace ps'))
       (assoc state :particles ps' :log-weights (mapv + (:log-weights state) (array/->clj ws))))
   (let [state (unbatched state)
         n (count (:traces state))
         move (fn [i tr key]
                (if proposal
                  (let [[k-propose k-update] (rng/split key)
                        {:keys [choices weight]} (gfi/propose proposal (into [tr] proposal-args) k-propose)
                        r (gfi/update tr args argdiffs (cm/merge observations choices) k-update)]
                    (if backward
                      (update r :weight + (- (:weight (gfi/assess backward (into [(:trace r)] backward-args) (:discard r) (rng/fold-in k-update 1)))
                                             weight))
                      (do (check-discard (:discard r) i)
                          (update r :weight - weight))))
                  (let [r (gfi/update tr args argdiffs observations key)]
                    (check-discard (:discard r) i)
                    r)))
         results (mapv move (range n) (:traces state) (particle-keys opts n))
         log-weights (mapv + (:log-weights state) (mapv :weight results))]
     ;; one step's graph at a time on a lazy backend
     (array/eval! [(mapv (comp gfi/eval-trace! :trace) results) log-weights])
     (assoc state
            :traces (mapv :trace results)
            :log-weights log-weights)))))

(defn ess
  "The effective sample size of the particles' weights."
  [state]
  (w/ess (:log-weights state)))

(defn- host-log-weights [state]
  (let [lw (vec (:log-weights state))]
    (mapv double (if (every? array/host? lw) lw (array/->clj (array/stack lw))))))

(defn- optimal-parents
  "Fearnhead and Clifford's (2003) resampling to n particles, at most the
   current number, each kept at most once: with c solving
   Σ min(c·W_i, 1) = n (W the normalized weights), the particles with
   c·W_i ≥ 1 are kept, and the other n - kept drawn systematically in
   proportion to W, at most once each since c·W_i < 1 for them.
   [parents log-weights], the weights relative to the total: W_i·n for a
   kept particle, n/c for a drawn one, so they add up to n exactly.

   In weight order, the kept particles are those from the first κ = W_(i)
   with Σ_{j≤i} W_(j)/κ + (number after i) ≤ n on: below κ, c·W < 1, so
   c = (n - kept)/(their total). (GenParticleFilters' find_inv_w_threshold
   returns (n - after)/(Σ_{j≤i} W_(j)), counting κ among the drawn although
   c·κ ≥ 1 then; for W = [0.1 0.2 0.7] and n = 2 it gives c = 2, where
   Σ min(c·W, 1) = 2 needs 10/3, and the weights add up to 1.2·n.)"
  [lw n key]
  (let [n-old (count lw)
        _ (when (> n n-old)
            (throw (ex-info (str "Optimal resampling cannot add particles: " n " from " n-old)
                            {:n n :particles n-old})))
        lw-normal (:log-weights (w/normalize lw))
        p (mapv #(Math/exp %) lw-normal)
        ;; a particle of weight 0 is never kept nor drawn
        order (vec (sort-by p (filter #(pos? (p %)) (range n-old))))
        n-old (count order)
        _ (when (> n n-old)
            (throw (ex-info (str "Optimal resampling to " n " particles needs that many of positive weight; there are "
                                 n-old)
                            {:n n :positive n-old})))
        below (vec (reductions clojure.core/+ (map p order)))
        i (or (first (filter (fn [i] (let [reach (+ (/ (below i) (p (order i))) (- n-old i 1))]
                                       (<= reach (+ n (Math/ulp (double reach))))))
                             (range n-old)))
              n-old)
        kept (subvec order i)
        others (subvec order 0 i)
        m (- n (count kept))
        c (when (pos? m) (/ m (reduce clojure.core/+ (map p others))))
        drawn (if (pos? m) (mapv others (w/systematic (mapv lw-normal others) m key)) [])]
    [(into kept drawn)
     (into (mapv #(+ (lw-normal %) (Math/log n)) kept) (when c (repeat m (Math/log (/ n c)))))]))

(defn resample
  "Particles drawn in proportion to their weights; the weights become
   equal, and the log marginal likelihood so far moves into :log-ml.
   Options: :method (:multinomial, :systematic, :stratified, :residual, or
   :optimal, below), :key, and from GenParticleFilters:
   - :priority, (fn [log-weights] → log priorities): draw in proportion to
     these instead (e.g. tempered weights, #(map (partial * 0.5) %)); each
     copy of particle a then keeps the weight w_a/π_a (π the normalized
     priorities), so the log marginal likelihood estimate stays unbiased;
   - :n, a new number of particles (as resize);
   - :method :optimal, Fearnhead and Clifford's resampling to :n at most
     the current number (pf_optimal_resize!): every particle survives at
     most once, and the weights stay unequal."
  ([state] (resample state {}))
  ([state {:keys [method key priority n] :or {method :multinomial}}]
   (let [lw (host-log-weights state)
         n-old (count lw)
         n (or n n-old)
         key (or key (gfi/next-key))
         {lw-n :log-weights log-total :log-total} (w/normalize lw)
         [parents log-weights]
         (if (= method :optimal)
           (if priority
             (throw (ex-info "Optimal resampling takes no :priority" {:method method}))
             (optimal-parents lw n key))
           (let [lp-n (if priority (:log-weights (w/normalize (mapv double (priority lw)))) lw-n)
                 _ (when (some (fn [i] (and (== ##-Inf (lp-n i)) (> (lw i) ##-Inf))) (range n-old))
                     (throw (ex-info "A priority of 0 for a particle of positive weight would bias the estimate"
                                     {:log-weights lw})))
                 parents (w/resample method lp-n n key)]
             ;; log W_a - log π_a, normalized: 0 when π are the weights themselves
             [parents (mapv #(- (lw-n %) (lp-n %)) parents)]))]
     (assoc (select-particles state parents log-weights)
            :log-ml (+ (:log-ml state) (- log-total (Math/log n-old)))))))

(defn resize
  "The state with n particles, resampled (GenParticleFilters' pf_resize!);
   options as for resample."
  ([state n] (resize state n {}))
  ([state n opts] (resample state (assoc opts :n n))))

(defn rejuvenate
  "Each particle's trace moved by kernel, an MCMC kernel ((fn [trace key] →
   {:trace …}), gen.inference.kernel) that leaves the posterior invariant,
   so the weights stay as they are (GenParticleFilters' pf_move_accept!).
   A batched state becomes one of scalar traces."
  ([state kernel] (rejuvenate state kernel (gfi/next-key)))
  ([state kernel key]
   (let [state (unbatched state)
         moved (mapv (fn [tr k] (:trace (kernel tr k))) (:traces state) (rng/split key (count (:traces state))))]
     (array/eval! (mapv gfi/eval-trace! moved))
     (assoc state :traces moved))))

(defn reweighting
  "A move-reweight kernel (GenParticleFilters' move_reweight): (fn [trace
   key] → {:trace trace' :weight w}), w the move's log importance weight,
   log p(t')/p(t) - log q_fwd(u; t) + log q_bwd(u'; t').
   - (reweighting selection): regenerate the selection from the internal
     proposal; w is regenerate's weight.
   - (reweighting proposal proposal-args): propose choices from
     (proposal trace args...), update the trace with them, and assess the
     discard under (proposal trace' args...): w = the update weight - the
     forward score + the backward one.
   - (reweighting proposal proposal-args opts), with :backward and
     :backward-args, a separate backward proposal with the same support;
     with :involution, (fn [trace fwd-choices fwd-retval proposal-args]) →
     {:trace :bwd-choices :weight} as for involutive MH, in place of the
     update (for moves that add or remove choices)."
  ([selection]
   (fn [trace key] (select-keys (gfi/regenerate trace selection key) [:trace :weight])))
  ([proposal proposal-args] (reweighting proposal proposal-args {}))
  ([proposal proposal-args opts]
   (fn [trace key]
     (let [[k-propose k-update] (rng/split key)]
       (mh/move trace proposal proposal-args opts k-propose k-update)))))

(defn move-reweight
  "Each particle's trace moved by a reweighting kernel (see reweighting),
   and its log weight increased by the move's (GenParticleFilters'
   pf_move_reweight!; Marques and Storvik 2013). The estimate of the
   marginal likelihood stays unbiased. A batched state becomes one of
   scalar traces."
  ([state kernel] (move-reweight state kernel (gfi/next-key)))
  ([state kernel key]
   (let [state (unbatched state)
         results (mapv kernel (:traces state) (rng/split key (count (:traces state))))
         log-weights (mapv + (:log-weights state) (mapv :weight results))]
     (array/eval! [(mapv (comp gfi/eval-trace! :trace) results) log-weights])
     (assoc state :traces (mapv :trace results) :log-weights log-weights))))

(defn replicate
  "Each particle replaced by k copies (GenParticleFilters' pf_replicate!):
   in contiguous blocks (:layout :contiguous, the default; particle 0's
   copies are 0 … k-1) or interleaved (:interleaved; they are 0, n, 2n, …)."
  ([state k] (replicate state k {}))
  ([state k {:keys [layout] :or {layout :contiguous}}]
   (let [idx (slots layout (count (:log-weights state)) k)]
     (select-particles state idx (mapv (vec (:log-weights state)) idx)))))

(defn dereplicate
  "One particle kept out of every group of k (GenParticleFilters'
   pf_dereplicate!), the groups laid out as replicate makes them (:layout).
   :method :keepfirst (the default) keeps each group's first particle and
   its weight, undoing replicate; :sample draws one in proportion to the
   group's weights and gives it the group's mean weight (:key)."
  ([state k] (dereplicate state k {}))
  ([state k {:keys [layout method key] :or {layout :contiguous method :keepfirst}}]
   (let [n-old (count (:log-weights state))
         n (quot n-old k)
         _ (when-not (zero? (rem n-old k))
             (throw (ex-info (str "dereplicate: " n-old " particles are not groups of " k)
                             {:particles n-old :k k})))
         groups (let [g (slots layout n k)] (mapv (group-by g (range n-old)) (range n)))]
     (case method
       :keepfirst (let [idx (mapv first groups)]
                    (select-particles state idx (mapv (vec (:log-weights state)) idx)))
       :sample (let [lw (host-log-weights state)
                     ;; a group of weight 0 keeps its first particle, at weight 0
                     pick (fn [group key] (if (== ##-Inf (w/log-sum-exp (mapv lw group)))
                                            (first group)
                                            (group (first (w/multinomial (mapv lw group) 1 key)))))
                     idx (mapv pick groups (rng/split (or key (gfi/next-key)) n))]
                 (select-particles state idx (mapv (fn [group] (- (w/log-sum-exp (mapv lw group)) (Math/log k))) groups)))))))

(defn- host-choices [trace]
  (let [c (:choices trace)]
    (zipmap (cm/addresses c) (map array/->clj (cm/values c)))))

(defn coalesce
  "The particles merged where (by trace) is equal (GenParticleFilters'
   pf_coalesce!; :by defaults to the trace's choices, as host values): one
   particle for each value, the first with it, weighing the sum of their
   weights times m/n (m particles left of n), so the estimate of the
   marginal likelihood stays as it was."
  ([state] (coalesce state {}))
  ([state {:keys [by] :or {by host-choices}}]
   (let [lw (host-log-weights state)
         trs (traces state)
         ;; in order of first appearance
         members (->> (range (count trs)) (group-by #(by (trs %))) vals (sort-by first))
         log-ratio (- (Math/log (count members)) (Math/log (count lw)))]
     (select-particles state (mapv first members)
                       (mapv (fn [is] (+ (w/log-sum-exp (mapv lw is)) log-ratio)) members)))))

(defn introduce
  "The state with n new particles generated with the observations
   (GenParticleFilters' pf_introduce!), from :model on :args (by default the
   first trace's), by its internal proposal or :proposal on
   :proposal-args. The marginal likelihood accumulated so far moves into
   the old particles' weights."
  ([state observations n] (introduce state observations n {}))
  ([state observations n {:keys [model args] :as opts}]
   (let [state (unbatched state)
         first-trace (first (:traces state))
         fresh (initialize (or model (:gen-fn first-trace)) (or args (:args first-trace)) observations n
                           (select-keys opts [:proposal :proposal-args :key]))
         log-ml (:log-ml state)]
     (assoc (dissoc state :parents)
            :traces (into (:traces state) (:traces fresh))
            :log-weights (into (mapv #(+ % log-ml) (:log-weights state)) (:log-weights fresh))
            :log-ml 0.0))))

(defn- probabilities [state] (mapv #(Math/exp %) (:log-weights (w/normalize (:log-weights state)))))

(defn mean
  "The weighted mean of (f trace) over the particles."
  [state f]
  (reduce + (map #(* %1 (f %2)) (probabilities state) (traces state))))

(defn variance
  "The weighted variance of (f trace) over the particles."
  [state f]
  (let [m (mean state f)]
    (reduce + (map #(* %1 (array/square (- (f %2) m))) (probabilities state) (traces state)))))

(defn proportions
  "The total weight of each value of (f trace) (GenParticleFilters'
   proportionmap): a map of host values to probabilities."
  [state f]
  (reduce (fn [m [p tr]] (update m (array/->clj (f tr)) (fnil + 0.0) p))
          {} (map vector (probabilities state) (traces state))))

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

(defn log-weights [state] (:log-weights state))

(defn sample-unweighted-traces
  "n traces drawn in proportion to the particles' weights."
  ([state n] (sample-unweighted-traces state n (gfi/next-key)))
  ([state n key]
   (let [idx (w/multinomial (:log-weights state) n key)]
     (if-let [ps (:particles state)]
       (batched/traces (batched/take ps idx))
       (mapv (:traces state) idx)))))
