(ns gen.inference.vi
  "Black-box variational inference with a generative function as the
   variational family (Gen.jl's black_box_vi! and black_box_vimco!).

   q is a generative function with a trainable-parameter store
   (gen.dynamic/with-params) whose choices are the model's latent ones:

     (black-box-vi model model-args observations q q-args store optimizer
                   {:iters 1000 :samples 20})

   Each iteration draws z ~ q, computes the log weight
   log p(observations, z) - log q(z), and ascends the ELBO, the mean log
   weight, by the score-function estimator: the mean of log w · ∇ log q(z)
   (Gen.jl's, with :baseline nil), or with a leave-one-out baseline
   (:baseline :leave-one-out), which keeps it unbiased and lowers its
   variance. With :estimator :reparam, by the pathwise (reparameterization)
   estimator instead, ADVI's: z is q's sample for a fixed key, a
   differentiable function of q's store, so the gradient of the mean log
   weight itself is unbiased, usually with far less variance. It needs every
   choice q makes from a family with a pathwise sampler (gen.dist.core/
   pathwise?: normal, uniform, exponential, …; a bernoulli is a loud
   error), and every latent choice of the model proposed by q (the model is
   assessed on the observations and q's choices). black-box-vimco ascends the K-sample bound
   E[log (1/K) Σ w_k] with VIMCO's leave-one-out control variates (Mnih and
   Rezende 2016). Both return {:store :history}, the history holding each
   iteration's objective estimate.

   The model's parameters train at once given :model-store and
   :model-optimizer (Gen.jl's model_update): the model runs under that
   store (with-params), and each iteration also ascends along the gradient
   in its parameters, ∇θ log p(observations, z) averaged over the draws
   (the ELBO's), or weighted by the normalized weights w̃_k (the K-sample
   bound's). q's draws do not depend on θ, but a latent r the model draws
   itself, which q does not propose, may: its score-function term,
   ∇θ log f_θ(r) times the learning signal (the ELBO's log weight less a
   leave-one-out baseline, VIMCO's own leave-one-out signal), is added,
   f_θ the density the model's internal proposal drew it with (see
   model-gradients). So each is an unbiased estimate of the objective's
   gradient. The result then holds :model-store too.

   With :batched? true, an iteration's draws run at once (gen.batched): q's
   n samples drawn as one batched run, and the model's weights and every
   gradient computed for all of them in one batched pass
   (gen.batched/map-particles; each draw's gradients its own, as vmap's),
   falling back to one draw at a time where q or the model cannot run
   batched, which the results record as :fallback (a written message);
   :batched? :required throws with that reason instead. For one key the
   draws differ from the unbatched ones, with the same distribution."
  {:api :stable}
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.weights :as w]
            [gen.learning :as learn]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(defn- unknown-baseline [baseline baselines]
  (throw (ex-info (str "The :baseline is one of " (pr-str baselines) ", not " (pr-str baseline))
                  {:baseline baseline :baselines baselines})))

(defn- model-under
  "The model under its store, when its parameters are trained."
  [model model-store]
  (if model-store (dyn/with-params model model-store) model))

(defn- own-draws-gradient
  "∇θ log f_θ(r) for the model's trace mt, generated with constraints: the
   gradient in the model's store of the density its internal proposal drew
   the other latents r with, score - project(constraints' addresses) at
   mt's choices; nil when the constraints hold every choice."
  [mt constraints]
  (when (some #(not (cm/has-value? constraints %)) (cm/addresses (:choices mt)))
    (let [gf (:gen-fn mt)
          constrained (sel/from-paths (cm/addresses constraints))
          log-f (fn [stores]
                  (gfi/with-exact-density "VI's model gradient"
                    (let [t (:trace (gfi/generate (gfi/with-stores gf stores) (:args mt) (:choices mt) (rng/key 0)))]
                      (- (:score t) (gfi/project t constrained)))))]
      ((array/grad log-f) (gfi/stores gf)))))

(defn- weighed
  "One draw's log weight and gradients, from qt, a trace of q: {:lw log
   p(observations, z, r) - log q(z), :q-grads ∇ log q(z) in q's store},
   and with model?, ∇θ log p as :joint and ∇θ log f_θ(r) as :own (or nil:
   own-draws-gradient) in the model's. The model generates with the
   observations and q's choices as constraints, so latent choices r that q
   does not make come from the model's internal proposal (as Gen.jl's)."
  [model model-args observations model? qt key]
  (let [constraints (cm/merge observations (:choices qt))
        {mt :trace lp :weight} (gfi/generate model model-args constraints key)]
    (cond-> {:lw (- lp (:score qt)) :q-grads (grad/param-gradients qt)}
      model? (assoc :joint (grad/param-gradients mt) :own (own-draws-gradient mt constraints)))))

(defn- draws
  "{:runs n draws of q, weighed (each :lw a host double) :fallback}: one at
   a time, each from its own key (key split n ways, then in two: q's and
   the model's), or with batched?, all at once (gen.batched: q's n
   particles from key's first split, the model's runs from its second),
   :fallback why they ran one at a time, if they did; batched? :required
   throws with that reason instead."
  [model model-args observations q q-args n model? batched? key]
  (if batched?
    (binding [batched/*required* (= :required batched?)]
      (let [[kq km] (rng/split key)
            ps (batched/simulate q q-args n kq)
            {all :result m :fallback} (batched/map-particles* ps #(weighed model model-args observations model? %1 %2) km)]
        (cond-> {:runs (mapv (fn [i lw] (assoc (array/tree-map #(some-> % (array/take i 0)) all) :lw lw))
                             (range n) (map double (array/->clj (:lw all))))}
          (or (batched/fallback ps) m) (assoc :fallback (or (batched/fallback ps) m)))))
    {:runs (mapv (fn [k]
                   (let [[kq km] (rng/split k)]
                     (update (weighed model model-args observations model? (gfi/simulate q q-args kq) km) :lw #(double (array/item %)))))
                 (rng/split key n))}))

(defn- model-gradients
  "Σ_i c_i ∇θ log w_i + Σ_i s_i ∇θ log f_θ(r_i) over the runs, the
   objective's gradient in the model's store (the post-M5 review's P3-01).
   The draw i has log w_i = log p(x, z_i, r_i) - log f_θ(r_i) - log q(z_i),
   r_i the latents the model drew itself from its internal proposal f_θ,
   so for an objective E[F(log w_1 … log w_n)]

     ∇θ E[F] = E[Σ_i ∂F/∂log w_i · ∇θ log w_i] + E[F · Σ_i ∇θ log f_θ(r_i)].

   c_i is ∂F/∂log w_i: 1/n for the ELBO, w̃_i for the K-sample bound. s_i
   is F less a baseline that does not depend on r_i, which keeps the
   estimate unbiased since E[∇θ log f_θ(r_i)] = 0: for the ELBO,
   (log w_i - b_i)/n with b_i the mean log weight of the other draws (none
   with one draw); for the bound, L - L₋ᵢ, VIMCO's own signal. With
   ∇θ log w_i = ∇θ log p - ∇θ log f_θ(r_i), this is Σ_i c_i ∇θ log p +
   Σ_i (s_i - c_i) ∇θ log f_θ(r_i). The -∇θ log f_θ(r_i) has mean 0 too,
   but keeping it, with the baseline, lowers the variance. The ELBO's sd
   per iteration of 10 draws, for a coin b ~ bern(θ) and a shift
   z ~ N(θ, 1): 0.279 and 1.918; with the term dropped, 0.680 and 1.892;
   with no baseline, 1.197 and 2.997, or 0.573 and 2.752 dropped."
  [cs signal runs]
  (let [joint (map (fn [c r] [c (:joint r)]) cs runs)
        own (for [[c s {g :own}] (map vector cs signal runs) :when g] [(- s c) g])
        terms (concat joint own)]
    (grad/weighted-sum (map first terms) (map second terms))))

(defn- pathwise-q
  "q with middleware that refuses a choice from a family without a pathwise
   sampler, where the pathwise gradient would silently be zero. (Its own
   sites: a generative function q calls keeps its own transitions.)"
  [q]
  (dyn/with-transitions
    q {:simulate (fn [state path dist key]
                   (when-not (dc/pathwise? dist)
                     (throw (ex-info (str "The reparameterization estimator needs every choice of q from a family with a"
                                          " pathwise sampler (gen.dist.core/pathwise?); the choice at " path " is from "
                                          (name (:type dist)) ": use :estimator :score")
                                     {:path path :type (:type dist)})))
                   ((dyn/site-transitions :simulate) state path dist key))}))

(defn- reparam-elbo-gradient
  "elbo-gradient by the pathwise estimator: the ELBO estimate from n draws,
   z_i q's sample for key i, a function of q's store, and its gradient with
   respect to the store (and the model's, given) by differentiating the
   estimate itself."
  [model model-args observations q q-args store n {:keys [key model-store batched?]}]
  (when batched?
    (throw (ex-info "The reparameterization estimator runs its draws one at a time: :batched? is not supported with it"
                    {:estimator :reparam})))
  (let [q (pathwise-q q)
        ks (rng/split (or key (gfi/next-key)) n)
        elbo (fn [q-store m-store]
               (gfi/with-exact-density "The reparameterization gradient"
                 (let [qs (gfi/with-stores q q-store)
                       m (if m-store (gfi/with-stores model m-store) model)
                       lw (fn [k]
                            (let [[kq km] (rng/split k)
                                  qt (gfi/simulate qs q-args kq)
                                  constraints (cm/merge observations (:choices qt))]
                              (- (:weight (gfi/assess m model-args constraints km)) (:score qt))))]
                   (/ (reduce + (map lw ks)) n))))]
    (if model-store
      (let [[v [g mg]] ((array/value-and-grad elbo {:argnums [0 1]}) store model-store)]
        {:objective (double (array/item v)) :grads g :model-grads mg})
      (let [[v g] ((array/value-and-grad #(elbo % nil)) store)]
        {:objective (double (array/item v)) :grads g}))))

(defn- score-elbo-gradient
  "elbo-gradient by the score-function estimator."
  [model model-args observations q q-args store n {:keys [baseline key model-store batched?]}]
  (when (and (= baseline :leave-one-out) (< n 2))
    (throw (ex-info (str "A leave-one-out baseline needs at least 2 samples, given " n) {:samples n})))
  (let [q (dyn/with-params q store)
        model (model-under model model-store)
        {runs :runs fallback :fallback} (draws model model-args observations q q-args n (some? model-store) batched? (or key (gfi/next-key)))
        lws (mapv :lw runs)
        total (reduce + lws)
        ;; log w less the mean log weight of the other draws
        loo (when (> n 1) (mapv #(- % (/ (- total %) (dec n))) lws))
        signal (case baseline
                 nil lws
                 :leave-one-out loo
                 (unknown-baseline baseline [nil :leave-one-out]))]
    (cond-> {:objective (/ total n)
             :grads (grad/weighted-sum (map #(/ % n) signal) (map :q-grads runs))}
      ;; the model's own draws' score term takes the leave-one-out
      ;; baseline whenever there are two draws or more, whatever q's
      model-store (assoc :model-grads (model-gradients (repeat n (/ 1.0 n)) (map #(/ % n) (or loo lws)) runs))
      fallback (assoc :fallback fallback))))

(defn elbo-gradient
  "One iteration's estimates: {:objective the mean log weight (the ELBO),
   :grads its gradient with respect to q's store} from n draws, by the
   score-function estimator, or with :estimator :reparam the pathwise one
   (see the namespace doc for :baseline, :estimator and :batched?); with
   :model-store, also :model-grads, its gradient with respect to the model's
   store; with :batched?, :fallback, why the draws ran one at a time, if
   they did (:batched? :required throws with it instead)."
  [model model-args observations q q-args store n {:keys [estimator] :as opts}]
  (case (or estimator :score)
    :score (score-elbo-gradient model model-args observations q q-args store n opts)
    :reparam (reparam-elbo-gradient model model-args observations q q-args store n opts)
    (throw (ex-info (str "The :estimator is :score or :reparam, not " (pr-str estimator)) {:estimator estimator}))))

(defn vimco-gradient
  "One iteration's estimates for the K-sample bound: {:objective
   log (1/K) Σ w_k, :grads its VIMCO gradient with respect to q's store}.
   Sample j's baseline replaces its weight by the geometric mean of the
   others' (:baseline :geometric, the default) or their arithmetic mean
   (:arithmetic), as Gen.jl's geometric keyword. With :model-store, also
   :model-grads, the bound's gradient with respect to the model's store;
   :batched? as elbo-gradient's."
  [model model-args observations q q-args store k {:keys [baseline key model-store batched?] :or {baseline :geometric}}]
  (when (< k 2)
    (throw (ex-info (str "VIMCO's leave-one-out control variates need K of at least 2, given " k) {:samples k})))
  (let [q (dyn/with-params q store)
        model (model-under model model-store)
        {runs :runs fallback :fallback} (draws model model-args observations q q-args k (some? model-store) batched? (or key (gfi/next-key)))
        lws (mapv :lw runs)
        bound (w/log-mean-exp lws)
        normalized (mapv #(Math/exp %) (:log-weights (w/normalize lws)))
        loo (fn [j] (let [others (keep-indexed #(when (not= j %1) %2) lws)
                          stand-in (case baseline
                                     :geometric (/ (reduce + others) (dec k))
                                     :arithmetic (w/log-mean-exp others)
                                     (unknown-baseline baseline [:geometric :arithmetic]))]
                      (w/log-mean-exp (assoc lws j stand-in))))
        signal (mapv #(- bound (loo %)) (range k))
        ;; ∇ of the bound: Σ_j (L - L_{-j}) ∇ log q_j - Σ_j w̃_j ∇ log q_j
        cs (mapv - signal normalized)]
    (cond-> {:objective bound
             :grads (grad/weighted-sum cs (map :q-grads runs))}
      model-store (assoc :model-grads (model-gradients normalized signal runs))
      fallback (assoc :fallback fallback))))

(defn- fit
  "Ascend from store (and model-store, given) along (estimate store
   model-store key)'s :grads (and :model-grads); :fallback, the first
   iteration's that fell back to one draw at a time."
  [estimate store optimizer {:keys [iters key model-store model-optimizer] :or {iters 1000}}]
  (when-not (= (some? model-store) (some? model-optimizer))
    (throw (ex-info "Training the model's parameters takes both :model-store and :model-optimizer"
                    {:model-store? (some? model-store) :model-optimizer? (some? model-optimizer)})))
  (loop [i 0 store store state (learn/init optimizer store)
         model-store model-store model-state (some->> model-store (learn/init model-optimizer))
         history [] key (or key (gfi/next-key)) fallback nil]
    (if (< i iters)
      (let [[k k-next] (rng/split key)
            {:keys [objective grads model-grads] m :fallback} (estimate store model-store k)
            [store state] (learn/step optimizer state store grads)
            [model-store model-state] (if model-store
                                        (learn/step model-optimizer model-state model-store model-grads)
                                        [nil nil])]
        (recur (inc i) store state model-store model-state (conj history objective) k-next (or fallback m)))
      (cond-> {:store store :history history}
        model-store (assoc :model-store model-store)
        fallback (assoc :fallback fallback)))))

(defn black-box-vi
  "Fit q's store by ascending the ELBO (Gen.jl's black_box_vi!). Options:
   :iters (1000), :samples per iteration (20), :estimator (:score, or
   :reparam: ADVI's pathwise gradient), :baseline (nil or :leave-one-out;
   the score estimator's), :model-store and :model-optimizer (to train the
   model's parameters too), :batched? (false: the draws one at a time; the
   score estimator's), :key."
  [model model-args observations q q-args store optimizer {:keys [samples] :or {samples 20} :as opts}]
  (fit (fn [store model-store key]
         (elbo-gradient model model-args observations q q-args store samples
                        (assoc opts :key key :model-store model-store)))
       store optimizer opts))

(defn black-box-vimco
  "Fit q's store by ascending the K-sample bound with VIMCO's control
   variates (Gen.jl's black_box_vimco!). Options: :iters (1000), :samples K
   (10), :estimates, the independent K-sample estimates averaged per
   iteration (1; Gen.jl's samples_per_iter), :baseline (:geometric or
   :arithmetic), :model-store and :model-optimizer (to train the model's
   parameters too), :batched?, :key."
  [model model-args observations q q-args store optimizer {:keys [samples estimates] :or {samples 10 estimates 1} :as opts}]
  (fit (fn [store model-store key]
         (let [runs (mapv #(vimco-gradient model model-args observations q q-args store samples
                                           (assoc opts :key % :model-store model-store))
                          (rng/split key estimates))]
           (cond-> {:objective (/ (reduce + (map :objective runs)) estimates)
                    :grads (grad/mean (map :grads runs))
                    :fallback (some :fallback runs)}
             model-store (assoc :model-grads (grad/mean (map :model-grads runs))))))
       store optimizer opts))
