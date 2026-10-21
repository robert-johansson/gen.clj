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
   variance. black-box-vimco ascends the K-sample bound
   E[log (1/K) Σ w_k] with VIMCO's leave-one-out control variates (Mnih and
   Rezende 2016). Both return {:store :history}, the history holding each
   iteration's objective estimate.

   The model's parameters train at once given :model-store and
   :model-optimizer (Gen.jl's model_update): the model runs under that
   store (with-params), and each iteration also ascends along the gradient
   in its parameters, ∇θ log p(observations, z) averaged over the draws
   (the ELBO's), or weighted by the normalized weights w̃_k (the K-sample
   bound's). q's draws do not depend on θ, so each is the exact gradient of
   the iteration's objective estimate, and unbiased. The result then holds
   :model-store too."
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.weights :as w]
            [gen.learning :as learn]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- draw
  "A trace of q, its log weight and the model's trace: [q-trace log-w
   model-trace]. The model generates with the observations and q's choices
   as constraints, so latent choices q does not make come from the model's
   internal proposal (as Gen.jl's)."
  [model model-args observations q q-args key]
  (let [[kq km] (rng/split key)
        qt (gfi/simulate q q-args kq)
        {mt :trace lp :weight} (gfi/generate model model-args (cm/merge observations (:choices qt)) km)]
    [qt (double (array/item (- lp (:score qt)))) mt]))

(defn- model-under
  "The model under its store, when its parameters are trained."
  [model model-store]
  (if model-store (dyn/with-params model model-store) model))

(defn- model-gradients
  "Σ_i c_i ∇θ log p of the runs' model traces."
  [cs runs]
  (grad/weighted-sum cs (map #(grad/param-gradients (nth % 2)) runs)))

(defn elbo-gradient
  "One iteration's estimates: {:objective the mean log weight (the ELBO),
   :grads its score-function gradient with respect to q's store} from n
   draws (see the namespace doc for :baseline); with :model-store, also
   :model-grads, its gradient with respect to the model's store."
  [model model-args observations q q-args store n {:keys [baseline key model-store]}]
  (when (and (= baseline :leave-one-out) (< n 2))
    (throw (ex-info (str "A leave-one-out baseline needs at least 2 samples, given " n) {:samples n})))
  (let [q (dyn/with-params q store)
        model (model-under model model-store)
        runs (mapv #(draw model model-args observations q q-args %) (rng/split (or key (gfi/next-key)) n))
        lws (mapv second runs)
        total (reduce clojure.core/+ lws)
        signal (case baseline
                 nil lws
                 :leave-one-out (mapv #(- % (/ (- total %) (dec n))) lws))]
    (cond-> {:objective (/ total n)
             :grads (grad/weighted-sum (map #(/ % n) signal) (map (comp grad/param-gradients first) runs))}
      model-store (assoc :model-grads (model-gradients (repeat n (/ 1.0 n)) runs)))))

(defn vimco-gradient
  "One iteration's estimates for the K-sample bound: {:objective
   log (1/K) Σ w_k, :grads its VIMCO gradient with respect to q's store}.
   Sample j's baseline replaces its weight by the geometric mean of the
   others' (:baseline :geometric, the default) or their arithmetic mean
   (:arithmetic), as Gen.jl's geometric keyword. With :model-store, also
   :model-grads, the bound's gradient with respect to the model's store."
  [model model-args observations q q-args store k {:keys [baseline key model-store] :or {baseline :geometric}}]
  (when (< k 2)
    (throw (ex-info (str "VIMCO's leave-one-out control variates need K of at least 2, given " k) {:samples k})))
  (let [q (dyn/with-params q store)
        model (model-under model model-store)
        runs (mapv #(draw model model-args observations q q-args %) (rng/split (or key (gfi/next-key)) k))
        lws (mapv second runs)
        bound (w/log-mean-exp lws)
        normalized (mapv #(Math/exp %) (:log-weights (w/normalize lws)))
        loo (fn [j] (let [others (keep-indexed #(when (not= j %1) %2) lws)
                          stand-in (case baseline
                                     :geometric (/ (reduce clojure.core/+ others) (dec k))
                                     :arithmetic (w/log-mean-exp others))]
                      (w/log-mean-exp (assoc lws j stand-in))))
        ;; ∇ of the bound: Σ_j (L - L_{-j}) ∇ log q_j - Σ_j w̃_j ∇ log q_j
        cs (mapv (fn [j] (- (- bound (loo j)) (normalized j))) (range k))]
    (cond-> {:objective bound
             :grads (grad/weighted-sum cs (map (comp grad/param-gradients first) runs))}
      ;; ∇θ of the bound: Σ_j w̃_j ∇θ log p(observations, z_j)
      model-store (assoc :model-grads (model-gradients normalized runs)))))

(defn- fit
  "Ascend from store (and model-store, given) along (estimate store
   model-store key)'s :grads (and :model-grads)."
  [estimate store optimizer {:keys [iters key model-store model-optimizer] :or {iters 1000}}]
  (when-not (= (some? model-store) (some? model-optimizer))
    (throw (ex-info "Training the model's parameters takes both :model-store and :model-optimizer"
                    {:model-store? (some? model-store) :model-optimizer? (some? model-optimizer)})))
  (loop [i 0 store store state (learn/init optimizer store)
         model-store model-store model-state (some->> model-store (learn/init model-optimizer))
         history [] key (or key (gfi/next-key))]
    (if (< i iters)
      (let [[k k-next] (rng/split key)
            {:keys [objective grads model-grads]} (estimate store model-store k)
            [store state] (array/eval! (learn/step optimizer state store grads))
            [model-store model-state] (if model-store
                                        (array/eval! (learn/step model-optimizer model-state model-store model-grads))
                                        [nil nil])]
        (recur (inc i) store state model-store model-state (conj history objective) k-next))
      (cond-> {:store store :history history}
        model-store (assoc :model-store model-store)))))

(defn black-box-vi
  "Fit q's store by ascending the ELBO (Gen.jl's black_box_vi!). Options:
   :iters (1000), :samples per iteration (20), :baseline (nil or
   :leave-one-out), :model-store and :model-optimizer (to train the
   model's parameters too), :key."
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
   parameters too), :key."
  [model model-args observations q q-args store optimizer {:keys [samples estimates] :or {samples 10 estimates 1} :as opts}]
  (fit (fn [store model-store key]
         (let [runs (mapv #(vimco-gradient model model-args observations q q-args store samples
                                           (assoc opts :key % :model-store model-store))
                          (rng/split key estimates))]
           (cond-> {:objective (/ (reduce clojure.core/+ (map :objective runs)) estimates)
                    :grads (grad/mean (map :grads runs))}
             model-store (assoc :model-grads (grad/mean (map :model-grads runs))))))
       store optimizer opts))
