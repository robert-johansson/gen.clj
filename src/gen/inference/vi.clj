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
   iteration's objective estimate. Only q's parameters are trained (Gen.jl's
   optional model_update, training the model's at once, is not here)."
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
  "A trace of q and its log weight: [q-trace log-w]. The model generates
   with the observations and q's choices as constraints, so latent choices
   q does not make come from the model's internal proposal (as Gen.jl's)."
  [model model-args observations q q-args key]
  (let [[kq km] (rng/split key)
        qt (gfi/simulate q q-args kq)
        {lp :weight} (gfi/generate model model-args (cm/merge observations (:choices qt)) km)]
    [qt (double (array/item (- lp (:score qt))))]))

(defn elbo-gradient
  "One iteration's estimates: {:objective the mean log weight (the ELBO),
   :grads its score-function gradient with respect to q's store} from n
   draws (see the namespace doc for :baseline)."
  [model model-args observations q q-args store n {:keys [baseline key]}]
  (when (and (= baseline :leave-one-out) (< n 2))
    (throw (ex-info (str "A leave-one-out baseline needs at least 2 samples, given " n) {:samples n})))
  (let [q (dyn/with-params q store)
        runs (mapv #(draw model model-args observations q q-args %) (rng/split (or key (gfi/next-key)) n))
        lws (mapv second runs)
        total (reduce clojure.core/+ lws)
        signal (case baseline
                 nil lws
                 :leave-one-out (mapv #(- % (/ (- total %) (dec n))) lws))]
    {:objective (/ total n)
     :grads (grad/weighted-sum (map #(/ % n) signal) (map (comp grad/param-gradients first) runs))}))

(defn vimco-gradient
  "One iteration's estimates for the K-sample bound: {:objective
   log (1/K) Σ w_k, :grads its VIMCO gradient with respect to q's store}.
   Sample j's baseline replaces its weight by the geometric mean of the
   others' (:baseline :geometric, the default) or their arithmetic mean
   (:arithmetic), as Gen.jl's geometric keyword."
  [model model-args observations q q-args store k {:keys [baseline key] :or {baseline :geometric}}]
  (when (< k 2)
    (throw (ex-info (str "VIMCO's leave-one-out control variates need K of at least 2, given " k) {:samples k})))
  (let [q (dyn/with-params q store)
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
    {:objective bound
     :grads (grad/weighted-sum cs (map (comp grad/param-gradients first) runs))}))

(defn- fit [estimate store optimizer {:keys [iters key] :or {iters 1000}}]
  (loop [i 0 store store state (learn/init optimizer store) history [] key (or key (gfi/next-key))]
    (if (< i iters)
      (let [[k k-next] (rng/split key)
            {:keys [objective grads]} (estimate store k)
            [store state] (learn/step optimizer state store grads)]
        (recur (inc i) store state (conj history objective) k-next))
      {:store store :history history})))

(defn black-box-vi
  "Fit q's store by ascending the ELBO (Gen.jl's black_box_vi!). Options:
   :iters (1000), :samples per iteration (20), :baseline (nil or
   :leave-one-out), :key."
  [model model-args observations q q-args store optimizer {:keys [samples] :or {samples 20} :as opts}]
  (fit #(elbo-gradient model model-args observations q q-args %1 samples (assoc opts :key %2))
       store optimizer opts))

(defn black-box-vimco
  "Fit q's store by ascending the K-sample bound with VIMCO's control
   variates (Gen.jl's black_box_vimco!). Options: :iters (1000), :samples K
   (10), :estimates, the independent K-sample estimates averaged per
   iteration (1; Gen.jl's samples_per_iter), :baseline (:geometric or
   :arithmetic), :key."
  [model model-args observations q q-args store optimizer {:keys [samples estimates] :or {samples 10 estimates 1} :as opts}]
  (fit (fn [store key]
         (let [runs (mapv #(vimco-gradient model model-args observations q q-args store samples (assoc opts :key %))
                          (rng/split key estimates))]
           {:objective (/ (reduce clojure.core/+ (map :objective runs)) estimates)
            :grads (grad/mean (map :grads runs))}))
       store optimizer opts))
