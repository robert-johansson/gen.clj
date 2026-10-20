(ns gen.learning
  "Optimizers for trainable-parameter stores (gen.dynamic/with-params), as
   pure functions: Gen.jl's ParamUpdate and apply!, without mutation. Each
   optimizer takes a gradient-ascent step on the objective whose gradients
   are given (e.g. log densities, from gen.gradients/param-gradients), and
   returns the new store with the optimizer's new state.

     (def opt (adam {:lr 0.01}))
     (loop [store store state (init opt store) …]
       (let [[store state] (step opt state store grads)] …))

   Stores and gradients are trees (maps of names to arrays or numbers) of
   the same shape. So one step updates several generative functions' stores
   together (Gen.jl's ParamUpdate with gf => params pairs) when they are
   nested in a map, {:model store :proposal store}, and only the
   parameters given gradients when the gradients name a subset; a
   gradient for a name the store lacks is an error.

   lecture and lecture-batched are the sleep phase of wake-sleep (Gen.jl's
   lecture! and lecture_batched!): the proposal's log density of the
   latent choices of the model's own traces, with its gradient, to train a
   proposal toward the model's posterior."
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.proposal :as proposal]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn- update-given
  "x with (f x-leaf g-leaf more-leaves…) at every leaf of the gradient tree
   g, the trees in more walked alongside g; x's other entries are kept, so
   a step updates the parameters it is given. A gradient for a name x lacks
   is an error."
  [f x g & more]
  (cond (map? g) (reduce-kv (fn [acc k gv]
                              (when-not (contains? x k)
                                (throw (ex-info (str "A gradient for " (pr-str k) ", which the store does not have")
                                                {:name k :store-names (vec (keys x))})))
                              (assoc acc k (apply update-given f (get x k) gv (map #(get % k) more))))
                            x g)
        (vector? g) (apply mapv #(apply update-given f %1 %2 %&) x g more)
        :else (apply f x g more)))

(defn sgd
  "Fixed-step gradient ascent: θ + lr·g (Gen.jl's FixedStepGradientDescent)."
  [{:keys [lr]}]
  {:kind :sgd :lr lr})

(defn gradient-descent
  "Gradient ascent with a decaying step: step-size-init · (beta + 1) /
   (beta + t) at step t (Gen.jl's GradientDescent). Options:
   :step-size-init, :step-size-beta."
  [{:keys [step-size-init step-size-beta]}]
  {:kind :gradient-descent :init step-size-init :beta step-size-beta})

(defn adam
  "Adam (Kingma and Ba 2015), ascending. Options: :lr (0.001), :beta1 (0.9),
   :beta2 (0.999), :eps (1e-8)."
  [{:keys [lr beta1 beta2 eps] :or {lr 0.001 beta1 0.9 beta2 0.999 eps 1e-8}}]
  {:kind :adam :lr lr :beta1 beta1 :beta2 beta2 :eps eps})

(defn init
  "The optimizer's initial state for store."
  [{:keys [kind]} store]
  (case kind
    :sgd nil
    :gradient-descent {:t 0}
    :adam {:t 0 :m (array/tree-map array/zeros-like store) :v (array/tree-map array/zeros-like store)}))

(defn step
  "[store' state']: one ascent step from store along the gradients."
  [{:keys [kind lr beta1 beta2 eps] :as optimizer} state store grads]
  (case kind
    :sgd [(update-given #(+ %1 (* lr %2)) store grads) nil]
    :gradient-descent (let [t (inc (:t state))
                            {:keys [init beta]} optimizer
                            size (/ (* init (+ beta 1.0)) (+ beta t))]
                        [(update-given #(+ %1 (* size %2)) store grads) {:t t}])
    :adam (let [t (inc (:t state))
                m (update-given #(+ (* beta1 %1) (* (- 1.0 beta1) %2)) (:m state) grads)
                v (update-given #(+ (* beta2 %1) (* (- 1.0 beta2) (* %2 %2))) (:v state) grads)
                c1 (- 1.0 (Math/pow beta1 t))
                c2 (- 1.0 (Math/pow beta2 t))]
            [(update-given (fn [theta _ mi vi] (+ theta (* lr (/ (/ mi c1) (+ (array/sqrt (/ vi c2)) eps)))))
                           store grads m v)
             {:t t :m m :v v}])))

(defn train
  "Fit the trainable parameters of gf by maximizing the log density of
   generated data (Gen.jl's train!): each of :epochs epochs takes
   :batches-per-epoch steps, each step drawing :batch-size examples
   [args constraints] from (data-fn key), averaging their param-gradients
   (through gradients-fn, gen.gradients/param-gradients) and taking one
   optimizer step from store. Returns {:store :history}, the history holding
   each epoch's mean log density of the last batch. Options also take :key."
  [gf store optimizer data-fn gradients-fn
   {:keys [epochs batches-per-epoch batch-size key] :or {epochs 1 batches-per-epoch 1 batch-size 1}}]
  (let [split rng/split]
    (loop [epoch 0 store store state (init optimizer store) history [] key (or key (gfi/next-key))]
      (if (< epoch epochs)
        (let [[k-epoch k-next] (split key)
              [store state score]
              (reduce (fn [[store state _] k-batch]
                        (let [results (map (fn [k]
                                             (let [[k-data k-gen] (split k)
                                                   [args constraints] (data-fn k-data)]
                                               (:trace (gfi/generate (dyn/with-params gf store) args constraints k-gen))))
                                           (split k-batch batch-size))
                              traces (vec results)
                              [store state] (array/eval! (step optimizer state store (grad/mean (map gradients-fn traces))))]
                          [store state (/ (reduce + (map #(double (array/item (:score %))) traces)) batch-size)]))
                      [store state nil]
                      (split k-epoch batches-per-epoch))]
          (recur (inc epoch) store state (conj history score) k-next))
        {:store store :history history}))))

(defn lecture
  "The sleep phase of wake-sleep (Gen.jl's lecture!): a trace of model on
   model-args, and the proposal, with its parameter store, scored on the
   choices of that trace it makes given (q-args-fn trace). {:score log q of
   those choices, :grads its gradient with respect to the store}. Ascending
   it trains the proposal toward the model's posterior given what q reads."
  [model model-args proposal store q-args-fn key]
  (let [[k-simulate k-generate] (rng/split key)
        tr (gfi/simulate model model-args k-simulate)
        q (dyn/with-params proposal store)
        q-args (q-args-fn tr)
        choices (proposal/choices-of q q-args (:choices tr))
        q-trace (:trace (gfi/generate q q-args choices k-generate))]
    {:score (:score q-trace) :grads (grad/param-gradients q-trace)}))

(defn lecture-batched
  "lecture over n of the model's traces: the mean score and gradient
   (Gen.jl's lecture_batched!)."
  [model model-args proposal store q-args-fn n key]
  (when-not (pos-int? n)
    (throw (ex-info (str "lecture-batched needs at least one trace, given " n) {:n n})))
  (let [runs (mapv #(lecture model model-args proposal store q-args-fn %) (rng/split key n))]
    {:score (/ (reduce + (map #(double (array/item (:score %))) runs)) n)
     :grads (grad/mean (map :grads runs))}))
