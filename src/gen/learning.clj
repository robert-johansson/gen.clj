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
   gradient for a name the store lacks is an error. A combinator's stores
   (gen.gfi/stores) are such a tree, and gfi/with-stores gives the stepped
   ones back to it.

   The optimizers: sgd, gradient-descent and adam (Gen.jl's three);
   momentum (with :nesterov?), rmsprop, adagrad and adamw (GenFluxOptimizers'
   common set, as PyTorch's update rules, ascending); and clip-norm, which
   clips the gradients by their global norm before another optimizer's
   step.

   An optimizer is a map with a :kind, which -init and -step dispatch on,
   so any namespace adds one (Gen.jl's init_update_state and
   apply_update!), and train and gen.inference.vi step it as they step the
   built-in ones:

     (defmethod learn/-init :my/sign [_ _store] nil)
     (defmethod learn/-step :my/sign [{:keys [lr]} state store grads]
       [(learn/update-given #(+ %1 (* lr (array/sign %2))) store grads) state])

     (learn/train gf store {:kind :my/sign :lr 0.01} data-fn grad/param-gradients {})

   update-given walks the gradients' tree, so such a step updates nested
   stores and the parameters given gradients, as the built-in ones do.

   train is Gen.jl's train!: epochs of minibatches drawn from each epoch's
   generated examples, and a history of the mean generate weight on
   held-out ones. lecture and lecture-batched are the sleep phase of
   wake-sleep (Gen.jl's lecture! and lecture_batched!): the proposal's log
   density of the latent choices of the model's own traces, with its
   gradient, to train a proposal toward the model's posterior."
  {:api :stable}
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.proposal :as proposal]
            [gen.random :as rng]))

(set! *warn-on-reflection* true)

(defn update-given
  "x with (f x-leaf g-leaf more-leaves…) at every leaf of the gradient tree
   g, the trees in more (an optimizer's state, of the store's shape) walked
   alongside g; x's other entries are kept, so a step updates the
   parameters it is given. A gradient for a name x lacks is an error. An
   optimizer's -step updates its store and state with it."
  [f x g & more]
  (cond (map? g) (reduce-kv (fn [acc k gv]
                              (when-not (contains? x k)
                                (throw (ex-info (str "A gradient for " (pr-str k) ", which the store does not have")
                                                {:name k :store-names (vec (keys x))})))
                              (assoc acc k (apply update-given f (get x k) gv (map #(get % k) more))))
                            x g)
        (vector? g) (apply mapv #(apply update-given f %1 %2 %&) x g more)
        :else (apply f x g more)))

;; ---------------------------------------------------------------------------
;; The interface
;; ---------------------------------------------------------------------------

(defmulti -init
  "The initial state of the optimizer for store (Gen.jl's
   init_update_state), dispatched on its :kind; nil for an optimizer with
   none."
  (fn [optimizer _store] (:kind optimizer)))

(defmulti -step
  "[store' state']: one ascent step of the optimizer from store along the
   gradient tree grads, with its state, dispatched on its :kind (Gen.jl's
   apply_update!). A pure function: step evaluates its result."
  (fn [optimizer _state _store _grads] (:kind optimizer)))

(defn- kinds
  "The :kinds of optimizer that have a -step."
  []
  (vec (sort-by str (remove #{:default} (keys (methods -step))))))

(defn- unknown-kind [kind]
  (throw (ex-info (str "No optimizer of :kind " (pr-str kind) ": the kinds are " (pr-str (kinds))
                       " (made by sgd, adam … or added with -init and -step)")
                  {:kind kind :kinds (kinds)})))

(defmethod -init :default [{:keys [kind]} _] (unknown-kind kind))

(defmethod -step :default [{:keys [kind]} _ _ _] (unknown-kind kind))

(defn init
  "The optimizer's initial state for store."
  [optimizer store]
  (-init optimizer store))

(defn step
  "[store' state']: one ascent step of the optimizer from store along the
   gradients, evaluated. A step is an inference loop's boundary: on a lazy
   backend a loop of unevaluated steps would make each store a graph over
   every step before it, which no collection frees (genclj-9e9e)."
  [optimizer state store grads]
  (let [result (-step optimizer state store grads)]
    (when-not (and (vector? result) (= 2 (count result)))
      (throw (ex-info (str "An optimizer's -step returns [store' state'], not " (pr-str (type result))
                           " (the :kind " (pr-str (:kind optimizer)) ")")
                      {:kind (:kind optimizer) :result-type (type result)})))
    (array/eval! result)))

;; ---------------------------------------------------------------------------
;; The optimizers
;; ---------------------------------------------------------------------------

(def ^:private at-least-0 [#(clojure.core/>= % 0) "a number of at least 0"])

(def ^:private above-0 [pos? "a number above 0"])

(def ^:private below-1 [#(clojure.core/<= 0 % (Math/nextDown 1.0)) "a number in [0, 1)"])

(defn- checked
  "opts, when the option at each key of ranges is a host number its range
   [pred description] holds for; a loud error naming who otherwise."
  [who opts ranges]
  (doseq [[k [ok? description]] ranges :let [x (get opts k)]]
    (when-not (and (number? x) (ok? x))
      (throw (ex-info (str who "'s " k " is " description ", not " (pr-str x)) {:option k :value x}))))
  opts)

(defn- zeros-like [store] (array/tree-map array/zeros-like store))

(defn sgd
  "Fixed-step gradient ascent: θ + lr·g (Gen.jl's FixedStepGradientDescent).
   Options: :lr."
  [{:keys [lr]}]
  (checked "sgd" {:kind :sgd :lr lr} {:lr at-least-0}))

(defmethod -init :sgd [_ _] nil)

(defmethod -step :sgd [{:keys [lr]} _ store grads]
  [(update-given #(+ %1 (* lr %2)) store grads) nil])

(defn gradient-descent
  "Gradient ascent with a decaying step: step-size-init · (beta + 1) /
   (beta + t) at step t (Gen.jl's GradientDescent). Options:
   :step-size-init, :step-size-beta."
  [{:keys [step-size-init step-size-beta]}]
  (checked "gradient-descent" {:step-size-init step-size-init :step-size-beta step-size-beta}
           {:step-size-init at-least-0 :step-size-beta at-least-0})
  {:kind :gradient-descent :init step-size-init :beta step-size-beta})

(defmethod -init :gradient-descent [_ _] {:t 0})

(defmethod -step :gradient-descent [{:keys [init beta]} {:keys [t]} store grads]
  (let [t (inc t)
        size (/ (* init (+ beta 1.0)) (+ beta t))]
    [(update-given #(+ %1 (* size %2)) store grads) {:t t}]))

(defn momentum
  "Gradient ascent with momentum (Polyak's heavy ball), as PyTorch's SGD
   with momentum: v' = μ·v + g, then θ + lr·v'; with :nesterov? true,
   Nesterov's, θ + lr·(g + μ·v'). Options: :lr, :momentum μ (0.9),
   :nesterov? (false). The state: {:v}, of the store's shape."
  [{:keys [lr momentum nesterov?] :or {momentum 0.9 nesterov? false}}]
  (checked "momentum" {:kind :momentum :lr lr :momentum momentum :nesterov? (boolean nesterov?)}
           {:lr at-least-0 :momentum below-1}))

(defmethod -init :momentum [_ store] {:v (zeros-like store)})

(defmethod -step :momentum [{:keys [lr momentum nesterov?]} {:keys [v]} store grads]
  (let [v (update-given #(+ (* momentum %1) %2) v grads)]
    [(update-given (if nesterov?
                     (fn [theta g vi] (+ theta (* lr (+ g (* momentum vi)))))
                     (fn [theta _ vi] (+ theta (* lr vi))))
                   store grads v)
     {:v v}]))

(defn rmsprop
  "RMSProp (Hinton's), ascending, as PyTorch's: v' = ρ·v + (1 - ρ)·g², then
   θ + lr·g / (√v' + eps). Options: :lr (0.01), :decay ρ (0.99, PyTorch's
   alpha), :eps (1e-8). The state: {:v}, of the store's shape."
  [{:keys [lr decay eps] :or {lr 0.01 decay 0.99 eps 1e-8}}]
  (checked "rmsprop" {:kind :rmsprop :lr lr :decay decay :eps eps} {:lr at-least-0 :decay below-1 :eps above-0}))

(defmethod -init :rmsprop [_ store] {:v (zeros-like store)})

(defmethod -step :rmsprop [{:keys [lr decay eps]} {:keys [v]} store grads]
  (let [v (update-given #(+ (* decay %1) (* (- 1.0 decay) (* %2 %2))) v grads)]
    [(update-given (fn [theta g vi] (+ theta (* lr (/ g (+ (array/sqrt vi) eps))))) store grads v)
     {:v v}]))

(defn adagrad
  "AdaGrad (Duchi, Hazan and Singer 2011), ascending, as PyTorch's: s' = s +
   g², then θ + lr·g / (√s' + eps). Options: :lr (0.01), :eps (1e-10). The
   state: {:sum}, of the store's shape."
  [{:keys [lr eps] :or {lr 0.01 eps 1e-10}}]
  (checked "adagrad" {:kind :adagrad :lr lr :eps eps} {:lr at-least-0 :eps above-0}))

(defmethod -init :adagrad [_ store] {:sum (zeros-like store)})

(defmethod -step :adagrad [{:keys [lr eps]} state store grads]
  (let [s (update-given #(+ %1 (* %2 %2)) (:sum state) grads)]
    [(update-given (fn [theta g si] (+ theta (* lr (/ g (+ (array/sqrt si) eps))))) store grads s)
     {:sum s}]))

(def ^:private adam-ranges {:lr at-least-0 :beta1 below-1 :beta2 below-1 :eps above-0})

(defn adam
  "Adam (Kingma and Ba 2015), ascending. Options: :lr (0.001), :beta1 (0.9),
   :beta2 (0.999), :eps (1e-8). The state: {:t :m :v}, the step count and
   the moments, of the store's shape."
  [{:keys [lr beta1 beta2 eps] :or {lr 0.001 beta1 0.9 beta2 0.999 eps 1e-8}}]
  (checked "adam" {:kind :adam :lr lr :beta1 beta1 :beta2 beta2 :eps eps} adam-ranges))

(defn adamw
  "AdamW (Loshchilov and Hutter 2019): Adam with decoupled weight decay, as
   PyTorch's, ascending: θ·(1 - lr·λ) + lr·m̂ / (√v̂ + eps), so the decay
   pulls each parameter toward 0. Options: adam's and :weight-decay λ
   (0.01). The state: adam's."
  [{:keys [lr beta1 beta2 eps weight-decay] :or {lr 0.001 beta1 0.9 beta2 0.999 eps 1e-8 weight-decay 0.01}}]
  (checked "adamw" {:kind :adamw :lr lr :beta1 beta1 :beta2 beta2 :eps eps :weight-decay weight-decay}
           (assoc adam-ranges :weight-decay at-least-0)))

(defn- adam-init [_ store] {:t 0 :m (zeros-like store) :v (zeros-like store)})

(defn- adam-step
  "Adam's step, with the decoupled decay λ (0 for Adam's)."
  [{:keys [lr beta1 beta2 eps]} decay state store grads]
  (let [t (inc (:t state))
        m (update-given #(+ (* beta1 %1) (* (- 1.0 beta1) %2)) (:m state) grads)
        v (update-given #(+ (* beta2 %1) (* (- 1.0 beta2) (* %2 %2))) (:v state) grads)
        c1 (- 1.0 (Math/pow beta1 t))
        c2 (- 1.0 (Math/pow beta2 t))
        shrink (- 1.0 (* lr decay))]
    [(update-given (fn [theta _ mi vi]
                     (+ (if (zero? decay) theta (* shrink theta))
                        (* lr (/ (/ mi c1) (+ (array/sqrt (/ vi c2)) eps)))))
                   store grads m v)
     {:t t :m m :v v}]))

(defmethod -init :adam [optimizer store] (adam-init optimizer store))

(defmethod -step :adam [optimizer state store grads] (adam-step optimizer 0.0 state store grads))

(defmethod -init :adamw [optimizer store] (adam-init optimizer store))

(defmethod -step :adamw [optimizer state store grads]
  (adam-step optimizer (:weight-decay optimizer) state store grads))

(defn- global-norm
  "The Euclidean norm of every leaf of the gradient tree grads together."
  [grads]
  (array/sqrt (reduce + (map #(array/sum (* % %)) (array/tree-leaves grads)))))

(defn clip-norm
  "optimizer, with each step's gradients scaled to a global norm of at most
   :max-norm first: g·min(1, max-norm / ‖g‖), ‖g‖ the Euclidean norm of
   every gradient together (optax's clip_by_global_norm, PyTorch's
   clip_grad_norm_). Its state is optimizer's."
  [optimizer {:keys [max-norm]}]
  (checked "clip-norm" {:kind :clip-norm :max-norm max-norm :optimizer optimizer} {:max-norm above-0}))

(defmethod -init :clip-norm [{:keys [optimizer]} store] (-init optimizer store))

(defmethod -step :clip-norm [{:keys [max-norm optimizer]} state store grads]
  (let [scale (/ max-norm (array/maximum (global-norm grads) max-norm))]
    (-step optimizer state store (array/tree-map #(* scale %) grads))))

;; ---------------------------------------------------------------------------
;; Training
;; ---------------------------------------------------------------------------

(defn- check-sizes [{:keys [epochs epoch-size batches-per-epoch batch-size evaluation-size]}]
  (doseq [[k x] [[:epochs epochs] [:epoch-size epoch-size] [:batches-per-epoch batches-per-epoch]
                 [:batch-size batch-size] [:evaluation-size evaluation-size]]]
    (when-not (if (= k :epochs) (nat-int? x) (pos-int? x))
      (throw (ex-info (str "train's " k " is a " (if (= k :epochs) "non-negative" "positive") " integer, not " (pr-str x))
                      {:option k :value x}))))
  (when (clojure.core/> batch-size epoch-size)
    (throw (ex-info (str "train draws each minibatch of " batch-size " examples from the epoch's " epoch-size
                         " without replacement: :batch-size is at most :epoch-size")
                    {:batch-size batch-size :epoch-size epoch-size}))))

(defn- subset
  "b distinct indices below n, uniform (the first b of a uniform random
   permutation, Gen.jl's randperm(n)[1:b])."
  [key n b]
  (->> (array/->clj (rng/uniform key [n]))
       (map-indexed vector)
       (sort-by peek)
       (map first)
       (clojure.core/take b)))

(defn- mean-weight
  "The mean of the generate weights, a host double."
  [runs]
  (/ (reduce + (map #(double (array/item (:weight %))) runs)) (count runs)))

(defn train
  "Fit the trainable parameters of gf by maximizing the expected
   conditional log density of generated data, as Gen.jl's train!. Each of
   :epochs epochs generates :epoch-size examples [args constraints] once,
   from (data-fn key), then takes :batches-per-epoch steps of the
   optimizer from store, each along the mean of the param-gradients
   (through gradients-fn, gen.gradients/param-gradients) of :batch-size
   examples drawn from the epoch's without replacement. After the epoch it
   generates :evaluation-size fresh examples (default :epoch-size), held
   out from training, and the epoch's entry in the history is their mean
   generate weight under the new store: the log density of the
   constraints when they hold every choice, otherwise a log importance
   weight, whose mean is a lower bound on it (Gen.jl's objective too).

   Returns {:store :state :history}, :state the optimizer's, which :state
   takes to resume (init's when absent). :callback, given, is called after
   each step with {:epoch :minibatch :objective :store}: the epoch and the
   minibatch within it (from 0), the minibatch's mean generate weight
   before the step and the store after it; its value is ignored.
   Defaults: 1 epoch of 1 minibatch of 1 example (Gen.jl's). Options also
   take :key."
  [gf store optimizer data-fn gradients-fn
   {:keys [epochs epoch-size batches-per-epoch batch-size evaluation-size callback state key]
    :or {epochs 1 epoch-size 1 batches-per-epoch 1 batch-size 1}}]
  (let [evaluation-size (or evaluation-size epoch-size)]
    (check-sizes {:epochs epochs :epoch-size epoch-size :batches-per-epoch batches-per-epoch
                  :batch-size batch-size :evaluation-size evaluation-size})
    (loop [epoch 0 store store state (if (some? state) state (init optimizer store)) history []
           key (or key (gfi/next-key))]
      (if (< epoch epochs)
        (let [[k-data k-batches k-evaluation k-next] (rng/split key 4)
              examples (mapv data-fn (rng/split k-data epoch-size))
              minibatch (fn [[store state] [b k]]
                          (let [[k-pick k-generate] (rng/split k)
                                model (dyn/with-params gf store)
                                runs (mapv (fn [i k] (let [[args constraints] (examples i)]
                                                       (gfi/generate model args constraints k)))
                                           (subset k-pick epoch-size batch-size)
                                           (rng/split k-generate batch-size))
                                [store' state'] (step optimizer state store (grad/mean (map (comp gradients-fn :trace) runs)))]
                            (when callback
                              (callback {:epoch epoch :minibatch b :objective (mean-weight runs) :store store'}))
                            [store' state']))
              [store state] (reduce minibatch [store state] (map-indexed vector (rng/split k-batches batches-per-epoch)))
              model (dyn/with-params gf store)
              held-out (mapv (fn [k] (let [[k-data k-generate] (rng/split k)
                                           [args constraints] (data-fn k-data)]
                                       (gfi/generate model args constraints k-generate)))
                             (rng/split k-evaluation evaluation-size))]
          (recur (inc epoch) store state (conj history (mean-weight held-out)) k-next))
        {:store store :state state :history history}))))

;; ---------------------------------------------------------------------------
;; Wake-sleep's sleep phase
;; ---------------------------------------------------------------------------

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
