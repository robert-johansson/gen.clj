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
   the same shape."
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as a :refer [+ - * /]]))

(set! *warn-on-reflection* true)

(defn- tree-map2 [f x y]
  (cond (map? x) (reduce-kv (fn [m k v] (assoc m k (tree-map2 f v (get y k)))) x x)
        (vector? x) (mapv #(tree-map2 f %1 %2) x y)
        :else (f x y)))

(defn sgd
  "Fixed-step gradient ascent: θ + lr·g (Gen.jl's FixedStepGradientDescent)."
  [{:keys [lr]}]
  {:kind :sgd :lr lr})

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
    :adam {:t 0 :m (a/tree-map a/zeros-like store) :v (a/tree-map a/zeros-like store)}))

(defn step
  "[store' state']: one ascent step from store along the gradients."
  [{:keys [kind lr beta1 beta2 eps]} state store grads]
  (case kind
    :sgd [(tree-map2 #(+ %1 (* lr %2)) store grads) nil]
    :adam (let [t (inc (:t state))
                m (tree-map2 #(+ (* beta1 %1) (* (- 1.0 beta1) %2)) (:m state) grads)
                v (tree-map2 #(+ (* beta2 %1) (* (- 1.0 beta2) (* %2 %2))) (:v state) grads)
                c1 (- 1.0 (Math/pow beta1 t))
                c2 (- 1.0 (Math/pow beta2 t))
                delta (tree-map2 (fn [mi vi] (/ (/ mi c1) (+ (a/sqrt (/ vi c2)) eps))) m v)]
            [(tree-map2 #(+ %1 (* lr %2)) store delta) {:t t :m m :v v}])))
