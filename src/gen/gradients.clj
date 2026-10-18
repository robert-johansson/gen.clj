(ns gen.gradients
  "Gradients of a trace's log density, as Gen.jl's choice_gradients and
   accumulate_param_gradients!, for any generative function: each is the
   gradient of assess's weight through reverse-mode autodiff
   (gen.array/value-and-grad), so selections may reach nested addresses.

     (choice-gradients trace selection)          → {:values :gradients :arg-grads}
     (choice-gradients trace selection retgrad {:argnums [0]})
     (param-gradients trace)                     → {name gradient}
     (param-gradients trace retgrad)
     (log-abs-det-jacobian f x)                  → log |det ∂f/∂x|

   The objective is log p(τ; x) plus ⟨retgrad, retval⟩ when retgrad (a
   cotangent of the return value, of the same tree shape) is given, as in
   Gen.jl. Selected choices must be continuous."
  (:refer-clojure :exclude [+ *])
  (:require [gen.array :as a :refer [+ *]]
            [gen.array.dtype :as dt]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(defn- float-tree?
  "Is every leaf of x a float array or floating-point number (and there is
   at least one)?"
  [x]
  (let [leaves (a/tree-leaves x)]
    (and (seq leaves) (every? #(and (a/array? %) (= :float (dt/kind (a/dtype %)))) leaves))))

(defn- objective
  "assess's weight of gf on args and choices, plus ⟨retgrad, retval⟩."
  [gf args choices retgrad]
  (let [{w :weight rv :retval} (gfi/assess gf args choices)]
    (if (nil? retgrad)
      w
      (reduce + w (map (fn [g v] (a/sum (* g v))) (a/tree-leaves retgrad) (a/tree-leaves rv))))))

(defn choice-gradients
  "The gradient of the trace's log density (plus ⟨retgrad, retval⟩) with
   respect to the selected choices, and to the arguments at :argnums (none
   by default, as Gen.jl differentiates only arguments annotated (grad)):
   {:values the selected choices, :gradients a choice map of the same shape,
   :arg-grads one entry per argument, nil where not differentiated}.
   Differentiated arguments must be trees of floats, and the model must use
   them through gen.array (host math on them cannot be differentiated)."
  ([trace selection] (choice-gradients trace selection nil nil))
  ([trace selection retgrad] (choice-gradients trace selection retgrad nil))
  ([trace selection retgrad {:keys [argnums]}]
   (let [{gf :gen-fn args :args choices :choices} trace
         selected (sel/get-selected choices selection)
         others (sel/get-selected choices (sel/complement selection))]
     (doseq [p (cm/addresses selected)]
       (when-not (float-tree? (cm/get-choice selected p))
         (throw (ex-info (str "choice-gradients: the choice at " p " is not continuous")
                         {:path p :value (cm/get-choice selected p)}))))
     (doseq [i argnums]
       (when-not (float-tree? (nth args i))
         (throw (ex-info (str "choice-gradients: argument " i " is not a tree of floats") {:argnum i}))))
     (let [arg-positions (vec argnums)
           f (fn [values & dargs]
               (let [args' (reduce (fn [as [i v]] (assoc as i v)) (vec args) (map vector arg-positions dargs))]
                 (objective gf args' (cm/merge others (cm/from-map values)) retgrad)))
           [_ [g & arg-gs]] (apply (a/value-and-grad f {:argnums (vec (range (inc (count arg-positions))))})
                                   (cm/to-map selected) (map #(nth args %) arg-positions))]
       {:values selected
        :gradients (if (cm/empty? selected) cm/EMPTY (cm/from-map g))
        :arg-grads (reduce (fn [gs [i ag]] (assoc gs i ag)) (vec (repeat (count args) nil))
                           (map vector arg-positions arg-gs))}))))

(defn param-gradients
  "The gradient of the trace's log density (plus ⟨retgrad, retval⟩) with
   respect to the trainable parameters of its generative function's store
   (gen.dynamic/with-params): a map of name to gradient. A functional
   accumulate_param_gradients!: sum the maps over traces to accumulate."
  ([trace] (param-gradients trace nil))
  ([trace retgrad]
   (let [{gf :gen-fn args :args choices :choices} trace
         store (:params gf)]
     (when (empty? store)
       (throw (ex-info "param-gradients: the trace's generative function has no parameter store" {})))
     ((a/grad (fn [ps] (objective (dyn/with-params gf ps) args choices retgrad))) store))))

(defn log-abs-det-jacobian
  "log |det J| of f: a 1-d float array → a 1-d array of the same size, at x,
   by one reverse pass per output (for the volume change of an involution,
   thesis Eq 3.15)."
  [f x]
  (let [n (a/size x)
        rows (mapv (fn [i] ((a/grad (fn [v] (a/take (f v) i))) x)) (range n))]
    (second (la/slogdet (a/stack rows)))))
