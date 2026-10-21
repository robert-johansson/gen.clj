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
  (:require [gen.array :as array :refer [+ *]]
            [gen.array.dtype :as dt]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
            [gen.custom :as custom]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(defn- float-tree?
  "Is every leaf of x a float array or floating-point number (and there is
   at least one)?"
  [x]
  (let [leaves (array/tree-leaves x)]
    (and (seq leaves) (every? #(and (array/array? %) (= :float (dt/kind (array/dtype %)))) leaves))))

(defn- objective
  "assess's weight of gf on args and choices, plus ⟨retgrad, retval⟩. The
   key is fixed, so the objective is a function: of the choices alone for an
   exact density, and at one draw of the encapsulated randomness otherwise."
  [gf args choices retgrad]
  (let [{w :weight rv :retval} (gfi/with-exact-density "A gradient" (gfi/assess gf args choices (rng/key 0)))]
    (if (nil? retgrad)
      w
      (reduce + w (map (fn [g v] (array/sum (* g v))) (array/tree-leaves retgrad) (array/tree-leaves rv))))))

(defn continuous-selection
  "The trace's choices at selection, which must be continuous (loudly): the
   values choice-gradients differentiates."
  [trace selection]
  (let [selected (sel/get-selected (:choices trace) selection)]
    (doseq [p (cm/addresses selected)]
      (when-not (float-tree? (cm/get-choice selected p))
        (throw (ex-info (str "choice-gradients: the choice at " p " is not continuous")
                        {:path p :value (cm/get-choice selected p)}))))
    selected))

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
         selected (continuous-selection trace selection)
         others (sel/get-selected choices (sel/complement selection))]
     (doseq [i argnums]
       (when-not (float-tree? (nth args i))
         (throw (ex-info (str "choice-gradients: argument " i " is not a tree of floats") {:argnum i}))))
     (let [arg-positions (vec argnums)
           f (fn [values & dargs]
               (let [args' (reduce (fn [as [i v]] (assoc as i v)) (vec args) (map vector arg-positions dargs))]
                 (objective gf args' (cm/merge others (cm/from-map values)) retgrad)))
           [_ [g & arg-gs]] (apply (array/value-and-grad f {:argnums (vec (range (inc (count arg-positions))))})
                                   (cm/to-map selected) (map #(nth args %) arg-positions))]
       {:values selected
        :gradients (if (cm/empty? selected) cm/EMPTY (cm/from-map g))
        :arg-grads (reduce (fn [gs [i ag]] (assoc gs i ag)) (vec (repeat (count args) nil))
                           (map vector arg-positions arg-gs))}))))

(defn param-gradients
  "The gradient of the trace's log density (plus ⟨retgrad, retval⟩) with
   respect to the trainable parameters of its generative function's store
   (gen.dynamic/with-params, or gen.custom/with-params for a deterministic
   function's own): a map of name to gradient. A functional
   accumulate_param_gradients!: sum the maps over traces to accumulate, or
   weigh them (weighted-sum) for its scale factor."
  ([trace] (param-gradients trace nil))
  ([trace retgrad]
   (let [{gf :gen-fn args :args choices :choices} trace
         store (:params gf)
         with-store (if (custom/deterministic? gf) custom/with-params dyn/with-params)]
     (when (empty? store)
       (throw (ex-info "param-gradients: the trace's generative function has no parameter store" {})))
     ((array/grad (fn [ps] (objective (with-store gf ps) args choices retgrad))) store))))

(defn- tree-add
  "x + y over gradient trees, leaf by leaf; a name in one tree only keeps
   its gradient (a run that did not read a parameter has none for it)."
  [x y]
  (cond (map? x) (merge-with tree-add x y)
        (vector? x) (mapv tree-add x y)
        :else (+ x y)))

(defn weighted-sum
  "Σ_i c_i·g_i over gradient trees gs (maps of names to arrays, numbers or
   nested trees, as param-gradients gives)."
  [cs gs]
  (reduce tree-add (map (fn [c g] (array/tree-map #(* c %) g)) cs gs)))

(defn mean
  "The mean of gradient trees gs."
  [gs]
  (let [n (count gs)]
    (weighted-sum (repeat n (/ 1.0 n)) gs)))

(defn log-abs-det-jacobian
  "log |det J| of f: a 1-d float array → a 1-d array of the same size, at x,
   by one reverse pass per output (for the volume change of an involution,
   thesis Eq 3.15)."
  [f x]
  (let [n (array/size x)
        rows (mapv (fn [i] ((array/grad (fn [v] (array/take (f v) i))) x)) (range n))]
    (second (la/slogdet (array/stack rows)))))
