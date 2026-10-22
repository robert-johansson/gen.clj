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
   Gen.jl. The trees pair by structure, maps by key and vectors by
   position, with nil for no cotangent (Gen.jl's nothing); another
   structure is a loud error. Selected choices must be continuous."
  (:refer-clojure :exclude [+ *])
  (:require [gen.array :as array :refer [+ *]]
            [gen.array.dtype :as dt]
            [gen.array.linalg :as la]
            [gen.choicemap :as cm]
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

(defn- node-kind [x]
  (cond (map? x) "a map" (sequential? x) "a sequence" :else "a leaf"))

(defn- inner-terms
  "The terms of ⟨retgrad, retval⟩ at path: the two trees walked together,
   maps by key and vectors or seqs by position, a nil cotangent adding
   nothing. A key in one tree only, a length mismatch or a node against
   another kind is a loud error; a leaf of retgrad broadcasts against an
   array leaf."
  [path g v]
  (let [mismatch (fn [what data]
                   (throw (ex-info (str "retgrad: " what " at " path ", not the return value's structure")
                                   (merge {:path path :retgrad g :retval v} data))))]
    (cond (nil? g) []
          (not= (node-kind g) (node-kind v)) (mismatch (str (node-kind g) " against " (node-kind v)) {})
          (map? g) (let [gk (set (keys g)) vk (set (keys v))]
                     (when (not= gk vk)
                       (mismatch "other keys" {:only-retgrad (vec (remove vk gk)) :only-retval (vec (remove gk vk))}))
                     (mapcat (fn [[k gv]] (inner-terms (conj path k) gv (get v k))) g))
          (sequential? g) (do (when (not= (count g) (count v))
                                (mismatch (str (count g) " elements against " (count v)) {}))
                              (mapcat #(inner-terms (conj path %1) %2 %3) (range) g v))
          :else [(array/sum (* g v))])))

(defn- objective
  "assess's weight of gf on args and choices, plus ⟨retgrad, retval⟩. The
   key is fixed, so the objective is a function: of the choices alone for an
   exact density, and at one draw of the encapsulated randomness otherwise."
  [gf args choices retgrad]
  (let [{w :weight rv :retval} (gfi/with-exact-density "A gradient" (gfi/assess gf args choices (rng/key 0)))]
    (reduce + w (inner-terms [] retgrad rv))))

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
   respect to the trainable parameters its generative function holds
   (gen.gfi/stores): a dynamic or deterministic function's own store
   (with-params), or a combinator's callees', accumulated over its
   elements; a tree of the stores' shape, a map of name to gradient for one
   store. A functional accumulate_param_gradients!: sum the trees over
   traces to accumulate, or weigh them (weighted-sum) for its scale factor,
   and step the stores with gen.learning (gfi/with-stores gives them back
   to a combinator)."
  ([trace] (param-gradients trace nil))
  ([trace retgrad]
   (let [{gf :gen-fn args :args choices :choices} trace
         stores (gfi/stores gf)]
     (when (empty? stores)
       (throw (ex-info "param-gradients: the trace's generative function holds no parameter store" {})))
     ((array/grad (fn [ps] (objective (gfi/with-stores gf ps) args choices retgrad))) stores))))

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
