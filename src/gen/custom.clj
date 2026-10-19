(ns gen.custom
  "Gen.jl's custom deterministic generative functions (CustomDetermGF,
   CustomUpdateGF and CustomGradientGF): no random choices, a return value
   computed by your function, an update you write to reuse earlier work,
   and a gradient you write for code differentiation cannot see through.

     (deterministic {:apply-with-state  (fn [args] [retval state])
                     :update-with-state (fn [state args argdiffs] [state retval retdiff])})

   apply-with-state computes the return value from the arguments and keeps
   whatever state helps later updates; update-with-state gets that state,
   the new arguments and their argdiffs (gen.diff), and returns the new
   state, the new return value and its retdiff. Without update-with-state,
   an update recomputes with apply-with-state. An update whose argdiffs are
   all no-change keeps the trace as it is.

   As a generative function it makes no choices: its score and every weight
   are 0, and constraints or choices given to it are a loud error. It
   composes as any other: splice it in a model, map it, unfold it.

   Gradients (choice-gradients, HMC) pass through it as through the code of
   its function, or as you give them (Gen.jl's CustomGradientGF):

     (deterministic {:apply    (fn [args] retval)
                     :gradient (fn [args retval retgrad] [grad-of-arg …])})

   or, with state, :gradient-with-state (fn [state args retgrad] [grads]).
   The gradient has one entry per argument, nil for an argument without one:
   an array for an array or number, a tree of the argument's shape for a
   tree; it passes through a return value that is a float array or number. With a gradient, the
   function runs on the arguments' values (array/stop-gradient), so it may
   use host arithmetic, a solver, anything (array/custom-vjp)."
  (:require [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defn- no-choices! [op choices]
  (when-not (cm/empty? choices)
    (throw (ex-info (str (name op) ": a deterministic generative function makes no choices, but was given "
                         (pr-str (vec (cm/addresses choices))))
                    {:op op :addresses (vec (cm/addresses choices))}))))

(defn- detached [arg] (array/tree-map #(if (array/array? %) (array/stop-gradient %) %) arg))

(defn- leaves-like
  "The leaves of g in the order of t's leaves, g a tree of t's shape (or nil,
   for none)."
  [t g]
  (cond (map? t) (mapcat (fn [[k v]] (leaves-like v (get g k))) t)
        (or (vector? t) (seq? t)) (mapcat leaves-like t (if (nil? g) (repeat nil) g))
        :else [g]))

(defn- with-gradient
  "retval, which apply-with-state computed from plain (the detached args),
   with its gradient with respect to the arguments' array leaves given by
   (gradient state plain retval retgrad): one entry per argument, an array
   for an array argument, a tree of its shape for a tree (nil for none)."
  [retval state args plain gradient]
  (let [leaves (mapv #(vec (leaves-like % %)) args)
        positions (vec (for [i (range (count args)) [j leaf] (map-indexed vector (leaves i)) :when (array/array? leaf)]
                         [i j]))]
    (when-not (array/array? retval)
      (throw (ex-info (str "A deterministic generative function with a gradient returns an array or a number, not "
                           (pr-str retval))
                      {:retval retval})))
    (if (empty? positions)
      retval
      (apply (array/custom-vjp (fn [& _] retval)
                               (fn [_ _ g] (let [grads (vec (gradient state plain retval g))]
                                             (when-not (= (count grads) (count args))
                                               (throw (ex-info (str "The gradient returned " (count grads) " entries for "
                                                                    (count args) " arguments (nil for none)")
                                                               {:gradients grads})))
                                             (let [by-leaf (mapv #(vec (leaves-like (args %) (grads %))) (range (count args)))]
                                               (mapv (fn [[i j]] ((by-leaf i) j)) positions)))))
             (mapv (fn [[i j]] ((leaves i) j)) positions)))))

(defn- run [gf args]
  (let [args (vec args)
        gradient (:gradient gf)
        plain (if gradient (mapv detached args) args)
        [retval state] ((:apply-with-state gf) plain)
        retval (if gradient (with-gradient retval state args plain gradient) retval)]
    (trace/make-trace {:gen-fn gf :args args :choices cm/EMPTY :retval retval :score 0.0 ::state state})))

(defn- updated
  "tr moved to args (argdiffs), with update-with-state, or kept when nothing
   changed: [trace retdiff]."
  [gf tr args argdiffs]
  (let [args (vec args)]
    (if (and (every? diff/no-change? argdiffs) (= (count args) (count (:args tr))))
      [tr diff/no-change]
      (let [[state retval retdiff] ((:update-with-state gf) (::state tr) args (vec argdiffs))]
        [(assoc tr :args args :retval retval ::state state) retdiff]))))

(defrecord DeterministicGF [apply-with-state update-with-state gradient]
  gfi/IGenerativeFunction
  (-simulate [this args _key] (run this args))
  (-generate [this args constraints _key]
    (no-choices! :generate constraints)
    {:trace (run this args) :weight 0.0})
  (-update [this tr args argdiffs constraints _key]
    (no-choices! :update constraints)
    (let [[t retdiff] (updated this tr args argdiffs)]
      {:trace t :weight 0.0 :retdiff retdiff :discard cm/EMPTY}))
  (-regenerate [this tr args argdiffs _selection _key]
    (let [[t retdiff] (updated this tr args argdiffs)]
      {:trace t :weight 0.0 :retdiff retdiff}))
  (-assess [this args choices]
    (no-choices! :assess choices)
    {:weight 0.0 :retval (:retval (run this args))})
  (-propose [this args _key] {:choices cm/EMPTY :weight 0.0 :retval (:retval (run this args))})
  (-project [_ _tr _selection] 0.0)

  clojure.lang.IFn
  (invoke [this] (:retval (run this [])))
  (invoke [this a] (:retval (run this [a])))
  (invoke [this a b] (:retval (run this [a b])))
  (invoke [this a b c] (:retval (run this [a b c])))
  (invoke [this a b c d] (:retval (run this [a b c d])))
  (invoke [this a b c d e] (:retval (run this [a b c d e])))
  (invoke [this a b c d e f] (:retval (run this [a b c d e f])))
  (applyTo [this args] (:retval (run this (vec args)))))

(defn deterministic
  "A deterministic generative function (see the namespace doc), from
   :apply-with-state, (fn [args] [retval state]), or :apply, (fn [args]
   retval); optionally :update-with-state, (fn [state args argdiffs] [state
   retval retdiff]); and optionally a gradient: :gradient, (fn [args retval
   retgrad] grads), or :gradient-with-state, (fn [state args retgrad]
   grads)."
  [{:keys [apply-with-state update-with-state gradient gradient-with-state] :as spec}]
  (let [f (:apply spec)
        _ (when-not (= 1 (count (filter ifn? [apply-with-state f])))
            (throw (ex-info "deterministic needs :apply-with-state, (fn [args] [retval state]), or :apply, (fn [args] retval)"
                            {:keys (vec (keys spec))})))
        _ (when (and update-with-state (not (ifn? update-with-state)))
            (throw (ex-info "deterministic: :update-with-state must be (fn [state args argdiffs] [state retval retdiff])" {})))
        _ (when (and gradient gradient-with-state)
            (throw (ex-info "deterministic takes :gradient or :gradient-with-state, not both" {})))
        _ (doseq [[k g] {:gradient gradient :gradient-with-state gradient-with-state}]
            (when (and g (not (ifn? g)))
              (throw (ex-info (str "deterministic: " k " must be a function") {k g}))))
        _ (when (and gradient-with-state (not apply-with-state))
            (throw (ex-info "deterministic: :gradient-with-state needs :apply-with-state (with :apply, use :gradient)" {})))
        apply-with-state (or apply-with-state (fn [args] [(f args) nil]))]
    (->DeterministicGF apply-with-state
                       (or update-with-state
                           (fn [_ args _] (let [[retval state] (apply-with-state args)] [state retval diff/unknown-change])))
                       (cond gradient (fn [_ args retval g] (gradient args retval g))
                             gradient-with-state (fn [state args _ g] (gradient-with-state state args g))))))

(defmethod print-method DeterministicGF [_ ^java.io.Writer w]
  (.write w "#gen/deterministic"))
