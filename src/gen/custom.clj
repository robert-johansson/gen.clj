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
   use host arithmetic, a solver, anything (array/custom-vjp).

   A generative function whose density is estimated (thesis §4.5.3), such
   as a likelihood that a particle filter or importance sampler estimates
   for pseudo-marginal MCMC:

     (estimated {:addresses [:y]
                 :estimate (fn [args choices key] log-estimate)
                 :simulate (fn [args key] {:choices … :log-estimate …})})

   estimate draws the estimator's randomness ω from key and returns log ξ,
   an unbiased estimate of the density of choices (all of the function's
   choices): E_ω[ξ] = p(choices; args). simulate, optional, samples the
   choices and ω jointly, the choices from p and ω given them from the
   estimator's conditional (for an importance sampler, one sample is the
   choices' own latent, the others from the proposal), and returns the
   estimate at both. The trace keeps the key as ω.

   generate with constraints, all of the choices (others are a loud error),
   weighs log ξ; without constraints it is simulate, weight 0. update keeps the
   trace when nothing changed (no-change argdiffs, no constraints), and
   otherwise estimates afresh: weight log ξ' - log ξ, the pseudo-marginal
   ratio. regenerate redraws ω the same way; with every choice selected it
   draws the choices and ω afresh from simulate (weight 0, as generate's),
   and it cannot resample some of the choices only (there is no proposal
   density to weigh them by), a loud error. project takes all of the
   choices or none."
  (:refer-clojure :exclude [-])
  (:require [gen.array :as array :refer [-]]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defn- no-choices! [op choices]
  (when-not (cm/empty? choices)
    (throw (gfi/unvisited op (vec (cm/addresses choices)) "a deterministic generative function makes no choices"))))

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
  "tr moved to args (argdiffs) with update-with-state, or kept when no
   argument changed: [trace retdiff]. (A trace gf did not make, or a call
   whose store changed, is gfi/revise's: it runs again.)"
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
  (-assess [this args choices _key]
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

;; ---------------------------------------------------------------------------
;; Estimated densities (encapsulated randomness, thesis §4.5)
;; ---------------------------------------------------------------------------

(defn- estimated-trace [gf args choices log-estimate omega]
  (trace/make-trace {:gen-fn gf :args (vec args) :choices choices :score log-estimate
                     :retval ((:retval gf) args choices) ::trace/omega omega}))

(defn- check-unbatched!
  "An estimate in a batched run raises the fallback: a particle's ω must
   reproduce its own estimate, as for untraced draws."
  []
  (when (batch/batch-backend? array/*backend*)
    (throw (array/batched-branch "An estimated density in a batched run" {}))))

(defn- check-addresses!
  "choices are all of gf's choices (an estimate is of all of them), a loud
   error otherwise: an unvisited one for choices at other addresses."
  [gf op choices]
  (let [given (set (cm/addresses choices))]
    (when-let [others (seq (remove (:addresses gf) given))]
      (throw (gfi/unvisited op (vec (sort-by str others)) (str "the estimator's choices are " (vec (sort-by str (:addresses gf)))))))
    (when-not (= given (:addresses gf))
      (throw (ex-info (str "estimated: " (name op) " needs every choice of the estimator, " (vec (sort-by str (:addresses gf)))
                           ", given " (vec (sort-by str given)))
                      {:op op :addresses (:addresses gf) :given given})))))

(defn- estimator
  "(f & args), the user's :estimate or :simulate: a constraint left
   unvisited in there is its own error, not one of the estimated
   function's choices (which gfi/revise could drop)."
  [f & args]
  (try (apply f args)
       (catch clojure.lang.ExceptionInfo e
         (throw (if (:unvisited (ex-data e)) (ex-info (ex-message e) (dissoc (ex-data e) :unvisited) e) e)))))

(defn- estimate-at
  "The trace of gf on args with choices, ω drawn from key."
  [gf op args choices key]
  (gfi/check-exact-density! "an estimated density")
  (check-unbatched!)
  (check-addresses! gf op choices)
  (estimated-trace gf args choices (estimator (:estimate gf) (vec args) choices key) key))

(defn- simulated [gf args key]
  (gfi/check-exact-density! "an estimated density")
  (check-unbatched!)
  (if-let [simulate (:simulate gf)]
    (let [{:keys [choices log-estimate] :as r} (estimator simulate (vec args) key)]
      (when-not (and (cm/choicemap? choices) (some? log-estimate))
        (throw (ex-info "estimated: :simulate returns {:choices (a choice map) :log-estimate}"
                        {:returned (keys r)})))
      (check-addresses! gf :simulate choices)
      (estimated-trace gf args choices log-estimate key))
    (throw (ex-info (str "estimated: simulate (and generate without constraints) needs :simulate, a joint sampler of"
                         " the choices and the estimator's randomness (thesis §4.5.1); generate with the choices,"
                         " update and assess need only :estimate")
                    {}))))

(defn- unchanged? [argdiffs] (every? diff/no-change? argdiffs))

(defrecord EstimatedGF [estimate simulate retval addresses]
  gfi/IGenerativeFunction
  (-simulate [this args key] (simulated this args key))
  (-generate [this args constraints key]
    (if (cm/empty? constraints)
      {:trace (simulated this args key) :weight 0.0}
      (let [t (estimate-at this :generate args constraints key)] {:trace t :weight (:score t)})))
  (-update [this tr args argdiffs constraints key]
    (if (and (unchanged? argdiffs) (cm/empty? constraints))
      {:trace tr :weight 0.0 :retdiff diff/no-change :discard cm/EMPTY}
      (let [t (estimate-at this :update args (cm/override (:choices tr) constraints) key)]
        {:trace t :weight (- (:score t) (:score tr)) :retdiff diff/unknown-change
         :discard (sel/get-selected (:choices tr) (sel/from-choicemap constraints))})))
  (-regenerate [this tr args argdiffs selection key]
    (let [chosen (sel/get-selected (:choices tr) selection)]
      (cond
        ;; every choice: the joint sampler is the internal proposal, as in generate
        (= chosen (:choices tr)) {:trace (simulated this args key) :weight 0.0 :retdiff diff/unknown-change}
        (not (cm/empty? chosen))
        (throw (ex-info (str "estimated: regenerate cannot resample some of the choices only, since an estimator has no"
                             " proposal density to weigh them by; select all of them, or update with a proposal")
                        {:selected (vec (cm/addresses chosen))}))
        (unchanged? argdiffs) {:trace tr :weight 0.0 :retdiff diff/no-change}
        :else (let [t (estimate-at this :regenerate args (:choices tr) key)]
                {:trace t :weight (- (:score t) (:score tr)) :retdiff diff/unknown-change}))))
  (-assess [this args choices key]
    (let [t (estimate-at this :assess args choices key)] {:weight (:score t) :retval (:retval t)}))
  (-propose [this args key]
    (let [t (simulated this args key)] {:choices (:choices t) :weight (:score t) :retval (:retval t)}))
  (-project [_ tr selection]
    (let [chosen (sel/get-selected (:choices tr) selection)]
      (cond (cm/empty? chosen) 0.0
            (= chosen (:choices tr)) (:score tr)
            :else (throw (ex-info "estimated: project takes all of the choices or none (the estimate is of all of them)"
                                  {:selected (vec (cm/addresses chosen))}))))))

(defn estimated
  "A generative function whose density is estimated (see the namespace
   doc), from :addresses, its choices' addresses (keywords or paths), and
   :estimate, (fn [args choices key] log-estimate); optionally :simulate,
   (fn [args key] {:choices :log-estimate}), and :retval, (fn [args
   choices] retval) (nil by default). Choices other than all of them are a
   loud error."
  [{:keys [estimate simulate retval addresses] :as spec}]
  (doseq [[k f] {:estimate estimate :simulate simulate :retval retval}]
    (when (and (or f (= k :estimate)) (not (ifn? f)))
      (throw (ex-info (str "estimated: " k " must be a function") {:keys (vec (keys spec))}))))
  (when-not (and (sequential? addresses) (seq addresses))
    (throw (ex-info "estimated: :addresses names its choices, e.g. [:y] or [[:obs 0] [:obs 1]]" {:keys (vec (keys spec))})))
  (->EstimatedGF estimate simulate (or retval (constantly nil)) (set (map #(if (vector? %) % [%]) addresses))))

(defmethod print-method EstimatedGF [_ ^java.io.Writer w]
  (.write w "#gen/estimated"))
