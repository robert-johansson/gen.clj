(ns gen.custom
  "Gen.jl's custom deterministic generative functions (CustomDetermGF and
   CustomUpdateGF): no random choices, a return value computed by your
   function, and an update you write to reuse earlier work.

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
   composes as any other: splice it in a model, map it, unfold it. Gradients
   through it come with custom VJPs (Gen.jl's CustomGradientGF, bean genclj-8cp8)."
  (:require [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defn- no-choices! [op choices]
  (when-not (cm/empty? choices)
    (throw (ex-info (str (name op) ": a deterministic generative function makes no choices, but was given "
                         (pr-str (vec (cm/addresses choices))))
                    {:op op :addresses (vec (cm/addresses choices))}))))

(defn- run [gf args]
  (let [args (vec args)
        [retval state] ((:apply-with-state gf) args)]
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

(defrecord DeterministicGF [apply-with-state update-with-state]
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
  (applyTo [this args] (:retval (run this (vec args)))))

(defn deterministic
  "A deterministic generative function (see the namespace doc), from
   :apply-with-state, (fn [args] [retval state]), and optionally
   :update-with-state, (fn [state args argdiffs] [state retval retdiff])."
  [{:keys [apply-with-state update-with-state]}]
  (when-not (fn? apply-with-state)
    (throw (ex-info "deterministic needs :apply-with-state, (fn [args] [retval state])" {})))
  (when (and update-with-state (not (fn? update-with-state)))
    (throw (ex-info "deterministic: :update-with-state must be (fn [state args argdiffs] [state retval retdiff])" {})))
  (->DeterministicGF apply-with-state
                     (or update-with-state
                         (fn [_ args _] (let [[retval state] (apply-with-state args)] [state retval diff/unknown-change])))))

(defmethod print-method DeterministicGF [_ ^java.io.Writer w]
  (.write w "#gen/deterministic"))
