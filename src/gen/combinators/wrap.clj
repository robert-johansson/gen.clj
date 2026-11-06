(ns gen.combinators.wrap
  "The wrappers of gen.combinators that run another generative function
   as their own: Mix's body (a categorical and a Switch), and dimap's runs
   of its inner generative function, its arguments and return value
   mapped. (call-at and choice-at are gen bodies of a line, in
   gen.combinators.)"
  {:api :internal}
  (:require [gen.combinators :as-alias comb]
            [gen.combinators.calls :as calls :refer [stopped?]]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Mix
;; ---------------------------------------------------------------------------

(defn mix-body
  "Mix's model, a dynamic function: the component index a choice at :index,
   from (categorical weights) (or categorical-logits), and the component
   the branch of the Switch sw of the n components, its choices under
   :component."
  [sw n logits?]
  (let [index (if logits? d/categorical-logits d/categorical)]
    (dyn/gen [weights & args]
      (let [i (index weights)
            batch (dc/value-shape i)
            k (count (dc/support i))]
        (when-not (and (empty? batch) (== n k))
          (throw (ex-info (str "Mix's weights are a vector of one per component (" n "), not " k
                               (when (seq batch) (str " in a batch of shape " batch)))
                          {:components n :weights k :batch batch})))
        (apply splice :component sw (trace :index i) args)))))

;; ---------------------------------------------------------------------------
;; dimap
;; ---------------------------------------------------------------------------

(defn inner-args
  "The inner's arguments: (apply pre args) as a vector, args without pre."
  [{:keys [pre]} args]
  (if pre
    (let [a (apply pre args)]
      (when-not (sequential? a)
        (throw (ex-info (str "dimap's argument map returns the inner's arguments, a vector, not " (pr-str a))
                        {:value a})))
      (vec a))
    args))

(defn- outer-retval
  "The return value v of the inner's run mapped by post; a run that
   stopped (stopped?, given its score or weight s) has none to map."
  ([{:keys [post]} v] (if post (post v) v))
  ([gf v s] (if (stopped? s) nil (outer-retval gf v))))

(defn- dimap-trace [gf args sub retval]
  (assoc (trace/make-trace {:gen-fn gf :args args :choices (:choices sub) :retval retval :score (:score sub)})
         ::comb/subtrace sub))

(defn dimap-generate
  "generate (constraints, or nil to simulate) of a dimap on args."
  [gf args constraints key]
  (let [args (vec args)
        {sub :trace :as r} (calls/fresh-at (:inner gf) (inner-args gf args) constraints key nil)]
    (assoc r :trace (dimap-trace gf args sub (outer-retval gf (:retval sub) (:score sub))))))

(def ^:private absent (Object.))

(defn- inner-call
  "[arguments argdiffs] of the inner's call, from args and argdiffs. Without
   an argument map, the same. With one, which change hints cannot see
   through: when no argument changed (nor their number), the previous
   call's arguments, unchanged; otherwise the mapped ones, each no-change
   when it is identical to the previous call's, else unknown-change."
  [gf tr args argdiffs]
  (let [prev (:args (::comb/subtrace tr))]
    (cond (nil? (:pre gf)) [args argdiffs]
          (and (== (count args) (count (:args tr))) (every? diff/no-change? argdiffs)) [prev (diff/no-changes (count prev))]
          :else (let [ia (inner-args gf args)]
                  [ia (vec (map-indexed (fn [i a] (if (identical? a (nth prev i absent)) diff/no-change diff/unknown-change))
                                        ia))]))))

(defn dimap-revise
  "update or regenerate of a dimap trace: the inner's (inner-call). The
   return value is mapped again only when the inner's may have changed,
   and is no-change when it is the same value (gfi/same-value?)."
  [gf tr args argdiffs op cs key]
  (let [args (vec args)
        [ia iad] (inner-call gf tr args argdiffs)
        {sub' :trace rd :retdiff :as r} (calls/revise-at op (:inner gf) (::comb/subtrace tr) ia iad cs key nil)
        [v vd] (cond (nil? (:post gf)) [(:retval sub') rd]
                     (stopped? sub') [nil diff/unknown-change]
                     (diff/no-change? rd) [(:retval tr) diff/no-change]
                     :else (let [v (outer-retval gf (:retval sub'))]
                             [v (if (gfi/same-value? (:retval tr) v) diff/no-change diff/unknown-change)]))]
    (assoc r :trace (dimap-trace gf args sub' v) :retdiff vd)))

(defn dimap-assess
  "assess of a dimap on args: the inner's, its return value mapped."
  [gf args choices key]
  (let [r (calls/assess-at (:inner gf) (inner-args gf (vec args)) choices key nil)]
    (assoc r :retval (outer-retval gf (:retval r) (:weight r)))))
