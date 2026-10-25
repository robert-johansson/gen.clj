(ns gen.inference.mh
  "Metropolis-Hastings, as Gen.jl's mh (metropolis_hastings):

     (mh trace selection)                 resample the selected choices from
                                          the internal proposal (regenerate)
     (mh trace proposal proposal-args)    a custom proposal q, called on
                                          (trace, proposal-args...)
     (involutive-mh trace proposal proposal-args involution)

   Each returns {:trace :accepted? :log-alpha}: the new trace (the old one if
   the move is rejected), whether it was accepted, and the log acceptance
   ratio. The key is an optional last argument; omitting it draws one from
   the implicit key source. A NaN acceptance ratio (for example from a chain
   started at an impossible trace) is a loud error. A move that takes a
   choice outside its support is rejected: the operation stops there with
   weight -Inf (DESIGN.md §9), before anything downstream runs."
  (:refer-clojure :exclude [+ -])
  (:require [gen.array :as array :refer [+ -]]
            [gen.batched :as batched]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(defn accept
  "Accept proposed with probability min(1, exp(log-alpha)): {:trace
   :accepted? :log-alpha}."
  [trace proposed log-alpha key]
  (let [la (double (array/item log-alpha))]
    (when (Double/isNaN la)
      (throw (ex-info "The MH acceptance ratio is NaN" {:log-alpha la :score (:score trace)})))
    (let [accepted? (or (>= la 0.0) (< (Math/log (double (array/item (rng/uniform key)))) la))]
      {:trace (if accepted? proposed trace) :accepted? accepted? :log-alpha la})))

(defn- selection-mh [trace selection key]
  (let [[k-move k-accept] (rng/split key)
        {proposed :trace weight :weight} (gfi/regenerate trace selection k-move)]
    (accept trace proposed weight k-accept)))

(defn- check-proposal-args [proposal-args]
  (when-not (sequential? proposal-args)
    (throw (ex-info "MH with a proposal needs its proposal arguments as a vector"
                    {:proposal-args proposal-args}))))

(defn move
  "A proposed move and its log acceptance ratio, {:trace trace' :weight w}
   (also a move-reweight weight, gen.inference.particle-filter). Forward
   choices u come from (proposal trace args…) drawn with k-propose. With
   :involution, (involution trace u retval args k-update) gives trace', the
   backward choices u' and the move's weight (k-update for its update,
   which draws a model's encapsulated randomness afresh); otherwise the
   trace is updated with u (k-update) and u' is the discard. u' is assessed under
   :backward on (trace', :backward-args…), or under the proposal on
   (trace', args…). w = the move's weight - log q(u) + log q_bwd(u'). A
   move outside the support (its weight -Inf on the host,
   gfi/impossible?: the update stopped at a choice there, so trace' holds
   the choices up to it) has w -Inf, its backward choices unassessed."
  [trace proposal proposal-args {:keys [involution backward backward-args]} k-propose k-update]
  (let [{fwd :choices fwd-weight :weight fwd-retval :retval} (gfi/propose proposal (into [trace] proposal-args) k-propose)
        [moved weight bwd] (if involution
                             (let [r (involution trace fwd fwd-retval proposal-args k-update)]
                               [(:trace r) (:weight r) (:bwd-choices r)])
                             (let [r (gfi/update trace fwd k-update)]
                               [(:trace r) (:weight r) (:discard r)]))]
    (if (gfi/impossible? weight)
      {:trace moved :weight ##-Inf}
      (let [{bwd-weight :weight} (gfi/assess (or backward proposal)
                                             (into [moved] (if backward backward-args proposal-args))
                                             bwd (gfi/backward-key k-propose))]
        {:trace moved :weight (+ (- weight fwd-weight) bwd-weight)}))))

(defn- proposal-mh
  "Gen.jl's MH with a custom proposal: propose forward choices from
   q(trace, args...), update the trace with them, and assess the discard
   under q(trace', args...) as the backward move."
  [trace proposal proposal-args key]
  (when-not (gfi/generative-function? proposal)
    (throw (ex-info (str "MH takes a selection, such as (gen.selection/select :a), or a proposal generative"
                         " function, not " (pr-str proposal))
                    {:value proposal})))
  (check-proposal-args proposal-args)
  (let [[k-propose k-update k-accept] (rng/split key 3)
        {proposed :trace w :weight} (move trace proposal proposal-args {} k-propose k-update)]
    (accept trace proposed w k-accept)))

(defn mh
  "One Metropolis-Hastings move: with a selection, regenerate it; with a
   proposal generative function and its arguments, propose from it."
  ([trace selection-or-proposal]
   (mh trace selection-or-proposal (gfi/next-key)))
  ([trace x y]
   (if (sel/selection? x)
     (selection-mh trace x y)
     (proposal-mh trace x y (gfi/next-key))))
  ([trace proposal proposal-args key]
   (proposal-mh trace proposal proposal-args key)))

(defn involutive-mh
  "Involutive MCMC (Gen.jl's mh with an involution): propose auxiliary
   choices u from q(trace, args...), map (trace, u) to (trace', u') with the
   deterministic involution, and accept with ratio
   p(trace') q(u'; trace') / (p(trace) q(u; trace)) · |J|.

   involution is (fn [trace fwd-choices fwd-retval proposal-args key]) and
   returns {:trace trace' :bwd-choices u' :weight w}, with w = log p(trace')
   - log p(trace) + log |det J| (the update weight plus the log Jacobian, 0
   for discrete moves). Applying it twice must give back (trace, u). Gen.jl's
   involution takes no key; gen.clj's takes one for its update, as every
   operation does, which draws a model's encapsulated randomness afresh (a
   pseudo-marginal move)."
  ([trace proposal proposal-args involution]
   (involutive-mh trace proposal proposal-args involution (gfi/next-key)))
  ([trace proposal proposal-args involution key]
   (check-proposal-args proposal-args)
   ;; split's i-th key does not depend on n: the involution's is a third
   (let [[k-propose k-accept k-update] (rng/split key 3)
         {proposed :trace w :weight} (move trace proposal proposal-args {:involution involution} k-propose k-update)]
     (accept trace proposed w k-accept))))

(defn mh-particles
  "One selection-MH move for each of n particles (n independent chains,
   gen.batched), accepted or rejected per particle. Returns {:particles
   :accepted? (n booleans) :log-alpha (an [n] array)}."
  ([ps selection] (mh-particles ps selection (gfi/next-key)))
  ([ps selection key]
   (let [[k-move k-accept] (rng/split key)
         {proposed :particles la :weights}
         (batched/regenerate ps (:args ps) (vec (repeat (count (:args ps)) diff/no-change)) selection k-move)
         las (mapv double (array/->clj la))
         _ (when (some #(Double/isNaN %) las)
             (throw (ex-info "An MH acceptance ratio is NaN" {:log-alpha las})))
         us (array/->clj (rng/uniform k-accept [(batched/size ps)]))
         accepted (mapv (fn [l u] (or (>= l 0.0) (< (Math/log u) l))) las us)]
     {:particles (batched/select accepted proposed ps) :accepted? accepted :log-alpha la})))
