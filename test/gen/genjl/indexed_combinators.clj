(ns gen.genjl.indexed-combinators
  "The checks that Gen.jl 0.4.8's test/modeling_library/map.jl and
   unfold.jl share. Map's elements and Unfold's steps are indexed, 0-based
   here (D2: Gen.jl's i is i - 1), each with one choice at [i leaf], and
   both files assert the same kinds of facts of a result. (check leaf
   result expected) asserts Gen.jl's expectations, a map with a key per
   kind of assertion:

     :args         get_args(trace) == args
     :choices      {i v}: choices[i => leaf] == v
     :absent       [i ...]: !has_value(choices, i => leaf)
     :shallow      {:values n :submaps m}: the counts of get_values_shallow
                   and get_submaps_shallow
     :discard      {i v}: discard[i => leaf] == v, and {} isempty(discard)
     :score        isapprox(get_score(trace), score)
     :weight       isapprox(weight, w)
     :retval       [v ...]: length(retval) == n and retval[i] == v
     :retdiff      :no-change (retdiff == NoChange()) or :vector-diff
                   (isa(retdiff, VectorDiff))
     :updated      {i diff}: retdiff.updated[i] == diff
     :not-updated  [i ...]: !haskey(retdiff.updated, i)
     :lengths      [prev new]: retdiff.prev_length and new_length

   A result is a trace (simulate's), the result of generate, propose or
   assess, or a revision by update or regenerate here, which keeps the
   previous trace: a revision's retdiff is also checked with
   diff/consistent? against it, as D1 asks of gen.clj's precise retdiffs."
  (:refer-clojure :exclude [update])
  (:require [clojure.test :refer [is]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.genjl.expectations :as expectations]
            [gen.genjl.support :as support :refer [approx?]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.test-support :refer [close?]]))

(defn holding
  "A model holding a parameter store for gf, a combinator, as Gen.jl's
   set_param! sets its kernel's: one store per model (D7), and a
   combinator has none of its own, so a dynamic function splicing gf at
   addr holds it."
  [addr gf store]
  (dyn/with-params (gen [& args] (apply splice addr gf args)) store))

(defn update
  "update of tr, the result keeping tr for check."
  [tr args argdiffs constraints]
  (assoc (gfi/update tr args argdiffs constraints (rng/key 1)) :prev tr))

(defn regenerate
  "regenerate of tr, the result keeping tr for check."
  [tr args argdiffs selection]
  (assoc (gfi/regenerate tr args argdiffs selection (rng/key 2)) :prev tr))

(defn- observed
  "What a result shows: its trace's args, choices, return value and score,
   and its weight, discard and retdiff."
  [result]
  (merge (select-keys (:trace result result) [:args :choices :retval :score])
         (select-keys result [:weight :discard :retdiff])))

(defn choice
  "choices[i => leaf] of a result."
  [leaf result i]
  (cm/get-choice (:choices (observed result)) [i leaf]))

(def ^:private checks
  "Each key's assertions, from what the result shows (with at, the
   address of index i) and the expectation."
  {:args (fn [{:keys [args]} want] (is (= want args) "get_args(trace)"))
   :choices (fn [{:keys [choices at]} want]
              (doseq [[i v] want] (is (close? v (cm/get-choice choices (at i))) (str "choices[" i "]"))))
   :absent (fn [{:keys [choices at]} want]
             (doseq [i want] (is (not (support/has-value? choices (at i))) (str "!has_value(choices, " i ")"))))
   :shallow (fn [{:keys [choices]} want] (is (= want (support/shallow choices)) "the choices' top level"))
   :discard (fn [{:keys [discard at]} want]
              (if (empty? want)
                (is (cm/empty? discard) "isempty(discard)")
                (doseq [[i v] want] (is (close? v (cm/get-choice discard (at i))) (str "discard[" i "]")))))
   :score (fn [{:keys [score]} want] (is (approx? want score) "get_score(trace)"))
   :weight (fn [{:keys [weight]} want] (is (approx? want weight) "weight"))
   :retval (fn [{:keys [retval]} want]
             (is (= (count want) (count retval)) "length(retval)")
             (doseq [[i v] (map-indexed vector want)] (is (close? v (nth retval i)) (str "retval[" i "]"))))
   :retdiff (fn [{:keys [retdiff]} want] (is (= want (expectations/diff-kind retdiff)) "the retdiff's kind"))
   :updated (fn [{:keys [retdiff]} want]
              (doseq [[i d] want] (is (= d (get (:updated retdiff) i)) (str "retdiff.updated[" i "]"))))
   :not-updated (fn [{:keys [retdiff]} want]
                  (doseq [i want] (is (not (contains? (:updated retdiff) i)) (str "!haskey(retdiff.updated, " i ")"))))
   :lengths (fn [{:keys [retdiff]} [prev new]]
              (is (= prev (:prev-length retdiff)) "retdiff.prev_length")
              (is (= new (:new-length retdiff)) "retdiff.new_length"))})

(defn check
  "Assert Gen.jl's expectations of a result (see the namespace doc)."
  [leaf result expected]
  (let [seen (assoc (observed result) :at (fn [i] [i leaf]))]
    (expectations/check checks seen expected)
    (when-let [prev (:prev result)]
      (is (diff/consistent? (:retval prev) (:retval seen) (:retdiff seen)) "a consistent retdiff (D1)"))))
