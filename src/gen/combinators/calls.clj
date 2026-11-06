(ns gen.combinators.calls
  "What gen.combinators' kinds share: their calls of a kernel, a branch or
   an inner generative function, the keys of indexed calls, the indices a
   selection reaches, and whether a part's run stopped at a choice outside
   its support."
  {:api :internal}
  (:require [gen.choicemap :as cm]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]))

(set! *warn-on-reflection* true)

(defn element-key
  "The key of element, step or node i: (fold-in key i), so that changing
   the length never shifts another element's randomness."
  [key i]
  (rng/fold-in key i))

;; ---------------------------------------------------------------------------
;; Calls, through dyn/run-call (a batched run is this same code under the
;; batch tracer); at is the call's address, its choices' place in the
;; combinator's (nil for a Switch's branch)
;; ---------------------------------------------------------------------------

(defn simulate-at [gf args key] (:trace (dyn/run-call gf :simulate args {:key key})))
(defn generate-at [gf args cs key at] (dyn/run-call gf :generate args {:constraints cs :key key :at at}))
(defn assess-at [gf args choices key at]
  (dyn/run-call gf :assess args {:constraints choices :key key :at at}))
(defn update-at [gf old args argdiffs cs key at]
  (dyn/run-call gf :update args {:prev old :argdiffs argdiffs :constraints cs :key key :at at}))
(defn regenerate-at [gf old args argdiffs s key at]
  (dyn/run-call gf :regenerate args {:prev old :argdiffs argdiffs :selection s :key key :at at}))

(defn fresh-at
  "A new run of gf on args, {:trace :weight}: generate under the
   constraints cs, or, when cs is nil, simulate, at weight 0."
  [gf args cs key at]
  (if cs
    (generate-at gf args cs key at)
    {:trace (simulate-at gf args key) :weight 0.0}))

(defn revise-at
  "The trace old revised: update (op :update, cs the constraints) or
   regenerate (op :regenerate, cs the selection)."
  [op gf old args argdiffs cs key at]
  (if (= op :update)
    (update-at gf old args argdiffs cs key at)
    (regenerate-at gf old args argdiffs cs key at)))

(defn reaching
  "What reaches the call at address at of cs: the selection there (op
   :regenerate), else the constraints there."
  [op cs at]
  (if (= op :regenerate) (sel/get-subselection cs at) (cm/get-submap cs at cm/EMPTY)))

(defn selected-indices
  "The indices a selection may select something under: :all, or a sorted
   set. An index past the elements selects nothing (they may be there at
   another length), but a negative one is no element's, nor a node's: a
   loud error in every kind that numbers its calls (genclj-kdau)."
  [s]
  (let [as (sel/top-addresses s)]
    (if (= :all as)
      :all
      (let [is (filter integer? as)]
        (when-let [neg (seq (sort (filter neg? is)))]
          (throw (ex-info (str "selection: a negative index " (pr-str (vec neg))
                               ", where elements and nodes are numbered from 0")
                          {:reason :negative-index :indices (vec neg)})))
        (into (sorted-set) is)))))

;; A choice outside its support stops the run that makes it (DESIGN.md §9,
;; the ten decisions' fifth; gfi/impossible?): its trace's score is -Inf on
;; the host, and the kernel's return value is not there. Unfold, Scan and
;; Recurse pass a part's return value on to the next part, so they stop at
;; such a part: the parts after it are not run (an update discards their
;; old choices), and the trace holds the parts up to it. A dimap does not
;; map such a return value. Map's elements, and a Switch's branch, pass
;; nothing on: Map runs every element, and a caller stops at its score.
;; Batched or lazy scores are arrays, never read: such a run goes on, -Inf
;; per particle (gen.dynamic).

(defn stopped?
  "Did the run of a part (a kernel's trace, or an assess's weight) stop at
   a choice outside its support?"
  [tr-or-weight]
  (gfi/impossible? (if (map? tr-or-weight) (:score tr-or-weight) tr-or-weight)))
