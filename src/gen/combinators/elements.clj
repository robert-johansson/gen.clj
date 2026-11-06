(ns gen.combinators.elements
  "The indexed calls of Map, Unfold and Scan (gen.combinators), and Map.
   Such a trace keeps each element's or step's subtrace
   (:gen.combinators/subtraces), its choices at the element's index, and its
   score as a running total of the elements' (:gen.combinators/total). An
   update or regenerate changes in place the elements it revisits, in an
   accumulator made from the old trace (trace-acc), and drops those past a
   shorter length (drop-elements)."
  {:api :internal}
  (:refer-clojure :exclude [+ -])
  (:require [gen.array :as array :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.combinators :as-alias comb]
            [gen.combinators.calls :as calls :refer [element-key]]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Indexed elements
;; ---------------------------------------------------------------------------

(defn indices
  "The top-level addresses of choices, checked to be indices below n, as a
   sorted set."
  [op choices n]
  (when (cm/has-value? choices)
    (throw (gfi/unvisited op [[]] "a single value was given where indexed choices belong")))
  (let [addrs (map first (cm/submaps choices))]
    (when-let [outside (seq (remove #(and (integer? %) (< -1 % n)) addrs))]
      (throw (gfi/unvisited op (mapv vector outside) (str "the indices are 0 to " (dec n)))))
    (into (sorted-set) (map long) addrs)))

(defn targets
  "The indices an update's constraints (op :update) or a regenerate's
   selection (op :regenerate) reach, the constraints checked to be below
   n: :all, or a sorted set."
  [op cs n]
  (if (= op :update) (indices :update cs n) (calls/selected-indices cs)))

(defn element-call
  "Map's, Unfold's or Scan's call at path p: element (p 0)'s trace."
  [tr p]
  (let [subs (::comb/subtraces tr)
        i (p 0)]
    (when (and (integer? i) (< -1 i (count subs)))
      [(subs i) (subvec p 1)])))

(defn element-diff
  "The diff of element i of a vector whose diff is d."
  [d i]
  (cond (diff/no-change? d) diff/no-change
        (diff/vector-diff? d) (get (:updated d) i diff/no-change)
        :else diff/unknown-change))

(defn project
  "The projection of an elements trace tr on the selection s: its
   elements' under their subselections."
  [tr s]
  (let [subs (::comb/subtraces tr)
        is (calls/selected-indices s)]
    (reduce (fn [w i] (let [sub (subs i)] (+ w (gfi/-project (:gen-fn sub) sub (sel/get-subselection s i)))))
            0.0
            (if (= :all is) (range (count subs)) (filter #(< % (count subs)) is)))))

;; ---------------------------------------------------------------------------
;; The score, a running total
;; ---------------------------------------------------------------------------

;; The score of a Map, Unfold or Scan trace is a running total of its
;; elements' scores, so a change costs the elements it changes, as in
;; Gen.jl. A plain one (score + new - old) gives -Inf + Inf = NaN once an
;; element left its support and came back (genclj-awd0). So a total keeps
;; the sum of the finite element scores beside two counts: of the scores not
;; above -Inf, and of those not below +Inf. A NaN is in both, as -Inf + Inf
;; is NaN, so the score is the IEEE sum of the elements' scores, what assess
;; sums, the finite part rounded as a running total (genclj-ul72). The
;; finite part is a compensated sum (Knuth's two-sum), :finite and its rounding
;; error :comp: a plain running total drifts from the elements' sum by the
;; rounding of every step it took, past the laws' float64 tolerance after a
;; long chain of updates that take large scores in and out of a small total.

(def ^:private no-total {:finite 0.0 :comp 0.0 :neg-inf 0 :pos-inf 0})

(defn- compensated
  "[finite comp] with x added: Knuth's two-sum, branch-free, so finite +
   comp holds the exact sum up to one rounding."
  [finite comp x]
  (let [t (+ finite x)
        b (- t finite)]
    [t (+ comp (+ (- finite (- t b)) (- x b)))]))

(defn- add-finite
  "total with finite score x added to its compensated sum."
  [total x]
  (let [[f c] (compensated (:finite total) (:comp total 0.0) x)]
    (assoc total :finite f :comp c)))

(defn- tallied
  "total with score s added (op +) or taken out (op -)."
  [total op s]
  (if (array/host? s)
    (let [x (double s)]
      (if (< ##-Inf x ##Inf)
        (add-finite total (if (= op +) s (- s)))
        (cond-> total
          (not (> x ##-Inf)) (update :neg-inf op 1)
          (not (< x ##Inf)) (update :pos-inf op 1))))
    (let [above (array/> s ##-Inf)
          below (array/< s ##Inf)]
      (-> total
          (add-finite (let [f (array/where (array/logical-and above below) s 0.0)] (if (= op +) f (- f))))
          (update :neg-inf op (array/where above 0 1))
          (update :pos-inf op (array/where below 0 1))))))

(defn- total-score
  "The score of a total: its finite sum, -Inf with a score not above -Inf,
   +Inf with one not below +Inf, and so NaN with both."
  [{:keys [neg-inf pos-inf] :as total}]
  (let [finite (+ (:finite total) (:comp total 0.0))]
    (if (and (array/host? neg-inf) (array/host? pos-inf))
      (cond-> finite
        (pos? (long neg-inf)) (+ ##-Inf)
        (pos? (long pos-inf)) (+ ##Inf))
      (+ (+ finite (array/where (array/> neg-inf 0) ##-Inf 0.0)) (array/where (array/> pos-inf 0) ##Inf 0.0)))))

;; ---------------------------------------------------------------------------
;; The accumulator: {:subs :choices :retval :total :weight :discard
;; :retdiffs}, :retval the elements' outputs (Scan's ys)
;; ---------------------------------------------------------------------------

(defn elements-trace
  "The trace of gf on args that the accumulator holds, its return value
   the accumulator's."
  [gf args {:keys [subs choices retval total]}]
  (assoc (trace/make-trace {:gen-fn gf :args args :choices choices :retval retval :score (total-score total)})
         ::comb/subtraces subs
         ::comb/total total))

(def empty-acc
  "An accumulator of no elements."
  {:subs [] :choices cm/EMPTY :retval [] :total no-total :weight 0.0 :discard cm/EMPTY :retdiffs {}})

(defn trace-acc
  "An accumulator holding the elements of tr, to be changed in place, and
   their outputs (tr's return value, or Scan's ys)."
  [tr outputs]
  (assoc empty-acc :subs (::comb/subtraces tr) :choices (:choices tr) :retval outputs :total (::comb/total tr)))

(defn- assoc-element [choices i sub]
  (if (cm/empty? sub) (cm/remove-submap choices i) (cm/set-submap choices i sub)))

(defn put-run
  "acc with the run r of element i ({:trace :weight}, and update's
   :discard) in place of old (if given): its weight added, its output y
   (the element's return value, a Scan step's y) among the outputs, y's
   diff yd among the changes."
  [acc i old {sub :trace w :weight d :discard} y yd]
  (cond-> (-> acc
              (assoc-in [:subs i] sub)
              (update :choices assoc-element i (:choices sub))
              (assoc-in [:retval i] y)
              (update :total #(cond-> (tallied % + (:score sub)) old (tallied - (:score old))))
              (update :weight + w))
    (and d (not (cm/empty? d))) (update :discard cm/set-submap i d)
    (diff/changed? yd) (assoc-in [:retdiffs i] yd)))

(defn- keep-first
  "v's first n elements, as v itself when they are all of it: a vector of
   v's class, so a return value nothing changed is the same value
   (gfi/same-value?), and no subvec holds on to dropped elements."
  [v n]
  (if (== n (count v)) v (into [] (take n) v)))

(defn drop-elements
  "acc without the elements from n on. update discards them and subtracts
   their scores from the weight; in regenerate they cancel against the
   reverse move."
  [acc n op]
  (let [gone (subvec (:subs acc) n)]
    (-> (reduce (fn [acc [i sub]]
                  (cond-> (-> acc
                              (update :choices cm/remove-submap i)
                              (update :total tallied - (:score sub)))
                    (= op :update) (-> (update :weight - (:score sub))
                                       (cond-> (not (cm/empty? (:choices sub)))
                                         (update :discard cm/set-submap i (:choices sub))))))
                acc
                (map vector (range n (count (:subs acc))) gone))
        (update :subs keep-first n)
        (update :retval keep-first n))))

(defn retdiff
  "The diff of the outputs of acc, n elements, prev-n before."
  [{:keys [retdiffs]} n prev-n]
  (if (and (= n prev-n) (empty? retdiffs)) diff/no-change (diff/vector-diff n prev-n retdiffs)))

(defn incremental-result
  "An update's or regenerate's result from acc, the steps of a run that
   stopped (stopped?) at its last one with an unknown retdiff."
  [gf args acc op n prev-n stopped]
  (cond-> {:trace (elements-trace gf args acc) :weight (:weight acc)
           :retdiff (if stopped diff/unknown-change (retdiff acc n prev-n))}
    (= op :update) (assoc :discard (:discard acc))))

;; ---------------------------------------------------------------------------
;; Map
;; ---------------------------------------------------------------------------

(defn- map-args
  "Map's arguments as vectors: each is a sequential collection, one element
   per application."
  [args]
  (when-let [bad (first (remove sequential? args))]
    (throw (ex-info (str "Map's arguments are vectors, one element per application, not " (pr-str bad)) {:arg bad})))
  (mapv vec args))

(defn- map-length [args]
  (when (empty? args)
    (throw (ex-info "Map needs at least one argument vector" {})))
  (let [ns (mapv count args)]
    (when-not (apply = ns)
      (throw (ex-info "Map's argument vectors must have one length" {:lengths ns})))
    (first ns)))

(defn- element-args [args i] (mapv #(nth % i) args))

(defn- element-argdiffs
  "The kernel's argdiffs for element i, from Map's argdiffs."
  [argdiffs i]
  (mapv #(element-diff % i) argdiffs))

(defn- changed-elements
  "The indices whose arguments may have changed: :all, or a sorted set. A
   change in the number of arguments changes every element's (DESIGN.md §9,
   G3: no argdiff can say one went or came)."
  [argdiffs same-count?]
  (if (and same-count? (every? #(or (diff/no-change? %) (diff/vector-diff? %)) argdiffs))
    (into (sorted-set) (mapcat #(when (diff/vector-diff? %) (keys (:updated %)))) argdiffs)
    :all))

(defn- visits
  "The retained indices below kept to revisit: those in either set (or every
   one, if either is :all)."
  [xs ys kept]
  (if (or (= :all xs) (= :all ys)) (range kept) (subseq (into xs ys) < kept)))

(defn- fresh-elements
  "acc with the elements from from on, below n, run afresh: generated
   under the constraints cs, or simulated when cs is nil."
  [acc kernel args from n cs key]
  (reduce (fn [acc i]
            (let [r (calls/fresh-at kernel (element-args args i) (when cs (cm/get-submap cs i cm/EMPTY))
                                    (element-key key i) [i])]
              (put-run acc i nil r (:retval (:trace r)) diff/no-change)))
          acc
          (range from n)))

(defn map-generate
  "generate (constraints, or nil to simulate) of a Map on args."
  [gf args constraints key]
  (let [args (map-args args)
        n (map-length args)]
    (when constraints (indices :generate constraints n))
    (let [acc (fresh-elements empty-acc (:kernel gf) args 0 n constraints key)]
      {:trace (elements-trace gf args acc) :weight (:weight acc)})))

(defn map-revise
  "update (op :update, with constraints) or regenerate (op :regenerate, with
   a selection) of a Map trace: the retained elements whose arguments,
   constraints or selection may have changed revisited, the rest kept; then
   extended or truncated."
  [gf tr args argdiffs op cs key]
  (let [args (map-args args)
        n (map-length args)
        kernel (:kernel gf)
        old-subs (::comb/subtraces tr)
        prev-n (count old-subs)
        kept (min n prev-n)
        reached (targets op cs n)
        acc (reduce (fn [acc i]
                      (let [old (old-subs i)
                            r (calls/revise-at op kernel old (element-args args i) (element-argdiffs argdiffs i)
                                               (calls/reaching op cs i) (element-key key i) [i])]
                        (put-run acc i old r (:retval (:trace r)) (:retdiff r))))
                    (trace-acc tr (:retval tr))
                    (visits (changed-elements argdiffs (== (count args) (count (:args tr)))) reached kept))
        acc (fresh-elements (drop-elements acc kept op) kernel args kept n (when (= op :update) cs) key)]
    (incremental-result gf args acc op n prev-n false)))

(defn map-assess
  "assess of a Map on args: every element's, under its choices."
  [gf args choices key]
  (let [args (map-args args)
        n (map-length args)]
    (indices :assess choices n)
    (reduce (fn [{:keys [weight retval]} i]
              (let [r (calls/assess-at (:kernel gf) (element-args args i) (cm/get-submap choices i cm/EMPTY)
                                       (element-key key i) [i])]
                {:weight (+ weight (:weight r)) :retval (conj retval (:retval r))}))
            {:weight 0.0 :retval []}
            (range n))))
