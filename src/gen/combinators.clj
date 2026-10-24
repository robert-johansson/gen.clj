(ns gen.combinators
  "Gen.jl's generative function combinators.

     (map kernel)       Map: args are vectors of equal length; element i calls
                        (kernel (nth arg1 i) (nth arg2 i) …), its choices at
                        address i. Returns the vector of element return values.
     (unfold kernel)    Unfold: args (n init & params); step t calls
                        (kernel t state & params), which returns the next
                        state; step t's choices are at address t. Returns the
                        vector of the n states after each step.
     (switch & gfs)     Switch: args (index & args) call branch index.
     (switch {k gf …})  keyed Switch: args (k & args).
     (recurse production aggregation max-branch)
                        Recurse: a tree. Node i's production kernel, on
                        its input, returns {:value v :children [inputs …]}
                        (at most max-branch); its aggregation kernel, on
                        (v [the children's aggregations]), returns the
                        node's value. Args (input), the root's; node i's
                        choices are at [i :production] and [i :aggregation],
                        the root 0 and child j of node i at
                        i·max-branch + j + 1 (get-child; get-child-num
                        and get-parent invert it). Returns the root's
                        aggregation.
     (call-at gf)       Gen.jl's call_at: args (& args k) call gf on args,
                        its choices under address k.
     (choice-at dist)   Gen.jl's choice_at: args (& args k), one choice at
                        address k from (dist & args).

   GenMLX's (and GenJAX's) further combinators:

     (scan kernel)      Scan, JAX's scan: args (init xs & params); step t
                        calls (kernel carry (nth xs t) & params), carry init
                        at step 0, and the kernel returns [carry' y]; step
                        t's choices are at address t. Returns [carry ys],
                        the last carry and the vector of the outputs.
     (mix gf0 gf1 …)    Mix: args (weights & args). The component index is a
                        choice at :index, from (categorical weights), and
                        the component, called on args, makes its choices
                        under :component (a Switch's). Returns the
                        component's return value. mix-logits takes logits.
     (dimap gf f g)     gf called on (apply f args), returning (g v) of its
                        return value v; the choices are gf's, at gf's
                        addresses. (contramap gf f) and (map-retval gf g)
                        map one side.

   Addresses and indices are 0-based (DESIGN.md §9). A trace keeps each
   element's or step's subtrace, so update and regenerate are incremental, as
   in Gen.jl:
   - Map revisits only the elements whose arguments may have changed
     (vector-diff argdiffs), or that are constrained or selected.
   - Unfold revisits a step only when its state input (the previous step's
     retdiff), its parameters, its constraints or its selection may have
     changed. So does Scan, whose step inputs are its carry, its x (by a
     vector-diff of xs) and its parameters.
   - Elements and steps beyond a changed length are generated, or discarded.
   - The retdiff is a vector-diff of the returned vector (Scan's of [carry
     ys], ys's a vector-diff too).
   A Switch whose index changes runs the new branch against the old branch's
   choices, whatever their kinds (gfi/revise), so the addresses they share
   keep their values, as an if would and as Gen.jl does; the old choices
   the new branch does not make are discarded. In a batched run, a Switch
   whose index differs across the particles runs each branch batched on its
   own particles (by partition: DESIGN.md §9, gen.partition).

   Element or step i draws from (fold-in key i), so changing the length never
   shifts another element's randomness."
  (:refer-clojure :exclude [map + -])
  (:require [gen.array :as array :refer [+ -]]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.partition :as part]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Indexed elements (Map and Unfold)
;; ---------------------------------------------------------------------------

(defn- element-key [key i] (rng/fold-in key i))

;; Kernel calls, through dyn/run-call (a batched run is this same code
;; under the batch tracer); at is the call's address, its choices' place in
;; the combinator's (nil for a Switch's branch).
(defn- simulate-at [gf args key] (:trace (dyn/run-call gf :simulate args {:key key})))
(defn- generate-at [gf args cs key at] (dyn/run-call gf :generate args {:constraints cs :key key :at at}))
(defn- assess-at [gf args choices key at]
  (dyn/run-call gf :assess args {:constraints choices :key key :at at}))
(defn- update-at [gf old args argdiffs cs key at]
  (dyn/run-call gf :update args {:prev old :argdiffs argdiffs :constraints cs :key key :at at}))
(defn- regenerate-at [gf old args argdiffs s key at]
  (dyn/run-call gf :regenerate args {:prev old :argdiffs argdiffs :selection s :key key :at at}))

(defn- assoc-element [choices i sub]
  (if (cm/empty? sub) (cm/remove-submap choices i) (cm/set-submap choices i sub)))

(defn- indices
  "The top-level addresses of choices, checked to be indices below n, as a
   sorted set."
  [op choices n]
  (when (cm/has-value? choices)
    (throw (gfi/unvisited op [[]] "a single value was given where indexed choices belong")))
  (let [addrs (clojure.core/map first (cm/submaps choices))]
    (when-let [outside (seq (remove #(and (integer? %) (< -1 % n)) addrs))]
      (throw (gfi/unvisited op (mapv vector outside) (str "the indices are 0 to " (dec n)))))
    (into (sorted-set) (clojure.core/map long) addrs)))

(defn- selected-indices
  "The indices a selection may select something under: :all, or a sorted set."
  [s]
  (let [as (sel/top-addresses s)]
    (if (= :all as) :all (into (sorted-set) (filter integer?) as))))

(defn- element-call
  "Map's or Unfold's call at path p: element (p 0)'s trace."
  [tr p]
  (let [subs (::subtraces tr)
        i (p 0)]
    (when (and (integer? i) (< -1 i (count subs)))
      [(subs i) (subvec p 1)])))

;; The score of a Map, Unfold or Scan trace is a running total of its
;; elements' scores, so a change costs the elements it changes, as in
;; Gen.jl. A plain one (score + new - old) gives -Inf + Inf = NaN once an
;; element left its support and came back (genclj-awd0). So a total keeps
;; the sum of the finite element scores beside two counts: of the scores not
;; above -Inf, and of those not below +Inf. A NaN is in both, as -Inf + Inf
;; is NaN, so the score is the IEEE sum of the elements' scores, what assess
;; sums, the finite part rounded as a running total (genclj-ul72).

(def ^:private no-total {:finite 0.0 :neg-inf 0 :pos-inf 0})

(defn- tallied
  "total with score s added (op +) or taken out (op -)."
  [total op s]
  (if (array/host? s)
    (let [x (double s)]
      (if (< ##-Inf x ##Inf)
        (update total :finite op s)
        (cond-> total
          (not (> x ##-Inf)) (update :neg-inf op 1)
          (not (< x ##Inf)) (update :pos-inf op 1))))
    (let [above (array/> s ##-Inf)
          below (array/< s ##Inf)]
      (-> total
          (update :finite op (array/where (array/logical-and above below) s 0.0))
          (update :neg-inf op (array/where above 0 1))
          (update :pos-inf op (array/where below 0 1))))))

(defn- total-score
  "The score of a total: its finite sum, -Inf with a score not above -Inf,
   +Inf with one not below +Inf, and so NaN with both."
  [{:keys [finite neg-inf pos-inf]}]
  (if (and (array/host? neg-inf) (array/host? pos-inf))
    (cond-> finite
      (pos? (long neg-inf)) (+ ##-Inf)
      (pos? (long pos-inf)) (+ ##Inf))
    (+ (+ finite (array/where (array/> neg-inf 0) ##-Inf 0.0)) (array/where (array/> pos-inf 0) ##Inf 0.0))))

(defn- elements-trace [gf args {:keys [subs choices retval total]}]
  (assoc (trace/make-trace {:gen-fn gf :args args :choices choices :retval retval :score (total-score total)})
         ::subtraces subs
         ::total total))

(def ^:private empty-acc
  {:subs [] :choices cm/EMPTY :retval [] :total no-total :weight 0.0 :discard cm/EMPTY :retdiffs {}})

(defn- trace-acc
  "An accumulator holding the elements of tr, to be changed in place."
  [tr]
  (assoc empty-acc :subs (::subtraces tr) :choices (:choices tr) :retval (:retval tr) :total (::total tr)))

(defn- put-element
  "acc with sub as element i (replacing old, if given), its weight w added."
  [acc i sub old w]
  (-> acc
      (assoc-in [:subs i] sub)
      (update :choices assoc-element i (:choices sub))
      (assoc-in [:retval i] (:retval sub))
      (update :total #(cond-> (tallied % + (:score sub)) old (tallied - (:score old))))
      (update :weight + w)))

(defn- put-update
  "acc with element i updated: the result of update or regenerate."
  [acc i old {sub :trace w :weight d :discard rd :retdiff}]
  (cond-> (put-element acc i sub old w)
    (and d (not (cm/empty? d))) (update :discard cm/set-submap i d)
    (diff/changed? rd) (assoc-in [:retdiffs i] rd)))

(defn- keep-first
  "v's first n elements, as v itself when they are all of it: a vector of
   v's class, so a return value nothing changed is the same value
   (gfi/same-value?), and no subvec holds on to dropped elements."
  [v n]
  (if (== n (count v)) v (into [] (take n) v)))

(defn- drop-elements
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
                (clojure.core/map vector (range n (count (:subs acc))) gone))
        (update :subs keep-first n)
        (update :retval keep-first n))))

(defn- retdiff [{:keys [retdiffs]} n prev-n]
  (if (and (= n prev-n) (empty? retdiffs)) diff/no-change (diff/vector-diff n prev-n retdiffs)))

(defn- incremental-result [gf args acc op n prev-n]
  (cond-> {:trace (elements-trace gf args acc) :weight (:weight acc) :retdiff (retdiff acc n prev-n)}
    (= op :update) (assoc :discard (:discard acc))))

(defn- project-elements [tr s]
  (let [subs (::subtraces tr)
        is (selected-indices s)]
    (reduce (fn [w i] (let [sub (subs i)] (+ w (gfi/-project (:gen-fn sub) sub (sel/get-subselection s i)))))
            0.0
            (if (= :all is) (range (count subs)) (filter #(< % (count subs)) is)))))

(defn- propose-from-trace [tr]
  {:choices (:choices tr) :weight (:score tr) :retval (:retval tr)})

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

(defn- unknown-diffs [args] (vec (repeat (count args) diff/unknown-change)))

(defn- element-diff
  "The diff of element i of a vector whose diff is d."
  [d i]
  (cond (diff/no-change? d) diff/no-change
        (diff/vector-diff? d) (get (:updated d) i diff/no-change)
        :else diff/unknown-change))

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

(defn- map-generate [gf args constraints key]
  (let [args (map-args args)
        n (map-length args)
        kernel (:kernel gf)]
    (when constraints (indices :generate constraints n))
    (let [acc (reduce (fn [acc i]
                        (let [ea (element-args args i)
                              k (element-key key i)]
                          (if constraints
                            (let [{sub :trace w :weight} (generate-at kernel ea (cm/get-submap constraints i cm/EMPTY) k [i])]
                              (put-element acc i sub nil w))
                            (put-element acc i (simulate-at kernel ea k) nil 0.0))))
                      empty-acc
                      (range n))]
      {:trace (elements-trace gf args acc) :weight (:weight acc)})))

(defn- map-revise
  "update (op :update, with constraints) or regenerate (op :regenerate, with
   a selection) of a Map trace."
  [gf tr args argdiffs op cs key]
  (let [args (map-args args)
        n (map-length args)
        kernel (:kernel gf)
        old-subs (::subtraces tr)
        prev-n (count old-subs)
        kept (min n prev-n)
        targets (if (= op :update) (indices :update cs n) (selected-indices cs))
        acc (reduce (fn [acc i]
                      (let [old (old-subs i)
                            ea (element-args args i)
                            ead (element-argdiffs argdiffs i)
                            k (element-key key i)]
                        (put-update acc i old
                                    (if (= op :update)
                                      (update-at kernel old ea ead (cm/get-submap cs i cm/EMPTY) k [i])
                                      (regenerate-at kernel old ea ead (sel/get-subselection cs i) k [i])))))
                    (trace-acc tr)
                    (visits (changed-elements argdiffs (== (count args) (count (:args tr)))) targets kept))
        acc (drop-elements acc kept op)
        acc (reduce (fn [acc i]
                      (let [ea (element-args args i)
                            k (element-key key i)]
                        (if (= op :update)
                          (let [{sub :trace w :weight} (generate-at kernel ea (cm/get-submap cs i cm/EMPTY) k [i])]
                            (put-element acc i sub nil w))
                          (put-element acc i (simulate-at kernel ea k) nil 0.0))))
                    acc
                    (range kept n))]
    (incremental-result gf args acc op n prev-n)))

(defn- map-assess [gf args choices key]
  (let [args (map-args args)
        n (map-length args)]
    (indices :assess choices n)
    (reduce (fn [{:keys [weight retval]} i]
              (let [r (assess-at (:kernel gf) (element-args args i) (cm/get-submap choices i cm/EMPTY) (element-key key i) [i])]
                {:weight (+ weight (:weight r)) :retval (conj retval (:retval r))}))
            {:weight 0.0 :retval []}
            (range n))))

;; A combinator's trainable-parameter stores (gfi/IStores): its callees',
;; keyed as it holds them, a Map's and an Unfold's those of their one kernel

(defn- callee-stores
  "{key stores} over the callees, a vector or map of generative functions
   by key, that hold stores; nil when none does."
  [callees]
  (not-empty (into {} (keep (fn [[k gf]] (when-let [s (gfi/stores gf)] [k s])))
                   (if (map? callees) callees (map-indexed vector callees)))))

(defn- with-callee-stores
  "callees (a vector or map of generative functions by key) with stores,
   {key stores}, given to them."
  [callees stores]
  (reduce-kv (fn [cs k s] (assoc cs k (gfi/with-stores (get cs k) s))) callees stores))

(gfi/defgf MapGF [kernel]
  gfi/IStores
  (-stores [gf] (gfi/stores (:kernel gf)))
  (-with-stores [gf stores] (assoc gf :kernel (gfi/with-stores (:kernel gf) stores)))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (map-generate gf args nil key)))
  (-generate [gf args constraints key] (map-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (map-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (map-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (map-assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (map-generate gf args nil key))))
  (-project [_ tr selection] (project-elements tr selection))
  trace/ICalls
  (-call [_ tr p] (element-call tr p)))

(defn map
  "Gen.jl's Map combinator: the kernel applied independently to each element
   of its argument vectors."
  [kernel]
  (->MapGF kernel))

;; ---------------------------------------------------------------------------
;; Unfold
;; ---------------------------------------------------------------------------

(defn- check-steps
  "The number of steps as a host integer (a one-element array is read; a
   number that differs across a batch's particles raises the batched-branch
   error, so batched inference falls back)."
  [n]
  (let [n (if (array/backend-array? n) (array/item n) n)]
    (when-not (nat-int? n)
      (throw (ex-info "Unfold's first argument, the number of steps, must be a non-negative integer" {:n n})))
    n))

(defn- step-args [t state params] (into [t state] params))

(defn- unfold-generate [gf [n init & params :as args] constraints key]
  (let [n (check-steps n)
        kernel (:kernel gf)]
    (when constraints (indices :generate constraints n))
    (let [acc (loop [t 0 state init acc empty-acc]
                (if (< t n)
                  (let [sa (step-args t state params)
                        k (element-key key t)
                        [sub w] (if constraints
                                  (let [r (generate-at kernel sa (cm/get-submap constraints t cm/EMPTY) k [t])]
                                    [(:trace r) (:weight r)])
                                  [(simulate-at kernel sa k) 0.0])]
                    (recur (inc t) (:retval sub) (put-element acc t sub nil w)))
                  acc))]
      {:trace (elements-trace gf (vec args) acc) :weight (:weight acc)})))

(defn- unfold-revise
  "update or regenerate of an Unfold trace. Walks the retained steps,
   revisiting a step when its state input, the parameters, its constraints or
   its selection may have changed, and skipping straight to the next
   targeted step otherwise; then extends or truncates."
  [gf tr [n init & params :as args] argdiffs op cs key]
  (let [n (check-steps n)
        kernel (:kernel gf)
        [_ dinit & dparams] argdiffs
        ;; a change in their number is a change (DESIGN.md §9, G3)
        params-changed? (or (not= (count args) (count (:args tr))) (some diff/changed? dparams))
        old-subs (::subtraces tr)
        prev-n (count old-subs)
        kept (min n prev-n)
        targets (if (= op :update) (indices :update cs n) (selected-indices cs))
        target? (fn [t] (or (= :all targets) (contains? targets t)))
        next-target (fn [t] (if (= :all targets) t (or (first (subseq targets >= t)) kept)))
        [state acc]
        (loop [t 0 state init sdiff dinit acc (trace-acc tr)]
          (cond
            (>= t kept) [state acc]

            (or params-changed? (diff/changed? sdiff) (target? t))
            (let [old (old-subs t)
                  sa (step-args t state params)
                  sad (into [diff/no-change sdiff] dparams)
                  k (element-key key t)
                  r (if (= op :update)
                      (update-at kernel old sa sad (cm/get-submap cs t cm/EMPTY) k [t])
                      (regenerate-at kernel old sa sad (sel/get-subselection cs t) k [t]))]
              (recur (inc t) (:retval (:trace r)) (:retdiff r) (put-update acc t old r)))

            :else
            (let [t' (min (next-target t) kept)]
              (recur (long t') (nth (:retval acc) (dec t')) diff/no-change acc))))
        acc (drop-elements acc kept op)
        acc (loop [t kept state state acc acc]
              (if (< t n)
                (let [sa (step-args t state params)
                      k (element-key key t)
                      [sub w] (if (= op :update)
                                (let [r (generate-at kernel sa (cm/get-submap cs t cm/EMPTY) k [t])] [(:trace r) (:weight r)])
                                [(simulate-at kernel sa k) 0.0])]
                  (recur (inc t) (:retval sub) (put-element acc t sub nil w)))
                acc))]
    (incremental-result gf (vec args) acc op n prev-n)))

(defn- unfold-assess [gf [n init & params] choices key]
  (let [n (check-steps n)]
    (indices :assess choices n)
    ;; weight starts boxed: it accumulates arrays on array backends
    (loop [t 0 state init weight (Double/valueOf 0.0) retval []]
      (if (< t n)
        (let [r (assess-at (:kernel gf) (step-args t state params) (cm/get-submap choices t cm/EMPTY) (element-key key t) [t])]
          (recur (inc t) (:retval r) (+ weight (:weight r)) (conj retval (:retval r))))
        {:weight weight :retval retval}))))

(gfi/defgf UnfoldGF [kernel]
  gfi/IStores
  (-stores [gf] (gfi/stores (:kernel gf)))
  (-with-stores [gf stores] (assoc gf :kernel (gfi/with-stores (:kernel gf) stores)))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (unfold-generate gf args nil key)))
  (-generate [gf args constraints key] (unfold-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (unfold-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (unfold-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (unfold-assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (unfold-generate gf args nil key))))
  (-project [_ tr selection] (project-elements tr selection))
  trace/ICalls
  (-call [_ tr p] (element-call tr p)))

(defn unfold
  "Gen.jl's Unfold combinator: a state-space chain of n kernel steps."
  [kernel]
  (->UnfoldGF kernel))

;; ---------------------------------------------------------------------------
;; Switch
;; ---------------------------------------------------------------------------

(declare own-constraints)

(defn- branch-index
  "The branch index as a host value: a one-element array is read. (A
   batched one runs by partition: index-keys.)"
  [idx]
  (if (array/backend-array? idx) (array/item idx) idx))

(defn- branch [gf k]
  (or (get (:branches gf) k)
      (throw (ex-info (str "Switch has no branch " (pr-str k)) {:index k :branches (count (:branches gf))}))))

(defn- switch-trace [gf args k sub]
  (assoc (trace/make-trace {:gen-fn gf :args args :choices (:choices sub) :retval (:retval sub) :score (:score sub)})
         ::branch k
         ::subtrace sub))

(defn- generate-branch
  "generate (constraints, or nil to simulate) of a Switch on args by its
   branch k."
  [gf args k constraints key]
  (let [b (branch gf k)
        bargs (vec (rest args))]
    (if constraints
      (let [{sub :trace w :weight} (generate-at b bargs (own-constraints constraints array/*backend*) key nil)]
        {:trace (switch-trace gf args k sub) :weight w})
      {:trace (switch-trace gf args k (simulate-at b bargs key)) :weight 0.0})))

(defn- revise-branch
  "update or regenerate of a Switch trace by the branch k: the branch at the
   new index runs against the old branch's trace, another branch by
   gfi/revise, so the addresses both make keep their values (Def 2.3.1), as
   the equivalent if would and as Gen.jl does."
  [gf tr args argdiffs op cs key k]
  (let [bargs (vec (rest args))
        b (branch gf k)
        sub (::subtrace tr)
        bdiffs (if (= k (::branch tr)) (vec (rest argdiffs)) (unknown-diffs bargs))
        r (if (= op :update)
            (update-at b sub bargs bdiffs (own-constraints cs array/*backend*) key nil)
            (regenerate-at b sub bargs bdiffs cs key nil))]
    (assoc r :trace (switch-trace gf args k (:trace r)))))

;; A Switch whose index differs across the particles of a batched run runs
;; by partition (DESIGN.md §9): the particles are grouped by branch, each
;; branch runs batched over its group under a part's backend, which draws
;; for a particle what the batch's own run would, and the results are
;; scattered back (gen.partition). The trace holds each group's branch, rows
;; and subtrace (::parts, a gen.array.batch/Parts), and its choices by
;; group, a Parts value at the call's address, which gen.batched reads per
;; particle. An update or regenerate runs together each old group's
;; particles that go to one branch (a cell), against their old subtraces,
;; and joins a branch's cells (gen.partition/merge-groups). Every particle
;; in one branch is the plain trace.

(defn- index-keys
  "The branch indices of the particles of the batch in effect, host values
   (as branch-index reads one), from a batched index."
  [idx]
  (let [s (array/shape idx)
        v (batch/value idx)]
    (when-not (== 1 (reduce * 1 s))
      (throw (ex-info (str "A Switch's index is one value, not an array of shape " s) {:shape s})))
    (array/->clj (array/reshape v [(first (array/shape v))]))))

(defn- one-key? [ks] (every? #(= (first ks) %) ks))

(defn- choices-by-part
  "The choices of groups whose data are choice maps: a Parts value, or EMPTY
   when there are none."
  [groups]
  (if (every? (comp cm/empty? :data) groups) cm/EMPTY (cm/->Value (batch/->Parts groups))))

(defn- parts-trace
  "The trace of a Switch on args whose particles ran in groups ({:key :rows
   :data}, data the group's subtrace), with its score and return value."
  [gf args groups score retval]
  (assoc (trace/make-trace {:gen-fn gf :args args :score score :retval retval
                            :choices (choices-by-part (mapv #(update % :data :choices) groups))})
         ::parts (batch/->Parts groups)))

(defn- scattered-trace
  "parts-trace, its score and return value scattered back from the groups'
   onto the batch backend b."
  [gf args groups b]
  (let [rows (mapv :rows groups)
        subs (mapv :data groups)]
    (parts-trace gf args groups (batch/scatter (mapv :score subs) rows b) (part/scatter (mapv :retval subs) rows b))))

(defn- in-order
  "The order taking particles held at rows to the batch's own: nil when
   they are in it."
  [rows]
  (when-not (= rows (range (count rows)))
    (let [inv (long-array (count rows))]
      (dotimes [j (count rows)] (aset inv (long (rows j)) (long j)))
      (vec inv))))

(defn- own-constraints
  "constraints, for a run of the batch backend b's particles: choices held
   by part (a Parts value: a batched trace's choices given back, as a
   gradient's assess or a rebuild gives them) whose particles are all in
   one group are that group's, taken into b; any other as they are (where
   the particles' choices differ in structure, the branch's run fails,
   loudly)."
  [constraints b]
  (let [v (cm/get-value constraints nil)]
    (if (and (batch/parts? v) (== 1 (count (:groups v))))
      (let [{rows :rows data :data} (first (:groups v))]
        (part/gather-into data (in-order rows) b))
      constraints)))

(defn- in-part
  "(f pb) under pb, the part backend of the particles at rows of the batch
   backend b."
  [b rows f]
  (let [pb (batch/part-backend b rows)]
    (array/with-backend pb (f pb))))

(defn- generate-parts
  "generate (constraints, or nil to simulate) of a Switch on args whose
   index keys ks differ across the particles: each branch on its group."
  [gf args ks constraints key]
  (let [b array/*backend*
        gs (part/groups ks)
        rs (mapv (fn [[k rows]]
                   (in-part b rows
                            (fn [pb]
                              (let [bargs (part/gather-into (vec (rest args)) rows pb)]
                                (if constraints
                                  (generate-at (branch gf k) bargs (own-constraints (part/gather-into constraints rows pb) pb) key nil)
                                  {:trace (simulate-at (branch gf k) bargs key) :weight 0.0})))))
                 gs)]
    {:trace (scattered-trace gf args (mapv (fn [[k rows] r] {:key k :rows rows :data (:trace r)}) gs rs) b)
     :weight (batch/scatter (mapv :weight rs) (mapv second gs) b)}))

(defn- old-groups
  "The groups of a Switch trace of m particles: its ::parts, or its one
   branch's."
  [tr m]
  (if-let [ps (::parts tr)] (:groups ps) [{:key (::branch tr) :rows (vec (range m)) :data (::subtrace tr)}]))

(defn- revise-parts
  "update or regenerate of a Switch trace whose particles' branches differ,
   before or after: each cell, the particles of an old group that go to one
   branch, revised by that branch against their old subtraces (another
   branch's by gfi/revise, as revise-branch), and a branch's cells joined."
  [gf tr args argdiffs op cs key ks]
  (let [b array/*backend*
        bargs (vec (rest args))
        cells (vec (for [{old :key rows :rows data :data} (old-groups tr (count ks))
                         [k places] (part/groups (mapv ks rows))]
                     {:key k :old old :rows (mapv rows places) :data data
                      :places (when-not (= places (range (count rows))) places)}))
        rs (mapv (fn [{k :key old :old rows :rows data :data places :places}]
                   (in-part b rows
                            (fn [pb]
                              (let [prev (part/gather-into data places pb)
                                    ba (part/gather-into bargs rows pb)
                                    bdiffs (if (= k old) (vec (rest argdiffs)) (unknown-diffs bargs))]
                                (if (= op :update)
                                  (update-at (branch gf k) prev ba bdiffs (own-constraints (part/gather-into cs rows pb) pb) key nil)
                                  (regenerate-at (branch gf k) prev ba bdiffs cs key nil))))))
                 cells)
        groups (part/merge-groups (mapv (fn [c r] {:key (:key c) :rows (:rows c) :data (:trace r)}) cells rs) b)
        t (if (== 1 (count groups))
            (let [{k :key rows :rows data :data} (first groups)]
              (switch-trace gf args k (part/gather-into data (in-order rows) b)))
            (scattered-trace gf args groups b))]
    (cond-> {:trace t
             :weight (batch/scatter (mapv :weight rs) (mapv :rows cells) b)
             :retdiff (if (every? #(diff/no-change? (:retdiff %)) rs) diff/no-change diff/unknown-change)}
      (= op :update) (assoc :discard (choices-by-part (mapv (fn [c r] {:key [(:old c) (:key c)] :rows (:rows c) :data (:discard r)})
                                                            cells rs))))))

(defn- switch-generate [gf [idx :as args] constraints key]
  (let [args (vec args)]
    (if (batch/batched? idx)
      (let [ks (index-keys idx)]
        (if (one-key? ks)
          (generate-branch gf args (first ks) constraints key)
          (generate-parts gf args ks constraints key)))
      (generate-branch gf args (branch-index idx) constraints key))))

(defn- switch-revise
  "update or regenerate of a Switch trace: by its branch, or by partition
   when the particles' branches differ, before or after."
  [gf tr [idx :as args] argdiffs op cs key]
  (let [args (vec args)
        ks (when (batch/batched? idx) (index-keys idx))]
    (if (or (::parts tr) (and ks (not (one-key? ks))))
      (revise-parts gf tr args argdiffs op cs key
                    (or ks (vec (repeat (batch/size array/*backend*) (branch-index idx)))))
      (revise-branch gf tr args argdiffs op cs key (if ks (first ks) (branch-index idx))))))

(defn- switch-assess [gf [idx & bargs] choices key]
  (let [bargs (vec bargs)
        ks (when (batch/batched? idx) (index-keys idx))]
    (if (and ks (not (one-key? ks)))
      (let [b array/*backend*
            gs (part/groups ks)
            rs (mapv (fn [[k rows]]
                       (in-part b rows #(gfi/-assess (branch gf k) (part/gather-into bargs rows %)
                                                     (own-constraints (part/gather-into choices rows %) %) key)))
                     gs)]
        {:weight (batch/scatter (mapv :weight rs) (mapv second gs) b)
         :retval (part/scatter (mapv :retval rs) (mapv second gs) b)})
      (gfi/-assess (branch gf (if ks (first ks) (branch-index idx))) bargs (own-constraints choices array/*backend*) key))))

(defn- switch-project [tr selection]
  (if-let [ps (::parts tr)]
    (let [b array/*backend*
          gs (:groups ps)]
      (batch/scatter (mapv (fn [{rows :rows sub :data}]
                             (array/with-backend (part/trace-backend sub (count rows) b)
                               (gfi/-project (:gen-fn sub) sub selection)))
                           gs)
                     (mapv :rows gs) b))
    (let [sub (::subtrace tr)] (gfi/-project (:gen-fn sub) sub selection))))

(defn- switch-join
  "Switch traces of consecutive parts of a batch as one (gen.partition's
   IJoin): their groups merged by branch, the plain trace when every
   particle is in one."
  [gf trs ns b]
  (let [groups (part/merge-groups (part/consecutive (mapv old-groups trs ns) ns) b)
        args (part/join (mapv :args trs) ns b)]
    (if (== 1 (count groups))
      (let [{k :key data :data} (first groups)] (switch-trace gf args k data))
      (parts-trace gf args groups (part/join (mapv :score trs) ns b) (part/join (mapv :retval trs) ns b)))))

(gfi/defgf SwitchGF [branches]
  gfi/IStores
  (-stores [gf] (callee-stores (:branches gf)))
  (-with-stores [gf stores] (update gf :branches with-callee-stores stores))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (switch-generate gf args nil key)))
  (-generate [gf args constraints key] (switch-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (switch-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (switch-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (switch-assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (switch-generate gf args nil key))))
  (-project [_ tr selection] (switch-project tr selection))
  ;; the branch's addresses are the switch's (a trace by partition has no
  ;; one call)
  trace/ICalls
  (-call [_ tr p] (when-let [sub (::subtrace tr)] (trace/-call (trace/maker sub) sub p)))
  part/IJoin
  (-join [gf trs ns b] (switch-join gf trs ns b)))

(defn switch
  "Gen.jl's Switch combinator: (switch gf0 gf1 …) is called with a branch
   index first, (switch {k gf …}) with a branch key first. In a batched run,
   an index that differs across the particles runs each branch on its own
   particles (by partition)."
  [& gfs]
  (if (and (= 1 (count gfs)) (map? (first gfs)) (not (record? (first gfs))))
    (->SwitchGF (first gfs))
    (->SwitchGF (vec gfs))))

;; ---------------------------------------------------------------------------
;; Recurse
;; ---------------------------------------------------------------------------

(defn- child-of
  "The node number of child j of node node, unchecked: Recurse's own
   loops, whose max-branch and number of children are checked already."
  [node j max-branch]
  (clojure.core/+ (* node max-branch) j 1))

(defn- check-numbering
  "Loud unless every number is a natural number, max-branch positive."
  [f max-branch & numbers]
  (when-not (and (pos-int? max-branch) (every? nat-int? numbers))
    (throw (ex-info (str f ": node numbers are natural numbers and max-branch a positive integer")
                    {:numbers (vec numbers) :max-branch max-branch}))))

(defn get-child
  "Gen.jl's get_child, 0-based: the node number of child j (0 to
   max-branch - 1) of node parent, parent·max-branch + j + 1."
  [parent j max-branch]
  (check-numbering "get-child" max-branch parent j)
  (when-not (< j max-branch)
    (throw (ex-info (str "get-child: child " j " of at most " max-branch) {:j j :max-branch max-branch})))
  (child-of parent j max-branch))

(defn- check-not-root [f child]
  (when (zero? child)
    (throw (ex-info (str f ": the root, node 0, has no parent") {:child child}))))

(defn get-child-num
  "Gen.jl's get_child_num, 0-based: which child of its parent node child
   (not the root) is."
  [child max-branch]
  (check-numbering "get-child-num" max-branch child)
  (check-not-root "get-child-num" child)
  (mod (dec child) max-branch))

(defn get-parent
  "Gen.jl's get_parent, 0-based: the parent of node child (not the root)."
  [child max-branch]
  (check-numbering "get-parent" max-branch child)
  (check-not-root "get-parent" child)
  (quot (dec child) max-branch))

(defn- put-part
  "choices with sub at [node part], unless sub is empty."
  [choices node part sub]
  (if (cm/empty? sub) choices (cm/set-choice choices [node part] sub)))

(defn- production-output
  "The production kernel's return value at node, checked."
  [gf node {out :retval}]
  (when-not (and (map? out) (contains? out :value) (sequential? (:children out)))
    (throw (ex-info (str "Recurse's production kernel returns {:value v :children [inputs …]}, not " (pr-str out)
                         " (node " node ")")
                    {:node node :retval out})))
  (when (> (count (:children out)) (:max-branch gf))
    (throw (ex-info (str "Recurse: node " node " has " (count (:children out)) " children, more than max-branch "
                         (:max-branch gf))
                    {:node node :children (count (:children out)) :max-branch (:max-branch gf)})))
  out)

(defn- recurse-targets
  "The nodes the constraints (update, generate) or the selection
   (regenerate) reach: {node #{:production :aggregation}}, or :all."
  [op cs]
  (cond
    (nil? cs) {}
    (= op :regenerate)
    (let [is (selected-indices cs)]
      (if (= :all is) :all (into {} (clojure.core/map (fn [i] [i #{:production :aggregation}])) is)))
    :else
    (do (when (cm/has-value? cs)
          (throw (gfi/unvisited op [[]] "a single value was given where Recurse's nodes belong")))
        (into {} (clojure.core/map (fn [[i sub]]
                                     (when-not (nat-int? i)
                                       (throw (gfi/unvisited op [[i]] "Recurse's addresses are node indices")))
                                     (when (cm/has-value? sub)
                                       (throw (gfi/unvisited op [[i]] (str "a value at node " i
                                                                           ": a node has :production and :aggregation"))))
                                     (let [parts (set (clojure.core/map first (cm/submaps sub)))]
                                       (when-let [bad (seq (remove #{:production :aggregation} parts))]
                                         (throw (gfi/unvisited op (mapv #(vector i %) bad) "a node has :production and :aggregation")))
                                       [i parts])))
              (cm/submaps cs)))))

(defn- targeted? [targets node part]
  (or (= :all targets) (contains? (get targets node) part)))

(defn- node-cs [op cs node part]
  (when cs (if (= op :regenerate) (sel/get-subselection cs [node part]) (cm/get-submap cs [node part] cm/EMPTY))))

(defn- run-node
  "One kernel run at node: revised from old when given (update or
   regenerate), else new (generate, or simulate for regenerate and
   unconstrained runs). {:trace :weight :retdiff :discard}."
  [op kernel old args argdiffs cs key at]
  (cond
    old (if (= op :regenerate)
          (regenerate-at kernel old args argdiffs cs key at)
          (update-at kernel old args argdiffs cs key at))
    (and cs (not= op :regenerate)) (assoc (generate-at kernel args cs key at) :retdiff diff/unknown-change)
    :else {:trace (simulate-at kernel args key) :weight 0.0 :retdiff diff/unknown-change}))

(defn- production-phase
  "The productions from the root, in node order (parents before children):
   a retained node is revisited only when its input may have changed or it
   is targeted. {:prods :p-diffs :weight :discard}."
  [{:keys [gf old-p targets op cs node-key]} input input-diff]
  (let [{:keys [production max-branch]} gf]
    ;; weight starts boxed: it accumulates arrays on array backends
    (loop [todo (sorted-map 0 [input input-diff]) prods {} p-diffs {} weight (Double/valueOf 0.0) discard cm/EMPTY]
      (if-let [[node [in d]] (first todo)]
        (let [old (old-p node)
              revisit? (or (nil? old) (diff/changed? d) (targeted? targets node :production))
              r (if revisit?
                  (run-node op production old [in] [d] (node-cs op cs node :production) (node-key node :production)
                            [node :production])
                  {:trace old :weight 0.0 :retdiff diff/no-change})
              out (production-output gf node (:trace r))
              child-diff (if (diff/changed? (:retdiff r)) diff/unknown-change diff/no-change)]
          (recur (into (dissoc todo node)
                       (map-indexed (fn [j c] [(child-of node j max-branch) [c child-diff]]) (:children out)))
                 (assoc prods node (:trace r)) (assoc p-diffs node (:retdiff r)) (+ weight (:weight r))
                 (put-part discard node :production (:discard r))))
        {:prods prods :p-diffs p-diffs :weight weight :discard discard}))))

(defn- aggregation-phase
  "The aggregations from the leaves (children before parents): a retained
   node is revisited when its value, its children or its target may have
   changed. {:aggs :a-diffs :weight :discard}, the weight and discard
   continuing the production phase's."
  [{:keys [gf old-a targets op cs node-key]} {:keys [prods p-diffs weight discard]}]
  (let [{:keys [aggregation max-branch]} gf]
    (reduce (fn [{:keys [aggs a-diffs weight discard]} node]
              (let [{v :value children :children} (:retval (prods node))
                    kids (clojure.core/map #(child-of node % max-branch) (range (count children)))
                    old (old-a node)
                    v-changed? (diff/changed? (p-diffs node))
                    ws-changed? (or (nil? old) (not= (count children) (count (second (:args old))))
                                    (some #(diff/changed? (a-diffs %)) kids))
                    r (if (or (nil? old) v-changed? ws-changed? (targeted? targets node :aggregation))
                        (run-node op aggregation old [v (mapv #(:retval (aggs %)) kids)]
                                  [(if v-changed? diff/unknown-change diff/no-change)
                                   (if ws-changed? diff/unknown-change diff/no-change)]
                                  (node-cs op cs node :aggregation) (node-key node :aggregation) [node :aggregation])
                        {:trace old :weight 0.0 :retdiff diff/no-change})]
                {:aggs (assoc aggs node (:trace r)) :a-diffs (assoc a-diffs node (:retdiff r))
                 :weight (+ weight (:weight r)) :discard (put-part discard node :aggregation (:discard r))}))
            {:aggs {} :a-diffs {} :weight weight :discard discard}
            (sort > (keys prods)))))

(defn- node-choices [choices node p a]
  (-> choices (put-part node :production (:choices p)) (put-part node :aggregation (:choices a))))

(defn- node-key [key node part] (element-key (element-key key node) (if (= part :production) 0 1)))

(defn- recurse-run
  "generate (tr nil), update or regenerate of a Recurse trace: the
   production phase, then the aggregation phase. Nodes no longer reached
   are dropped: update discards them and subtracts their scores; in
   regenerate they cancel against the reverse move."
  [gf tr [input] [input-diff] op cs key]
  (let [ctx {:gf gf :old-p (::productions tr {}) :old-a (::aggregations tr {}) :targets (recurse-targets op cs)
             :op op :cs cs
             :node-key (partial node-key key)}
        {:keys [prods] :as produced} (production-phase ctx input input-diff)
        _ (when (and cs (not= op :regenerate))
            (doseq [node (keys (:targets ctx))]
              (when-not (contains? prods node)
                (throw (gfi/unvisited op [[node]] (str "the tree has nodes " (vec (sort (keys prods)))))))))
        {:keys [aggs a-diffs weight discard]} (aggregation-phase ctx produced)
        {:keys [old-p old-a]} ctx
        gone (remove prods (keys old-p))
        [weight discard] (if (= op :update)
                           [(reduce (fn [w node] (- w (+ (:score (old-p node)) (:score (old-a node))))) weight gone)
                            (reduce #(node-choices %1 %2 (old-p %2) (old-a %2)) discard gone)]
                           [weight discard])
        nodes (sort (keys prods))
        t (assoc (trace/make-trace {:gen-fn gf :args [input]
                                    :choices (reduce #(node-choices %1 %2 (prods %2) (aggs %2)) cm/EMPTY nodes)
                                    :retval (:retval (aggs 0))
                                    :score (reduce + 0.0 (concat (clojure.core/map :score (vals prods))
                                                                 (clojure.core/map :score (vals aggs))))})
                 ::productions prods ::aggregations aggs)]
    (cond-> {:trace t :weight weight}
      (not= op :generate) (assoc :retdiff (if tr (a-diffs 0) diff/unknown-change))
      (= op :update) (assoc :discard discard))))

(defn- recurse-args [args]
  (when-not (= 1 (count args))
    (throw (ex-info (str "Recurse takes one argument, the root's input; given " (count args)) {:args args})))
  (vec args))

(defn- recurse-assess [gf [input] choices key]
  (let [{:keys [production aggregation max-branch]} gf
        targets (recurse-targets :assess choices)
        [ps weight] (loop [todo (sorted-map 0 input) ps {} weight (Double/valueOf 0.0)]
                      (if-let [[node in] (first todo)]
                        (let [r (assess-at production [in] (cm/get-submap choices [node :production] cm/EMPTY)
                                           (node-key key node :production) [node :production])
                              out (production-output gf node r)]
                          (recur (into (dissoc todo node)
                                       (map-indexed (fn [j c] [(child-of node j max-branch) c]) (:children out)))
                                 (assoc ps node out) (+ weight (:weight r))))
                        [ps weight]))
        _ (doseq [node (keys targets)]
            (when-not (contains? ps node)
              (throw (gfi/unvisited :assess [[node]] (str "the tree has nodes " (vec (sort (keys ps))))))))
        [as weight] (reduce (fn [[as weight] node]
                              (let [{v :value children :children} (ps node)
                                    ws (mapv #(as (child-of node % max-branch)) (range (count children)))
                                    r (assess-at aggregation [v ws] (cm/get-submap choices [node :aggregation] cm/EMPTY)
                                                 (node-key key node :aggregation) [node :aggregation])]
                                [(assoc as node (:retval r)) (+ weight (:weight r))]))
                            [{} weight] (sort > (keys ps)))]
    {:weight weight :retval (as 0)}))

(defn- recurse-project [tr s]
  (let [part (fn [subs part]
               (reduce (fn [w [node sub]]
                         (+ w (gfi/-project (:gen-fn sub) sub (sel/get-subselection s [node part]))))
                       0.0 subs))]
    (+ (part (::productions tr) :production) (part (::aggregations tr) :aggregation))))

(gfi/defgf RecurseGF [production aggregation max-branch]
  gfi/IStores
  (-stores [gf] (callee-stores (select-keys gf [:production :aggregation])))
  (-with-stores [gf stores] (with-callee-stores gf stores))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (recurse-run gf nil (recurse-args args) [diff/unknown-change] :generate nil key)))
  (-generate [gf args constraints key] (recurse-run gf nil (recurse-args args) [diff/unknown-change] :generate constraints key))
  (-update [gf tr args argdiffs constraints key] (recurse-run gf tr (recurse-args args) argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (recurse-run gf tr (recurse-args args) argdiffs :regenerate selection key))
  (-assess [gf args choices key] (recurse-assess gf (recurse-args args) choices key))
  (-propose [gf args key] (propose-from-trace (gfi/-simulate gf args key)))
  (-project [_ tr selection] (recurse-project tr selection))
  ;; node i's production at [i :production], its aggregation at [i :aggregation]
  trace/ICalls
  (-call [_ tr p]
    (when (<= 2 (count p))
      (when-let [sub (get (case (p 1) :production (::productions tr) :aggregation (::aggregations tr) nil) (p 0))]
        [sub (subvec p 2)]))))

(defn recurse
  "Gen.jl's Recurse combinator (see the namespace doc): a tree grown by the
   production kernel and folded by the aggregation kernel, each node with at
   most max-branch children."
  [production aggregation max-branch]
  (when-not (pos-int? max-branch)
    (throw (ex-info "Recurse's max-branch is a positive integer" {:max-branch max-branch})))
  (->RecurseGF production aggregation max-branch))

;; ---------------------------------------------------------------------------
;; Printing
;; ---------------------------------------------------------------------------

(defmethod print-method RecurseGF [_ ^java.io.Writer w]
  (.write w "#gen/recurse"))

(defmethod print-method MapGF [gf ^java.io.Writer w]
  (.write w "#gen/map[") (print-method (:kernel gf) w) (.write w "]"))

(defmethod print-method UnfoldGF [gf ^java.io.Writer w]
  (.write w "#gen/unfold[") (print-method (:kernel gf) w) (.write w "]"))

(defmethod print-method SwitchGF [gf ^java.io.Writer w]
  (.write w "#gen/switch") (print-method (:branches gf) w))

;; ---------------------------------------------------------------------------
;; call-at, choice-at
;; ---------------------------------------------------------------------------

(defn call-at
  "Gen.jl's call_at: gf with an address as its last argument, gf's choices
   under it. ((call-at gf) a b k) calls (gf a b), its choices at k. A dynamic
   generative function splicing gf, so update, regenerate and project are
   the handler's own."
  [gf]
  (when-not (gfi/generative-function? gf)
    (throw (ex-info "call-at needs a generative function" {:value gf})))
  (dyn/gen [& args]
    (apply splice (last args) gf (butlast args))))

(defn choice-at
  "Gen.jl's choice_at: a distribution constructor with an address as its
   last argument, one choice there. ((choice-at d/normal) 0 1 :x) is the
   value of (trace :x (d/normal 0 1))."
  [dist]
  (when-not (fn? dist)
    (throw (ex-info "choice-at needs a distribution constructor (d/normal, not (d/normal 0 1))" {:value dist})))
  (dyn/gen [& args]
    (trace (last args) (apply dist (butlast args)))))

;; ---------------------------------------------------------------------------
;; Scan
;; ---------------------------------------------------------------------------

(defn- scan-inputs
  "Scan's inputs, its second argument, as a vector of one per step."
  [args]
  (when (< (count args) 2)
    (throw (ex-info (str "Scan takes (init xs & params), the initial carry and the inputs; given " (count args)
                         " arguments")
                    {:args (count args)})))
  (let [xs (second args)]
    (when-not (sequential? xs)
      (throw (ex-info (str "Scan's second argument is its inputs, a vector of one per step, not " (pr-str xs))
                      {:xs xs})))
    (vec xs)))

(defn- step-output
  "Step t's kernel return value, checked to be [carry y]."
  [t out]
  (when-not (and (vector? out) (== 2 (count out)))
    (throw (ex-info (str "Scan's kernel returns [carry y], not " (pr-str out) " (step " t ")") {:step t :retval out})))
  out)

(defn- put-step
  "acc with step t's run r ({:trace :weight}, and update's :discard) in
   place of old (if given): its output y among the outputs, y's diff yd."
  [acc t old {sub :trace w :weight d :discard} y yd]
  (cond-> (assoc-in (put-element acc t sub old w) [:retval t] y)
    (and d (not (cm/empty? d))) (update :discard cm/set-submap t d)
    (diff/changed? yd) (assoc-in [:retdiffs t] yd)))

(defn- fresh-steps
  "[carry acc]: acc with the steps from from on run afresh after carry,
   generated (op :generate with constraints cs, or :update) or simulated."
  [kernel xs params from carry acc op cs key]
  (loop [t from carry carry acc acc]
    (if (< t (count xs))
      (let [sa (into [carry (xs t)] params)
            k (element-key key t)
            r (if (or (= op :update) (and cs (= op :generate)))
                (generate-at kernel sa (cm/get-submap cs t cm/EMPTY) k [t])
                {:trace (simulate-at kernel sa k) :weight 0.0})
            [c y] (step-output t (:retval (:trace r)))]
        (recur (inc t) c (put-step acc t nil r y diff/no-change)))
      [carry acc])))

(defn- scan-trace [gf args carry acc]
  (elements-trace gf args (assoc acc :retval [carry (:retval acc)])))

(defn- scan-generate [gf [init _ & params :as args] constraints key]
  (let [xs (scan-inputs args)]
    (when constraints (indices :generate constraints (count xs)))
    (let [[carry acc] (fresh-steps (:kernel gf) xs params 0 init empty-acc :generate constraints key)]
      {:trace (scan-trace gf (vec args) carry acc) :weight (:weight acc)})))

(defn- changed-steps
  "The steps whose input may have changed, by the inputs' argdiff d: a
   sorted set, or :all."
  [d]
  (cond (diff/no-change? d) (sorted-set)
        (diff/vector-diff? d) (into (sorted-set) (keys (:updated d)))
        :else :all))

(defn- among? [steps t] (or (= :all steps) (contains? steps t)))

(defn- next-among
  "The first of steps (a sorted set, or :all) from t on, else end."
  [steps t end]
  (if (= :all steps) t (or (first (subseq steps >= t)) end)))

(defn- part-diff
  "The diff of part i (0 the carry, 1 the output) of a step's [carry y],
   prev before and now after, from the kernel's retdiff rd: the part's own
   (from a vector-diff), or no-change when it is the same value
   (gfi/same-value?), so that a changed output alone leaves the next step
   be."
  [rd prev now i]
  (let [d (element-diff rd i)]
    (if (and (diff/changed? d) (gfi/same-value? (nth prev i) (nth now i))) diff/no-change d)))

(defn- scan-retdiff
  "The diff of [carry ys] from its parts'."
  [carry-diff ys-diff]
  (if (and (diff/no-change? carry-diff) (diff/no-change? ys-diff))
    diff/no-change
    (diff/vector-diff 2 2 (cond-> {}
                            (diff/changed? carry-diff) (assoc 0 carry-diff)
                            (diff/changed? ys-diff) (assoc 1 ys-diff)))))

(defn- scan-revise
  "update or regenerate of a Scan trace, as Unfold's: walks the retained
   steps, revisiting a step when its carry, its input, the parameters, its
   constraints or its selection may have changed, and skipping straight to
   the next such step otherwise; then extends or truncates."
  [gf tr [init _ & params :as args] argdiffs op cs key]
  (let [xs (scan-inputs args)
        n (count xs)
        kernel (:kernel gf)
        [dinit dxs & dparams] argdiffs
        ;; a change in their number is a change (DESIGN.md §9, G3)
        params-changed? (or (not= (count args) (count (:args tr))) (some diff/changed? dparams))
        old-subs (::subtraces tr)
        prev-n (count old-subs)
        kept (min n prev-n)
        targets (if (= op :update) (indices :update cs n) (selected-indices cs))
        inputs (changed-steps dxs)
        [carry cdiff acc]
        (loop [t 0 carry init cdiff dinit acc (assoc (trace-acc tr) :retval (second (:retval tr)))]
          (cond
            (>= t kept) [carry cdiff acc]

            (or params-changed? (diff/changed? cdiff) (among? inputs t) (among? targets t))
            (let [old (old-subs t)
                  sa (into [carry (xs t)] params)
                  sad (into [cdiff (element-diff dxs t)] dparams)
                  k (element-key key t)
                  r (if (= op :update)
                      (update-at kernel old sa sad (cm/get-submap cs t cm/EMPTY) k [t])
                      (regenerate-at kernel old sa sad (sel/get-subselection cs t) k [t]))
                  [c y :as out] (step-output t (:retval (:trace r)))
                  rd (:retdiff r)]
              (recur (inc t) c (part-diff rd (:retval old) out 0)
                     (put-step acc t old r y (part-diff rd (:retval old) out 1))))

            :else
            (let [t' (long (min (next-among targets t kept) (next-among inputs t kept) kept))]
              (recur t' (first (:retval ((:subs acc) (dec t')))) diff/no-change acc))))
        [carry acc] (fresh-steps kernel xs params kept carry (drop-elements acc kept op) op cs key)
        ;; the last carry is another step's when the length changed
        carry-diff (cond (== n prev-n) cdiff
                         (gfi/same-value? (first (:retval tr)) carry) diff/no-change
                         :else diff/unknown-change)]
    (cond-> {:trace (scan-trace gf (vec args) carry acc) :weight (:weight acc)
             :retdiff (scan-retdiff carry-diff (retdiff acc n prev-n))}
      (= op :update) (assoc :discard (:discard acc)))))

(defn- scan-assess [gf [init _ & params :as args] choices key]
  (let [xs (scan-inputs args)]
    (indices :assess choices (count xs))
    ;; weight starts boxed: it accumulates arrays on array backends
    (loop [t 0 carry init weight (Double/valueOf 0.0) ys []]
      (if (< t (count xs))
        (let [r (assess-at (:kernel gf) (into [carry (xs t)] params) (cm/get-submap choices t cm/EMPTY)
                           (element-key key t) [t])
              [c y] (step-output t (:retval r))]
          (recur (inc t) c (+ weight (:weight r)) (conj ys y)))
        {:weight weight :retval [carry ys]}))))

(gfi/defgf ScanGF [kernel]
  gfi/IStores
  (-stores [gf] (gfi/stores (:kernel gf)))
  (-with-stores [gf stores] (assoc gf :kernel (gfi/with-stores (:kernel gf) stores)))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (scan-generate gf args nil key)))
  (-generate [gf args constraints key] (scan-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (scan-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (scan-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (scan-assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (scan-generate gf args nil key))))
  (-project [_ tr selection] (project-elements tr selection))
  trace/ICalls
  (-call [_ tr p] (element-call tr p)))

(defn- check-gen-fn [what gf]
  (when-not (gfi/generative-function? gf)
    (throw (ex-info (str what " needs a generative function, not " (pr-str gf)) {:value gf}))))

(defn scan
  "JAX's scan as a combinator (GenMLX's Scan): Unfold with inputs and
   outputs. ((scan kernel) init xs & params) runs (kernel carry x & params)
   for each x of xs in order, carry init at the first step and the carry
   the previous step returned after; the kernel returns [carry' y]. Returns
   [carry ys]: the last carry, and the vector of the ys."
  [kernel]
  (check-gen-fn "scan" kernel)
  (->ScanGF kernel))

;; ---------------------------------------------------------------------------
;; Mix
;; ---------------------------------------------------------------------------

(defn- mix-body
  "Mix's model, a dynamic function: the component index a choice at :index,
   from (categorical weights) (or categorical-logits), and the component a
   Switch's branch, its choices under :component."
  [components logits?]
  (let [sw (->SwitchGF components)
        n (count components)
        index (if logits? d/categorical-logits d/categorical)]
    (dyn/gen [weights & args]
      (let [i (index weights)
            batch (dc/value-shape i)
            k (count (dc/support i))]
        (when-not (and (empty? batch) (== n k))
          (throw (ex-info (str "Mix's weights are a vector of one per component (" n "), not " k
                               (when (seq batch) (str " in a batch of shape " batch)))
                          {:components n :weights k :batch batch})))
        (apply splice :component sw (trace :index i) args)))))

(declare mix-of)

(defn- as-mix [r gf] (assoc r :trace (trace/as-made-by (:trace r) gf)))

;; Its traces are its body's, as the Mix's own (trace/as-made-by); its
;; parameter stores are the components', as a Switch's are its branches'
(gfi/defgf MixGF [components logits? body]
  gfi/IStores
  (-stores [_] (callee-stores components))
  (-with-stores [_ stores] (mix-of (with-callee-stores components stores) logits?))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (trace/as-made-by (gfi/-simulate body args key) gf))
  (-generate [gf args constraints key] (as-mix (gfi/-generate body args constraints key) gf))
  (-update [gf tr args argdiffs constraints key]
    (as-mix (gfi/-update body (trace/as-made-by tr body) args argdiffs constraints key) gf))
  (-regenerate [gf tr args argdiffs selection key]
    (as-mix (gfi/-regenerate body (trace/as-made-by tr body) args argdiffs selection key) gf))
  (-assess [_ args choices key] (gfi/-assess body args choices key))
  (-propose [_ args key] (gfi/-propose body args key))
  (-project [_ tr selection] (gfi/-project body tr selection))
  trace/ICalls
  (-call [_ tr p] (trace/-call body tr p))
  ;; the body follows from the components
  gfi/IRevisable
  (-same-gen-fn? [_ other budget]
    (and (= logits? (:logits? other)) (gfi/same-value? components (:components other) budget)))
  (-revise [gf op args opts project-old] (as-mix (gfi/-revise body op args opts project-old) gf)))

(defn- mix-of [components logits?]
  (let [components (vec components)]
    (when (empty? components)
      (throw (ex-info "Mix needs at least one component" {})))
    (run! #(check-gen-fn "mix" %) components)
    (->MixGF components logits? (mix-body components logits?))))

(defn mix
  "GenMLX's Mix, a mixture of generative functions: ((mix gf0 gf1 …)
   weights & args) draws an index i from (categorical weights), at :index,
   and calls component i on args, its choices under :component; it returns
   the component's return value. weights are the components' probabilities
   (mix-logits takes logits). A categorical and a Switch: the same model as
     (gen [weights & args]
       (apply splice :component (switch gf0 gf1 …)
              (trace :index (categorical weights)) args))"
  [& components]
  (mix-of components false))

(defn mix-logits
  "mix, its weights logits: the index is (categorical-logits weights)."
  [& components]
  (mix-of components true))

;; ---------------------------------------------------------------------------
;; dimap, contramap, map-retval
;; ---------------------------------------------------------------------------

(defn- inner-args
  "The inner's arguments: (apply pre args) as a vector, args without pre."
  [{:keys [pre]} args]
  (if pre
    (let [a (apply pre args)]
      (when-not (sequential? a)
        (throw (ex-info (str "dimap's argument map returns the inner's arguments, a vector, not " (pr-str a))
                        {:value a})))
      (vec a))
    args))

(defn- outer-retval [{:keys [post]} v] (if post (post v) v))

(defn- dimap-trace [gf args sub retval]
  (assoc (trace/make-trace {:gen-fn gf :args args :choices (:choices sub) :retval retval :score (:score sub)})
         ::subtrace sub))

(defn- dimap-generate [gf args constraints key]
  (let [args (vec args)
        ia (inner-args gf args)
        {sub :trace :as r} (if constraints
                             (generate-at (:inner gf) ia constraints key nil)
                             {:trace (simulate-at (:inner gf) ia key) :weight 0.0})]
    (assoc r :trace (dimap-trace gf args sub (outer-retval gf (:retval sub))))))

(def ^:private absent (Object.))

(defn- inner-call
  "[arguments argdiffs] of the inner's call, from args and argdiffs. Without
   an argument map, the same. With one, which change hints cannot see
   through: when no argument changed (nor their number), the previous
   call's arguments, unchanged; otherwise the mapped ones, each no-change
   when it is identical to the previous call's, else unknown-change."
  [gf tr args argdiffs]
  (let [prev (:args (::subtrace tr))]
    (cond (nil? (:pre gf)) [args argdiffs]
          (and (== (count args) (count (:args tr))) (every? diff/no-change? argdiffs)) [prev (diff/no-changes (count prev))]
          :else (let [ia (inner-args gf args)]
                  [ia (vec (map-indexed (fn [i a] (if (identical? a (nth prev i absent)) diff/no-change diff/unknown-change))
                                        ia))]))))

(defn- dimap-revise
  "update or regenerate of a dimap trace: the inner's (inner-call). The
   return value is mapped again only when the inner's may have changed,
   and is no-change when it is the same value (gfi/same-value?)."
  [gf tr args argdiffs op cs key]
  (let [args (vec args)
        sub (::subtrace tr)
        [ia iad] (inner-call gf tr args argdiffs)
        {sub' :trace rd :retdiff :as r} (if (= op :update)
                                          (update-at (:inner gf) sub ia iad cs key nil)
                                          (regenerate-at (:inner gf) sub ia iad cs key nil))
        [v vd] (cond (nil? (:post gf)) [(:retval sub') rd]
                     (diff/no-change? rd) [(:retval tr) diff/no-change]
                     :else (let [v (outer-retval gf (:retval sub'))]
                             [v (if (gfi/same-value? (:retval tr) v) diff/no-change diff/unknown-change)]))]
    (assoc r :trace (dimap-trace gf args sub' v) :retdiff vd)))

(gfi/defgf DimapGF [inner pre post]
  gfi/IStores
  (-stores [gf] (gfi/stores (:inner gf)))
  (-with-stores [gf stores] (assoc gf :inner (gfi/with-stores (:inner gf) stores)))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (dimap-generate gf args nil key)))
  (-generate [gf args constraints key] (dimap-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (dimap-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (dimap-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key]
    (update (assess-at (:inner gf) (inner-args gf (vec args)) choices key nil) :retval #(outer-retval gf %)))
  (-propose [gf args key] (propose-from-trace (:trace (dimap-generate gf args nil key))))
  (-project [_ tr selection] (let [sub (::subtrace tr)] (gfi/-project (:gen-fn sub) sub selection)))
  ;; the inner's addresses are the wrapper's
  trace/ICalls
  (-call [_ tr p] (let [sub (::subtrace tr)] (trace/-call (trace/maker sub) sub p))))

(defn- check-fn [what f]
  (when-not (ifn? f)
    (throw (ex-info (str what " needs a function, not " (pr-str f)) {:value f}))))

(defn dimap
  "gf with its arguments and its return value mapped (GenJAX's and
   GenMLX's dimap): called on args, it calls gf on (apply f args), a vector
   of gf's arguments, and returns (g v), v gf's return value. Its choices
   are gf's, at gf's addresses. f and g are pure functions; f is opaque to
   change hints, so an argument it maps changes unless it is identical to
   the previous call's (or no argument changed)."
  [gf f g]
  (check-gen-fn "dimap" gf)
  (check-fn "dimap" f)
  (check-fn "dimap" g)
  (->DimapGF gf f g))

(defn contramap
  "gf with its arguments mapped: called on args, it calls gf on (apply f
   args), a vector of gf's arguments (dimap without the return map)."
  [gf f]
  (check-gen-fn "contramap" gf)
  (check-fn "contramap" f)
  (->DimapGF gf f nil))

(defn map-retval
  "gf with its return value mapped: it returns (g v), v gf's return value
   (dimap without the argument map)."
  [gf g]
  (check-gen-fn "map-retval" gf)
  (check-fn "map-retval" g)
  (->DimapGF gf nil g))

(defmethod print-method ScanGF [gf ^java.io.Writer w]
  (.write w "#gen/scan[") (print-method (:kernel gf) w) (.write w "]"))

(defmethod print-method MixGF [gf ^java.io.Writer w]
  (.write w (if (:logits? gf) "#gen/mix-logits" "#gen/mix")) (print-method (:components gf) w))

(defmethod print-method DimapGF [gf ^java.io.Writer w]
  (.write w "#gen/dimap[") (print-method (:inner gf) w) (.write w "]"))
