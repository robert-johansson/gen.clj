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
                        i·max-branch + j + 1. Returns the root's aggregation.
     (call-at gf)       Gen.jl's call_at: args (& args k) call gf on args,
                        its choices under address k.
     (choice-at dist)   Gen.jl's choice_at: args (& args k), one choice at
                        address k from (dist & args).

   Addresses and indices are 0-based (DESIGN.md §9). A trace keeps each
   element's or step's subtrace, so update and regenerate are incremental, as
   in Gen.jl:
   - Map revisits only the elements whose arguments may have changed
     (vector-diff argdiffs), or that are constrained or selected.
   - Unfold revisits a step only when its state input (the previous step's
     retdiff), its parameters, its constraints or its selection may have
     changed.
   - Elements and steps beyond a changed length are generated, or discarded.
   - The retdiff is a vector-diff of the returned vector.
   A Switch whose index changes runs the new branch against the old branch's
   choices when the branches are compatible (two dynamic generative
   functions, two distributions, or two combinators of one type), so the
   addresses they share keep their values, as an if would and as Gen.jl does.
   Branches of different kinds share no addresses: the new one is generated
   from the constraints and every old choice is discarded.

   Element or step i draws from (fold-in key i), so changing the length never
   shifts another element's randomness."
  (:refer-clojure :exclude [map + -])
  (:require [gen.array :as array :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
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
  (try (gfi/-assess gf args choices key)
       (catch clojure.lang.ExceptionInfo e (throw (gfi/under e at)))))
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
    (throw (ex-info (str (name op) ": a single value was given where indexed choices belong")
                    {:op op})))
  (let [addrs (clojure.core/map first (cm/submaps choices))]
    (when-let [outside (seq (remove #(and (integer? %) (< -1 % n)) addrs))]
      (throw (gfi/unvisited op (mapv vector outside) (str "the indices are 0 to " (dec n)))))
    (into (sorted-set) (clojure.core/map long) addrs)))

(defn- selected-indices
  "The indices a selection may select something under: :all, or a sorted set."
  [s]
  (cond (instance? gen.selection.None s) (sorted-set)
        (instance? gen.selection.Hierarchical s) (into (sorted-set) (filter integer?) (keys (:m s)))
        :else :all))

(defn- elements-trace [gf args {:keys [subs choices retval score]}]
  (assoc (trace/make-trace {:gen-fn gf :args args :choices choices :retval retval :score score})
         ::subtraces subs))

(def ^:private empty-acc
  {:subs [] :choices cm/EMPTY :retval [] :score 0.0 :weight 0.0 :discard cm/EMPTY :retdiffs {}})

(defn- trace-acc
  "An accumulator holding the elements of tr, to be changed in place."
  [tr]
  (assoc empty-acc :subs (::subtraces tr) :choices (:choices tr) :retval (:retval tr) :score (:score tr)))

(defn- put-element
  "acc with sub as element i (replacing old, if given), its weight w added."
  [acc i sub old w]
  (-> acc
      (assoc-in [:subs i] sub)
      (update :choices assoc-element i (:choices sub))
      (assoc-in [:retval i] (:retval sub))
      (update :score #(cond-> (+ % (:score sub)) old (- (:score old))))
      (update :weight + w)))

(defn- put-update
  "acc with element i updated: the result of update or regenerate."
  [acc i old {sub :trace w :weight d :discard rd :retdiff}]
  (cond-> (put-element acc i sub old w)
    (and d (not (cm/empty? d))) (update :discard cm/set-submap i d)
    (diff/changed? rd) (assoc-in [:retdiffs i] rd)))

(defn- drop-elements
  "acc without the elements from n on. update discards them and subtracts
   their scores from the weight; in regenerate they cancel against the
   reverse move."
  [acc n op]
  (let [gone (subvec (:subs acc) n)]
    (-> (reduce (fn [acc [i sub]]
                  (cond-> (-> acc
                              (update :choices cm/remove-submap i)
                              (update :score - (:score sub)))
                    (= op :update) (-> (update :weight - (:score sub))
                                       (cond-> (not (cm/empty? (:choices sub)))
                                         (update :discard cm/set-submap i (:choices sub))))))
                acc
                (clojure.core/map vector (range n (count (:subs acc))) gone))
        (update :subs subvec 0 n)
        (update :retval subvec 0 n))))

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

(defn- element-argdiffs
  "The kernel's argdiffs for element i, from Map's argdiffs."
  [argdiffs i]
  (mapv #(cond (diff/no-change? %) diff/no-change
               (diff/vector-diff? %) (get (:updated %) i diff/no-change)
               :else diff/unknown-change)
        argdiffs))

(defn- changed-elements
  "The indices whose arguments may have changed: :all, or a sorted set."
  [argdiffs]
  (if (every? #(or (diff/no-change? %) (diff/vector-diff? %)) argdiffs)
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
                            (let [{sub :trace w :weight} (generate-at kernel ea (cm/get-submap constraints i) k [i])]
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
                                      (update-at kernel old ea ead (cm/get-submap cs i) k [i])
                                      (regenerate-at kernel old ea ead (sel/get-subselection cs i) k [i])))))
                    (trace-acc tr)
                    (visits (changed-elements argdiffs) targets kept))
        acc (drop-elements acc kept op)
        acc (reduce (fn [acc i]
                      (let [ea (element-args args i)
                            k (element-key key i)]
                        (if (= op :update)
                          (let [{sub :trace w :weight} (generate-at kernel ea (cm/get-submap cs i) k [i])]
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
              (let [r (assess-at (:kernel gf) (element-args args i) (cm/get-submap choices i) (element-key key i) [i])]
                {:weight (+ weight (:weight r)) :retval (conj retval (:retval r))}))
            {:weight 0.0 :retval []}
            (range n))))

(gfi/defgf MapGF [kernel]
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (map-generate gf args nil key)))
  (-generate [gf args constraints key] (map-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (map-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (map-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (map-assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (map-generate gf args nil key))))
  (-project [_ tr selection] (project-elements tr selection)))

(defn map
  "Gen.jl's Map combinator: the kernel applied independently to each element
   of its argument vectors."
  [kernel]
  (->MapGF kernel))

;; ---------------------------------------------------------------------------
;; Unfold
;; ---------------------------------------------------------------------------

(declare check-steps*)

(defn- check-steps
  "The number of steps as a host integer (a one-element array is read; a
   number that differs across a batch's particles raises the batched-branch
   error, so batched inference falls back)."
  [n]
  (let [n (if (and (array/array? n) (not (array/host? n))) (array/item n) n)]
    (check-steps* n)))

(defn- check-steps* [n]
  (when-not (nat-int? n)
    (throw (ex-info "Unfold's first argument, the number of steps, must be a non-negative integer" {:n n})))
  n)

(defn- step-args [t state params] (into [t state] params))

(defn- unfold-generate [gf [n init & params :as args] constraints key]
  (let [n (check-steps n)]
  (when constraints (indices :generate constraints n))
  (let [kernel (:kernel gf)
        acc (loop [t 0 state init acc empty-acc]
              (if (< t n)
                (let [sa (step-args t state params)
                      k (element-key key t)
                      [sub w] (if constraints
                                (let [r (generate-at kernel sa (cm/get-submap constraints t) k [t])] [(:trace r) (:weight r)])
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
        params-changed? (some diff/changed? dparams)
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
                      (update-at kernel old sa sad (cm/get-submap cs t) k [t])
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
                                (let [r (generate-at kernel sa (cm/get-submap cs t) k [t])] [(:trace r) (:weight r)])
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
      (let [r (assess-at (:kernel gf) (step-args t state params) (cm/get-submap choices t) (element-key key t) [t])]
        (recur (inc t) (:retval r) (+ weight (:weight r)) (conj retval (:retval r))))
      {:weight weight :retval retval}))))

(gfi/defgf UnfoldGF [kernel]
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (unfold-generate gf args nil key)))
  (-generate [gf args constraints key] (unfold-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (unfold-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (unfold-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (unfold-assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (unfold-generate gf args nil key))))
  (-project [_ tr selection] (project-elements tr selection)))

(defn unfold
  "Gen.jl's Unfold combinator: a state-space chain of n kernel steps."
  [kernel]
  (->UnfoldGF kernel))

;; ---------------------------------------------------------------------------
;; Switch
;; ---------------------------------------------------------------------------

(defn- branch-index
  "The branch index as a host value: a one-element array is read; an index
   that differs across the particles of a batch cannot pick one branch for
   all, and array/item raises the batched-branch error (gen.array.batch), so
   batched inference falls back to one particle at a time."
  [idx]
  (if (or (array/host? idx) (not (array/array? idx))) idx (array/item idx)))

(defn- branch [gf idx]
  (or (get (:branches gf) (branch-index idx))
      (throw (ex-info (str "Switch has no branch " (pr-str idx)) {:index idx :branches (count (:branches gf))}))))

(defn- switch-trace [gf args idx sub]
  (assoc (trace/make-trace {:gen-fn gf :args args :choices (:choices sub) :retval (:retval sub) :score (:score sub)})
         ::branch (branch-index idx)
         ::subtrace sub))

(defn- switch-generate [gf [idx & bargs :as args] constraints key]
  (let [b (branch gf idx)]
    (if constraints
      (let [{sub :trace w :weight} (generate-at b (vec bargs) constraints key nil)]
        {:trace (switch-trace gf (vec args) idx sub) :weight w})
      {:trace (switch-trace gf (vec args) idx (simulate-at b (vec bargs) key)) :weight 0.0})))

(defn- switch-revise [gf tr [idx & bargs :as args] argdiffs op cs key]
  (let [args (vec args)
        bargs (vec bargs)]
    (if (or (= (branch-index idx) (::branch tr))
            ;; compatible branches (two dynamic GFs, two distributions, two
            ;; combinators of one type): the new one runs against the old one's
            ;; choices, so the addresses they share keep their values (Def
            ;; 2.3.1), as the equivalent if would and as Gen.jl does
            (dyn/compatible-callee? (:gen-fn (::subtrace tr)) (branch gf idx)))
      (let [b (branch gf idx)
            sub (::subtrace tr)
            bdiffs (if (= (branch-index idx) (::branch tr)) (vec (rest argdiffs)) (unknown-diffs bargs))
            r (if (= op :update)
                (update-at b sub bargs bdiffs cs key nil)
                (regenerate-at b sub bargs bdiffs cs key nil))]
        (assoc r :trace (switch-trace gf args idx (:trace r))))
      ;; a new branch: generated afresh; the old choices are discarded
      (let [{t :trace w :weight} (switch-generate gf args (when (= op :update) cs) key)]
        (cond-> {:trace t :retdiff diff/unknown-change}
          (= op :update) (assoc :weight (- w (:score tr)) :discard (:choices tr))
          (= op :regenerate) (assoc :weight 0.0))))))

(gfi/defgf SwitchGF [branches]
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (switch-generate gf args nil key)))
  (-generate [gf args constraints key] (switch-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (switch-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (switch-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf [idx & bargs] choices key] (gfi/-assess (branch gf idx) (vec bargs) choices key))
  (-propose [gf args key] (propose-from-trace (:trace (switch-generate gf args nil key))))
  (-project [_ tr selection] (let [sub (::subtrace tr)] (gfi/-project (:gen-fn sub) sub selection))))

(defn switch
  "Gen.jl's Switch combinator: (switch gf0 gf1 …) is called with a branch
   index first, (switch {k gf …}) with a branch key first."
  [& gfs]
  (if (and (= 1 (count gfs)) (map? (first gfs)) (not (record? (first gfs))))
    (->SwitchGF (first gfs))
    (->SwitchGF (vec gfs))))

;; ---------------------------------------------------------------------------
;; Recurse
;; ---------------------------------------------------------------------------

(defn- child-of [max-branch node j] (+ (* node max-branch) j 1))

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
          (throw (ex-info (str (name op) ": a single value was given where Recurse's nodes belong") {:op op})))
        (into {} (clojure.core/map (fn [[i sub]]
                                     (when-not (nat-int? i)
                                       (throw (gfi/unvisited op [[i]] "Recurse's addresses are node indices")))
                                     (when (cm/has-value? sub)
                                       (throw (gfi/unvisited op [[i]] (str "a value at node " i ": a node has :production and :aggregation"))))
                                     (let [parts (set (clojure.core/map first (cm/submaps sub)))]
                                       (when-let [bad (seq (remove #{:production :aggregation} parts))]
                                         (throw (gfi/unvisited op (mapv #(vector i %) bad) "a node has :production and :aggregation")))
                                       [i parts])))
              (cm/submaps cs)))))

(defn- targeted? [targets node part]
  (or (= :all targets) (contains? (get targets node) part)))

(defn- node-cs [op cs node part]
  (when cs (if (= op :regenerate) (sel/get-subselection (sel/get-subselection cs node) part) (cm/get-submap cs [node part]))))

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
                       (map-indexed (fn [j c] [(child-of max-branch node j) [c child-diff]]) (:children out)))
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
                    kids (clojure.core/map #(child-of max-branch node %) (range (count children)))
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
                        (let [r (assess-at production [in] (cm/get-submap choices [node :production])
                                           (node-key key node :production) [node :production])
                              out (production-output gf node r)]
                          (recur (into (dissoc todo node)
                                       (map-indexed (fn [j c] [(child-of max-branch node j) c]) (:children out)))
                                 (assoc ps node out) (+ weight (:weight r))))
                        [ps weight]))
        _ (doseq [node (keys targets)]
            (when-not (contains? ps node)
              (throw (gfi/unvisited :assess [[node]] (str "the tree has nodes " (vec (sort (keys ps))))))))
        [as weight] (reduce (fn [[as weight] node]
                              (let [{v :value children :children} (ps node)
                                    ws (mapv #(as (child-of max-branch node %)) (range (count children)))
                                    r (assess-at aggregation [v ws] (cm/get-submap choices [node :aggregation])
                                                 (node-key key node :aggregation) [node :aggregation])]
                                [(assoc as node (:retval r)) (+ weight (:weight r))]))
                            [{} weight] (sort > (keys ps)))]
    {:weight weight :retval (as 0)}))

(defn- recurse-project [tr s]
  (let [part (fn [subs part]
               (reduce (fn [w [node sub]]
                         (+ w (gfi/-project (:gen-fn sub) sub (sel/get-subselection (sel/get-subselection s node) part))))
                       0.0 subs))]
    (+ (part (::productions tr) :production) (part (::aggregations tr) :aggregation))))

(gfi/defgf RecurseGF [production aggregation max-branch]
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (recurse-run gf nil (recurse-args args) [diff/unknown-change] :generate nil key)))
  (-generate [gf args constraints key] (recurse-run gf nil (recurse-args args) [diff/unknown-change] :generate constraints key))
  (-update [gf tr args argdiffs constraints key] (recurse-run gf tr (recurse-args args) argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (recurse-run gf tr (recurse-args args) argdiffs :regenerate selection key))
  (-assess [gf args choices key] (recurse-assess gf (recurse-args args) choices key))
  (-propose [gf args key] (propose-from-trace (gfi/-simulate gf args key)))
  (-project [_ tr selection] (recurse-project tr selection)))

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
