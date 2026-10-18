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
   choices when both branches are dynamic generative functions, so the
   addresses they share keep their values (as an if would, and as Gen.jl
   does); for other branches it generates the new branch from the
   constraints and discards every old choice.

   Element or step i draws from (fold-in key i), so changing the length never
   shifts another element's randomness."
  (:refer-clojure :exclude [map + -])
  (:require [gen.array :as a :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

(defmacro ^:private defcombinator
  "defrecord, plus untraced calls as in Gen.jl: (gf a b) is the return value
   of a simulated run."
  [name fields & impls]
  `(defrecord ~name ~fields
     ~@impls
     clojure.lang.IFn
     ~@(for [n (range 7)]
         (let [as (vec (repeatedly n #(gensym "arg")))]
           `(~'invoke [this# ~@as] (:retval (gfi/simulate this# ~as)))))
     (~'applyTo [this# args#] (:retval (gfi/simulate this# (vec args#))))))

;; ---------------------------------------------------------------------------
;; Indexed elements (Map and Unfold)
;; ---------------------------------------------------------------------------

(defn- element-key [key i] (rng/fold-in key i))

;; Kernel calls. (A batched run is this same code under the batch tracer.)
(defn- simulate-at [gf args key] (gfi/-simulate gf args key))
(defn- generate-at [gf args cs key] (gfi/-generate gf args cs key))
(defn- assess-at [gf args choices] (gfi/-assess gf args choices))
(defn- update-at [gf old args argdiffs cs key] (gfi/-update gf old args argdiffs cs key))
(defn- regenerate-at [gf old args argdiffs s key] (gfi/-regenerate gf old args argdiffs s key))

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
    (doseq [i addrs]
      (when-not (and (integer? i) (< -1 i n))
        (throw (ex-info (str (name op) ": the constraint at " i " was not visited (the indices are 0 to "
                             (dec n) ")")
                        {:op op :address i :n n}))))
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
  (let [args (mapv vec args)
        n (map-length args)
        kernel (:kernel gf)]
    (when constraints (indices :generate constraints n))
    (let [acc (reduce (fn [acc i]
                        (let [ea (element-args args i)
                              k (element-key key i)]
                          (if constraints
                            (let [{sub :trace w :weight} (generate-at kernel ea (cm/get-submap constraints i) k)]
                              (put-element acc i sub nil w))
                            (put-element acc i (simulate-at kernel ea k) nil 0.0))))
                      empty-acc
                      (range n))]
      {:trace (elements-trace gf args acc) :weight (:weight acc)})))

(defn- map-revise
  "update (op :update, with constraints) or regenerate (op :regenerate, with
   a selection) of a Map trace."
  [gf tr args argdiffs op cs key]
  (let [args (mapv vec args)
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
                                      (update-at kernel old ea ead (cm/get-submap cs i) k)
                                      (regenerate-at kernel old ea ead (sel/get-subselection cs i) k)))))
                    (trace-acc tr)
                    (visits (changed-elements argdiffs) targets kept))
        acc (drop-elements acc kept op)
        acc (reduce (fn [acc i]
                      (let [ea (element-args args i)
                            k (element-key key i)]
                        (if (= op :update)
                          (let [{sub :trace w :weight} (generate-at kernel ea (cm/get-submap cs i) k)]
                            (put-element acc i sub nil w))
                          (put-element acc i (simulate-at kernel ea k) nil 0.0))))
                    acc
                    (range kept n))]
    (incremental-result gf args acc op n prev-n)))

(defn- map-assess [gf args choices]
  (let [args (mapv vec args)
        n (map-length args)]
    (indices :assess choices n)
    (reduce (fn [{:keys [weight retval]} i]
              (let [r (assess-at (:kernel gf) (element-args args i) (cm/get-submap choices i))]
                {:weight (+ weight (:weight r)) :retval (conj retval (:retval r))}))
            {:weight 0.0 :retval []}
            (range n))))

(defcombinator MapGF [kernel]
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (map-generate gf args nil key)))
  (-generate [gf args constraints key] (map-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (map-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (map-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices] (map-assess gf args choices))
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

(defn- check-steps [n]
  (when-not (nat-int? n)
    (throw (ex-info "Unfold's first argument, the number of steps, must be a non-negative integer" {:n n})))
  n)

(defn- step-args [t state params] (into [t state] params))

(defn- unfold-generate [gf [n init & params :as args] constraints key]
  (check-steps n)
  (when constraints (indices :generate constraints n))
  (let [kernel (:kernel gf)
        acc (loop [t 0 state init acc empty-acc]
              (if (< t n)
                (let [sa (step-args t state params)
                      k (element-key key t)
                      [sub w] (if constraints
                                (let [r (generate-at kernel sa (cm/get-submap constraints t) k)] [(:trace r) (:weight r)])
                                [(simulate-at kernel sa k) 0.0])]
                  (recur (inc t) (:retval sub) (put-element acc t sub nil w)))
                acc))]
    {:trace (elements-trace gf (vec args) acc) :weight (:weight acc)}))

(defn- unfold-revise
  "update or regenerate of an Unfold trace. Walks the retained steps,
   revisiting a step when its state input, the parameters, its constraints or
   its selection may have changed, and skipping straight to the next
   targeted step otherwise; then extends or truncates."
  [gf tr [n init & params :as args] argdiffs op cs key]
  (check-steps n)
  (let [kernel (:kernel gf)
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
                      (update-at kernel old sa sad (cm/get-submap cs t) k)
                      (regenerate-at kernel old sa sad (sel/get-subselection cs t) k))]
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
                                (let [r (generate-at kernel sa (cm/get-submap cs t) k)] [(:trace r) (:weight r)])
                                [(simulate-at kernel sa k) 0.0])]
                  (recur (inc t) (:retval sub) (put-element acc t sub nil w)))
                acc))]
    (incremental-result gf (vec args) acc op n prev-n)))

(defn- unfold-assess [gf [n init & params] choices]
  (check-steps n)
  (indices :assess choices n)
  ;; weight starts boxed: it accumulates arrays on array backends
  (loop [t 0 state init weight (Double/valueOf 0.0) retval []]
    (if (< t n)
      (let [r (assess-at (:kernel gf) (step-args t state params) (cm/get-submap choices t))]
        (recur (inc t) (:retval r) (+ weight (:weight r)) (conj retval (:retval r))))
      {:weight weight :retval retval})))

(defcombinator UnfoldGF [kernel]
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (unfold-generate gf args nil key)))
  (-generate [gf args constraints key] (unfold-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (unfold-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (unfold-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices] (unfold-assess gf args choices))
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
   all, and a/item raises the batched-branch error (gen.array.batch), so
   batched inference falls back to one particle at a time."
  [idx]
  (if (or (a/host? idx) (not (a/array? idx))) idx (a/item idx)))

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
      (let [{sub :trace w :weight} (generate-at b (vec bargs) constraints key)]
        {:trace (switch-trace gf (vec args) idx sub) :weight w})
      {:trace (switch-trace gf (vec args) idx (simulate-at b (vec bargs) key)) :weight 0.0})))

(defn- switch-revise [gf tr [idx & bargs :as args] argdiffs op cs key]
  (let [args (vec args)
        bargs (vec bargs)]
    (if (or (= (branch-index idx) (::branch tr))
            ;; two dynamic branches: the new one runs against the old one's
            ;; choices, so the addresses they share keep their values (Def
            ;; 2.3.1), as the equivalent if would and as Gen.jl does
            (and (dyn/dynamic-gf? (branch gf idx)) (dyn/dynamic-gf? (:gen-fn (::subtrace tr)))))
      (let [b (branch gf idx)
            sub (::subtrace tr)
            bdiffs (if (= (branch-index idx) (::branch tr)) (vec (rest argdiffs)) (unknown-diffs bargs))
            r (if (= op :update)
                (update-at b sub bargs bdiffs cs key)
                (regenerate-at b sub bargs bdiffs cs key))]
        (assoc r :trace (switch-trace gf args idx (:trace r))))
      ;; a new branch: generated afresh; the old choices are discarded
      (let [{t :trace w :weight} (switch-generate gf args (when (= op :update) cs) key)]
        (cond-> {:trace t :retdiff diff/unknown-change}
          (= op :update) (assoc :weight (- w (:score tr)) :discard (:choices tr))
          (= op :regenerate) (assoc :weight 0.0))))))

(defcombinator SwitchGF [branches]
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (switch-generate gf args nil key)))
  (-generate [gf args constraints key] (switch-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (switch-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (switch-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf [idx & bargs] choices] (gfi/-assess (branch gf idx) (vec bargs) choices))
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
;; Printing
;; ---------------------------------------------------------------------------

(defmethod print-method MapGF [gf ^java.io.Writer w]
  (.write w "#gen/map[") (print-method (:kernel gf) w) (.write w "]"))

(defmethod print-method UnfoldGF [gf ^java.io.Writer w]
  (.write w "#gen/unfold[") (print-method (:kernel gf) w) (.write w "]"))

(defmethod print-method SwitchGF [gf ^java.io.Writer w]
  (.write w "#gen/switch") (print-method (:branches gf) w))
