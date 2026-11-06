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
   shifts another element's randomness.

   A choice outside its support stops the run of the part that makes it
   (gen.dynamic; DESIGN.md §9). Unfold, Scan and Recurse, which pass a
   part's return value on, stop at such a part: their trace holds the parts
   up to it, an update discards the old ones after it, and the score is the
   sum of the parts that ran. A dimap does not map such a part's return
   value. Map runs every element (they pass nothing on), and a dynamic
   caller stops at the call's score."
  {:api :stable}
  (:refer-clojure :exclude [map])
  (:require [gen.combinators.chain :as chain]
            [gen.combinators.elements :as el]
            [gen.combinators.recurse :as recurse]
            [gen.combinators.switch :as switch]
            [gen.combinators.wrap :as wrap]
            [gen.diff :as diff]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.partition :as part]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

;; The kinds, their constructors and their printing. Each kind's operations
;; are its namespace's: Map's gen.combinators.elements (the indexed calls
;; Unfold and Scan share), Unfold's and Scan's gen.combinators.chain,
;; Switch's gen.combinators.switch, Recurse's gen.combinators.recurse, and
;; Mix's body and dimap's gen.combinators.wrap; their calls, the keys of
;; indexed calls and a stop at a choice outside its support are
;; gen.combinators.calls'.

(defn- check-gen-fn
  "gf, checked to be a generative function: every constructor's argument,
   refused when it is made, not when it is first called."
  [what gf]
  (when-not (gfi/generative-function? gf)
    (throw (ex-info (str what " needs a generative function, not " (pr-str gf))
                    {:reason :not-a-generative-function :value gf})))
  gf)

(defn- propose-from-trace [tr]
  {:choices (:choices tr) :weight (:score tr) :retval (:retval tr)})

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

;; ---------------------------------------------------------------------------
;; Map
;; ---------------------------------------------------------------------------

(gfi/defgf MapGF [kernel]
  gfi/IStores
  (-stores [gf] (gfi/stores (:kernel gf)))
  (-with-stores [gf stores] (assoc gf :kernel (gfi/with-stores (:kernel gf) stores)))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (el/map-generate gf args nil key)))
  (-generate [gf args constraints key] (el/map-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (el/map-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (el/map-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (el/map-assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (el/map-generate gf args nil key))))
  (-project [_ tr selection] (el/project tr selection))
  trace/ICalls
  (-call [_ tr p] (el/element-call tr p)))

(defn map
  "Gen.jl's Map combinator: the kernel applied independently to each element
   of its argument vectors."
  [kernel]
  (->MapGF (check-gen-fn "map" kernel)))

;; ---------------------------------------------------------------------------
;; Unfold
;; ---------------------------------------------------------------------------

(gfi/defgf UnfoldGF [kernel]
  gfi/IStores
  (-stores [gf] (gfi/stores (:kernel gf)))
  (-with-stores [gf stores] (assoc gf :kernel (gfi/with-stores (:kernel gf) stores)))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (chain/unfold-generate gf args nil key)))
  (-generate [gf args constraints key] (chain/unfold-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (chain/unfold-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (chain/unfold-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (chain/unfold-assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (chain/unfold-generate gf args nil key))))
  (-project [_ tr selection] (el/project tr selection))
  trace/ICalls
  (-call [_ tr p] (el/element-call tr p)))

(defn unfold
  "Gen.jl's Unfold combinator: a state-space chain of n kernel steps."
  [kernel]
  (->UnfoldGF (check-gen-fn "unfold" kernel)))

;; ---------------------------------------------------------------------------
;; Switch
;; ---------------------------------------------------------------------------

(gfi/defgf SwitchGF [branches]
  gfi/IStores
  (-stores [gf] (callee-stores (:branches gf)))
  (-with-stores [gf stores] (update gf :branches with-callee-stores stores))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (switch/generate gf args nil key)))
  (-generate [gf args constraints key] (switch/generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (switch/revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (switch/revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (switch/assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (switch/generate gf args nil key))))
  (-project [_ tr selection] (switch/project tr selection))
  ;; the branch's addresses are the switch's (a trace by partition has no
  ;; one call)
  trace/ICalls
  (-call [_ tr p] (when-let [sub (::subtrace tr)] (trace/-call (trace/maker sub) sub p)))
  part/IJoin
  (-join [gf trs ns b] (switch/join gf trs ns b)))

(defn switch
  "Gen.jl's Switch combinator: (switch gf0 gf1 …) is called with a branch
   index first, (switch {k gf …}) with a branch key first. In a batched run,
   an index that differs across the particles runs each branch on its own
   particles (by partition)."
  [& gfs]
  (let [branches (if (and (= 1 (count gfs)) (map? (first gfs)) (not (record? (first gfs))))
                   (first gfs)
                   (vec gfs))]
    (run! #(check-gen-fn "switch" %) (if (map? branches) (vals branches) branches))
    (->SwitchGF branches)))

;; ---------------------------------------------------------------------------
;; Recurse
;; ---------------------------------------------------------------------------

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
  (recurse/child-of parent j max-branch))

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

(gfi/defgf RecurseGF [production aggregation max-branch]
  gfi/IStores
  (-stores [gf] (callee-stores (select-keys gf [:production :aggregation])))
  (-with-stores [gf stores] (with-callee-stores gf stores))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (recurse/run gf nil (recurse/check-args args) [diff/unknown-change] :generate nil key)))
  (-generate [gf args constraints key]
    (recurse/run gf nil (recurse/check-args args) [diff/unknown-change] :generate constraints key))
  (-update [gf tr args argdiffs constraints key]
    (recurse/run gf tr (recurse/check-args args) argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key]
    (recurse/run gf tr (recurse/check-args args) argdiffs :regenerate selection key))
  (-assess [gf args choices key] (recurse/assess gf (recurse/check-args args) choices key))
  (-propose [gf args key] (propose-from-trace (gfi/-simulate gf args key)))
  (-project [_ tr selection] (recurse/project tr selection))
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
  (->RecurseGF (check-gen-fn "recurse's production" production) (check-gen-fn "recurse's aggregation" aggregation)
               max-branch))

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

(gfi/defgf ScanGF [kernel]
  gfi/IStores
  (-stores [gf] (gfi/stores (:kernel gf)))
  (-with-stores [gf stores] (assoc gf :kernel (gfi/with-stores (:kernel gf) stores)))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (chain/scan-generate gf args nil key)))
  (-generate [gf args constraints key] (chain/scan-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (chain/scan-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (chain/scan-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (chain/scan-assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (chain/scan-generate gf args nil key))))
  (-project [_ tr selection] (el/project tr selection))
  trace/ICalls
  (-call [_ tr p] (el/element-call tr p)))

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
    (->MixGF components logits? (wrap/mix-body (->SwitchGF components) (count components) logits?))))

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

(gfi/defgf DimapGF [inner pre post]
  gfi/IStores
  (-stores [gf] (gfi/stores (:inner gf)))
  (-with-stores [gf stores] (assoc gf :inner (gfi/with-stores (:inner gf) stores)))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (wrap/dimap-generate gf args nil key)))
  (-generate [gf args constraints key] (wrap/dimap-generate gf args constraints key))
  (-update [gf tr args argdiffs constraints key] (wrap/dimap-revise gf tr args argdiffs :update constraints key))
  (-regenerate [gf tr args argdiffs selection key] (wrap/dimap-revise gf tr args argdiffs :regenerate selection key))
  (-assess [gf args choices key] (wrap/dimap-assess gf args choices key))
  (-propose [gf args key] (propose-from-trace (:trace (wrap/dimap-generate gf args nil key))))
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

;; ---------------------------------------------------------------------------
;; Printing
;; ---------------------------------------------------------------------------

(defmethod print-method MapGF [gf ^java.io.Writer w]
  (.write w "#gen/map[") (print-method (:kernel gf) w) (.write w "]"))

(defmethod print-method UnfoldGF [gf ^java.io.Writer w]
  (.write w "#gen/unfold[") (print-method (:kernel gf) w) (.write w "]"))

(defmethod print-method SwitchGF [gf ^java.io.Writer w]
  (.write w "#gen/switch") (print-method (:branches gf) w))

(defmethod print-method RecurseGF [_ ^java.io.Writer w]
  (.write w "#gen/recurse"))

(defmethod print-method ScanGF [gf ^java.io.Writer w]
  (.write w "#gen/scan[") (print-method (:kernel gf) w) (.write w "]"))

(defmethod print-method MixGF [gf ^java.io.Writer w]
  (.write w (if (:logits? gf) "#gen/mix-logits" "#gen/mix")) (print-method (:components gf) w))

(defmethod print-method DimapGF [gf ^java.io.Writer w]
  (.write w "#gen/dimap[") (print-method (:inner gf) w) (.write w "]"))
