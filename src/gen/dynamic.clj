(ns gen.dynamic
  "The dynamic modeling language: generative functions written as Clojure
   code with the gen macro.

     (gen [xs]
       (let [slope (trace :slope (normal 0 10))]
         (doseq [[i x] (map-indexed vector xs)]
           (trace [:y i] (normal (* slope x) 1)))
         slope))

   Inside a body, (trace addr dist) makes a random choice at an address (or a
   vector path), and (splice addr gf & args) calls another generative
   function, its choices nested under addr. (param name) reads a trainable
   parameter from the store in effect (with-params). Conditionals branch on
   the values of 0-d arrays (docs/array-api.md §8a).

   Every GFI operation runs the body under a handler: a pure transition from
   (state, address, distribution) to [value state'], one per operation,
   threaded through a single volatile! for the duration of one call. Each site
   consumes one key from a sequence split off the operation's key. The same
   transitions serve one execution or a batch of particles: a batched run is
   the same code under the batch tracer (gen.array.batch), whose values have
   per-particle shapes and draws a batch axis. A trace records every choice's log
   probability and every spliced sub-trace, so regenerate weights use the old
   log probabilities exactly and project reads the trace without re-running
   the model."
  (:refer-clojure :exclude [+ -])
  (:require [gen.array :as array :refer [+ -]]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist.core :as dc]
            [gen.dynamic.core]
            [gen.dynamic.static :as static]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Addresses
;; ---------------------------------------------------------------------------

(defn- path [addr] (if (vector? addr) addr [addr]))

(defn- check-address!
  "A path may be used once, and never under or above another used path."
  [state p]
  (let [visited (:visited state)]
    (when (empty? p)
      (throw (ex-info "An address path must not be empty" {:path p})))
    (when (contains? visited p)
      (throw (ex-info (str "Address " p " is used twice") {:path p})))
    (doseq [n (range 1 (count p))]
      (when (contains? visited (subvec p 0 n))
        (throw (ex-info (str "Address " p " lies under the choice at " (subvec p 0 n))
                        {:path p :prefix (subvec p 0 n)}))))
    (when (cm/has-submap? (:choices state) p)
      (throw (ex-info (str "Address " p " already holds nested choices") {:path p})))))

;; ---------------------------------------------------------------------------
;; Pure transitions: (state, path, dist) -> [value state']
;; ---------------------------------------------------------------------------

(defn- site-log-prob
  "A site's log probability: the distribution's log-prob summed over all its
   axes (a site with array parameters is one choice). Under the batch tracer
   the axes are per particle, so a batched site sums per particle."
  [dist v]
  (let [lp (dc/log-prob dist v)]
    (if (pos? (array/ndim lp)) (array/sum lp) lp)))

(defn- given-value
  "[v lp] of the value in the constraint c at a site. (Under the batch tracer
   an unbatched value is shared by every particle, and a batched one is per
   particle.)"
  [state dist c]
  (let [v (cm/get-value c)
        expected (dc/value-shape dist)]
    (dc/check-not-nan (:mode state) v)
    ;; a value of another shape would be scored as several draws, silently
    (when (and expected (not= expected (array/shape v)))
      (throw (ex-info (str (name (:mode state)) ": the value at this site has shape " (array/shape v)
                           ", but the distribution's values have shape " expected)
                      {:value-shape (array/shape v) :expected expected})))
    [v (site-log-prob dist v)]))

(defn- draw [_state dist k]
  (dc/sample dist k []))

(defn- record
  "state with the choice v (log probability lp) at path p."
  [state p v lp]
  (-> state
      (update :choices cm/set-choice p v)
      (update :scores assoc p lp)
      (update :score + lp)))

(defn- add-weight [state w] (update state :weight + w))

(defn- constraint-at
  "The constraint value at a primitive site, or nil. A nested choice map there
   is an error: it would otherwise be silently ignored. A value on the way
   to p (a constraint at an address where the run calls) is no constraint
   here, so it stays the unvisited error, which a revise drops."
  [state p op]
  (let [c (cm/get-submap (:constraints state) p cm/EMPTY)]
    (cond (cm/has-value? c) c
          (cm/empty? c) nil
          :else (throw (ex-info (str (name op) ": the constraint at " p " is a nested choice map,"
                                     " but the site is a single random choice")
                                {:path p :op op})))))

(defn- prev-choice
  "[value lp] of the previous trace's choice at p, or nil."
  [state p]
  (when-let [prev (:prev state)]
    (let [scores (::scores prev)]
      (when (contains? scores p)
        [(cm/get-choice (:choices prev) p) (get scores p)]))))

(defn- simulate-site [state p dist k]
  (let [v (draw state dist k)]
    [v (record state p v (site-log-prob dist v))]))

(defn- generate-site [state p dist k]
  (if-let [c (constraint-at state p :generate)]
    (let [[v lp] (given-value state dist c)]
      [v (-> state (record p v lp) (add-weight lp))])
    (simulate-site state p dist k)))

(defn- assess-site [state p dist _k]
  (let [c (constraint-at state p :assess)]
    (when-not c
      (throw (ex-info (str "assess: no value for the choice at " p) {:path p :missing-choice true})))
    (let [[v lp] (given-value state dist c)]
      [v (-> state (record p v lp) (add-weight lp))])))

(defn- update-site [state p dist k]
  (let [c (constraint-at state p :update)
        prev (prev-choice state p)]
    (cond
      c (let [[v lp] (given-value state dist c)]
          [v (cond-> (-> state (record p v lp) (add-weight lp))
               prev (-> (add-weight (array/negate (second prev)))
                        (update :discard cm/set-choice p (first prev))))])
      prev (let [[v prev-lp] prev
                 _ (dc/check-value-shape dist v)
                 lp (site-log-prob dist v)]
             [v (-> state (record p v lp) (add-weight (- lp prev-lp)))])
      :else (simulate-site state p dist k))))

(defn- regenerate-site [state p dist k]
  (let [prev (prev-choice state p)]
    (if (and prev (not (sel/selected? (:selection state) p)))
      (let [[v prev-lp] prev
            _ (dc/check-value-shape dist v)
            lp (site-log-prob dist v)]
        [v (-> state (record p v lp) (add-weight (- lp prev-lp)))])
      ;; selected, or a new address: a fresh draw from the prior, which cancels
      (simulate-site state p dist k))))

;; Every site consumes one key from the sequence, whether it samples or not, so
;; constraining or retaining a choice never shifts another site's randomness:
;; generate with part of a trace's choices and that trace's key reproduces it.
(def site-transitions
  "The standard handler transitions, one per operation: (fn [state path dist
   key] → [value state']). Middleware (with-transitions) wraps them."
  {:simulate simulate-site :generate generate-site :assess assess-site
   :update update-site :regenerate regenerate-site})

;; ---------------------------------------------------------------------------
;; Splices: calling another generative function at an address
;; ---------------------------------------------------------------------------


(defn- site-key
  "[the key of the run's next site or call, state with the key after it]:
   every site and call consumes one, whether it samples or not."
  [state]
  (let [[k-next k-site] (rng/split (:key state))]
    [k-site (assoc state :key k-next)]))

(def ^:dynamic *params*
  "The trainable parameters in effect: the store of the outermost generative
   function that has one, merged with any inner one's (with-params)."
  nil)

(defn- with-store
  "The store in effect within a generative function with its own store ps:
   ps over the outer one (not a copy when there is none, so a store keeps
   its identity from run to run)."
  [ps]
  (if *params* (merge *params* ps) ps))

(defn- arg-diffs
  "Argdiffs for a splice's new args against its previous call's: no-change
   for an identical argument (sound, and O(1)), else the hint given (the
   static language's propagated diffs), else unknown-change."
  ([prev-args args] (arg-diffs prev-args args nil))
  ([prev-args args hints]
   (mapv (fn [a pa h] (cond (identical? a pa) diff/no-change h h :else diff/unknown-change))
         args (concat prev-args (repeat nil)) (concat hints (repeat nil)))))

(defn- record-call
  "state with the sub-trace tr at path p (score sc)."
  [state p tr]
  (cond-> (-> state
              (update :subtraces assoc p tr)
              (update :score + (:score tr)))
    (not (cm/empty? (:choices tr))) (update :choices cm/set-choice p (:choices tr))))

(declare dynamic-gf?)

(defn- project-under-its-store
  "tr's projection by its maker under the store at the call that made it
   (a callee's project may run its model)."
  [tr s]
  (binding [*params* (::site-store tr)] (gfi/project-by-maker tr s)))

(defn ^:no-doc run-call
  "op on gf at a call site, a splice or a combinator's kernel, through
   gfi/run-op (opts as there, and :at, the call's address in its caller):
   the one place a call's context is checked. Each trace a call makes is
   marked with the store in effect at the call. An update or regenerate
   under another store than prev's mark keeps nothing of prev but its
   choices (gfi/revise, prev projected under its own store), whatever
   kind gf is: no argument shows a store, so no callee's own skipping
   could see it. (A gf that does not compute as prev's maker is
   gfi/run-op's to revise.) In a revise's generate (gfi/*revising*), a
   generate drops the old choices its callee leaves unvisited
   (gfi/generate-dropping), never the revise's constraints: the call's path
   extends the revise's prefix. An unvisited-constraint error from the callee
   is raised with its paths under :at (gfi/under)."
  [gf op args opts]
  (let [prev (:prev opts)
        r (try (cond
                 (and prev (not (gfi/same-value? *params* (::site-store prev))))
                 (gfi/revise gf op args opts project-under-its-store)
                 (and gfi/*revising* (= op :generate))
                 (let [prefix (into (:prefix gfi/*revising*) (:at opts))]
                   (binding [gfi/*revising* (assoc gfi/*revising* :prefix prefix)]
                     (first (gfi/generate-dropping #(gfi/run-op gf op args (assoc opts :constraints %))
                                                   (or (:constraints opts) cm/EMPTY) prefix))))
                 :else (gfi/run-op gf op args opts))
               (catch clojure.lang.ExceptionInfo e (throw (gfi/under e (:at opts)))))
        t (:trace r)]
    (cond-> r
      (and t (not (identical? (::site-store t) *params*))) (assoc :trace (assoc t ::site-store *params*)))))

(defn- splice-call
  "Call gf at path p under the run's operation, scalar or batched (through
   run-call), nesting its choices and recording its sub-trace: [retval
   state' retdiff]. hints are argdiffs for the arguments that are not
   identical to the previous call's. The previous call's sub-trace is
   revised whatever gf is (another generative function by gfi/revise), so
   the addresses both make keep their values (Def 2.3.1)."
  [state p gf args & [hints]]
  (let [mode (:mode state)
        cons (cm/get-submap (:constraints state) p cm/EMPTY)
        prev-sub (get (::subtraces (:prev state)) p)
        argdiffs (when prev-sub (arg-diffs (:args prev-sub) args hints))
        ;; a batched run's callees run batched too: they inherit the batch tracer
        base {}
        ;; the callee's key, the same in every mode: assess draws what
        ;; generate would with the same key
        [k state] (site-key state)
        opts (assoc base :key k :at p)]
    (if (= mode :assess)
      (let [{w :weight rv :retval} (try (gfi/run-op gf :assess args (assoc opts :constraints cons))
                                        (catch clojure.lang.ExceptionInfo e (throw (gfi/under e p))))]
        [rv (cond-> (-> state (update :score + w) (add-weight w))
              (not (cm/empty? cons)) (update :choices cm/set-choice p cons))])
      (let [r (cond
                (= mode :simulate) (run-call gf :simulate args opts)
                (= mode :generate) (run-call gf :generate args (assoc opts :constraints cons))
                (nil? prev-sub) (if (= mode :update)
                                  (run-call gf :generate args (assoc opts :constraints cons))
                                  (run-call gf :simulate args opts))
                (= mode :update) (run-call gf :update args (assoc opts :prev prev-sub :constraints cons :argdiffs argdiffs))
                :else (run-call gf :regenerate args (assoc opts :prev prev-sub :argdiffs argdiffs
                                                           :selection (sel/get-subselection (:selection state) p))))
            {tr :trace w :weight d :discard rd :retdiff} r]
        [(:retval tr) (cond-> (record-call state p tr)
                        w (add-weight w)
                        (and d (not (cm/empty? d))) (update :discard cm/set-choice p d))
         (or rd diff/unknown-change)]))))

;; ---------------------------------------------------------------------------
;; Running a body
;; ---------------------------------------------------------------------------

(defn- check-constraints-visited!
  "Every constraint must be consumed by a choice or a splice (as in Gen.jl)."
  [state op]
  (let [visited (:visited state)
        missed (remove (fn [p] (some #(contains? visited (subvec p 0 %)) (range 1 (inc (count p)))))
                       (cm/addresses (:constraints state)))]
    (when (seq missed)
      (throw (gfi/unvisited op missed)))))

(defn- finish
  "Account for the choices and calls of the previous trace that this run no
   longer visits, and check the constraints."
  [state]
  (let [mode (:mode state)]
    (when (#{:generate :update :assess} mode)
      (check-constraints-visited! state mode))
    (if (= mode :update)
      ;; gone by kind: a path that held a choice and now holds a call (or the
      ;; reverse) is visited, but its old record is still gone
      (let [prev (:prev state)
            gone-choices (remove #(contains? (:scores state) %) (keys (::scores prev)))
            gone-calls (remove #(contains? (:subtraces state) %) (keys (::subtraces prev)))]
        (as-> state s
          (reduce (fn [s p] (-> s (add-weight (array/negate (get (::scores prev) p)))
                                (update :discard cm/set-choice p (cm/get-choice (:choices prev) p))))
                  s gone-choices)
          (reduce (fn [s p] (let [sub (get (::subtraces prev) p)]
                              (cond-> (add-weight s (array/negate (:score sub)))
                                (not (cm/empty? (:choices sub)))
                                (update :discard cm/set-choice p (:choices sub)))))
                  s gone-calls)))
      ;; regenerate: choices no longer visited cancel against the reverse move
      state)))

;; ---------------------------------------------------------------------------
;; Encapsulated randomness: untraced draws (thesis §4.5)
;; ---------------------------------------------------------------------------

(def ^:private omega-salt
  "Folded into an operation's key for its encapsulated randomness ω, apart
   from the sites' sequence, so untraced draws never shift a site's key."
  0x6f6d6567)

(defn- keeps-omega?
  "Does an update or regenerate keep the previous trace's ω? Only when
   nothing the function depends on changed: no-change argdiffs, no
   constraint (update) or selection (regenerate) under it, and not a
   revise (a changed store or generative function at the call, which no
   hint shows). Otherwise ω is drawn afresh, a pseudo-marginal refresh
   (DESIGN.md §9)."
  [mode {:keys [argdiffs constraints selection] :as opts}]
  (and (not (::revise opts)) (some? argdiffs) (every? diff/no-change? argdiffs)
       (case mode
         :update (or (nil? constraints) (cm/empty? constraints))
         :regenerate (= sel/none selection)
         false)))

(defn- initial-state [mode {:keys [key constraints prev selection argdiffs] :as opts}]
  {:mode mode :key key :argdiffs argdiffs
   :choices cm/EMPTY :scores {} :subtraces {} :score 0.0 :weight 0.0
   :constraints (or constraints cm/EMPTY) :prev prev :selection (or selection sel/none)
   :discard cm/EMPTY :visited #{}
   ;; ω: the previous trace's when kept, else folded from the key at the first
   ;; untraced draw
   :omega (when (keeps-omega? mode opts) (::trace/omega prev)) :omega-key key :omega-count 0})

(defn- read-param
  "(param name) in a body: the parameter's value in the store in effect."
  [name]
  (let [v (get *params* name ::missing)]
    (if (identical? ::missing v)
      (throw (ex-info (str "The parameter " (pr-str name) " has no value: give the generative function a store"
                           " with gen.dynamic/with-params")
                      {:param name :available (keys *params*)}))
      v)))

(declare run-handler)

(def ^:private realize-limit 10000)

(defn- realize!
  "Force the lazy sequences in a body's return value (and in the vectors,
   lists and maps holding them), so that trace calls in them happen during
   the run. Lazy sequences cache their elements, so this changes no value.
   Only lazy sequences proper are forced, and only their first realize-limit
   elements: the rest of a longer (or infinite) one stays lazy, and a trace
   call in it is a loud error after the run (check-running!)."
  [x]
  (cond (instance? clojure.lang.LazySeq x) (run! realize! (take realize-limit x))
        (or (vector? x) (list? x)) (run! realize! x)
        (map? x) (run! realize! (vals x)))
  x)

(defn- check-running! [state p]
  (when (::finished state)
    (throw (ex-info (str "trace or splice at " p " after the generative function returned: a lazy"
                         " sequence escaped the body (realize it in the body, e.g. with mapv or doall)")
                    {:path p}))))

(defn- check-arity!
  "args must fit gf's parameter vector (the body fn's own arity error would
   count its hidden parameters and name an anonymous fn)."
  [gf args]
  (let [params (first (:source gf))]
    (when (vector? params)
      (let [n (count args)
            i (.indexOf ^java.util.List params '&)
            fixed (if (neg? i) (count params) i)]
        (when-not (if (neg? i) (== n fixed) (>= n fixed))
          (throw (ex-info (str "The generative function " (pr-str params) " takes "
                               (when-not (neg? i) "at least ") fixed " argument" (when-not (== 1 fixed) "s")
                               ", given " n)
                          {:params params :args n})))))))

;; ---------------------------------------------------------------------------
;; The static language: a walk of the graph that skips what did not change
;; ---------------------------------------------------------------------------

(defn- arg-name-diffs
  "Each argument name's diff: an argument bound to a symbol takes its
   argdiff; a destructured one's names no-change or unknown-change. Every
   name unknown-change without argdiffs."
  [{:keys [fixed rest]} params argdiffs]
  (let [d-of (fn [d] (if (diff/no-change? d) diff/no-change diff/unknown-change))
        positional (map (fn [names form i]
                          (let [d (if argdiffs (nth argdiffs i diff/unknown-change) diff/unknown-change)]
                            (if (symbol? form) {form d} (zipmap names (repeat (d-of d))))))
                        fixed params (range))
        rest-d (if (and argdiffs (every? diff/no-change? (drop (count fixed) argdiffs))) diff/no-change diff/unknown-change)]
    ;; in binding order: a name bound again (by a later parameter, or the
    ;; rest parameter) holds its last binding, as in let
    (apply merge (concat positional [(zipmap rest (repeat rest-d))]))))

(defn- spec-diff
  "An argument's diff from its spec: its slot's, none for a constant, or
   unknown-change unless every slot its expression reads is unchanged."
  [diffs {:keys [slot constant in]}]
  (cond slot (nth diffs slot)
        constant diff/no-change
        (every? #(diff/no-change? (nth diffs %)) in) diff/no-change
        :else diff/unknown-change))

(defn- computed
  "[value diff] of a compute node or the return expression on env: one call
   of a var gets its diff from gen.diff/propagate."
  [{:keys [f rule args-f arg-specs alias]} env diffs old skip?]
  (cond
    alias [(nth env alias) (if skip? (nth diffs alias) diff/unknown-change)]
    ;; the value from the node's guarded fn, as the body computes it; the
    ;; rule sees the arguments only for the diff
    rule
    (let [v (f read-param env)]
      [v (if skip? (diff/propagate rule (args-f read-param env) (mapv #(spec-diff diffs %) arg-specs)) diff/unknown-change)])
    :else
    (let [v (f read-param env)]
      [v (if (and skip? (identical? v old)) diff/no-change diff/unknown-change)])))

(defn- keep-node!
  "A node of kind at path p, kept: its previous value old, with its site's
   or call's previous record, as an unchanged rerun would leave them
   (weight 0), and the same key consumed, so that what follows draws what
   the dynamic body would."
  [vol kind p old]
  (when-not (= kind :compute)
    (let [state @vol]
      (check-running! state p)
      (check-address! state p)
      (let [state (second (site-key state))
            state (if (= kind :trace)
                    (let [[v lp] (prev-choice state p)] (record state p v lp))
                    (record-call state p (get (::subtraces (:prev state)) p)))]
        (vreset! vol (update state :visited conj p)))))
  old)

(defn- run-static
  "A static body (gen.dynamic.static), node by node, with the handler's own
   trace and splice. In update and regenerate, a node keeps its value,
   score and sub-trace when its inputs' diffs are all no-change and no
   constraint or selection reaches it; the trace keeps every node's value
   for the next update. Diffs come from the argdiffs, from a callee's
   retdiff, from gen.diff/propagate for a binding that is one call, and a
   splice hands its callee its arguments' diffs. A gf with middleware
   (with-transitions) reruns every node, so its transitions see every site."
  [{:keys [nodes ret args-bind params slots size] :as st} gf vol trace-fn splice-at args]
  (let [{:keys [mode prev constraints selection argdiffs]} @vol
        values (::node-values prev)
        ;; the values are this graph's, made by a gf that computes as this
        ;; one (not another static gf's at the same address, a Switch's
        ;; other branch, the same body under other middleware: gfi/revise
        ;; runs those here too), read under an equal store (a callee's
        ;; param reads, a param in an argument)
        skip? (boolean (and values (#{:update :regenerate} mode) (nil? (:transitions gf))
                            (gfi/same-gen-fn? gf (trace/maker prev))
                            (gfi/same-value? *params* (::params prev))))
        reached? (fn [kind p]
                   (cond (= mode :update) (not (cm/empty? (cm/get-submap constraints p cm/EMPTY)))
                         (= kind :trace) (sel/selected? selection p)
                         :else (not= sel/none (sel/get-subselection selection p))))
        unchanged? (fn [diffs in] (every? #(diff/no-change? (nth diffs %)) in))
        [env diffs vals]
        (reduce (fn [[env diffs vals] [i {:keys [kind addr keys-f in out f bind symbol? arg-specs] :as n}]]
                  (let [old (when skip? (nth values i))
                        ;; a computed address's path, from the inputs (the
                        ;; previous one when the node is kept)
                        p (when (some? addr) (if keys-f (into (path addr) (keys-f read-param env)) (path addr)))
                        keep? (and skip? (unchanged? diffs in) (not (and p (reached? kind p))))
                        [v d] (cond
                                keep? [(keep-node! vol kind p old) diff/no-change]
                                (= kind :trace) (let [v (trace-fn p (f read-param env))]
                                                  [v (if (and skip? (identical? v old)) diff/no-change diff/unknown-change)])
                                (= kind :splice) (let [[g as] (f read-param env)]
                                                   (splice-at p g as (when skip? (mapv #(spec-diff diffs %) arg-specs))))
                                :else (computed n env diffs old skip?))]
                    [(bind env v)
                     (let [d (if (or symbol? (diff/no-change? d)) d diff/unknown-change)]
                       (reduce #(assoc %1 %2 d) diffs out))
                     (conj vals v)]))
                [(args-bind args)
                 (reduce-kv (fn [ds x d] (assoc ds (slots x) d))
                            (vec (repeat size diff/no-change))
                            (if skip? (arg-name-diffs (:args st) params argdiffs) {}))
                 []]
                (map-indexed vector nodes))]
    (vswap! vol assoc ::node-values vals)
    (if (and skip? (unchanged? diffs (:in ret)))
      (do (vswap! vol assoc ::retdiff diff/no-change) (:retval prev))
      (let [[v d] (computed ret env diffs (:retval prev) skip?)]
        (vswap! vol assoc ::retdiff d)
        v))))

(defn- execute
  "Run gf's body on args under the transitions of mode; the final state with
   :retval."
  [gf mode args opts]
  (check-arity! gf args)
  (let [;; the transition in effect: a splice without an address swaps in its
        ;; callee's middleware for the callee's body
        transition (volatile! (or (get (:transitions gf) mode) (site-transitions mode)))
        vol (volatile! (initial-state mode opts))
        ;; the n-th untraced draw's key is ω folded with n
        untraced-fn (fn untraced
                      ([] (let [state @vol]
                            (when (::finished state)
                              (throw (ex-info "An untraced draw after the generative function returned: a lazy sequence escaped the body (realize it in the body, e.g. with mapv or doall)" {})))
                            (gfi/check-exact-density! "an untraced draw")
                            ;; each particle's ω must reproduce its own draws
                            (when (batch/batch-backend? array/*backend*)
                              (throw (array/batched-branch "An untraced draw in a batched run" {})))
                            (let [omega (or (:omega state) (rng/fold-in (:omega-key state) omega-salt))
                                  n (long (:omega-count state))]
                              (vreset! vol (assoc state :omega omega :omega-count (inc n)))
                              (rng/fold-in omega n))))
                      ([dist]
                       (when-not (instance? gen.dist.core.Distribution dist)
                         (throw (ex-info "untraced draws from a distribution; give other stochastic code a key with (untraced-key)"
                                         {:value dist})))
                       (dc/sample dist (untraced) [])))
        trace-fn (fn [addr dist]
                   (let [p (path addr)
                         state @vol]
                     (check-running! state p)
                     (when-not (instance? gen.dist.core.Distribution dist)
                       (throw (ex-info (str "trace at " p " needs a distribution; call generative functions with splice")
                                       {:path p :value dist})))
                     (check-address! state p)
                     (let [[k-site state] (site-key state)
                           [v state] (@transition state p dist k-site)]
                       (vreset! vol (update state :visited conj p))
                       v)))
        ;; a call at an address: [value retdiff], hints the arguments' argdiffs
        splice-at (fn [addr gf args hints]
                    (let [args (realize! (vec args))
                          p (path addr)
                          state @vol]
                      (check-running! state p)
                      (when-not (gfi/generative-function? gf)
                        (throw (ex-info (str "splice at " p " needs a generative function") {:path p :value gf})))
                      (check-address! state p)
                      (let [[v state' rd] (splice-call state p gf args hints)]
                        (when-not (identical? state @vol)
                          (throw (ex-info (str "The caller's trace was used during the splice at " p
                                               ": a function passed to the callee calls trace")
                                          {:path p})))
                        (vreset! vol (update state' :visited conj p))
                        [v rd])))
        splice-fn (fn splice-fn [& xs]
                    (if (gfi/generative-function? (first xs))
                      ;; (splice gf & args): Gen.jl's {*} ~ gf(args), the callee's choices in
                      ;; this namespace. Its body runs under this handler, so constraints,
                      ;; previous choices, selections and address checks apply unchanged.
                      (let [[sub & args] xs
                            args (realize! (vec args))]
                        (when-not (dynamic-gf? sub)
                          (throw (ex-info "A splice without an address needs a dynamic generative function (others take an address)"
                                          {:gen-fn (type sub)})))
                        (check-arity! sub args)
                        (let [outer @transition
                              run-body #(apply (:body-fn sub) trace-fn splice-fn read-param untraced-fn args)]
                          (vreset! transition (or (get (:transitions sub) mode) (site-transitions mode)))
                          (try
                            (if-let [ps (:params sub)]
                              (binding [*params* (with-store ps)] (run-body))
                              (run-body))
                            (finally (vreset! transition outer)))))
                      ;; trace calls in lazy arguments happen now, in this body (splice-at)
                      (let [[addr gf & args] xs] (first (splice-at addr gf args nil)))))
        retval (realize! (if-let [st (:static gf)]
                           (run-static st gf vol trace-fn splice-at args)
                           (apply (:body-fn gf) trace-fn splice-fn read-param untraced-fn args)))]
    ;; the store in effect, which the next update compares with its own
    (vswap! vol assoc ::finished true ::params *params*)
    (assoc (finish @vol) :retval retval)))

(defn- make-trace [gf args state]
  (cond-> (assoc (trace/make-trace {:gen-fn gf :args args :choices (:choices state)
                                    :retval (:retval state) :score (:score state)})
                 ::scores (:scores state)
                 ::subtraces (:subtraces state))
    (pos? (long (:omega-count state))) (assoc ::trace/omega (:omega state))
    (::node-values state) (assoc ::node-values (::node-values state))
    (::params state) (assoc ::params (::params state))))

(defn- holds-batch?
  "Does x hold a batched value (in its maps, vectors and realized seqs)?"
  [x]
  (cond (batch/batched? x) true
        (array/array? x) false
        (map? x) (boolean (some holds-batch? (vals x)))
        (seq? x) (boolean (some holds-batch? (take realize-limit x)))
        (coll? x) (boolean (some holds-batch? x))
        :else false))

(defn- retdiff
  "A static body's propagated retdiff when it knows better than
   unknown-change; otherwise no-change when the return value is equal to
   the previous one (sound: equal values cannot differ), unknown-change
   otherwise."
  [prev-trace state]
  (let [d (::retdiff state)]
    (cond (and d (not (diff/unknown-change? d))) d
          ;; batched return values cannot be compared (gen.array.batch): unknown
          (and (not (holds-batch? (:retval prev-trace)))
               (not (holds-batch? (:retval state)))
               (= (:retval prev-trace) (:retval state)))
          diff/no-change
          :else diff/unknown-change)))

(defn run
  "Run a GFI operation on a dynamic generative function. opts: :key,
   :batch-shape ([] by default; [N] runs N particles at once), and per
   operation :constraints, :prev (the previous trace) and :selection. Returns
   the operation's result map."
  [gf op args opts]
  (if (seq (:batch-shape opts))
    (batch/batched (:batch-shape opts) (run gf op args (dissoc opts :batch-shape)))
    (or (when-let [dispatch (:dispatch gf)] (dispatch gf op args opts))
        (run-handler gf op args opts))))

(defn- run-handler [gf op args opts]
  (let [state (if-let [ps (:params gf)]
                (binding [*params* (with-store ps)] (execute gf op args opts))
                (execute gf op args opts))]
    (case op
      :simulate {:trace (make-trace gf args state)}
      :generate {:trace (make-trace gf args state) :weight (:weight state)}
      :assess {:weight (:weight state) :retval (:retval state)}
      :update {:trace (make-trace gf args state) :weight (:weight state)
               :retdiff (retdiff (:prev opts) state) :discard (:discard state)}
      :regenerate {:trace (make-trace gf args state) :weight (:weight state)
                   :retdiff (retdiff (:prev opts) state)})))

;; ---------------------------------------------------------------------------
;; A static function's score: one generated function of its choices
;; ---------------------------------------------------------------------------

(declare static-scored? static-score)

(defn ^:no-doc static-site
  "[value log-density] of a static body's trace node at path p of choices,
   with assess's checks."
  [dist choices p]
  (let [c (cm/get-submap choices p cm/EMPTY)]
    (when-not (cm/has-value? c)
      (throw (ex-info (str "assess: no value for the choice at " p) {:path p :missing-choice true})))
    (given-value {:mode :assess} dist c)))

(defn ^:no-doc static-call
  "[weight retval] of a static body's splice at path p: the callee's assess
   on args with sub, under key k (nil under an exact-density operation,
   where no draw may happen), with the handler's checks (an unvisited
   choice named under p)."
  [p g args sub k]
  (when-not (gfi/generative-function? g)
    (throw (ex-info (str "splice at " p " needs a generative function") {:path p :value g})))
  (let [args (realize! (vec args))]
    (try
      (if (static-scored? g)
        (static-score g args sub k)
        (let [r (gfi/run-op g :assess args {:key (or k (rng/key 0)) :constraints sub})] [(:weight r) (:retval r)]))
      (catch clojure.lang.ExceptionInfo e (throw (gfi/under e p))))))

(defn ^:no-doc check-computed!
  "assess's check of a node at a computed address (a constant prefix and
   computed keys): every choice under the prefix lies at the node's path p
   (a choice), or under it (a call), so none goes unvisited."
  [choices prefix p exact?]
  (doseq [q (cm/addresses (cm/get-submap choices prefix cm/EMPTY))
          :let [full (into prefix q)]]
    (when-not (if exact? (= full p) (= p (subvec full 0 (min (count full) (count p)))))
      (throw (gfi/unvisited :assess [full])))))

(defn- check-static-choices!
  "Every choice must sit at a node's address, as assess checks: the top
   level here (a callee checks its own), every path when a node's address
   is a path."
  [{:keys [roots paths]} choices]
  (if paths
    (doseq [p (cm/addresses choices)]
      (when-not (some (fn [[kind q]] (if (= kind :trace) (= p q) (= q (subvec p 0 (min (count p) (count q)))))) paths)
        (throw (gfi/unvisited :assess [p]))))
    (doseq [[k _] (cm/submaps choices)]
      (when-not (contains? roots k)
        (throw (gfi/unvisited :assess [[k]]))))))

(defn- static-score
  "assess of a static gf by its generated score fn, without the handler:
   [weight retval]. Keys are consumed as the handler's assess consumes
   them, so a callee's encapsulated randomness draws what it would there;
   under an operation that needs exact densities (gfi/with-exact-density),
   where a draw is an error anyway, no key is split."
  [gf args choices key]
  (check-arity! gf args)
  (let [st (:static gf)
        run #(do (check-static-choices! st choices)
                 (let [[w rv] ((:score st) read-param (vec args) choices (when-not gfi/*exact-density* key))]
                   [w (realize! rv)]))]
    (if-let [ps (:params gf)] (binding [*params* (with-store ps)] (run)) (run))))

(defn- static-scored?
  "Is gf a static function whose assess runs its score fn? Not with
   middleware or a dispatcher, which must see the run, nor batched."
  [gf]
  (and (dynamic-gf? gf) (:static gf) (nil? (:transitions gf)) (nil? (:dispatch gf))
       (not (batch/batch-backend? array/*backend*))))

(defn- project* [tr s]
  (let [from-choices (reduce (fn [acc [p lp]] (if (sel/selected? s p) (+ acc lp) acc))
                             0.0 (::scores tr))]
    (reduce (fn [acc [p sub]] (+ acc (gfi/-project (:gen-fn sub) sub (sel/get-subselection s p))))
            from-choices (::subtraces tr))))

;; ---------------------------------------------------------------------------
;; DynamicGF
;; ---------------------------------------------------------------------------

;; an untraced call, as in Gen.jl: (gf a b) is the return value of a
;; simulated run (gfi/defgf)
(gfi/defgf DynamicGF [body-fn source params transitions dispatch static]
  gfi/IStores
  (-stores [gf] (:params gf))
  (-with-stores [gf stores] (assoc gf :params stores))
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (run gf :simulate args {:key key})))
  (-generate [gf args constraints key] (run gf :generate args {:key key :constraints constraints}))
  (-update [gf tr args argdiffs constraints key]
    (run gf :update args {:key key :prev tr :argdiffs argdiffs :constraints constraints}))
  (-regenerate [gf tr args argdiffs selection key]
    (run gf :regenerate args {:key key :prev tr :argdiffs argdiffs :selection selection}))
  (-assess [gf args choices key]
    (if (static-scored? gf)
      (let [[w rv] (static-score gf args choices key)] {:weight w :retval rv})
      (run gf :assess args {:key key :constraints choices})))
  (-propose [gf args key]
    (let [tr (:trace (run gf :simulate args {:key key}))]
      {:choices (:choices tr) :weight (:score tr) :retval (:retval tr)}))
  ;; under gf's own store, as its run: a callee's project may run its model
  (-project [gf tr selection]
    (if-let [ps (:params gf)] (binding [*params* (with-store ps)] (project* tr selection)) (project* tr selection)))

  ;; a call is at a path of one or more addresses
  trace/ICalls
  (-call [_ tr p]
    (let [subs (::subtraces tr)]
      (some #(when-let [sub (get subs (subvec p 0 %))] [sub (subvec p %)]) (range 1 (inc (count p))))))

  gfi/IRevisable
  ;; the source and the static graph follow from the body fn's class
  (-same-gen-fn? [_ other budget]
    (and (gfi/same-value? body-fn (:body-fn other) budget) (gfi/same-value? params (:params other) budget)
         (gfi/same-value? transitions (:transitions other) budget) (gfi/same-value? dispatch (:dispatch other) budget)))
  ;; a dynamic trace: the handler's own update or regenerate, which reruns
  ;; every site against it (as an if would; a static body skips nothing,
  ;; run-static), weighs with its recorded log probabilities, draws ω
  ;; afresh, as every kind's revise does, and leaves each call to check its
  ;; own context (run-call); middleware sees the operation asked for
  (-revise [gf op args opts project-old]
    (if (contains? (:prev opts) ::scores)
      (run-handler gf op args (assoc opts ::revise true))
      (gfi/generate-again gf op args opts project-old))))

(defn dynamic-gf? [x] (instance? DynamicGF x))

(defn make-gen-fn
  "A dynamic generative function from a body fn of (trace splice param
   untraced & args) and its source form, and optionally the graph of a
   static body (the gen macro's, gen.dynamic.static)."
  ([body-fn source] (make-gen-fn body-fn source nil))
  ([body-fn source static]
   ;; a body fn of other hidden parameters would take the arguments shifted,
   ;; silently: its arities must fit the source's parameters after the four
   (let [params (first source)
         variadic? (instance? clojure.lang.RestFn body-fn)
         required (when variadic? (.getRequiredArity ^clojure.lang.RestFn body-fn))
         fixed (set (keep #(when (= "invoke" (.getName ^java.lang.reflect.Method %))
                             (alength (.getParameterTypes ^java.lang.reflect.Method %)))
                          (.getDeclaredMethods (class body-fn))))
         fits? (if (vector? params)
                 (let [i (.indexOf ^java.util.List params '&)
                       n (if (neg? i) (count params) i)]
                   ;; a variadic fn needs the four, and may name some arguments
                   (or (and (neg? i) (contains? fixed (+ 4 n)))
                       (and variadic? (<= 4 required (+ 4 n)))))
                 (or (not variadic?) (>= required 4)))]
     (when-not fits?
       (throw (ex-info "make-gen-fn: the body fn takes trace, splice, param and untraced before the arguments its source names"
                       {:params params :arities fixed :required-arity required})))
     (->DynamicGF body-fn source nil nil nil static))))

(defn- check-dynamic! [what gf]
  (when-not (dynamic-gf? gf)
    (throw (ex-info (str what " needs a dynamic generative function (gen), not a " (.getName (class gf))
                         ": wrap it, (gen [& args] (splice :at gf …)), or give it to the model that calls it")
                    {:gen-fn (type gf)}))))

(defn with-params
  "gf with the trainable-parameter store params, a map of name to value read
   by (param name) in gf's body and in every generative function it calls.
   gf is a dynamic generative function: the store is the body's (another
   kind reads the store of the dynamic function that calls it; a
   deterministic function that names :params may have its own,
   gen.custom/with-params)."
  [gf params]
  (check-dynamic! "with-params" gf)
  (assoc gf :params params))

(defn with-transitions
  "gf, a dynamic generative function, with handler middleware: a map from
   operation (:simulate :generate :assess :update :regenerate) to a
   transition (fn [state path dist key] → [value state']) used in place of
   the standard one (site-transitions), typically by wrapping it.
   Middleware may change the model (e.g. masking a distribution); an
   optimization must not (see with-dispatch). A call whose store or
   generative function changed (gfi/revise) still runs the operation asked
   for, so update middleware sees it."
  [gf transitions]
  (check-dynamic! "with-transitions" gf)
  (assoc gf :transitions transitions))

(defn with-dispatch
  "gf with an execution seam: (dispatch gf op args opts) runs before the
   handler and returns the operation's result map, or nil to leave it to the
   handler (opts as for run, without :batch-shape: a batched run calls the
   dispatcher under the batch tracer, where (gen.array.batch/batch-backend?
   gen.array/*backend*) holds). A dispatcher is an optimization: it must be
   observationally equivalent to the handler, which gen.laws checks
   (dispatch-matches-handler)."
  [gf dispatch]
  (assoc gf :dispatch dispatch))

(defmethod print-method DynamicGF [gf ^java.io.Writer w]
  (.write w (str "#gen/fn" (pr-str (first (:source gf))))))

;; ---------------------------------------------------------------------------
;; The gen macro: conditionals branch on values
;; ---------------------------------------------------------------------------

(defn case-key
  "case's dispatch value, checked: a value that differs across the particles
   of a batch cannot pick one clause (gen.array.batch), so it raises the
   batched-branch error rather than falling to the default clause; a 0-d
   backend array (an MLX scalar) dispatches on its host value, as on the JVM
   backend."
  [x]
  (if (instance? gen.array.protocols.IBatchedValue x)
    (throw (array/batched-branch "case on a batched value: it differs across particles" {}))
    (gen.dynamic.core/host-value "case on" x)))

(declare guard-branches)

(def ^:private guarded-core
  "The functions that test truthiness, look up by key or test
   representation, to their guarded versions in gen.dynamic.core (each names
   the function it guards in its :guards metadata)."
  (into {} (keep (fn [[s v]] (when-let [g (:guards (meta v))] [g (symbol "gen.dynamic.core" (name s))])))
        (ns-publics 'gen.dynamic.core)))

(def ^:dynamic ^:private *outer-locals*
  "The locals of the gen body being guarded, while forms inside it are
   macroexpanded: a gen (or guarded-fn) nested in it sees them."
  #{})

(defn- expand
  "form macroexpanded, with the locals around it visible to nested gens."
  [form locals]
  (binding [*outer-locals* locals] (macroexpand form)))

(defn- guard-symbol
  "sym, or its guarded version when it names one of guarded-core's
   functions (and is not a local)."
  [sym locals]
  (if (contains? locals sym)
    sym
    (let [v (try (resolve sym) (catch Exception _ nil))]
      (or (when (var? v) (guarded-core (symbol v))) sym))))

(defn- guard-body [forms locals] (map #(guard-branches % locals) forms))

(def ^:private lazy-fns
  '#{clojure.core/map clojure.core/map-indexed clojure.core/mapcat clojure.core/keep
     clojure.core/keep-indexed clojure.core/filter clojure.core/remove clojure.core/for
     gen.dynamic.core/filter gen.dynamic.core/remove})

(defn- discarded-lazy-trace?
  "Is form a lazy call (map, for, keep, …) whose trace or splice calls would
   never run because its value is discarded?"
  [form locals]
  (and (seq? form)
       (let [head (first form)]
         (and (symbol? head) (not (contains? locals head))
              (when-let [v (resolve head)] (lazy-fns (symbol v)))
              (some #{'trace 'splice 'untraced 'untraced-key} (tree-seq coll? seq (rest form)))))))

(defn- discarded-lazy-call
  "The lazy call with trace calls that would be discarded with form's value:
   form itself (as written, (for …), or macroexpanded, (->> xs (map …))), or
   the value of an if, do, let or letfn it expands to (when, cond, …)."
  [form locals]
  (let [expanded (if (and (seq? form) (symbol? (first form)) (not (contains? locals (first form))))
                   (expand form locals)
                   form)]
    (cond (or (discarded-lazy-trace? form locals) (discarded-lazy-trace? expanded locals)) form
          (not (seq? expanded)) nil
          :else (case (first expanded)
                  if (some #(discarded-lazy-call % locals) (drop 2 expanded))
                  do (some-> (last (rest expanded)) (discarded-lazy-call locals))
                  (let* loop* letfn*) (let [[_ bindings & body] expanded]
                                        (some-> (last body)
                                                (discarded-lazy-call (into locals (take-nth 2 bindings)))))
                  nil))))

(defn- guard-statements
  "A body's forms, guarded; every form but the last is a statement, whose
   value is discarded, so a lazy call there with trace calls is an error."
  [forms locals]
  (doseq [f (butlast forms)]
    (when-let [lazy (discarded-lazy-call f locals)]
      (throw (ex-info (str "In gen: " (pr-str (first lazy)) " here is lazy and its value is discarded,"
                           " so its trace calls would never run. Use doseq, run! or mapv.")
                      {:form f}))))
  (guard-body forms locals))

(defn- check-lazy-bindings
  "A let binding of a lazy call with trace calls whose symbol is never used:
   its trace calls would never run, so it is an error, as for a statement."
  [bindings body locals]
  (let [pairs (vec (partition 2 bindings))]
    (doseq [[i [sym init]] (map-indexed vector pairs)
            :let [expanded (if (and (seq? init) (symbol? (first init)) (not (contains? locals (first init))))
                             (expand init locals)
                             init)]]
      (when (and (symbol? sym)
                 (or (discarded-lazy-trace? init locals) (discarded-lazy-trace? expanded locals))
                 (not (some #{sym} (tree-seq coll? seq (concat (map second (subvec pairs (inc i))) body)))))
        (throw (ex-info (str "In gen: " sym " is bound to a lazy " (pr-str (first init)) " and never used,"
                             " so its trace calls would never run. Use doseq, run! or mapv.")
                        {:form init}))))))

(defn- guard-bindings
  "[bindings' locals'] for a let*/loop* binding vector."
  [bindings locals]
  (reduce (fn [[acc ls] [sym init]] [(conj acc sym (guard-branches init ls)) (conj ls sym)])
          [[] locals]
          (partition 2 bindings)))

(defn- guard-fn [[_ & more] locals]
  (let [[fname more] (if (symbol? (first more)) [(first more) (rest more)] [nil more])
        locals (cond-> locals fname (conj fname))
        arities (if (vector? (first more)) [more] more)]
    (list* 'fn* (concat (when fname [fname])
                        (for [[params & body] arities]
                          (list* params (guard-statements body (into locals (remove #{'&} params)))))))))

(defn- guard-special [form locals]
  (case (first form)
    (quote var) form
    if (let [[_ test then else] form]
         (list* 'if (list `array/truthy? (guard-branches test locals))
                (guard-branches then locals)
                (when (= 4 (count form)) [(guard-branches else locals)])))
    (let* loop*) (let [[op bindings & body] form
                       _ (when (= 'let* op) (check-lazy-bindings bindings body locals))
                       [bindings locals] (guard-bindings bindings locals)]
                   (list* op bindings (guard-statements body locals)))
    do (list* 'do (guard-statements (rest form) locals))
    fn* (guard-fn form locals)
    ;; (reify* [interfaces] (method [this params…] body…) …): the parameters are locals
    reify* (let [[_ interfaces & methods] form]
             (list* 'reify* interfaces
                    (for [[mname params & body] methods]
                      (list* mname params (guard-statements body (into locals params))))))
    ;; (. target member args…) or (. target (member args…)): a member name is
    ;; not a var, even when it is spelled like one (.get)
    . (let [[_ target member & args] form]
        (list* '. (guard-branches target locals)
               (if (seq? member)
                 (list* (first member) (guard-body (rest member) locals))
                 member)
               (guard-body args locals)))
    letfn* (let [[_ bindings & body] form
                 locals (into locals (take-nth 2 bindings))]
             (list* 'letfn*
                    (vec (mapcat (fn [[n f]] [n (guard-branches f locals)]) (partition 2 bindings)))
                    (guard-body body locals)))
    ;; a caught exception may be thrown for some particles only: in a batched
    ;; run, a catch clause raises the fallback first (gen.dynamic.core/caught)
    try (list* 'try (for [f (rest form)]
                      (cond (and (seq? f) (= 'catch (first f)))
                            (let [[_ cls sym & body] f]
                              (list* 'catch cls sym (list `gen.dynamic.core/caught sym)
                                     (guard-body body (conj locals sym))))
                            (and (seq? f) (= 'finally (first f)))
                            (list* 'finally (guard-body (rest f) locals))
                            :else (guard-branches f locals))))
    ;; case*'s dispatch value is a local symbol (case binds it): check it first
    ;; dispatching on case-key's value (case* needs a local: rebind it)
    case* (let [[_ ge shift mask default imap & more] form]
            (list 'let* [ge (list `case-key ge)]
                  (list* 'case* ge shift mask (guard-branches default locals)
                   (into (empty imap) (map (fn [[k [test then]]] [k [test (guard-branches then locals)]])) imap)
                   more)))
    (apply list (guard-body form locals))))

(defn- keep-meta
  "guarded, with the metadata of the form it came from (type hints, lines)."
  [form guarded]
  (if-let [m (meta form)]
    (if (instance? clojure.lang.IObj guarded) (with-meta guarded (merge (meta guarded) m)) guarded)
    guarded))

(defn- guard-branches
  "form, macroexpanded, with every if test wrapped in gen.array/truthy?, and
   the truthiness tests, lookups and representation tests of
   gen.dynamic.core in place of the functions they guard. locals are the
   symbols bound around form (never macroexpanded as heads)."
  [form locals]
  (cond
    (seq? form) (let [head (first form)]
                  (if (and (symbol? head) (contains? locals head))
                    (keep-meta form (apply list (guard-body form locals)))
                    (let [expanded (expand form locals)]
                      (if (seq? expanded)
                        (keep-meta expanded (guard-special expanded locals))
                        (guard-branches expanded locals)))))
    (vector? form) (keep-meta form (mapv #(guard-branches % locals) form))
    (map? form) (into (empty form) (map (fn [[k v]] [(guard-branches k locals) (guard-branches v locals)])) form)
    (set? form) (into (empty form) (map #(guard-branches % locals)) form)
    (symbol? form) (guard-symbol form locals)
    :else form))

(def ^:private primitive-coercions
  '{double clojure.core/double long clojure.core/long float clojure.core/float int clojure.core/int
    short clojure.core/short byte clojure.core/byte char clojure.core/char boolean clojure.core/boolean})

(defn- names-map
  "The form of a map from each of names (quoted) to its value."
  [names]
  (into {} (map (fn [s] [(list 'quote s) s])) names))

(defn- score-form
  "The form of a static body's score fn, (fn [param args choices key] →
   [weight retval]): the body's bindings in order as locals, a trace node's
   value from choices with its log density, a splice's callee assessed on
   its sub-map, keys split as the handler's assess splits them (none when
   key is nil)."
  [st params coercions locals]
  (let [args (gensym "args") choices (gensym "choices") key (gensym "key")
        {:keys [bindings terms]}
        (reduce (fn [{:keys [bindings terms k]} {:keys [kind addr binding expr] computed :keys}]
                  (let [;; a computed address's path, checked to hold every
                        ;; choice under its prefix
                        p (gensym "p")
                        at (when (some? addr)
                             (if computed
                               [p `(into ~(path addr) [~@computed])
                                '_ `(check-computed! ~choices ~(path addr) ~p ~(= kind :trace))]
                               [p (path addr)]))]
                    (case kind
                      :trace (let [v (gensym "v") lp (gensym "lp") k' (gensym "k")]
                               {:bindings (-> bindings (into at)
                                              (into [[v lp] `(static-site ~expr ~choices ~p)
                                                     binding v
                                                     k' `(when ~k (first (rng/split ~k)))]))
                                :terms (conj terms lp) :k k'})
                      :splice (let [[g as] expr w (gensym "w") rv (gensym "rv") k' (gensym "k") ks (gensym "ks")]
                                {:bindings (-> bindings (into at)
                                               (into [[k' ks] `(if ~k (rng/split ~k) [nil nil])
                                                      [w rv] `(static-call ~p ~g ~as (cm/get-submap ~choices ~p cm/EMPTY) ~ks)
                                                      binding rv]))
                                 :terms (conj terms w) :k k'})
                      {:bindings (into bindings [binding expr]) :terms terms :k k})))
                {:bindings [] :terms [] :k key}
                (:nodes st))]
    (guard-branches
     `(fn [~'param ~args ~choices ~key]
        (let [~params ~args ~@coercions ~@bindings]
          [(reduce array/+ 0.0 [~@terms]) ~(:expr (:ret st))]))
     locals)))

(defn- static-form
  "The runtime graph of the static body st (gen.dynamic.static/analyze),
   its names in slots (the arguments' first): per node, its kind, address,
   names and inputs, the slots it reads and writes, a guarded fn of (param,
   env) for its distribution, call or value, a fn writing its names' slots
   from its value, and how its diff is found (a rule, an alias, argument
   specs)."
  [st params coercions locals]
  (let [;; each name once: a name bound twice ([w w z]) holds its last binding, as let's
        arg-names (vec (distinct (concat (apply concat (:fixed (:args st))) (:rest (:args st)))))
        names (vec (distinct (concat arg-names (mapcat :names (:nodes st)))))
        slots (zipmap names (range))
        env (gensym "env")
        reads (fn [inputs] (vec (mapcat (fn [x] [x `(nth ~env ~(slots x))]) inputs)))
        node-fn (fn [{:keys [inputs expr]}]
                  (guard-branches `(fn [~'param ~env] (let ~(reads inputs) ~expr)) (into locals inputs)))
        keys-fn (fn [{:keys [inputs] ks :keys}]
                  (guard-branches `(fn [~'param ~env] (let ~(reads inputs) [~@ks])) (into locals inputs)))
        param? (fn [expr] (boolean (some #{'param} (tree-seq coll? seq expr))))
        spec (fn [{:keys [symbol constant inputs]}]
               (cond (and symbol (contains? slots symbol)) {:slot (slots symbol)}
                     (or symbol constant) {:constant true}
                     :else {:in (mapv slots inputs)}))
        call (fn [{:keys [call args arg-specs inputs alias]}]
               (cond call {:rule `(var ~call) :arg-specs (mapv spec arg-specs)
                           :args-f `(fn [~'param ~env] (let ~(reads inputs) [~@args]))}
                     alias {:alias (slots alias)}))
        v (gensym "v")
        args (gensym "args")]
    `{:args '~(:args st)
      :params '~params
      :slots '~slots
      :size ~(count names)
      :args-bind (fn [~args] (let [~params ~args ~@coercions] [~@arg-names ~@(repeat (- (count names) (count arg-names)) nil)]))
      :nodes [~@(for [n (:nodes st)]
                  (merge `{:kind ~(:kind n) :addr ~(:addr n) :names '~(:names n) :inputs '~(:inputs n) :form '~(:form n)
                           :in ~(mapv slots (:inputs n)) :out ~(mapv slots (:names n))
                           :symbol? ~(symbol? (:binding n))
                           :param? ~(param? [(:expr n) (:keys n)])
                           :f ~(node-fn n)
                           :keys-f ~(when (:keys n) (keys-fn n))
                           :bind (fn [~env ~v] (let [~(:binding n) ~v] (-> ~env ~@(map (fn [x] `(assoc ~(slots x) ~x)) (:names n)))))}
                         (when (= :splice (:kind n)) {:arg-specs (mapv spec (:arg-specs n))})
                         (when (:keys n) {:keys (list 'quote (:keys n))})
                         (call n)))]
      :ret ~(merge {:inputs `'~(:inputs (:ret st)) :in (mapv slots (:inputs (:ret st)))
                    :param? (param? (:expr (:ret st))) :f (node-fn (:ret st))}
                   (call (:ret st)))
      :score ~(score-form st params coercions locals)
      :roots ~(set (keep #(when-some [a (:addr %)] (first (path a))) (:nodes st)))
      ;; a node at a computed address holds its prefix, as a call does
      ;; (its score fn checks the rest: check-computed!)
      :paths ~(when (some #(vector? (:addr %)) (:nodes st))
                (vec (keep #(when-some [a (:addr %)] [(if (:keys %) :splice (:kind %)) (path a)]) (:nodes st))))}))

(defmacro gen
  "A generative function. params are the arguments (destructuring allowed);
   in body, (trace addr dist) and (splice addr gf & args) make random choices
   and calls, (param name) reads a trainable parameter (with-params), and
   conditionals branch on the values of 0-d arrays.

   (untraced dist) draws from dist without an address, and (untraced-key)
   gives other stochastic code (a simulator, an inner sampler) a key: the
   function's encapsulated randomness ω (thesis §4.5.2). Untraced values
   are not choices and add nothing to the score, so the score is an unbiased
   estimate of the choices' density given ω. The trace keeps ω, and an
   update or regenerate keeps it unless the change hints say something
   changed, or the store or generative function at the call did
   (gfi/revise): ω is then drawn afresh (DESIGN.md §9). Untraced values
   must not decide which addresses are visited.

   (gen ^:static [args] (let [bindings…] ret)) is a static body (Gen.jl's
   @gen (static)): each binding a call at a constant address or a pure
   expression, so update and regenerate skip the nodes that did not change,
   with the same results as without the flag (gen.dynamic.static, which
   also renders the graph). A body that is not static is a compile error.

   Conditionals, catch clauses, and clojure.core's truthiness tests and
   lookups (not, filter, every?, get, …; gen.dynamic.core), branch per
   particle in a batched run: on a value that differs across particles they
   raise the fallback to one particle at a time (gen.batched). Type tests
   (class, instance?, number?, …) answer for a particle's value.

   Helper functions the body calls are not rewritten: define one that
   branches on values with defn-guarded (or guarded-fn).

   Trace calls in a lazy sequence run when it is realized (chunking
   included); the body's return value is realized during the run. A lazy
   call with trace calls whose value is discarded (also as the value of a
   when, if, let or do in statement position), or bound and never used, is
   a compile error. Prefer mapv, doseq and run!."
  [params & body]
  ;; the body's untraced and untraced-key would silently shadow a var of
  ;; that name (an enclosing body's own are locals, and fine)
  (doseq [s '[untraced untraced-key]]
    (when (and (some #{s} (tree-seq coll? seq body)) (not (contains? &env s)) (var? (resolve &env s)))
      (throw (ex-info (str "gen: " s " in a body is gen's own (encapsulated randomness), but it would hide the var "
                           (resolve &env s) ": rename the var")
                      {:symbol s :var (resolve &env s)}))))
  (let [;; the body fn has four hidden parameters, and a fn with primitive
        ;; parameters takes at most four: coerce them in the body instead
        prim (filter #(and (symbol? %) (primitive-coercions (:tag (meta %)))) params)
        plain (mapv #(if (some #{%} prim) (vary-meta % dissoc :tag) %) params)
        coercions (mapcat (fn [p] [(vary-meta p dissoc :tag) (list (primitive-coercions (:tag (meta p))) (vary-meta p dissoc :tag))]) prim)
        locals (into (set (keys &env)) *outer-locals*)
        st (when (:static (meta params)) (static/analyze plain body &env locals))
        body (if (seq prim) [`(let [~@coercions] ~@body)] body)
        body-fn (guard-branches `(fn [~'trace ~'splice ~'param ~'untraced ~@plain]
                                   (let [~'untraced-key (fn [] (~'untraced))] ~@body))
                                locals)]
    ;; the body fn takes the four hidden parameters: no check (make-gen-fn's
    ;; reflects, and an inline gen is built on every run of its body)
    `(->DynamicGF ~body-fn '~(cons params body) nil nil nil ~(when st (static-form st plain coercions locals)))))

(defmacro guarded-fn
  "fn, with a gen body's guards: if and case tests branch through truthy?,
   catch clauses raise the fallback in a batched run, and clojure.core's
   truthiness, lookup and type tests are gen.dynamic.core's. A helper
   function that branches on values, defined so, raises the fallback in a
   batched run (gen.batched) where a plain fn would take one branch for
   every particle."
  [& fn-tail]
  (guard-branches `(fn ~@fn-tail) (into (set (keys &env)) *outer-locals*)))

(defmacro defn-guarded
  "defn with guarded-fn's guards: (defn-guarded name doc? [params] body…)."
  [fname & more]
  (let [[doc more] (if (string? (first more)) [(first more) (rest more)] [nil more])
        arglists (if (vector? (first more)) (list (first more)) (map first more))]
    `(def ~(vary-meta fname assoc :doc doc :arglists (list 'quote arglists))
       (guarded-fn ~fname ~@more))))
