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
  (:require [gen.array :as a]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist.core :as dc]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Addresses
;; ---------------------------------------------------------------------------

(defn- path [addr] (if (vector? addr) addr [addr]))

(defn- subselection-at
  "The selection for the subtree at path p."
  [s p]
  (reduce sel/get-subselection s p))

(defn- path-selected? [s p]
  (sel/selected? (subselection-at s (pop p)) (peek p)))

(defn- check-address!
  "A path may be used once, and never under or above another used path."
  [state p]
  (let [visited (:visited state)]
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
    (if (pos? (a/ndim lp)) (a/sum lp) lp)))

(defn- given-value
  "[v lp] of the value in the constraint c at a site. (Under the batch tracer
   an unbatched value is shared by every particle, and a batched one is per
   particle.)"
  [state dist c]
  (let [v (cm/get-value c)
        expected (dc/value-shape dist)]
    ;; a value of another shape would be scored as several draws, silently
    (when (and expected (not= expected (a/shape v)))
      (throw (ex-info (str (name (:mode state)) ": the value at this site has shape " (a/shape v)
                           ", but the distribution's values have shape " expected)
                      {:value-shape (a/shape v) :expected expected})))
    [v (site-log-prob dist v)]))

(defn- draw [_state dist k]
  (dc/sample dist k []))

(defn- record
  "state with the choice v (log probability lp) at path p."
  [state p v lp]
  (-> state
      (update :choices cm/set-choice p v)
      (update :scores assoc p lp)
      (update :score a/+ lp)))

(defn- add-weight [state w] (update state :weight a/+ w))

(defn- constraint-at
  "The constraint value at a primitive site, or nil. A nested choice map there
   is an error: it would otherwise be silently ignored."
  [state p op]
  (let [c (cm/get-submap (:constraints state) p)]
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
      (throw (ex-info (str "assess: no value for the choice at " p) {:path p})))
    (let [[v lp] (given-value state dist c)]
      [v (-> state (record p v lp) (add-weight lp))])))

(defn- update-site [state p dist k]
  (let [c (constraint-at state p :update)
        prev (prev-choice state p)]
    (cond
      c (let [[v lp] (given-value state dist c)]
          [v (cond-> (-> state (record p v lp) (add-weight lp))
               prev (-> (add-weight (a/negate (second prev)))
                        (update :discard cm/set-choice p (first prev))))])
      prev (let [[v prev-lp] prev
                 _ (dc/check-value-shape dist v)
                 lp (site-log-prob dist v)]
             [v (-> state (record p v lp) (add-weight (a/- lp prev-lp)))])
      :else (simulate-site state p dist k))))

(defn- regenerate-site [state p dist k]
  (let [prev (prev-choice state p)]
    (if (and prev (not (path-selected? (:selection state) p)))
      (let [[v prev-lp] prev
            _ (dc/check-value-shape dist v)
            lp (site-log-prob dist v)]
        [v (-> state (record p v lp) (add-weight (a/- lp prev-lp)))])
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


(defn- sub-key [state]
  (let [[k-next k-sub] (rng/split (:key state))]
    [k-sub (assoc state :key k-next)]))

(defn- arg-diffs
  "Argdiffs for a splice's new args against its previous call's: no-change
   for an identical argument (sound, and O(1)), unknown-change otherwise."
  [prev-args args]
  (mapv #(if (identical? %1 %2) diff/no-change diff/unknown-change)
        args (concat prev-args (repeat nil))))

(defn- record-call
  "state with the sub-trace tr at path p (score sc)."
  [state p tr]
  (cond-> (-> state
              (update :subtraces assoc p tr)
              (update :score a/+ (:score tr)))
    (not (cm/empty? (:choices tr))) (update :choices cm/set-choice p (:choices tr))))

(declare dynamic-gf?)

(defn compatible-callee?
  "Can gf update (or regenerate) old's sub-trace? A sub-trace holds choices
   by address, so any dynamic generative function can run against another's
   (the addresses they share keep their values, as with if), and any
   distribution against another's single value; a combinator needs one of its
   own type, with compatible kernels."
  [old gf]
  (cond (identical? old gf) true
        (and (instance? gen.dist.core.Distribution old) (instance? gen.dist.core.Distribution gf)) true
        (and (dynamic-gf? old) (dynamic-gf? gf)) true
        (and (record? old) (gfi/generative-function? old))
        (and (= (class old) (class gf))
             (every? (fn [k] (let [x (get old k)]
                               (or (not (gfi/generative-function? x)) (compatible-callee? x (get gf k)))))
                     (keys old)))
        :else (= old gf)))

(defn- replace-call
  "state without the previous call at p (its choices discarded and, in
   update, its score subtracted): a different generative function now sits
   at p, so the old call is gone and the new one is new."
  [state p old]
  (if (= (:mode state) :update)
    (cond-> (add-weight state (a/negate (:score old)))
      (not (cm/empty? (:choices old))) (update :discard cm/set-choice p (:choices old)))
    state))

(defn- splice-call
  "Call gf at path p under the run's operation, scalar or batched (through
   gfi/run-op), nesting its choices and recording its sub-trace."
  [state p gf args]
  (let [mode (:mode state)
        cons (cm/get-submap (:constraints state) p)
        old-sub (get (::subtraces (:prev state)) p)
        replaced? (and old-sub (not (compatible-callee? (:gen-fn old-sub) gf)))
        prev-sub (when-not replaced? old-sub)
        state (if replaced? (replace-call state p old-sub) state)
        ;; a batched run's callees run batched too: they inherit the batch tracer
        base {}]
    (if (= mode :assess)
      (let [{w :weight rv :retval} (gfi/run-op gf :assess args (assoc base :constraints cons :key (rng/key 0)))]
        [rv (cond-> (-> state (update :score a/+ w) (add-weight w))
              (not (cm/empty? cons)) (update :choices cm/set-choice p cons))])
      (let [[k state] (sub-key state)
            opts (assoc base :key k)
            r (cond
                (= mode :simulate) (gfi/run-op gf :simulate args opts)
                (= mode :generate) (gfi/run-op gf :generate args (assoc opts :constraints cons))
                (nil? prev-sub) (if (= mode :update)
                                  (gfi/run-op gf :generate args (assoc opts :constraints cons))
                                  (gfi/run-op gf :simulate args opts))
                (= mode :update) (gfi/run-op gf :update args (assoc opts :prev prev-sub :constraints cons
                                                                    :argdiffs (arg-diffs (:args prev-sub) args)))
                :else (gfi/run-op gf :regenerate args (assoc opts :prev prev-sub
                                                             :argdiffs (arg-diffs (:args prev-sub) args)
                                                             :selection (subselection-at (:selection state) p))))
            {tr :trace w :weight d :discard} r]
        [(:retval tr) (cond-> (record-call state p tr)
                        w (add-weight w)
                        (and d (not (cm/empty? d))) (update :discard cm/set-choice p d))]))))

;; ---------------------------------------------------------------------------
;; Running a body
;; ---------------------------------------------------------------------------

(defn- check-constraints-visited!
  "Every constraint must be consumed by a choice or a splice (as in Gen.jl)."
  [state op]
  (let [visited (:visited state)]
    (doseq [p (cm/addresses (:constraints state))]
      (when-not (some #(contains? visited (subvec p 0 %)) (range 1 (inc (count p))))
        (throw (ex-info (str (name op) ": the constraint at " p " was not visited")
                        {:path p :op op}))))))

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
          (reduce (fn [s p] (-> s (add-weight (a/negate (get (::scores prev) p)))
                                (update :discard cm/set-choice p (cm/get-choice (:choices prev) p))))
                  s gone-choices)
          (reduce (fn [s p] (let [sub (get (::subtraces prev) p)]
                              (cond-> (add-weight s (a/negate (:score sub)))
                                (not (cm/empty? (:choices sub)))
                                (update :discard cm/set-choice p (:choices sub)))))
                  s gone-calls)))
      ;; regenerate: choices no longer visited cancel against the reverse move
      state)))

(defn- initial-state [mode {:keys [key constraints prev selection]}]
  {:mode mode :key key
   :choices cm/EMPTY :scores {} :subtraces {} :score 0.0 :weight 0.0
   :constraints (or constraints cm/EMPTY) :prev prev :selection (or selection sel/none)
   :discard cm/EMPTY :visited #{}})

(def ^:dynamic *params*
  "The trainable parameters in effect: the store of the outermost generative
   function that has one, merged with any inner one's (with-params)."
  nil)

(defn- read-param
  "(param name) in a body: the parameter's value in the store in effect."
  [name]
  (let [v (get *params* name ::missing)]
    (if (= ::missing v)
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

(defn- execute
  "Run gf's body on args under the transitions of mode; the final state with
   :retval."
  [gf mode args opts]
  (let [;; the transition in effect: a splice without an address swaps in its
        ;; callee's middleware for the callee's body
        transition (volatile! (or (get (:transitions gf) mode) (site-transitions mode)))
        vol (volatile! (initial-state mode opts))
        trace-fn (fn [addr dist]
                   (let [p (path addr)
                         state @vol]
                     (check-running! state p)
                     (when-not (instance? gen.dist.core.Distribution dist)
                       (throw (ex-info (str "trace at " p " needs a distribution; call generative functions with splice")
                                       {:path p :value dist})))
                     (check-address! state p)
                     (let [[k-next k-site] (rng/split (:key state))
                           [v state] (@transition (assoc state :key k-next) p dist k-site)]
                       (vreset! vol (update state :visited conj p))
                       v)))
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
                        (let [outer @transition
                              run-body #(apply (:body-fn sub) trace-fn splice-fn read-param args)]
                          (vreset! transition (or (get (:transitions sub) mode) (site-transitions mode)))
                          (try
                            (if-let [ps (:params sub)]
                              (binding [*params* (merge *params* ps)] (run-body))
                              (run-body))
                            (finally (vreset! transition outer)))))
                      (let [[addr gf & args] xs
                            ;; trace calls in lazy arguments happen now, in this body
                            args (realize! (vec args))
                            p (path addr)
                            state @vol]
                        (check-running! state p)
                        (when-not (gfi/generative-function? gf)
                          (throw (ex-info (str "splice at " p " needs a generative function") {:path p :value gf})))
                        (check-address! state p)
                        (let [[v state'] (splice-call state p gf args)]
                          (when-not (identical? state @vol)
                            (throw (ex-info (str "The caller's trace was used during the splice at " p
                                                 ": a function passed to the callee calls trace")
                                            {:path p})))
                          (vreset! vol (update state' :visited conj p))
                          v))))
        retval (realize! (apply (:body-fn gf) trace-fn splice-fn read-param args))]
    (vswap! vol assoc ::finished true)
    (assoc (finish @vol) :retval retval)))

(defn- make-trace [gf args state]
  (assoc (trace/make-trace {:gen-fn gf :args args :choices (:choices state)
                            :retval (:retval state) :score (:score state)})
         ::scores (:scores state)
         ::subtraces (:subtraces state)))

(defn- retdiff
  "no-change when the return value is equal to the previous one (sound: equal
   values cannot differ), unknown-change otherwise."
  [prev-trace state]
  ;; batched return values cannot be compared (gen.array.batch): unknown
  (if (try (= (:retval prev-trace) (:retval state))
           (catch clojure.lang.ExceptionInfo e
             (if (:gen.array/batched-branch (ex-data e)) false (throw e))))
    diff/no-change
    diff/unknown-change))

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
                (binding [*params* (merge *params* ps)] (execute gf op args opts))
                (execute gf op args opts))]
    (case op
      :simulate {:trace (make-trace gf args state)}
      :generate {:trace (make-trace gf args state) :weight (:weight state)}
      :assess {:weight (:weight state) :retval (:retval state)}
      :update {:trace (make-trace gf args state) :weight (:weight state)
               :retdiff (retdiff (:prev opts) state) :discard (:discard state)}
      :regenerate {:trace (make-trace gf args state) :weight (:weight state)
                   :retdiff (retdiff (:prev opts) state)})))

(defn- project* [tr s]
  (let [from-choices (reduce (fn [acc [p lp]] (if (path-selected? s p) (a/+ acc lp) acc))
                             0.0 (::scores tr))]
    (reduce (fn [acc [p sub]] (a/+ acc (gfi/-project (:gen-fn sub) sub (subselection-at s p))))
            from-choices (::subtraces tr))))

;; ---------------------------------------------------------------------------
;; DynamicGF
;; ---------------------------------------------------------------------------

(defrecord DynamicGF [body-fn source params transitions dispatch]
  gfi/IGenerativeFunction
  (-simulate [gf args key] (:trace (run gf :simulate args {:key key})))
  (-generate [gf args constraints key] (run gf :generate args {:key key :constraints constraints}))
  (-update [gf tr args _argdiffs constraints key]
    (run gf :update args {:key key :prev tr :constraints constraints}))
  (-regenerate [gf tr args _argdiffs selection key]
    (run gf :regenerate args {:key key :prev tr :selection selection}))
  (-assess [gf args choices] (run gf :assess args {:key (rng/key 0) :constraints choices}))
  (-propose [gf args key]
    (let [tr (:trace (run gf :simulate args {:key key}))]
      {:choices (:choices tr) :weight (:score tr) :retval (:retval tr)}))
  (-project [_ tr selection] (project* tr selection))

  ;; an untraced call, as in Gen.jl: (gf a b) is the return value of a simulated run
  clojure.lang.IFn
  (invoke [this] (:retval (gfi/simulate this [])))
  (invoke [this a] (:retval (gfi/simulate this [a])))
  (invoke [this a b] (:retval (gfi/simulate this [a b])))
  (invoke [this a b c] (:retval (gfi/simulate this [a b c])))
  (invoke [this a b c d] (:retval (gfi/simulate this [a b c d])))
  (invoke [this a b c d e] (:retval (gfi/simulate this [a b c d e])))
  (invoke [this a b c d e f] (:retval (gfi/simulate this [a b c d e f])))
  (applyTo [this args] (:retval (gfi/simulate this (vec args)))))

(defn dynamic-gf? [x] (instance? DynamicGF x))

(defn make-gen-fn
  "A dynamic generative function from a body fn of (trace splice & args) and
   its source form."
  [body-fn source]
  (->DynamicGF body-fn source nil nil nil))

(defn with-params
  "gf with the trainable-parameter store params, a map of name to value read
   by (param name) in gf's body and in every generative function it calls."
  [gf params]
  (assoc gf :params params))

(defn with-transitions
  "gf with handler middleware: a map from operation (:simulate :generate
   :assess :update :regenerate) to a transition (fn [state path dist key] →
   [value state']) used in place of the standard one (site-transitions),
   typically by wrapping it. Middleware may change the model (e.g. masking
   a distribution); an optimization must not (see with-dispatch)."
  [gf transitions]
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
   batched-branch error rather than falling to the default clause."
  [x]
  (if (instance? gen.array.protocols.IBatchedValue x)
    (throw (ex-info "case on a batched value: it differs across particles" {:gen.array/batched-branch true}))
    x))

(declare guard-branches)

(defn- guard-body [forms locals] (map #(guard-branches % locals) forms))

(def ^:private lazy-fns
  '#{clojure.core/map clojure.core/map-indexed clojure.core/mapcat clojure.core/keep
     clojure.core/keep-indexed clojure.core/filter clojure.core/remove clojure.core/for})

(defn- discarded-lazy-trace?
  "Is form a lazy call (map, for, keep, …) whose trace or splice calls would
   never run because its value is discarded?"
  [form locals]
  (and (seq? form)
       (let [head (first form)]
         (and (symbol? head) (not (contains? locals head))
              (when-let [v (resolve head)] (lazy-fns (symbol v)))
              (some #{'trace 'splice} (tree-seq coll? seq (rest form)))))))

(defn- guard-statements
  "A body's forms, guarded; every form but the last is a statement, whose
   value is discarded, so a lazy call there with trace calls is an error."
  [forms locals]
  (doseq [f (butlast forms)
          ;; as written ((for …)) and macroexpanded ((->> xs (map …)))
          :let [expanded (if (and (seq? f) (symbol? (first f)) (not (contains? locals (first f)))) (macroexpand f) f)]]
    (when (or (discarded-lazy-trace? f locals) (discarded-lazy-trace? expanded locals))
      (throw (ex-info (str "In gen: " (pr-str (first f)) " here is lazy and its value is discarded,"
                           " so its trace calls would never run. Use doseq, run! or mapv.")
                      {:form f}))))
  (guard-body forms locals))

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
         (list* 'if (list `a/truthy? (guard-branches test locals))
                (guard-branches then locals)
                (when (= 4 (count form)) [(guard-branches else locals)])))
    (let* loop*) (let [[op bindings & body] form
                       [bindings locals] (guard-bindings bindings locals)]
                   (list* op bindings (guard-statements body locals)))
    do (list* 'do (guard-statements (rest form) locals))
    fn* (guard-fn form locals)
    letfn* (let [[_ bindings & body] form
                 locals (into locals (take-nth 2 bindings))]
             (list* 'letfn*
                    (vec (mapcat (fn [[n f]] [n (guard-branches f locals)]) (partition 2 bindings)))
                    (guard-body body locals)))
    try (list* 'try (for [f (rest form)]
                      (cond (and (seq? f) (= 'catch (first f)))
                            (let [[_ cls sym & body] f] (list* 'catch cls sym (guard-body body (conj locals sym))))
                            (and (seq? f) (= 'finally (first f)))
                            (list* 'finally (guard-body (rest f) locals))
                            :else (guard-branches f locals))))
    ;; case*'s dispatch value is a local symbol (case binds it): check it first
    case* (let [[_ ge shift mask default imap & more] form]
            (list 'do (list `case-key ge)
                  (list* 'case* (guard-branches ge locals) shift mask (guard-branches default locals)
                   (into (empty imap) (map (fn [[k [test then]]] [k [test (guard-branches then locals)]])) imap)
                   more)))
    (apply list (guard-body form locals))))

(defn- guard-branches
  "form, macroexpanded, with every if test wrapped in gen.array/truthy?.
   locals are the symbols bound around form (never macroexpanded as heads)."
  [form locals]
  (cond
    (seq? form) (let [head (first form)]
                  (if (and (symbol? head) (contains? locals head))
                    (apply list (guard-body form locals))
                    (let [expanded (macroexpand form)]
                      (if (seq? expanded) (guard-special expanded locals) (guard-branches expanded locals)))))
    (vector? form) (mapv #(guard-branches % locals) form)
    (map? form) (into (empty form) (map (fn [[k v]] [(guard-branches k locals) (guard-branches v locals)])) form)
    (set? form) (into (empty form) (map #(guard-branches % locals)) form)
    :else form))

(defmacro gen
  "A generative function. params are the arguments (destructuring allowed);
   in body, (trace addr dist) and (splice addr gf & args) make random choices
   and calls, (param name) reads a trainable parameter (with-params), and
   conditionals branch on the values of 0-d arrays."
  [params & body]
  (let [body-fn (guard-branches `(fn [~'trace ~'splice ~'param ~@params] ~@body) (set (keys &env)))]
    `(make-gen-fn ~body-fn '~(cons params body))))
