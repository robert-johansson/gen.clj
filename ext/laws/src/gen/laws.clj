(ns gen.laws
  "The GFI's laws (Cusumano-Towner 2020; docs/research/genjl-parity.md §3,
   audited in docs/research/law-audit.md) as test.check properties over any
   generative functions, and the means to run them on yours.

   The module ext/laws holds it, outside gen.clj's core: it depends on
   gen.clj and test.check (ext/laws/deps.edn). A project takes it as a git
   dependency on gen.clj's repository with :deps/root \"ext/laws\".

   Checking a generative function. (verify models) runs every law of all
   over cases of models, and returns the failures: a map from each law that
   failed to what it says (the check that failed and the values it
   compared, or the error thrown, and the smallest failing case), {} when
   every law holds. (verify models opts laws) runs a law set on opts'
   cases. In a test namespace, (deflaws prefix models opts laws) defines a
   test.check spec per law.

   Models are [label gf args alt-args]: alt-args is a vector of other
   argument vectors, so that update and regenerate are also checked with
   changed arguments. A law set is a map from a law's name to the law, a
   function from cases (cases) to a test.check property. The sets:

   - general: the thesis's definitions, true of every generative function
     (Def 2.1.16, Def 2.3.1, Prop 2.3.1, Def 2.3.2, Eq 4.1's retention);
   - forward-sampling: true when the internal proposal samples unconstrained
     choices from the prior, each site from its own key, as every gen.clj
     generative function does so far. A generative function with an optimal
     or overridden internal proposal (thesis pp. 145-152) need satisfy only
     the general set;
   - all: general and forward-sampling;
   - fixed-support: the support condition, for models whose every address
     has a fixed support;
   - static: a static generative function against the same body without
     ^:static;
   - encapsulated: for a generative function with encapsulated randomness
     (§4.5), whose score is an estimate, the laws that hold for one draw of
     it; estimated: those of them that hold for an estimated density with a
     joint sampler (gen.custom/estimated with :simulate);
   - across: an update or regenerate of a trace that another generative
     function made (gf under another store, say), which keeps only its
     choices (gfi/revise);
   - implementation: properties of gen.clj's own generative functions that
     the thesis does not require, so that a caller can skip work.

   Writing a law. A law is a function from cases to a test.check property
   (prop/for-all over them) whose value is a test.check result: (checks …)
   of its comparisons names the one that failed and the values it
   compared, and vacuous marks a case the law says nothing about. check
   runs one law's property, counting its vacuous cases, and fails a law
   more than max-vacuous of whose cases were vacuous: it has stopped
   testing anything. A map {:name law} is a law set that verify and deflaws
   run as they run gen.clj's.

   Vacuity. A law draws its constraints from another run, which may make
   choices this run cannot visit, and the interface rejects them loudly.
   Such a case is vacuous (vacuous-on-unvisitable): the operation threw an
   ex-info whose message says a constraint was not visited, or that one is
   a nested choice map or a single value where the other is expected
   (unvisitable?). The rule reads the message, so a generative function
   written by hand must throw gfi/unvisited's error for constraints it does
   not visit: another error fails the law."
  {:api :stable}
  (:refer-clojure :exclude [+ -])
  (:require [clojure.test.check :as tc]
            [clojure.test.check.clojure-test :as ctc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.results :as results]
            [gen.array :as array :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

;; ---------------------------------------------------------------------------
;; Values on the host, within the backend's precision
;; ---------------------------------------------------------------------------

(defn- f32?
  "Is the default float of the default backend float32 (MLX's GPUs, or
   MLX without -Dgen.mlx.dtypes=64)?"
  []
  (= :float32 (:float (array/defaults array/*backend*))))

(defn close?
  "Are weights x and y equal within the backend's precision? Within
   float64's round-off (a relative and an absolute 1e-12), or, on a float32
   backend, a relative 1e-4 and an absolute 1e-3: a weight is a difference
   of scores, so its error scales with the scores rather than with it, a
   few ulps of scores of up to ~100."
  [x y]
  (array/allclose? x y (if (f32?) {:rtol 1e-4 :atol 1e-3} {:rtol 1e-12 :atol 1e-12})))

(defn- number
  "The host double of a 0-d (or one-element) array or number."
  ^double [x]
  (double (array/item x)))

(defn- host
  "x with every backend array in it (in nested maps, vectors and seqs, and
   choice maps, as maps) read on the host."
  [x]
  (cond (array/backend-array? x) (array/->clj x)
        (cm/choicemap? x) (host (cm/to-map x))
        (map? x) (if (record? x) x (into (empty x) (map (fn [[k v]] [k (host v)])) x))
        (vector? x) (mapv host x)
        (seq? x) (doall (map host x))
        :else x))

(defn- zero-weight?
  "Is the weight w exactly zero (a host number, or a 0-d array of any
   backend)?"
  [w]
  (== 0.0 (number w)))

;; ---------------------------------------------------------------------------
;; Cases
;; ---------------------------------------------------------------------------

(def seeds
  "The seeds of the cases' keys, any of 2^31. gen/nat is bounded by
   test.check's size, so a run of 100 cases drew its keys from 0-99, half
   of them twice, and every run the same ones (2026-10-24). Planted bug: a
   normal sampler whose draws beyond 3.25 standard deviations are NaN (an
   erfinv edge) failed 2 of 240 runs of a dynamic law (24 laws, 10 runs
   each) under those keys, and 46 under these."
  (gen/choose 0 Integer/MAX_VALUE))

(defn- model-entries? [models]
  (and (sequential? models) (seq models)
       (every? #(and (sequential? %) (<= 3 (count %))) models)))

(defn cases
  "A test.check generator of cases [[label gf args alt-args] seed picks
   alt?] over models: a vector of model entries, or a generator of them
   (e.g. random models). A law's operations take keys from seed; picks says
   for each address, in order, whether a law constrains or selects it,
   cycled after 32 (with 8, addresses 8 apart were always picked together,
   and a rule bug whose case needs one picked and the other not passed: the
   confirmation review, 2026-10-20); alt? whether it changes the arguments
   to one of alt-args. Models that are neither are a loud error."
  [models]
  (when-not (or (gen/generator? models) (model-entries? models))
    (throw (ex-info (str "laws: models are a non-empty vector of model entries [label gf args alt-args],"
                         " or a generator of them")
                    {:models models})))
  (gen/tuple (if (gen/generator? models) models (gen/elements models))
             seeds (gen/vector gen/boolean 32) gen/boolean))

;; ---------------------------------------------------------------------------
;; Results: failures that say what failed, and vacuous cases
;; ---------------------------------------------------------------------------

(defn- failed
  "A failing test.check result whose data is data."
  [data]
  (reify results/Result
    (pass? [_] false)
    (result-data [_] data)))

(def vacuous
  "The result of a case a law says nothing about: it passes, and check and
   deflaws count it."
  (reify results/Result
    (pass? [_] true)
    (result-data [_] {::vacuous true})))

(defn vacuous?
  "Is r a vacuous case's result?"
  [r]
  (identical? vacuous r))

(defn- printable
  "x with each seq cut to its first 10 elements, so that a failure report
   prints: a return value may be infinite."
  [x]
  (cond (seq? x) (let [xs (take 11 x)]
                   (cond-> (map printable (take 10 xs)) (= 11 (count xs)) (concat ['...])))
        (map? x) (into x (map (fn [[k v]] [k (printable v)])) x)
        (coll? x) (into (empty x) (map printable) x)
        :else x))

(defn- constant? [form]
  (or (nil? form) (boolean? form) (number? form) (keyword? form) (string? form)
      (and (seq? form) (contains? '#{fn fn*} (first form)))))

(defn- call-of
  "The var or local a check form calls as a function, or nil when the form
   is not a call (a special form, a macro, a value)."
  [env form]
  (when (and (seq? form) (symbol? (first form)) (not (special-symbol? (first form))))
    (let [f (first form)]
      (if (contains? env f)
        f
        (let [v (resolve env f)]
          (when-not (:macro (meta v)) (or v f)))))))

(defn ^:no-doc failure
  "The failure of a check (checks' expansion calls it): v's own when v is a
   failure (a nested checks), else what explain says, or the values of the
   check's arguments."
  [v form values explain]
  (if (results/result-data v)
    v
    (failed (merge {:check form} (printable (if explain (explain) {:values values}))))))

(defmacro checks
  "and over a law's checks, as a test.check result: true when every check
   passes, vacuous when every one was vacuous, and otherwise a failure
   naming the first that failed, as its form (:check), with the values of
   its arguments (:values) when it is a function call. A function whose
   var has :gen.laws/explain metadata, a function of the same arguments,
   gives a map of what to show in place of the values. A check whose value
   is a result (a nested checks, or vacuous) passes or fails as that
   result.

     (prop/for-all [[[_ gf args] seed] cases]
       (let [tr (gfi/simulate gf args (rng/key seed))]
         (laws/checks (laws/close? (gfi/project tr sel/all) (:score tr)))))"
  [& forms]
  (letfn [(chain [[form & more] checked]
            (let [v (gensym "v")
                  f (call-of &env form)
                  args (when f (rest form))
                  as (repeatedly (count args) #(gensym "arg"))
                  checked' (gensym "checked")
                  explain (when (and (var? f) (::explain (meta f)))
                            `(fn [] ((::explain (meta (var ~(symbol f)))) ~@as)))
                  values `(into {} ~(vec (keep (fn [[a s]] (when-not (constant? a) [(list 'quote a) s])) (map vector args as))))]
              `(let [~@(interleave as args)
                     ~v ~(if f (cons (first form) as) form)]
                 (if (results/pass? ~v)
                   (let [~checked' (or ~checked (not (vacuous? ~v)))]
                     ~(if (seq more) (chain more checked') `(if ~checked' true vacuous)))
                   (failure ~v '~form ~values ~explain)))))]
    (if (seq forms) (chain forms false) true)))

(defn unvisitable?
  "Is e an error for constraints drawn from another run that this run
   cannot visit (a structural mismatch the GFI correctly rejects)? Its
   message says a constraint was not visited (gfi/unvisited's error), or
   that one is a nested choice map or a single value where the other is
   expected."
  [^Throwable e]
  (boolean (re-find #"not visited|nested choice map|single value" (or (.getMessage e) ""))))

(defmacro vacuous-on-unvisitable
  "body's value, or vacuous when body throws an ex-info that unvisitable?
   says is an unvisitable constraint (any other error is thrown): for an
   operation given constraints from another run.

     (let [r (laws/vacuous-on-unvisitable (gfi/generate gf args sigma key))]
       (if (laws/vacuous? r) r (laws/checks …)))"
  [& body]
  `(try ~@body
        (catch clojure.lang.ExceptionInfo e#
          (if (unvisitable? e#) vacuous (throw e#)))))


;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- addrs [tr] (cm/addresses (:choices tr)))

(defn- picked [paths picks] (vec (keep (fn [[p pick?]] (when pick? p)) (map vector paths (cycle picks)))))

(defn- picked-apart
  "Two disjoint parts of paths: those picks picks, and of the rest those
   picks picks one place on."
  [paths picks]
  (let [ps (map vector paths (cycle picks) (rest (cycle picks)))]
    [(vec (keep (fn [[p a]] (when a p)) ps))
     (vec (keep (fn [[p a b]] (when (and (not a) b) p)) ps))]))

(defn- restrict [choices paths] (sel/get-selected choices (sel/from-paths paths)))

(defn- lp
  "The log density of the choices at paths in tr."
  [tr paths]
  (gfi/project tr (sel/from-paths paths)))

(defn- next-args
  "The arguments to update or regenerate with: an alternative, or the same."
  [[_ _ args alt] seed alt?]
  (if (and alt? (seq alt)) (nth alt (mod seed (count alt))) args))

(defn- unknown-diffs [args] (vec (repeat (count args) diff/unknown-change)))

(defn- finite? [w] (array/truthy? (array/all (array/isfinite w))))

(defn- in-support? [tr] (finite? (:score tr)))

(defn- outside?
  "Is t2, a move's new trace, outside the support (its score -Inf)? The
   move then weighs -Inf (the ten decisions' fifth, DESIGN.md §9), which is
   all the weight laws check of it: a run that stops there holds the
   choices up to it, but a batched or lazy one (MLX) cannot stop, and its
   densities after that choice are what it computed (NaN where it made a
   parameter invalid), so no projection of them is a weight. Planted bug
   (2026-10-25): the weight laws comparing projections there too fail the
   out-of-support models on MLX's CPU."
  [t2]
  (= ##-Inf (number (:score t2))))

(defn- minus-infinity? [w] (= ##-Inf (number w)))

(defn- same-retval?
  "Are return values a and b equal (=), two seqs on the part of them a run
   realizes, their first gfi/realize-limit elements? = would walk an
   infinite one forever."
  [a b]
  (if (and (seq? a) (seq? b))
    (= (take gfi/realize-limit a) (take gfi/realize-limit b))
    (= a b)))

(defn- assessment
  "tr's score and return value, and those gf assesses its choices to have."
  [gf tr]
  (let [{w :weight rv :retval} (gfi/assess gf (:args tr) (:choices tr) (rng/key 0))]
    {:score (:score tr) :assessed w :retval (:retval tr) :assessed-retval rv}))

(defn- assessed?
  "Is tr's score gf's assessed density of its choices, and its return value
   the one they determine?"
  {::explain assessment}
  [gf tr]
  (let [{:keys [score assessed retval assessed-retval]} (assessment gf tr)]
    (and (close? score assessed) (same-retval? retval assessed-retval))))

(defn- consistent?
  "Trace consistency (Def 2.1.16): tr's score and return value are those its
   gen-fn assesses."
  {::explain #(assessment (:gen-fn %) %)}
  [tr]
  (assessed? (:gen-fn tr) tr))

(defn- sound-retdiff?
  "Def 2.3.2: the retdiff describes the change of the return value
   (gen.diff/consistent?): no-change an equal value; a vector-diff the two
   lengths and equal elements below both outside :updated (law audit gap
   F: only no-change was checked); an int-, set- or dict-diff its change."
  [old {tr :trace rd :retdiff}]
  (boolean (diff/consistent? (:retval old) (:retval tr) rd)))

(defn- update-case
  "A random update: constraints are a random subset of another run's choices
   (with the new arguments), so they may change values, add addresses and
   change structure. Given gf', the other run is gf''s and the trace is
   updated as gf''s (gfi/revise). nil when the update rejects the
   constraints as unvisitable."
  ([model seed picks alt?] (update-case model seed picks alt? nil))
  ([[_ gf args :as model] seed picks alt? gf']
   (let [k (rng/key seed)
         tr (gfi/simulate gf args k)
         args' (next-args model seed alt?)
         other (gfi/simulate (or gf' gf) args' (rng/fold-in k 1))
         sigma (restrict (:choices other) (picked (addrs other) picks))
         r (vacuous-on-unvisitable
            (gfi/update (cond-> tr gf' (assoc :gen-fn gf')) args' (unknown-diffs args') sigma (rng/fold-in k 2)))]
     (when-not (vacuous? r)
       {:tr tr :args' args' :sigma sigma :result r}))))

(defn- fresh
  "The choices of t2, an update of tr with sigma, drawn afresh: neither tr's
   nor constrained."
  [tr t2 sigma]
  (let [was (set (addrs tr))
        constrained (set (cm/addresses sigma))]
    (remove #(or (was %) (constrained %)) (addrs t2))))

(defn- kept
  "The choices of t2, an update of tr with sigma, not drawn afresh: kept
   from tr or constrained."
  [tr t2 sigma]
  (remove (set (fresh tr t2 sigma)) (addrs t2)))

(defn- exact-discard
  "Def 2.3.1: the discard of an update of tr to t2 with sigma is τ
   restricted to (A_τ ∩ A_σ) ∪ (A_τ ∖ A_τ')."
  [tr t2 sigma]
  (let [now (set (addrs t2))
        constrained (set (cm/addresses sigma))]
    (restrict (:choices tr) (filter #(or (constrained %) (not (now %))) (addrs tr)))))

(defn- reached
  "sigma, the constraints of an update to t2: those t2 holds when its run
   stopped at a choice outside its support (its score -Inf on the host,
   gfi/impossible?), which leaves the constraints after it unvisited (the
   ten decisions' fifth, DESIGN.md §9). Planted bug (2026-10-25): an update
   that goes on past such a choice (gen.dynamic's weigh never stopping)
   throws in a model's downstream parameter check and fails the dynamic
   suite's out-of-support laws."
  [sigma t2]
  (if (gfi/impossible? (:score t2)) (restrict sigma (addrs t2)) sigma))

(defn- applied?
  "Does t2, an update of tr with sigma, hold sigma's values (those it
   reached), and tr's unconstrained choices that remain unchanged?"
  [tr t2 sigma]
  (let [now (set (addrs t2))
        constrained (set (cm/addresses (reached sigma t2)))]
    (and (every? #(= (cm/get-choice sigma %) (cm/get-choice (:choices t2) %)) constrained)
         (every? #(= (cm/get-choice (:choices tr) %) (cm/get-choice (:choices t2) %))
                 (remove constrained (filter now (addrs tr)))))))

;; ---------------------------------------------------------------------------
;; General laws
;; ---------------------------------------------------------------------------

(defn score-is-the-assessed-density
  "Trace consistency (Def 2.1.16): a simulated trace's score is the
   density assess gives its choices, and the return values agree."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (checks (consistent? (gfi/simulate gf args (rng/key seed))))))

(defn generate-with-every-choice-is-deterministic
  "generate constrained on every choice of a trace makes those choices,
   with the trace's score as its weight."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {g :trace w :weight} (gfi/generate gf args (:choices tr) (rng/key (inc seed)))]
      (checks (= (:choices tr) (:choices g)) (close? w (:score tr))))))

(defn generated-traces-are-consistent
  "generate constrained on some of a trace's choices keeps them and makes
   a consistent trace, and its weight is not NaN."
  [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          sigma (restrict (:choices tr) (picked (addrs tr) picks))
          ;; another key may take another branch, leaving a constraint unvisited
          r (vacuous-on-unvisitable (gfi/generate gf args sigma (rng/key (inc seed))))]
      (if (vacuous? r)
        r
        (let [{g :trace w :weight} r]
          (checks (consistent? g)
                  (not (array/truthy? (array/isnan w)))
                  (every? #(= (cm/get-choice sigma %) (cm/get-choice (:choices g) %)) (cm/addresses (reached sigma g)))))))))

(defn identity-update-changes-nothing
  "An update without constraints has weight 0, keeps every choice,
   discards nothing, and reports a sound change of its return value."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {t2 :trace w :weight disc :discard :as r} (gfi/update tr cm/EMPTY (rng/key (inc seed)))]
      (checks (zero-weight? w) (= (:choices tr) (:choices t2)) (cm/empty? disc) (sound-retdiff? tr r)))))

(defn updates-are-reversible-with-exact-discards
  "An update with constraints from another run, on other arguments, makes
   a consistent trace whose discard is Def 2.3.1's, and updating back
   with the discard restores the trace (Prop 2.3.1)."
  [cases]
  (prop/for-all [[model seed picks alt?] cases]
    (if-let [{:keys [tr sigma args'] {t2 :trace disc :discard :as r} :result} (update-case model seed picks alt?)]
      (let [{t3 :trace disc3 :discard} (gfi/update t2 (:args tr) (unknown-diffs (:args tr)) disc (rng/key seed))]
        (checks (consistent? t2)
                (= args' (:args t2))
                (sound-retdiff? tr r)
                (applied? tr t2 sigma)
                (= (exact-discard tr t2 sigma) disc)
                ;; Prop 2.3.1: updating back with the discard restores the trace,
                ;; and discards σ (as far as t2 reached it) with the freshly
                ;; proposed choices
                (= (:choices tr) (:choices t3))
                (= (cm/merge (reached sigma t2) (restrict (:choices t2) (fresh tr t2 sigma))) disc3)))
      vacuous)))

(defn regenerate-keeps-unselected-choices
  "regenerate makes a consistent trace and keeps every unselected choice."
  [cases]
  (prop/for-all [[[_ gf args :as model] seed picks alt?] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          args' (next-args model seed alt?)
          s (sel/from-paths (picked (addrs tr) picks))
          {t2 :trace :as r} (gfi/regenerate tr args' (unknown-diffs args') s (rng/fold-in k 1))
          selected (set (cm/addresses (sel/get-selected (:choices tr) s)))
          now (set (addrs t2))]
      (checks (consistent? t2)
              (= args' (:args t2))
              (sound-retdiff? tr r)
              (every? #(= (cm/get-choice (:choices tr) %) (cm/get-choice (:choices t2) %))
                      (remove selected (filter now (addrs tr))))))))

(defn regenerating-nothing-changes-nothing
  "A regenerate with nothing selected has weight 0 and keeps the choices."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {t2 :trace w :weight} (gfi/regenerate tr sel/none (rng/key (inc seed)))]
      (checks (zero-weight? w) (= (:choices tr) (:choices t2))))))

(defn project-of-everything-is-the-score
  "A trace projected on every address gives its score."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))]
      (checks (close? (gfi/project tr sel/all) (:score tr))))))

(defn generate-weight-is-the-projection
  "generate's weight given sigma is the projection of its trace on sigma's
   addresses: both are log p(τ)/q(τ∖σ | σ), q the internal proposal,
   whatever it is (forward sampling, an exact or a custom proposal)."
  [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [k (rng/key seed)
          tr0 (gfi/simulate gf args k)
          sigma (restrict (:choices tr0) (picked (addrs tr0) picks))]
      ;; the same key, so forward sampling takes tr0's branches; a custom
      ;; proposal may take others, under which some of sigma is unused: the
      ;; law says nothing about such a case
      (if-let [{tr :trace w :weight} (try (gfi/generate gf args sigma k)
                                          (catch clojure.lang.ExceptionInfo e
                                            (when-not (re-find #"was not visited" (ex-message e)) (throw e))))]
        (checks (close? w (gfi/project tr (sel/from-paths (cm/addresses sigma)))))
        vacuous))))

(defn regenerate-weight-is-the-projection-ratio
  "regenerate's weight is the ratio of the projections of the new and the
   old trace on the retained choices (Eq. 4.1): with forward sampling, the
   retained density ratio; for any internal proposal, its correction."
  [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          chosen (set (picked (addrs tr) picks))
          {t2 :trace w :weight} (gfi/regenerate tr (sel/from-paths chosen) (rng/key (inc seed)))
          now (set (addrs t2))
          retained (sel/from-paths (filter #(and (now %) (not (chosen %))) (addrs tr)))]
      (checks (if (outside? t2)
                (minus-infinity? w)
                (close? w (- (gfi/project t2 retained) (gfi/project tr retained))))))))

(defn update-weight-is-the-projection-change
  "update's weight is the projection of the new trace on its choices that
   are not fresh (kept from the old trace or constrained), less the old
   score: log p(τ')/q(fresh | the rest) - log p(τ) (Def. 2.3.1, p. 69,
   with an internal proposal), whatever the proposal. Before this law the
   general set left update's weight unchecked: a planted +1 passed all of
   it (law audit, 2026-10-20)."
  [cases]
  (prop/for-all [[model seed picks alt?] cases]
    (if-let [{:keys [tr sigma] {t2 :trace w :weight} :result} (update-case model seed picks alt?)]
      (checks (if (outside? t2)
                (minus-infinity? w)
                (close? w (- (gfi/project t2 (sel/from-paths (kept tr t2 sigma))) (:score tr)))))
      vacuous)))

(defn regenerate-weight-with-new-arguments-is-the-projection-ratio
  "Eq. 4.1 with new arguments (p. 144): the projection of the new trace, on
   the new arguments, on the retained choices, less the old trace's on its
   own arguments (the reverse move generates at the old arguments: the
   thesis's Alg. 16 reads x' there, an erratum). A move whose new arguments
   leave an unselected choice unvisited is loud and the case vacuous."
  [cases]
  (prop/for-all [[[_ gf args :as model] seed picks] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          args' (next-args model seed true)
          chosen (set (picked (addrs tr) picks))
          r (vacuous-on-unvisitable
             (gfi/regenerate tr args' (unknown-diffs args') (sel/from-paths chosen) (rng/key (inc seed))))]
      (if (vacuous? r)
        r
        (let [{t2 :trace w :weight} r
              now (set (addrs t2))
              retained (sel/from-paths (filter #(and (now %) (not (chosen %))) (addrs tr)))]
          (checks (if (outside? t2)
                    (minus-infinity? w)
                    (close? w (- (gfi/project t2 retained) (gfi/project tr retained))))))))))

(defn propose-weight-is-the-assessed-density
  "propose's weight is the density assess gives the choices it proposed."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [{c :choices w :weight} (gfi/propose gf args (rng/key seed))]
      (checks (close? w (:weight (gfi/assess gf args c (rng/key (inc seed)))))))))

(defn assess-is-generate-with-every-choice
  "assess's weight is generate's on the same choices and key: the key draws
   the same encapsulated randomness in both (§4.5.1)."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          k (rng/key (inc seed))]
      (checks (close? (:weight (gfi/assess gf args (:choices tr) k)) (:weight (gfi/generate gf args (:choices tr) k)))))))

(defn updates-and-regenerates-keep-the-assessed-score
  "Def 2.1.16 after chains of moves: a trace updated to other arguments and
   constraints and back with its discard, and one regenerated to other
   arguments and back, has the assessed density of its choices as its
   score. The first move may take a choice out of its support and the
   second bring it back: a score kept as a running total (score + new -
   old) was then -Inf + Inf = NaN where the density is finite
   (genclj-awd0); updates-are-reversible-with-exact-discards compared only
   the choices coming back. Planted bug (2026-10-24): that running total
   unchecked fails it on the combinators' and the extras' bounded kernels."
  [cases]
  (prop/for-all [[[_ gf args :as model] seed picks alt?] cases]
    (let [k (rng/key seed)]
      (checks (if-let [{:keys [tr] {t2 :trace disc :discard} :result} (update-case model seed picks alt?)]
                (let [r (vacuous-on-unvisitable (gfi/update t2 (:args tr) (unknown-diffs (:args tr)) disc (rng/fold-in k 3)))]
                  (checks (consistent? t2) (if (vacuous? r) r (checks (consistent? (:trace r))))))
                vacuous)
              (let [tr (gfi/simulate gf args k)
                    args' (next-args model seed alt?)
                    s (sel/from-paths (picked (addrs tr) picks))
                    r (vacuous-on-unvisitable (gfi/regenerate tr args' (unknown-diffs args') s (rng/fold-in k 4)))]
                (if (vacuous? r)
                  r
                  (let [r2 (vacuous-on-unvisitable (gfi/regenerate (:trace r) args (unknown-diffs args) sel/none (rng/fold-in k 5)))]
                    (checks (consistent? (:trace r)) (if (vacuous? r2) r2 (checks (consistent? (:trace r2))))))))))))

;; ---------------------------------------------------------------------------
;; Forward-sampling laws
;; ---------------------------------------------------------------------------

(defn generate-without-constraints-has-weight-zero
  "Forward sampling: generate without constraints has weight 0."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (checks (zero-weight? (:weight (gfi/generate gf args cm/EMPTY (rng/key seed)))))))

(defn generate-weight-is-the-constrained-density
  "Forward sampling: generate's weight is the log density of the
   constrained choices, and the open ones are drawn as simulate draws
   them on the same key."
  [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          sigma (restrict (:choices tr) (picked (addrs tr) picks))
          {g :trace w :weight} (gfi/generate gf args sigma k)]
      ;; one key per site: the unconstrained draws are the same as tr's
      (checks (= (:choices tr) (:choices g))
              (close? w (lp g (cm/addresses sigma)))))))

(defn update-weight-is-the-score-change-less-fresh-choices
  "Forward sampling: update's weight is the change of score less the log
   density of the choices drawn afresh, and an update back adds the
   negated fresh density to it."
  [cases]
  (prop/for-all [[model seed picks alt?] cases]
    (if-let [{:keys [tr sigma] {t2 :trace w :weight disc :discard} :result} (update-case model seed picks alt?)]
      (let [fresh-lp (lp t2 (fresh tr t2 sigma))
            {w3 :weight} (gfi/update t2 (:args tr) (unknown-diffs (:args tr)) disc (rng/key seed))]
        (checks (if (outside? t2) (minus-infinity? w) (close? w (- (- (:score t2) (:score tr)) fresh-lp)))
                ;; the reverse weight is defined only inside the support (Def
                ;; 4.1.1): a move to density 0 has w = -Inf and no finite reverse
                (if (in-support? t2) (checks (close? (+ w w3) (array/negate fresh-lp))) true)))
      vacuous)))

(defn regenerate-weight-is-the-retained-density-ratio
  "Forward sampling: regenerate's weight is the log density ratio of the
   retained choices (Eq 4.1, Alg 18)."
  [cases]
  (prop/for-all [[[_ gf args :as model] seed picks alt?] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          args' (next-args model seed alt?)
          s (sel/from-paths (picked (addrs tr) picks))
          {t2 :trace w :weight} (gfi/regenerate tr args' (unknown-diffs args') s (rng/fold-in k 1))
          selected (set (cm/addresses (sel/get-selected (:choices tr) s)))
          now (set (addrs t2))
          ;; Eq 4.1, Alg 18: selected, fresh and removed choices cancel
          retained (remove selected (filter now (addrs tr)))]
      (checks (if (outside? t2) (minus-infinity? w) (close? w (- (lp t2 retained) (lp tr retained))))))))

(defn regenerating-everything-has-weight-zero
  "Forward sampling: a regenerate of every choice has weight 0."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))]
      (checks (zero-weight? (:weight (gfi/regenerate tr sel/all (rng/key (inc seed)))))))))

(defn project-splits-the-score
  "project splits the score over a partition of the choices: none projects
   to zero, and a selection and its complement add up to the score; so do
   two disjoint selections s and t, their union, and the rest, by De
   Morgan's law the intersection of their complements. A union of selects
   is one hierarchical selection; s with t less s is a union of kinds, and
   the union less t an intersection of them, which every kind of
   generative function reads through gen.selection (a Map's or Unfold's
   top addresses, a distribution's root). Planted bug (2026-10-25): an
   intersection whose top addresses are none when a part may select
   anything fails it on the combinator models."
  [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          p #(gfi/project tr %)
          [ps pt] (picked-apart (addrs tr) picks)
          s (sel/from-paths ps)
          t (sel/from-paths pt)
          both (sel/union s t)]
      (checks (zero-weight? (p sel/none))
              (close? (+ (p s) (p (sel/complement s))) (:score tr))
              (close? (p both) (+ (p s) (p t)))
              (close? (p (sel/union s (sel/intersection t (sel/complement s)))) (p both))
              (close? (p (sel/intersection both (sel/complement t))) (p s))
              (close? (+ (p both) (p (sel/intersection (sel/complement s) (sel/complement t)))) (:score tr))))))

(defn runs-are-pure-functions-of-the-key
  "The same key makes the same choices."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (checks (= (:choices (gfi/simulate gf args (rng/key seed)))
               (:choices (gfi/simulate gf args (rng/key seed)))))))

(defn dispatch-matches-handler
  "A dispatcher (gen.dynamic/with-dispatch) is observationally equivalent to
   the handler: each operation gives the same choices, weights and return
   values with the same key. Over cases of models whose gf has a dispatcher."
  [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [plain (assoc gf :dispatch nil)
          k (rng/key seed)
          same? (fn [r1 r2] (and (= (:choices (:trace r1)) (:choices (:trace r2)))
                                 (= (:retval (:trace r1)) (:retval (:trace r2)))
                                 (close? (or (:weight r1) 0.0) (or (:weight r2) 0.0))
                                 (= (:discard r1) (:discard r2))))
          tr (gfi/simulate plain args k)
          sigma (restrict (:choices tr) (picked (addrs tr) picks))
          s (sel/from-paths (picked (addrs tr) picks))]
      (checks (= (:choices (gfi/simulate gf args k)) (:choices tr))
              (same? (gfi/generate gf args sigma k) (gfi/generate plain args sigma k))
              (close? (:weight (gfi/assess gf args (:choices tr) k)) (:weight (gfi/assess plain args (:choices tr) k)))
              (same? (gfi/update (assoc tr :gen-fn gf) sigma k) (gfi/update tr sigma k))
              (same? (gfi/regenerate (assoc tr :gen-fn gf) s k) (gfi/regenerate tr s k))))))

(def general
  "The laws of every generative function: the thesis's definitions (Def
   2.1.16's trace consistency, Def 2.3.1's update and discard, Prop 2.3.1's
   reversibility, Def 2.3.2's retdiffs, Eq 4.1's regenerate weight), for
   any internal proposal."
  {:score-is-the-assessed-density score-is-the-assessed-density
   :generate-with-every-choice-is-deterministic generate-with-every-choice-is-deterministic
   :generated-traces-are-consistent generated-traces-are-consistent
   :identity-update-changes-nothing identity-update-changes-nothing
   :updates-are-reversible-with-exact-discards updates-are-reversible-with-exact-discards
   :regenerate-keeps-unselected-choices regenerate-keeps-unselected-choices
   :regenerating-nothing-changes-nothing regenerating-nothing-changes-nothing
   :project-of-everything-is-the-score project-of-everything-is-the-score
   :generate-weight-is-the-projection generate-weight-is-the-projection
   :regenerate-weight-is-the-projection-ratio regenerate-weight-is-the-projection-ratio
   :update-weight-is-the-projection-change update-weight-is-the-projection-change
   :regenerate-weight-with-new-arguments-is-the-projection-ratio regenerate-weight-with-new-arguments-is-the-projection-ratio
   :propose-weight-is-the-assessed-density propose-weight-is-the-assessed-density
   :assess-is-generate-with-every-choice assess-is-generate-with-every-choice
   :updates-and-regenerates-keep-the-assessed-score updates-and-regenerates-keep-the-assessed-score})

(def forward-sampling
  "The laws of a generative function whose internal proposal samples the
   unconstrained choices from the prior, each site from its own key, as
   gen and the combinators do. Another internal proposal need satisfy only
   general."
  {:generate-without-constraints-has-weight-zero generate-without-constraints-has-weight-zero
   :generate-weight-is-the-constrained-density generate-weight-is-the-constrained-density
   :update-weight-is-the-score-change-less-fresh-choices update-weight-is-the-score-change-less-fresh-choices
   :regenerate-weight-is-the-retained-density-ratio regenerate-weight-is-the-retained-density-ratio
   :regenerating-everything-has-weight-zero regenerating-everything-has-weight-zero
   :project-splits-the-score project-splits-the-score
   :runs-are-pure-functions-of-the-key runs-are-pure-functions-of-the-key})

(def all
  "general and forward-sampling: the laws of a generative function that
   samples forward, verify's and deflaws' default."
  (merge general forward-sampling))

(defn generate-weights-are-finite-on-a-fixed-support
  "Def. 4.1.1's support condition, for a model whose every address has a
   fixed support (thesis pp. 169-170): constraints drawn from a simulated
   trace are in the support, so the trace's score and generate's weight
   are finite (law audit gap E: not-NaN let a -Inf weight pass). Planted
   bug (2026-10-20): normal's log density -Inf beyond 2.5 standard
   deviations fails it, on the dynamic and the combinator models."
  [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          sigma (restrict (:choices tr) (picked (addrs tr) picks))
          r (vacuous-on-unvisitable (gfi/generate gf args sigma (rng/key (inc seed))))]
      (checks (finite? (:score tr))
              (if (vacuous? r) r (checks (finite? (:weight r))))))))

(def fixed-support
  "For models whose every address has a fixed support."
  {:generate-weights-are-finite-on-a-fixed-support generate-weights-are-finite-on-a-fixed-support})

(defn static-matches-dynamic
  "The static language is an optimization of the dynamic one (DESIGN.md
   §9): every operation on a static gf gives what the same body without
   ^:static gives under the same keys (choices, scores, weights, return
   values and discards, and a sound retdiff), also when an update or
   regenerate skips nodes (no-change argdiffs for unchanged arguments)."
  [cases]
  (prop/for-all [[[_ gf args :as model] seed picks alt?] cases]
    (let [plain (assoc gf :static nil)
          k (rng/key seed)
          tr (gfi/simulate gf args k)
          tp (gfi/simulate plain args k)
          host-choices #(host (cm/to-map (:choices %)))
          same-trace? (fn [t1 t2] (and (= (host-choices t1) (host-choices t2)) (close? (:score t1) (:score t2))
                                       (= (host (:retval t1)) (host (:retval t2)))))
          ;; the static retdiff may be more precise than the dynamic one (a
          ;; propagated diff), so it need only be sound
          same? (fn [r1 r2] (and (same-trace? (:trace r1) (:trace r2))
                                 (close? (or (:weight r1) 0.0) (or (:weight r2) 0.0))
                                 (= (some-> (:discard r1) cm/to-map host) (some-> (:discard r2) cm/to-map host))
                                 (or (nil? (:retdiff r1)) (sound-retdiff? tr r1))))
          args' (next-args model seed alt?)
          diffs (if (= args' args) (diff/no-changes (count args)) (unknown-diffs args'))
          ;; precise argdiffs too, which the static walk propagates (review,
          ;; 2026-10-20); an argument the previous call did not have is new
          precise (mapv (fn [i a'] (if (< i (count args)) (diff/compute (nth args i) a') diff/unknown-change))
                        (range) args')
          other (gfi/simulate gf args' (rng/fold-in k 1))
          sigma (restrict (:choices other) (picked (addrs other) picks))
          s (sel/from-paths (picked (addrs tr) picks))
          both (fn [f] (let [r1 (vacuous-on-unvisitable (f gf tr)) r2 (vacuous-on-unvisitable (f plain tp))]
                         (if (and (vacuous? r1) (vacuous? r2)) true (checks (same? r1 r2)))))]
      (checks (same-trace? tr tp)
              (both (fn [g _] (gfi/generate g args (restrict (:choices tr) (picked (addrs tr) picks)) (rng/fold-in k 2))))
              (close? (:weight (gfi/assess gf args (:choices tr) k)) (:weight (gfi/assess plain args (:choices tr) k)))
              (both (fn [_ t] (gfi/update t args' diffs sigma (rng/fold-in k 3))))
              (both (fn [_ t] (gfi/update t args' precise sigma (rng/fold-in k 3))))
              (both (fn [_ t] (gfi/regenerate t args' precise s (rng/fold-in k 6))))
              (both (fn [_ t] (gfi/update t cm/EMPTY (rng/fold-in k 4))))
              (both (fn [_ t] (gfi/regenerate t args' diffs s (rng/fold-in k 5))))))))

(def static
  "For a static generative function (gen ^:static)."
  {:static-matches-dynamic static-matches-dynamic})

;; ---------------------------------------------------------------------------
;; Encapsulated randomness (thesis §4.5)
;; ---------------------------------------------------------------------------

(defn generate-with-every-choice-weighs-its-score
  "Encapsulated randomness: generate constrained on every choice of a
   trace makes those choices, its weight the score of the trace it makes."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {g :trace w :weight} (gfi/generate gf args (:choices tr) (rng/key (inc seed)))]
      (checks (= (:choices tr) (:choices g)) (close? w (:score g))))))

(defn generated-traces-keep-their-constraints
  "Encapsulated randomness: generate keeps its constraints, and its weight
   is not NaN."
  [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          sigma (restrict (:choices tr) (picked (addrs tr) picks))
          r (vacuous-on-unvisitable (gfi/generate gf args sigma (rng/key (inc seed))))]
      (if (vacuous? r)
        r
        (let [{g :trace w :weight} r]
          (checks (not (array/truthy? (array/isnan w)))
                  (every? #(= (cm/get-choice sigma %) (cm/get-choice (:choices g) %)) (cm/addresses (reached sigma g)))))))))

(defn updates-restore-their-choices
  "Prop 2.3.1 for the choices: the discards of an update, given back,
   restore them (the weights agree only in expectation)."
  [cases]
  (prop/for-all [[model seed picks alt?] cases]
    (if-let [{:keys [tr sigma] {t2 :trace disc :discard :as r} :result} (update-case model seed picks alt?)]
      (let [{t3 :trace} (gfi/update t2 (:args tr) (unknown-diffs (:args tr)) disc (rng/key seed))]
        (checks (sound-retdiff? tr r)
                (applied? tr t2 sigma)
                (= (exact-discard tr t2 sigma) disc)
                (= (:choices tr) (:choices t3))))
      vacuous)))

(defn regenerate-keeps-unselected-values
  "Encapsulated randomness: regenerate keeps every unselected choice and
   reports a sound change of its return value."
  [cases]
  (prop/for-all [[[_ gf args :as model] seed picks alt?] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          args' (next-args model seed alt?)
          s (sel/from-paths (picked (addrs tr) picks))
          {t2 :trace :as r} (gfi/regenerate tr args' (unknown-diffs args') s (rng/fold-in k 1))
          selected (set (cm/addresses (sel/get-selected (:choices tr) s)))
          now (set (addrs t2))]
      (checks (sound-retdiff? tr r)
              (every? #(= (cm/get-choice (:choices tr) %) (cm/get-choice (:choices t2) %))
                      (remove selected (filter now (addrs tr))))))))

(defn update-weight-is-the-score-change-less-fresh
  "Encapsulated randomness: update's weight is the change of score less
   the log density of the choices drawn afresh, for one draw of ω."
  [cases]
  (prop/for-all [[model seed picks alt?] cases]
    (if-let [{:keys [tr sigma] {t2 :trace w :weight} :result} (update-case model seed picks alt?)]
      (checks (if (outside? t2) (minus-infinity? w) (close? w (- (- (:score t2) (:score tr)) (lp t2 (fresh tr t2 sigma))))))
      vacuous)))

(defn score-is-assessed-on-the-same-key
  "Def 2.1.16 pointwise for untraced draws: simulate and assess on one key
   draw the same ω, so the score is assess's weight (law audit gap 12). An
   estimator's joint sampler draws ω otherwise than its estimate."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)]
      (checks (close? (:score tr) (:weight (gfi/assess gf args (:choices tr) k)))))))

(defn identity-update-keeps-omega
  "An update that changes nothing keeps ω and the score (DESIGN.md §9)."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          t2 (:trace (gfi/update tr cm/EMPTY (rng/key (inc seed))))]
      (checks (= (:gen.trace/omega tr) (:gen.trace/omega t2)) (= (number (:score tr)) (number (:score t2)))))))

(def encapsulated
  "The laws of a generative function with encapsulated randomness ω, whose
   score is an estimate: those of general and forward-sampling that hold for
   one ω, and the others without their comparisons of separate runs (assess
   against a trace's score, propose against assess, an update and its
   reverse), which hold only in expectation (Eqs. 4.3-4.4, tested
   statistically)."
  (merge (apply dissoc all [:score-is-the-assessed-density :generate-with-every-choice-is-deterministic
                            :generated-traces-are-consistent :updates-are-reversible-with-exact-discards
                            :regenerate-keeps-unselected-choices :propose-weight-is-the-assessed-density
                            :update-weight-is-the-score-change-less-fresh-choices
                            :updates-and-regenerates-keep-the-assessed-score])
         {:generate-with-every-choice-weighs-its-score generate-with-every-choice-weighs-its-score
          :generated-traces-keep-their-constraints generated-traces-keep-their-constraints
          :updates-restore-their-choices updates-restore-their-choices
          :regenerate-keeps-unselected-values regenerate-keeps-unselected-values
          :update-weight-is-the-score-change-less-fresh update-weight-is-the-score-change-less-fresh
          :identity-update-keeps-omega identity-update-keeps-omega
          :score-is-assessed-on-the-same-key score-is-assessed-on-the-same-key}))

(def estimated
  "The laws of an estimated density with a joint sampler
   (gen.custom/estimated with :simulate, or a model that calls one):
   encapsulated without score-is-assessed-on-the-same-key. The sampler
   draws ω otherwise than the estimate does, so assess on a trace's key
   does not reproduce its score; the estimate is unbiased only in
   expectation."
  (dissoc encapsulated :score-is-assessed-on-the-same-key))

;; ---------------------------------------------------------------------------
;; Across generative functions (gfi/revise)
;; ---------------------------------------------------------------------------

(defn update-across-generative-functions
  "An update of a trace that gf made, given gf' (another generative function,
   such as gf under another store): only its choices are kept (gfi/revise),
   so Def. 2.3.1 holds against gf': the new trace is gf''s, the weight is
   log p'(τ') - log q'(fresh | the rest) - log p(τ) (the projection of τ' on
   its kept choices, less τ's score), the discard is the old choices
   constrained or gone, and updating back with gf and the discard restores
   τ, the two weights adding to -log q'(fresh) (law audit F5: store-test had
   examples only). Models are [label gf args alt-args gf']."
  [cases]
  (prop/for-all [[[_ gf _ _ gf' :as model] seed picks alt?] cases]
    (if-let [{:keys [tr sigma] {t2 :trace w :weight disc :discard} :result} (update-case model seed picks alt? gf')]
      (let [kept-lp (gfi/project t2 (sel/from-paths (kept tr t2 sigma)))
            {t3 :trace w3 :weight} (gfi/update (assoc t2 :gen-fn gf) (:args tr) (unknown-diffs (:args tr)) disc
                                               (rng/fold-in (rng/key seed) 3))]
        (checks (gfi/same-gen-fn? gf' (trace/maker t2))
                (assessed? gf' t2)
                (applied? tr t2 sigma)
                (if (outside? t2) (minus-infinity? w) (close? w (- kept-lp (:score tr))))
                (= (exact-discard tr t2 sigma) disc)
                (= (:choices tr) (:choices t3))
                (assessed? gf t3)
                (if (in-support? t2) (checks (close? (+ w w3) (- kept-lp (:score t2)))) true)))
      vacuous)))

(defn regenerate-across-generative-functions
  "A regenerate of a trace that gf made, given gf': the unselected choices
   are kept, and the weight is Eq. 4.1's with each trace projected by its
   own generative function, on the retained choices."
  [cases]
  (prop/for-all [[[_ gf args _ gf' :as model] seed picks alt?] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          args' (next-args model seed alt?)
          chosen (set (picked (addrs tr) picks))
          r (vacuous-on-unvisitable
             (gfi/regenerate (assoc tr :gen-fn gf') args' (unknown-diffs args') (sel/from-paths chosen) (rng/fold-in k 1)))]
      (if (vacuous? r)
        r
        (let [{t2 :trace w :weight} r
              now (set (addrs t2))
              retained (filter #(and (now %) (not (chosen %))) (addrs tr))]
          (checks (gfi/same-gen-fn? gf' (trace/maker t2))
                  (assessed? gf' t2)
                  (every? #(= (cm/get-choice (:choices tr) %) (cm/get-choice (:choices t2) %)) retained)
                  (if (outside? t2)
                    (minus-infinity? w)
                    (close? w (- (gfi/project t2 (sel/from-paths retained)) (gfi/project tr (sel/from-paths retained)))))))))))

(def across
  "For pairs of generative functions: models [label gf args alt-args gf'],
   gf' typically gf under another store (gen.dynamic/with-params)."
  {:update-across-generative-functions update-across-generative-functions
   :regenerate-across-generative-functions regenerate-across-generative-functions})

;; ---------------------------------------------------------------------------
;; gen.clj's implementation properties
;; ---------------------------------------------------------------------------

(defn- holds-array?
  "Does x hold a backend array (in its maps, vectors and the first 10,000
   elements of a seq)?"
  [x]
  (cond (array/backend-array? x) true
        (map? x) (boolean (some holds-array? (vals x)))
        (seq? x) (boolean (some holds-array? (take 10000 x)))
        (coll? x) (boolean (some holds-array? x))
        :else false))

(defn- no-change-unless-an-array?
  "Is rd, an identity update's retdiff of the return value retval,
   no-change, or unknown-change of a value that holds a backend array?"
  [rd retval]
  (or (diff/no-change? rd)
      (and (diff/unknown-change? rd) (holds-array? retval))))

(defn identity-update-reports-no-change
  "An implementation property, not a thesis law (Def 2.3.2 allows
   unknown-change): an identity update's retdiff is no-change. A return
   value that holds a backend array may be unknown-change: a body run again
   builds a new array, and unchanged is gfi/same-value?, which reads no
   array (DESIGN.md §9, D1). On MLX even a 0-d value is such an array."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          rd (:retdiff (gfi/update tr cm/EMPTY (rng/key (inc seed))))]
      (checks (no-change-unless-an-array? rd (:retval tr))))))

(def implementation
  "Properties of gen.clj's own generative functions that the thesis does
   not require: an identity update reports no-change (unless its return
   value holds an array), so that a caller can skip work. A correct
   generative function that always reports unknown-change fails it; one of
   yours need satisfy it only to behave as gen.clj's do."
  {:identity-update-reports-no-change identity-update-reports-no-change})

;; ---------------------------------------------------------------------------
;; Running the laws
;; ---------------------------------------------------------------------------

(def max-vacuous
  "The share of a law's cases that may be vacuous before check fails it.
   The highest share measured (2026-10-24, five runs of every law) was the
   fuzz's 0.32 of 200 cases, then Recurse's 0.23 of 100: 0.6 is 8 and 9
   standard errors above them."
  0.6)

(defn ^:no-doc options
  "The options of verify and of a deflaws spec, opts being a number of
   cases or a map of :num-tests and check's options: a map with
   :num-tests (test.check's defspec count, 100, by default)."
  [opts]
  (let [known #{:num-tests :seed :max-size :max-vacuous :reporter-fn}]
    (cond (nil? opts) {:num-tests ctc/*default-test-count*}
          (number? opts) {:num-tests opts}
          (and (map? opts) (every? known (keys opts))) (merge {:num-tests ctc/*default-test-count*} opts)
          (map? opts) (throw (ex-info (str "laws: unknown options " (pr-str (vec (remove known (keys opts))))
                                           " (the options are " (pr-str (vec (sort known)))
                                           "; a law set comes after them)")
                                      {:options (vec (keys opts))}))
          :else (throw (ex-info (str "laws: options are a number of cases or a map, not " (pr-str opts))
                                {:options opts})))))

(defn- law-set!
  "laws, when it is a law set (a map from a law's name to the law); else a
   loud error."
  [laws]
  (if (and (map? laws) (every? fn? (vals laws)))
    laws
    (throw (ex-info "laws: a law set is a map from a law's name to the law, a function of cases"
                    {:laws (if (map? laws) (vec (keys laws)) laws)}))))

(defn check
  "quick-check of a law's property over num-tests cases (100 by default),
   with the count of its vacuous cases as :vacuous: a law more than
   :max-vacuous (max-vacuous by default) of whose cases were vacuous fails,
   its :result-data {:vacuous :of :max-vacuous}. Options are quick-check's
   (:seed, :max-size, :reporter-fn), as keywords or a map. A run draws its
   seed from the clock unless :seed is given, and its result's :seed
   reproduces it. Nothing is printed.

     (laws/check 20 (laws/score-is-the-assessed-density (laws/cases models)) :seed 1)

   Planted bug (2026-10-24): an update that rejects every constraint as
   unvisited passed the dynamic and combinator suites' update-weight laws,
   their every case vacuous (only an error in a reverse update failed one
   other law); it fails them now."
  ([property] (check ctc/*default-test-count* property))
  ([num-tests property & {:as opts}]
   (let [{:keys [reporter-fn] :or {reporter-fn (fn [_])} :as opts} (merge {:max-vacuous max-vacuous} opts)
         n (volatile! 0)
         count-vacuous (fn [m]
                         (when (and (= :trial (:type m)) (::vacuous (:result-data m))) (vswap! n inc))
                         (reporter-fn m))
         qc-opts (assoc (dissoc opts :max-vacuous :num-tests) :reporter-fn count-vacuous)
         r (assoc (apply tc/quick-check num-tests property (apply concat qc-opts)) :vacuous @n)]
     (if (and (:pass? r) (> @n (* (:max-vacuous opts) num-tests)))
       (assoc r :pass? false :result false
              :result-data {:vacuous @n :of num-tests :max-vacuous (:max-vacuous opts)})
       r))))

(defn- the-case
  "A case [[label gf args alt-args] seed picks alt?] as a failure shows it:
   the model by its label, its seed (of the keys) as :key-seed."
  [[[label] seed picks alt?]]
  {:model label :key-seed seed :picks picks :alt? alt?})

(defn failure-of
  "What check's failing result r says, readably: the failing check's
   :check and :values (or what its explain gives), or the :error thrown
   (its message, ex-data as :data, and the :exception), or the counts of a
   law that failed for its vacuous cases; with the smallest failing case
   test.check shrank it to (:case, {:model :key-seed :picks :alt?}: the
   model by its label, the seed of the case's keys, as cases has them),
   and the run's :seed, test.check's."
  [r]
  (let [{:keys [seed shrunk fail result-data]} r
        data (if shrunk (:result-data shrunk) result-data)
        error (::prop/error data)
        smallest (first (or (:smallest shrunk) fail))]
    (cond-> (dissoc data ::prop/error)
      error (assoc :error (or (ex-message error) (.getName ^Class (class error))) :exception error)
      (ex-data error) (assoc :data (ex-data error))
      smallest (assoc :case (the-case smallest))
      true (assoc :seed seed))))

(defn verify
  "Runs every law of laws (all by default) over cases of models, and
   returns the failures: a sorted map from each law that failed to what it
   says (failure-of: the check that failed and the values it compared, or
   the error thrown, the smallest failing case and the run's seed); {} when
   every law holds. opts is the number of cases of each law (100 by
   default), or a map of it (:num-tests) and check's options: :seed,
   :max-size and :max-vacuous. Every law runs from one seed, drawn from the
   clock unless :seed is given, so a failure's :seed, given back,
   reproduces the run. The arguments are deflaws' but the prefix.

     (laws/verify [[:model model [0.0] [[1.5]]]])
     (laws/verify models {:num-tests 20 :seed 1} laws/general)"
  ([models] (verify models nil all))
  ([models opts] (verify models opts all))
  ([models opts laws]
   (let [{:keys [num-tests] :as opts} (options opts)
         opts (update opts :seed #(or % (System/currentTimeMillis)))
         cs (cases models)]
     (into (sorted-map)
           (keep (fn [[law-name law]]
                   (let [r (check num-tests (vary-meta (law cs) assoc :name law-name) (dissoc opts :num-tests))]
                     (when-not (:pass? r) [law-name (failure-of r)]))))
           (law-set! laws)))))

(defmacro deflaws
  "A test.check spec per law of laws (all by default), named prefix-law,
   each over cases of models, with prefix's metadata (^:slow, say), run by
   check. opts is the number of cases, or a map of it (:num-tests) and
   check's options. laws is evaluated when the deflaws compiles: a law
   set's var, a map, or an expression of them, such as (merge laws/all
   laws/fixed-support).

   A spec is a clojure.test test, and a function: (spec) runs its cases
   from a seed drawn from the clock, a fresh one at each call, as
   test.check's defspec does; (spec n :seed s) runs n cases from the seed
   s, which a result's :seed reproduces. It returns check's result and
   prints nothing; a failure is reported as clojure.test reports one."
  ([prefix models opts] `(deflaws ~prefix ~models ~opts all))
  ([prefix models opts laws]
   (let [laws-map (if (map? laws) laws (eval laws))
         cs (gensym "cases")
         ls (gensym "laws")]
     `(let [~cs (cases ~models)
            ~ls ~laws]
        ~@(for [law (keys laws-map)
                :let [spec-name (with-meta (symbol (str prefix "-" (name law))) (meta prefix))]]
            ;; defspec's var, with check in place of quick-check
            `(defn ~(vary-meta spec-name assoc ::ctc/defspec true ::law law
                               :doc (str "The law " law " over the models of (gen.laws/deflaws " prefix " …): ("
                                         spec-name ") runs its cases from a fresh seed, (" spec-name
                                         " n :seed s) n cases from the seed s.")
                               :test `(fn [] (ctc/assert-check (assoc (~spec-name) :test-var (str '~spec-name)))))
               ([] (~spec-name (:num-tests (options ~opts))))
               ([times# & {:as check-opts#}]
                (check times# (vary-meta ((get ~ls ~law) ~cs) assoc :name '~spec-name)
                       (merge (dissoc (options ~opts) :num-tests) check-opts#)))))))))
