(ns gen.laws
  "The GFI's laws (Cusumano-Towner 2020; docs/research/genjl-parity.md §3,
   audited in docs/research/law-audit.md) as test.check properties over any
   generative functions, in five sets:

   - general: the thesis's definitions, true of every generative function
     (Def 2.1.16, Def 2.3.1, Prop 2.3.1, Def 2.3.2, Eq 4.1's retention);
   - forward-sampling: true when the internal proposal samples unconstrained
     choices from the prior, each site from its own key, as every gen.clj
     generative function does so far. A generative function with an optimal
     or overridden internal proposal (thesis pp. 145-152) need satisfy only
     the general set;
   - fixed-support: the support condition, for models whose every address
     has a fixed support;
   - static: a static generative function against the same body without
     ^:static;
   - encapsulated: for a generative function with encapsulated randomness
     (§4.5), whose score is an estimate, the laws that hold for one draw of
     it.

   Models are [label gf args alt-args]: alt-args is a vector of other
   argument vectors, so that update and regenerate are also checked with
   changed arguments. (deflaws prefix models n) defines a defspec per law."
  (:refer-clojure :exclude [+ -])
  (:require [clojure.test.check.clojure-test]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(defn close?
  "Weights within the backend's precision. On a float32 backend a weight is a
   difference of scores, so its error scales with the scores rather than
   with it: a few ulps of scores of up to ~100."
  [x y]
  (if (ts/f32?) (ts/close? x y {:rtol 1e-4 :atol 1e-3}) (ts/close? x y)))

(defn- zero-weight?
  "Is the weight w exactly zero (a host number, or a 0-d array of any
   backend)?"
  [w]
  (== 0.0 (ts/number w)))

(defn cases
  "Cases [[label gf args alt-args] seed picks alt?] over models: a vector of
   model entries, or a generator of them (e.g. random models)."
  [models]
  (gen/tuple (if (gen/generator? models) models (gen/elements models))
             gen/nat (gen/vector gen/boolean 8) gen/boolean))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- addrs [tr] (cm/addresses (:choices tr)))

(defn- picked [paths picks] (vec (keep (fn [[p pick?]] (when pick? p)) (map vector paths (cycle picks)))))

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

(defn- in-support? [tr] (array/truthy? (array/all (array/isfinite (:score tr)))))

(defn- consistent?
  "Trace consistency (Def 2.1.16): the score is the assessed density of the
   choices, and the return value is the one they determine."
  [tr]
  (let [{w :weight rv :retval} (gfi/assess (:gen-fn tr) (:args tr) (:choices tr) (rng/key 0))]
    (and (close? (:score tr) w) (= (:retval tr) rv))))

(defn- sound-retdiff?
  "Def 2.3.2: the retdiff describes the change of the return value
   (gen.diff/consistent?): no-change an equal value; a vector-diff the two
   lengths and equal elements below both outside :updated (law audit gap
   F: only no-change was checked); an int-, set- or dict-diff its change."
  [old {tr :trace rd :retdiff}]
  (boolean (diff/consistent? (:retval old) (:retval tr) rd)))

(defn- unvisitable?
  "Constraints drawn from another run that this run cannot visit (a
   structural mismatch the GFI correctly rejects): the case is vacuous."
  [^Throwable e]
  (boolean (re-find #"not visited|nested choice map|single value" (or (.getMessage e) ""))))

(defmacro ^:private vacuous-on-unvisitable [& body]
  `(try ~@body
        (catch clojure.lang.ExceptionInfo e#
          (if (unvisitable? e#) ::vacuous (throw e#)))))

(defn- update-case
  "A random update: constraints are a random subset of another run's choices
   (with the new arguments), so they may change values, add addresses and
   change structure. nil when the update rejects them as unvisitable."
  [[_ gf args :as model] seed picks alt?]
  (let [k (rng/key seed)
        tr (gfi/simulate gf args k)
        args' (next-args model seed alt?)
        other (gfi/simulate gf args' (rng/fold-in k 1))
        sigma (restrict (:choices other) (picked (addrs other) picks))
        r (vacuous-on-unvisitable (gfi/update tr args' (unknown-diffs args') sigma (rng/fold-in k 2)))]
    (when-not (= ::vacuous r)
      {:tr tr :args' args' :sigma sigma :result r})))

;; ---------------------------------------------------------------------------
;; General laws
;; ---------------------------------------------------------------------------

(defn score-is-the-assessed-density [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (consistent? (gfi/simulate gf args (rng/key seed)))))

(defn generate-with-every-choice-is-deterministic [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {g :trace w :weight} (gfi/generate gf args (:choices tr) (rng/key (inc seed)))]
      (and (= (:choices tr) (:choices g)) (close? w (:score tr))))))

(defn generated-traces-are-consistent [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          sigma (restrict (:choices tr) (picked (addrs tr) picks))
          ;; another key may take another branch, leaving a constraint unvisited
          r (vacuous-on-unvisitable (gfi/generate gf args sigma (rng/key (inc seed))))]
      (or (= ::vacuous r)
          (let [{g :trace w :weight} r]
            (and (consistent? g)
                 (not (array/truthy? (array/isnan w)))
                 (every? #(= (cm/get-choice sigma %) (cm/get-choice (:choices g) %)) (cm/addresses sigma))))))))

(defn identity-update-changes-nothing [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {t2 :trace w :weight disc :discard :as r} (gfi/update tr cm/EMPTY (rng/key (inc seed)))]
      (and (zero-weight? w) (= (:choices tr) (:choices t2)) (cm/empty? disc) (sound-retdiff? tr r)))))

(defn updates-are-reversible-with-exact-discards [cases]
  (prop/for-all [[model seed picks alt?] cases]
    (if-let [{:keys [tr sigma] {t2 :trace disc :discard :as r} :result} (update-case model seed picks alt?)]
      (let [was (set (addrs tr))
            now (set (addrs t2))
            constrained (set (cm/addresses sigma))
            ;; Def 2.3.1: the discard is τ restricted to (A_τ ∩ A_σ) ∪ (A_τ ∖ A_τ')
            expected-discard (restrict (:choices tr) (filter #(or (constrained %) (not (now %))) (addrs tr)))
            fresh (remove #(or (was %) (constrained %)) (addrs t2))
            {t3 :trace disc3 :discard} (gfi/update t2 (:args tr) (unknown-diffs (:args tr)) disc (rng/key seed))]
        (and (consistent? t2)
             (sound-retdiff? tr r)
             (every? #(= (cm/get-choice sigma %) (cm/get-choice (:choices t2) %)) constrained)
             ;; unconstrained choices that remain keep their values
             (every? #(= (cm/get-choice (:choices tr) %) (cm/get-choice (:choices t2) %))
                     (remove constrained (filter now (addrs tr))))
             (= expected-discard disc)
             ;; Prop 2.3.1: updating back with the discard restores the trace,
             ;; and discards σ with the freshly proposed choices
             (= (:choices tr) (:choices t3))
             (= (cm/merge sigma (restrict (:choices t2) fresh)) disc3)))
      true)))

(defn regenerate-keeps-unselected-choices [cases]
  (prop/for-all [[[_ gf args :as model] seed picks alt?] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          args' (next-args model seed alt?)
          s (sel/from-paths (picked (addrs tr) picks))
          {t2 :trace :as r} (gfi/regenerate tr args' (unknown-diffs args') s (rng/fold-in k 1))
          selected (set (cm/addresses (sel/get-selected (:choices tr) s)))
          now (set (addrs t2))]
      (and (consistent? t2)
           (sound-retdiff? tr r)
           (every? #(= (cm/get-choice (:choices tr) %) (cm/get-choice (:choices t2) %))
                   (remove selected (filter now (addrs tr))))))))

(defn regenerating-nothing-changes-nothing [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {t2 :trace w :weight} (gfi/regenerate tr sel/none (rng/key (inc seed)))]
      (and (zero-weight? w) (= (:choices tr) (:choices t2))))))

(defn project-of-everything-is-the-score [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))]
      (close? (gfi/project tr sel/all) (:score tr)))))

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
        (close? w (gfi/project tr (sel/from-paths (cm/addresses sigma))))
        true))))

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
      (close? w (- (gfi/project t2 retained) (gfi/project tr retained))))))

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
      (let [was (set (addrs tr))
            constrained (set (cm/addresses sigma))
            kept (filter #(or (was %) (constrained %)) (addrs t2))]
        (close? w (- (gfi/project t2 (sel/from-paths kept)) (:score tr))))
      true)))

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
      (or (= ::vacuous r)
          (let [{t2 :trace w :weight} r
                now (set (addrs t2))
                retained (sel/from-paths (filter #(and (now %) (not (chosen %))) (addrs tr)))]
            (close? w (- (gfi/project t2 retained) (gfi/project tr retained))))))))

(defn propose-weight-is-the-assessed-density [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [{c :choices w :weight} (gfi/propose gf args (rng/key seed))]
      (close? w (:weight (gfi/assess gf args c))))))

(defn assess-is-generate-with-every-choice
  "assess's weight is generate's on the same choices and key: the key draws
   the same encapsulated randomness in both (§4.5.1)."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          k (rng/key (inc seed))]
      (close? (:weight (gfi/assess gf args (:choices tr) k)) (:weight (gfi/generate gf args (:choices tr) k))))))

;; ---------------------------------------------------------------------------
;; Forward-sampling laws
;; ---------------------------------------------------------------------------

(defn generate-without-constraints-has-weight-zero [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (zero-weight? (:weight (gfi/generate gf args cm/EMPTY (rng/key seed))))))

(defn generate-weight-is-the-constrained-density [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          sigma (restrict (:choices tr) (picked (addrs tr) picks))
          {g :trace w :weight} (gfi/generate gf args sigma k)]
      ;; one key per site: the unconstrained draws are the same as tr's
      (and (= (:choices tr) (:choices g))
           (close? w (lp g (cm/addresses sigma)))))))

(defn update-weight-is-the-score-change-less-fresh-choices [cases]
  (prop/for-all [[model seed picks alt?] cases]
    (if-let [{:keys [tr sigma] {t2 :trace w :weight disc :discard} :result} (update-case model seed picks alt?)]
      (let [was (set (addrs tr))
            constrained (set (cm/addresses sigma))
            fresh-lp (lp t2 (remove #(or (was %) (constrained %)) (addrs t2)))
            {w3 :weight} (gfi/update t2 (:args tr) (unknown-diffs (:args tr)) disc (rng/key seed))]
        (and (close? w (- (- (:score t2) (:score tr)) fresh-lp))
             ;; the reverse weight is defined only inside the support (Def
             ;; 4.1.1): a move to density 0 has w = -Inf and no finite reverse
             (or (not (in-support? t2)) (close? (+ w w3) (array/negate fresh-lp)))))
      true)))

(defn regenerate-weight-is-the-retained-density-ratio [cases]
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
      (close? w (- (lp t2 retained) (lp tr retained))))))

(defn regenerating-everything-has-weight-zero [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))]
      (zero-weight? (:weight (gfi/regenerate tr sel/all (rng/key (inc seed))))))))

(defn project-splits-the-score [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          s (sel/from-paths (picked (addrs tr) picks))]
      (and (zero-weight? (gfi/project tr sel/none))
           (close? (+ (gfi/project tr s) (gfi/project tr (sel/complement s))) (:score tr))))))

(defn runs-are-pure-functions-of-the-key [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (= (:choices (gfi/simulate gf args (rng/key seed)))
       (:choices (gfi/simulate gf args (rng/key seed))))))

(defn identity-update-reports-no-change
  "An implementation property, not a thesis law (Def 2.3.2 allows
   unknown-change)."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (diff/no-change? (:retdiff (gfi/update (gfi/simulate gf args (rng/key seed)) cm/EMPTY (rng/key (inc seed)))))))

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
      (and (= (:choices (gfi/simulate gf args k)) (:choices tr))
           (same? (gfi/generate gf args sigma k) (gfi/generate plain args sigma k))
           (close? (:weight (gfi/assess gf args (:choices tr))) (:weight (gfi/assess plain args (:choices tr))))
           (same? (gfi/update (assoc tr :gen-fn gf) sigma k) (gfi/update tr sigma k))
           (same? (gfi/regenerate (assoc tr :gen-fn gf) s k) (gfi/regenerate tr s k))))))

(def general
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
   :assess-is-generate-with-every-choice assess-is-generate-with-every-choice})

(def forward-sampling
  {:generate-without-constraints-has-weight-zero generate-without-constraints-has-weight-zero
   :generate-weight-is-the-constrained-density generate-weight-is-the-constrained-density
   :update-weight-is-the-score-change-less-fresh-choices update-weight-is-the-score-change-less-fresh-choices
   :regenerate-weight-is-the-retained-density-ratio regenerate-weight-is-the-retained-density-ratio
   :regenerating-everything-has-weight-zero regenerating-everything-has-weight-zero
   :project-splits-the-score project-splits-the-score
   :runs-are-pure-functions-of-the-key runs-are-pure-functions-of-the-key
   :identity-update-reports-no-change identity-update-reports-no-change})

(def all (merge general forward-sampling))

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
      (and (in-support? tr)
           (or (= ::vacuous r) (array/truthy? (array/all (array/isfinite (:weight r)))))))))

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
          host-choices #(ts/host (cm/to-map (:choices %)))
          same-trace? (fn [t1 t2] (and (= (host-choices t1) (host-choices t2)) (close? (:score t1) (:score t2))
                                       (= (ts/host (:retval t1)) (ts/host (:retval t2)))))
          ;; the static retdiff may be more precise than the dynamic one (a
          ;; propagated diff), so it need only be sound
          same? (fn [r1 r2] (and (same-trace? (:trace r1) (:trace r2))
                                 (close? (or (:weight r1) 0.0) (or (:weight r2) 0.0))
                                 (= (some-> (:discard r1) cm/to-map ts/host) (some-> (:discard r2) cm/to-map ts/host))
                                 (or (nil? (:retdiff r1)) (sound-retdiff? tr r1))))
          args' (next-args model seed alt?)
          diffs (if (= args' args) (diff/no-changes (count args)) (unknown-diffs args'))
          ;; precise argdiffs too, which the static walk propagates (review, 2026-10-20)
          precise (mapv diff/compute args args')
          other (gfi/simulate gf args' (rng/fold-in k 1))
          sigma (restrict (:choices other) (picked (addrs other) picks))
          s (sel/from-paths (picked (addrs tr) picks))
          both (fn [f] (let [r1 (vacuous-on-unvisitable (f gf tr)) r2 (vacuous-on-unvisitable (f plain tp))]
                         (or (= ::vacuous r1 r2) (same? r1 r2))))]
      (and (same-trace? tr tp)
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

(defn generate-with-every-choice-weighs-its-score [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {g :trace w :weight} (gfi/generate gf args (:choices tr) (rng/key (inc seed)))]
      (and (= (:choices tr) (:choices g)) (close? w (:score g))))))

(defn generated-traces-keep-their-constraints [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          sigma (restrict (:choices tr) (picked (addrs tr) picks))
          r (vacuous-on-unvisitable (gfi/generate gf args sigma (rng/key (inc seed))))]
      (or (= ::vacuous r)
          (let [{g :trace w :weight} r]
            (and (not (array/truthy? (array/isnan w)))
                 (every? #(= (cm/get-choice sigma %) (cm/get-choice (:choices g) %)) (cm/addresses sigma))))))))

(defn updates-restore-their-choices
  "Prop 2.3.1 for the choices: the discards of an update, given back,
   restore them (the weights agree only in expectation)."
  [cases]
  (prop/for-all [[model seed picks alt?] cases]
    (if-let [{:keys [tr sigma] {t2 :trace disc :discard :as r} :result} (update-case model seed picks alt?)]
      (let [now (set (addrs t2))
            constrained (set (cm/addresses sigma))
            expected-discard (restrict (:choices tr) (filter #(or (constrained %) (not (now %))) (addrs tr)))
            {t3 :trace} (gfi/update t2 (:args tr) (unknown-diffs (:args tr)) disc (rng/key seed))]
        (and (sound-retdiff? tr r)
             (every? #(= (cm/get-choice sigma %) (cm/get-choice (:choices t2) %)) constrained)
             (every? #(= (cm/get-choice (:choices tr) %) (cm/get-choice (:choices t2) %))
                     (remove constrained (filter now (addrs tr))))
             (= expected-discard disc)
             (= (:choices tr) (:choices t3))))
      true)))

(defn regenerate-keeps-unselected-values [cases]
  (prop/for-all [[[_ gf args :as model] seed picks alt?] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          args' (next-args model seed alt?)
          s (sel/from-paths (picked (addrs tr) picks))
          {t2 :trace :as r} (gfi/regenerate tr args' (unknown-diffs args') s (rng/fold-in k 1))
          selected (set (cm/addresses (sel/get-selected (:choices tr) s)))
          now (set (addrs t2))]
      (and (sound-retdiff? tr r)
           (every? #(= (cm/get-choice (:choices tr) %) (cm/get-choice (:choices t2) %))
                   (remove selected (filter now (addrs tr))))))))

(defn update-weight-is-the-score-change-less-fresh [cases]
  (prop/for-all [[model seed picks alt?] cases]
    (if-let [{:keys [tr sigma] {t2 :trace w :weight} :result} (update-case model seed picks alt?)]
      (let [was (set (addrs tr))
            constrained (set (cm/addresses sigma))]
        (close? w (- (- (:score t2) (:score tr)) (lp t2 (remove #(or (was %) (constrained %)) (addrs t2))))))
      true)))

(defn score-is-assessed-on-the-same-key
  "Def 2.1.16 pointwise for untraced draws: simulate and assess on one key
   draw the same ω, so the score is assess's weight (law audit gap 12). An
   estimator's joint sampler draws ω otherwise than its estimate."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)]
      (close? (:score tr) (:weight (gfi/assess gf args (:choices tr) k))))))

(defn identity-update-keeps-omega
  "An update that changes nothing keeps ω and the score (DESIGN.md §9)."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          t2 (:trace (gfi/update tr cm/EMPTY (rng/key (inc seed))))]
      (and (= (:gen.trace/omega tr) (:gen.trace/omega t2)) (= (ts/number (:score tr)) (ts/number (:score t2)))))))

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
                            :update-weight-is-the-score-change-less-fresh-choices])
         {:generate-with-every-choice-weighs-its-score generate-with-every-choice-weighs-its-score
          :generated-traces-keep-their-constraints generated-traces-keep-their-constraints
          :updates-restore-their-choices updates-restore-their-choices
          :regenerate-keeps-unselected-values regenerate-keeps-unselected-values
          :update-weight-is-the-score-change-less-fresh update-weight-is-the-score-change-less-fresh
          :identity-update-keeps-omega identity-update-keeps-omega
          :score-is-assessed-on-the-same-key score-is-assessed-on-the-same-key}))

(defmacro deflaws
  "A defspec per law (from gen.laws/all, or the given law map: a var's
   symbol or a literal map), named prefix-law, each over cases of models."
  ([prefix models num-tests] `(deflaws ~prefix ~models ~num-tests all))
  ([prefix models num-tests laws]
   (let [laws-map (if (symbol? laws) @(resolve laws) laws)
         cs (gensym "cases")]
     `(let [~cs (cases ~models)]
        ~@(for [law (keys laws-map)]
            `(clojure.test.check.clojure-test/defspec ~(symbol (str prefix "-" (name law))) ~num-tests
               ((get ~laws ~law) ~cs)))))))
