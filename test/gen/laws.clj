(ns gen.laws
  "The GFI's laws (Cusumano-Towner 2020; docs/research/genjl-parity.md §3,
   audited in docs/research/law-audit.md) as test.check properties over any
   generative functions, in two sets:

   - general: the thesis's definitions, true of every generative function
     (Def 2.1.16, Def 2.3.1, Prop 2.3.1, Def 2.3.2, Eq 4.1's retention);
   - forward-sampling: true when the internal proposal samples unconstrained
     choices from the prior, each site from its own key, as every gen.clj
     generative function does so far. A generative function with an optimal
     or overridden internal proposal (thesis pp. 145-152) need satisfy only
     the general set.

   Models are [label gf args alt-args]: alt-args is a vector of other
   argument vectors, so that update and regenerate are also checked with
   changed arguments. (deflaws prefix models n) defines a defspec per law."
  (:require [clojure.test.check.clojure-test]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
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
  (let [{w :weight rv :retval} (gfi/assess (:gen-fn tr) (:args tr) (:choices tr))]
    (and (close? (:score tr) w) (= (:retval tr) rv))))

(defn- sound-retdiff?
  "Def 2.3.2: no-change promises an equal return value."
  [old {tr :trace rd :retdiff}]
  (or (not (diff/no-change? rd)) (= (:retval old) (:retval tr))))

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

(defn propose-weight-is-the-assessed-density [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [{c :choices w :weight} (gfi/propose gf args (rng/key seed))]
      (close? w (:weight (gfi/assess gf args c))))))

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
        (and (close? w (array/- (array/- (:score t2) (:score tr)) fresh-lp))
             ;; the reverse weight is defined only inside the support (Def
             ;; 4.1.1): a move to density 0 has w = -Inf and no finite reverse
             (or (not (in-support? t2)) (close? (array/+ w w3) (array/negate fresh-lp)))))
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
      (close? w (array/- (lp t2 retained) (lp tr retained))))))

(defn regenerating-everything-has-weight-zero [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))]
      (zero-weight? (:weight (gfi/regenerate tr sel/all (rng/key (inc seed))))))))

(defn project-splits-the-score [cases]
  (prop/for-all [[[_ gf args] seed picks] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          s (sel/from-paths (picked (addrs tr) picks))]
      (and (zero-weight? (gfi/project tr sel/none))
           (close? (array/+ (gfi/project tr s) (gfi/project tr (sel/complement s))) (:score tr))))))

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
   :propose-weight-is-the-assessed-density propose-weight-is-the-assessed-density})

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
