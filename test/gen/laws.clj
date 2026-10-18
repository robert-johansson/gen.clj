(ns gen.laws
  "The GFI's laws (Cusumano-Towner's thesis; docs/research/genjl-parity.md
   §3) as test.check properties over any generative function. Each law takes
   a generator of cases [[label gf args] seed], so every feature's suite can
   check them on its own models:

     (defspec map-score-is-the-assessed-density 100
       (laws/score-is-the-assessed-density gen-case))"
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as a]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]))

(defn close? [x y] (a/allclose? x y {:rtol 1e-12 :atol 1e-12}))

(defn cases
  "A generator of [[label gf args] seed] over models, a vector of
   [label gf args]."
  [models]
  (gen/tuple (gen/elements models) gen/nat))

(defn- pick-paths [addrs picks]
  (keep (fn [[p pick?]] (when pick? p)) (map vector addrs (cycle picks))))

(defn score-is-the-assessed-density [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {w :weight rv :retval} (gfi/assess gf args (:choices tr))]
      (and (close? (:score tr) w) (= (:retval tr) rv)))))

(defn generate-with-every-choice-is-deterministic [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {g :trace w :weight} (gfi/generate gf args (:choices tr) (rng/key (inc seed)))]
      (and (= (:choices tr) (:choices g)) (close? w (:score tr))))))

(defn generate-without-constraints-has-weight-zero [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (zero? (:weight (gfi/generate gf args cm/EMPTY (rng/key seed))))))

(defn generate-with-part-of-a-trace-and-its-key-reproduces-it [cases]
  (prop/for-all [[[_ gf args] seed] cases
                 picks (gen/vector gen/boolean 8)]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          s (sel/from-paths (pick-paths (cm/addresses (:choices tr)) picks))
          constraints (sel/get-selected (:choices tr) s)
          {g :trace w :weight} (gfi/generate gf args constraints k)]
      ;; every site consumes one key, so the unconstrained draws are the same,
      ;; and the weight is the log density of the constrained choices
      (and (= (:choices tr) (:choices g))
           (close? w (gfi/project g (sel/from-choicemap constraints)))))))

(defn identity-update-has-weight-zero [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {t2 :trace w :weight disc :discard rd :retdiff} (gfi/update tr cm/EMPTY (rng/key (inc seed)))]
      (and (zero? w) (= (:choices tr) (:choices t2)) (cm/empty? disc) (diff/no-change? rd)))))

(defn updates-are-reversible [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          ;; constrain one existing address to a value from a second run
          other (gfi/simulate gf args (rng/fold-in k 1))
          ;; an address both runs visited
          p (first (filter (set (cm/addresses (:choices other))) (cm/addresses (:choices tr))))
          v (cm/get-choice (:choices other) p)
          {t2 :trace w :weight disc :discard} (gfi/update tr (cm/choicemap p v) (rng/fold-in k 2))
          fresh (remove (set (cm/addresses (:choices tr))) (cm/addresses (:choices t2)))
          fresh-lp (reduce a/+ 0.0 (map #(gfi/project t2 (sel/select %)) fresh))
          {t3 :trace w3 :weight} (gfi/update t2 disc (rng/fold-in k 3))]
      (and
       ;; the weight is the new score minus the old, less the freshly proposed choices
       (close? w (a/- (a/- (:score t2) (:score tr)) fresh-lp))
       ;; updating back with the discard restores the trace
       (= (:choices tr) (:choices t3))
       (close? (a/+ w w3) (a/negate fresh-lp))))))

(defn regenerating-nothing-changes-nothing [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {t2 :trace w :weight} (gfi/regenerate tr sel/none (rng/key (inc seed)))]
      (and (zero? w) (= (:choices tr) (:choices t2))))))

(defn regenerate-weight-is-the-retained-density-ratio [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          s (sel/from-paths (take 1 (cm/addresses (:choices tr))))
          {t2 :trace w :weight} (gfi/regenerate tr s (rng/fold-in k 1))
          retained (sel/from-paths (filter (set (cm/addresses (:choices tr)))
                                           (cm/addresses (sel/get-selected (:choices t2) (sel/complement s)))))]
      ;; Σ over retained choices of log p_new - log p_old: selected, fresh and
      ;; removed choices cancel against the proposal
      (close? w (a/- (gfi/project t2 retained) (gfi/project tr retained))))))

(defn project-splits-the-score [cases]
  (prop/for-all [[[_ gf args] seed] cases
                 picks (gen/vector gen/boolean 8)]
    (let [tr (gfi/simulate gf args (rng/key seed))
          s (sel/from-paths (pick-paths (cm/addresses (:choices tr)) picks))]
      (and (close? (gfi/project tr sel/all) (:score tr))
           (zero? (gfi/project tr sel/none))
           (close? (a/+ (gfi/project tr s) (gfi/project tr (sel/complement s))) (:score tr))))))

(defn propose-weight-is-the-assessed-density [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [{c :choices w :weight} (gfi/propose gf args (rng/key seed))]
      (close? w (:weight (gfi/assess gf args c))))))

(defn runs-are-pure-functions-of-the-key [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (= (:choices (gfi/simulate gf args (rng/key seed)))
       (:choices (gfi/simulate gf args (rng/key seed))))))

(def all
  "Every law above, by name."
  {:score-is-the-assessed-density score-is-the-assessed-density
   :generate-with-every-choice-is-deterministic generate-with-every-choice-is-deterministic
   :generate-without-constraints-has-weight-zero generate-without-constraints-has-weight-zero
   :generate-with-part-of-a-trace-and-its-key-reproduces-it generate-with-part-of-a-trace-and-its-key-reproduces-it
   :identity-update-has-weight-zero identity-update-has-weight-zero
   :updates-are-reversible updates-are-reversible
   :regenerating-nothing-changes-nothing regenerating-nothing-changes-nothing
   :regenerate-weight-is-the-retained-density-ratio regenerate-weight-is-the-retained-density-ratio
   :project-splits-the-score project-splits-the-score
   :propose-weight-is-the-assessed-density propose-weight-is-the-assessed-density
   :runs-are-pure-functions-of-the-key runs-are-pure-functions-of-the-key})

(defmacro deflaws
  "A defspec per law in gen.laws/all, named prefix-law, each over cases."
  [prefix cases num-tests]
  `(do ~@(for [law (keys all)]
           `(clojure.test.check.clojure-test/defspec ~(symbol (str prefix "-" (name law))) ~num-tests
              ((get all ~law) ~cases)))))
