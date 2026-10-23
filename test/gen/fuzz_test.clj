(ns gen.fuzz-test
  "The GFI laws, and batched path equivalence, over random generative
   functions (gen.random-models): random families, dependencies, branches,
   splices, shared-namespace splices, Map and Unfold."
  (:refer-clojure :exclude [-])
  (:require [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.batched :as b]
            [gen.array :as array :refer [-]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.selection :as sel]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.random-models :as rm]))

(laws/deflaws random rm/gen-model 200)

(defspec batched-particles-of-random-models-are-their-scalar-traces 100
  ;; with spot checks off, so that the batched scores themselves are tested
  ;; rather than the fallback a wrong one would trigger
  (prop/for-all [[_ gf args] rm/gen-model
                 seed gen/nat]
    (binding [b/*spot-checks* 0]
      (let [ps (b/simulate gf args 3 (rng/key seed))]
        (every? true? (map #(laws/close? %1 (:weight (gfi/assess gf args (:choices %2))))
                           (array/->clj (b/scores ps)) (b/traces ps)))))))

;; per particle: w = score' - score - log p(fresh), and the discard is
;; exactly what the constraint overwrote or removed (fresh draws may differ
;; from a scalar run's, so the law, not equality with one, is the check)
(defn- obeys-the-update-law?
  "Does the update of old to new under the constraints sigma, with weight w
   and discard d, obey the law?"
  [old new w d sigma]
  (let [was (set (cm/addresses (:choices old)))
        constrained (set (cm/addresses sigma))
        now (set (cm/addresses (:choices new)))
        fresh (remove #(or (was %) (constrained %)) now)
        fresh-lp (if (seq fresh) (gfi/project new (sel/from-paths fresh)) 0.0)
        expected-d (sel/get-selected (:choices old)
                                     (sel/from-paths (filter #(or (constrained %) (not (now %))) was)))]
    (and (laws/close? w (- (- (:score new) (:score old)) fresh-lp))
         (= expected-d d))))

(defspec batched-updates-of-random-models-obey-the-update-law 100
  (prop/for-all [[_ gf args] rm/gen-model
                 seed gen/nat]
    (binding [b/*spot-checks* 0]
      (let [ps (b/simulate gf args 3 (rng/key seed))
            probe (gfi/simulate gf args (rng/key (inc seed)))
            p (first (cm/addresses (:choices probe)))
            sigma (if p (cm/choicemap p (cm/get-choice (:choices probe) p)) cm/EMPTY)
            olds (b/traces ps)
            r (try (b/update ps args (vec (repeat (count args) diff/no-change)) sigma (rng/key 2))
                   (catch clojure.lang.ExceptionInfo _ ::rejected))]
        (or (= ::rejected r)
            (let [{ps' :particles ws :weights ds :discards} r]
              (every? true? (map #(obeys-the-update-law? %1 %2 %3 %4 sigma) olds (b/traces ps') (array/->clj ws) ds))))))))

(defspec per-particle-generates-of-random-models-are-their-scalar-generates 100
  ;; particle i constrained by every choice of its own scalar run: it holds
  ;; them, and its weight is its scalar generate's (runs of different
  ;; structures, which do not stack, fall back, which must hold too)
  (prop/for-all [[_ gf args] rm/gen-model
                 seed gen/nat]
    (binding [b/*spot-checks* 0]
      (let [own (mapv #(:choices (gfi/simulate gf args %)) (rng/split (rng/key seed) 3))
            {ps :particles ws :weights} (b/generate gf args cm/EMPTY 3 (rng/key 1) {:per-particle own})]
        (every? true? (map (fn [c w tr] (and (= c (:choices tr)) (laws/close? w (:weight (gfi/generate gf args c (rng/key 0))))))
                           own (array/->clj ws) (b/traces ps)))))))

(defspec per-particle-updates-of-random-models-obey-the-update-law 100
  ;; particle i's constraint at the first address of its own scalar run
  (prop/for-all [[_ gf args] rm/gen-model
                 seed gen/nat]
    (binding [b/*spot-checks* 0]
      (let [ps (b/simulate gf args 3 (rng/key seed))
            own (mapv #(let [c (:choices (gfi/simulate gf args %))
                             p (first (cm/addresses c))]
                         (if p (cm/choicemap p (cm/get-choice c p)) cm/EMPTY))
                      (rng/split (rng/key (inc seed)) 3))
            olds (b/traces ps)
            r (try (b/update ps args (vec (repeat (count args) diff/no-change)) cm/EMPTY (rng/key 2) {:per-particle own})
                   (catch clojure.lang.ExceptionInfo _ ::rejected))]
        (or (= ::rejected r)
            (let [{ps' :particles ws :weights ds :discards} r]
              (every? true? (map obeys-the-update-law? olds (b/traces ps') (array/->clj ws) ds own))))))))
