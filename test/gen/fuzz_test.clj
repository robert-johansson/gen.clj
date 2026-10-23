(ns gen.fuzz-test
  "The GFI laws, and batched path equivalence (batched gradients too), over
   random generative functions (gen.random-models): random families,
   dependencies, branches, splices, shared-namespace splices, Map and
   Unfold."
  (:refer-clojure :exclude [-])
  (:require [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.batched :as b]
            [gen.array :as array :refer [-]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.gradients :as grad]
            [gen.selection :as sel]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.random-models :as rm]
            [gen.test-support :as ts]))

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

(defspec every-particle-s-choice-gradients-of-random-models-are-each-trace-s 100
  ;; with respect to the continuous choices every particle holds: one
  ;; batched gradient, or one per trace where the collection fell back
  ;; (vmap's semantics either way; a NaN where the trace's own gradient has
  ;; one). A batched collection's gradient must not fall back itself
  (prop/for-all [[_ gf args] rm/gen-model
                 seed gen/nat]
    (binding [b/*spot-checks* 0]
      (let [ps (b/simulate gf args 3 (rng/key seed))
            trs (b/traces ps)
            continuous? #(every? grad/continuous? (array/tree-leaves %))
            paths (filter (fn [p] (every? #(and (cm/has-value? (:choices %) p) (continuous? (cm/get-choice (:choices %) p))) trs))
                          (cm/addresses (:choices (first trs))))
            s (sel/from-paths paths)
            same? (fn [x y] (ts/close? x y (merge {:equal-nan? true} (if (ts/f32?) {:rtol 1e-4 :atol 1e-3} {}))))]
        (or (empty? paths)
            (let [r (if (b/batched? ps)
                      (with-redefs [b/traces (fn [_] (throw (ex-info "the gradient fell back" {})))]
                        (grad/choice-gradients ps s))
                      (grad/choice-gradients ps s))
                  each (map #(:gradients (grad/choice-gradients % s)) trs)]
              (every? (fn [p] (same? (cm/get-choice (:gradients r) p) (array/stack (mapv #(cm/get-choice % p) each))))
                      paths)))))))
