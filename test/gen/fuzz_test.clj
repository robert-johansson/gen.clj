(ns gen.fuzz-test
  "The GFI laws, and batched path equivalence (batched gradients too), over
   random generative functions (gen.random-models): random families,
   dependencies, branches, splices, shared-namespace splices, Map, Unfold,
   Switch, Scan and Mix; and, over Switch-heavy models whose Switches run by
   partition in a batched run, every batched operation particle by particle
   against the scalar handler."
  (:refer-clojure :exclude [+ -])
  (:require [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.batched :as b]
            [gen.array :as array :refer [+ -]]
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
;; from a scalar run's, so the law, not equality with one, is the check).
;; A particle the update takes outside its support has w -Inf, and the
;; batched run, which cannot stop for it (gen.dynamic), discards what the
;; whole run overwrote, where its rebuilt scalar trace (b/traces) stops at
;; that choice (DESIGN.md §9): its discard is not compared.
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
         (or (gfi/impossible? (double w)) (= expected-d d)))))

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

;; ---------------------------------------------------------------------------
;; Switches by partition: Switch-heavy models, particle by particle
;; ---------------------------------------------------------------------------

;; Every Switch of these models has an index that differs across the
;; particles, so a batched run partitions them (gen.combinators), Switches
;; nested in branches and in Map kernels too. Spot checks are off, so the
;; batched results themselves are compared with the scalar handler's, and a
;; model must stay batched.

(def ^:private switch-case (gen/tuple rm/gen-switch-model gen/nat (gen/choose 1 6)))

(defn- index-paths
  "The paths of a trace's top-level Switch and Mix indices, which every
   particle makes."
  [tr]
  (filterv #(and (= 2 (count %)) (#{:i :index} (peek %))) (cm/addresses (:choices tr))))

(defn- call-of
  "The path of the call an index at p picks: a Switch's at :s, a Mix's
   component at :component."
  [p]
  (conj (pop p) (if (= :index (peek p)) :component :s)))

(defn- own-indices
  "n particles' own values at the index paths, from scalar runs."
  [gf args paths n seed]
  (mapv #(sel/get-selected (:choices (gfi/simulate gf args %)) (sel/from-paths paths)) (rng/split (rng/key seed) n)))

(defn- consistent? [gf args tr] (laws/close? (:score tr) (:weight (gfi/assess gf args (:choices tr)))))

(defspec switch-heavy-particles-are-their-scalar-traces 100
  (prop/for-all [[[_ gf args] seed n] switch-case]
    (binding [b/*spot-checks* 0]
      (let [ps (b/simulate gf args n (rng/key seed))]
        (and (b/batched? ps)
             (every? true? (map #(laws/close? %1 (:weight (gfi/assess gf args (:choices %2))))
                                (array/->clj (b/scores ps)) (b/traces ps))))))))

(defspec switch-heavy-generates-are-each-particle-s 100
  ;; each particle constrained to its own indices: it holds them, and its
  ;; weight is their density given its other choices
  (prop/for-all [[[_ gf args] seed n] switch-case]
    (binding [b/*spot-checks* 0]
      (let [paths (index-paths (gfi/simulate gf args (rng/key seed)))
            own (own-indices gf args paths n (inc seed))
            {ps :particles ws :weights} (b/generate gf args cm/EMPTY n (rng/key seed) {:per-particle own})]
        (and (b/batched? ps)
             (every? true? (map (fn [tr w c] (and (= (ts/host c) (ts/host (sel/get-selected (:choices tr) (sel/from-paths paths))))
                                                  (consistent? gf args tr)
                                                  (laws/close? w (gfi/project tr (sel/from-paths paths)))))
                                (b/traces ps) (array/->clj ws) own)))))))

(defspec switch-heavy-updates-are-each-particle-s-update 100
  ;; each particle given its own new indices, which moves particles between
  ;; branches: its scalar update, with the choices the batched one drew
  ;; afresh constrained too, makes its choices and discard, and its weight
  ;; less those fresh choices' density (a particle the update takes outside
  ;; its support, w -Inf, has the whole batched run's discard: see
  ;; obeys-the-update-law?)
  (prop/for-all [[[_ gf args] seed n] switch-case]
    (binding [b/*spot-checks* 0]
      (let [ps (b/simulate gf args n (rng/key seed))
            olds (b/traces ps)
            own (own-indices gf args (index-paths (first olds)) n (inc seed))
            {ps' :particles ws :weights ds :discards} (b/update ps args (diff/no-changes (count args)) cm/EMPTY (rng/key 2) {:per-particle own})]
        (and (b/batched? ps')
             (every? true?
                     (map (fn [old new w d c]
                            (let [was (set (cm/addresses (:choices old)))
                                  fresh (remove #(or (was %) (cm/has-value? c %)) (cm/addresses (:choices new)))
                                  sigma (cm/merge c (sel/get-selected (:choices new) (sel/from-paths fresh)))
                                  r (gfi/update old sigma (rng/key 0))]
                              (and (= (ts/host (:choices new)) (ts/host (:choices (:trace r))))
                                   (or (gfi/impossible? (double w)) (= (ts/host d) (ts/host (:discard r))))
                                   (laws/close? (:weight r) (+ w (if (seq fresh) (gfi/project new (sel/from-paths fresh)) 0.0))))))
                          olds (b/traces ps') (array/->clj ws) ds own)))))))

(defn- selected-parts
  "Some of a trace's top-level Switch and Mix indices and calls, by the
   booleans picks."
  [tr picks]
  (sel/from-paths (keep (fn [[p pick?]] (when pick? p))
                        (map vector (mapcat (fn [p] [p (call-of p)]) (index-paths tr)) (cycle picks)))))

(defspec switch-heavy-regenerates-obey-the-regenerate-law 100
  ;; some indices and Switch calls regenerated: w = log p(new)/p(old) over
  ;; the choices kept, those both make that were not selected (fresh and
  ;; dropped ones cancel)
  (prop/for-all [[[_ gf args] seed n] switch-case
                 picks (gen/vector gen/boolean 1 6)]
    (binding [b/*spot-checks* 0]
      (let [ps (b/simulate gf args n (rng/key seed))
            s (selected-parts (first (b/traces ps)) picks)
            {ps' :particles ws :weights} (b/regenerate ps args (diff/no-changes (count args)) s (rng/key 3))]
        (and (b/batched? ps')
             (every? true?
                     (map (fn [old new w]
                            (let [was (set (cm/addresses (:choices old)))
                                  kept (sel/from-paths (filter #(and (was %) (not (sel/selected? s %))) (cm/addresses (:choices new))))]
                              (and (consistent? gf args new)
                                   (laws/close? w (- (gfi/project new kept) (gfi/project old kept))))))
                          (b/traces ps) (b/traces ps') (array/->clj ws))))))))

(defspec switch-heavy-take-select-and-project-are-each-particle-s 60
  (prop/for-all [[[_ gf args] seed n] switch-case
                 picks (gen/vector gen/nat 1 8)
                 mask (gen/vector gen/boolean 6)]
    (binding [b/*spot-checks* 0]
      (let [ps (b/simulate gf args n (rng/key seed))
            trs (b/traces ps)
            paths (index-paths (first trs))
            qs (:particles (b/regenerate ps args (diff/no-changes (count args)) (sel/from-paths paths) (rng/key 4)))
            idx (mapv #(mod % n) picks)
            keep (vec (take n mask))
            taken (b/take ps idx)
            chosen (b/select keep ps qs)
            choices (fn [ps] (mapv (comp ts/host :choices) (b/traces ps)))
            s (selected-parts (first trs) mask)]
        (and (b/batched? taken) (b/batched? chosen)
             (= (mapv (choices ps) idx) (choices taken))
             (= (mapv (fn [keep? p q] (if keep? p q)) keep (choices ps) (choices qs)) (choices chosen))
             (every? true? (map laws/close? (array/->clj (b/project ps s)) (map #(gfi/project % s) trs))))))))

(defn- same-weight?
  "laws/close?, NaN equal to NaN: a chain of moves may reach a trace of
   density zero (a kept value outside its new support), after which a
   weight is -Inf - -Inf."
  [x y]
  (or (and (Double/isNaN (ts/number x)) (Double/isNaN (ts/number y))) (laws/close? x y)))

(defspec switch-heavy-particles-stay-their-scalar-traces-through-chains-of-operations 60
  ;; MH-like moves (regenerate, then select), resampling (take) and updates
  ;; that move particles between branches, in a random order: a particle's
  ;; data is then joined, merged and regrouped before later runs read it,
  ;; and each operation's weights obey its law
  (prop/for-all [[[_ gf args] seed n] switch-case
                 ops (gen/vector (gen/tuple (gen/elements [:regenerate :select :take :update]) (gen/vector gen/boolean 6)) 1 6)]
    (binding [b/*spot-checks* 0]
      (loop [ps (b/simulate gf args n (rng/key seed))
             [[op picks] & more] ops
             k 0]
        (let [olds (b/traces ps)
              n (b/size ps)
              s (selected-parts (first olds) picks)
              obeys? (fn [ps' ws kept]
                       (every? true? (map (fn [old new w]
                                            (let [was (set (cm/addresses (:choices old)))
                                                  kept (sel/from-paths (filter #(and (was %) (kept %)) (cm/addresses (:choices new))))]
                                              (and (consistent? gf args new)
                                                   (same-weight? w (- (gfi/project new kept) (gfi/project old kept))))))
                                          olds (b/traces ps') (array/->clj ws))))
              [ps' ok?]
              (case op
                :regenerate (let [{ps' :particles ws :weights} (b/regenerate ps args (diff/no-changes (count args)) s (rng/key k))]
                              [ps' (obeys? ps' ws #(not (sel/selected? s %)))])
                :select (let [qs (:particles (b/regenerate ps args (diff/no-changes (count args)) s (rng/key k)))
                              keep (vec (take n (cycle picks)))
                              chosen (b/select keep ps qs)]
                          [chosen (= (mapv (fn [keep? p q] (if keep? (ts/host (:choices p)) (ts/host (:choices q)))) keep olds (b/traces qs))
                                     (mapv (comp ts/host :choices) (b/traces chosen)))])
                :take (let [idx (vec (keep-indexed (fn [i pick?] (when pick? (mod i n))) (take (max n 6) (cycle picks))))
                            idx (if (seq idx) idx [0])]
                        [(b/take ps idx) true])
                :update (let [own (own-indices gf args (index-paths (first olds)) n (+ seed k))
                              {ps' :particles ws :weights} (b/update ps args (diff/no-changes (count args)) cm/EMPTY (rng/key k) {:per-particle own})]
                          ;; the constrained indices and the fresh choices
                          ;; are not kept
                          [ps' (every? true? (map (fn [old new w c]
                                                    (let [was (set (cm/addresses (:choices old)))
                                                          fresh (remove #(or (was %) (cm/has-value? c %)) (cm/addresses (:choices new)))
                                                          sigma (cm/merge c (sel/get-selected (:choices new) (sel/from-paths fresh)))]
                                                      (same-weight? (:weight (gfi/update old sigma (rng/key 0)))
                                                                    (+ w (if (seq fresh) (gfi/project new (sel/from-paths fresh)) 0.0)))))
                                                  olds (b/traces ps') (array/->clj ws) own))]))]
          (cond (not (and ok? (b/batched? ps') (every? #(consistent? gf args %) (b/traces ps'))
                          (every? true? (map #(laws/close? %1 (:score %2)) (array/->clj (b/scores ps')) (b/traces ps')))))
                false
                (seq more) (recur ps' more (inc k))
                :else true))))))
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
