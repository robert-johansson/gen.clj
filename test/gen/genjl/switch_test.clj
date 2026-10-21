(ns gen.genjl.switch-test
  "Gen.jl 0.4.8's test/modeling_library/switch.jl: a bare Switch, a Switch
   called from a generative function, branches that call another, and the
   discard of a branch change. Branch indices are 0-based (D2): Gen.jl's
   branch 1 is branch 0 here. Twice Gen.jl constrains an address its model
   lacks, which it ignores there and gen.clj rejects (D11): the ports drop
   those constraints. Most assertions are Gen.jl's
   isapprox(get_score(tr), get_score(new_tr) - w): a revision's weight is
   its change of score."
  (:refer-clojure :exclude [+ - *])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :refer [+ - *]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.genjl.finite-diff :refer [logpdf-grad]]
            [gen.genjl.support :as support :refer [approx?]]
            [gen.gfi :as gfi]
            [gen.gradients :as g]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :refer [close?]]))

(def ^:private x 5.0)
(def ^:private y 3.0)
(def ^:private std 3.0)

(defn- lp
  "logpdf(normal, z, mean, 3.0)"
  [z mean]
  (support/logpdf d/normal z mean std))

(defn- dz
  "logpdf_grad(normal, z, mean, 3.0)[1]"
  [z mean]
  (first (logpdf-grad d/normal z mean std)))

(def ^:private unknown [diff/unknown-change])

(defn- balances?
  "isapprox(get_score(tr), get_score(new_tr) - w) for the result of a
   revision of tr."
  [tr {t :trace w :weight}]
  (approx? (:score tr) (- (:score t) w)))

(defn- gradient-at
  "choice_gradients' gradient at addr, of gf's trace on args with addr
   constrained to z."
  [gf args addr z]
  (let [tr (:trace (gfi/generate gf args (cm/choicemap addr z) (rng/key 1)))]
    (cm/get-choice (:gradients (g/choice-gradients tr (sel/select addr))) addr)))

(def ^:private zs
  "The values the gradient testsets loop over."
  [1.0 3.0 5.0 10.0])

;; ---------------------------------------------------------------------------
;; The bare combinator
;; ---------------------------------------------------------------------------

(def bang
  "switch.jl:21 and 218 (bang0, bang2): z ~ normal(x + y, 3)."
  (gen [x y] (trace :z (d/normal (+ x y) std))))

(def fuzz
  "switch.jl:27 (fuzz0): z ~ normal(x + 2y, 3)."
  (gen [x y] (trace :z (d/normal (+ x (* 2 y)) std))))

(def sc (comb/switch bang fuzz))

;; switch.jl:35
(deftest bare-simulate
  (let [tr (gfi/simulate sc [0 x y] (rng/key 2))]
    (is (approx? (lp (tr :z) (+ x y)) (:score tr))))
  (let [tr (gfi/simulate sc [1 x y] (rng/key 3))]
    (is (approx? (lp (tr :z) (+ x (* 2 y))) (:score tr)))))

;; switch.jl:42
(deftest bare-generate
  (let [{tr :trace w :weight} (gfi/generate sc [1 x y] (cm/choicemap :z 5.0) (rng/key 4))]
    (is (close? 5.0 (tr :z)))
    (is (approx? (lp 5.0 (+ x (* 2 y))) w))))

;; switch.jl:51
(deftest bare-assess
  (is (approx? (lp 5.0 (+ x (* 2 y))) (:weight (gfi/assess sc [1 x y] (cm/choicemap :z 5.0) (rng/key 5))))))

;; switch.jl:58
(deftest bare-propose
  (let [{c :choices w :weight} (gfi/propose sc [1 x y] (rng/key 6))]
    (is (approx? (lp (cm/get-choice c :z) (+ x (* 2 y))) w))))

;; switch.jl:63: Gen.jl constrains :x => :z, an address the bare Switch
;; lacks (its choice is :z), and update ignores it; dropped here (D11)
(deftest bare-update
  (let [tr (gfi/simulate sc [0 x y] (rng/key 7))
        argdiffs [diff/unknown-change diff/no-change diff/no-change]]
    (is (balances? tr (gfi/update tr [1 x y] argdiffs cm/EMPTY (rng/key 8))))
    (is (balances? tr (gfi/update tr [0 x y] argdiffs cm/EMPTY (rng/key 9))))))

;; switch.jl:97
(deftest bare-choice-gradients
  (doseq [z zs]
    (is (approx? (dz z (+ x y)) (gradient-at sc [0 x y] :z z)) (str "branch 0, z " z))
    (is (approx? (dz z (+ x (* 2 y))) (gradient-at sc [1 x y] :z z)) (str "branch 1, z " z))))

;; ---------------------------------------------------------------------------
;; A Switch called from a generative function
;; ---------------------------------------------------------------------------

(def bang1
  "switch.jl:115: bang with its own std, a parameter."
  (gen [x y] (trace :z (d/normal (+ x y) (param :bang1-std)))))

(def fuzz1
  "switch.jl:121: fuzz with its own std, a parameter."
  (gen [x y] (trace :z (d/normal (+ x (* 2 y)) (param :fuzz1-std)))))

(def bam
  "switch.jl:128, holding both branches' parameters, as init_param! sets
   them (D7: one store per model)."
  (dyn/with-params (gen [s] (splice :x (comb/switch bang1 fuzz1) s x y)) {:bang1-std 3.0 :fuzz1-std 3.0}))

;; switch.jl:134
(deftest hierarchy-simulate
  (let [tr (gfi/simulate bam [1] (rng/key 10))]
    (is (approx? (lp (tr [:x :z]) (+ x (* 2 y))) (:score tr)))))

;; switch.jl:139
(deftest hierarchy-generate
  (let [{tr :trace w :weight} (gfi/generate bam [1] (cm/choicemap [:x :z] 5.0) (rng/key 11))]
    (is (close? 5.0 (tr [:x :z])))
    (is (approx? (lp 5.0 (+ x (* 2 y))) w))))

;; switch.jl:148
(deftest hierarchy-assess
  (is (approx? (lp 5.0 (+ x (* 2 y))) (:weight (gfi/assess bam [1] (cm/choicemap [:x :z] 5.0) (rng/key 12))))))

;; switch.jl:155
(deftest hierarchy-propose
  (let [{c :choices w :weight} (gfi/propose bam [1] (rng/key 13))]
    (is (approx? (lp (cm/get-choice c [:x :z]) (+ x (* 2 y))) w))))

;; switch.jl:160
(deftest hierarchy-update
  (let [tr (gfi/simulate bam [1] (rng/key 14))]
    (is (balances? tr (gfi/update tr [1] unknown (cm/choicemap [:x :z] 5.0) (rng/key 15))))
    (is (balances? tr (gfi/update tr [0] unknown (cm/choicemap [:x :z] 10.0) (rng/key 16))))))

;; switch.jl:171: the branches share :z, so nothing is resampled
(deftest hierarchy-regenerate
  (let [tr (gfi/simulate bam [1] (rng/key 17))]
    (doseq [s [0 1 0]]
      (is (balances? tr (gfi/regenerate tr [s] unknown sel/none (rng/key 18))) (str "to branch " s)))))

;; switch.jl:182
(deftest hierarchy-choice-gradients
  (doseq [z zs]
    (is (approx? (dz z (+ x y)) (gradient-at bam [0] [:x :z] z)) (str "branch 0, z " z))
    (is (approx? (dz z (+ x (* 2 y))) (gradient-at bam [1] [:x :z] z)) (str "branch 1, z " z))))

;; switch.jl:199: Gen.jl constrains :z, which bam lacks (its choice is
;; :x => :z), and generate ignores it; dropped here (D11), so each of the
;; four runs draws z
(deftest hierarchy-accumulate-param-gradients
  (doseq [k (range 4)]
    (let [dstd (fn [tr mean] (nth (logpdf-grad d/normal (tr [:x :z]) mean std) 2))
          tr0 (gfi/simulate bam [0] (rng/fold-in (rng/key 19) k))
          tr1 (gfi/simulate bam [1] (rng/fold-in (rng/key 20) k))]
      (is (approx? (dstd tr0 (+ x y)) (:bang1-std (g/param-gradients tr0 1.0))) (str "run " k))
      (is (approx? (dstd tr1 (+ x (* 2 y))) (:fuzz1-std (g/param-gradients tr1 1.0))) (str "run " k)))))

;; ---------------------------------------------------------------------------
;; Branches that call another generative function
;; ---------------------------------------------------------------------------

(def fuzz2
  "switch.jl:223: fuzz, then bang at :q on z."
  (gen [x y] (let [z (trace :z (d/normal (+ x (* 2 y)) std))] (splice :q bang z y) z)))

(def bam2
  "switch.jl:230"
  (gen [s] (splice :x (comb/switch bang fuzz2) s x y)))

(defn- fuzz2-density
  "fuzz2's log density at z and its call's q."
  [z q]
  (+ (lp z (+ x (* 2 y))) (lp q (+ z y))))

;; switch.jl:236
(deftest nested-simulate
  (let [tr (gfi/simulate bam2 [0] (rng/key 21))]
    (is (approx? (lp (tr [:x :z]) (+ x y)) (:score tr))))
  (let [tr (gfi/simulate bam2 [1] (rng/key 22))]
    (is (approx? (fuzz2-density (tr [:x :z]) (tr [:x :q :z])) (:score tr)))))

;; switch.jl:243
(deftest nested-generate
  (let [cs (cm/choicemap [:x :z] 5.0)]
    (let [{tr :trace w :weight} (gfi/generate bam2 [0] cs (rng/key 23))]
      (is (close? 5.0 (tr [:x :z])))
      (is (approx? (lp 5.0 (+ x y)) w)))
    (let [{tr :trace w :weight} (gfi/generate bam2 [1] cs (rng/key 24))]
      (is (close? 5.0 (tr [:x :z])))
      (is (approx? (lp (tr [:x :z]) (+ x (* 2 y))) w)))))

;; switch.jl:256
(deftest nested-assess
  (let [cs (cm/choicemap [:x :z] 5.0)]
    (is (approx? (lp 5.0 (+ x y)) (:weight (gfi/assess bam2 [0] cs (rng/key 25)))))
    (is (approx? (fuzz2-density 5.0 5.0) (:weight (gfi/assess bam2 [1] (cm/set-choice cs [:x :q :z] 5.0) (rng/key 26)))))))

;; switch.jl:266
(deftest nested-propose
  (let [{c :choices w :weight} (gfi/propose bam2 [0] (rng/key 27))]
    (is (approx? (lp (cm/get-choice c [:x :z]) (+ x y)) w)))
  (let [{c :choices w :weight} (gfi/propose bam2 [1] (rng/key 28))]
    (is (approx? (fuzz2-density (cm/get-choice c [:x :z]) (cm/get-choice c [:x :q :z])) w))))

;; switch.jl:273
(deftest nested-update
  (let [tr (gfi/simulate bam2 [1] (rng/key 29))]
    (is (balances? tr (gfi/update tr [1] unknown (cm/choicemap [:x :z] 5.0) (rng/key 30))))
    (is (balances? tr (gfi/update tr [0] unknown (cm/choicemap [:x :z] 10.0) (rng/key 31))))))

;; switch.jl:284: the same branch with nothing selected (the other three
;; assertions are pending)
(deftest nested-regenerate
  (let [tr (gfi/simulate bam2 [1] (rng/key 32))]
    (is (balances? tr (gfi/regenerate tr [1] unknown sel/none (rng/key 33))))))

;; switch.jl:297
(deftest nested-choice-gradients
  (doseq [z zs]
    (is (approx? (dz z (+ x y)) (gradient-at bam2 [0] [:x :z] z)) (str "z " z))))

;; ---------------------------------------------------------------------------
;; The discard of a branch change
;; ---------------------------------------------------------------------------

(def bang3
  "switch.jl:311: bang, then bang at :q on z."
  (gen [x y] (let [z (trace :z (d/normal (+ x y) std))] (splice :q bang z y) z)))

(def fuzz3
  "switch.jl:317: fuzz, :m, then bang3 at :q on z."
  (gen [x y]
    (let [z (trace :z (d/normal (+ x (* 2 y)) std))]
      (trace :m (d/normal (+ x (* 3 y)) std))
      (splice :q bang3 z y)
      z)))

(def bam3
  "switch.jl:325"
  (gen [s] (splice :x (comb/switch bang3 fuzz3) s x y)))

;; switch.jl:331
(deftest discard-update
  (let [tr (gfi/simulate bam3 [1] (rng/key 34))]
    (testing "the same branch: the constrained choice is discarded"
      (let [{d :discard :as r} (gfi/update tr [1] unknown (cm/choicemap [:x :z] 5.0) (rng/key 35))]
        (is (close? (tr [:x :z]) (cm/get-choice d [:x :z])))
        (is (balances? tr r))))
    (testing "branch 1 to 0: bang3's call at :q, bang, lacks bang3's own :q"
      (let [{d :discard :as r} (gfi/update tr [0] unknown (cm/choicemap [:x :z] 10.0) (rng/key 36))]
        (is (close? (tr [:x :q :q :z]) (cm/get-choice d [:x :q :q :z])))
        (is (balances? tr r))))))

;; Pending
;; - switch.jl:85, 89 and 94 (bare regenerate) and 288, 292 and 294
;;   (regenerate of branches that call another), isapprox(get_score(tr),
;;   get_score(new_tr) - w): Gen.jl's regenerate weight there is wrong (U1),
;;   genclj-c5rl.
;;
;; Not ported
;; - switch.jl:10, a SwitchTrace built by hand: Gen.jl's internal
;;   constructor (n/a).
