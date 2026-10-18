(ns gen.dynamic-test
  "The dynamic modeling language against the GFI's laws (Cusumano-Towner's
   thesis; docs/research/genjl-parity.md §3), on fixed-structure,
   structure-changing, spliced and discrete models."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as a]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def linreg
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (a/+ (a/* slope x) intercept) 1)))
      slope)))

(def branching
  (gen []
    (if (trace :switch (d/bernoulli 0.4))
      (trace :a (d/normal 0 1))
      (+ (trace :b (d/gamma 2 1)) (trace :c (d/poisson 3.0))))))

(def observation (gen [mu] (trace :x (d/normal mu 1))))

(def hierarchical
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (mapv #(splice [:obs %] observation mu) (range n)))))

(def discrete
  (gen []
    (let [k (trace :k (d/categorical [0.2 0.5 0.3]))
          n (trace :n (d/poisson (a/+ 1.0 k)))]
      (trace :flip (d/bernoulli (a// 1.0 (a/+ 2.0 n)))))))

(def models
  [[:linreg linreg [[1.0 2.0 3.0]]]
   [:branching branching []]
   [:hierarchical hierarchical [3]]
   [:discrete discrete []]])

(def gen-case (gen/tuple (gen/elements models) gen/nat))

(defn- close? [x y] (a/allclose? x y {:rtol 1e-12 :atol 1e-12}))

;; ---------------------------------------------------------------------------
;; Laws
;; ---------------------------------------------------------------------------

(defspec score-is-the-assessed-density 100
  (prop/for-all [[[_ gf args] seed] gen-case]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {w :weight rv :retval} (gfi/assess gf args (:choices tr))]
      (and (close? (:score tr) w) (= (:retval tr) rv)))))

(defspec generate-with-every-choice-is-deterministic 100
  (prop/for-all [[[_ gf args] seed] gen-case]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {g :trace w :weight} (gfi/generate gf args (:choices tr) (rng/key (inc seed)))]
      (and (= (:choices tr) (:choices g)) (close? w (:score tr))))))

(defspec generate-without-constraints-has-weight-zero 100
  (prop/for-all [[[_ gf args] seed] gen-case]
    (zero? (:weight (gfi/generate gf args cm/EMPTY (rng/key seed))))))

(defspec generate-with-part-of-a-trace-and-its-key-reproduces-it 150
  (prop/for-all [[[_ gf args] seed] gen-case
                 picks (gen/vector gen/boolean 8)]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          s (sel/from-paths (keep (fn [[p pick?]] (when pick? p))
                                  (map vector (cm/addresses (:choices tr)) (cycle picks))))
          constraints (sel/get-selected (:choices tr) s)
          {g :trace w :weight} (gfi/generate gf args constraints k)]
      ;; every site consumes one key, so the unconstrained draws are the same,
      ;; and the weight is the log density of the constrained choices
      (and (= (:choices tr) (:choices g))
           (close? w (gfi/project g (sel/from-choicemap constraints)))))))

(defspec identity-update-has-weight-zero 100
  (prop/for-all [[[_ gf args] seed] gen-case]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {t2 :trace w :weight disc :discard rd :retdiff} (gfi/update tr cm/EMPTY (rng/key (inc seed)))]
      (and (zero? w) (= (:choices tr) (:choices t2)) (cm/empty? disc) (diff/no-change? rd)))))

(defspec updates-are-reversible 150
  (prop/for-all [[[_ gf args] seed] gen-case]
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

(defspec regenerating-nothing-changes-nothing 100
  (prop/for-all [[[_ gf args] seed] gen-case]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {t2 :trace w :weight} (gfi/regenerate tr sel/none (rng/key (inc seed)))]
      (and (zero? w) (= (:choices tr) (:choices t2))))))

(defspec regenerate-weight-is-the-retained-density-ratio 150
  (prop/for-all [[[_ gf args] seed] gen-case]
    (let [k (rng/key seed)
          tr (gfi/simulate gf args k)
          s (sel/from-paths (take 1 (cm/addresses (:choices tr))))
          {t2 :trace w :weight} (gfi/regenerate tr s (rng/fold-in k 1))
          retained (sel/from-paths (filter (set (cm/addresses (:choices tr)))
                                           (cm/addresses (sel/get-selected (:choices t2) (sel/complement s)))))]
      ;; Σ over retained choices of log p_new - log p_old: selected, fresh and
      ;; removed choices cancel against the proposal
      (close? w (a/- (gfi/project t2 retained) (gfi/project tr retained))))))

(defspec project-splits-the-score 100
  (prop/for-all [[[_ gf args] seed] gen-case
                 picks (gen/vector gen/boolean 8)]
    (let [tr (gfi/simulate gf args (rng/key seed))
          addrs (cm/addresses (:choices tr))
          s (sel/from-paths (keep (fn [[p pick?]] (when pick? p)) (map vector addrs (cycle picks))))]
      (and (close? (gfi/project tr sel/all) (:score tr))
           (zero? (gfi/project tr sel/none))
           (close? (a/+ (gfi/project tr s) (gfi/project tr (sel/complement s))) (:score tr))))))

(defspec propose-weight-is-the-assessed-density 100
  (prop/for-all [[[_ gf args] seed] gen-case]
    (let [{c :choices w :weight} (gfi/propose gf args (rng/key seed))]
      (close? w (:weight (gfi/assess gf args c))))))

(defspec runs-are-pure-functions-of-the-key 50
  (prop/for-all [[[_ gf args] seed] gen-case]
    (= (:choices (gfi/simulate gf args (rng/key seed)))
       (:choices (gfi/simulate gf args (rng/key seed))))))

;; ---------------------------------------------------------------------------
;; Structure, splices, errors
;; ---------------------------------------------------------------------------

(deftest structure-changing-update
  (let [k (rng/key 4)
        tr (first (filter #(true? (% :switch)) (map #(gfi/simulate branching [] (rng/fold-in k %)) (range 50))))
        {t2 :trace w :weight disc :discard} (gfi/update tr (cm/choicemap :switch false) (rng/key 5))]
    (is (= #{[:switch] [:b] [:c]} (set (cm/addresses (:choices t2)))) "the other branch is sampled fresh")
    (is (= (cm/choicemap :switch true :a (tr :a)) disc) "the old value and the removed choice are discarded")
    (is (close? w (a/- (gfi/project t2 (sel/select :switch)) (:score tr))))
    (is (= (:choices tr) (:choices (:trace (gfi/update t2 disc (rng/key 6))))) "and the update reverses")))

(deftest splices-nest-choices
  (let [tr (gfi/simulate hierarchical [2] (rng/key 1))]
    (is (= [[:mu] [:obs 0 :x] [:obs 1 :x]] (cm/addresses (:choices tr))))
    (is (= (tr [:obs 1 :x]) (second (:retval tr))))
    (let [{t2 :trace} (gfi/update tr (cm/choicemap [:obs 0 :x] 9.0) (rng/key 2))]
      (is (= 9.0 (t2 [:obs 0 :x])))
      (is (= (tr [:obs 1 :x]) (t2 [:obs 1 :x]))))
    (testing "selections reach into splices"
      (let [{t2 :trace} (gfi/regenerate tr (sel/select [:obs 1 :x]) (rng/key 3))]
        (is (= (tr :mu) (t2 :mu)))
        (is (= (tr [:obs 0 :x]) (t2 [:obs 0 :x])))
        (is (not= (tr [:obs 1 :x]) (t2 [:obs 1 :x])))))
    (testing "an update with new arguments adds calls"
      (let [{t2 :trace w :weight} (gfi/update tr [3] [diff/unknown-change] cm/EMPTY (rng/key 4))]
        (is (= 3 (count (:retval t2))))
        (is (close? w 0.0) "the new call is a fresh proposal, which cancels")))))

(deftest errors-are-loud
  (let [k (rng/key 0)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                          (gfi/generate linreg [[1.0]] (cm/choicemap :typo 1.0) k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"used twice"
                          (gfi/simulate (gen [] (trace :x (d/normal 0 1)) (trace :x (d/normal 0 1))) [] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"lies under"
                          (gfi/simulate (gen [] (trace :x (d/normal 0 1)) (trace [:x :y] (d/normal 0 1))) [] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"nested choice map"
                          (gfi/generate linreg [[1.0]] (cm/choicemap [:slope :oops] 1.0) k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no value"
                          (gfi/assess linreg [[1.0]] (cm/choicemap :slope 1.0))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs a distribution"
                          (gfi/simulate (gen [] (trace :x 3.0)) [] k)))))

(deftest conditionals-branch-on-values
  (let [model (gen [c] (if c (trace :a (d/normal 0 1)) (trace :b (d/normal 0 1))))]
    (testing "a 1-element array that holds false takes the else branch (plain if would not)"
      (is (= [[:b]] (cm/addresses (:choices (gfi/simulate model [(a/array [false])] (rng/key 1)))))))
    (is (= [[:a]] (cm/addresses (:choices (gfi/simulate model [true] (rng/key 1))))))
    (testing "when, cond, and and or expand to if, so they are guarded too"
      (let [m (gen [c] (when c (trace :a (d/normal 0 1))) (cond (and c true) 1 :else 2))]
        (is (cm/empty? (:choices (gfi/simulate m [(a/array [false])] (rng/key 1)))))))
    (testing "local functions shadowing macro names are not expanded"
      (let [m (gen [] (let [when (fn [x] (* 2 x))] (when 21)))]
        (is (= 42 (m)))))))

(deftest batched-handler
  (testing "the same transitions run N particles at once"
    (let [k (rng/key 7)
          {tr :trace} (dyn/run linreg :simulate [[1.0 2.0]] {:key k :batch-shape [5]})]
      (is (= [5] (a/shape (tr :slope))))
      (is (= [5] (a/shape (:score tr))))
      (testing "each particle's score is its own assessed density"
        (doseq [[i c] (map-indexed vector (cm/unstack (:choices tr) 5))]
          (is (close? (a/take (:score tr) i) (:weight (gfi/assess linreg [[1.0 2.0]] c))))))))
  (testing "branching on a batch of values is loud, not silently wrong"
    (is (:gen.array/batched-branch
         (ex-data (try (dyn/run branching :simulate [] {:key (rng/key 1) :batch-shape [4]})
                       (catch clojure.lang.ExceptionInfo e e)))))))

(deftest untraced-calls
  (is (number? (linreg [1.0 2.0])) "calling a generative function runs it, like Gen.jl"))
