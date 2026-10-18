(ns gen.dynamic-test
  "The dynamic modeling language against the GFI's laws (Cusumano-Towner's
   thesis; docs/research/genjl-parity.md §3), on fixed-structure,
   structure-changing, spliced and discrete models."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as a]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
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

(def sub-x (gen [] (trace :y (d/normal 0 1))))

(def kind-changing
  "An address that holds a choice in one branch and a splice in the other."
  (gen []
    (if (trace :s (d/bernoulli 0.5))
      (trace :x (d/normal 0 1))
      (splice :x sub-x))))

(def spliced-distribution
  "A distribution called through the GFI, its parameter an argument."
  (gen [mu]
    (splice :x (d/normal mu 1))
    (trace :y (d/normal 0 1))))

(def p-mh
  "The thesis's p_mh (p. 92): b exists only when a is true."
  (gen []
    (let [z (trace :a (d/bernoulli 0.5))
          z (and z (trace :b (d/bernoulli 0.5)))]
      (trace :c (d/bernoulli (if z 0.9 0.1))))))

(def models
  [[:linreg linreg [[1.0 2.0 3.0]] [[[1.0 2.5]] [[0.5 1.0 1.5 2.0]]]]
   [:branching branching []]
   [:hierarchical hierarchical [3] [[1] [5]]]
   [:discrete discrete []]
   [:kind-changing kind-changing []]
   [:spliced-distribution spliced-distribution [0.0] [[2.0] [-1.5]]]
   [:p-mh p-mh []]])

(defn- close? [x y] (a/allclose? x y {:rtol 1e-12 :atol 1e-12}))

;; ---------------------------------------------------------------------------
;; Laws (gen.laws), and their checks on structure changes
;; ---------------------------------------------------------------------------

(laws/deflaws dynamic models 100)

(deftest update-without-fresh-choices-is-exactly-reversible
  ;; constraints covering the new branch: w = Δscore, the reverse weight is
  ;; -w and the reverse discard is σ (Prop 2.3.1, Def 2.3.1)
  (let [tr (first (filter #(true? (% :switch)) (map #(gfi/simulate branching [] (rng/key %)) (range 50))))
        sigma (cm/choicemap :switch false :b 2.0 :c 3)
        {t2 :trace w :weight disc :discard} (gfi/update tr sigma (rng/key 1))
        {t3 :trace w3 :weight disc3 :discard} (gfi/update t2 disc (rng/key 2))]
    (is (close? w (a/- (:score t2) (:score tr))))
    (is (= (cm/choicemap :switch true :a (tr :a)) disc))
    (is (= (:choices tr) (:choices t3)))
    (is (close? w3 (a/negate w)))
    (is (= sigma disc3))))

(deftest a-choice-that-becomes-a-call-is-discarded
  ;; regression for the audit's E1
  (let [tr (first (filter #(true? (% :s)) (map #(gfi/simulate kind-changing [] (rng/key %)) (range 50))))
        {t2 :trace w :weight disc :discard} (gfi/update tr (cm/choicemap :s false) (rng/key 1))]
    (is (= (cm/choicemap :s true :x (tr :x)) disc))
    (is (close? w (a/- (a/- (:score t2) (:score tr)) (gfi/project t2 (sel/select [:x :y])))))
    (is (= (:choices tr) (:choices (:trace (gfi/update t2 disc (rng/key 2))))))))

(def inlined
  "hierarchical with its spliced calls written inline."
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (mapv #(trace [:obs % :x] (d/normal mu 1)) (range n)))))

(deftest splicing-is-compositional
  ;; Prop 2.3.2: a splice updates as its body written inline would
  (let [tr (gfi/simulate hierarchical [3] (rng/key 3))
        tr-inline (:trace (gfi/generate inlined [3] (:choices tr) (rng/key 4)))]
    (is (close? (:score tr) (:score tr-inline)))
    (doseq [sigma [(cm/choicemap :mu 0.7) (cm/choicemap [:obs 1 :x] -2.0) (cm/choicemap :mu -0.3 [:obs 0 :x] 1.0)]]
      (let [r (gfi/update tr sigma (rng/key 5))
            ri (gfi/update tr-inline sigma (rng/key 5))]
        (is (close? (:weight r) (:weight ri)))
        (is (= (:discard r) (:discard ri)))
        (is (= (:choices (:trace r)) (:choices (:trace ri))))))
    (testing "and with changed arguments that remove calls"
      (let [r (gfi/update tr [1] [diff/unknown-change] cm/EMPTY (rng/key 6))
            ri (gfi/update tr-inline [1] [diff/unknown-change] cm/EMPTY (rng/key 6))]
        (is (close? (:weight r) (:weight ri)))
        (is (= (:discard r) (:discard ri)))))))

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

(deftest the-implicit-key-source
  (let [model (gen [] (trace :x (d/normal 0 1)))
        before (gfi/implicit-seed)]
    (gfi/seed! 123)
    (let [a [(model) (model)]]
      (gfi/seed! 123)
      (is (= a [(model) (model)]) "seed! makes calls without a key repeat exactly")
      (is (not= (first a) (second a)) "successive calls draw fresh keys")
      (is (= 123 (gfi/implicit-seed))))
    (gfi/seed! before)))

(deftest untraced-calls
  (is (number? (linreg [1.0 2.0])) "calling a generative function runs it, like Gen.jl"))
