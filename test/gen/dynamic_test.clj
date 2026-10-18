(ns gen.dynamic-test
  "The dynamic modeling language against the GFI's laws (Cusumano-Towner's
   thesis; docs/research/genjl-parity.md §3), on fixed-structure,
   structure-changing, spliced and discrete models."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check]
            [gen.array :as a]
            [gen.array.batch :as batch]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dist.core :as dc]
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

(def shared-body
  "A dynamic generative function spliced without an address."
  (gen [mu]
    (let [z (trace :z (d/normal mu 1))]
      (when (trace :more (d/bernoulli 0.5))
        (trace :w (d/normal z 1)))
      z)))

(def shared-namespace
  (gen []
    (let [mu (trace :mu (d/normal 0 1))
          z (splice shared-body mu)]
      (trace :obs (d/normal z 0.5)))))

(def mixtures
  "Mixture and product distributions at trace sites."
  (gen []
    (let [x (trace :x ((d/homogeneous-mixture d/normal) [0.3 0.7] [-2.0 1.0] [0.5 1.0]))
          [u b] (trace :pair ((d/product d/normal d/bernoulli) [x 1.0] [0.4]))]
      (trace :h ((d/heterogeneous-mixture [d/normal d/uniform]) [0.5 0.5] [u 1.0] [-1.0 1.0]))
      b)))

(def models
  [[:linreg linreg [[1.0 2.0 3.0]] [[[1.0 2.5]] [[0.5 1.0 1.5 2.0]]]]
   [:shared-namespace shared-namespace []]
   [:mixtures mixtures []]
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
                          (gfi/simulate (gen [] (trace :x 3.0)) [] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"has shape \[3\], but the distribution's values have shape \[\]"
                          (gfi/generate (gen [] (trace :y (d/normal 0 1))) [] (cm/choicemap :y (a/array [1.0 2.0 3.0])) k))
        "a constraint of another shape is not scored as several draws")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"nested choices"
                          (gfi/generate (gen [] (splice :x (d/normal 0 1))) [] (cm/choicemap [:x :y] 3.0) k))
        "constraints under a spliced distribution are not ignored")
    (testing "a NaN observation is an error, not a NaN weight"
      (let [m (gen [] (trace :y (d/normal 0 1)))
            nan (cm/choicemap :y ##NaN)]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"generate: a constrained value is NaN" (gfi/generate m [] nan k)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"update: a constrained value is NaN"
                              (gfi/update (gfi/simulate m [] k) nan k)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is NaN" (gfi/assess m [] nan)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is NaN"
                              (gfi/generate (gen [] (trace :v (d/mvnormal [0 0] (a/eye 2)))) []
                                            (cm/choicemap :v (a/array [0.0 ##NaN])) k)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is NaN" (gfi/generate (d/normal 0 1) [] (cm/->Value ##NaN) k)))))
    (testing "update and regenerate take one argdiff per argument"
      (let [tr (gfi/simulate linreg [[1.0]] k)]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"1 arguments but 0 argdiffs"
                              (gfi/update tr [[2.0]] [] cm/EMPTY k)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"1 arguments but 2 argdiffs"
                              (gfi/regenerate tr [[2.0]] [diff/unknown-change diff/unknown-change] (sel/select :slope) k)))))))

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
  (testing "the same transitions run N particles at once, under the batch tracer"
    (let [k (rng/key 7)
          {tr :trace} (dyn/run linreg :simulate [[1.0 2.0]] {:key k :batch-shape [5]})]
      (is (= [] (a/shape (tr :slope))) "inside the trace, values have their per-particle shapes")
      (is (= [5] (a/shape (batch/unwrap (tr :slope) [5]))) "with a leading batch axis underneath")
      (testing "each particle's score is its own assessed density"
        (doseq [[i c] (map-indexed vector (cm/unstack (cm/from-map (a/tree-map #(batch/unwrap % [5]) (cm/to-map (:choices tr)))) 5))]
          (is (close? (a/take (batch/unwrap (:score tr) [5]) i) (:weight (gfi/assess linreg [[1.0 2.0]] c))))))))
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

;; ---------------------------------------------------------------------------
;; Splices without an address; middleware and dispatch
;; ---------------------------------------------------------------------------

(deftest splices-without-an-address-share-the-namespace
  (let [tr (gfi/simulate shared-namespace [] (rng/key 1))]
    (is (every? #{[:mu] [:z] [:more] [:w] [:obs]} (cm/addresses (:choices tr))) "the callee's choices join the caller's")
    (is (= 2.5 ((:trace (gfi/generate shared-namespace [] (cm/choicemap :z 2.5) (rng/key 2))) :z)) "constraints reach them")
    (let [{t2 :trace} (gfi/update tr (cm/choicemap :z -1.0) (rng/key 3))]
      (is (= -1.0 (t2 :z)))
      (is (= (tr :mu) (t2 :mu))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"used twice"
                          (gfi/simulate (gen [] (trace :z (d/normal 0 1)) (splice shared-body 0.0)) [] (rng/key 4)))
        "addresses still may not collide")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"without an address needs a dynamic"
                          (gfi/simulate (gen [] (splice (d/normal 0 1))) [] (rng/key 5))))))

(def doubled-site
  "Middleware for simulate: every value drawn at :x is doubled (a different
   model, as middleware may be)."
  (fn [state p dist k]
    (let [[v s] ((dyn/site-transitions :simulate) state p dist k)]
      (if (= p [:x])
        [(a/* 2.0 v) (assoc-in s [:choices] (cm/set-choice (:choices s) p (a/* 2.0 v)))]
        [v s]))))

(deftest middleware-of-a-callee-spliced-without-an-address-applies-to-its-body
  (let [inner (dyn/with-transitions (gen [] (trace :x (d/normal 0 1))) {:simulate doubled-site})
        outer (gen [] (splice inner) (trace :z (d/normal 0 1)))
        k (rng/key 7)
        tr (gfi/simulate outer [] k)
        plain (gfi/simulate (gen [] (splice (gen [] (trace :x (d/normal 0 1)))) (trace :z (d/normal 0 1))) [] k)]
    (is (= (* 2.0 (plain :x)) (tr :x)) "the callee's middleware doubles its :x")
    (is (= (plain :z) (tr :z)) "and the caller's own sites are unaffected")))

(deftest middleware-replaces-a-transition-for-one-operation
  (let [m (gen [] (trace :x (d/normal 0 1)))
        k (rng/key 6)
        mw (dyn/with-transitions m {:simulate doubled-site})]
    (is (= (* 2.0 ((gfi/simulate m [] k) :x)) ((gfi/simulate mw [] k) :x)))
    (is (= ((:trace (gfi/generate m [] cm/EMPTY k)) :x) ((:trace (gfi/generate mw [] cm/EMPTY k)) :x))
        "the other operations keep the standard transitions")))

(def closed-form-assess
  "A dispatcher computing linreg's assess in closed form; everything else
   is left to the handler."
  (fn [gf op [xs] {:keys [constraints batch-shape]}]
    (when (and (= op :assess) (empty? batch-shape))
      (let [c #(cm/get-choice constraints %)
            slope (c :slope) intercept (c :intercept)
            lp (fn [v mu sd] (- (- (Math/log sd)) 0.9189385332046728 (* 0.5 (Math/pow (/ (- v mu) sd) 2))))]
        {:weight (+ (lp slope 0.0 10.0) (lp intercept 0.0 10.0)
                    (reduce + (map-indexed (fn [i x] (lp (c [:y i]) (+ (* slope x) intercept) 1.0)) xs)))
         :retval slope}))))

(laws/deflaws dispatched [[:linreg (dyn/with-dispatch linreg closed-form-assess) [[1.0 2.0 3.0]]]] 100
  {:dispatch-matches-handler laws/dispatch-matches-handler})

(deftest a-wrong-dispatcher-is-caught
  (let [wrong (fn [gf op args opts] (when (= op :assess) (update (closed-form-assess gf op args opts) :weight + 0.1)))
        cases (laws/cases [[:linreg (dyn/with-dispatch linreg wrong) [[1.0 2.0 3.0]]]])]
    (is (false? (:pass? (clojure.test.check/quick-check 20 (laws/dispatch-matches-handler cases)))))))

(deftest lazy-bodies-are-run-to-completion
  ;; trace calls inside a returned lazy sequence happen during the run
  (let [m (gen [] (map (fn [i] (trace [:x i] (d/normal 0 1))) (range 3)))
        tr (gfi/simulate m [] (rng/key 1))]
    (is (= [[:x 0] [:x 1] [:x 2]] (cm/addresses (:choices tr))))
    (is (close? (:score tr) (:weight (gfi/assess m [] (:choices tr))))))
  (testing "an infinite return value does not hang"
    (is (= [1.0 1.0] (vec (take 2 (:retval (gfi/simulate (gen [] (repeat 1.0)) [] (rng/key 3))))))))
  (testing "and laziness that escapes otherwise is loud"
    (let [escaped (atom nil)
          m (gen [] (reset! escaped (map (fn [i] (trace [:x i] (d/normal 0 1))) (range 3))) nil)]
      (gfi/simulate m [] (rng/key 2))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"after the generative function returned" (doall @escaped))))))

(deftest discarded-lazy-trace-calls-are-compile-errors
  (let [compile-error (fn [form] (try (eval form) nil
                                      (catch Throwable e (.getMessage ^Throwable (or (.getCause e) e)))))]
    (is (re-find #"lazy and its value is discarded"
                 (str (compile-error '(gen.dynamic/gen [xs] (map (fn [x] (trace [:y x] (gen.dist/normal x 1))) xs) nil)))))
    (is (re-find #"lazy and its value is discarded"
                 (str (compile-error '(gen.dynamic/gen [xs] (let [n 1] (for [x xs] (trace [:y x] (gen.dist/normal x n))) n))))))
    (testing "mapv, a lazy tail value and lazy calls without trace are fine"
      (is (nil? (compile-error '(gen.dynamic/gen [xs] (mapv (fn [x] (trace [:y x] (gen.dist/normal x 1))) xs) nil))))
      (is (nil? (compile-error '(gen.dynamic/gen [xs] (map (fn [x] (trace [:y x] (gen.dist/normal x 1))) xs)))))
      (is (nil? (compile-error '(gen.dynamic/gen [xs] (map inc xs) nil)))))))

(def returns-seven (gen [] (trace :x (d/normal 0 1)) 7.0))

(deftest a-different-callee-at-an-address-is-a-new-call
  ;; review 3: the old sub-trace was handed to the new generative function
  (let [m (gen [] (if (trace :c (d/bernoulli 0.5)) (splice :s returns-seven) (splice :s (d/normal 0 1))))
        tr (:trace (gfi/generate m [] (cm/choicemap :c true [:s :x] 0.5) (rng/key 1)))
        {t :trace w :weight disc :discard} (gfi/update tr (cm/choicemap :c false) (rng/key 2))]
    (is (not= 7.0 (t :s)) "the new distribution draws its own value")
    (is (close? w (a/- (a/- (:score t) (:score tr)) (gfi/project t (sel/select :s)))))
    (is (= (cm/choicemap :c true [:s :x] 0.5) disc))
    (is (= (:choices tr) (:choices (:trace (gfi/update t disc (rng/key 3))))) "and the update reverses")))

(deftest third-review-regressions
  (testing "B2: a splice whose dynamic callee changes keeps shared addresses"
    (let [ma (gen [] (trace :obs (d/normal 0 1)))
          mb (gen [] (trace :obs (d/normal 2 1)))
          m (gen [] (splice :s (if (trace :i (d/bernoulli 0.5)) ma mb)))
          tr (:trace (gfi/generate m [] (cm/choicemap :i true [:s :obs] 1.5) (rng/key 1)))
          {t :trace w :weight} (gfi/regenerate tr (sel/select :i) (rng/key 2))]
      (is (= 1.5 (t [:s :obs])) "the unselected observation keeps its value")
      (when-not (t :i)
        (is (close? w (- (dc/log-prob (d/normal 2 1) 1.5) (dc/log-prob (d/normal 0 1) 1.5)))))))
  (testing "B5: values of the wrong shape are errors everywhere"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape"
                          (gfi/generate (gen [] (splice :x (d/normal 0 1))) [] (cm/choicemap :x (a/array [1.0 2.0 3.0])) (rng/key 1))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape" (gfi/assess (d/normal 0 1) [] (cm/->Value (a/array [1.0 2.0])))))
    (let [m (gen [d] (trace :v (d/normal (a/zeros [d]) 1)))
          tr (gfi/simulate m [1] (rng/key 2))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape" (gfi/update tr [3] [diff/unknown-change] cm/EMPTY (rng/key 3)))
          "a retained value whose parameters change shape"))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape"
                          (dc/log-prob ((d/product d/normal d/bernoulli) [0 1] [0.3]) [(a/array [1.0 2.0 3.0]) true]))))
  (testing "B6: trace calls in lazy splice arguments are recorded"
    (let [m (gen [] (splice :m (comb/map observation) (vec (map #(trace [:x %] (d/normal 0 1)) (range 3)))))
          tr (gfi/simulate m [] (rng/key 4))]
      (is (= 3 (count (filter #(= :x (first %)) (cm/addresses (:choices tr))))))
      (is (close? (:score tr) (:weight (gfi/assess m [] (:choices tr))))))
    (is (re-find #"lazy and its value is discarded"
                 (str (try (eval '(gen.dynamic/gen [xs] (->> xs (map-indexed (fn [i x] (trace [:y i] (gen.dist/normal x 1))))) nil))
                           nil (catch Throwable e (.getMessage ^Throwable (or (.getCause e) e))))))))
  (testing "B7: an infinite lazy return value costs little"
    (let [m (gen [] (let [s (trace :s (d/normal 0 1))] (map #(* s %) (range))))
          t0 (System/nanoTime)]
      (dotimes [i 20] (gfi/simulate m [] (rng/key i)))
      (is (< (/ (- (System/nanoTime) t0) 1e6) 400.0) "under 20 ms a run"))))
