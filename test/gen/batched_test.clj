(ns gen.batched-test
  "Batched execution against scalar execution: each particle of a batched run
   is exactly the scalar trace of its choices, batched weights are the scalar
   weights particle by particle, the per-particle fallback carries on, and
   batched inference reaches the exact answers."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as a]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as imp]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.laws :refer [close?]]
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

(def observation (gen [mu] (trace :x (d/normal mu 1))))

(def hierarchical
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (mapv #(splice [:obs %] observation mu) (range n)))))

(def step
  (gen [t state]
    (let [s (trace :s (d/normal (a/* 0.8 state) 1))]
      (trace :obs (d/normal s 0.5))
      s)))

(def unfolded (comb/unfold step))

(def spliced
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))
          xs (splice :xs (comb/map observation) (vec (repeat n mu)))]
      (splice :path unfolded n mu)
      (splice :pick (comb/switch (gen [m] (trace :a (d/normal m 1))) (gen [m] (trace :b (d/gamma 2 1)))) 1 mu)
      xs)))

(def vector-latent
  (gen [] (let [z (trace :z (d/mvnormal [0.0 1.0] [[2.0 0.3] [0.3 1.0]]))] (trace :y (d/normal (a/sum z {:axis -1}) 1)))))

(def models
  [[:linreg linreg [[1.0 2.0 3.0]]]
   [:hierarchical hierarchical [3]]
   [:unfold unfolded [3 0.0]]
   [:spliced spliced [2]]
   [:vector-latent vector-latent []]])

(def gen-case (gen/tuple (gen/elements models) gen/nat (gen/choose 1 6)))

(defn- consistent? [tr]
  (close? (:score tr) (:weight (gfi/assess (:gen-fn tr) (:args tr) (:choices tr)))))

;; ---------------------------------------------------------------------------
;; Path equivalence
;; ---------------------------------------------------------------------------

(defspec every-particle-is-the-scalar-trace-of-its-choices 60
  (prop/for-all [[[_ gf args] seed n] gen-case]
    (let [ps (b/simulate gf args n (rng/key seed))
          trs (b/traces ps)]
      (and (b/batched? ps)
           (= n (count trs))
           (every? consistent? trs)
           (every? true? (map #(close? (:score %1) %2) trs (a/->clj (b/scores ps))))))))

(defspec batched-generate-weights-are-the-observations-densities 60
  (prop/for-all [[[_ gf args] seed n] gen-case]
    ;; observe the leaf choices of a scalar run that every particle visits
    (let [probe (gfi/simulate gf args (rng/key (inc seed)))
          obs-paths (filter #(#{:obs :y :x} (peek %)) (cm/addresses (:choices probe)))
          obs (sel/get-selected (:choices probe) (sel/from-paths obs-paths))
          {ps :particles ws :weights} (b/generate gf args obs n (rng/key seed))]
      (every? true? (map (fn [tr w] (and (consistent? tr)
                                         (close? w (gfi/project tr (sel/from-paths obs-paths)))
                                         (every? #(= (cm/get-choice obs %) (tr %)) obs-paths)))
                         (b/traces ps) (a/->clj ws))))))

(defspec batched-updates-are-the-scalar-updates 60
  (prop/for-all [[[_ gf args] seed n] gen-case]
    (let [ps (b/simulate gf args n (rng/key seed))
          probe (gfi/simulate gf args (rng/key (inc seed)))
          p (first (cm/addresses (:choices probe)))
          sigma (cm/choicemap p (cm/get-choice (:choices probe) p))
          {ps' :particles ws :weights ds :discards} (b/update ps args (vec (repeat (count args) diff/no-change)) sigma (rng/key 2))
          scalar (map #(gfi/update % sigma (rng/key 0)) (b/traces ps))]
      (and (every? true? (map #(close? (:weight %1) %2) scalar (a/->clj ws)))
           (= (map :discard scalar) ds)
           (= (map (comp :choices :trace) scalar) (map :choices (b/traces ps')))))))

(defspec batched-regenerate-weights-are-retained-density-ratios 60
  (prop/for-all [[[_ gf args] seed n] gen-case]
    (let [ps (b/simulate gf args n (rng/key seed))
          s (sel/from-paths (take 1 (cm/addresses (:choices (first (b/traces ps))))))
          {ps' :particles ws :weights} (b/regenerate ps args (vec (repeat (count args) diff/no-change)) s (rng/key 3))]
      (every? true?
              (map (fn [old new w]
                     (let [retained (sel/from-paths (remove (set (cm/addresses (sel/get-selected (:choices old) s)))
                                                            (cm/addresses (:choices new))))]
                       (and (consistent? new)
                            (close? w (a/- (gfi/project new retained) (gfi/project old retained))))))
                   (b/traces ps) (b/traces ps') (a/->clj ws))))))

(deftest batched-statistics
  (let [ps (b/simulate linreg [[1.0]] 20000 (rng/key 1))
        xs (a/->clj (b/choice ps :slope))
        m (/ (reduce + xs) 20000)
        v (/ (reduce + (map #(* (- % m) (- % m)) xs)) 19999)]
    (is (< (Math/abs m) (* 5 (/ 10 (Math/sqrt 20000)))))
    (is (< (Math/abs (- v 100.0)) 5.0))))

;; ---------------------------------------------------------------------------
;; Population operations and the fallback
;; ---------------------------------------------------------------------------

(deftest take-and-select
  (let [ps (b/simulate linreg [[1.0 2.0]] 5 (rng/key 4))
        slopes (a/->clj (b/choice ps :slope))]
    (is (= (mapv slopes [4 4 0]) (a/->clj (b/choice (b/take ps [4 4 0]) :slope))))
    (is (b/batched? (b/take ps [1 2])))
    (let [qs (b/simulate linreg [[1.0 2.0]] 5 (rng/key 5))
          mixed (b/select [true false true false true] ps qs)]
      (is (= (mapv (fn [keep? p q] (if keep? p q)) [true false true false true] slopes (a/->clj (b/choice qs :slope)))
             (a/->clj (b/choice mixed :slope))))
      (is (every? consistent? (b/traces mixed))))))

(def branching
  (gen []
    (if (trace :switch (d/bernoulli 0.4))
      (trace :a (d/normal 0 1))
      (trace :b (d/gamma 2 1)))))

(deftest models-that-branch-on-particles-fall-back
  (let [{ps :particles ws :weights} (b/generate branching [] cm/EMPTY 8 (rng/key 6))]
    (is (not (b/batched? ps)))
    (is (= 8 (count (b/traces ps))))
    (is (= (vec (repeat 8 0.0)) (a/->clj ws)))
    (testing "and later operations carry on per particle"
      (let [{ws :weights} (b/update ps [] [] (cm/choicemap :switch true) (rng/key 7))
            scalar (map #(:weight (gfi/update % (cm/choicemap :switch true) (rng/key 0))) (b/traces ps))]
        (is (= 8 (count (a/->clj ws))))
        (is (every? true? (map #(and (Double/isFinite %1) (Double/isFinite %2)) (a/->clj ws) scalar))))))
  (testing "a Switch whose index differs across particles falls back"
    (let [sw (gen [] (splice :s (comb/switch observation observation)
                             (if (trace :c (d/bernoulli 0.5)) 0 1) 0.0))]
      (is (not (b/batched? (b/simulate sw [] 6 (rng/key 8)))))))
  (testing "while a shared index stays batched"
    (is (b/batched? (b/simulate (gen [] (splice :s (comb/switch observation observation) 1 0.0)) [] 6 (rng/key 9))))))

;; ---------------------------------------------------------------------------
;; Batched inference
;; ---------------------------------------------------------------------------

(def conjugate (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x 1)))))
(def y 1.3)
(def evidence (dc/log-prob (d/normal 0 (Math/sqrt 2)) y))

(deftest batched-importance-sampling
  (let [{:keys [particles log-ml-estimate log-normalized-weights]}
        (imp/importance-sampling conjugate [] (cm/choicemap :y y) 20000 {:batched? true :key (rng/key 10)})]
    (is (b/batched? particles))
    (is (= 20000 (count log-normalized-weights)))
    (is (< (Math/abs (- log-ml-estimate evidence)) 0.03))
    (let [xs (a/->clj (b/choice particles :x))
          mean (reduce + (map #(* (Math/exp %1) %2) log-normalized-weights xs))]
      (is (< (Math/abs (- mean 0.65)) 0.03) "the weighted posterior mean")))
  (testing "resampling one trace"
    (is (number? ((:trace (imp/importance-resampling conjugate [] (cm/choicemap :y y) 100 {:batched? true :key (rng/key 11)})) :x)))))

(deftest a-batched-particle-filter-matches-the-kalman-filter
  (let [ys [0.3 1.1 0.4 -0.6 0.2]
        obs (fn [t] (cm/choicemap [t :obs] (ys t)))
        ;; Kalman filter for step: x_t ~ N(0.8 x_{t-1}, 1), obs ~ N(x_t, 0.5)
        exact (first (reduce (fn [[lml m p] yv]
                               (let [m- (* 0.8 m) p- (+ (* 0.64 p) 1.0) s (+ p- 0.25) k (/ p- s)]
                                 [(+ lml (dc/log-prob (d/normal m- (Math/sqrt s)) yv)) (+ m- (* k (- yv m-))) (* (- 1 k) p-)]))
                             [0.0 0.0 0.0] ys))
        run (fn [n key]
              (let [ks (rng/split key 10)
                    init (pf/initialize unfolded [1 0.0] (obs 0) n {:batched? true :key (ks 0)})]
                (reduce (fn [s t] (-> s (pf/maybe-resample {:key (ks (dec (* 2 t)))})
                                      (pf/step [(inc t) 0.0] [diff/unknown-change diff/no-change] (obs t) {:key (ks (* 2 t))})))
                        init (range 1 (count ys)))))
        state (run 40000 (rng/key 12))]
    (is (b/batched? (:particles state)) "the filter stays batched")
    ;; the estimate's sd is about sqrt(12.6/n) here (measured on runs of 20
    ;; particles): four of them at n = 40000 is 0.07
    (is (< (Math/abs (- (pf/log-ml-estimate state) exact)) 0.07))
    (is (= 40000 (count (:log-weights state))))
    (testing "unbiased in exp space (law 20)"
      (let [runs 400
            ratios (map #(Math/exp (- (pf/log-ml-estimate (run 20 %)) exact)) (rng/split (rng/key 14) runs))
            mean (/ (reduce + ratios) runs)
            sd (Math/sqrt (/ (reduce + (map #(* (- % mean) (- % mean)) ratios)) (dec runs)))]
        (is (< (Math/abs (- mean 1.0)) (* 5 (/ sd (Math/sqrt runs)))))))))

(deftest parallel-mh-chains
  (let [n 4000
        start (fn [xs] (b/take (:particles (b/generate conjugate [] (cm/choicemap :y y) 1 (rng/key 0))) (repeat n 0)))
        moments (fn [ps] (let [xs (a/->clj (b/choice ps :x)) m (/ (reduce + xs) n)]
                           [m (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))]))
        chains (reduce (fn [ps k] (:particles (mh/mh-particles ps (sel/select :x) k)))
                       (start nil) (rng/split (rng/key 13) 60))
        [m v] (moments chains)]
    (is (b/batched? chains))
    (is (< (Math/abs (- m 0.65)) (* 5 (Math/sqrt (/ 0.5 n)))))
    (is (< (Math/abs (- v 0.5)) 0.06))))

;; ---------------------------------------------------------------------------
;; Regressions from the M1 review (docs/research/review-2026-10-18.md)
;; ---------------------------------------------------------------------------

(defn- batched-matches-scalar?
  "Do batched importance weights equal the scalar generate weights of the
   same particles? (Per particle, from the batched particles' own choices.)"
  [model args obs n key]
  (let [{ps :particles ws :weights} (b/generate model args obs n key)]
    (every? true? (map (fn [tr w] (close? w (gfi/project tr (sel/from-choicemap obs))))
                       (b/traces ps) (a/->clj ws)))))

(deftest host-predicates-on-particles-fall-back
  ;; review 1: (= k 0) on a batch was silently false for every particle
  (let [m (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))
                        mu (if (= k 0) 0.0 3.0)]
                    (trace :y (d/normal mu 1))))
        obs (cm/choicemap :y 0.0)
        exact (Math/log (+ (* 0.5 (Math/exp (dc/log-prob (d/normal 0 1) 0.0))) (* 0.5 (Math/exp (dc/log-prob (d/normal 3 1) 0.0)))))
        {:keys [particles log-ml-estimate]} (imp/importance-sampling m [] obs 4000 {:batched? true :key (rng/key 30)})]
    (is (not (b/batched? particles)) "the batch falls back to one particle at a time")
    (is (< (Math/abs (- log-ml-estimate exact)) 0.05))
    (testing "and case too"
      (let [m (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))]
                        (trace :y (d/normal (case k 0 0.0 1 3.0) 1))))]
        (is (batched-matches-scalar? m [] obs 6 (rng/key 31)))))))

(deftest vector-sites-are-per-particle
  ;; review 2: an [N] per-particle parameter met a [3] value element by element
  (let [ys (a/array [0.5 1.0 1.5])
        ;; three i.i.d. observations: vector parameters (Gen.jl's broadcasted_normal)
        m (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :ys (d/normal (a/+ mu (a/zeros [3])) 1))))]
    (doseq [n [3 4]]
      (let [{ps :particles ws :weights} (b/generate m [] (cm/choicemap :ys ys) n (rng/key 32))]
        (is (b/batched? ps))
        (is (every? true? (map (fn [mu w] (close? w (a/sum (dc/log-prob (d/normal mu 1) ys))))
                               (a/->clj (b/choice ps :mu)) (a/->clj ws))))))
    (testing "a vector site's draws have the batch axis in front"
      (let [v (gen [] (trace :z (d/normal (a/zeros [3]) 1)))]
        (is (= [5 3] (a/shape (b/choice (b/simulate v [] 5 (rng/key 33)) :z))))
        (is (batched-matches-scalar? v [] (cm/choicemap :z (a/array [0.1 0.2 0.3])) 3 (rng/key 34)))))
    (testing "a per-particle scalar meets a shared vector in model arithmetic"
      (let [m (gen [] (let [x (trace :x (d/normal 0 1))]
                        (trace :y (d/normal (a/sum (a/* x (a/array [1.0 2.0 3.0]))) 1))))]
        (is (batched-matches-scalar? m [] (cm/choicemap :y 0.7) 3 (rng/key 35)))
        (is (every? true? (map (fn [tr] (close? (:score tr) (:weight (gfi/assess m [] (:choices tr)))))
                               (b/traces (:particles (b/generate m [] (cm/choicemap :y 0.7) 3 (rng/key 35)))))))))))

(deftest switches-take-and-spliced-distributions-in-batches
  (let [sw (gen [] (splice :s (comb/switch observation observation) (trace :i (d/categorical [0.5 0.5])) 0.0))]
    (testing "review 11: a Switch whose index is a choice, N = 1"
      (is (= 1 (count (b/traces (b/simulate sw [] 1 (rng/key 36)))))))
    (testing "review 12: take of a branching model's particles"
      (let [ps (:particles (b/generate branching [] cm/EMPTY 1 (rng/key 37)))]
        (is (= 3 (count (b/traces (b/take ps [0 0 0])))))))
    (testing "review 13: a spliced distribution runs batched"
      (let [m (gen [mu] (splice :x (d/normal mu 1)) (trace :y (d/normal 0 1)))
            ps (b/simulate m [0.5] 4 (rng/key 38))]
        (is (b/batched? ps))
        (is (every? #(close? (:score %) (:weight (gfi/assess m [0.5] (:choices %)))) (b/traces ps)))))))

;; ---------------------------------------------------------------------------
;; Regressions from the third review (docs/research/review-2026-10-19.md)
;; ---------------------------------------------------------------------------

(deftest host-predicates-in-either-order-fall-back
  ;; B1: (= 0 k), condp, case with integer keys and a default, vector lookups
  (let [obs (cm/choicemap :y -3.0)
        exact (Math/log (+ (* 0.5 (Math/exp (dc/log-prob (d/normal -3 1) -3.0))) (* 0.5 (Math/exp (dc/log-prob (d/normal 3 1) -3.0)))))]
    ;; =, not= and condp reach the batch through = even in helper functions;
    ;; case (and if) only inside gen bodies, which guard them
    (doseq [[label mu-of] [[:=-on-the-right (fn [k] (if (= 0 k) -3.0 3.0))]
                           [:not= (fn [k] (if (not= 0 k) 3.0 -3.0))]
                           [:condp (fn [k] (condp = k 0 -3.0 3.0))]
                           [:vector-lookup (fn [k] ([-3.0 3.0] k))]]]
      (let [m (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))] (trace :y (d/normal (mu-of k) 1))))
            {:keys [log-ml-estimate]} (imp/importance-sampling m [] obs 4000 {:batched? true :key (rng/key 40)})]
        (is (< (Math/abs (- log-ml-estimate exact)) 0.1) (str label))))
    (let [m (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))] (trace :y (d/normal (case k 0 -3.0 3.0) 1))))
          {:keys [log-ml-estimate]} (imp/importance-sampling m [] obs 4000 {:batched? true :key (rng/key 40)})]
      (is (< (Math/abs (- log-ml-estimate exact)) 0.1) "case with a default, in a gen body"))))

(deftest batched-runs-validate-parameters
  ;; B3: a batched run threw nothing where a scalar run throws
  (let [m (gen [] (let [x (trace :x (d/uniform 0 1))] (trace :k (d/categorical (a/stack [x x])))))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"probabilities" (gfi/simulate m [] (rng/key 1))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"probabilities" (b/simulate m [] 4 (rng/key 1))))))

(deftest more-batched-forms
  (testing "B9: scans and arg-reductions of a per-particle scalar"
    (let [m (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (a/sum (a/cumsum x)) 1))))]
      (is (b/batched? (b/simulate m [] 3 (rng/key 41))))))
  (testing "B10: an Unfold whose length is a choice falls back"
    (let [m (gen [] (splice :u unfolded (trace :n (d/uniform-discrete 1 3)) 0.0))]
      (is (every? #(close? (:score %) (:weight (gfi/assess m [] (:choices %)))) (b/traces (b/simulate m [] 4 (rng/key 42)))))))
  (testing "B11: a collection of a distribution's particles"
    (is (= 3 (count (b/traces (b/simulate (d/normal 0 1) [] 3 (rng/key 43)))))))
  (testing "a per-particle fill value stays batched"
    (let [m (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (a/full [2] x) 1))))]
      (is (b/batched? (b/simulate m [] 3 (rng/key 44))))))
  (testing "B8: resampled particles are the chosen ones, exactly"
    (let [ps (b/simulate linreg [[1.0 2.0]] 5 (rng/key 45))
          taken (b/take ps [3 3 1])]
      (is (= (mapv (vec (b/traces ps)) [3 3 1]) (b/traces taken)))
      (is (every? #(close? (:score %) (:weight (gfi/assess linreg [[1.0 2.0]] (:choices %)))) (b/traces taken))))))

;; ---------------------------------------------------------------------------
;; Branches the tracer alone cannot see
;; ---------------------------------------------------------------------------

(defn- scores-exact?
  "Are the collection's scores (batched ones as the batched run made them)
   the assessed densities of its particles' choices?"
  [gf args ps]
  (every? true? (map #(close? %1 (:weight (gfi/assess gf args (:choices %2)))) (a/->clj (b/scores ps)) (b/traces ps))))

(defn- falls-back-exactly?
  "Does gf fall back in batched importance sampling, giving the scalar
   estimate exactly (same key)? Spot checks are off: the guard under test
   must cause the fallback by itself."
  [gf args obs]
  (let [s (imp/importance-sampling gf args obs 200 {:key (rng/key 60)})
        r (binding [b/*spot-checks* 0]
            (imp/importance-sampling gf args obs 200 {:key (rng/key 60) :batched? true}))]
    (and (not (b/batched? (:particles r)))
         (= (:log-ml-estimate s) (:log-ml-estimate r)))))

(deftest clojure-core-tests-in-gen-bodies-branch-per-particle
  ;; second acceptance test B1: these test truthiness inside clojure.core,
  ;; where a batch is one true object; the gen macro substitutes guarded
  ;; versions (gen.dynamic.core)
  (let [obs (cm/choicemap :y 4.0)]
    (doseq [[label m] [[:if-not (gen [] (let [b (trace :b (d/bernoulli 0.9))] (trace :y (d/normal (if-not b 5.0 0.0) 1))))]
                       [:not (gen [] (let [b (trace :b (d/bernoulli 0.9))] (trace :y (d/normal (if (not b) 5.0 0.0) 1))))]
                       [:boolean (gen [] (let [b (trace :b (d/bernoulli 0.9))] (trace :y (d/normal (if (boolean b) 0.0 5.0) 1))))]
                       [:true? (gen [] (let [b (trace :b (d/bernoulli 0.9))] (trace :y (d/normal (if (true? b) 0.0 5.0) 1))))]
                       [:every? (gen [] (let [b (trace :b (d/bernoulli 0.9))] (trace :y (d/normal (if (every? identity [b]) 0.0 5.0) 1))))]
                       [:filter (gen [] (let [b (trace :b (d/bernoulli 0.9))] (trace :y (d/normal (if (seq (filter identity [b])) 0.0 5.0) 1))))]
                       [:some (gen [] (let [b (trace :b (d/bernoulli 0.9))] (trace :y (d/normal (if (some #(when % :yes) [b]) 0.0 5.0) 1))))]
                       [:complement (gen [] (let [b (trace :b (d/bernoulli 0.9))] (trace :y (d/normal (if ((complement identity) b) 0.0 5.0) 1))))]
                       [:get (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))] (trace :y (d/normal (get [0.0 5.0] k 2.0) 1))))]
                       [:contains? (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))] (trace :y (d/normal (if (contains? [0.0] k) 0.0 5.0) 1))))]]]
      (is (falls-back-exactly? m [] obs) (str label)))
    (testing "and ordinary values behave as clojure.core's"
      (let [m (gen [xs] (let [{:keys [a]} {:a 2.0}
                              ys (filterv #(> % 1) xs)]
                          (trace :y (d/normal (+ a (count ys) (if (every? pos? xs) 1 0) (get xs 0)) 1))))]
        (is (close? (dc/log-prob (d/normal (+ 2.0 2 1 1) 1) 4.0)
                    (:weight (gfi/generate m [[1 2 3]] (cm/choicemap :y 4.0) (rng/key 61)))))
        (is (b/batched? (b/simulate m [[1 2 3]] 4 (rng/key 62))) "and stay batched")))
    (testing "interop member names are left alone"
      (let [m (gen [] (let [x (trace :x (d/normal 0 1))] (.get ^java.util.List [x] 0)))]
        (is (number? (:retval (gfi/simulate m [] (rng/key 63)))))))))

(defn- drift
  "A raw if on an array comparison, in a helper outside gen: under the
   tracer the comparison is a batch, which is truthy, so every particle would
   take the first branch."
  [x]
  (if (a/> x 0) 1.0 -1.0))

(def hidden-branch (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (drift x) 1)))))

(deftest a-comparison-nothing-reads-falls-back
  ;; second acceptance test B3: a comparison that only a raw if consumed
  (let [ps (b/simulate hidden-branch [] 50 (rng/key 50))]
    (is (not (b/batched? ps)))
    (is (re-find #"comparison was never read" (:message (b/fallback-reason ps))))
    (is (scores-exact? hidden-branch [] ps)))
  (let [rare (fn [x] (if (a/< x 2.0) 1.0 0.1))
        m (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x (rare x)))))]
    (is (falls-back-exactly? m [] (cm/choicemap :y 2.5)) "however rare the other branch"))
  (testing "while comparisons that are read, or handed back, stay batched"
    (is (b/batched? (b/simulate (gen [] (let [x (trace :x (d/normal 0 1))] (a/> x 0))) [] 8 (rng/key 51))))
    (is (b/batched? (b/simulate (gen [] (let [x (trace :x (d/normal 0 1))]
                                          (trace :y (d/normal (a/where (a/> x 0) 1.0 -1.0) 1))))
                                [] 8 (rng/key 52))))))

(defn- level
  "A vector lookup keyed by a choice, in a helper outside gen: a batched key
   is no integer to a vector, which silently gives the default."
  [k]
  (get [-3.0 3.0] k 0.0))

(def hidden-lookup (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))] (trace :y (d/normal (level k) 1)))))

(dyn/defn-guarded guarded-level
  "level, defined with a body's guards."
  [k]
  (get [-3.0 3.0] k 0.0))

(deftest helpers-defined-with-guards-branch-per-particle
  (let [m (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))] (trace :y (d/normal (guarded-level k) 1))))]
    (is (falls-back-exactly? m [] (cm/choicemap :y 0.5)))
    (is (= 3.0 (guarded-level 1)) "and are ordinary functions otherwise"))
  (let [sgn (dyn/guarded-fn [x] (if (a/> x 0) 1.0 -1.0))
        m (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (sgn x) 1))))]
    (is (falls-back-exactly? m [] (cm/choicemap :y 0.5)))))

(deftest values-the-trace-holds-are-handed-back
  ;; third acceptance test: needless fallbacks
  (let [callee (gen [] (let [x (trace :x (d/normal 0 1))] {:x x :big? (a/> x 0)}))
        m (gen [] (let [{:keys [x]} (splice :c callee)] (trace :y (d/normal x 1))))]
    (is (b/batched? (b/simulate m [] 8 (rng/key 70))) "a sub-trace's return value"))
  (let [m (gen [] (let [x (trace :x (d/normal 0 1))] (assert (number? x)) (trace :y (d/normal x 1))))]
    (is (b/batched? (b/simulate m [] 8 (rng/key 71))) "type predicates answer from the dtype")))

(deftest spot-checks-catch-what-the-tracer-cannot-see
  (testing "simulate and generate fall back, and say why"
    (let [ps (b/simulate hidden-lookup [] 50 (rng/key 50))
          {gs :particles} (b/generate hidden-lookup [] (cm/choicemap :y 0.5) 50 (rng/key 51))]
      (is (not (b/batched? ps)))
      (is (not (b/batched? gs)))
      (is (re-find #"disagrees with the scalar handler" (:message (b/fallback-reason ps))))
      (is (scores-exact? hidden-lookup [] ps))))
  (testing "without spot checks the hole shows: the documented limit"
    (binding [b/*spot-checks* 0]
      (let [ps (b/simulate hidden-lookup [] 50 (rng/key 50))]
        (is (b/batched? ps))
        (is (not (scores-exact? hidden-lookup [] ps))))))
  (testing "batched importance sampling then reaches the scalar answer"
    (let [obs (cm/choicemap :y 0.5)
          exact (Math/log (+ (* 0.5 (Math/exp (dc/log-prob (d/normal 3 1) 0.5)))
                             (* 0.5 (Math/exp (dc/log-prob (d/normal -3 1) 0.5)))))
          {:keys [log-ml-estimate]} (imp/importance-sampling hidden-lookup [] obs 4000 {:batched? true :key (rng/key 52)})]
      (is (< (Math/abs (- log-ml-estimate exact)) 0.05))))
  (testing "a branch that first runs in an early update falls back there"
    (let [m (gen [t] (let [k (trace :k (d/categorical [0.5 0.5]))]
                       (when (pos? t) (trace :y (d/normal (level k) 1)))
                       k))
          ps (b/simulate m [0] 20 (rng/key 53))
          obs (cm/choicemap :y 0.5)
          {ps' :particles ws :weights} (b/update ps [1] [diff/unknown-change] obs (rng/key 54))]
      (is (b/batched? ps))
      (is (not (b/batched? ps')))
      (is (every? true? (map (fn [tr w] (close? w (:weight (gfi/update tr [1] [diff/unknown-change] obs (rng/key 0)))))
                             (b/traces ps) (a/->clj ws))))))
  (testing "one found after unchecked updates is a loud error: their results may be wrong too"
    (let [m (gen [t] (let [k (trace :k (d/categorical [0.5 0.5]))]
                       (when (> t 2) (trace :y (d/normal (level k) 1)))
                       k))
          step (fn [ps t] (:particles (b/update ps [t] [diff/unknown-change] cm/EMPTY (rng/key t))))
          ps (reduce step (b/simulate m [0] 20 (rng/key 55)) [1 2 3])]
      ;; updates 1 and 2 are checked, 3 is not (the branch starts there), 4 is
      (is (b/batched? ps))
      (is (:gen.batched/diverged (ex-data (try (step ps 4) nil (catch clojure.lang.ExceptionInfo e e)))))))
  (testing "so is a fallback after unchecked updates, which starts from their results"
    (let [m (gen [t] (let [k (trace :k (d/categorical [0.5 0.5]))]
                       (when (> t 2) (trace :y (d/normal (level k) 1)))
                       (when (> t 3) (when (trace :b (d/bernoulli 0.5)) (trace :z (d/normal 0 1))))
                       k))
          step (fn [ps t] (:particles (b/update ps [t] [diff/unknown-change] cm/EMPTY (rng/key t))))
          ps (reduce step (b/simulate m [0] 20 (rng/key 55)) [1 2 3])]
      ;; update 3 is unchecked and wrong; update 4 falls back on the body's if
      (is (b/batched? ps))
      (is (:gen.batched/diverged (ex-data (try (step ps 4) nil (catch clojure.lang.ExceptionInfo e e)))))))
  (testing "a batched-branch fallback records the error's data"
    (is (:gen.array/batched-branch (:data (b/fallback-reason (b/simulate branching [] 8 (rng/key 56)))))))
  (testing "well-behaved models stay batched, also when every particle is checked"
    (binding [b/*spot-checks* :all]
      (let [ps (b/simulate linreg [[1.0 2.0]] 16 (rng/key 57))]
        (is (b/batched? ps))
        (is (nil? (b/fallback-reason ps)))))))
