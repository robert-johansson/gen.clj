(ns gen.batched-test
  "Batched execution against scalar execution: each particle of a batched run
   is exactly the scalar trace of its choices, batched weights are the scalar
   weights particle by particle (with per-particle constraints too, each
   particle's own), the per-particle fallback carries on, and batched
   inference reaches the exact answers."
  (:refer-clojure :exclude [+ - *])
  (:require [gen.test-support :as ts]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array :refer [+ - *]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.inference.proposal :as proposal]
            [gen.inference.vi :as vi]
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
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      slope)))

(def observation (gen [mu] (trace :x (d/normal mu 1))))

(def hierarchical
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (mapv #(splice [:obs %] observation mu) (range n)))))

(def step
  (gen [t state]
    (let [s (trace :s (d/normal (* 0.8 state) 1))]
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
  (gen [] (let [z (trace :z (d/mvnormal [0.0 1.0] [[2.0 0.3] [0.3 1.0]]))] (trace :y (d/normal (array/sum z {:axis -1}) 1)))))

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
           (every? true? (map #(close? (:score %1) %2) trs (array/->clj (b/scores ps))))))))

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
                         (b/traces ps) (array/->clj ws))))))

(defspec batched-updates-are-the-scalar-updates 60
  (prop/for-all [[[_ gf args] seed n] gen-case]
    (let [ps (b/simulate gf args n (rng/key seed))
          probe (gfi/simulate gf args (rng/key (inc seed)))
          p (first (cm/addresses (:choices probe)))
          sigma (cm/choicemap p (cm/get-choice (:choices probe) p))
          {ps' :particles ws :weights ds :discards} (b/update ps args (vec (repeat (count args) diff/no-change)) sigma (rng/key 2))
          scalar (map #(gfi/update % sigma (rng/key 0)) (b/traces ps))]
      (and (every? true? (map #(close? (:weight %1) %2) scalar (array/->clj ws)))
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
                            (close? w (- (gfi/project new retained) (gfi/project old retained))))))
                   (b/traces ps) (b/traces ps') (array/->clj ws))))))

(deftest batched-statistics
  (let [ps (b/simulate linreg [[1.0]] 20000 (rng/key 1))
        xs (array/->clj (b/choice ps :slope))
        m (/ (reduce + xs) 20000)
        v (/ (reduce + (map #(* (- % m) (- % m)) xs)) 19999)]
    (is (< (Math/abs m) (* 5 (/ 10 (Math/sqrt 20000)))))
    (is (< (Math/abs (- v 100.0)) 5.0))))

;; ---------------------------------------------------------------------------
;; Population operations and the fallback
;; ---------------------------------------------------------------------------

(deftest take-and-select
  (let [ps (b/simulate linreg [[1.0 2.0]] 5 (rng/key 4))
        slopes (array/->clj (b/choice ps :slope))]
    (is (= (mapv slopes [4 4 0]) (array/->clj (b/choice (b/take ps [4 4 0]) :slope))))
    (is (b/batched? (b/take ps [1 2])))
    (let [qs (b/simulate linreg [[1.0 2.0]] 5 (rng/key 5))
          mixed (b/select [true false true false true] ps qs)]
      (is (= (mapv (fn [keep? p q] (if keep? p q)) [true false true false true] slopes (array/->clj (b/choice qs :slope)))
             (array/->clj (b/choice mixed :slope))))
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
    (is (= (vec (repeat 8 0.0)) (array/->clj ws)))
    (testing "and later operations carry on per particle"
      (let [{ws :weights} (b/update ps [] [] (cm/choicemap :switch true) (rng/key 7))
            scalar (map #(:weight (gfi/update % (cm/choicemap :switch true) (rng/key 0))) (b/traces ps))]
        (is (= 8 (count (array/->clj ws))))
        (is (every? true? (map #(and (Double/isFinite %1) (Double/isFinite %2)) (array/->clj ws) (map ts/number scalar)))))))
  (testing "a Switch whose index differs across particles stays batched, by partition"
    (let [sw (gen [] (splice :s (comb/switch observation observation) (trace :c (d/categorical [0.5 0.5])) 0.0))
          ps (b/simulate sw [] 6 (rng/key 8))]
      (is (b/batched? ps))
      (is (nil? (b/fallback-reason ps)))))
  (testing "as does a shared index"
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
    (let [xs (array/->clj (b/choice particles :x))
          mean (reduce + (map #(* (Math/exp %1) %2) log-normalized-weights xs))]
      (is (< (Math/abs (- mean 0.65)) 0.03) "the weighted posterior mean")))
  (testing "resampling one trace"
    (is (number? (ts/host ((:trace (imp/importance-resampling conjugate [] (cm/choicemap :y y) 100 {:batched? true :key (rng/key 11)})) :x))))))

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
        moments (fn [ps] (let [xs (array/->clj (b/choice ps :x)) m (/ (reduce + xs) n)]
                           [m (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))]))
        chains (reduce (fn [ps k] (:particles (mh/mh-particles ps (sel/select :x) k)))
                       (start nil) (rng/split (rng/key 13) 60))
        [m v] (moments chains)]
    (is (b/batched? chains))
    (is (< (Math/abs (- m 0.65)) (* 5 (Math/sqrt (/ 0.5 n)))))
    (is (< (Math/abs (- v 0.5)) 0.06))))

;; ---------------------------------------------------------------------------
;; Per-particle constraints
;; ---------------------------------------------------------------------------

(defn- runs-at
  "The choices at paths of n scalar runs of gf on args, keys from seed: n
   particles' own constraints."
  [gf args paths n seed]
  (mapv #(sel/get-selected (:choices (gfi/simulate gf args %)) (sel/from-paths paths)) (rng/split (rng/key seed) n)))

(defn- holds?
  "Does the trace tr hold the choices c?"
  [tr c]
  (every? #(close? (cm/get-choice c %) (tr %)) (cm/addresses c)))

(defspec per-particle-generates-are-each-particle-s-generate 60
  ;; particle i constrained by its own choices at every other address and
  ;; by one shared choice: it holds them, and its weight is its scalar
  ;; generate's, their density given its other choices
  (prop/for-all [[[_ gf args] seed n] gen-case]
    (let [addrs (cm/addresses (:choices (gfi/simulate gf args (rng/key (inc seed)))))
          own-paths (take-nth 2 addrs)
          shared-paths (take 1 (remove (set own-paths) addrs))
          own (runs-at gf args own-paths n seed)
          shared (first (runs-at gf args shared-paths 1 (+ seed 1000)))
          given (sel/from-paths (concat own-paths shared-paths))
          {ps :particles ws :weights} (b/generate gf args shared n (rng/key seed) {:per-particle own})]
      (and (b/batched? ps)
           (every? true? (map (fn [tr w c] (and (consistent? tr) (holds? tr (cm/merge shared c)) (close? w (gfi/project tr given))))
                              (b/traces ps) (array/->clj ws) own))))))

(defspec per-particle-updates-are-each-particle-s-update 60
  ;; particle i updated with its own value at one address and a shared one
  ;; at another: the weights, discards and choices of its scalar update
  (prop/for-all [[[_ gf args] seed n] gen-case]
    (let [ps (b/simulate gf args n (rng/key seed))
          addrs (cm/addresses (:choices (gfi/simulate gf args (rng/key (inc seed)))))
          own (runs-at gf args (take 1 addrs) n (+ seed 7))
          shared (first (runs-at gf args (take-last 1 (rest addrs)) 1 (+ seed 11)))
          argdiffs (vec (repeat (count args) diff/no-change))
          {ps' :particles ws :weights ds :discards} (b/update ps args argdiffs shared (rng/key 2) {:per-particle own})
          scalar (map #(gfi/update %1 (cm/merge shared %2) (rng/key 0)) (b/traces ps) own)]
      (and (b/batched? ps')
           (every? true? (map #(close? (:weight %1) %2) scalar (array/->clj ws)))
           (= (map (comp ts/host :discard) scalar) (map ts/host ds))
           (= (map (comp ts/host :choices :trace) scalar) (map (comp ts/host :choices) (b/traces ps')))))))

(deftest per-particle-is-said-so-never-read-from-a-shape
  ;; three particles and a site of shape [3]: a shared [3] value is one
  ;; value for every particle, per-particle ones are one each
  (let [m (gen [] (let [z (trace :z (d/normal (array/zeros [3]) 1))] (trace :y (d/normal (array/sum z) 1))))
        v (array/array [0.1 0.2 0.3])
        vs [(array/array [1.0 0.0 0.0]) (array/array [0.0 2.0 0.0]) (array/array [0.0 0.0 3.0])]
        shared (:particles (b/generate m [] (cm/choicemap :z v) 3 (rng/key 80)))
        own (:particles (b/generate m [] cm/EMPTY 3 (rng/key 80) {:per-particle (mapv #(cm/choicemap :z %) vs)}))]
    (is (b/batched? shared))
    (is (b/batched? own))
    (is (close? (array/stack [v v v]) (b/choice shared :z)))
    (is (close? (array/stack vs) (b/choice own :z)))))

(deftest per-particle-constraints-that-cannot-share-a-run-fall-back
  (testing "choice maps of different addresses"
    (let [own [(cm/choicemap :slope 1.0) (cm/choicemap :intercept 2.0) (cm/choicemap :slope -1.0)]
          obs (cm/choicemap [:y 0] 0.5)
          {ps :particles ws :weights} (b/generate linreg [[1.0 2.0]] obs 3 (rng/key 81) {:per-particle own})]
      (is (not (b/batched? ps)))
      (is (re-find #"do not stack" (:message (b/fallback-reason ps))))
      (is (every? true? (map (fn [tr w c] (and (holds? tr (cm/merge obs c))
                                               (close? w (gfi/project tr (sel/from-choicemap (cm/merge obs c))))))
                             (b/traces ps) (array/->clj ws) own)))))
  (testing "a model that branches on a particle's value"
    (let [{ps :particles ws :weights} (b/generate branching [] cm/EMPTY 3 (rng/key 82)
                                                  {:per-particle (mapv #(cm/choicemap :switch %) [true false true])})]
      (is (not (b/batched? ps)))
      (is (= [true false true] (mapv #(ts/host (% :switch)) (b/traces ps))))
      (is (close? (array/log (array/array [0.4 0.6 0.4])) ws)))))

(deftest per-particle-constraints-are-checked
  (let [own (mapv #(cm/choicemap :slope %) [1.0 2.0])
        ps (b/simulate linreg [[1.0]] 2 (rng/key 83))
        fails (fn [re f] (is (thrown-with-msg? clojure.lang.ExceptionInfo re (f))))]
    (fails #"one choice map for each of the 3 particles" #(b/generate linreg [[1.0]] cm/EMPTY 3 (rng/key 1) {:per-particle own}))
    (fails #"one choice map for each of the 2 particles" #(b/generate linreg [[1.0]] cm/EMPTY 2 (rng/key 1) {:per-particle (first own)}))
    (fails #"must be a choice map" #(b/generate linreg [[1.0]] cm/EMPTY 2 (rng/key 1) {:per-particle [{:slope 1.0} {:slope 2.0}]}))
    (fails #"constraints of particle 0 overlap the shared ones"
           #(b/generate linreg [[1.0]] (cm/choicemap :slope 0.0) 2 (rng/key 1) {:per-particle own}))
    (fails #"one choice map for each of the 2 particles" #(b/update ps [[1.0]] [diff/no-change] cm/EMPTY (rng/key 1) {:per-particle [cm/EMPTY]}))
    (fails #"constraints of particle 0 overlap the shared ones"
           #(b/update ps [[1.0]] [diff/no-change] (cm/choicemap :slope 0.0) (rng/key 1) {:per-particle own}))))

(deftest projections-are-each-particle-s
  (let [each (fn [ps s] (array/stack (mapv #(gfi/project % s) (b/traces ps))))]
    (testing "of a batched collection, under the batch tracer"
      (let [ps (b/simulate linreg [[1.0 2.0]] 5 (rng/key 84))
            s (sel/select :slope [:y 0])]
        (is (b/batched? ps))
        (is (close? (each ps s) (b/project ps s)))))
    (testing "per particle when that fails: an exact proposal's projection reads a support that differs across them"
      (let [m (enum/with-exact-proposal (gen [] (let [n (trace :n (d/uniform-discrete 1 3))]
                                                  (trace :k (d/uniform-discrete 0 n))))
                                        [:k])
            ps (b/simulate m [] 6 (rng/key 85))]
        (is (b/batched? ps))
        (is (close? (each ps (sel/select :k)) (b/project ps (sel/select :k))))))
    (testing "and of particles that fell back"
      (let [ps (b/simulate branching [] 5 (rng/key 86))]
        (is (not (b/batched? ps)))
        (is (close? (each ps (sel/select :switch)) (b/project ps (sel/select :switch))))))))

;; ---------------------------------------------------------------------------
;; Regressions from the M1 review (docs/research/review-2026-10-18.md)
;; ---------------------------------------------------------------------------

(defn- batched-matches-scalar?
  "Do batched importance weights equal the scalar generate weights of the
   same particles? (Per particle, from the batched particles' own choices.)"
  [model args obs n key]
  (let [{ps :particles ws :weights} (b/generate model args obs n key)]
    (every? true? (map (fn [tr w] (close? w (gfi/project tr (sel/from-choicemap obs))))
                       (b/traces ps) (array/->clj ws)))))

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
  (let [ys (array/array [0.5 1.0 1.5])
        ;; three i.i.d. observations: vector parameters (Gen.jl's broadcasted_normal)
        m (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :ys (d/normal (+ mu (array/zeros [3])) 1))))]
    (doseq [n [3 4]]
      (let [{ps :particles ws :weights} (b/generate m [] (cm/choicemap :ys ys) n (rng/key 32))]
        (is (b/batched? ps))
        (is (every? true? (map (fn [mu w] (close? w (array/sum (dc/log-prob (d/normal mu 1) ys))))
                               (array/->clj (b/choice ps :mu)) (array/->clj ws))))))
    (testing "a vector site's draws have the batch axis in front"
      (let [v (gen [] (trace :z (d/normal (array/zeros [3]) 1)))]
        (is (= [5 3] (array/shape (b/choice (b/simulate v [] 5 (rng/key 33)) :z))))
        (is (batched-matches-scalar? v [] (cm/choicemap :z (array/array [0.1 0.2 0.3])) 3 (rng/key 34)))))
    (testing "a per-particle scalar meets a shared vector in model arithmetic"
      (let [m (gen [] (let [x (trace :x (d/normal 0 1))]
                        (trace :y (d/normal (array/sum (* x (array/array [1.0 2.0 3.0]))) 1))))]
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
    ;; case (and if) only inside gen bodies, which guard them. The helpers
    ;; test host numbers: what the JVM backend's 0-d values are, and MLX's
    ;; are not (on MLX a helper must read a value with array/item)
    (doseq [[label mu-of] (if (ts/jvm?) [[:=-on-the-right (fn [k] (if (= 0 k) -3.0 3.0))]
                           [:not= (fn [k] (if (not= 0 k) 3.0 -3.0))]
                           [:condp (fn [k] (condp = k 0 -3.0 3.0))]
                           [:vector-lookup (fn [k] ([-3.0 3.0] k))]] [])]
      (let [m (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))] (trace :y (d/normal (mu-of k) 1))))
            {:keys [log-ml-estimate]} (imp/importance-sampling m [] obs 4000 {:batched? true :key (rng/key 40)})]
        (is (< (Math/abs (- log-ml-estimate exact)) 0.1) (str label))))
    (let [m (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))] (trace :y (d/normal (case k 0 -3.0 3.0) 1))))
          {:keys [log-ml-estimate]} (imp/importance-sampling m [] obs 4000 {:batched? true :key (rng/key 40)})]
      (is (< (Math/abs (- log-ml-estimate exact)) 0.1) "case with a default, in a gen body"))))

(deftest batched-runs-validate-parameters
  ;; B3: a batched run threw nothing where a scalar run throws
  ;; (on eager backends: a lazy one does not read array parameters, rule 5)
  (let [m (gen [] (let [x (trace :x (d/uniform 0 1))] (trace :k (d/categorical (array/stack [x x])))))]
    (when (array/concrete? (array/zeros [1]))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"probabilities" (gfi/simulate m [] (rng/key 1))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"probabilities" (b/simulate m [] 4 (rng/key 1)))))))

(deftest more-batched-forms
  (testing "B9: scans and arg-reductions of a per-particle scalar"
    (let [m (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (array/sum (array/cumsum x)) 1))))]
      (is (b/batched? (b/simulate m [] 3 (rng/key 41))))))
  (testing "B10: an Unfold whose length is a choice falls back"
    (let [m (gen [] (splice :u unfolded (trace :n (d/uniform-discrete 1 3)) 0.0))]
      (is (every? #(close? (:score %) (:weight (gfi/assess m [] (:choices %)))) (b/traces (b/simulate m [] 4 (rng/key 42)))))))
  (testing "B11: a collection of a distribution's particles"
    (is (= 3 (count (b/traces (b/simulate (d/normal 0 1) [] 3 (rng/key 43)))))))
  (testing "a per-particle fill value stays batched"
    (let [m (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (array/full [2] x) 1))))]
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
  (every? true? (map #(close? %1 (:weight (gfi/assess gf args (:choices %2)))) (array/->clj (b/scores ps)) (b/traces ps))))

(defn- falls-back-exactly?
  "Does gf fall back in batched importance sampling, giving the scalar
   estimate exactly (same key)? Spot checks are off: the guard under test
   must cause the fallback by itself."
  [gf args obs]
  (let [s (imp/importance-sampling gf args obs 200 {:key (rng/key 60)})
        r (binding [b/*spot-checks* 0]
            (imp/importance-sampling gf args obs 200 {:key (rng/key 60) :batched? true}))]
    (and (not (b/batched? (:particles r)))
         ;; exactly, but for float32 storage: a scalar run of host parameters
         ;; computes on the host in float64, the fallback stores its weights
         ;; as an array of the backend's float
         (if (ts/f32?)
           (ts/close? (:log-ml-estimate s) (:log-ml-estimate r))
           (= (:log-ml-estimate s) (:log-ml-estimate r))))))

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
        (is (number? (ts/host (:retval (gfi/simulate m [] (rng/key 63))))))))))

(defn- drift
  "A raw if on an array comparison, in a helper outside gen: under the
   tracer the comparison is a batch, which is truthy, so every particle would
   take the first branch."
  [x]
  (if (array/> x 0) 1.0 -1.0))

(def hidden-branch (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (drift x) 1)))))

(deftest a-comparison-nothing-reads-falls-back
  ;; second acceptance test B3: a comparison that only a raw if consumed
  (let [ps (b/simulate hidden-branch [] 50 (rng/key 50))]
    (is (not (b/batched? ps)))
    (is (re-find #"comparison was never read" (:message (b/fallback-reason ps))))
    (is (scores-exact? hidden-branch [] ps)))
  (let [rare (fn [x] (if (array/< x 2.0) 1.0 0.1))
        m (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x (rare x)))))]
    (is (falls-back-exactly? m [] (cm/choicemap :y 2.5)) "however rare the other branch"))
  (testing "while comparisons that are read, or handed back, stay batched"
    (is (b/batched? (b/simulate (gen [] (let [x (trace :x (d/normal 0 1))] (array/> x 0))) [] 8 (rng/key 51))))
    (is (b/batched? (b/simulate (gen [] (let [x (trace :x (d/normal 0 1))]
                                          (trace :y (d/normal (array/where (array/> x 0) 1.0 -1.0) 1))))
                                [] 8 (rng/key 52))))))

(defn- level
  "A vector lookup keyed by a choice, in a helper outside gen: a batched key
   is no integer to a vector, which silently gives the default."
  [k]
  (get [-3.0 3.0] k 0.0))

(defn- address-of
  "A case on a choice, in a helper outside gen: a batched key matches no
   clause, silently."
  [k]
  (case k 1 :rare :common))

(def hidden-lookup (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))] (trace :y (d/normal (level k) 1)))))

(dyn/defn-guarded guarded-level
  "level, defined with a body's guards."
  [k]
  (get [-3.0 3.0] k 0.0))

(deftest helpers-defined-with-guards-branch-per-particle
  (let [m (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))] (trace :y (d/normal (guarded-level k) 1))))]
    (is (falls-back-exactly? m [] (cm/choicemap :y 0.5)))
    (is (= 3.0 (guarded-level 1)) "and are ordinary functions otherwise"))
  (let [sgn (dyn/guarded-fn [x] (if (array/> x 0) 1.0 -1.0))
        m (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (sgn x) 1))))]
    (is (falls-back-exactly? m [] (cm/choicemap :y 0.5)))))

(defn- safe-item
  "A plain helper that catches what a batched value raises."
  [x]
  (try (array/item x) (catch Exception _ 0.0)))

(deftest caught-exceptions-are-branches
  ;; fourth acceptance test B1: a caught batched-branch error, or any
  ;; exception caught in a body, may be thrown for some particles only
  (let [obs (cm/choicemap :y 3.0)]
    (is (falls-back-exactly? (gen [] (let [x (trace :x (d/normal 0 1))]
                                       (trace :y (d/normal (try (if (array/> x 2.0) 3.0 0.0) (catch Exception _ 0.0)) 1))))
                             [] obs)
        "a try in a body")
    (is (falls-back-exactly? (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (safe-item x) 1))))
                             [] obs)
        "a plain helper catching the batched-branch error")
    (is (b/batched? (b/simulate (gen [] (let [x (trace :x (d/normal 0 1))] (try x (finally nil)))) [] 8 (rng/key 72)))
        "a try that catches nothing stays batched")))

(defprotocol ^:private Scalar (scalar-kind [x]))
(extend-protocol Scalar Double (scalar-kind [_] :double))

(def ^:private vector-class "The class of a [2] array." (class (array/zeros [2])))

(deftest representation-tests-answer-for-the-particles
  ;; fourth acceptance test B2
  (doseq [[label m] [[:instance? (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (if (instance? Double x) 3.0 0.0) 1))))]
                     [:host? (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (if (array/host? x) 3.0 0.0) 1))))]
                     [:coll? (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (if (coll? x) 0.0 3.0) 1))))]
                     [:class (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (if (= Double (class x)) 3.0 0.0) 1))))]
                     [:satisfies? (gen [] (let [x (trace :x (d/normal 0 1))]
                                            (trace :y (d/normal (if (satisfies? Scalar x) 3.0 0.0) 1))))]
                     ;; the post-M5 review's I2-04: one predicate, and a vector
                     ;; particle's prototype made through gen.array; plants: the
                     ;; unguarded backend-array? (true of a batch) fails the
                     ;; first, a prototype that is the batch itself the second
                     [:backend-array? (gen [] (let [x (trace :x (d/normal 0 1))]
                                                (trace :y (d/normal (if (array/backend-array? x) 0.0 3.0) 1))))]
                     [:class-of-a-vector (gen [] (let [x (trace :x (d/normal (array/zeros [2]) 1))]
                                                   (trace :y (d/normal (if (= vector-class (class x)) 3.0 0.0) 1))))]]]
    (let [obs (cm/choicemap :y 3.0)
          s (imp/importance-sampling m [] obs 200 {:key (rng/key 73)})
          r (binding [b/*spot-checks* 0] (imp/importance-sampling m [] obs 200 {:key (rng/key 73) :batched? true}))]
      ;; every particle takes the 3.0 branch, so both estimates are exact
      (is (b/batched? (:particles r)) (str label " answers without falling back"))
      (is (close? (dc/log-prob (d/normal 3.0 1) 3.0) (:log-ml-estimate s)) (str label))
      (is (close? (dc/log-prob (d/normal 3.0 1) 3.0) (:log-ml-estimate r)) (str label)))))

(deftest values-the-trace-holds-are-handed-back
  ;; third acceptance test: needless fallbacks
  (let [callee (gen [] (let [x (trace :x (d/normal 0 1))] {:x x :big? (array/> x 0)}))
        m (gen [] (let [{:keys [x]} (splice :c callee)] (trace :y (d/normal x 1))))]
    (is (b/batched? (b/simulate m [] 8 (rng/key 70))) "a sub-trace's return value"))
  (let [m (gen [] (let [x (trace :x (d/normal 0 1))] (assert (number? x)) (trace :y (d/normal x 1))))]
    (is (b/batched? (b/simulate m [] 8 (rng/key 71))) "type predicates answer from the dtype")))

(deftest spot-checks-catch-what-the-tracer-cannot-see
  ;; The planted helpers branch on host numbers (case, vector lookups), which
  ;; only the JVM backend's 0-d values are; on MLX such a helper branches
  ;; alike in batched and scalar runs (it never matches), so there is nothing
  ;; for a spot check to find
  (when (ts/jvm?)
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
                             (b/traces ps) (array/->clj ws))))))
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
  (testing "results read after unchecked updates are checked"
    (let [m (gen [t] (let [k (trace :k (d/categorical [0.5 0.5]))]
                       (when (> t 2) (trace :y (d/normal (level k) 1)))
                       k))
          step (fn [ps t] (:particles (b/update ps [t] [diff/unknown-change] cm/EMPTY (rng/key t))))
          ps (reduce step (b/simulate m [0] 20 (rng/key 55)) [1 2 3])]
      ;; update 3 is unchecked and wrong, and no later update checks it
      (is (b/batched? ps))
      (doseq [read [b/scores b/traces #(b/choice % :k)]]
        (is (:gen.batched/diverged (ex-data (try (read ps) nil (catch clojure.lang.ExceptionInfo e e))))))))
  (testing "and a divergence in structure is one too"
    (let [m (gen [t] (let [k (trace :k (d/categorical [0.5 0.5]))]
                       (when (> t 2) (trace (address-of k) (d/normal 0 1)))
                       k))
          step (fn [ps t] (:particles (b/update ps [t] [diff/unknown-change] cm/EMPTY (rng/key t))))
          ps (reduce step (b/simulate m [0] 20 (rng/key 56)) [1 2 3])]
      (is (:gen.batched/diverged (ex-data (try (b/scores ps) nil (catch clojure.lang.ExceptionInfo e e)))))))
  (testing "a batched-branch fallback records the error's data"
    (is (:gen.array/batched-branch (:data (b/fallback-reason (b/simulate branching [] 8 (rng/key 56)))))))
  (testing "well-behaved models stay batched, also when every particle is checked"
    (binding [b/*spot-checks* :all]
      (let [ps (b/simulate linreg [[1.0 2.0]] 16 (rng/key 57))]
        (is (b/batched? ps))
        (is (nil? (b/fallback-reason ps))))))))

(defn- spot-checked-on
  "f's value, its batched runs spot-checked as on a lazy backend (lazy?:
   re-run on the JVM backend) or as on an eager one (re-run on their own),
   whatever the backend."
  [lazy? f]
  (with-redefs [gen.batched/lazy-batch? (constantly lazy?)] (f)))

(deftest one-spot-check-for-every-backend
  ;; Both kinds of backend check the same host reads of the checked
  ;; particles, re-run on the JVM backend or on the batch's own (the post-M5
  ;; review's I2-07, which made the two loops one).
  (when (ts/jvm?)
    (testing "either way, what the tracer cannot see falls back"
      ;; plant: a re-run that compares nothing passes both
      (doseq [lazy? [true false]]
        (let [ps (spot-checked-on lazy? #(b/simulate hidden-lookup [] 50 (rng/key 50)))]
          (is (re-find #"disagrees with the scalar handler" (:message (b/fallback-reason ps))) (str lazy?))))))
  (testing "on its own backend a choice is re-run in its dtype"
    ;; the verifier's caveat (a dirichlet of float32 parameters draws
    ;; float32); plant: a choice made again from its host values by
    ;; array/array alone, float64 on the JVM backend, takes the other branch
    ;; and falls back
    (let [m (gen [] (let [x (trace :x (d/dirichlet (array/array [1.0 1.0] {:dtype :float32})))]
                      (trace :y (d/normal (if (= :float32 (array/dtype x)) 0.0 5.0) 1))))]
      (binding [b/*spot-checks* :all]
        (is (b/batched? (spot-checked-on false #(b/simulate m [] 8 (rng/key 58)))))))))

(deftest float32-rounding-is-no-mismatch
  ;; The M2 review's finding 4 (genclj-i8ui): a lazy backend's batch is
  ;; checked against float64 re-runs on the JVM backend. A float32 batch of
  ;; an ill-conditioned model (an intercept near 1, x = 1000..1999 as an
  ;; argument, y near 2x + 1 with sd 0.01) rounds its means where those
  ;; runs do not: on the JVM backend 0.03-0.56 nats apart at the 8 checked
  ;; particles, against float32's tolerance of 0.034 (the JVM backend's
  ;; prior draws float64, so its score is float64 and its tolerance 3e-5).
  ;; The model computes in float32 on every backend (its arrays and the
  ;; cast), and the JVM re-run is taken on every backend. Plant: the JVM
  ;; re-run's mismatch taken as final, as before the fix: generate falls
  ;; back, and the update that adds the observations, being unchecked,
  ;; leaves a collection that diverges when read (on the JVM and MLX).
  (let [f32 #(array/array % {:dtype :float32})
        xs (f32 (mapv double (range 1000 2000)))
        ys (f32 (mapv #(+ (* 2.0 %) 1.0 (* 0.01 (Math/sin %))) (range 1000 2000)))
        mean (fn [b xs] (+ (array/astype b :float32) (* 2 xs)))]
    (testing "a batched generate stays batched"
      (let [m (gen [xs] (let [b (trace :b (d/normal (f32 [1.0]) 0.001))] (trace :y (d/normal (mean b xs) 0.01))))
            {ps :particles} (spot-checked-on true #(b/generate m [xs] (cm/choicemap :y ys) 64 (rng/key 80)))]
        (is (nil? (b/fallback-reason ps)))))
    (testing "observations added by an unchecked update do not diverge"
      ;; updates 1 and 2 are checked (one observation of b each, float64 on
      ;; the JVM backend), 3 (the ill-conditioned observations) is not
      (let [m (gen [xs k] (let [b (trace :b (d/normal (f32 [1.0]) 0.001))]
                            (doseq [j (range k)]
                              (if (= j 2)
                                (trace [:y j] (d/normal (mean b xs) 0.01))
                                (trace [:y j] (d/normal b 1.0))))))
            step (fn [ps k] (:particles (b/update ps [xs k] [diff/no-change diff/unknown-change]
                                                  (cm/choicemap [:y (dec k)] (if (= k 3) ys (f32 [1.0]))) (rng/key k))))
            [ps error] (spot-checked-on true #(let [ps (reduce step (b/simulate m [xs 0] 64 (rng/key 81)) [1 2 3])]
                                                [ps (try (b/scores ps) nil (catch clojure.lang.ExceptionInfo e (ex-message e)))]))]
        (is (nil? (b/fallback-reason ps)))
        (is (nil? error))))))
;; ---------------------------------------------------------------------------
;; Switches by partition (genclj-7tqw)
;; ---------------------------------------------------------------------------

;; A Switch whose index differs across the particles runs each branch on its
;; own particles (DESIGN.md §9). The fuzz tests (gen.fuzz-test) check every
;; operation particle by particle on random Switch-heavy models; these check
;; the draws, the loud edges, and inference against exact answers.

(def component-a (gen [m] (trace :x (d/normal m 1.0))))

(def component-b
  "Structurally different from component-a: x's mean is a choice of its own."
  (gen [m] (let [u (trace :u (d/normal 0 1))] (trace :x (d/normal (+ m u 3.0) 0.5)))))

(def components (comb/switch component-a component-b))

(def mixture
  (gen [] (let [k (trace :k (d/categorical [0.3 0.7]))]
            (trace :y (d/normal (splice :c components k 0.0) 1.0)))))

;; y ~ 0.3 N(0, 2) + 0.7 N(3, 2.25): x is N(0, 1) under a and N(3, 1.25)
;; under b, and y | x is N(x, 1)
(def mixture-y 1.2)
(def mixture-parts [(* 0.3 (Math/exp (dc/log-prob (d/normal 0 (Math/sqrt 2.0)) mixture-y)))
                    (* 0.7 (Math/exp (dc/log-prob (d/normal 3 1.5) mixture-y)))])
(def mixture-evidence (Math/log (reduce + mixture-parts)))
(def mixture-p0 (/ (first mixture-parts) (reduce + mixture-parts)))

(deftest a-switch-of-one-branch-draws-what-the-branch-would
  ;; A partition runs each branch on its own particles, under a part's
  ;; backend whose draws are the whole batch's at their rows: a Switch whose
  ;; branches are all one generative function makes every particle the
  ;; choices the branch alone would, whatever the partition, in every
  ;; operation that draws. Exactly so where an elementwise op rounds alike
  ;; at every array length; MLX's CPU rounds log1p and pow an ulp apart in
  ;; its SIMD lanes and its scalar tail (2026-10-24), so values compare
  ;; within the backend's tolerance. Plant: parts that draw their own batch
  ;; (not the batch's rows) fail every comparison.
  (let [g (gen [m] (let [z (trace :z (d/normal m 1))]
                     (trace :w (d/gamma 2 1))
                     (splice :v (comb/map (gen [x] (trace :v (d/normal x 1)))) [z z])
                     z))
        ps [0.2 0.3 0.5]
        direct (gen [] (trace :i (d/categorical ps)) (splice :s g 0.5))
        parted (gen [] (splice :s (comb/switch g g g) (trace :i (d/categorical ps)) 0.5))
        nested (gen [] (splice :s (comb/switch (gen [m] (splice :t (comb/switch g g) (trace :j (d/categorical [0.5 0.5])) m)) g g)
                               (trace :i (d/categorical ps)) 0.5))
        alike? (fn [a b] (and (= (cm/addresses a) (cm/addresses b))
                              (every? #(ts/close? (cm/get-choice a %) (cm/get-choice b %)) (cm/addresses a))))
        same? (fn [ps qs] (every? true? (map alike? (map :choices (b/traces ps)) (map :choices (b/traces qs)))))
        n 9
        p (b/simulate direct [] n (rng/key 1))
        q (b/simulate parted [] n (rng/key 1))]
    (is (b/batched? q))
    (is (< 1 (count (distinct (array/->clj (b/choice q :i))))) "the index differs across the particles")
    (testing "simulate"
      (is (same? p q)))
    (testing "generate, constrained by each particle's own index"
      (let [own (mapv #(cm/choicemap :i %) [0 1 2 2 1 0 0 1 2])
            gp (b/generate direct [] cm/EMPTY n (rng/key 2) {:per-particle own})
            gq (b/generate parted [] cm/EMPTY n (rng/key 2) {:per-particle own})]
        (is (same? (:particles gp) (:particles gq)))
        (is (close? (:weights gp) (:weights gq)))))
    (testing "update, moving particles between branches"
      (let [own (mapv #(cm/choicemap :i %) [2 2 0 1 0 1 2 0 1])
            up (b/update p [] [] (cm/choicemap [:s :z] 0.3) (rng/key 3) {:per-particle own})
            uq (b/update q [] [] (cm/choicemap [:s :z] 0.3) (rng/key 3) {:per-particle own})]
        (is (same? (:particles up) (:particles uq)))
        (is (close? (:weights up) (:weights uq)))))
    (testing "regenerate, of each branch's choices"
      (let [rp (b/regenerate p [] [] (sel/select [:s :z] [:s :v]) (rng/key 4))
            rq (b/regenerate q [] [] (sel/select [:s :z] [:s :v]) (rng/key 4))]
        (is (same? (:particles rp) (:particles rq)))
        (is (close? (:weights rp) (:weights rq)))))
    (testing "and through nested Switches"
      (let [r (b/simulate nested [] n (rng/key 1))
            in-g (fn [tr] (if (cm/has-submap? (:choices tr) [:s :t]) (cm/get-submap (:choices tr) [:s :t]) (cm/get-submap (:choices tr) :s)))]
        (is (b/batched? r))
        (is (every? true? (map alike? (map in-g (b/traces r)) (map #(cm/get-submap (:choices %) :s) (b/traces p)))))))))

(deftest switches-by-partition-at-the-edges
  (testing "the fallback no longer triggers for a per-particle index"
    (binding [b/*spot-checks* :all]
      (let [ps (b/simulate mixture [] 40 (rng/key 70))]
        (is (b/batched? ps))
        (is (nil? (b/fallback-reason ps))))))
  (testing "a choice every particle makes reads through the partition, one only some make is loud"
    (let [ps (b/simulate mixture [] 12 (rng/key 71))
          trs (b/traces ps)]
      (is (close? (array/stack (mapv #(% [:c :x]) trs)) (b/choice ps [:c :x])))
      (is (= (mapv #(ts/host (% :k)) trs) (array/->clj (b/choice ps :k))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not every particle has a choice at \[:c :u\]"
                            (b/choice ps [:c :u])))))
  (testing "return values that differ in shape across the branches fall back, loudly"
    (let [m (gen [] (splice :s (comb/switch (gen [] (trace :x (d/normal 0 1)))
                                            (gen [] (trace :x (d/normal (array/zeros [2]) 1))))
                            (trace :i (d/categorical [0.5 0.5]))))
          ps (b/simulate m [] 8 (rng/key 72))]
      (is (not (b/batched? ps)))
      (is (re-find #"differ in shape or dtype" (:message (b/fallback-reason ps))))
      (is (every? #(close? (:score %) (:weight (gfi/assess m [] (:choices %)))) (b/traces ps)))))
  (testing "a partitioned call revised by another generative function falls back, loudly"
    ;; its choices are held by part: no one choice map to generate from
    (let [other (gen [_ m] (trace :x (d/normal m 2.0)))
          m (gen [flag] (let [k (trace :k (d/categorical [0.5 0.5]))]
                          (splice :c (if flag components other) k 0.0)))
          ps (b/simulate m [true] 8 (rng/key 73))
          {ps' :particles ws :weights} (b/update ps [false] [diff/unknown-change] cm/EMPTY (rng/key 74))]
      (is (b/batched? ps))
      (is (not (b/batched? ps')))
      (is (re-find #"revised by another generative function" (:message (b/fallback-reason ps'))))
      ;; per particle the scalar update: :x kept, the rest dropped
      (is (every? true? (map (fn [before after w] (and (= (ts/host (before [:c :x])) (ts/host (after [:c :x])))
                                                      (close? w (- (:score after) (:score before)))))
                             (b/traces ps) (b/traces ps') (array/->clj ws))))))
  (testing "spot checks see a branch's hidden lookup through the partition, re-run either way"
    (when (ts/jvm?)
      (doseq [lazy? [true false]]
        (let [m (gen [] (splice :s (comb/switch (gen [_] (let [k (trace :k (d/categorical [0.5 0.5]))]
                                                          (trace :y (d/normal (level k) 1))))
                                                observation)
                                (trace :i (d/categorical [0.5 0.5])) 0.0))
              ps (spot-checked-on lazy? #(b/simulate m [] 50 (rng/key 75)))]
          (is (re-find #"disagrees with the scalar handler" (:message (b/fallback-reason ps))) (str lazy?)))))))

(deftest a-batched-mixture-of-different-structures-reaches-the-exact-answers
  (let [{:keys [particles log-ml-estimate log-normalized-weights]}
        (imp/importance-sampling mixture [] (cm/choicemap :y mixture-y) 20000 {:batched? true :key (rng/key 76)})
        w (map #(Math/exp %) log-normalized-weights)
        ;; the estimator's standard errors, from its weights (delta method)
        se-log-ml (Math/sqrt (- (reduce + (map * w w)) (/ 1.0 20000)))
        zero? (map #(= 0 %) (array/->clj (b/choice particles :k)))
        p0 (reduce + (map (fn [wi z] (if z wi 0.0)) w zero?))
        se-p0 (Math/sqrt (reduce + (map (fn [wi z] (let [d (- (if z 1.0 0.0) p0)] (* wi wi d d))) w zero?)))]
    (is (b/batched? particles))
    (is (nil? (b/fallback-reason particles)))
    (is (< (Math/abs (- log-ml-estimate mixture-evidence)) (* 5 se-log-ml)))
    (is (< (Math/abs (- p0 mixture-p0)) (* 5 se-p0)))))

(deftest parallel-mh-chains-on-a-mixture-move-between-branches
  ;; independent MH chains whose moves change the index (moving particles
  ;; between branches, the shared :x kept), or redraw the index and the
  ;; branch together; each chain's last k is a draw from the posterior
  (let [n 2000
        start (:particles (b/generate mixture [] (cm/choicemap :y mixture-y) n (rng/key 77)))
        chains (reduce (fn [ps [i k]] (:particles (mh/mh-particles ps (if (even? i) (sel/select :k) (sel/select :k :c)) k)))
                       start (map-indexed vector (rng/split (rng/key 78) 40)))
        p0 (/ (count (filter #(= 0 %) (array/->clj (b/choice chains :k)))) (double n))]
    (is (b/batched? chains))
    (is (< (Math/abs (- p0 mixture-p0)) (* 5 (Math/sqrt (/ (* mixture-p0 (- 1 mixture-p0)) n)))))))

(def shifted-mixture
  "The mixture, shifted by a choice outside the Switch."
  (gen [] (let [mu (trace :mu (d/normal 0 1))
                k (trace :k (d/categorical [0.3 0.7]))]
            (trace :y (d/normal (+ mu (splice :c components k 0.0)) 1.0)))))

(def fed-mixture
  "The mixture, its components' mean a choice: what is differentiated flows
   into the branches."
  (gen [] (let [mu (trace :mu (d/normal 0 1))
                k (trace :k (d/categorical [0.3 0.7]))]
            (trace :y (d/normal (splice :c components k mu) 1.0)))))

(deftest gradients-and-set-choices-across-a-partitioned-switch
  ;; vmap's semantics either way: a selection outside the Switch whose
  ;; values do not flow into it is one batched gradient (the trace's choices
  ;; by part given back to the branches as constraints, gen.combinators'
  ;; own-constraints); one inside it, or flowing into its branches, falls
  ;; back to each trace's gradient. Planted: without own-constraints the
  ;; batched gradient falls back
  (let [n 8
        obs (cm/choicemap :y 1.7)
        each (fn [ps s] (map #(:gradients (grad/choice-gradients % s)) (b/traces ps)))
        stacked (fn [gs p] (array/stack (mapv #(cm/get-choice % p) gs)))
        unbatched (fn [f] (with-redefs [b/traces (fn [_] (throw (ex-info "fell back" {})))] (f)))]
    (testing "outside the Switch: batched"
      (let [ps (:particles (b/generate shifted-mixture [] obs n (rng/key 90)))
            s (sel/select :mu)
            want (each ps s)
            r (unbatched #(grad/choice-gradients ps s))]
        (is (b/batched? ps))
        (is (< 1 (count (distinct (array/->clj (b/choice ps :k))))))
        (is (close? (stacked want [:mu]) (cm/get-choice (:gradients r) :mu)))))
    (testing "inside the Switch, or flowing into it: each trace's"
      (doseq [[m s p] [[shifted-mixture (sel/select [:c :x]) [:c :x]] [fed-mixture (sel/select :mu) [:mu]]]]
        (let [ps (:particles (b/generate m [] obs n (rng/key 91)))
              r (grad/choice-gradients ps s)]
          (is (b/batched? ps))
          (is (close? (stacked (each ps s) p) (cm/get-choice (:gradients r) p))))))
    (testing "set-choices outside the Switch stays batched, inside sets each particle's"
      (let [ps (:particles (b/generate shifted-mixture [] obs n (rng/key 92)))
            vs (array/array (mapv #(* 0.25 %) (range n)))]
        (doseq [[p batched?] [[:mu true] [[:c :x] false]]]
          (let [ps' (b/set-choices ps (cm/choicemap p vs))
                trs (b/traces ps')]
            (is (= batched? (b/batched? ps')) (str p))
            (is (close? vs (array/stack (mapv #(% p) trs))) (str p))
            (is (= (map #(ts/host (sel/get-selected (:choices %) (sel/complement (sel/from-paths [(cm/path p)])))) (b/traces ps))
                   (map #(ts/host (sel/get-selected (:choices %) (sel/complement (sel/from-paths [(cm/path p)])))) trs))
                "every other choice kept")
            (is (every? #(close? (:score %) (:weight (gfi/assess shifted-mixture [] (:choices %)))) trs))))))
    (testing "HMC chains on the shift stay batched"
      (let [ps (:particles (b/generate shifted-mixture [] obs n (rng/key 93)))
            {ps' :particles} (mcmc/hmc-particles ps (sel/select :mu) {:eps 0.1 :L 5} (rng/key 94))]
        (is (b/batched? ps'))
        (is (= (map #(ts/host (% :k)) (b/traces ps)) (map #(ts/host (% :k)) (b/traces ps'))))))))

(def regimes
  "Two regimes of different structure: y directly, or around a choice of
   its own."
  (comb/switch (gen [] (trace :y (d/normal -1.0 0.6)))
               (gen [] (let [u (trace :u (d/normal 0 1))] (trace :y (d/normal (+ 1.5 u) 0.5))))))

(def regime-step
  (gen [t k] (let [k' (trace :k (d/categorical (array/where (array/== k 0) (array/array [0.8 0.2]) (array/array [0.3 0.7]))))]
               (splice :e regimes k')
               k')))

(deftest ^:slow a-batched-particle-filter-over-switching-regimes-is-unbiased
  ;; an HMM of two regimes whose emissions differ in structure: the forward
  ;; algorithm gives p(ys), and the filter's estimate is unbiased for it
  ;; (law 20): the mean of Z/p(ys) over runs is 1, within 5 standard errors
  ;; of the runs' measured spread
  (let [ys [0.4 -1.2 1.9 2.2 -0.3 1.0]
        emit [#(dc/log-prob (d/normal -1.0 0.6) %) #(dc/log-prob (d/normal 1.5 (Math/sqrt 1.25)) %)]
        trans [[0.8 0.2] [0.3 0.7]]
        exact (let [step (fn [alpha y] (mapv (fn [j] (* (Math/exp ((emit j) y))
                                                         (reduce + (map-indexed (fn [i a] (* a ((trans i) j))) alpha))))
                                             [0 1]))]
                (Math/log (reduce + (reduce step [1.0 0.0] ys))))
        unfolded (comb/unfold regime-step)
        obs (fn [t] (cm/choicemap [t :e :y] (ys t)))
        run (fn [n key]
              (let [ks (rng/split key 12)
                    init (pf/initialize unfolded [1 0] (obs 0) n {:batched? true :key (ks 0)})]
                (reduce (fn [s t] (-> s (pf/maybe-resample {:key (ks (dec (* 2 t)))})
                                      (pf/step [(inc t) 0] [diff/unknown-change diff/no-change] (obs t) {:key (ks (* 2 t))})))
                        init (range 1 (count ys)))))
        runs (mapv #(run 100 %) (rng/split (rng/key 79) 200))
        ratios (map #(Math/exp (- (pf/log-ml-estimate %) exact)) runs)
        mean (/ (reduce + ratios) (count runs))
        sd (Math/sqrt (/ (reduce + (map #(* (- % mean) (- % mean)) ratios)) (dec (count runs))))]
    (is (every? #(b/batched? (:particles %)) runs) "every filter stays batched")
    (is (< (Math/abs (- mean 1.0)) (* 5 (/ sd (Math/sqrt (count runs))))))))

;; ---------------------------------------------------------------------------
;; Functions of every particle: map-particles and set-choices
;; ---------------------------------------------------------------------------

(def ^:private observed (cm/choicemap [:y 0] 2.1 [:y 1] 3.9 [:y 2] 6.2))

(defn- unbatched
  "ps as the per-particle collection a fallback leaves."
  [ps]
  (assoc ps :trace nil :traces (b/traces ps)))

(defn- each-trace
  "f of each of ps's traces, with its row of x, stacked: map-particles by
   definition (no draws)."
  [ps f x]
  (let [rs (map-indexed (fn [i t] (f t nil (array/take x i 0))) (b/traces ps))]
    {:score (array/stack (mapv :score rs))
     :shifted (array/stack (mapv :shifted rs))
     :pair [(array/stack (mapv (comp first :pair) rs)) (array/stack (mapv (comp second :pair) rs))]
     :doubled (array/stack (mapv #(cm/get-choice (:choices %) :s) rs))}))

(defn- same-results? [r want]
  (and (close? (:score r) (:score want))
       (close? (:shifted r) (:shifted want))
       (every? true? (map close? (:pair r) (:pair want)))
       (close? (cm/get-choice (:choices r) :s) (:doubled want))))

(deftest map-particles-is-f-of-each-trace
  ;; vmap's semantics: batched or not, particle i's result is f's on its
  ;; trace and its rows. Planted (2026-10-23): every particle given
  ;; particle 0's rows fails the per-particle cases and the fallbacks; a
  ;; shared value left unbroadcast fails its shape; the guard without its
  ;; check of unread comparisons fails the raw if
  (let [n 5
        ps (:particles (b/generate linreg [[1.0 2.0 3.0]] observed n (rng/key 80)))
        shifts (array/array (mapv #(* 0.5 %) (range n)))
        f (fn [t _ shift] {:score (:score t) :shifted (+ (t :slope) shift) :pair [(t :intercept) 2.0]
                           :choices (cm/choicemap :s (* 2.0 (t :slope)))})
        want (each-trace ps f shifts)]
    (is (b/batched? ps))
    (testing "batched"
      (let [r (b/map-particles ps f nil shifts)]
        (is (same-results? r want))
        (is (= [n] (array/shape (second (:pair r)))) "a shared value, broadcast to every particle")))
    (testing "per particle, the form a fallback leaves"
      (is (same-results? (b/map-particles (unbatched ps) f nil shifts) want)))
    (testing "where f reads a particle's value on the host, or branches on one with a raw if, one trace at a time"
      ;; batched, the raw if would take its first branch for every particle.
      ;; Each g against itself on each trace: on MLX a scalar comparison is
      ;; an array, which a raw if takes for true too (docs/array-api.md §8a,
      ;; plain helpers), so there g's own scalar semantics add 100 throughout
      (doseq [g [(fn [t k shift] (assoc (f t k shift) :read (array/item (t :slope))))
                 (fn [t k shift] (update (f t k shift) :shifted #(if (array/> (t :slope) 1e9) (+ % 100.0) %)))]]
        (is (same-results? (b/map-particles ps g nil shifts) (each-trace ps g shifts)))))
    (testing "draws are per particle"
      (let [r (b/map-particles ps (fn [_ k] (rng/normal k [2])) (rng/key 81))]
        (is (= [n 2] (array/shape r)))
        (is (= n (count (distinct (map first (ts/host r))))))))))

(deftest set-choices-sets-each-particle-s-own
  ;; each trace is its generative function's trace of its choices with the
  ;; new values, batched or not. Planted (2026-10-23): the new values
  ;; merged under the old ones keep the old slopes
  (let [n 4
        ps (:particles (b/generate linreg [[1.0 2.0 3.0]] observed n (rng/key 82)))
        slopes (array/array [-1.0 0.5 2.0 3.5])
        new (cm/choicemap :slope slopes)]
    (doseq [[label ps] [["batched" ps] ["per particle" (unbatched ps)]]]
      (testing label
        (let [ps' (b/set-choices ps new)
              trs (b/traces ps')]
          (is (= (b/batched? ps) (b/batched? ps')))
          (is (close? slopes (b/choice ps' :slope)))
          (is (close? (b/choice ps :intercept) (b/choice ps' :intercept)) "the other choices kept")
          (is (every? consistent? trs))
          (is (close? (b/scores ps') (array/stack (mapv :score trs)))))))))

;; ---------------------------------------------------------------------------
;; A particle outside its support (DESIGN.md §9, the ten decisions' fifth)
;; ---------------------------------------------------------------------------

;; A batched run cannot stop for one particle: it goes on, and the handler
;; sets the score and weight of a particle whose choice left its support
;; to -Inf at the end (gen.dynamic's weigh), what the particle's scalar run,
;; which stops there, gives. What it computed after that choice is masked:
;; here b's beta density at 0, +Inf, which with the -Inf would sum to NaN.
;; Its trace holds the whole run's choices and their densities as computed
;; (NaN where a parameter the impossible value made was invalid); b/traces
;; rebuilds the scalar trace, which stops at the choice. Where a parameter check downstream
;; reads the batch (the JVM's concrete arrays) and fails for such a
;; particle, the run falls back to one particle at a time, which stops it.
;; Planted bugs (2026-10-25): the mask read but not applied gives NaN
;; weights for the particles outside, which the spot checks catch, so the
;; run falls back, failing this test (and the Map's +Inf stages,
;; gen.combinators-test); the mask never read is an unread batched
;; comparison, which fails every batched run (gen.batched's guard).
(deftest a-particle-outside-its-support-weighs-minus-infinity
  (let [m (gen [lo] (let [a (trace :a (d/uniform lo (+ lo 1.0)))] (trace :b (d/beta 0.5 1.0)) a))
        ;; the even particles' a leaves [0.5, 1.5], and their b is at 0
        own (mapv #(if (even? %) (cm/choicemap :a 0.25 :b 0.0) (cm/choicemap :a 0.75 :b 0.5)) (range 6))
        ps (:particles (b/generate m [0.0] cm/EMPTY 6 (rng/key 1) {:per-particle own}))
        {out :particles ws :weights} (b/update ps [0.5] [diff/unknown-change] cm/EMPTY (rng/key 2))
        ws (array/->clj ws)]
    (is (b/batched? out) "it stays batched")
    (is (every? #(= ##-Inf %) (take-nth 2 ws)) "a at 0.25 is outside [0.5, 1.5]")
    (is (every? #(close? 0.0 %) (take-nth 2 (rest ws))) "a at 0.75 is inside")
    (is (every? #(= ##-Inf %) (take-nth 2 (array/->clj (b/scores out)))))
    (is (every? true? (map #(= (ts/number %1) (ts/number (:weight (gfi/update %2 [0.5] [diff/unknown-change] cm/EMPTY (rng/key 3)))))
                           ws (b/traces ps)))
        "each particle's weight is its scalar update's")
    (when (ts/jvm?)
      ;; (a lazy backend's scalar run goes on, gen.dynamic)
      (is (every? #(= [[:a]] (cm/addresses (:choices %))) (take-nth 2 (b/traces out)))
          "a particle outside is rebuilt as its scalar trace, which stops at a")))
  (testing "a parameter check downstream that fails for such a particle falls back to one at a time"
    (let [m (gen [] (let [s (trace :s (d/gamma 1 1))] (trace :y (d/normal 0 s))))
          ps (:particles (b/generate m [] (cm/choicemap :y 0.1) 4 (rng/key 4)))
          own (mapv #(cm/choicemap :s (if (even? %) -1.0 0.5)) (range 4))
          {out :particles ws :weights} (b/update ps [] [] cm/EMPTY (rng/key 5) {:per-particle own})]
      ;; (a lazy backend checks no parameter it would have to read)
      (when (ts/jvm?)
        (is (not (b/batched? out)))
        (is (re-find #"std must be positive" (:message (b/fallback-reason out)))))
      (is (= [##-Inf ##-Inf] (vec (take-nth 2 (array/->clj ws)))))
      (is (every? ts/finite? (take-nth 2 (rest (array/->clj ws)))))))
  (testing "MH chains reject the particles' moves outside the support"
    (let [m (gen [] (let [a (trace :a (d/uniform 0 1)) b (trace :b (d/uniform 0 a))] (trace :c (d/normal b 1.0)) b))
          ps (:particles (b/generate m [] (cm/choicemap :a 0.9 :b 0.6 :c 0.0) 16 (rng/key 6)))
          {out :particles acc :accepted? la :log-alpha} (mh/mh-particles ps (sel/select :a) (rng/key 7))
          la (array/->clj la)]
      (is (b/batched? out))
      (is (some #(= ##-Inf %) la) "some a below b")
      (is (every? (fn [[l a?]] (or (not= ##-Inf l) (not a?))) (map vector la acc)))
      (is (every? #(< (ts/number (% :b)) (ts/number (% :a))) (b/traces out))))))

;; ---------------------------------------------------------------------------
;; A fallback is recorded, and :batched? :required refuses it (DESIGN.md §9,
;; the ten decisions' ninth; genclj-ynpc)
;; ---------------------------------------------------------------------------

;; Every site that runs particles batched and may fall back to one at a
;; time records why in its result, :fallback, a written message (never a
;; raw exception's), and under :batched? :required throws ex-info with it
;; instead. Nothing prints. Planted bugs (2026-10-25): fallen-back
;; recording no message fails 17 of the checks here; batched-or ignoring
;; *required* fails the :required checks of the sites that run through it
;; (8); host math's reason left raw fails its check.

(def host-math
  "Host math on a choice: Math/exp takes a batched value as one number,
   which Java's cast refuses (029's report)."
  (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (Math/exp x) 1)))))

(defn- written? [m] (and (string? m) (not (re-find #"class |cannot be cast" m))))

(defn- refused? [f]
  (try (f) false
       (catch clojure.lang.ExceptionInfo e
         (boolean (and (re-find #"cannot be batched \(:batched\? :required\)" (ex-message e)) (written? (:fallback (ex-data e))))))))

(deftest a-fallback-is-recorded-and-required-refuses-it
  (testing "gen.batched's operations"
    (let [ps (b/simulate branching [] 8 (rng/key 1))
          {gps :particles :as g} (b/generate branching [] cm/EMPTY 8 (rng/key 2))
          u (b/update gps [] [] (cm/choicemap :switch true) (rng/key 3))
          r (b/regenerate gps [] [] (sel/select :switch) (rng/key 4))]
      (is (re-find #"Cannot branch on a batched value" (b/fallback ps)))
      (is (= (b/fallback gps) (:fallback g)))
      (is (= (b/fallback gps) (:fallback u) (:fallback r)) "carried on by the operations after it")
      (is (= {:message (b/fallback ps)} (select-keys (b/fallback-reason ps) [:message])))
      (is (nil? (b/fallback (b/simulate (gen [] (trace :x (d/normal 0 1))) [] 8 (rng/key 5)))) "nil while batched")
      (binding [b/*required* true]
        (is (refused? #(b/simulate branching [] 8 (rng/key 1))))
        (is (refused? #(b/generate branching [] cm/EMPTY 8 (rng/key 2))))
        (is (refused? #(b/update gps [] [] cm/EMPTY (rng/key 3))) "particles held one at a time")
        (is (refused? #(b/map-particles gps (fn [t _] (:score t)) nil))))))
  (testing "host math's reason is written (029)"
    (let [m (b/fallback (b/simulate host-math [] 4 (rng/key 6)))]
      (is (written? m))
      (is (re-find #"host math" m))))
  (testing "the batched generate under an internal proposal (028)"
    (let [m (gen [] (let [z (trace :z (d/normal 0 1))] (trace :y (d/normal z 1))))
          q (gen [_cs] (trace :z (d/normal 0.5 1)))
          {ps :particles g :fallback} (b/generate (proposal/with-internal-proposal m q) [] (cm/choicemap :y 0.3) 6 (rng/key 7))]
      (is (= g (b/fallback ps)))
      (is (written? g))))
  (testing "importance sampling"
    (let [obs cm/EMPTY]
      (doseq [[label f] [["importance-sampling" #(imp/importance-sampling branching [] obs 8 %)]
                         ["importance-resampling" #(imp/importance-resampling branching [] obs 8 %)]
                         ["estimate" #(imp/estimate (fn [t] (if (ts/host (t :switch)) 1.0 0.0)) branching [] obs 8 %)]]]
        (testing label
          (is (re-find #"Cannot branch" (:fallback (f {:batched? true :key (rng/key 8)}))))
          (is (refused? #(f {:batched? :required :key (rng/key 8)})))
          (is (nil? (:fallback (f {:key (rng/key 8)}))) "one at a time by request records nothing"))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"a custom proposal runs one sample at a time"
                          (imp/importance-sampling branching [] cm/EMPTY 4 {:batched? :required :proposal (gen [] nil)})))
    (is (nil? (:fallback (imp/importance-sampling (gen [] (trace :x (d/normal 0 1))) [] cm/EMPTY 8 {:batched? :required :key (rng/key 9)})))
        "a model that runs batched passes :required"))
  (testing "the particle filter: initialize, and a step that branches on particles"
    (let [m (gen [t] (let [x (trace :x (d/normal 0 1))]
                       (when (> t 1) (if (trace :b (d/bernoulli 0.5)) (trace :c (d/normal x 1)) (trace :d (d/normal 0 1))))))
          s0 (pf/initialize m [1] cm/EMPTY 6 {:batched? true :key (rng/key 10)})
          s1 (pf/step s0 [2] [diff/unknown-change] cm/EMPTY {:key (rng/key 11)})]
      (is (nil? (:fallback s0)))
      (is (re-find #"Cannot branch" (:fallback s1)))
      (is (re-find #"Cannot branch" (:fallback (pf/initialize branching [] cm/EMPTY 6 {:batched? true :key (rng/key 12)})))))
    (let [m (gen [t] (let [x (trace :x (d/normal 0 1))]
                       (when (> t 1) (if (trace :b (d/bernoulli 0.5)) (trace :c (d/normal x 1)) (trace :d (d/normal 0 1))))))
          s0 (pf/initialize m [1] cm/EMPTY 6 {:batched? :required :key (rng/key 10)})]
      (is (= :required (:batched? s0)))
      (is (refused? #(pf/step s0 [2] [diff/unknown-change] cm/EMPTY {:key (rng/key 11)})))
      (is (refused? #(pf/initialize branching [] cm/EMPTY 6 {:batched? :required :key (rng/key 12)})))))
  (testing "MH, HMC and MAP on particles, gradients, VI, enumeration"
    (let [m (gen [] (let [x (trace :x (d/normal 0 1))] (if (array/> x 0.0) (trace :y (d/normal x 1)) (trace :y (d/normal (- x) 1)))))
          ps (b/simulate (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x 1)))) [] 6 (rng/key 13))
          bps (b/simulate m [] 6 (rng/key 13))]
      (is (written? (:fallback (mh/mh-particles bps (sel/select :x) (rng/key 14)))))
      (is (nil? (:fallback (mh/mh-particles ps (sel/select :x) (rng/key 14)))))
      (is (written? (:fallback (mcmc/hmc-particles bps (sel/select :x) {:eps 0.1 :L 2 :key (rng/key 15)}))))
      (is (written? (b/fallback (mcmc/map-optimize-particles bps (sel/select :x)))))
      (is (written? (:fallback (grad/choice-gradients bps (sel/select :x)))))
      (is (nil? (:fallback (grad/choice-gradients ps (sel/select :x)))))
      (let [q (gen [] (trace :x (d/normal (param :m) 1.0)))]
        (is (written? (:fallback (vi/elbo-gradient m [] (cm/choicemap :y 0.5) q [] {:m 0.0} 4 {:batched? true :key (rng/key 16)}))))
        (is (refused? #(vi/elbo-gradient m [] (cm/choicemap :y 0.5) q [] {:m 0.0} 4 {:batched? :required :key (rng/key 16)}))))
      (let [grid (enum/choice-vol-grid [:switch [true false]])]
        (is (re-find #"Cannot branch" (:fallback (enum/enumerative-inference branching [] cm/EMPTY grid {:batched? true :key (rng/key 17)}))))
        (is (refused? #(enum/enumerative-inference branching [] cm/EMPTY grid {:batched? :required :key (rng/key 17)})))))))
