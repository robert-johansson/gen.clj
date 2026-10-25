(ns gen.state-space-test
  "The Kalman filter's and the forward algorithm's generative functions:
   their marginal likelihoods against answers computed without a filter
   (test/fixtures/state-space.edn: the joint Gaussian of the whole series,
   and every path of states enumerated), a particle filter's estimate and
   an enumeration of the model with its states as choices; incremental
   updates; the GFI laws; and forward filtering, backward sampling against
   the smoothing marginals."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.particle-filter :as pf]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.state-space :as ss]
            [gen.test-support :as ts]
            [gen.trace :as trace]))

(def ^:private fixtures (edn/read-string (slurp "test/fixtures/state-space.edn")))

(defn- fixture [kind name] (first (filter #(= name (:name %)) (kind fixtures))))

(defn- value
  "A fixture's number, or its vector as an array (constraints are arrays)."
  [x]
  (if (number? x) x (array/array x)))

(defn- observations [ys] (apply cm/choicemap (mapcat (fn [t y] [t (value y)]) (range) ys)))

;; ---------------------------------------------------------------------------
;; The models of the fixtures (test/fixtures/generate.py's state_space)
;; ---------------------------------------------------------------------------

(def kalman-models
  {:local-level [{:mean 0.5 :cov 2.0} {:transition 1.0 :transition-cov 0.5 :emission 1.0 :emission-cov 1.0}]
   :offsets [{:mean 1.0 :cov 0.5} {:transition 0.8 :transition-cov 0.3 :transition-offset 5.0
                                   :emission 2.0 :emission-cov 0.7 :emission-offset -1.0}]
   :constant-velocity [{:mean [0.0 1.0] :cov [[1.0 0.2] [0.2 0.5]]}
                       {:transition [[1.0 1.0] [0.0 1.0]] :transition-cov [[0.05 0.02] [0.02 0.1]]
                        :emission [1.0 0.0] :emission-cov 0.5}]
   :vector [{:mean [0.3 -0.2] :cov [[0.8 0.1] [0.1 0.6]]}
            {:transition [[0.9 0.1] [-0.1 0.9]] :transition-cov [[0.2 0.05] [0.05 0.1]] :transition-offset [0.2 -0.1]
             :emission [[1.0 0.5] [0.0 1.0]] :emission-cov [[0.5 0.1] [0.1 0.4]] :emission-offset [1.0 0.0]}]
   :time-varying [{:mean [0.0 0.0] :cov [[4.0 0.0] [0.0 4.0]]}
                  (let [xs (:xs (fixture :kalman :time-varying))]
                    (fn [t] {:transition [[1.0 0.0] [0.0 1.0]] :transition-cov [[0.01 0.0] [0.0 0.02]]
                             :emission [1.0 (xs t)] :emission-cov 0.25}))]})

(def ^:private a2 [[0.7 0.3] [0.2 0.8]])

(defn- hmm-model [{:keys [name B mus sigmas covs]}]
  (case name
    :categorical [[0.5 0.3 0.2] {:transition [[0.8 0.1 0.1] [0.2 0.6 0.2] [0.1 0.3 0.6]] :emission (d/categorical B)}]
    :normal [[0.6 0.4] {:transition a2 :emission (d/normal mus sigmas)}]
    :mvnormal [[0.6 0.4] {:transition a2 :emission (d/mvnormal mus covs)}]))

(defn- kalman-case [name]
  (let [{:keys [ys] :as f} (fixture :kalman name)
        [prior params] (kalman-models name)]
    (assoc f :args [(count ys) prior params] :obs (observations ys))))

(defn- hmm-case [name]
  (let [{:keys [ys] :as f} (fixture :hmm name)
        [prior params] (hmm-model f)]
    (assoc f :args [(count ys) prior params] :obs (observations ys))))

;; ---------------------------------------------------------------------------
;; Exact marginal likelihoods
;; ---------------------------------------------------------------------------

(deftest kalman-likelihoods-are-the-joint-gaussian-s
  (doseq [name (keys kalman-models)
          :let [{:keys [args obs log-likelihood smoothed-means smoothed-variances]} (kalman-case name)]]
    (testing (str name)
      (let [{:keys [trace weight]} (gfi/generate ss/kalman args obs (rng/key 1))]
        (is (laws/close? weight log-likelihood) "generate with every observation: the marginal likelihood")
        (is (laws/close? (:score trace) log-likelihood))
        (is (laws/close? (:weight (gfi/assess ss/kalman args obs)) log-likelihood))
        (is (laws/close? (:log-ml-estimate (imp/importance-sampling ss/kalman args obs 1 {:key (rng/key 2)}))
                         log-likelihood)
            "so importance sampling's estimate is exact with one sample")
        (testing "the last filtered belief is the last smoothing marginal"
          (let [{:keys [mean cov]} (peek (:retval trace))]
            (is (laws/close? mean (value (last smoothed-means))))
            (is (laws/close? (if (zero? (array/ndim cov)) cov (array/diag cov)) (value (last smoothed-variances))))))))))

(deftest hmm-likelihoods-are-the-sum-over-every-path
  (doseq [name [:categorical :normal :mvnormal]
          :let [{:keys [args obs log-likelihood smoothed]} (hmm-case name)]]
    (testing (str name)
      (let [{:keys [trace weight]} (gfi/generate ss/hmm args obs (rng/key 1))]
        (is (laws/close? weight log-likelihood))
        (is (laws/close? (:weight (gfi/assess ss/hmm args obs)) log-likelihood))
        (is (laws/close? (peek (:retval trace)) (value (last smoothed)))
            "the last filtered probabilities are the last smoothing marginal")))))

(deftest a-batched-run-is-the-same-filter
  (let [{:keys [args obs log-likelihood]} (kalman-case :vector)
        ws (:weights (b/generate ss/kalman args obs 20 (rng/key 3)))]
    (is (every? #(laws/close? % log-likelihood) (ts/host ws))))
  (let [{:keys [args obs log-likelihood]} (hmm-case :normal)
        ws (:weights (b/generate ss/hmm args obs 20 (rng/key 3)))]
    (is (every? #(laws/close? % log-likelihood) (ts/host ws)))))

;; ---------------------------------------------------------------------------
;; Incremental updates
;; ---------------------------------------------------------------------------

(deftest an-update-to-more-steps-adds-their-predictive-densities
  (doseq [[model {:keys [ys] [n prior params] :args}] [[ss/kalman (kalman-case :vector)] [ss/hmm (hmm-case :categorical)]]]
    (let [k (dec n)
          tr (:trace (gfi/generate model [k prior params] (observations (take k ys)) (rng/key 4)))
          {t2 :trace w :weight rd :retdiff disc :discard}
          (gfi/update tr [n prior params] [diff/unknown-change diff/no-change diff/no-change] (cm/choicemap k (value (ys k)))
                      (rng/key 5))
          full (:weight (gfi/assess model [n prior params] (observations ys)))]
      (is (laws/close? w (- full (:score tr))) "the weight is log p(y_n-1 | the earlier observations)")
      (is (laws/close? (:score t2) full))
      (is (cm/empty? disc))
      (is (= (diff/vector-diff n k {}) rd) "the earlier beliefs are unchanged")
      (is (every? #(identical? (trace/subtrace tr %) (trace/subtrace t2 %)) (range k))
          "no earlier step is revisited")))
  (testing "a changed observation revisits the steps after it, whose beliefs change"
    (let [{:keys [args ys obs]} (kalman-case :local-level)
          tr (:trace (gfi/generate ss/kalman args obs (rng/key 6)))
          {t2 :trace w :weight} (gfi/update tr (cm/choicemap 2 0.0))
          full (:weight (gfi/assess ss/kalman args (observations (assoc ys 2 0.0))))]
      (is (laws/close? w (- full (:score tr))))
      (is (identical? (trace/subtrace tr 1) (trace/subtrace t2 1)))
      (is (not (identical? (trace/subtrace tr 4) (trace/subtrace t2 4)))))))

;; ---------------------------------------------------------------------------
;; Against inference on the models with their states as choices
;; ---------------------------------------------------------------------------

(def local-level-step
  "The local-level model with its state traced: step t's state at [t :x]."
  (gen [t x {:keys [m0 p0 q r]}]
    (let [x (trace :x (d/normal (if (= t 0) m0 x) (if (= t 0) (array/sqrt p0) (array/sqrt q))))]
      (trace :y (d/normal x (array/sqrt r)))
      x)))

(deftest ^:slow a-particle-filter-s-estimate-is-unbiased-for-it
  ;; Ẑ is unbiased for p(ys), the Kalman generative function's density: the
  ;; mean of Ẑ/Z over independent filters is 1 within 5 standard errors
  (let [{:keys [args obs ys]} (kalman-case :local-level)
        log-likelihood (ts/number (:weight (gfi/assess ss/kalman args obs)))
        explicit (comb/unfold local-level-step)
        ps {:m0 0.5 :p0 2.0 :q 0.5 :r 1.0}
        n (count ys)
        estimate (fn [key]
                   (let [ks (rng/split key (* 2 n))
                         at (fn [t] (cm/choicemap [t :y] (ys t)))]
                     (pf/log-ml-estimate
                      (reduce (fn [state t]
                                (-> state
                                    (pf/maybe-resample {:key (ks (+ n t))})
                                    (pf/step [(inc t) 0.0 ps] [diff/unknown-change diff/no-change diff/no-change]
                                             (at t) {:key (ks t)})))
                              (pf/initialize explicit [1 0.0 ps] (at 0) 300 {:batched? true :key (ks 0)})
                              (range 1 n)))))
        ratios (mapv #(Math/exp (- (ts/number (estimate %)) log-likelihood)) (rng/split (rng/key 7) 40))
        mean (/ (reduce clojure.core/+ ratios) (count ratios))
        sd (Math/sqrt (/ (reduce clojure.core/+ (map #(Math/pow (clojure.core/- % mean) 2) ratios)) (dec (count ratios))))]
    (is (< 0.0 sd) "the estimates vary, so the band is not vacuous")
    (is (< (Math/abs (clojure.core/- mean 1.0)) (* 5 (/ sd (Math/sqrt (count ratios))))))))

(def hmm-step
  "The categorical HMM with its state traced: step t's state at [t :x]."
  (gen [t s {:keys [init a b]}]
    (let [x (trace :x (d/categorical (if (= t 0) init (array/take a s 0))))]
      (trace :y (d/categorical (array/take b x 0)))
      x)))

(deftest an-hmm-s-likelihood-is-the-enumerated-sum
  (let [{:keys [ys B] [n init params] :args} (hmm-case :categorical)
        explicit (comb/unfold hmm-step)
        ps {:init (array/array init) :a (array/array (:transition params)) :b (array/array B)}
        obs (apply cm/choicemap (mapcat (fn [t] [[t :y] (ys t)]) (range n)))
        grid (apply enum/choice-vol-grid (map (fn [t] [[t :x] [0 1 2]]) (range n)))
        {:keys [log-ml-estimate]} (enum/enumerative-inference explicit [n 0 ps] obs grid {:key (rng/key 8)})]
    (is (= 729 (count grid)))
    (is (laws/close? (:weight (gfi/assess ss/hmm [n init params] (observations ys))) log-ml-estimate))))

;; ---------------------------------------------------------------------------
;; Forward filtering, backward sampling
;; ---------------------------------------------------------------------------

(defn- within-5-se?
  "Is the sample mean of xs within 5 standard errors of mu, for draws of
   variance v?"
  [xs mu v]
  (< (Math/abs (clojure.core/- (/ (reduce clojure.core/+ xs) (count xs)) mu)) (* 5 (Math/sqrt (/ v (count xs))))))

(deftest ^:slow ^:heavy sampled-states-follow-the-smoothing-marginals
  (let [draws 2000]
    (doseq [name [:local-level :offsets :vector]
            :let [{:keys [args obs smoothed-means smoothed-variances]} (kalman-case name)
                  tr (:trace (gfi/generate ss/kalman args obs (rng/key 9)))
                  paths (mapv #(ts/host (ss/sample-states tr %)) (rng/split (rng/key 10) draws))]]
      (testing (str name)
        (is (= (count smoothed-means) (count (first paths))))
        (doseq [t (range (count smoothed-means))
                :let [mu (smoothed-means t) v (smoothed-variances t)]
                [i m var] (if (number? mu) [[nil mu v]] (map vector (range) mu v))]
          (is (within-5-se? (map #(if i (get-in % [t i]) (% t)) paths) m var) (str "x_" t (when i (str "[" i "]")))))))
    (doseq [name [:categorical :normal]
            :let [{:keys [args obs smoothed]} (hmm-case name)
                  tr (:trace (gfi/generate ss/hmm args obs (rng/key 9)))
                  paths (mapv #(ts/host (ss/sample-states tr %)) (rng/split (rng/key 11) draws))]]
      (testing (str name)
        (doseq [t (range (count smoothed))
                [s p] (map-indexed vector (smoothed t))]
          (is (within-5-se? (map #(if (= s (% t)) 1.0 0.0) paths) p (* p (clojure.core/- 1.0 p)))
              (str "P(x_" t " = " s ")")))))))

;; ---------------------------------------------------------------------------
;; The GFI laws (the full set: both are forward sampling)
;; ---------------------------------------------------------------------------

(def law-models
  (let [[ll-prior ll-params] (kalman-models :local-level)
        [v-prior v-params] (kalman-models :vector)
        [c-init c-params] (hmm-model (fixture :hmm :categorical))
        [n-init n-params] (hmm-model (fixture :hmm :normal))]
    [[:kalman-scalar ss/kalman [3 ll-prior ll-params]
      [[4 ll-prior ll-params] [2 ll-prior ll-params] [3 {:mean -1.0 :cov 0.5} (assoc ll-params :emission-cov 0.2)]]]
     [:kalman-vector ss/kalman [2 v-prior v-params] [[3 v-prior v-params] [2 v-prior (dissoc v-params :transition-offset)]]]
     [:kalman-time-varying ss/kalman [3 ll-prior (fn [t] (assoc ll-params :emission (+ 1.0 t)))] [[4 ll-prior ll-params]]]
     [:hmm-categorical ss/hmm [3 c-init c-params] [[4 c-init c-params] [2 [0.2 0.3 0.5] c-params]]]
     [:hmm-normal ss/hmm [3 n-init n-params] [[2 n-init n-params] [3 n-init (assoc n-params :transition [[0.5 0.5] [0.5 0.5]])]]]]))

(laws/deflaws state-space law-models 40 (merge laws/all laws/conventions))
(laws/deflaws state-space law-models 40 laws/fixed-support)

;; ---------------------------------------------------------------------------
;; Loud errors
;; ---------------------------------------------------------------------------

(deftest misuses-are-loud
  (let [[prior params] (kalman-models :local-level)
        [h-init h-params] (hmm-model (fixture :hmm :categorical))
        err (fn [re gf args] (is (thrown-with-msg? clojure.lang.ExceptionInfo re (gfi/simulate gf args (rng/key 1)))))]
    (err #"params lack \[:emission-cov\]" ss/kalman [2 prior (dissoc params :emission-cov)])
    (testing "a misspelled optional key, which would otherwise be an offset of 0 (genmlx-rmy7)"
      (err #"unknown keys \[:transition-ofset\]" ss/kalman [2 prior (assoc params :transition-ofset 5.0)]))
    (err #"does not fit \[2 2\]" ss/kalman [2 {:mean [0.0 0.0] :cov [1.0 1.0 1.0]} (second (kalman-models :vector))])
    (err #"takes \(t belief params\)" ss/kalman [2 prior])
    ;; checks that read values, which a lazy backend does not (as
    ;; distributions validate their parameters)
    (when (ts/jvm?)
      (err #"predictive variance must be positive" ss/kalman [2 prior (assoc params :emission-cov -3.0)])
      (err #"each row of :transition must be probabilities" ss/hmm
           [2 h-init (assoc h-params :transition [[0.5 0.6 0.1] [0.2 0.6 0.2] [0.1 0.3 0.6]])])
      (err #"the prior must be probabilities" ss/hmm [2 [0.5 0.6 0.1] h-params]))
    (err #"batched over the 3 states" ss/hmm [2 h-init (assoc h-params :emission (d/categorical [0.2 0.8]))])
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"takes a trace of gen.state-space/kalman or hmm"
                          (ss/sample-states (gfi/simulate (comb/unfold local-level-step) [2 0.0 {:m0 0 :p0 1 :q 1 :r 1}]
                                                          (rng/key 1)))))))
