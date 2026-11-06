(ns gen.llm.awrs-test
  "Adaptive weighted rejection sampling (gen.llm.awrs) on rows of six
   tokens, where Σ_t p(t)·1[accepted]·f(t) is a sum.

   - Proper weighting: for each estimator (recursive, geometric with 2 and
     3 accepts), unbounded and under budgets of rejections, the mean of
     Ẑ·exp(w(x))·f(x) over 4,000 keys is within 5 standard errors of
     Σ_t p(t)·exp(w(t))·f(t) over the accepted tokens, for f each token's
     indicator and 1 (so Ẑ is unbiased for Z, and the token's distribution
     given its weight is the restricted row's), for constraints that accept
     most of the mass, a rare token, every token, and a soft weight.
   - Nothing accepted: Ẑ = 0 at every key. Everything accepted: the
     recursive estimate is 1 exactly, with two checks.
   - Budgets: at most R + 1 checks (recursive), R + max-accepts
     (geometric); determinism by the key; misuse is loud.

   Planted bugs (2026-11-05, each applied, the suite run, reverted): the
   recursive estimate m_n where the extra draw is rejected (in place of
   p(x_n)); the geometric estimate (s - 1)/(n + 1) (a reading of the
   paper's Def. 8 that is biased); phantoms not counted. Each fails the
   proper-weighting test."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.llm.awrs :as awrs]
            [gen.llm.branching-contract :as bc]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def ^:private row
  "Six tokens' log-probabilities, normalized in double then stored as
   floats."
  (let [ps [0.35 0.25 0.15 0.12 0.08 0.05]]
    (float-array (map #(Math/log %) ps))))

(def ^:private whole
  "The row's mass, 1 up to its floats' rounding, which the sampler's
   shares are of."
  (reduce + (map #(Math/exp %) row)))

(defn- p ^double [^long t] (/ (Math/exp (aget ^floats row t)) (double whole)))

(def ^:private conditions
  "Accepted tokens, and a log weight for each."
  {:most {:accepted #{0 2 3} :w (constantly 0.0)}
   :rare {:accepted #{5} :w (constantly 0.0)}
   :all {:accepted (set (range 6)) :w (constantly 0.0)}
   :soft {:accepted #{1 2 4 5} :w (fn [t] (* 0.4 (double t)))}})

(defn- weigh-of [{:keys [accepted w]}]
  (fn [t] (if (accepted t) (w t) ##-Inf)))

(def ^:private estimators
  {:recursive {}
   :recursive-budget-1 {:max-rejects 1}
   :recursive-budget-2 {:max-rejects 2}
   :geometric {:method :geometric}
   :geometric-3-accepts {:method :geometric :max-accepts 3}
   :geometric-budget-2 {:method :geometric :max-rejects 2}})

;; the battery's sample statistics: a mean and its standard error
(def ^:private mean ts/mean)
(def ^:private se ts/se)

(def ^:private n-keys 4000)

(defn- runs [condition opts]
  (let [weigh (weigh-of condition)]
    (mapv #(awrs/sample row weigh (rng/key %) opts) (range n-keys))))

(deftest the-estimates-are-properly-weighted
  (doseq [[cname condition] conditions
          [ename opts] estimators
          :let [rs (runs condition opts)
                {:keys [accepted w]} condition]
          f (cons nil (range 6))]
    (let [f-of (fn [t] (if (nil? f) 1.0 (if (= f t) 1.0 0.0)))
          exact (reduce + (for [t accepted] (* (p t) (Math/exp (w t)) (f-of t))))
          xs (mapv (fn [{:keys [token weight log-z]}]
                     (if (= ##-Inf log-z) 0.0 (* (Math/exp log-z) (Math/exp weight) (f-of token))))
                   rs)
          band (max (* 5 (se xs)) 1e-12)]
      (is (<= (Math/abs (- (mean xs) exact)) band)
          (str cname " " ename " f " (or f "1") ": mean " (mean xs) ", exact " exact ", band " band)))))

(deftest edge-cases-are-exact
  (testing "nothing accepted: Ẑ = 0"
    (doseq [[ename opts] estimators]
      (is (every? #(= ##-Inf (:log-z %)) (runs {:accepted #{} :w (constantly 0.0)} opts)) ename)))
  (testing "everything accepted: the recursive estimate is 1, after two checks"
    (is (every? #(and (< (Math/abs (:log-z %)) 1e-6) (= 2 (:checks %))) (runs (:all conditions) {})))))

(deftest budgets-bound-the-checks
  (doseq [[opts most] [[{:max-rejects 1} 1] [{:max-rejects 3} 4]
                       [{:method :geometric :max-rejects 2} 4] [{:method :geometric :max-rejects 2 :max-accepts 3} 5]]]
    (is (every? #(<= (:checks %) most) (runs (:rare conditions) opts)) (pr-str opts))))

(deftest runs-are-pure-functions-of-the-key
  (doseq [[_ opts] estimators]
    (is (= (runs (:soft conditions) opts) (runs (:soft conditions) opts)))))

(deftest misuse-is-loud
  (let [weigh (weigh-of (:most conditions))]
    (doseq [[what f reason] [["a NaN weight" #(awrs/sample row (constantly ##NaN) (rng/key 0)) :invalid-weight]
                             ["no row" #(awrs/sample [0.0] weigh (rng/key 0)) :invalid-row]
                             ["an empty row" #(awrs/sample (float-array 0) weigh (rng/key 0)) :invalid-row]
                             ["an unknown method" #(awrs/sample row weigh (rng/key 0) {:method :best}) :invalid-option]
                             ["no rejections allowed" #(awrs/sample row weigh (rng/key 0) {:max-rejects 0}) :invalid-option]
                             ["one accept" #(awrs/sample row weigh (rng/key 0) {:method :geometric :max-accepts 1}) :invalid-option]
                             ["accepts for the recursive" #(awrs/sample row weigh (rng/key 0) {:max-accepts 2}) :invalid-option]]]
      (is (= reason (bc/reason f)) what))))
