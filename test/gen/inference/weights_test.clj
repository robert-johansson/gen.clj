(ns gen.inference.weights-test
  (:require [gen.test-support :as ts]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as a]
            [gen.inference.weights :as w]
            [gen.random :as rng]))

(defn- close? [x y] (< (Math/abs (- (ts/number x) (ts/number y))) 1e-12))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (::w/invalid-weights (ex-data e)))))

(deftest normalization
  (let [{:keys [log-total log-weights]} (w/normalize [0.0 (Math/log 3.0)])]
    (is (close? (Math/log 4.0) log-total))
    (is (close? (Math/log 0.25) (first log-weights)))
    (is (close? (Math/log 0.75) (second log-weights))))
  (testing "shifting every weight changes only the total"
    (let [ws [-1.0 2.5 0.3]
          shifted (mapv #(+ % 1000.0) ws)]
      (is (every? true? (map close? (:log-weights (w/normalize ws)) (:log-weights (w/normalize shifted)))))
      (is (close? (+ 1000.0 (:log-total (w/normalize ws))) (:log-total (w/normalize shifted))))))
  (testing "arrays and seqs give the same result"
    (is (= (w/normalize [0.5 -0.5 ##-Inf]) (w/normalize (a/array [0.5 -0.5 ##-Inf]))))))

(deftest log-mean-exp-and-ess
  (is (close? 0.0 (w/log-mean-exp [0 0 0])))
  (is (close? (Math/log 2.0) (w/log-mean-exp [0.0 (Math/log 3.0)])))
  (is (close? 4.0 (w/ess [0 0 0 0])) "equal weights: n")
  (is (close? 1.0 (w/ess [0.0 ##-Inf ##-Inf])) "one sample carries all the weight: 1")
  (is (close? (w/ess [0.0 1.0 2.0]) (w/ess (:log-weights (w/normalize [0.0 1.0 2.0]))))
      "normalized or not"))

(deftest invalid-weights-fail-loudly
  (is (= :empty (reason #(w/normalize []))))
  (is (= :impossible (reason #(w/normalize [##-Inf ##-Inf]))))
  (is (= :nan (reason #(w/ess [0.0 ##NaN]))))
  (is (= :infinite (reason #(w/log-mean-exp [0.0 ##Inf]))))
  (is (= :impossible (reason #(w/systematic [##-Inf] 3 (rng/key 0)))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Unknown resampling method"
                        (w/resample :bogus [0.0] 1 (rng/key 0)))))

;; ---------------------------------------------------------------------------
;; Resampling properties
;; ---------------------------------------------------------------------------

(def gen-log-weights
  "1 to 12 log weights, some -Inf, at least one finite."
  (gen/such-that #(some (fn [x] (> x ##-Inf)) %)
                 (gen/vector (gen/frequency [[4 (gen/double* {:min -5.0 :max 5.0 :NaN? false :infinite? false})]
                                             [1 (gen/return ##-Inf)]])
                             1 12)))

(def gen-case (gen/tuple gen-log-weights (gen/choose 0 40) gen/nat))

(defn- probs [lws] (mapv #(Math/exp %) (:log-weights (w/normalize lws))))

(defn- counts [n-samples indices]
  (let [f (frequencies indices)] (mapv #(get f % 0) (range n-samples))))

(defspec every-resampler-draws-n-possible-indices 200
  (prop/for-all [[lws n seed] gen-case]
    (every? (fn [method]
              (let [is (w/resample method lws n (rng/key seed))]
                (and (= n (count is))
                     (every? #(and (<= 0 % (dec (count lws))) (> (lws %) ##-Inf)) is))))
            [:multinomial :systematic :stratified :residual])))

(defspec systematic-counts-are-floor-or-ceil 200
  (prop/for-all [[lws n seed] gen-case]
    (every? (fn [[c p]] (<= (Math/floor (- (* n p) 1e-9)) c (Math/ceil (+ (* n p) 1e-9))))
            (map vector (counts (count lws) (w/systematic lws n (rng/key seed))) (probs lws)))))

(defspec residual-keeps-the-whole-copies 200
  (prop/for-all [[lws n seed] gen-case]
    (every? (fn [[c p]] (>= c (Math/floor (- (* n p) 1e-9))))
            (map vector (counts (count lws) (w/residual lws n (rng/key seed))) (probs lws)))))

(defspec stratified-counts-are-within-two 200
  (prop/for-all [[lws n seed] gen-case]
    (every? (fn [[c p]] (< (Math/abs (- c (* n p))) 2.0))
            (map vector (counts (count lws) (w/stratified lws n (rng/key seed))) (probs lws)))))

(deftest resamplers-are-unbiased
  ;; fixed keys, so deterministic; band: five multinomial standard errors,
  ;; an upper bound for the lower-variance schemes
  ;; fractional n·p_i, so that no scheme is deterministic
  (let [ps [0.13 0.27 0.05 0.55 0.0]
        lws (mapv #(Math/log %) ps)
        n 10 runs 4000]
    (doseq [method [:multinomial :systematic :stratified :residual]]
      (let [total (reduce #(mapv + %1 %2)
                          (map #(counts 5 (w/resample method lws n %)) (rng/split (rng/key 17) runs)))]
        (doseq [[i p] (map-indexed vector ps)]
          (let [mean (/ (total i) (double runs))
                se (Math/sqrt (/ (* n p (- 1 p)) runs))]
            (is (<= (Math/abs (- mean (* n p))) (max (* 5 se) 1e-12))
                (str method " index " i ": mean count " mean ", expected " (* n p)))))))))
