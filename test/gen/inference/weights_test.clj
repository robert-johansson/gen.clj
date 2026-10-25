(ns gen.inference.weights-test
  (:require [gen.test-support :as ts]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
            [gen.inference.weights :as w]
            [gen.random :as rng]))

(defn- close? [x y] (< (Math/abs (- (ts/number x) (ts/number y))) 1e-12))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (::w/invalid-weights (ex-data e)))))

(deftest normalization
  (let [{:keys [log-total log-normalized-weights]} (w/normalize [0.0 (Math/log 3.0)])]
    (is (close? (Math/log 4.0) log-total))
    (is (close? (Math/log 0.25) (first log-normalized-weights)))
    (is (close? (Math/log 0.75) (second log-normalized-weights))))
  (testing "shifting every weight changes only the total"
    (let [ws [-1.0 2.5 0.3]
          shifted (mapv #(+ % 1000.0) ws)]
      (is (every? true? (map close? (:log-normalized-weights (w/normalize ws)) (:log-normalized-weights (w/normalize shifted)))))
      (is (close? (+ 1000.0 (:log-total (w/normalize ws))) (:log-total (w/normalize shifted))))))
  (testing "arrays and seqs give the same result"
    (is (= (w/normalize [0.5 -0.5 ##-Inf]) (w/normalize (array/array [0.5 -0.5 ##-Inf]))))))

(deftest log-mean-exp-and-ess
  (is (close? 0.0 (w/log-mean-exp [0 0 0])))
  (is (close? (Math/log 2.0) (w/log-mean-exp [0.0 (Math/log 3.0)])))
  (is (close? 4.0 (w/ess [0 0 0 0])) "equal weights: n")
  (is (close? 1.0 (w/ess [0.0 ##-Inf ##-Inf])) "one sample carries all the weight: 1")
  (is (close? (w/ess [0.0 1.0 2.0]) (w/ess (:log-normalized-weights (w/normalize [0.0 1.0 2.0]))))
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

(defn- probs [lws] (mapv #(Math/exp %) (:log-normalized-weights (w/normalize lws))))

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

(deftest residual-rounds-a-whole-copy-up-within-its-tolerance
  ;; genclj-tx1h (the ten decisions' eighth, DESIGN.md §9): n·p_i within a
  ;; relative 1e-9 below an integer is that many copies. Planted bug
  ;; (2026-10-25): the floor alone fails 22 checks, the 10 equal weights
  ;; (which gave [1 4 5 8 2 7 5 2 4 0]), the case within the tolerance and
  ;; most equal-weight counts; three thirds, whose product is exact, and
  ;; the case outside pass either way.
  (testing "10 equal weights give each index once"
    (let [lws (vec (repeat 10 0.0))]
      (is (< (* 10 (Math/exp (first (:log-normalized-weights (w/normalize lws))))) 1.0) "n·p is just below 1")
      (doseq [k (range 5)]
        (is (= (range 10) (w/residual lws 10 (rng/key k)))))))
  (testing "three thirds"
    (is (= [0 1 2] (w/residual (vec (repeat 3 0.0)) 3 (rng/key 1)))))
  (testing "at the tolerance's edge: n·p_0 = 2(1 - e), relative gap e"
    (let [lws (fn [e] [(Math/log (* 0.5 (- 1 e))) (Math/log (* 0.5 (+ 1 e)))])]
      (is (= [0 0 1 1] (w/residual (lws 0.5e-9) 4 (rng/key 2))) "within it: two copies each, nothing drawn")
      (is (= [0 1 1] (take 3 (w/residual (lws 2e-9) 4 (rng/key 2)))) "outside it: one copy, then a draw")))
  (testing "the copies stay at most n"
    (doseq [m (range 1 40)]
      (let [lws (vec (repeat m 0.0))]
        (doseq [n [1 m (* 3 m) 7]]
          (is (= n (count (w/residual lws n (rng/key m)))))
          (when (zero? (mod n m))
            (is (= (sort (mapcat #(repeat (quot n m) %) (range m))) (sort (w/residual lws n (rng/key m))))
                (str m " equal weights, " n " draws: n/m copies each"))))))))

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
