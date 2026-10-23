(ns gen.inference.diagnostics-test
  "MCMC diagnostics against ArviZ 1.3 (arviz_stats 1.3.3) on fixed draws
   (test/fixtures/diagnostics.edn, written by test/fixtures/generate.py),
   and against what is known without it: the ESS of long AR(1) chains, the
   split R-hat of chains with shifted means, and the invariances of the
   rank-based diagnostics. The diagnostics are host math, so these run the
   same on every backend; only draws reads backend arrays."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as dg]
            [gen.inference.kernel :as k]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(def ^:private fixtures (edn/read-string (slurp "test/fixtures/diagnostics.edn")))

(defn- fixture [name] (first (filter #(= name (:name %)) fixtures)))

(defn- agrees?
  "Equal within 1e-12 relative, NaN and the infinities exactly: ArviZ's
   FFT and these direct sums round differently (1.2e-14 apart at most,
   measured)."
  [expected actual]
  (let [expected (double expected) actual (double actual)]
    (or (== expected actual)
        (and (Double/isNaN expected) (Double/isNaN actual))
        (<= (Math/abs (- expected actual)) (* 1e-12 (Math/abs expected))))))

(def ^:private diagnostics
  {:ess-bulk dg/ess-bulk :ess-tail dg/ess-tail :ess-mean dg/ess-mean :r-hat dg/r-hat
   :split-r-hat dg/split-r-hat :mcse-mean dg/mcse-mean :mcse-sd dg/mcse-sd})

(deftest arviz-reference-values
  ;; AR(1) chains with positive and negative autocorrelation, shifted means
  ;; and a trend (R-hat above 1), an odd length, short chains, the minimum
  ;; of 4 draws (the cap), Cauchy draws, one chain, binary draws (ties, and
  ;; a constant 95% indicator), draws whose fold is constant, constant
  ;; draws and chains stuck at different values
  (doseq [{:keys [name draws quantiles] :as c} fixtures]
    (testing name
      (doseq [[k f] diagnostics]
        (is (agrees? (c k) (f draws)) (str k ": ArviZ " (c k) ", here " (f draws))))
      (let [s (dg/summary draws)]
        (doseq [k [:mean :sd :ess-bulk :ess-tail :r-hat :mcse-mean :mcse-sd]]
          (is (agrees? (c k) (s k)) (str "summary " k)))
        (is (= (map first quantiles) (keys (:quantiles s))))
        (doseq [[p q] quantiles]
          (is (agrees? q (get-in s [:quantiles p])) (str "quantile " p)))))))

(deftest edges-as-arviz
  (testing "constant draws: the nominal ESS, and R-hat 0/0"
    (let [draws [(repeat 20 1.5) (repeat 20 1.5)]]
      (is (= 40.0 (dg/ess-bulk draws) (dg/ess-tail draws) (dg/ess-mean draws)))
      (is (Double/isNaN (dg/r-hat draws)))
      (is (= 0.0 (dg/mcse-mean draws)))))
  (testing "chains stuck at different values: R-hat +Inf"
    (is (= ##Inf (dg/r-hat [(repeat 20 0.0) (repeat 20 1.0)]))))
  (testing "one chain: R-hat of its halves, and a vector of numbers is one chain"
    (let [chain (first (:draws (fixture "ar(1) 0.5")))]
      (is (Double/isFinite (dg/r-hat [chain])))
      (doseq [[k f] diagnostics]
        (is (= (f [chain]) (f chain)) (str k))))))

;; ---------------------------------------------------------------------------
;; Without ArviZ
;; ---------------------------------------------------------------------------

(def ^:private jvm (array/backend-named :jvm))

(defn- ar1
  "m chains of n draws of the AR(1) process x_t = ρ·x_{t-1} + e_t, e_t
   standard normal, from its stationary distribution N(0, 1/(1 - ρ²)). The
   innovations are the JVM backend's, so the chains are the same whatever
   the default backend."
  [rho m n key]
  (let [rho (double rho)]
    (mapv (fn [es]
            (vec (reductions (fn [x e] (+ (* rho (double x)) (double e)))
                             (/ (double (first es)) (Math/sqrt (- 1.0 (* rho rho))))
                             (rest es))))
          (array/->clj (rng/normal key [m n] {:backend jvm})))))

(defn- mean-sd [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)]
    [m (Math/sqrt (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n)))]))

(deftest ar1-ess-is-known
  ;; The ESS of n draws of an AR(1) chain is n(1 - ρ)/(1 + ρ) as n grows,
  ;; for the mean, and for the bulk too: rank normalization leaves a
  ;; Gaussian chain Gaussian. 16 runs of 4 chains of 5,000 draws for each
  ;; ρ; the runs' mean ESS per draw within 5 standard errors of it, the
  ;; standard error from the runs' spread. Measured: bands of 4.4% of the
  ;; ESS at ρ = 0.5, 13% at 0.9 and 4.1% at -0.3, the means 0.06 to 0.22
  ;; of a band away
  (doseq [[rho seed] [[0.5 1] [0.9 2] [-0.3 3]]]
    (let [runs 16 m 4 n 5000
          expected (/ (- 1.0 rho) (+ 1.0 rho))
          chains (mapv #(ar1 rho m n %) (rng/split (rng/key seed) runs))]
      (doseq [[label f] [["ess-mean" dg/ess-mean] ["ess-bulk" dg/ess-bulk]]]
        (let [[mean sd] (mean-sd (map #(/ (f %) (* m n)) chains))]
          (is (< (Math/abs (- mean expected)) (* 5 (/ sd (Math/sqrt runs))))
              (str label " at ρ = " rho ": " mean " per draw, expected " expected
                   ", band " (* 5 (/ sd (Math/sqrt runs))))))))))

(deftest split-r-hat-of-shifted-chains-is-known
  ;; Chains of independent N(μ_c, 1) draws: the split R-hat's square is
  ;; (h - 1)/h + B/(h·W) for split chains of h draws, which tends to 1 + v,
  ;; v the variance (n - 1 denominator) of the split chains' μ. Means
  ;; [0 0 1 1] split to [0 0 0 0 1 1 1 1]: v = 2/7. 16 runs of 4 chains of
  ;; 2,000 draws; their mean within 5 standard errors (measured: 0.15 and
  ;; 0.36 of a band away, the bands 0.0072 and 0.00026)
  (doseq [[offsets expected seed] [[[0.0 0.0 1.0 1.0] (Math/sqrt (+ 1.0 (/ 2.0 7.0))) 4]
                                   [[0.0 0.0 0.0 0.0] 1.0 5]]]
    (let [runs 16 n 2000
          r-hats (for [key (rng/split (rng/key seed) runs)]
                   (dg/split-r-hat (mapv (fn [mu chain] (mapv #(+ mu %) chain))
                                         offsets (ar1 0.0 4 n key))))
          [mean sd] (mean-sd r-hats)]
      (is (< (Math/abs (- mean expected)) (* 5 (/ sd (Math/sqrt runs))))
          (str "means " offsets ": " mean ", expected " expected)))))

(deftest rank-diagnostics-are-invariant-to-monotone-maps
  ;; ranks, and which draws lie below a quantile, are the same after any
  ;; increasing map
  (let [draws (:draws (fixture "ar(1) 0.5"))
        exp-draws (mapv #(mapv (fn [x] (Math/exp x)) %) draws)]
    (is (= (dg/ess-bulk draws) (dg/ess-bulk exp-draws)))
    (is (= (dg/ess-tail draws) (dg/ess-tail exp-draws)))
    (is (not= (dg/ess-mean draws) (dg/ess-mean exp-draws)) "the mean's ESS is not rank-based")))

(deftest chain-order-does-not-matter
  (let [draws (:draws (fixture "shifted means"))
        reordered (vec (reverse draws))]
    (doseq [[k f] diagnostics]
      (is (agrees? (f draws) (f reordered)) (str k)))))

;; ---------------------------------------------------------------------------
;; Invalid draws, and draws from traces
;; ---------------------------------------------------------------------------

(deftest invalid-draws-are-loud
  (let [reason #(:reason (ts/thrown-data (dg/ess-bulk %)))]
    (is (= :invalid-draws (reason [])))
    (is (= :invalid-draws (reason 1.0)))
    (is (= :invalid-draws (reason [[1 2 3 4] 5])))
    (is (= :non-numeric (reason [1 2 nil 4])))
    (is (= :non-numeric (reason [[1 2 3 4] [1 2 true 4]])))
    (is (= :non-finite (reason [[1 2 3 4] [1 2 ##NaN 4]])))
    (is (= {:reason :non-finite :chain 0 :draw 1 :value ##-Inf}
           (ts/thrown-data (dg/r-hat [1 ##-Inf 3 4]))))
    (is (= :unequal-chains (reason [[1 2 3 4] [1 2 3 4 5]])))
    (is (= {:reason :too-few-draws :draws 3 :minimum 4} (ts/thrown-data (dg/ess-tail [[1 2 3] [4 5 6]]))))
    (is (= :invalid-prob (:reason (ts/thrown-data (dg/summary [1 2 3 4] {:probs [0.5 1.5]})))))))

(deftest summary-quantiles
  (let [draws [[1.0 2.0 3.0 4.0] [5.0 6.0 7.0 8.0]]]
    (is (= {0.0 1.0 0.25 2.75 0.5 4.5 1.0 8.0}
           (:quantiles (dg/summary draws {:probs [0.5 0.0 1.0 0.25]}))))
    (is (= [0.05 0.5 0.95] (keys (:quantiles (dg/summary draws)))))))

(def ^:private model
  (gen [] [(trace :x (d/normal 0 1)) (trace :b (d/bernoulli 0.3))]))

(deftest draws-from-traces
  (let [runs (mapv (fn [key]
                     (let [[k1 k2] (rng/split key)]
                       (k/run (k/mh (sel/select :x :b)) (gfi/simulate model [] k1) {:samples 5 :key k2})))
                   (rng/split (rng/key 6) 3))
        chains (map :traces runs)
        xs (dg/draws chains :x)]
    (is (= (mapv (fn [c] (mapv #(double (ts/host (% :x))) c)) chains) xs))
    (is (= (first xs) (dg/draws (first chains) :x)) "one chain of traces gives one chain of draws")
    (is (= (mapv (fn [c] (mapv #(if (ts/host (% :b)) 1.0 0.0) c)) chains) (dg/draws chains :b))
        "booleans as 0 and 1")
    (is (Double/isFinite (dg/ess-bulk xs)))
    (is (= :non-scalar (:reason (ts/thrown-data (dg/draws chains [])))) "the empty path: the return value, a vector")
    (is (= :invalid-draws (:reason (ts/thrown-data (dg/draws [[1.0]] :x)))))
    (is (= :invalid-draws (:reason (ts/thrown-data (dg/draws (ffirst chains) :x)))) "a trace, not a seq of them")))
