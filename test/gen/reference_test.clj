(ns gen.reference-test
  "Reference log-densities from probcomp's Gen.clj and Gen.jl (as collected
   in GenMLX's gen_clj_compat_test, analytic and checked against Gen.jl and
   scipy), translated to Gen.jl's names and parameterizations. GenMLX ran
   them at float32 tolerances; the JVM backend is float64, so they hold here
   to 1e-9. Plus normalization: continuous densities integrate to 1 and
   discrete ones sum to 1."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as array]
            [gen.dist :as d]
            [gen.dist.core :as dc]))

(defn- lp [dist v] (double (array/item (dc/log-prob dist (if (vector? v) (array/array v) v)))))

(defn- close? [expected actual]
  (or (= expected actual)
      (< (Math/abs (- expected actual)) (* (ts/tol (ts/tol 1e-9)) (max 1.0 (Math/abs expected))))))

(def ^:private reference
  "[label distribution value expected-log-density]"
  [["normal(0,1) at 0.5" (d/normal 0 1) 0.5 -1.0439385332046727]
   ["normal(0,2) at 0.5" (d/normal 0 2) 0.5 -1.643335713764618]
   ["normal(0,2) at 0" (d/normal 0 2) 0 -1.612085713764618]
   ["beta(0.001,1) at 0.4" (d/beta 0.001 1) 0.4 -5.992380837839856]
   ["beta(1,0.001) at 0.4" (d/beta 1 0.001) 0.4 -6.397440480839912]
   ["gamma(0.001, scale 1) at 0.4" (d/gamma 0.001 1) 0.4 -6.391804444241573]
   ["gamma(1, scale 0.001) at 0.4" (d/gamma 1 0.001) 0.4 -393.0922447210179]
   ["exponential(2) at 2" (d/exponential 2.0) 2.0 -3.3068528194400546]
   ["exponential(2) at 3" (d/exponential 2.0) 3.0 -5.306852819440055]
   ["laplace(2,1) at 1" (d/laplace 2 1) 1 -1.6931471805599454]
   ["laplace(0,2) at 1" (d/laplace 0 2) 1 -1.8862943611198906]
   ["laplace(0,0.001) at 0.002" (d/laplace 0 0.001) 0.002 4.214608098422191]
   ["poisson(3) at 3" (d/poisson 3) 3 -1.495922603223724]
   ["poisson(1) at 0" (d/poisson 1) 0 -1.0]
   ["poisson(5) at 10" (d/poisson 5) 10 -4.010033448735005]
   ["poisson(0.1) at 0" (d/poisson 0.1) 0 -0.1]
   ["poisson(0.01) at 0" (d/poisson 0.01) 0 -0.01]
   ["categorical-logits uniform-3 at 0" (d/categorical-logits [0.0 0.0 0.0]) 0 -1.0986122886681098]
   ["categorical-logits uniform-3 at 2" (d/categorical-logits [0.0 0.0 0.0]) 2 -1.0986122886681098]
   ["categorical-logits peaked at 0" (d/categorical-logits [10.0 0.0 0.0]) 0 -0.00009079573746717529]
   ["categorical-logits peaked at 1" (d/categorical-logits [-1.0 2.0 -1.0]) 1 -0.0949229564209606]
   ["categorical-logits binary uniform at 1" (d/categorical-logits [0.0 0.0]) 1 -0.6931471805599453]
   ["categorical-logits uniform-10 at 5" (d/categorical-logits (vec (repeat 10 0.0))) 5 -2.302585092994046]
   ["dirichlet([1,1,1]) at [0.5,0.3,0.2]" (d/dirichlet [1.0 1.0 1.0]) [0.5 0.3 0.2] 0.6931471805599454]
   ["dirichlet([5,2,1]) at [0.7,0.2,0.1]" (d/dirichlet [5.0 2.0 1.0]) [0.7 0.2 0.1] 2.3109698425284577]
   ["dirichlet([1,1,1,1]) at uniform" (d/dirichlet [1.0 1.0 1.0 1.0]) [0.25 0.25 0.25 0.25] 1.791759469228055]
   ["dirichlet([1,2,3]) at [0.1,0.3,0.6]" (d/dirichlet [1.0 2.0 3.0]) [0.1 0.3 0.6] 1.8687205103641835]
   ["cauchy(0,1) at 0" (d/cauchy 0 1) 0 -1.1447298858494002]
   ["cauchy(0,1) at 5" (d/cauchy 0 1) 5 -4.4028264238708825]
   ["cauchy(2,0.5) at 0" (d/cauchy 2 0.5) 0 -3.284796049345671]
   ["cauchy(0,1) at 100" (d/cauchy 0 1) 100 -10.355170252825918]
   ["cauchy(5,2) at the mode" (d/cauchy 5 2) 5 -1.8378770664093453]
   ["cauchy(0,0.01) at 0" (d/cauchy 0 0.01) 0 3.460440300138691]
   ["inv-gamma(2,1) at 1" (d/inv-gamma 2 1) 1.0 -1.0]
   ["inv-gamma(3,2) at 0.5" (d/inv-gamma 3 2) 0.5 0.15888308335967238]
   ["inv-gamma(1,1) at 5" (d/inv-gamma 1 1) 5.0 -3.4188758248682003]
   ["inv-gamma(2,1) at 0.01" (d/inv-gamma 2 1) 0.01 -86.18448944203573]
   ["inv-gamma(2,1) at 100" (d/inv-gamma 2 1) 100.0 -13.825510557964275]
   ["geometric(0.5) at 0" (d/geometric 0.5) 0 -0.6931471805599453]
   ["geometric(0.3) at 3" (d/geometric 0.3) 3 -2.2739976361421332]
   ["geometric(0.9) at 1" (d/geometric 0.9) 1 -2.407945608651872]
   ["geometric(0.99) at 0" (d/geometric 0.99) 0 -0.01005033585350145]
   ["geometric(0.5) at 10" (d/geometric 0.5) 10 -7.6246189861593985]
   ["neg-binom(5,0.5) at 3" (d/neg-binom 5 0.5) 3 -1.9898293829901275]
   ["neg-binom(1,0.5) at 0" (d/neg-binom 1 0.5) 0 -0.6931471805599448]
   ["neg-binom(10,0.3) at 5" (d/neg-binom 10 0.3) 5 -6.221200803075214]
   ["neg-binom(1,0.9) at 0" (d/neg-binom 1 0.9) 0 -0.10536051565782584]
   ["neg-binom(3,0.5) at 10" (d/neg-binom 3 0.5) 10 -4.821258605251945]
   ["binom(10,0.5) at 3" (d/binom 10 0.5) 3 -2.14398006281693]
   ["binom(5,0.3) at 0" (d/binom 5 0.3) 0 -1.7833747196936622]
   ["binom(10,0.9) at 10" (d/binom 10 0.9) 10 -1.0536051565782625]
   ["binom(10,0.5) at 10" (d/binom 10 0.5) 10 -6.931471805599452]
   ["binom(10,0.9) at 0" (d/binom 10 0.9) 0 -23.02585092994046]
   ["uniform-discrete(1,6) at 3" (d/uniform-discrete 1 6) 3 -1.791759469228055]
   ["uniform-discrete(0,9) at 5" (d/uniform-discrete 0 9) 5 -2.302585092994046]
   ["uniform-discrete(0,1) at 0" (d/uniform-discrete 0 1) 0 -0.6931471805599453]
   ["uniform-discrete(-5,5) at 0" (d/uniform-discrete -5 5) 0 -2.3978952727983707]
   ["mvnormal(0,I2) at 0" (d/mvnormal [0.0 0.0] [[1.0 0.0] [0.0 1.0]]) [0.0 0.0] -1.8378770664093453]
   ["mvnormal(0,correlated) at [1,2]" (d/mvnormal [0.0 0.0] [[1.0 0.5] [0.5 1.0]]) [1.0 2.0] -3.694036030183455]
   ["mvnormal([1,2,3],I3) at 0" (d/mvnormal [1.0 2.0 3.0] [[1.0 0.0 0.0] [0.0 1.0 0.0] [0.0 0.0 1.0]]) [0.0 0.0 0.0] -9.756815599614018]
   ["mvnormal(0,high-corr) at [0.5,0.5]" (d/mvnormal [0.0 0.0] [[1.0 0.9] [0.9 1.0]]) [0.5 0.5] -1.1390904103669408]
   ["mvnormal([1,1],I2) at [3,3]" (d/mvnormal [1.0 1.0] [[1.0 0.0] [0.0 1.0]]) [3.0 3.0] -5.837877066409345]
   ["piecewise-uniform([0,1,2],[1,1]) at 0.5" (d/piecewise-uniform [0 1 2] [0.5 0.5]) 0.5 -0.6931471805599453]
   ["piecewise-uniform([0,1,2],[1,1]) at 1.5" (d/piecewise-uniform [0 1 2] [0.5 0.5]) 1.5 -0.6931471805599453]
   ["piecewise-uniform 3 bins, low-weight bin" (d/piecewise-uniform [0 1 2 3] [0.25 0.5 0.25]) 0.5 -1.3862943611198906]
   ["piecewise-uniform 3 bins, high-weight bin" (d/piecewise-uniform [0 1 2 3] [0.25 0.5 0.25]) 1.5 -0.6931471805599453]
   ["mixture N(0,1)+N(5,1), equal, at 0" ((d/heterogeneous-mixture [d/normal d/normal]) [0.5 0.5] [0 1] [5 1]) 0 -1.61208198711839]
   ["mixture N(0,1)+N(5,1), equal, at 2.5" ((d/homogeneous-mixture d/normal) [0.5 0.5] [0.0 5.0] [1.0 1.0]) 2.5 -4.0439385332046722]
   ["mixture 0.9 N(0,1) + 0.1 N(10,1) at 0" ((d/homogeneous-mixture d/normal) [0.9 0.1] [0.0 10.0] [1.0 1.0]) 0 -1.024299048862499]
   ["product N(0,1), Exp(1) at [0,1]" ((d/product d/normal d/exponential) [0 1] [1]) [0.0 1.0] -1.9189385332046727]
   ["product N(0,1), Exp(1) at [1,2]" ((d/product d/normal d/exponential) [0 1] [1]) [1.0 2.0] -3.4189385332046727]
   ["product N(0,1), U(0,1), Bern(0.7) at [0.5 0.3 true]" ((d/product d/normal d/uniform d/bernoulli) [0 1] [0 1] [0.7]) [0.5 0.3 true]
    -1.4006134771434051]])

(deftest reference-log-densities
  (doseq [[label dist v expected] reference]
    (let [v (if (and (vector? v) (some boolean? v)) (vec v) v)
          got (if (= ::d/product (:type dist)) (double (array/item (dc/log-prob dist v))) (lp dist v))]
      (is (close? expected got) (str label ": expected " expected ", got " got)))))

(deftest symmetries-and-identities
  (testing "normal is symmetric and shift invariant"
    (doseq [sd [0.5 1.0 2.0 5.0] v [0.1 1.0 3.0 10.0]]
      (is (close? (lp (d/normal 0 sd) v) (lp (d/normal 0 sd) (- v)))))
    (doseq [[mu sd v shift] [[2.0 1.0 3.0 5.0] [-1.0 0.5 2.0 -3.0] [0.0 3.0 1.0 10.0]]]
      (is (close? (lp (d/normal mu sd) v) (lp (d/normal (+ mu shift) sd) (+ v shift))))))
  (testing "uniform inside and outside its bounds"
    (doseq [[lo hi v] [[-5.0 5.0 0.0] [0.0 1.0 0.5] [-10.0 -0.1 -5.0] [0.1 10.0 5.0]]]
      (is (close? (- (Math/log (- hi lo))) (lp (d/uniform lo hi) v))))
    (doseq [[lo hi v] [[0.0 1.0 -0.1] [0.0 1.0 1.5] [-5.0 5.0 10.0]]]
      (is (= ##-Inf (lp (d/uniform lo hi) v)))))
  (testing "bernoulli"
    (doseq [p [0.2 0.5 0.8]]
      (is (close? (Math/log p) (lp (d/bernoulli p) true)))
      (is (close? (Math/log (- 1 p)) (lp (d/bernoulli p) false)))))
  (testing "exponential(1) is -v on the positive axis, -Inf below it"
    (doseq [v [0.5 1.0 2.0 5.0]] (is (close? (- v) (lp (d/exponential 1.0) v))))
    (doseq [v [-0.1 -1.0 -100.0]] (is (= ##-Inf (lp (d/exponential 1.0) v)))))
  (testing "laplace at its location is -log 2"
    (doseq [v [0.0 1.5 -3.0 10.0]] (is (close? (- (Math/log 2)) (lp (d/laplace v 1) v)))))
  (testing "log Γ(n) = log (n-1)! through the gamma density"
    (doseq [n (range 1 15)]
      (is (close? (Math/log (reduce * 1.0 (range 1 n))) (- -1.0 (lp (d/gamma n 1) 1.0))))))
  (testing "outside the support"
    (is (= ##-Inf (lp (d/uniform-discrete 1 6) 7)))
    (is (= ##-Inf (lp (d/piecewise-uniform [0 1 2] [0.5 0.5]) 5.0)))))

(defn- integral
  "∫ exp(log-prob) over [lo, hi] by Simpson's rule with n panels."
  [dist lo hi n]
  (let [h (/ (- hi lo) n)
        f #(Math/exp (lp dist (+ lo (* % h))))]
    (* (/ h 3) (+ (f 0) (f n)
                  (* 4 (reduce + (map f (range 1 n 2))))
                  (* 2 (reduce + (map f (range 2 n 2))))))))

(deftest densities-are-normalized
  (doseq [[label dist lo hi mass] [["normal" (d/normal 0.5 1.3) -12.0 13.0]
                              ["gamma" (d/gamma 2.5 1.5) 0.0 80.0]
                              ["beta" (d/beta 2.5 1.7) 0.0 1.0]
                              ["exponential" (d/exponential 1.5) 0.0 40.0]
                              ;; split at the kink, where Simpson's rule loses its order
                              ["laplace, left" (d/laplace 0.3 0.8) -40.0 0.3 0.5]
                              ["laplace, right" (d/laplace 0.3 0.8) 0.3 40.0 0.5]
                              ["inv-gamma" (d/inv-gamma 3.0 2.0) 1e-6 400.0]
                              ["uniform" (d/uniform -1.0 2.0) -1.0 2.0]
                              ["mixture" ((d/homogeneous-mixture d/normal) [0.3 0.7] [-2.0 1.0] [0.5 1.0]) -15.0 15.0]]]
    (is (< (Math/abs (- (or mass 1.0) (integral dist lo hi 20000))) 1e-6) label))
  (doseq [[label dist support]
          [["poisson" (d/poisson 3.5) (range 60)] ["binom" (d/binom 12 0.35) (range 13)]
           ["geometric" (d/geometric 0.3) (range 200)] ["neg-binom" (d/neg-binom 2.5 0.4) (range 300)]
           ["uniform-discrete" (d/uniform-discrete -2 4) (range -2 5)] ["categorical" (d/categorical [0.2 0.5 0.3]) (range 3)]]]
    (is (< (Math/abs (- 1.0 (reduce + (map #(Math/exp (lp dist %)) support)))) (ts/tol 1e-9)) label)))
