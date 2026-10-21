(ns gen.genjl.distributions-test
  "Gen.jl 0.4.8's test/modeling_library/distributions.jl, ported: each
   distribution's range, its density outside the support, and logpdf_grad
   against finite differences (gen.genjl.finite-diff). gen.clj's values are
   0-based (categorical: D2), its shapes NumPy's, Julia's reversed (D3),
   invalid parameters are errors (D5), and beta is exact at 0 and 1 (D6).
   Draws use fixed keys where Gen.jl uses its global generator."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.genjl.finite-diff :as fd]
            [gen.genjl.support :as genjl]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(defn- draw
  "A draw of dist, on the host."
  [dist]
  (ts/host (dc/sample dist (rng/key 1))))

(defn- grads
  "Gen.jl's logpdf_grad(dist, args...) on the host: a vector of the
   gradients, nil for an integer or a boolean."
  [make & args]
  (ts/host (apply fd/logpdf-grad make args)))

(def ^:private eps
  "Julia's eps(Float64)."
  (Math/ulp 1.0))

(deftest bernoulli
  ;; distributions.jl:4
  (is (fd/logpdf-grad-matches? d/bernoulli [false 0.3] 1))
  (is (fd/logpdf-grad-matches? d/bernoulli [true 0.3] 1)))

(deftest beta
  ;; distributions.jl:19; at 0 and 1 gen.clj is exact (D6, DESIGN §9): +Inf
  ;; where alpha (at 0) or beta (at 1) is below 1, where Gen.jl substitutes
  ;; a finite value
  (is (< 0 (draw (d/beta 0.5 0.5)) 1))
  (is (= ##-Inf (genjl/logpdf d/beta -1.0 0.5 0.5)))
  (testing "exact at the endpoints: Gen.jl's logpdf(eps) < logpdf(0) < Inf is logpdf(eps) < logpdf(0) = Inf"
    (doseq [[inside edge a b] [[eps 0.0 0.5 0.5] [eps 0.0 0.5 1.5] [(- 1 eps) 1.0 0.5 0.5] [(- 1 eps) 1.0 1.5 0.5]]]
      (let [at-edge (genjl/logpdf d/beta edge a b)]
        (is (= ##Inf at-edge) (str "at " edge " with alpha " a ", beta " b))
        (is (< (genjl/logpdf d/beta inside a b) at-edge)))))
  (doseq [i (range 3)]
    (is (fd/logpdf-grad-matches? d/beta [0.4 0.2 0.3] i) (str "d/darg" i))))

(deftest categorical
  ;; distributions.jl:43; values are 0-based (D2): Gen.jl's 1 to 3 are 0 to
  ;; 2, its value 2 is 1. Distributions.jl's own integer-probability bug,
  ;; and the value's nothing gradient, are n/a.
  (is (<= 0 (draw (d/categorical [0.2 0.3 0.5])) 2))
  (is (= ##-Inf (genjl/logpdf d/categorical -1 (array/array [0.2 0.3 0.5]))))
  (is (fd/logpdf-grad-matches? d/categorical [1 (array/array [0.2 0.3 0.5])] 1 {:dx fd/probability-dx})
      "the three probabilities' finite_diff_vec"))

(deftest gamma
  ;; distributions.jl:67
  (is (< 0 (draw (d/gamma 1.0 1.0))))
  (is (= ##-Inf (genjl/logpdf d/gamma -1.0 1.0 1.0)))
  (doseq [i (range 3)]
    (is (fd/logpdf-grad-matches? d/gamma [0.4 0.2 0.3] i) (str "d/darg" i))))

(deftest inv-gamma
  ;; distributions.jl:85
  (is (< 0 (draw (d/inv-gamma 1.0 1.0))))
  (is (= ##-Inf (genjl/logpdf d/inv-gamma -1.0 1.0 1.0)))
  (doseq [i (range 3)]
    (is (fd/logpdf-grad-matches? d/inv-gamma [0.4 0.2 0.3] i) (str "d/darg" i))))

(deftest normal
  ;; distributions.jl:103: no overflow at 1e13, exactly
  (when (ts/float64?)
    (is (= -5e25 (genjl/logpdf d/normal 1e13 0.0 1.0)))
    (is (= [-1e13 1e13 1e26] (grads d/normal 1e13 0.0 1.0))))
  (doseq [i (range 3)]
    (is (fd/logpdf-grad-matches? d/normal [0.4 0.2 0.3] i) (str "d/darg" i))))

(deftest zero-dimensional-broadcasted-normal
  ;; distributions.jl:121: 0-d values and parameters, whose gradients are
  ;; 0-d (host numbers on the JVM backend)
  (let [args (mapv #(array/array % {:dtype (ts/float-dtype)}) [0.4 0.2 0.3])]
    (doseq [g (apply fd/logpdf-grad d/broadcasted-normal args)]
      (is (= [] (array/shape g))))
    (doseq [i (range 3)]
      (is (fd/logpdf-grad-matches? d/broadcasted-normal args i) (str "d/darg" i)))))

(deftest array-normal
  ;; distributions.jl:140: all arguments of one shape
  (let [args (mapv array/array [[[1.0 2.0 3.0] [4.0 5.0 6.0]]
                                [[0.1 0.2 0.3] [0.4 0.5 0.6]]
                                [[0.01 0.02 0.03] [0.04 0.05 0.06]]])]
    (doseq [g (apply fd/logpdf-grad d/broadcasted-normal args)]
      (is (= [2 3] (array/shape g))))
    (doseq [i (range 3)]
      (is (fd/logpdf-grad-matches? d/broadcasted-normal args i {:rtol 1e-7}) (str "d/darg" i)))))

(defn- shape-of-draw [mu std] (array/shape (dc/sample (d/broadcasted-normal mu std) (rng/key 1))))

(defn- throws?
  "Does (f) throw an ex-info, where Gen.jl throws (DimensionMismatch) or
   gives -Inf for an invalid parameter (D5)?"
  [f]
  (try (f) false (catch clojure.lang.ExceptionInfo _ true)))

(deftest broadcasted-normal
  ;; distributions.jl:166; Julia pads shapes on the right and gen.clj
  ;; (NumPy) on the left, so each of Gen.jl's shapes is reversed here (D3)
  (testing "the shape of a draw"
    (is (= [3 1] (shape-of-draw (array/zeros [3 1]) 1.0)))
    (is (= [4 3 2] (shape-of-draw (array/zeros [4 3 1]) (array/ones [4 1 2]))))
    (is (= [1 3 2] (shape-of-draw (array/zeros [3 1]) (array/ones [1 1 2]))))
    (is (throws? #(shape-of-draw (array/zeros [3 1]) (array/ones [2 1]))))
    (is (throws? #(shape-of-draw (array/zeros [3 2]) (array/ones [3 2 1])))
        "Julia's (2, 3) and (1, 2, 3), which NumPy would broadcast the other way round"))
  (testing "logpdf is a number, its gradients the arguments' shapes"
    (let [args [(array/ones [4 2]) (array/ones [1 2]) (array/ones [4 1])]]
      (is (= [] (array/shape (apply genjl/log-density d/broadcasted-normal args))))
      (is (= [[4 2] [1 2] [4 1]] (mapv array/shape (apply fd/logpdf-grad d/broadcasted-normal args))))))
  (testing "a value of the wrong shape, or one only broadcast-compatible, or parameters that do not broadcast"
    (doseq [[label args] {"wrong" [(array/ones [2 1]) (array/ones [3 1]) (array/ones [1 2])]
                          "broadcast-compatible" [(array/ones [3 1]) (array/ones [3 1]) (array/ones [1 2])]
                          "incompatible parameters" [(array/ones [1 2]) (array/ones [2 1]) (array/ones [3 1])]}]
      (is (throws? #(apply genjl/logpdf d/broadcasted-normal args)) (str label ": logpdf"))
      (is (throws? #(apply fd/logpdf-grad d/broadcasted-normal args)) (str label ": logpdf_grad"))))
  (testing "logpdf with broadcast parameters equals logpdf with them expanded"
    ;; Julia's x (2, 4, 1), mu (1, 4), std (2, 1, 1), reversed
    (let [x (array/array [[[0.2 0.5] [0.3 0.4] [0.4 0.3] [0.5 0.2]]])
          mu (array/array [[0.7] [0.7] [0.8] [0.6]])
          std (array/array [[[0.2 0.1]]])
          logpdf (partial genjl/logpdf d/broadcasted-normal)]
      (is (= (logpdf x mu std)
             (logpdf x (array/broadcast-to mu [1 4 2]) (array/broadcast-to std [1 4 2])))))))

(defn- symmetrized
  "(g + gᵀ)/2: a covariance gradient from the tape sits on the triangle the
   Cholesky factorization reads, where Gen.jl's is symmetric (D3)."
  [g]
  (array/* 0.5 (array/+ g (array/transpose g))))

(deftest multivariate-normal
  ;; distributions.jl:217
  (is (= 2 (count (draw (d/mvnormal [0.0 0.0] [[1.0 0.2] [0.2 1.4]])))))
  (let [args [(array/array [0.1 0.2]) (array/array [0.3 0.4]) (array/array [[1.0 0.2] [0.2 1.4]])]
        f (partial genjl/logpdf d/mvnormal)
        [gx gmu gcov] (apply fd/logpdf-grad d/mvnormal args)]
    (is (fd/grad-close? gx (fd/finite-diff f args 0)) "x's two finite_diff_vec")
    (is (fd/grad-close? gmu (fd/finite-diff f args 1)) "mu's two")
    (is (fd/grad-close? (symmetrized gcov) (fd/finite-diff f args 2 {:symmetric? true}))
        "cov's four finite_diff_mat_sym")))

(def ^:private dirichlet-references
  "scipy's log densities for the values Gen.jl compares with
   Distributions.jl's (test/fixtures/generate.py)."
  (into {} (keep (fn [{:keys [dist params value log-prob]}]
                   (when (= :dirichlet dist) [[value (first params)] log-prob])))
        (edn/read-string (slurp "test/fixtures/distributions.edn"))))

(defn- softmax-jacobian
  "Gen.jl's softmax_grad at a point s of the simplex: diagm(s) - s sᵀ."
  [s]
  (array/- (array/* (array/eye (first (array/shape s))) s) (array/outer s s)))

(deftest dirichlet
  ;; distributions.jl:237; invalid alphas are errors (D5) where Gen.jl's
  ;; logpdf is -Inf
  (let [x (draw (d/dirichlet [1.0 1.0 1.0 1.0]))]
    (is (= 4 (count x)))
    (is (ts/close? 1.0 (reduce + x))))
  (testing "values off the simplex"
    (is (= ##-Inf (genjl/logpdf d/dirichlet (array/array [0.0 0.0]) (array/array [1.0 1.0]))))
    (is (= ##-Inf (genjl/logpdf d/dirichlet (array/array [1.0 1.0]) (array/array [1.0 1.0]))))
    (is (= ##-Inf (genjl/logpdf d/dirichlet (array/array [2.0 -1.0]) (array/array [1.0 1.0]))))
    ;; alpha as data, as Julia's literal: data is validated on every backend
    ;; (dc/->checkable), an array's values only where they can be read (rule 5)
    (is (throws? #(genjl/logpdf d/dirichlet (array/array [0.5 0.5]) [-1.0 1.0])) "a negative alpha")
    (is (throws? #(genjl/logpdf d/dirichlet (array/array [0.5 0.5]) [-1.0 1.0])) "(again, as Gen.jl)")
    (is (not= ##-Inf (genjl/logpdf d/dirichlet (array/array [0.0 1.0]) (array/array [1.0 1.0]))) "a corner, with alpha 1"))
  (testing "values against scipy's, as Gen.jl's against Distributions.jl's"
    (doseq [[x alpha] [[[0.01 0.99] [2.0 2.0]] [[0.01 0.99] [1.0 4.0]] [[0.01 0.99] [0.01 0.01]] [[0.2 0.2 0.6] [2.0 2.0 4.0]]]]
      (is (ts/close? (dirichlet-references [x alpha]) (genjl/logpdf d/dirichlet (array/array x) (array/array alpha))
                     {:rtol (ts/tol 1e-10)})
          (str x " under " alpha))))
  (testing "gradients: with respect to x through a softmax, and to alpha"
    (let [alpha (array/array [1.0 2.0 3.0 3.0])
          s (array/array [0.25 0.25 0.25 0.25])
          [gx galpha] (fd/logpdf-grad d/dirichlet s alpha)
          f (partial genjl/logpdf d/dirichlet)]
      (is (fd/grad-close? (array/matmul gx (softmax-jacobian s))
                          (fd/finite-diff (fn [x a] (f (array/softmax x) a)) [(array/zeros [4]) alpha] 0))
          "x's four, chained through softmax_grad")
      ;; the difference's rounding: lgamma(Σα) = lgamma(9) = 10.6, which
      ;; gen.clj computes to an ulp (1.8e-15) where Julia's is exact there,
      ;; makes each partial's difference uncertain by 1.8e-15/dx = 1.8e-9 (a
      ;; relative 1.1e-8 for -0.168, near √eps): atol 1e-8 allows a few ulps
      (is (fd/grad-close? galpha (fd/finite-diff f [s alpha] 1) {:atol 1e-8}) "alpha's four"))))

(deftest uniform
  ;; distributions.jl:292
  (is (< -0.5 (draw (d/uniform -0.5 0.5)) 0.5))
  (is (= ##-Inf (genjl/logpdf d/uniform -1.0 -0.5 0.5)))
  (doseq [i (range 3)]
    (is (fd/logpdf-grad-matches? d/uniform [0.0 -0.5 0.5] i) (str "d/darg" i))))

(deftest uniform-discrete
  ;; distributions.jl:310; its nothing gradients are n/a
  (is (<= 10 (draw (d/uniform-discrete 10 20)) 20))
  (is (= ##-Inf (genjl/logpdf d/uniform-discrete -1 10 20))))

(deftest piecewise-uniform
  ;; distributions.jl:325
  (is (< -0.5 (draw (d/piecewise-uniform [-0.5 0.5] [1.0])) 0.5))
  (is (= ##-Inf (genjl/logpdf d/piecewise-uniform -1.0 (array/array [-0.5 0.5]) (array/array [1.0]))))
  (let [args [0.5 (array/array [-1.0 0.0 1.0]) (array/array [0.4 0.6])]]
    (is (fd/logpdf-grad-matches? d/piecewise-uniform args 0) "x")
    (is (fd/logpdf-grad-matches? d/piecewise-uniform args 1) "the three bounds' finite_diff_vec")
    (is (fd/logpdf-grad-matches? d/piecewise-uniform args 2 {:dx fd/probability-dx}) "the two probabilities'")))

(deftest beta-uniform
  ;; distributions.jl:346
  (is (< 0 (draw (d/beta-uniform 0.5 0.5 0.5)) 1))
  (is (= ##-Inf (genjl/logpdf d/beta-uniform -1.0 0.5 0.5 0.5)))
  (doseq [i (range 4)]
    (is (fd/logpdf-grad-matches? d/beta-uniform [0.5 0.4 10.0 2.0] i) (str "d/darg" i))))

(deftest geometric
  ;; distributions.jl:365; the value's nothing gradient is n/a
  (is (<= 0 (draw (d/geometric 0.5))))
  (is (= ##-Inf (genjl/logpdf d/geometric -1 0.5)))
  (is (fd/logpdf-grad-matches? d/geometric [4 0.3] 1)))

(deftest binom
  ;; distributions.jl:381; the nothing gradients are n/a
  (is (<= 0 (draw (d/binom 5 0.3))))
  (is (= ##-Inf (genjl/logpdf d/binom -1 5 0.3)))
  (is (fd/logpdf-grad-matches? d/binom [2 5 0.3] 2)))

(deftest neg-binom
  ;; distributions.jl:398; r is real, so Gen.jl differentiates it (5 here
  ;; as a float); the value's nothing gradient is n/a
  (is (<= 0 (draw (d/neg-binom 5 0.3))))
  (is (= ##-Inf (genjl/logpdf d/neg-binom -1 5 0.3)))
  (is (fd/logpdf-grad-matches? d/neg-binom [2 5.0 0.3] 1))
  (is (fd/logpdf-grad-matches? d/neg-binom [2 5.0 0.3] 2)))

(deftest exponential
  ;; distributions.jl:415
  (is (< 0 (draw (d/exponential 0.5))))
  (is (= ##-Inf (genjl/logpdf d/exponential -1.0 0.5)))
  (is (fd/logpdf-grad-matches? d/exponential [1.2 0.5] 0))
  (is (fd/logpdf-grad-matches? d/exponential [1.2 0.5] 1)))

(deftest poisson
  ;; distributions.jl:431; the value's nothing gradient is n/a
  (is (<= 0 (draw (d/poisson 1.0))))
  (is (= ##-Inf (genjl/logpdf d/poisson -1 1.0)))
  (is (fd/logpdf-grad-matches? d/poisson [4 2.0] 1)))

(deftest laplace
  ;; distributions.jl:447
  (doseq [i (range 3)]
    (is (fd/logpdf-grad-matches? d/laplace [0.4 0.2 0.3] i) (str "d/darg" i))))

(deftest cauchy
  ;; distributions.jl:461
  (doseq [i (range 3)]
    (is (fd/logpdf-grad-matches? d/cauchy [0.4 0.2 0.3] i) (str "d/darg" i))))
