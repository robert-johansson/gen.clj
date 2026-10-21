(ns gen.genjl.mixture-test
  "Gen.jl 0.4.8's test/modeling_library/mixture.jl, ported: heterogeneous
   and homogeneous mixtures, their densities against the components', and
   their gradients against finite differences (gen.genjl.finite-diff).
   gen.clj's heterogeneous mixture groups each component's parameters (one
   vector per component), so the ports call it from Gen.jl's flat
   arguments; a multivariate family's components lead its parameters'
   axes (D3: means [K d], covariances [K d d]). has_output_grad and
   has_argument_grads, and nothing for a gradient, are n/a."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.genjl.finite-diff :as fd]
            [gen.gfi :as gfi]
            [gen.gradients :as grads]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(defn- density
  "Gen.jl's logpdf, on the host."
  [make x & args]
  (ts/number (apply (fd/logpdf make) x args)))

(defn- log-mix
  "log Σ wₖ exp(lₖ), on the host."
  [ws lps]
  (Math/log (reduce + (map #(* %1 (Math/exp %2)) ws lps))))

(def ^:private uniform-beta
  "Gen.jl's HeterogeneousMixture([uniform, beta]), from its flat arguments."
  (let [m (d/heterogeneous-mixture [d/uniform d/beta])]
    (fn [weights a b alpha beta] (m weights [a b] [alpha beta]))))

(deftest fixed-mixture-of-different-distributions
  ;; mixture.jl:3
  (let [[w1 w2 a b alpha beta] [0.4 0.6 0.0 3.4 2.3 1.0]
        ws (array/array [w1 w2])
        x 0.123
        args [x ws a b alpha beta]]
    (is (not (dc/discrete? (uniform-beta ws a b alpha beta))))
    (is (number? (ts/host (dc/sample (uniform-beta ws a b alpha beta) (rng/key 1)))) "a draw")
    (is (ts/close? (log-mix [w1 w2] [(density d/uniform x a b) (density d/beta x alpha beta)])
                   (density uniform-beta x ws a b alpha beta)))
    (testing "logpdf_grad against finite differences"
      (is (fd/logpdf-grad-matches? uniform-beta args 0) "x")
      (is (fd/logpdf-grad-matches? uniform-beta args 1 {:dx fd/probability-dx}) "the two weights")
      (doseq [i (range 2 6)]
        (is (fd/logpdf-grad-matches? uniform-beta args i) (str "argument " i))))))

(def ^:private mixture-of-normals (d/homogeneous-mixture d/normal [0 0]))
(def ^:private mixture-of-binomials (d/homogeneous-mixture d/binom [0 0]))
(def ^:private mixture-of-mvnormals (d/homogeneous-mixture d/mvnormal [1 2]))

(deftest mixture-of-normals-test
  ;; mixture.jl:44
  (let [[w1 w2 mu1 mu2 std1 std2] [0.4 0.6 0.0 1.0 2.3 1.0]
        [ws mus stds] (mapv array/array [[w1 w2] [mu1 mu2] [std1 std2]])
        x 1.123
        args [x ws mus stds]]
    (is (not (dc/discrete? (mixture-of-normals ws mus stds))))
    (is (number? (ts/host (dc/sample (mixture-of-normals ws mus stds) (rng/key 1)))) "a draw")
    (is (ts/close? (log-mix [w1 w2] [(density d/normal x mu1 std1) (density d/normal x mu2 std2)])
                   (density mixture-of-normals x ws mus stds)))
    (testing "logpdf_grad against finite differences"
      (let [[_ _ gmus gstds] (apply fd/logpdf-grad mixture-of-normals args)]
        (is (= [2] (array/shape gmus)))
        (is (= [2] (array/shape gstds))))
      (is (fd/logpdf-grad-matches? mixture-of-normals args 0) "x")
      (is (fd/logpdf-grad-matches? mixture-of-normals args 1 {:dx fd/probability-dx}) "the two weights")
      (is (fd/logpdf-grad-matches? mixture-of-normals args 2) "the two means")
      (is (fd/logpdf-grad-matches? mixture-of-normals args 3) "the two standard deviations"))
    (testing "the tape, through each scalar as Gen.jl tracks it"
      (let [scalar-logpdf (fn [x w1 w2 mu1 mu2 std1 std2]
                            ((fd/logpdf mixture-of-normals) x (array/stack [w1 w2]) (array/stack [mu1 mu2])
                                                            (array/stack [std1 std2])))
            scalars [x w1 w2 mu1 mu2 std1 std2]
            tracked (apply (array/grad scalar-logpdf {:argnums (vec (range 7))}) scalars)]
        (doseq [i (range 7)]
          (is (fd/grad-close? (nth tracked i)
                              (fd/finite-diff scalar-logpdf scalars i (when (#{1 2} i) {:dx fd/probability-dx})))
              (str "scalar " i)))))))

(deftest mixture-of-binomial
  ;; mixture.jl:104
  (let [[w1 w2 n1 n2 p1 p2] [0.4 0.6 10 20 0.2 0.3]
        ws (array/array [w1 w2])
        ns (array/array [n1 n2])
        ps (array/array [p1 p2])
        x 4
        args [x ws ns ps]]
    (is (dc/discrete? (mixture-of-binomials ws ns ps)))
    (is (<= 0 (ts/host (dc/sample (mixture-of-binomials ws ns ps) (rng/key 1)))) "a draw")
    (is (ts/close? (log-mix [w1 w2] [(density d/binom x n1 p1) (density d/binom x n2 p2)])
                   (density mixture-of-binomials x ws ns ps)))
    (testing "logpdf_grad against finite differences"
      (is (fd/logpdf-grad-matches? mixture-of-binomials args 1 {:dx fd/probability-dx}) "the two weights")
      (is (fd/logpdf-grad-matches? mixture-of-binomials args 3) "the two probabilities"))))

(def ^:private rotated-covariance-model
  "Gen.jl's f_cov_model: the mixture of mvnormals whose second covariance
   is diag(2, 1) rotated by theta, which stays positive definite as theta
   moves, where a step in the matrix itself would not. Its constants are
   made where it runs, so finite differences run it on the JVM backend."
  (gen [theta]
    (let [c (array/array [[2.0 0.0] [0.0 1.0]])
          rot (array/stack [(array/stack [(array/cos theta) (- (array/sin theta))])
                            (array/stack [(array/sin theta) (array/cos theta)])])]
      (trace :x (mixture-of-mvnormals (array/array [0.4 0.6]) (array/array [[0.0 0.0] [1.0 1.0]])
                                      (array/stack [c (array/matmul (array/matmul rot c) (array/transpose rot))]))))))

(deftest mixture-of-multivariate-normals
  ;; mixture.jl:138; Gen.jl's mus (d × K) and covs (d × d × K) lead with the
  ;; component here (D3)
  (let [[w1 w2] [0.4 0.6]
        ws (array/array [w1 w2])
        [mu1 mu2] [[0.0 0.0] [1.0 1.0]]
        [cov1 cov2] [[[2.3 0.0] [0.0 2.3]] [[1.0 0.0] [0.0 1.0]]]
        mus (array/array [mu1 mu2])
        covs (array/array [cov1 cov2])
        x (array/array [1.123 -2.123])
        args [x ws mus covs]]
    (is (not (dc/discrete? (mixture-of-mvnormals ws mus covs))))
    (is (= [2] (array/shape (dc/sample (mixture-of-mvnormals ws mus covs) (rng/key 1)))) "a draw")
    (is (ts/close? (log-mix [w1 w2] [(density d/mvnormal x mu1 cov1) (density d/mvnormal x mu2 cov2)])
                   (density mixture-of-mvnormals x ws mus covs)))
    (testing "logpdf_grad against finite differences"
      (let [[_ _ gmus gcovs] (apply fd/logpdf-grad mixture-of-mvnormals args)]
        (is (= [2 2] (array/shape gmus)))
        (is (= [2 2 2] (array/shape gcovs))))
      (is (fd/logpdf-grad-matches? mixture-of-mvnormals args 0) "x's two")
      (is (fd/logpdf-grad-matches? mixture-of-mvnormals args 1 {:dx fd/probability-dx}) "the two weights")
      (is (fd/logpdf-grad-matches? mixture-of-mvnormals args 2 {:rtol 1e-2}) "every mean's finite_diff_arr"))
    (testing "a covariance rotated by theta: choice_gradients' argument gradient at pi/2"
      (let [at (fn [theta] (:trace (gfi/generate rotated-covariance-model [theta]
                                                 (cm/choicemap :x (array/array [1.123 -2.123])) (rng/key 1))))
            theta (/ Math/PI 2)]
        (is (fd/grad-close? (first (:arg-grads (grads/choice-gradients (at theta) sel/none nil {:argnums [0]})))
                            (fd/finite-diff #(:score (at %)) [theta] 0)))))))
