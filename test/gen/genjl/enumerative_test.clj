(ns gen.genjl.enumerative-test
  "Gen.jl 0.4.8's test/inference/enumerative.jl (which Gen.jl's runtests.jl
   never includes, U2), ported: choice-volume grids over discrete,
   continuous and 2-d choices, and enumerative inference over them
   (gen.inference.enumerative).

   A grid is a flat vector in row-major order, the last spec varying
   fastest, with no N-d shape (D12): Gen.jl's size(grid) and size(traces)
   are n/a, and its first-index-fastest order changes nothing asserted
   here. Endpoints are computed as Julia's ranges are, correctly rounded
   (k/5 - 1, not -1 + 0.2k), so the grid's midpoints are Gen.jl's values."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.genjl.support :as genjl]
            [gen.inference.enumerative :as enum]
            [gen.inference.weights :as w]
            [gen.test-support :as ts]))

(defn- endpoints
  "Julia's lo:step:hi as doubles, lo + k·step correctly rounded: (k + lo/step)/(1/step)."
  [lo per-unit n]
  (mapv #(/ (+ % (* lo per-unit)) (double per-unit)) (range n)))

(defn- log-sum-exp [xs] (:log-total (w/normalize xs)))

;; ---------------------------------------------------------------------------
;; enumerative.jl:3, polynomial regression
;; ---------------------------------------------------------------------------

(def ^:private poly-model
  (gen [n xs]
    (let [degree (trace :degree (d/uniform-discrete 1 n))
          coeffs (mapv #(trace [:coeff %] (d/uniform -1 1)) (range (inc n)))]
      ;; the terms of degree 0 to degree
      (mapv (fn [i x]
              (let [y-mean (reduce + (map #(array/where (array/<= % degree) (* (coeffs %) (array/pow x %)) 0.0)
                                          (range (inc n))))]
                (trace [:y i] (d/normal y-mean 0.1))))
            (range) xs))))

(def ^:private true-coeffs [0.5 0.1 -0.5])

(def ^:private xs (endpoints 0.5 2 6))

(def ^:private observations
  (apply cm/choicemap (mapcat (fn [i x] [[:y i] (reduce + (map-indexed #(* %2 (Math/pow x %1)) true-coeffs))])
                              (range) xs)))

(defn- in-poly-box?
  "enumerative.jl:48, test_choices: the degree in 1..n, coefficients 1..n in [-1, 1]."
  [n cs]
  (and (<= 1 (ts/host (cs :degree)) n)
       (every? #(<= -1.0 (ts/host (cs [:coeff %])) 1.0) (range 1 (inc n)))))

(deftest polynomial-regression
  ;; enumerative.jl:1
  (let [coeff-points (endpoints -1 5 11)
        grid (enum/choice-vol-grid [:degree [1 2]]
                                   [[:coeff 0] coeff-points :continuous]
                                   [[:coeff 1] coeff-points :continuous]
                                   [[:coeff 2] coeff-points :continuous]
                                   {:anchor :midpoint})
        log-vol (Math/log (Math/pow 0.2 3))]
    (testing "the grid"
      (is (= 2000 (count grid)))
      (let [[choices vol] (first grid)
            {:keys [degree coeff]} (ts/host choices)]
        (is (and (== 1 degree) (= [0 1 2] (sort (keys coeff))) (every? #(ts/close? -0.9 %) (vals coeff))))
        (is (genjl/approx? log-vol vol)))
      (is (every? (fn [[choices _]] (in-poly-box? 2 #(cm/get-choice choices %))) grid))
      (is (every? (fn [[_ vol]] (genjl/approx? log-vol vol)) grid)))
    (testing "enumerative inference over it"
      (let [{:keys [traces log-normalized-weights log-ml-estimate]}
            (enum/enumerative-inference poly-model [2 xs] observations grid)
            joint (mapv #(+ (ts/number (:score %)) log-vol) traces)
            lml (log-sum-exp joint)]
        (is (= 2000 (count traces)))
        (is (every? #(in-poly-box? 2 %) traces))
        (is (genjl/approx? lml log-ml-estimate))
        (is (every? true? (map #(genjl/approx? (- %1 lml) %2) joint log-normalized-weights)))
        (testing "the polynomial is most likely quadratic"
          (let [degrees (enum/marginal {:traces traces :log-normalized-weights log-normalized-weights} :degree)]
            (is (= 2 (key (apply max-key val degrees))))))
        (testing "the MAP trace recovers the coefficients"
          (let [map-trace (traces (apply max-key log-normalized-weights (range (count traces))))]
            (is (== 2 (ts/host (map-trace :degree))))
            (is (ts/close? 0.5 (map-trace [:coeff 0])))
            (is (ts/close? 0.1 (map-trace [:coeff 1])))
            (is (ts/close? -0.5 (map-trace [:coeff 2])))))))))

;; ---------------------------------------------------------------------------
;; enumerative.jl:80, a 2-d mixture of normals
;; ---------------------------------------------------------------------------

(def ^:private mixture-model
  (gen [] (let [sign (trace :sign (d/bernoulli 0.5))
                mu (array/where sign (array/full [2] 0.5) (array/full [2] -0.5))]
            (trace :z (d/broadcasted-normal mu (array/ones [2]))))))

(defn- in-square? [z] (every? #(<= -2.0 % 2.0) (ts/host z)))

(deftest a-2-d-mixture-of-normals
  ;; enumerative.jl:80
  (let [z-points (endpoints -2 10 41)
        grid (enum/choice-vol-grid [:sign [false true]] [:z [z-points z-points] :continuous 2] {:anchor :left})
        log-vol (Math/log (Math/pow 0.1 2))]
    (testing "the grid"
      (is (= 3200 (count grid)))
      (let [[choices vol] (first grid)]
        (is (= {:sign false :z [-2.0 -2.0]} (ts/host choices)))
        (is (genjl/approx? log-vol vol)))
      (is (every? (fn [[choices _]] (in-square? (cm/get-choice choices :z))) grid))
      (is (every? (fn [[_ vol]] (genjl/approx? log-vol vol)) grid)))
    (testing "enumerative inference over it"
      (let [{:keys [traces log-normalized-weights log-ml-estimate]}
            (enum/enumerative-inference mixture-model [] cm/EMPTY grid)
            expected (fn [tr] (let [[x y] (ts/host (tr :z))
                                    mu (if (ts/host (tr :sign)) 0.5 -0.5)
                                    n (d/normal mu 1.0)]
                                (+ (Math/log 0.5) (ts/number (dc/log-prob n x)) (ts/number (dc/log-prob n y)))))
            joint (mapv #(+ (expected %) log-vol) traces)
            lml (log-sum-exp joint)]
        (is (= 3200 (count traces)))
        (is (every? #(in-square? (% :z)) traces))
        (is (genjl/approx? lml log-ml-estimate))
        (is (every? true? (map #(genjl/approx? (- %1 lml) %2) joint log-normalized-weights)))
        (testing "the largest weights are at the modes"
          (let [top (apply max log-normalized-weights)
                [first-mode second-mode] (map traces (filter #(= top (log-normalized-weights %)) (range (count traces))))]
            (is (false? (ts/host (first-mode :sign))))
            (is (= [-0.5 -0.5] (ts/host (first-mode :z))))
            (is (true? (ts/host (second-mode :sign))))
            (is (= [0.5 0.5] (ts/host (second-mode :z))))))))))

;; n/a (D12: a grid has no N-d shape): enumerative.jl:38 and :94,
;; size(grid); :58 and :108, size(traces).
