(ns gen.inference.importance-test
  "Importance sampling on a conjugate model: x ~ N(0, 1), y ~ N(x, 1), so
   p(y) = N(y; 0, √2) and x | y ~ N(y/2, √½)."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as a :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.inference.importance :as imp]
            [gen.random :as rng]))

(def model
  (gen []
    (let [x (trace :x (d/normal 0 1))]
      (trace :y (d/normal x 1)))))

(def y 1.3)
(def observations (cm/choicemap :y y))
(def log-evidence (dc/log-prob (d/normal 0 (Math/sqrt 2)) y))

(def posterior-proposal
  (gen [y] (trace :x (d/normal (/ y 2) (Math/sqrt 0.5)))))

(defn- density [mu sd x] (Math/exp (dc/log-prob (d/normal mu sd) x)))

(def relative-variance
  "Var(w)/E(w)² for prior-proposal weights w = N(y; x, 1): E(w²) is
   N(y; 0, √1.5)/(2√π)."
  (- (/ (/ (density 0 (Math/sqrt 1.5) y) (* 2 (Math/sqrt Math/PI)))
        (Math/pow (Math/exp log-evidence) 2))
     1))

(deftest the-exact-posterior-proposal-gives-equal-weights
  (let [n 50
        {:keys [traces log-normalized-weights log-ml-estimate]}
        (imp/importance-sampling model [] observations n
                                {:proposal posterior-proposal :proposal-args [y] :key (rng/key 1)})]
    (is (= n (count traces)))
    (is (every? #(= y (% :y)) traces))
    (is (every? #(< (ts/abs (- % (- (Math/log n)))) (ts/tol 1e-12)) log-normalized-weights))
    (is (< (ts/abs (- log-ml-estimate log-evidence)) (ts/tol 1e-12)))))

(deftest the-prior-proposal-estimates-the-evidence
  (let [n 10000
        {:keys [log-ml-estimate log-normalized-weights]}
        (imp/importance-sampling model [] observations n {:key (rng/key 2)})]
    (is (= n (count log-normalized-weights)))
    (is (< (ts/abs (- log-ml-estimate log-evidence)) (* 5 (Math/sqrt (/ relative-variance n)))))))

(deftest ^:heavy the-estimate-is-unbiased-in-exp-space
  ;; law 20: E[exp(log-ml-estimate)] = p(y), even for few samples, where the
  ;; log of the estimate is biased low by about relative-variance / (2n)
  (let [n 5 runs 4000
        mean (/ (reduce + (map #(Math/exp (:log-ml-estimate (imp/importance-sampling model [] observations n {:key %})))
                               (rng/split (rng/key 3) runs)))
                runs)]
    (is (< (ts/abs (- (/ mean (Math/exp log-evidence)) 1))
           (* 5 (Math/sqrt (/ relative-variance (* n runs))))))))

(deftest ^:heavy resampling-approaches-the-posterior
  (let [runs 2000
        xs (map #((:trace (imp/importance-resampling model [] observations 100 {:key %})) :x)
                (rng/split (rng/key 4) runs))
        mean (/ (reduce + xs) runs)]
    (is (< (ts/abs (- mean (/ y 2))) (* 5 (Math/sqrt (/ 0.5 runs)))))))

(deftest keys-and-errors
  (testing "the same key gives the same samples"
    (is (= (imp/importance-sampling model [] observations 5 {:key (rng/key 5)})
           (imp/importance-sampling model [] observations 5 {:key (rng/key 5)}))))
  (testing "a proposal may not propose an observed address"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"overlap the observations at \[\[:y\]\]"
                          (imp/importance-sampling model [] observations 3
                                                  {:proposal (gen [] (trace :y (d/normal 0 1)))
                                                   :proposal-args []}))
        "and the error names the address")))

(deftest traces-of-either-mode
  (let [proposal (gen [] (trace :x (d/normal 0 2)))]
    (doseq [opts [{} {:batched? true} {:batched? true :proposal proposal :proposal-args []}]]
      (let [result (imp/importance-sampling model [] observations 6 (assoc opts :key (rng/key 7)))
            trs (imp/traces result)]
        (is (= 6 (count trs)) (str opts))
        (is (every? #(= y (ts/host (% :y))) trs) (str opts))))
    (testing "a custom proposal runs one sample at a time, even when batched? is asked for"
      (is (contains? (imp/importance-sampling model [] observations 3 {:batched? true :proposal proposal :proposal-args []})
                     :traces)))))

;; ---------------------------------------------------------------------------
;; A proposal for only some of the latents (Alg 14, p. 147)
;; ---------------------------------------------------------------------------

(def chain-model
  "x ~ N(0, 1), z ~ N(x, 1), y ~ N(z, 1): p(y) = N(y; 0, √3)."
  (gen []
    (let [x (trace :x (d/normal 0 1))
          z (trace :z (d/normal x 1))]
      (trace :y (d/normal z 1)))))

(def x-proposal (gen [] (trace :x (d/normal 0.5 1.5))))

(deftest ^:heavy a-partial-proposal-leaves-the-rest-to-the-internal-proposal
  (let [n 20
        {:keys [traces log-normalized-weights log-ml-estimate]}
        (imp/importance-sampling chain-model [] observations n {:proposal x-proposal :proposal-args [] :key (rng/key 6)})
        log-total (+ log-ml-estimate (Math/log n))]
    (testing "log w = log p(τ) - log q_ext(x) - log q_int(z | x)"
      (doseq [[tr lnw] (map vector traces log-normalized-weights)]
        (let [expected (- (+ (dc/log-prob (d/normal 0 1) (tr :x)) (dc/log-prob (d/normal (tr :z) 1) y))
                          (dc/log-prob (d/normal 0.5 1.5) (tr :x)))]
          (is (< (ts/abs (- (+ lnw log-total) expected)) (ts/tol 1e-12))))))
    (testing "and the estimate is unbiased in exp space"
      (let [runs 4000
            evidence (Math/exp (dc/log-prob (d/normal 0 (Math/sqrt 3)) y))
            ratios (map #(/ (Math/exp (:log-ml-estimate (imp/importance-sampling chain-model [] observations 5
                                                                                 {:proposal x-proposal :proposal-args [] :key %})))
                            evidence)
                        (rng/split (rng/key 7) runs))
            mean (/ (reduce + ratios) runs)
            sd (Math/sqrt (/ (reduce + (map #(* (- % mean) (- % mean)) ratios)) (dec runs)))]
        (is (< (ts/abs (- mean 1.0)) (* 5 (/ sd (Math/sqrt runs)))))))))
