(ns gen.inference.importance-test
  "Importance sampling on a conjugate model: x ~ N(0, 1), y ~ N(x, 1), so
   p(y) = N(y; 0, √2) and x | y ~ N(y/2, √½)."
  (:require [clojure.test :refer [deftest is testing]]
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
    (is (every? #(< (Math/abs (- % (- (Math/log n)))) 1e-12) log-normalized-weights))
    (is (< (Math/abs (- log-ml-estimate log-evidence)) 1e-12))))

(deftest the-prior-proposal-estimates-the-evidence
  (let [n 10000
        {:keys [log-ml-estimate log-normalized-weights]}
        (imp/importance-sampling model [] observations n {:key (rng/key 2)})]
    (is (= n (count log-normalized-weights)))
    (is (< (Math/abs (- log-ml-estimate log-evidence)) (* 5 (Math/sqrt (/ relative-variance n)))))))

(deftest the-estimate-is-unbiased-in-exp-space
  ;; law 20: E[exp(log-ml-estimate)] = p(y), even for few samples, where the
  ;; log of the estimate is biased low by about relative-variance / (2n)
  (let [n 5 runs 4000
        mean (/ (reduce + (map #(Math/exp (:log-ml-estimate (imp/importance-sampling model [] observations n {:key %})))
                               (rng/split (rng/key 3) runs)))
                runs)]
    (is (< (Math/abs (- (/ mean (Math/exp log-evidence)) 1))
           (* 5 (Math/sqrt (/ relative-variance (* n runs))))))))

(deftest resampling-approaches-the-posterior
  (let [runs 2000
        xs (map #((:trace (imp/importance-resampling model [] observations 100 {:key %})) :x)
                (rng/split (rng/key 4) runs))
        mean (/ (reduce + xs) runs)]
    (is (< (Math/abs (- mean (/ y 2))) (* 5 (Math/sqrt (/ 0.5 runs)))))))

(deftest keys-and-errors
  (testing "the same key gives the same samples"
    (is (= (imp/importance-sampling model [] observations 5 {:key (rng/key 5)})
           (imp/importance-sampling model [] observations 5 {:key (rng/key 5)}))))
  (testing "a proposal may not propose an observed address"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"overlap"
                          (imp/importance-sampling model [] observations 3
                                                  {:proposal (gen [] (trace :y (d/normal 0 1)))
                                                   :proposal-args []})))))
