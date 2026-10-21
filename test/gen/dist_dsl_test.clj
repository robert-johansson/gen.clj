(ns gen.dist-dsl-test
  "Pushforward distributions (Gen.jl's @dist): densities against closed
   forms, normalization (which a wrong Jacobian breaks), the discrete case
   with repeated labels, labels that are not numbers, sampling, batched
   runs, the GFI laws inside a model, loud errors."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dist.dsl :refer [dist]]
            [gen.dynamic :refer [gen]]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(dist shifted-exponential [rate shift] (+ (d/exponential rate) shift))
(dist log-normal [mu sd] (array/exp (d/normal mu sd)))
(dist reciprocal [mu sd c] (/ c (d/normal mu sd)))
(dist chained [mu sd] (* 2.0 (array/exp (+ (d/normal mu sd) 1.0))))
(dist labeled [probs labels] (array/take labels (d/categorical probs)))
(dist labeled-cat [labels probs] (nth labels (d/categorical probs)))
(def ^:private colors {false :red true :green})
(dist color [p] (get colors (d/bernoulli p)))
(defrecord ^:private Label [name])

(dist doubled-poisson [rate] (* 2.0 (d/poisson rate)))
(dist exp-geometric [p] (array/exp (d/geometric p)))
;; dist_dsl.jl:20's vec_normal, through Gen.jl's alias: (def broadcasted-normal
;; normal) lacked the constructor mark, so this failed to expand (M5)
(dist vec-normal [mu sigma] (d/broadcasted-normal (+ mu 1.0) (* sigma 2.0)))

(defn- lp [d v] (ts/number (dc/log-prob d v)))

(deftest discrete-draws-without-a-support-invert-without-jacobians
  ;; the review's finding: a discrete draw is no longer taken for continuous
  (is (ts/close? (lp (d/poisson 3.0) 2) (lp (doubled-poisson 3.0) 4.0)))
  (is (= ##-Inf (lp (doubled-poisson 3.0) 3.0)) "an odd value is no image")
  (is (ts/close? (lp (d/geometric 0.5) 2) (lp (exp-geometric 0.5) (Math/exp 2.0))))
  (is (ts/close? 1.0 (reduce + (map #(Math/exp (lp (doubled-poisson 3.0) (* 2.0 %))) (range 60)))) "normalized")
  (is (dc/discrete? (doubled-poisson 3.0)))
  (is (not (dc/discrete? (log-normal 0.2 0.7)))))
(defn- normal-lp [x mu sd] (- (- (* 0.5 (Math/pow (/ (- x mu) sd) 2))) (Math/log (* sd (Math/sqrt (* 2 Math/PI))))))

(deftest densities-match-the-closed-forms
  (testing "a shift"
    (doseq [v [1.5 2.0 3.7]]
      (is (ts/close? (- (Math/log 2.0) (* 2.0 (- v 1.0))) (lp (shifted-exponential 2.0 1.0) v) {:rtol 1e-12 :atol 1e-12})))
    (is (= ##-Inf (lp (shifted-exponential 2.0 1.0) 0.5)) "below the shift"))
  (testing "exp"
    (doseq [v [0.3 1.0 4.2]]
      (is (ts/close? (- (normal-lp (Math/log v) 0.2 0.7) (Math/log v)) (lp (log-normal 0.2 0.7) v) {:rtol 1e-12 :atol 1e-12})))
    (is (= ##-Inf (lp (log-normal 0.2 0.7) -1.0)) "outside exp's range"))
  (testing "the draw in a denominator"
    (doseq [v [-2.0 0.5 3.0]]
      (is (ts/close? (+ (normal-lp (/ 1.5 v) 0.4 1.1) (Math/log 1.5) (* -2 (Math/log (Math/abs v))))
                     (lp (reciprocal 0.4 1.1 1.5) v) {:rtol 1e-12 :atol 1e-12}))))
  (testing "a chain of steps"
    (doseq [v [0.5 3.0 9.0]]
      (let [x (- (Math/log (/ v 2.0)) 1.0)]
        (is (ts/close? (- (normal-lp x 0.1 0.6) (Math/log (/ v 2.0)) (Math/log 2.0)) (lp (chained 0.1 0.6) v)
                       {:rtol 1e-12 :atol 1e-12}))))))

(deftest continuous-densities-are-normalized
  ;; a wrong Jacobian (a factor 2, a missing log v) leaves the density
  ;; integrating to something else than 1
  (doseq [[label d lo hi] [["shift" (shifted-exponential 2.0 1.0) 1.0 15.0]
                           ["exp" (log-normal 0.2 0.5) 1e-6 30.0]
                           ["chain" (chained 0.1 0.4) 1e-6 60.0]]]
    (let [n 200000
          h (/ (- hi lo) n)
          total (* h (reduce clojure.core/+ (for [i (range n)] (Math/exp (lp d (+ lo (* h (+ i 0.5))))))))]
      (is (< (Math/abs (- total 1.0)) 2e-3) (str label ": " total)))))

(deftest a-discrete-draw-sums-the-mass-of-its-labels
  (let [d (labeled [0.2 0.5 0.3] [10 20 30])]
    (is (ts/close? (Math/log 0.5) (lp d 20)))
    (is (= ##-Inf (lp d 25)) "not a label")
    (is (= [10.0 20.0 30.0] (mapv ts/number (dc/support d)))))
  (testing "repeated labels add up, and the support lists each once"
    (let [d (labeled [0.2 0.5 0.3] [10 10 30])]
      (is (ts/close? (Math/log 0.7) (lp d 10)))
      (is (= [10.0 30.0] (mapv double (dc/support d))))
      (is (ts/close? 1.0 (reduce + (map #(Math/exp (lp d %)) (dc/support d)))))))
  (testing "samples are labels"
    (is (every? #{10.0 20.0 30.0} (map double (ts/host (dc/sample (labeled [0.2 0.5 0.3] [10 20 30]) (rng/key 1) [50])))))))

(deftest labels-that-are-not-numbers
  (let [d (labeled-cat [:a :b :c] [0.2 0.5 0.3])]
    (is (ts/close? (Math/log 0.5) (lp d :b)))
    (is (= ##-Inf (lp d :z)) "not a label")
    (is (= ##-Inf (lp d 1)) "nor a number, the index of one")
    (is (= [:a :b :c] (dc/support d)))
    (is (ts/close? 1.0 (reduce + (map #(Math/exp (lp d %)) (dc/support d)))) "normalized"))
  (testing "repeated labels add up, and the support lists each once"
    (let [d (labeled-cat [:a :b :a] [0.2 0.5 0.3])]
      (is (ts/close? (Math/log 0.5) (lp d :a)))
      (is (= [:a :b] (dc/support d)))))
  (testing "a map's labels, and records, which compare by value"
    (is (ts/close? (Math/log 0.25) (lp (color 0.25) :green)))
    (is (ts/close? 0.0 (lp (labeled-cat [(->Label :a)] [1.0]) (->Label :a)))))
  (testing "a label has a gradient for its probabilities: categorical's at its index"
    (let [probs (array/array [0.2 0.5 0.3])
          g (fn [f v] ((array/grad #(f v %)) probs))]
      (is (ts/close? (g #(dc/log-prob (d/categorical %2) %1) 1) (g #(dc/log-prob (labeled-cat [:a :b :c] %2) %1) :b)))))
  (testing "draws are labels, in proportion, on every backend (the draw is read where a lookup needs it)"
    (let [n 2000
          draws (frequencies (map #(dc/sample (labeled-cat [:a :b :c] [0.2 0.5 0.3]) %) (rng/split (rng/key 5) n)))]
      (is (= #{:a :b :c} (set (keys draws))))
      ;; the count of :b, binomial(n, 0.5): five standard errors
      (is (< (Math/abs (- (draws :b) (* 0.5 n))) (* 5 (Math/sqrt (* n 0.25)))))))
  (testing "a batch of labels is an error, as no array holds them"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not numbers, so it draws one at a time"
                          (dc/sample (labeled-cat [:a :b] [0.5 0.5]) (rng/key 1) [4]))))
  (testing "a family's parameters that are not ^:host must be arrays or their data"
    (is (thrown? clojure.lang.ExceptionInfo (d/categorical [:a :b])))))

(def ^:private labeled-model
  (gen [] (let [k (trace :k (labeled-cat [:x :y :z] [0.2 0.5 0.3]))]
            (trace :v (d/normal (if (= k :y) 1.0 0.0) 1))
            k)))

(deftest labels-in-a-batched-run
  (testing "a draw is read per particle: the run falls back"
    (let [ps (b/simulate labeled-model [] 8 (rng/key 6))]
      (is (not (b/batched? ps)))
      (is (:gen.array/batched-branch (:data (b/fallback-reason ps))))
      (is (every? #{:x :y :z} (map #(% :k) (:traces ps))))))
  (testing "a given label is scored on the host for every particle: no fallback"
    (let [{ps :particles w :weights} (b/generate labeled-model [] (cm/choicemap :k :z) 8 (rng/key 7))]
      (is (b/batched? ps))
      (is (ts/close? (array/full [8] (Math/log 0.3)) w)))))

(deftest sampling-follows-the-body
  (let [n 20000
        xs (ts/host (dc/sample (shifted-exponential 2.0 1.0) (rng/key 2) [n]))
        mean (/ (reduce clojure.core/+ xs) n)]
    (is (every? #(>= % 1.0) xs) "every draw is past the shift")
    ;; mean 1 + 1/2, sd 1/2: five standard errors
    (is (< (Math/abs (- mean 1.5)) (* 5 (/ 0.5 (Math/sqrt n)))))))

(def models
  [[:pushforward-in-a-model
    (gen [rate] (let [x (trace :x (shifted-exponential rate 1.0))] (trace :y (d/normal x 1)) x))
    [2.0] [[3.0]]]
   [:discrete-pushforward
    (gen [] (let [k (trace :k (labeled [0.2 0.5 0.3] [1 2 3]))] (trace :y (d/normal k 1)) k))
    [] []]
   [:labels-in-a-model labeled-model [] []]])

(laws/deflaws pushforward models 60)

(deftest misuse-is-loud
  (letfn [(definition-error [form]
            (try (eval form) nil
                 (catch Exception e (ex-message (or (ex-cause e) e)))))]
    (is (re-find #"exactly one random draw, it makes 0"
                 (definition-error '(gen.dist.dsl/dist no-draw [a] (gen.array/+ a 1.0)))))
    (is (re-find #"exactly one random draw, it makes 2"
                 (definition-error '(gen.dist.dsl/dist two-draws [] (gen.array/+ (gen.dist/normal 0 1) (gen.dist/normal 0 1))))))
    (is (re-find #"the body is one expression"
                 (definition-error '(gen.dist.dsl/dist two-forms [] (gen.dist/normal 0 1) 2.0)))))
  (testing "a continuous draw through a step that cannot be inverted"
    (dist through-sin [] (array/sin (d/normal 0 1)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not an invertible step"
                          (dc/log-prob (through-sin) 0.5))))
  (testing "the draw's own checks, at construction"
    (is (thrown? clojure.lang.ExceptionInfo (shifted-exponential -1.0 0.0)))))

(deftest an-alias-of-a-constructor-is-a-draw
  (let [mu (array/array [0.0 1.0])
        sigma (array/array [1.0 2.0])
        v (array/array [0.5 -1.0])]
    (is (ts/close? (dc/log-prob (d/normal (+ mu 1.0) (* sigma 2.0)) v)
                   (dc/log-prob (vec-normal mu sigma) v)))))
