;; The code of book/013-conditioning.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).
;; The region that shows an error is the last, so that the namespace loads.

;; book: ns
(ns book.conditioning
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as imp]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: foo
(def foo
  (gen [prob-a]
    (let [val (if (trace :a (d/bernoulli prob-a))
                (trace :b (d/bernoulli 0.6))
                true)]
      (and (trace :c (d/bernoulli (if val 0.9 0.2))) val))))

;; book: table
(def every-choice-map
  [(cm/choicemap :a true :b true :c true)
   (cm/choicemap :a true :b true :c false)
   (cm/choicemap :a true :b false :c true)
   (cm/choicemap :a true :b false :c false)
   (cm/choicemap :a false :c true)
   (cm/choicemap :a false :c false)])

(defn probability
  "p(t; prob-a): the probability that foo makes the choice map t."
  [t prob-a]
  (array/exp (:weight (gfi/assess foo [prob-a] t))))

(def table (mapv #(probability % 0.3) every-choice-map))

table
;=> [0.162 0.018 0.024 0.096 0.63 0.07]

(array/allclose? 1.0 (reduce + table))
;=> true

(mapv #(:retval (gfi/assess foo [0.3] %)) every-choice-map)
;=> [true false false false true false]

;; book: score
(def tr (gfi/simulate foo [0.3] (rng/key 1)))

(cm/to-map (:choices tr))
;=> {:a false, :c true} ; float64

(array/allclose? (:score tr) (array/log (probability (:choices tr) 0.3)))
;=> true

;; book: generate
(def constraints (cm/choicemap :a true :c false))

(def generated (gfi/generate foo [0.3] constraints (rng/key 2)))

(cm/to-map (:choices (:trace generated)))
;=> {:a true, :b false, :c false} ; float64

(:weight generated)
;=> -1.427 ; float64

;; book: extremes
(let [t (gfi/simulate foo [0.3] (rng/key 4))
      free (gfi/generate foo [0.3] cm/EMPTY (rng/key 4))
      fixed (gfi/generate foo [0.3] (:choices t) (rng/key 5))]
  [(= (mapv array/->clj (cm/values (:choices t)))
      (mapv array/->clj (cm/values (:choices (:trace free)))))
   (:weight free)
   (array/allclose? (:weight fixed) (:score t))])
;=> [true 0.0 true]

;; book: two-runs
(def two-runs (mapv #(gfi/generate foo [0.3] constraints (rng/key %)) [2 3]))

(mapv #((:trace %) :b) two-runs)
;=> [false true] ; float64

(mapv :weight two-runs)
;=> [-1.427 -3.507] ; float64

;; book: weight-by-hand
(defn weight-by-hand
  "log p(t) - log q(t) for a trace t of foo under the constraints: the
   internal proposal q drew only :b, from bernoulli(0.6)."
  [t]
  (let [q (if (array/truthy? (t :b)) 0.6 0.4)]
    (- (array/log (probability (:choices t) 0.3)) (Math/log q))))

(every? #(array/allclose? (:weight %) (weight-by-hand (:trace %))) two-runs)
;=> true

;; book: weight-as-project
(def constrained (sel/from-choicemap constraints))

(every? #(array/allclose? (:weight %) (gfi/project (:trace %) constrained))
        two-runs)
;=> true

;; book: sensor
(def sensor
  (gen [n]
    (let [mu (trace :mu (d/normal 0 2))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(defn observations
  "The choice map fixing each [addr i] to (xs i)."
  [addr xs]
  (apply cm/choicemap (mapcat (fn [i x] [[addr i] x]) (range) xs)))

(def readings [2.3 1.6 2.9 2.2])

;; book: sensor-weight
(let [{:keys [trace weight]}
      (gfi/generate sensor [4] (observations :y readings) (rng/key 6))
      mu (trace :mu)]
  (array/allclose? weight
                   (reduce + (map #(dc/log-prob (d/normal mu 1) %) readings))))
;=> true

;; book: partition
(let [t (gfi/simulate sensor [4] (rng/key 7))
      s (sel/select :mu)]
  (array/allclose? (:score t)
                   (+ (gfi/project t s) (gfi/project t (sel/complement s)))))
;=> true

;; book: underflow
(zero? (Math/exp -1000.0))
;=> true

(Math/exp (- -1001.0 -1000.0))
;=> 0.3679

(defn importance-by-hand
  "n traces of model on args under the observations, their normalized
   weights, and the log of their average weight: {:traces :weights :log-ml}."
  [model args observations n key]
  (let [samples (mapv #(gfi/generate model args observations %)
                      (rng/split key n))
        log-ws (mapv #(array/item (:weight %)) samples)
        top (apply max log-ws)
        ws (mapv #(Math/exp (- % top)) log-ws)
        total (reduce + ws)]
    {:traces (mapv :trace samples)
     :weights (mapv #(/ % total) ws)
     :log-ml (+ top (Math/log (/ total n)))}))

;; book: estimate
(defn estimate
  "The weighted average of f over traces with normalized weights (equal
   weights when none are given), and its standard error: {:mean :se}."
  ([f traces]
   (estimate f traces (vec (repeat (count traces) (/ 1.0 (count traces))))))
  ([f traces weights]
   (let [xs (mapv #(array/item (f %)) traces)
         mean (reduce + (map * weights xs))
         sq (fn [p x] (let [e (* p (- x mean))] (* e e)))]
     {:mean mean :se (Math/sqrt (reduce + (map sq weights xs)))})))

;; book: posterior-a
(def given-not-c (cm/choicemap :c false))

(def p-not-c (let [[_ tt _ tf _ f] table] (+ tt tf f)))

(def exact-a (let [[_ tt _ tf] table] (/ (+ tt tf) p-not-c)))

[p-not-c exact-a]
;=> [0.184 0.6196]

(def by-hand (importance-by-hand foo [0.3] given-not-c 2000 (rng/key 8)))

(defn a-true
  "1 when the trace's :a is true, 0 when it is false."
  [t]
  (if (array/truthy? (t :a)) 1.0 0.0))

(def a-estimate (estimate a-true (:traces by-hand) (:weights by-hand)))

a-estimate
;=> {:mean 0.612 :se 0.0146} ; float64

(< (abs (- (:mean a-estimate) exact-a)) (* 5 (:se a-estimate)))
;=> true

;; book: log-ml
(defn log-ml-se
  "The standard error of the log marginal likelihood estimate of samples
   with these normalized weights: about sqrt(1/ESS - 1/n)."
  [weights]
  (let [ess (/ 1.0 (reduce + (map * weights weights)))]
    (Math/sqrt (- (/ 1.0 ess) (/ 1.0 (count weights))))))

(< (abs (- (:log-ml by-hand) (Math/log p-not-c)))
   (* 5 (log-ml-se (:weights by-hand))))
;=> true

;; book: sensor-exact
(defn sensor-posterior
  "The exact posterior of mu given readings ys: a normal whose precision is
   the prior's, 1/2^2, plus one for each reading."
  [ys]
  (let [precision (+ (/ 1.0 4.0) (count ys))]
    (d/normal (/ (reduce + ys) precision) (Math/sqrt (/ 1.0 precision)))))

(:params (sensor-posterior readings))
;=> {:mu 2.1176 :std 0.4851}

(defn log-ml-by-bayes
  "log p(observations) by Bayes' rule at a value v of the choice at addr,
   given that choice's exact posterior:
   log p(v, observations) - log p(v | observations)."
  [model args observations addr posterior v]
  (- (:weight (gfi/assess model args (cm/set-choice observations addr v)))
     (dc/log-prob posterior v)))

(defn sensor-log-ml
  "log p(readings), by Bayes' rule at mu."
  [mu]
  (log-ml-by-bayes sensor [4] (observations :y readings)
                   :mu (sensor-posterior readings) mu))

[(sensor-log-ml 0.0) (sensor-log-ml 3.0)]
;=> [-6.113 -6.113]

;; book: built-in
(def result
  (imp/importance-sampling sensor [4] (observations :y readings) 10000
                           {:key (rng/key 12) :batched? true}))

(defn probabilities
  "The normalized weights of an importance-sampling result."
  [result]
  (mapv #(Math/exp %) (:log-normalized-weights result)))

(def sensor-traces (imp/traces result))

(def sensor-weights (probabilities result))

(:log-ml-estimate result)
;=> -6.067 ; float64

(< (abs (- (:log-ml-estimate result) (sensor-log-ml 0.0)))
   (* 5 (log-ml-se sensor-weights)))
;=> true

(w/ess (:log-normalized-weights result))
;=> 1968.4 ; float64

(< (w/ess (:log-normalized-weights result)) 5000)
;=> true

;; book: prior-vs-posterior
(def prior-mu (estimate #(% :mu) sensor-traces))

(def posterior-mu (estimate #(% :mu) sensor-traces sensor-weights))

[(< (abs (:mean prior-mu)) (* 5 (:se prior-mu)))
 (< (abs (- (:mean posterior-mu) (:mu (:params (sensor-posterior readings)))))
    (* 5 (:se posterior-mu)))]
;=> [true true]

;; book: prior-posterior-figure
;; figure: figures/013-prior-posterior.svg
(let [mus (mapv #(% :mu) sensor-traces)
      density (fn [dist] #(array/exp (dc/log-prob dist %)))]
  (svg/histogram
   [{:values mus :label "prior" :density (density (d/normal 0 2))}
    {:values mus :weights sensor-weights :label "posterior"
     :density (density (sensor-posterior readings))}]
   {:range [-6 6] :bins 60 :x-label "mu"}))

;; book: coin
(def coin
  (gen [n]
    (let [p (trace :p (d/beta 2 2))]
      (dotimes [i n]
        (trace [:flip i] (d/bernoulli p)))
      p)))

(def flips [true true false])

(defn coin-posterior
  "The exact posterior of the coin's p: beta(2 + heads, 2 + tails)."
  [flips]
  (let [heads (count (filter true? flips))]
    (d/beta (+ 2 heads) (+ 2 (- (count flips) heads)))))

(defn coin-ml
  "p(flips), by Bayes' rule at p."
  [flips p]
  (array/exp (log-ml-by-bayes coin [(count flips)] (observations :flip flips)
                              :p (coin-posterior flips) p)))

(every? #(array/allclose? 0.1 (coin-ml flips %)) [0.2 0.7])
;=> true

;; book: rejection
(defn agrees?
  "Does trace t hold the observations' values?"
  [t observations]
  (every? #(= (array/->clj (cm/get-choice (:choices t) %))
              (array/->clj (cm/get-choice observations %)))
          (cm/addresses observations)))

(defn rejection
  "The traces, among n simulations of model on args, that agree with the
   observations."
  [model args observations n key]
  (filterv #(agrees? % observations)
           (mapv #(gfi/simulate model args %) (rng/split key n))))

(def kept (rejection coin [3] (observations :flip flips) 5000 (rng/key 13)))

(count kept)
;=> 513 ; float64

(let [rate (/ (count kept) 5000.0)
      {:keys [mean se]} (estimate #(% :p) kept)]
  [(< (abs (- rate 0.1)) (* 5 (Math/sqrt (/ (* 0.1 0.9) 5000))))
   (< (abs (- mean (/ 4.0 7.0))) (* 5 se))])
;=> [true true]

;; book: solution-1
(* 3 (- (array/digamma 2.0) (array/digamma 4.0)))
;=> -2.5

(let [one-sample #(:weight (gfi/generate coin [3] (observations :flip flips) %))
      estimates (mapv one-sample (rng/split (rng/key 18) 3000))
      {:keys [mean se]} (estimate identity estimates)]
  [(< (abs (- mean -2.5)) (* 5 se))
   (< mean (- (Math/log 0.1) (* 5 se)))])
;=> [true true]

(let [r (imp/importance-sampling coin [3] (observations :flip flips) 5000
                                 {:key (rng/key 14) :batched? true})
      ws (probabilities r)
      {:keys [mean se]} (estimate #(% :p) (imp/traces r) ws)]
  [(< (abs (- (:log-ml-estimate r) (Math/log 0.1))) (* 5 (log-ml-se ws)))
   (< (abs (- mean (/ 4.0 7.0))) (* 5 se))])
;=> [true true]

;; book: solution-2
(def six [true true false true true true])

(array/allclose? (/ 1.0 42.0) (coin-ml six 0.5))
;=> true

(let [n 5000
      kept (rejection coin [6] (observations :flip six) n (rng/key 19))
      p (/ 1.0 42.0)]
  (< (abs (- (/ (count kept) (double n)) p))
     (* 5 (Math/sqrt (/ (* p (- 1 p)) n)))))
;=> true

;; book: solution-3
(def far-readings (mapv #(+ % 6.0) readings))

(def far
  (imp/importance-sampling sensor [4] (observations :y far-readings) 10000
                           {:key (rng/key 20) :batched? true}))

(w/ess (:log-normalized-weights far))
;=> 4.13 ; float64

(< (w/ess (:log-normalized-weights far)) 50)
;=> true

[(:mu (:params (sensor-posterior far-readings)))
 (estimate #(% :mu) (imp/traces far) (probabilities far))]
;=> [7.765 {:mean 7.205 :se 0.045}] ; float64

;; book: unvisited
;; error: generate: the constraint at [:b] was not visited
#_(gfi/generate foo [0.3] (cm/choicemap :a false :b true) (rng/key 3))
