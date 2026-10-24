;; The code of book/016-update-regenerate.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.update-regenerate
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as diag]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: line-model
(def line-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

(def xs [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0])

(def ys [8.23 5.87 3.99 2.59 0.23 -0.66 -3.53 -6.91 -7.24 -9.90])

(defn observations
  "The choice map fixing each [:y i] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

;; book: line-trace
(def tr
  (:trace (gfi/generate line-model [xs]
                        (cm/merge (cm/choicemap :slope -2.0 :intercept 10.0)
                                  (observations ys))
                        (rng/key 1))))

(:score tr)
;=> -16.975

;; book: log-joint
(defn log-likelihood
  "The log density of the points (xs, ys) given a line, by hand."
  [slope intercept xs ys]
  (reduce + (map (fn [x y]
                   (dc/log-prob (d/normal (+ (* slope x) intercept) 1) y))
                 xs ys)))

(defn log-joint
  "log p(slope, intercept, ys; xs), by hand: the priors and the likelihood."
  [slope intercept xs ys]
  (+ (dc/log-prob (d/normal 0 2) slope)
     (dc/log-prob (d/normal 0 10) intercept)
     (log-likelihood slope intercept xs ys)))

(array/allclose? (:score tr) (log-joint -2.0 10.0 xs ys))
;=> true

;; book: coins
(def coins
  (gen [prob-a]
    (let [a (trace :a (d/bernoulli prob-a))
          b (when a (trace :b (d/bernoulli 0.6)))
          val (if a b true)
          c (trace :c (d/bernoulli (if val 0.9 0.2)))]
      (and val c))))

;; book: table
(def table
  "Every choice map coins can make on 0.3, with its probability, by hand
   from the model."
  {{:a true :b true :c true} (* 0.3 0.6 0.9)
   {:a true :b true :c false} (* 0.3 0.6 0.1)
   {:a true :b false :c true} (* 0.3 0.4 0.2)
   {:a true :b false :c false} (* 0.3 0.4 0.8)
   {:a false :c true} (* 0.7 0.9)
   {:a false :c false} (* 0.7 0.1)})

(array/allclose? (reduce + (vals table)) 1.0)
;=> true

;; book: assess
(defn assessed
  "The probability of the choice map m (a Clojure map) under coins on 0.3,
   by assess."
  [m]
  (array/exp (:weight (gfi/assess coins [0.3] (cm/from-map m)))))

(assessed {:a true :b false :c true})
;=> 0.024

(every? (fn [[m p]] (array/allclose? (assessed m) p)) table)
;=> true

;; book: assess-missing
;; error: assess: no value for the choice at [:b]
(gfi/assess coins [0.3] (cm/choicemap :a true :c true))

;; book: update
(def moved (gfi/update tr (cm/choicemap :slope -1.5) (rng/key 2)))

(keys moved)
;=> (:trace :weight :retdiff :discard)

[((:trace moved) :slope) ((:trace moved) :intercept)]
;=> [-1.5 10.0]

(:weight moved)
;=> -40.236

(diff/unknown-change? (:retdiff moved))
;=> true

;; book: update-weight
(array/allclose? (:weight moved) (- (:score (:trace moved)) (:score tr)))
;=> true

(array/allclose? (:weight moved)
                 (- (log-joint -1.5 10.0 xs ys) (log-joint -2.0 10.0 xs ys)))
;=> true

;; book: update-args
(def xs+ (conj xs 11.0))

(def extended
  (gfi/update tr [xs+] [diff/unknown-change]
              (cm/choicemap [:y 10] -12.1) (rng/key 4)))

(array/allclose? (:weight extended)
                 (dc/log-prob (d/normal (+ (* -2.0 11.0) 10.0) 1) -12.1))
;=> true

(cm/empty? (:discard extended))
;=> true

;; book: discard
(cm/to-map (:discard moved))
;=> {:slope -2.0}

(def restored (gfi/update (:trace moved) (:discard moved) (rng/key 6)))

[((:trace restored) :slope) (cm/to-map (:discard restored))]
;=> [-2.0 {:slope -1.5}]

(array/allclose? (:weight restored) (array/negate (:weight moved)))
;=> true

;; book: outcome
(defn outcome
  "The choices of a trace of coins, as a Clojure map of host values."
  [t]
  (update-vals (cm/to-map (:choices t)) array/->clj))

(def t
  (:trace (gfi/generate coins [0.3] (cm/from-map {:a true :b false :c true})
                        (rng/key 7))))

;; book: discard-structure
(def a-false (gfi/update t (cm/choicemap :a false) (rng/key 8)))

[(outcome (:trace a-false)) (cm/to-map (:discard a-false))]
;=> [{:a false, :c true} {:a true, :b false}]

(array/allclose? (:weight a-false)
                 (array/log (/ (table {:a false :c true})
                               (table {:a true :b false :c true}))))
;=> true

(def a-back (gfi/update (:trace a-false) (:discard a-false) (rng/key 9)))

[(outcome (:trace a-back)) (cm/to-map (:discard a-back))]
;=> [{:a true, :b false, :c true} {:a false}]

;; book: update-new-choice
(def a-true (gfi/update (:trace a-false) (cm/choicemap :a true) (rng/key 10)))

(let [t' (:trace a-true)
      fresh-b (dc/log-prob (d/bernoulli 0.6) (t' :b))]
  (array/allclose? (:weight a-true)
                   (- (array/log (table (outcome t')))
                      fresh-b
                      (array/log (table {:a false :c true})))))
;=> true

;; book: regenerate
(defn moves
  "n regenerates of trace t's selection s, one per key split from key:
   each as [the new choices, the weight], in host values."
  [t s n key]
  (mapv (fn [k]
          (let [{t' :trace w :weight} (gfi/regenerate t s k)]
            [(outcome t') (array/item w)]))
        (rng/split key n)))

(def from-t (moves t (sel/select :a) 1000 (rng/key 11)))

(count (set from-t))
;=> 2

(def weights (into {} from-t))

weights
;=> {{:a true, :b false, :c true} 0.0, {:a false, :c true} 1.504}

;; book: regenerate-frequency
(let [n (count from-t)
      dropped (count (filter #(= {:a false :c true} (first %)) from-t))]
  (< (abs (- (/ dropped (double n)) 0.7))
     (* 5 (Math/sqrt (/ (* 0.7 0.3) n)))))
;=> true

;; book: regenerate-gen-jl
(let [p (table {:a true :b false :c true})
      p' (table {:a false :c true})
      q-forward 0.7
      q-back (* 0.3 0.4)]
  (array/log (/ (* p' q-back) (* p q-forward))))
;=> 1.504

;; book: project
(array/allclose? (gfi/project tr sel/all) (:score tr))
;=> true

(gfi/project tr sel/none)
;=> 0.0

(array/allclose? (gfi/project tr (sel/select :y))
                 (log-likelihood -2.0 10.0 xs ys))
;=> true

;; book: eq-4-1
(defn retained
  "The selection of the choices a regenerate of t to t' with the selection
   s kept: t's unselected addresses that t' still has."
  [t t' s]
  (let [now (set (cm/addresses (:choices t')))]
    (sel/from-paths (filter #(and (now %) (not (sel/selected? s %)))
                            (cm/addresses (:choices t))))))

(defn retained-ratio
  "Eq. 4.1 for a model that samples forward: the log ratio of the
   densities of the retained choices in t' and in t."
  [t t' s]
  (let [r (retained t t' s)]
    (- (gfi/project t' r) (gfi/project t r))))

;; book: eq-4-1-coins
(every? (fn [k]
          (let [{t' :trace w :weight} (gfi/regenerate t (sel/select :a) k)]
            (array/allclose? w (retained-ratio t t' (sel/select :a)))))
        (rng/split (rng/key 12) 50))
;=> true

;; book: regenerate-line
(def resampled (gfi/regenerate tr (sel/select :slope) (rng/key 13)))

(def new-slope ((:trace resampled) :slope))

[new-slope (:weight resampled)]
;=> [1.129 -1836.39] ; float64

(array/allclose? (:weight resampled)
                 (- (log-likelihood new-slope 10.0 xs ys)
                    (log-likelihood -2.0 10.0 xs ys))
                 {:atol 1e-4})
;=> true

;; book: regenerate-not-score
(array/allclose? (- (:score (:trace resampled)) (:score tr) (:weight resampled))
                 (- (dc/log-prob (d/normal 0 2) new-slope)
                    (dc/log-prob (d/normal 0 2) -2.0))
                 {:atol 1e-4})
;=> true

;; book: reverse
(def t' (:trace (gfi/generate coins [0.3] (cm/from-map {:a false :c true})
                              (rng/key 14))))

(def from-t' (moves t' (sel/select :a) 400 (rng/key 15)))

(def back-weights (into {} from-t'))

(count back-weights)
;=> 3

(back-weights {:a true :b false :c true})
;=> -1.504

(mapv back-weights [{:a false :c true} {:a true :b true :c true}])
;=> [0.0 0.0]

;; book: mean-model
(def mean-model
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(def measurements [2.1 1.4])

(def start
  (:trace (gfi/generate mean-model [(count measurements)]
                        (observations measurements) (rng/key 16))))

;; book: mh-step
(defn mh-step
  "One Metropolis-Hastings move on the trace tr: regenerate the selection
   s, and accept the proposal with probability min(1, exp(weight)). The
   key splits in two, for the move and for the acceptance, as mh/mh's."
  [tr s key]
  (let [[k-move k-accept] (rng/split key)
        {proposed :trace w :weight} (gfi/regenerate tr s k-move)
        u (array/item (rng/uniform k-accept))]
    (if (< (Math/log u) (array/item w)) proposed tr)))

;; book: run-chain
(defn run-chain
  "The traces of n moves from the trace tr, each (move tr s key)."
  [move tr s n key]
  (vec (reductions (fn [t k] (move t s k)) tr (rng/split key n))))

(def by-hand (run-chain mh-step start (sel/select :mu) 3000 (rng/key 17)))

;; book: same-as-library
(def by-library
  (run-chain (fn [t s k] (:trace (mh/mh t s k)))
             start (sel/select :mu) 3000 (rng/key 17)))

(= (diag/draws by-hand :mu) (diag/draws by-library :mu))
;=> true

;; book: mean-exact
(def exact
  (let [precision (+ 1.0 (count measurements))]
    {:mean (/ (reduce + measurements) precision)
     :sd (Math/sqrt (/ 1.0 precision))}))

exact
;=> {:mean 1.1667 :sd 0.5774}

;; book: histogram-figure
;; figure: figures/016-mh-histogram.svg
(let [posterior (d/normal (:mean exact) (:sd exact))]
  (svg/histogram (diag/draws (subvec by-hand 200) :mu)
                 #(array/exp (dc/log-prob posterior %))
                 {:x-label "mu"}))

;; book: mean-check
(def mu-summary (diag/summary (diag/draws (subvec by-hand 200) :mu)))

[(< (abs (- (:mean mu-summary) (:mean exact))) (* 5 (:mcse-mean mu-summary)))
 (< (abs (- (:sd mu-summary) (:sd exact))) (* 5 (:mcse-sd mu-summary)))]
;=> [true true]

(:ess-bulk mu-summary)
;=> 269.6 ; float64

;; book: solution-1
(def same (gfi/update tr (cm/choicemap :slope -2.0) (rng/key 20)))

[(:weight same) (cm/to-map (:discard same))]
;=> [0.0 {:slope -2.0}]

;; book: solution-2
(defn wrong-step
  "mh-step weighed by the score difference: the proposal forgotten."
  [tr s key]
  (let [[k-move k-accept] (rng/split key)
        {proposed :trace} (gfi/regenerate tr s k-move)
        w (- (:score proposed) (:score tr))
        u (array/item (rng/uniform k-accept))]
    (if (< (Math/log u) (array/item w)) proposed tr)))

(let [chain (run-chain wrong-step start (sel/select :mu) 3000 (rng/key 19))
      {:keys [mean mcse-mean]}
      (diag/summary (diag/draws (subvec chain 200) :mu))]
  [(< (abs (- mean (:mean exact))) (* 5 mcse-mean))
   (< (abs (- mean 0.875)) (* 5 mcse-mean))])
;=> [false true]

;; book: solution-3
(defn reach
  "For every choice map of coins, the choice maps that regenerating the
   selection s reached from it in n tries."
  [s n key]
  (into {} (map (fn [m k]
                  (let [[k-trace k-moves] (rng/split k)
                        t (:trace (gfi/generate coins [0.3] (cm/from-map m)
                                                k-trace))]
                    [m (set (map first (moves t s n k-moves)))]))
                (keys table)
                (rng/split key (count table)))))

;; book: solution-3-check
(def by-a (reach (sel/select :a) 300 (rng/key 22)))

(def by-c (reach (sel/select :c) 300 (rng/key 23)))

[(every? (fn [[m to]] (every? #(= (:c m) (:c %)) to)) by-a)
 (every? (fn [[m to]] (= (count to) (if (:a m) 2 3))) by-a)
 (every? (fn [[m to]] (= to #{m (update m :c not)})) by-c)]
;=> [true true true]

(count (set (map first (moves t' (sel/select :a :b :c) 1000 (rng/key 24)))))
;=> 6

;; book: solution-4
(def coin-chain
  (let [[k-start k-chain] (rng/split (rng/key 25))
        start (:trace (gfi/generate coins [0.3] (cm/choicemap :c true)
                                    k-start))]
    (run-chain mh-step start (sel/select :a) 2000 k-chain)))

(def a-exact
  (let [with-a (+ (table {:a true :b true :c true})
                  (table {:a true :b false :c true}))]
    (/ with-a (+ with-a (table {:a false :c true})))))

a-exact
;=> 0.2279

(let [{:keys [mean mcse-mean]} (diag/summary (diag/draws coin-chain :a))]
  (< (abs (- mean a-exact)) (* 5 mcse-mean)))
;=> true
