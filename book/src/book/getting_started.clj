;; The code of book/011-getting-started.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.getting-started
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as diag]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: model
(def line-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

;; book: simulate
(def xs [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0])

(def tr (gfi/simulate line-model [xs] (rng/key 1)))

;; book: same-key
(= (mapv array/->clj (cm/values (:choices tr)))
   (mapv array/->clj (cm/values (:choices (gfi/simulate line-model [xs] (rng/key 1))))))
;=> true

;; book: choices
[(tr :slope) (tr :intercept) (tr [:y 0])]
;=> [1.432 -5.475 -5.395] ; float64

;; book: addresses
(count (cm/addresses (:choices tr)))
;=> 12

(take 4 (cm/addresses (:choices tr)))
;=> [[:intercept] [:slope] [:y 0] [:y 1]]

;; book: retval
(= (:retval tr) [(tr :slope) (tr :intercept)])
;=> true

;; book: score
(def score-by-hand
  (let [[slope intercept] (:retval tr)]
    (reduce + (+ (dc/log-prob (d/normal 0 2) slope)
                 (dc/log-prob (d/normal 0 10) intercept))
            (map-indexed (fn [i x] (dc/log-prob (d/normal (+ (* slope x) intercept) 1) (tr [:y i])))
                         xs))))

(- (:score tr) score-by-hand)
;=> 0.0

;; book: data
(def ys [8.23 5.87 3.99 2.59 0.23 -0.66 -3.53 -6.91 -7.24 -9.90])

(def prior-lines
  (mapv #(:retval (gfi/simulate line-model [xs] %)) (rng/split (rng/key 2) 5)))

;; book: prior-figure
;; figure: figures/011-prior-lines.svg
(svg/lines xs ys prior-lines {:title "Five lines from the prior, and the data"})

;; book: observations
(defn observations
  "The choice map fixing each [:y i] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

;; book: mh
(defn mh-sweep
  "One MH move on the slope, then one on the intercept."
  [tr key]
  (let [[k1 k2] (rng/split key)]
    (-> tr
        (mh/mh (sel/select :slope) k1) :trace
        (mh/mh (sel/select :intercept) k2) :trace)))

(defn inference-program
  "The chain of traces from num-iters MH sweeps, starting from a trace
   consistent with the observations ys."
  [xs ys num-iters key]
  (let [[k-start k-chain] (rng/split key)
        start (:trace (gfi/generate line-model [xs] (observations ys) k-start))]
    (vec (reductions mh-sweep start (rng/split k-chain num-iters)))))

(def chain (inference-program xs ys 1000 (rng/key 3)))

;; book: answer
(:retval (peek chain))

;; book: exact
(def exact
  (let [X (array/array (mapv (fn [x] [x 1.0]) xs))
        Xt (array/transpose X)
        precision (+ (array/diag (array/array [(/ 1.0 4.0) (/ 1.0 100.0)])) (array/matmul Xt X))
        covariance (linalg/inv precision)
        [[v-slope c] [_ v-intercept]] (array/->clj covariance)]
    {:mean (array/->clj (array/matmul covariance (array/matmul Xt (array/array ys))))
     :sd [(Math/sqrt v-slope) (Math/sqrt v-intercept)]
     :correlation (/ c (Math/sqrt (* v-slope v-intercept)))}))

(:mean exact)
;=> [-1.979 10.143]

(:sd exact)
;=> [0.1097 0.6807]

;; book: answer-check
(let [[slope intercept] (:retval (peek chain))
      [mu-slope mu-intercept] (:mean exact)
      [sd-slope sd-intercept] (:sd exact)]
  [(< (abs (- (array/item slope) mu-slope)) (* 5 sd-slope))
   (< (abs (- (array/item intercept) mu-intercept)) (* 5 sd-intercept))])
;=> [true true]

;; book: summary
(def burned (subvec chain 200))

(def slope-summary (diag/summary (diag/draws [burned] :slope)))

(select-keys slope-summary [:mean :mcse-mean :ess-bulk])

;; book: ess
(< (:ess-bulk slope-summary) 20)
;=> true

;; book: chain-figure
;; figure: figures/011-slope-chain.svg
(svg/series (map #(% :slope) chain) {:title "The slope along the chain" :y-label "slope"})

;; book: acceptance
(defn acceptance
  "The fraction of sweeps that moved the choice at addr."
  [chain addr]
  (/ (count (filter true? (map #(not= (array/->clj (%1 addr)) (array/->clj (%2 addr))) chain (rest chain))))
     (double (dec (count chain)))))

[(acceptance chain :slope) (acceptance chain :intercept)]

(< (max (acceptance chain :slope) (acceptance chain :intercept)) 0.1)
;=> true

;; book: correlation
(:correlation exact)
;=> -0.886

;; book: posterior-figure
;; figure: figures/011-posterior-lines.svg
(svg/lines xs ys (mapv :retval (take-nth 40 burned)) {:title "Lines from the posterior"})

;; book: solution-1
(def line-model-2
  (gen [xs]
    (let [slope (trace :slope (d/normal 1 3))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

(let [t (gfi/simulate line-model-2 [xs] (rng/key 4))
      [slope intercept] (:retval t)]
  (- (:score t)
     (reduce + (+ (dc/log-prob (d/normal 1 3) slope) (dc/log-prob (d/normal 0 10) intercept))
             (map-indexed (fn [i x] (dc/log-prob (d/normal (+ (* slope x) intercept) 1) (t [:y i]))) xs))))
;=> 0.0

;; book: solution-2
(defn exact-sd
  "The exact posterior's standard deviations, from points xs."
  [xs]
  (let [X (array/array (mapv (fn [x] [x 1.0]) xs))
        Xt (array/transpose X)]
    (array/->clj (array/sqrt (array/diag (linalg/inv (+ (array/diag (array/array [0.25 0.01])) (array/matmul Xt X))))))))

[(first (exact-sd (subvec xs 0 3))) (first (exact-sd xs))]
;=> [0.661 0.1097]

;; book: solution-3
(def random-walk
  (gen [tr]
    (trace :slope (d/normal (tr :slope) 0.1))
    (trace :intercept (d/normal (tr :intercept) 0.5))))

(def walk-chain
  (let [[k-start k-chain] (rng/split (rng/key 5))]
    (vec (reductions (fn [tr k] (:trace (mh/mh tr random-walk [] k)))
                     (:trace (gfi/generate line-model [xs] (observations ys) k-start))
                     (rng/split k-chain 2000)))))

(def walk-summary (diag/summary (diag/draws [(subvec walk-chain 200)] :slope)))

[(> (acceptance walk-chain :slope) 0.3)
 (> (:ess-bulk walk-summary) (* 3 (:ess-bulk slope-summary)))]
;=> [true true]

;; book: solution-3-check
(< (abs (- (:mean walk-summary) (first (:mean exact)))) (* 5 (:mcse-mean walk-summary)))
;=> true
