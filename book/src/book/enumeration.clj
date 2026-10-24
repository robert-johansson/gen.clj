;; The code of book/024-enumeration.md. Each region, from a `;; book:` line
;; to the next, is included there by name (gen.book: `clojure -M:test -m
;; gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.enumeration
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
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.mh :as mh]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: model
(defn robust-regression
  "A line through points, some of them outliers: a slope and an intercept
   from their priors, then for each x a flag, true with probability
   prob-outlier, and a y, normal about 0 with sd 5 for an outlier and
   about the line with sd 1 for an inlier."
  [{:keys [slope-prior intercept-prior prob-outlier]}]
  (gen [xs]
    (let [slope (trace :slope slope-prior)
          intercept (trace :intercept intercept-prior)]
      (doseq [[i x] (map-indexed vector xs)]
        (let [outlier? (trace [:data i :outlier?] (d/bernoulli prob-outlier))]
          (trace [:data i :y]
                 (d/normal (array/where outlier? 0.0 (+ (* slope x) intercept))
                           (array/where outlier? 5.0 1.0))))))))

(def discrete-priors
  {:slope-prior (d/uniform-discrete -2 2)
   :intercept-prior (d/uniform-discrete -2 2)
   :prob-outlier 0.1})

(def discrete-regression (robust-regression discrete-priors))

;; book: discrete-data
(def xs [-2.0 -1.0 0.0 1.0 2.0])

(def discrete-ys [-1.59 -0.92 4.0 0.14 2.18])

(defn observations
  "The choice map fixing each [:data i :y] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:data i :y] y]) (range) ys)))

;; book: discrete-data-figure
;; figure: figures/024-discrete-data.svg
(let [points (map vector xs discrete-ys)]
  (svg/plot [{:curves [(fn [x] x)] :color "gray"}
             {:points (remove #{[0.0 4.0]} points)}
             {:points [[0.0 4.0]] :color "firebrick"}]))

;; book: discrete-grid
(def flag-specs
  "A grid spec for each point's outlier flag: both of its values."
  (for [i (range (count xs))] [[:data i :outlier?] [false true]]))

(def discrete-grid
  (apply enum/choice-vol-grid
         [:slope (range -2 3)]
         [:intercept (range -2 3)]
         flag-specs))

(count discrete-grid)
;=> 800

(defn describe
  "A grid point as [slope intercept flags log-volume]."
  [[choices log-volume]]
  [(cm/get-choice choices :slope)
   (cm/get-choice choices :intercept)
   (mapv #(cm/get-choice choices [:data % :outlier?]) (range (count xs)))
   log-volume])

(describe (first discrete-grid))
;=> [-2 -2 [false false false false false] 0.0]

(describe (second discrete-grid))
;=> [-2 -2 [false false false false true] 0.0]

(describe (peek discrete-grid))
;=> [2 2 [true true true true true] 0.0]

;; book: discrete-run
(def discrete-result
  (enum/enumerative-inference discrete-regression [xs]
                              (observations discrete-ys) discrete-grid
                              {:key (rng/key 1) :batched? true}))

(sort (keys discrete-result))
;=> (:batches :log-ml-estimate :log-normalized-weights)

(count (enum/traces discrete-result))
;=> 800

(:log-ml-estimate discrete-result)
;=> -12.444

;; book: discrete-exact
(defn point-likelihood
  "p(y | slope, intercept) for the point (x, y) with its flag summed out:
   an outlier with probability prob-outlier, an inlier otherwise."
  [prob-outlier slope intercept x y]
  (let [density (fn [mean sd] (array/exp (dc/log-prob (d/normal mean sd) y)))]
    (+ (* prob-outlier (density 0.0 5.0))
       (* (- 1.0 prob-outlier) (density (+ (* slope x) intercept) 1.0)))))

(defn discrete-by-hand
  "The discrete model's answers by hand: the flags summed out point by
   point, then the 25 equally probable lines. {:log-ml, and :outliers, the
   probability that each point is an outlier}."
  [ys]
  (let [lines (for [slope (range -2 3) intercept (range -2 3)]
                [slope intercept])
        likelihood (fn [[s b]]
                     (reduce * (map #(point-likelihood 0.1 s b %1 %2) xs ys)))
        joint (mapv #(/ (likelihood %) 25.0) lines)
        evidence (reduce + joint)
        outlier (fn [i [s b] p]
                  (let [y (ys i)
                        out (* 0.1 (array/exp (dc/log-prob (d/normal 0 5) y)))]
                    (* (/ p evidence)
                       (/ out (point-likelihood 0.1 s b (xs i) y)))))]
    {:log-ml (array/log evidence)
     :outliers (mapv (fn [i] (reduce + (map #(outlier i %1 %2) lines joint)))
                     (range (count xs)))}))

(def discrete-exact (discrete-by-hand discrete-ys))

(array/allclose? (:log-ml-estimate discrete-result) (:log-ml discrete-exact))
;=> true

;; book: discrete-table
(defn probabilities
  "The posterior probabilities of an enumeration's points, in grid order."
  [result]
  (mapv #(Math/exp %) (:log-normalized-weights result)))

(def discrete-table
  "The posterior of the line: the points' probabilities in the grid's
   shape, summed over the 32 assignments of the flags."
  (-> (array/array (probabilities discrete-result))
      (array/reshape [5 5 32])
      (array/sum {:axis 2})
      array/->clj))

(get-in discrete-table [3 2])
;=> 0.684

;; book: discrete-figure
;; figure: figures/024-discrete-posterior.svg
(svg/heatmap (range -2 3) (range -2 3) discrete-table
             {:labels? true :mark [0 1] :marginals? true
              :x-label "intercept" :y-label "slope"})

;; book: discrete-marginal
(enum/marginal discrete-result :slope)
;=> {-2 1.1E-4 -1 7.0E-4 0 0.0449 1 0.9464 2 0.008}

(array/allclose?
 (array/array (mapv (enum/marginal discrete-result :slope) (range -2 3)))
 (array/array (mapv #(reduce + %) discrete-table)))
;=> true

;; book: discrete-outliers
(defn outlier-probabilities
  "The posterior probability that each point is an outlier."
  [result]
  (mapv #(get (enum/marginal result [:data % :outlier?]) true 0.0)
        (range (count xs))))

(outlier-probabilities discrete-result)
;=> [0.035 0.028 0.917 0.047 0.042]

(array/allclose? (array/array (outlier-probabilities discrete-result))
                 (array/array (:outliers discrete-exact)))
;=> true

;; book: queries
(defn expectation
  "The posterior mean of the choice at addr, from its marginal: each value
   weighted by its probability."
  [result addr]
  (reduce + (map (fn [[v p]] (* v p)) (enum/marginal result addr))))

(defn entropy
  "The entropy of the probabilities ps, in nats: the sum of -p log p."
  [ps]
  (- (reduce + (map #(if (pos? %) (* % (Math/log %)) 0.0) ps))))

[(expectation discrete-result :slope) (expectation discrete-result :intercept)]
;=> [0.961 0.03]

[(Math/log 25.0) (entropy (apply concat discrete-table))]
;=> [3.219 1.063]

;; book: continuous-model
(def continuous-priors
  {:slope-prior (d/normal 0 1)
   :intercept-prior (d/normal 0 2)
   :prob-outlier 0.1})

(def continuous-regression (robust-regression continuous-priors))

;; book: continuous-grid
(defn endpoints
  "The endpoints of the intervals of width step from lo to hi."
  [lo hi step]
  (mapv #(+ lo (* step %)) (range (inc (Math/round (/ (- hi lo) step))))))

(def line-specs
  [[:slope (endpoints -3 3 0.25) :continuous]
   [:intercept (endpoints -4 4 0.5) :continuous]])

(def continuous-grid
  (apply enum/choice-vol-grid (concat line-specs flag-specs)))

(count continuous-grid)
;=> 12288

(describe (first continuous-grid))
;=> [-2.875 -3.75 [false false false false false] -2.0794]

(array/allclose? (peek (first continuous-grid)) (Math/log (* 0.25 0.5)))
;=> true

;; book: continuous-data
(def continuous-ys [5.74 0.0 1.58 1.9 -0.64])

;; book: continuous-data-figure
;; figure: figures/024-continuous-data.svg
(let [points (map vector xs continuous-ys)]
  (svg/plot [{:curves [(fn [x] (+ (* -1.21 x) 2.56))] :color "gray"}
             {:points (remove #{[-1.0 0.0]} points)}
             {:points [[-1.0 0.0]] :color "firebrick"}]))

;; book: continuous-run
(def continuous-result
  (enum/enumerative-inference continuous-regression [xs]
                              (observations continuous-ys) continuous-grid
                              {:key (rng/key 2) :batched? true}))

(:log-ml-estimate continuous-result)
;=> -13.0441

;; book: exact-posterior
(def configurations
  "Every assignment of the five flags, in the grid's order."
  (vec (for [bits (range 32)] (mapv #(bit-test bits %) (range 4 -1 -1)))))

(defn given-flags
  "A robust regression's answers given which points are outliers (flags),
   for normal priors about 0: {:log-p, log p(flags) + log p(ys | flags),
   and :mean, the posterior mean of [slope intercept]}. The outliers' ys
   are normal about 0 with sd 5. The inliers' are jointly normal, with
   covariance X S X' + I, X their rows [x 1] and S the line's prior
   covariance, and the line's posterior is as in Getting Started."
  [{:keys [slope-prior intercept-prior prob-outlier]} ys flags]
  (let [ins (keep-indexed (fn [i out?] (when-not out? i)) flags)
        outs (keep-indexed (fn [i out?] (when out? i)) flags)
        log-flags (+ (* (count outs) (Math/log prob-outlier))
                     (* (count ins) (Math/log (- 1.0 prob-outlier))))
        log-outs (reduce + 0.0 (map #(dc/log-prob (d/normal 0 5) (ys %)) outs))
        variance #(array/square (:std (:params %)))
        S (array/diag (array/array [(variance slope-prior)
                                    (variance intercept-prior)]))]
    (if (empty? ins)
      {:log-p (+ log-flags log-outs) :mean [0.0 0.0]}
      (let [X (array/array (mapv (fn [i] [(xs i) 1.0]) ins))
            Xt (array/transpose X)
            y (array/array (mapv ys ins))
            cov (+ (array/matmul X (array/matmul S Xt)) (array/eye (count ins)))
            post (linalg/inv (+ (linalg/inv S) (array/matmul Xt X)))]
        {:log-p (+ log-flags log-outs
                   (dc/log-prob (d/mvnormal (array/zeros [(count ins)]) cov) y))
         :mean (array/->clj (array/matmul post (array/matmul Xt y)))}))))

(defn exact-posterior
  "A robust regression's exact answers, for normal priors about 0: the
   answers given each assignment of the flags, weighted by its posterior
   probability. {:log-ml :outliers :mean}"
  [priors ys]
  (let [parts (mapv #(given-flags priors ys %) configurations)
        {:keys [log-total log-weights]} (w/normalize (mapv :log-p parts))
        ps (mapv #(Math/exp %) log-weights)
        weighted (fn [f] (reduce + (map * ps (map f configurations parts))))]
    {:log-ml log-total
     :outliers (mapv (fn [i] (weighted (fn [c _] (if (c i) 1.0 0.0))))
                     (range (count xs)))
     :mean (mapv (fn [j] (weighted (fn [_ part] ((:mean part) j)))) [0 1])}))

(def continuous-exact (exact-posterior continuous-priors continuous-ys))

(:log-ml continuous-exact)
;=> -13.0439

(- (:log-ml-estimate continuous-result) (:log-ml continuous-exact))
;=> -2.37E-4 ; float64

(< 1e-4
   (- (:log-ml continuous-exact) (:log-ml-estimate continuous-result))
   1e-3)
;=> true

(outlier-probabilities continuous-result)
;=> [0.464 0.437 0.035 0.057 0.089]

(:outliers continuous-exact)
;=> [0.464 0.437 0.035 0.057 0.089]

[(expectation continuous-result :slope)
 (expectation continuous-result :intercept)]
;=> [-0.687 1.521]

(:mean continuous-exact)
;=> [-0.687 1.521]

;; book: grid-error
(defn log-ml-error
  "The error of the continuous model's log marginal likelihood from a
   grid over the line given by specs, and over the flags."
  [specs]
  (let [grid (apply enum/choice-vol-grid (concat specs flag-specs))
        result (enum/enumerative-inference continuous-regression [xs]
                                           (observations continuous-ys) grid
                                           {:key (rng/key 3) :batched? true})]
    (- (:log-ml-estimate result) (:log-ml continuous-exact))))

(def finer
  [[:slope (endpoints -3 3 0.125) :continuous]
   [:intercept (endpoints -4 4 0.25) :continuous]])

(def wider
  [[:slope (endpoints -5 5 0.25) :continuous]
   [:intercept (endpoints -8 8 0.5) :continuous]])

(log-ml-error finer)
;=> -2.97E-4 ; float64

(< (abs (log-ml-error wider)) 5e-5)
;=> true

;; book: continuous-table
(def continuous-table
  "The posterior of the line on the continuous grid: [slope][intercept]."
  (-> (array/array (probabilities continuous-result))
      (array/reshape [24 16 32])
      (array/sum {:axis 2})
      array/->clj))

(defn midpoints
  "The midpoints of the intervals between endpoints."
  [ends]
  (mapv #(/ (+ %1 %2) 2.0) ends (rest ends)))

;; book: continuous-figure
;; figure: figures/024-continuous-posterior.svg
(svg/heatmap (midpoints (endpoints -4 4 0.5)) (midpoints (endpoints -3 3 0.25))
             continuous-table
             {:mark [2.56 -1.21] :marginals? true
              :x-label "intercept" :y-label "slope"})

;; book: conditional
(def given-outlier
  (cm/merge (observations continuous-ys)
            (cm/choicemap [:data 1 :outlier?] true)))

(def conditional-grid
  (apply enum/choice-vol-grid
         (concat line-specs
                 (remove #(= [:data 1 :outlier?] (first %)) flag-specs))))

(count conditional-grid)
;=> 6144

(def conditional-result
  (enum/enumerative-inference continuous-regression [xs] given-outlier
                              conditional-grid
                              {:key (rng/key 4) :batched? true}))

(def flag-1-probability
  (Math/exp (- (:log-ml-estimate conditional-result)
               (:log-ml-estimate continuous-result))))

flag-1-probability
;=> 0.437

(array/allclose? flag-1-probability
                 ((outlier-probabilities continuous-result) 1))
;=> true

;; book: conditional-slice
(def conditional-table
  (-> (array/array (probabilities conditional-result))
      (array/reshape [24 16 16])
      (array/sum {:axis 2})))

(def sliced-table
  "The full enumeration's points where point 1 is an outlier: axis 3 of
   the grid's shape at index 1 (true), renormalized."
  (let [slice (-> (array/array (probabilities continuous-result))
                  (array/reshape [24 16 2 2 2 2 2])
                  (array/take 1 3)
                  (array/sum {:axis [2 3 4 5]}))]
    (/ slice (array/sum slice))))

(array/allclose? conditional-table sliced-table)
;=> true

;; book: conditional-figure
;; figure: figures/024-conditional-posterior.svg
(svg/heatmap (midpoints (endpoints -4 4 0.5)) (midpoints (endpoints -3 3 0.25))
             (array/->clj conditional-table)
             {:mark [2.56 -1.21] :marginals? true
              :x-label "intercept" :y-label "slope"})

;; book: misspecified-model
(def misspecified-priors
  {:slope-prior (d/normal 0 (Math/sqrt 0.5))
   :intercept-prior (d/normal 0 (Math/sqrt 0.5))
   :prob-outlier 0.25})

(def misspecified (robust-regression misspecified-priors))

(def steep-ys [-8.02 -5.34 -2.13 -1.19 2.57])

;; book: resampling
(defn line-of
  "A trace's line and flags as host values: {:slope :intercept :outliers}."
  [tr]
  {:slope (array/item (tr :slope))
   :intercept (array/item (tr :intercept))
   :outliers (mapv #(tr [:data % :outlier?]) (range (count xs)))})

(def resampled
  (mapv (fn [k]
          (line-of (:trace (imp/importance-resampling
                            misspecified [xs] (observations steep-ys) 2000
                            {:key k :batched? true}))))
        (rng/split (rng/key 5) 100)))

(defn mean-and-se
  "The mean of independent draws vs and its standard error."
  [vs]
  (let [n (count vs)
        mean (/ (reduce + vs) n)
        squares (reduce + (map #(array/square (- % mean)) vs))]
    {:mean mean :se (Math/sqrt (/ squares (* n (dec n))))}))

[(mean-and-se (map :slope resampled)) (mean-and-se (map :intercept resampled))]
;=> [{:mean 0.638 :se 0.086} {:mean -0.763 :se 0.074}] ; float64

;; book: resampled-outliers
(defn outlier-frequencies
  "The fraction of draws (line-of maps) in which each point is an outlier."
  [draws]
  (mapv (fn [i] (/ (count (filter #(get-in % [:outliers i]) draws))
                   (double (count draws))))
        (range (count xs))))

(outlier-frequencies resampled)
;=> [0.91 0.83 0.21 0.2 0.46] ; float64

;; book: mcmc
(def line-proposal
  (gen [tr]
    (trace :slope (d/normal (tr :slope) 0.5))
    (trace :intercept (d/normal (tr :intercept) 0.5))))

(defn mcmc-sweep
  "A random-walk MH move on the line, then an MH move on each flag that
   resamples it from its prior."
  [tr key]
  (let [[k-line & k-flags] (rng/split key (inc (count xs)))]
    (reduce (fn [tr [i k]]
              (:trace (mh/mh tr (sel/select [:data i :outlier?]) k)))
            (:trace (mh/mh tr line-proposal [] k-line))
            (map vector (range) k-flags))))

(defn mcmc-draws
  "The lines and flags (line-of) of n sweeps of a chain from the trace
   start, after burn-in sweeps."
  [sweep start n burn-in key]
  (->> (rng/split key (+ burn-in n))
       (reductions sweep start)
       (drop (inc burn-in))
       (mapv line-of)))

;; book: mcmc-run
(def at-the-truth
  (:trace (gfi/generate misspecified [xs]
                        (cm/merge (observations steep-ys)
                                  (cm/choicemap :slope 2.8 :intercept -2.4))
                        (rng/key 6))))

(def chain (mcmc-draws mcmc-sweep at-the-truth 10000 500 (rng/key 7)))

(defn chain-mean
  "The mean of a chain's draws vs, with its Monte Carlo standard error."
  [vs]
  (let [{:keys [mean mcse-mean]} (diag/summary vs)]
    {:mean mean :se mcse-mean}))

[(chain-mean (map :slope chain)) (chain-mean (map :intercept chain))]
;=> [{:mean 0.63 :se 0.083} {:mean -0.854 :se 0.058}] ; float64

(outlier-frequencies chain)
;=> [0.845 0.784 0.201 0.163 0.467] ; float64

;; book: misspecified-enumeration
(def wide-specs
  [[:slope (endpoints -4 4 0.25) :continuous]
   [:intercept (endpoints -4 4 0.25) :continuous]])

(def misspecified-grid
  (apply enum/choice-vol-grid (concat wide-specs flag-specs)))

(count misspecified-grid)
;=> 32768

(def misspecified-result
  (enum/enumerative-inference misspecified [xs] (observations steep-ys)
                              misspecified-grid
                              {:key (rng/key 8) :batched? true}))

[(expectation misspecified-result :slope)
 (expectation misspecified-result :intercept)]
;=> [0.689 -0.867]

(outlier-probabilities misspecified-result)
;=> [0.84 0.771 0.194 0.163 0.434]

(def misspecified-table
  (-> (array/array (probabilities misspecified-result))
      (array/reshape [32 32 32])
      (array/sum {:axis 2})
      array/->clj))

;; book: misspecified-figure
;; figure: figures/024-misspecified-posterior.svg
(svg/heatmap (midpoints (endpoints -4 4 0.25)) (midpoints (endpoints -4 4 0.25))
             misspecified-table
             {:mark [-2.4 2.8] :marginals? true
              :x-label "intercept" :y-label "slope"})

;; book: misspecified-checks
(def misspecified-exact (exact-posterior misspecified-priors steep-ys))

(array/allclose? (:log-ml-estimate misspecified-result)
                 (:log-ml misspecified-exact))
;=> true

(defn agrees?
  "Is the estimate {:mean :se} within five standard errors of x?"
  [{:keys [mean se]} x]
  (< (abs (- mean x)) (* 5 se)))

(let [[slope intercept] (:mean misspecified-exact)]
  [(agrees? (mean-and-se (map :slope resampled)) slope)
   (agrees? (mean-and-se (map :intercept resampled)) intercept)
   (agrees? (chain-mean (map :slope chain)) slope)
   (agrees? (chain-mean (map :intercept chain)) intercept)])
;=> [true true true true]

;; book: steep-mass
(defn probability-above
  "The posterior probability that the choice at addr exceeds x."
  [result addr x]
  (reduce + (keep (fn [[v p]] (when (> v x) p)) (enum/marginal result addr))))

(probability-above misspecified-result :slope 2.0)
;=> 0.114

;; book: misspecified-lines
;; figure: figures/024-misspecified-lines.svg
(let [picks (w/multinomial (:log-normalized-weights misspecified-result) 40
                           (rng/key 9))
      line (fn [i]
             (let [[choices] (misspecified-grid i)
                   slope (cm/get-choice choices :slope)
                   intercept (cm/get-choice choices :intercept)]
               (fn [x] (+ (* slope x) intercept))))]
  (svg/plot [{:curves (map line picks) :opacity 0.25}
             {:curves [(fn [x] (+ (* 2.8 x) -2.4))] :color "black"}
             {:points (map vector xs steep-ys)}]))

;; book: outliers-figure
;; figure: figures/024-outliers.svg
(let [shifted (fn [dx ps] (map vector (map #(+ % dx) xs) ps))]
  (svg/plot [{:points (shifted -0.15 (outlier-frequencies resampled))
              :color "steelblue" :r 4}
             {:points (shifted 0.0 (outlier-frequencies chain))
              :color "darkorange" :r 4}
             {:points (shifted 0.15 (outlier-probabilities misspecified-result))
              :r 4}]
            {:x-range [-2.5 2.5] :y-range [0 1] :y-label "P(outlier)"}))

;; book: hierarchical-model
(def hierarchical
  (gen [xs]
    (let [slope-var (trace :slope-var (d/inv-gamma 1 1))
          intercept-var (trace :intercept-var (d/inv-gamma 1 1))
          slope (trace :slope (d/normal 0 (array/sqrt slope-var)))
          intercept (trace :intercept (d/normal 0 (array/sqrt intercept-var)))
          prob-outlier (trace :prob-outlier (d/beta 1 1))]
      (doseq [[i x] (map-indexed vector xs)]
        (let [outlier? (trace [:data i :outlier?] (d/bernoulli prob-outlier))]
          (trace [:data i :y]
                 (d/normal (array/where outlier? 0.0 (+ (* slope x) intercept))
                           (array/where outlier? 5.0 1.0))))))))

;; book: hierarchical-enumeration
(def coarse-specs
  [[:slope (endpoints -4 4 1) :continuous]
   [:intercept (endpoints -4 4 1) :continuous]
   [:slope-var (endpoints 0 5 1) :continuous]
   [:intercept-var (endpoints 0 5 1) :continuous]
   [:prob-outlier (endpoints 0 1 0.2) :continuous]])

(def coarse-grid (apply enum/choice-vol-grid (concat coarse-specs flag-specs)))

(count coarse-grid)
;=> 256000

(def hierarchical-result
  (enum/enumerative-inference hierarchical [xs] (observations steep-ys)
                              coarse-grid
                              {:key (rng/key 10) :batched? true
                               :chunk-size 32000}))

(def hierarchical-table
  (-> (array/array (probabilities hierarchical-result))
      (array/reshape [8 8 (* 5 5 5 32)])
      (array/sum {:axis 2})
      array/->clj))

;; book: hierarchical-figure
;; figure: figures/024-hierarchical-posterior.svg
(svg/heatmap (midpoints (endpoints -4 4 1)) (midpoints (endpoints -4 4 1))
             hierarchical-table
             {:labels? true :mark [-2.4 2.8] :marginals? true
              :x-label "intercept" :y-label "slope"})

;; book: hierarchical-mode
(defn most-probable-cell
  "The [row column] of a table's largest value."
  [table]
  (apply max-key #(get-in table %)
         (for [i (range (count table)) j (range (count (first table)))] [i j])))

(let [[i j] (most-probable-cell hierarchical-table)]
  [(nth (midpoints (endpoints -4 4 1)) i) (nth (midpoints (endpoints -4 4 1)) j)
   (get-in hierarchical-table [i j])])
;=> [2.5 -2.5 0.622]

;; book: hyperparameters
(def at-truth
  (cm/merge (observations steep-ys) (cm/choicemap :slope 2.8 :intercept -2.4)))

(def hyper-specs
  [[:slope-var (endpoints 0 5 0.5) :continuous]
   [:intercept-var (endpoints 0 5 0.5) :continuous]
   [:prob-outlier (endpoints 0 1 0.1) :continuous]])

(def hyper-result
  (enum/enumerative-inference hierarchical [xs] at-truth
                              (apply enum/choice-vol-grid
                                     (concat hyper-specs flag-specs))
                              {:key (rng/key 11) :batched? true}))

(defn cell-probabilities
  "The marginal of the choice at addr on a continuous grid, in the order of
   its cells."
  [result addr]
  (mapv val (sort (enum/marginal result addr))))

;; book: conjugate
(defn normalized
  "The numbers xs scaled to sum to 1."
  [xs]
  (let [total (reduce + xs)] (mapv #(/ % total) xs)))

(defn on-cells
  "The probabilities a density f gives cells of equal width: its values at
   their midpoints, normalized, as the grid weights them."
  [f mids]
  (normalized (mapv #(array/item (f %)) mids)))

(def variance-cells (midpoints (endpoints 0 5 0.5)))

(array/allclose?
 (array/array (cell-probabilities hyper-result :slope-var))
 (array/array (on-cells #(array/exp (dc/log-prob (d/inv-gamma 1.5 4.92) %))
                        variance-cells)))
;=> true

;; book: prob-outlier
(def probability-cells (midpoints (endpoints 0 1 0.1)))

(def prob-outlier-cells
  (on-cells (fn [p] (reduce * (map #(point-likelihood p 2.8 -2.4 %1 %2)
                                   xs steep-ys)))
            probability-cells))

(array/allclose? (array/array (cell-probabilities hyper-result :prob-outlier))
                 (array/array prob-outlier-cells))
;=> true

(defn cell-mean
  "The mean of cells with these midpoints and probabilities."
  [mids ps]
  (reduce + (map * mids ps)))

[(cell-mean variance-cells (cell-probabilities hyper-result :slope-var))
 (cell-mean variance-cells
            (on-cells #(array/exp (dc/log-prob (d/inv-gamma 1 1) %))
                      variance-cells))
 (cell-mean probability-cells prob-outlier-cells)]
;=> [2.728 1.468 0.188]

;; book: prob-outlier-figure
;; figure: figures/024-prob-outlier.svg
(svg/histogram [{:values probability-cells :label "prior"}
                {:values probability-cells :label "posterior"
                 :weights (cell-probabilities hyper-result :prob-outlier)}]
               {:range [0 1] :bins 10 :x-label "prob-outlier"})

;; book: hierarchical-mcmc
(defn hierarchical-sweep
  "mcmc-sweep, then an MH move on each hyperparameter that resamples it
   from its prior."
  [tr key]
  (let [[k-sweep & ks] (rng/split key 4)]
    (reduce (fn [tr [addr k]] (:trace (mh/mh tr (sel/select addr) k)))
            (mcmc-sweep tr k-sweep)
            (map vector [:slope-var :intercept-var :prob-outlier] ks))))

(def hierarchical-chain
  (mcmc-draws hierarchical-sweep
              (:trace (gfi/generate hierarchical [xs] (observations steep-ys)
                                    (rng/key 12)))
              10000 500 (rng/key 13)))

(def steep-fraction
  (chain-mean (map #(if (> (:slope %) 1.5) 1.0 0.0) hierarchical-chain)))

steep-fraction
;=> {:mean 0.905 :se 0.031} ; float64

(> (- (:mean steep-fraction) (* 5 (:se steep-fraction))) 0.5)
;=> true

;; book: hierarchical-mcmc-figure
;; figure: figures/024-hierarchical-mcmc.svg
(let [cells (midpoints (endpoints -4 4 0.25))
      bin (fn [v] (long (Math/floor (* 4 (+ v 4)))))
      counts (frequencies (for [{:keys [slope intercept]} hierarchical-chain
                                :when (and (< -4 slope 4) (< -4 intercept 4))]
                            [(bin slope) (bin intercept)]))]
  (svg/heatmap cells cells
               (mapv (fn [i] (mapv (fn [j] (get counts [i j] 0)) (range 32)))
                     (range 32))
               {:mark [-2.4 2.8] :marginals? true
                :x-label "intercept" :y-label "slope"}))

;; book: exact-proposal
(def flag-addresses (mapv #(vector :data % :outlier?) (range (count xs))))

(def every-choice
  (enum/with-exact-proposal discrete-regression
                            (into [:slope :intercept] flag-addresses)
                            {:batched? true}))

(mapv #(:weight (gfi/generate every-choice [xs] (observations discrete-ys) %))
      [(rng/key 14) (rng/key 15)])
;=> [-12.444 -12.444]

(array/allclose? (:weight (gfi/generate every-choice [xs]
                                        (observations discrete-ys)
                                        (rng/key 14)))
                 (:log-ml discrete-exact))
;=> true

;; book: summed-grid
(def summed-regression
  (enum/with-exact-proposal continuous-regression flag-addresses
                            {:batched? true}))

(def line-grid (apply enum/choice-vol-grid line-specs))

(count line-grid)
;=> 384

(def summed-result
  (enum/enumerative-inference summed-regression [xs]
                              (observations continuous-ys) line-grid
                              {:key (rng/key 16) :batched? true}))

(array/allclose? (:log-ml-estimate summed-result)
                 (:log-ml-estimate continuous-result))
;=> true

(def summed-flags
  "The full grid's probabilities summed over the flags, point by point of
   the line's grid."
  (map #(reduce + %) (partition 32 (probabilities continuous-result))))

(array/allclose? (array/array (probabilities summed-result))
                 (array/array summed-flags))
;=> true

;; book: summed-hierarchical
(def summed-hierarchical
  (enum/enumerative-inference (enum/with-exact-proposal hierarchical
                                                         flag-addresses)
                              [xs] (observations steep-ys)
                              (apply enum/choice-vol-grid coarse-specs)
                              {:key (rng/key 17) :batched? true}))

(array/allclose? (:log-ml-estimate summed-hierarchical)
                 (:log-ml-estimate hierarchical-result))
;=> true

;; book: summed-importance
(defn importance-estimate
  "The importance estimate of the posterior mean of the choice at addr, n
   samples of model given ys, with its standard error, about the root of
   the posterior variance over the ESS: {:mean :se :ess}."
  [model ys addr n key]
  (let [run (fn [f] (imp/estimate f model [xs] (observations ys) n
                                  {:key key :batched? true}))
        {mean :estimate ess :ess} (run #(array/item (% addr)))
        {square :estimate} (run #(array/square (array/item (% addr))))]
    {:mean mean :se (Math/sqrt (/ (- square (* mean mean)) ess)) :ess ess}))

(def plain-estimate
  (importance-estimate continuous-regression continuous-ys :slope 2000
                       (rng/key 18)))

(def summed-estimate
  (importance-estimate summed-regression continuous-ys :slope 2000
                       (rng/key 18)))

[(:ess plain-estimate) (:ess summed-estimate)]
;=> [28.8 254.2] ; float64

[(< (:ess plain-estimate) 100) (> (:ess summed-estimate) 150)]
;=> [true true]

(let [slope (first (:mean continuous-exact))]
  [(agrees? plain-estimate slope) (agrees? summed-estimate slope)])
;=> [true true]

;; book: gibbs
(def summed-trace
  (:trace (gfi/generate summed-regression [xs] (observations continuous-ys)
                        (rng/key 19))))

(select-keys (mh/mh summed-trace (apply sel/select flag-addresses) (rng/key 20))
             [:accepted? :log-alpha])
;=> {:accepted? true :log-alpha 0.0}

;; book: solution-1
(def narrow-result
  (enum/enumerative-inference continuous-regression [xs]
                              (observations continuous-ys)
                              (apply enum/choice-vol-grid
                                     [:slope (endpoints -1 1 0.25) :continuous]
                                     (second line-specs)
                                     flag-specs)
                              {:key (rng/key 21) :batched? true}))

[(expectation narrow-result :slope) (first (:mean continuous-exact))]
;=> [-0.233 -0.687]

(defn edge-mass
  "The posterior probability of the first and last cells of the choice at
   addr on a continuous grid."
  [result addr]
  (let [ps (cell-probabilities result addr)] (+ (first ps) (peek ps))))

[(edge-mass narrow-result :slope) (edge-mass continuous-result :slope)]
;=> [0.223 2.8E-5]

;; book: solution-2
(defn line-sweep
  "mcmc-sweep without its moves on the flags."
  [tr key]
  (:trace (mh/mh tr line-proposal [] key)))

(def no-outliers
  (apply cm/choicemap (mapcat (fn [addr] [addr false]) flag-addresses)))

(def stuck
  (mcmc-draws line-sweep
              (:trace (gfi/generate misspecified [xs]
                                    (cm/merge at-truth no-outliers)
                                    (rng/key 22)))
              5000 500 (rng/key 23)))

(def all-inliers
  (given-flags misspecified-priors steep-ys [false false false false false]))

(:mean all-inliers)
;=> [2.111 -2.016]

(chain-mean (map :slope stuck))
;=> {:mean 2.111 :se 0.01} ; float64

(let [slope (chain-mean (map :slope stuck))]
  [(agrees? slope (first (:mean all-inliers)))
   (agrees? slope (first (:mean misspecified-exact)))])
;=> [true false]

;; book: solution-3
(defn enumerate-steep
  "The misspecified model's enumeration on the wide grid, with other priors."
  [priors]
  (enum/enumerative-inference (robust-regression priors) [xs]
                              (observations steep-ys) misspecified-grid
                              {:key (rng/key 25) :batched? true}))

(def wide-priors
  (assoc misspecified-priors
         :slope-prior (d/normal 0 2) :intercept-prior (d/normal 0 2)))

(def rare-outliers (assoc misspecified-priors :prob-outlier 0.02))

(for [priors [wide-priors rare-outliers]
      :let [result (enumerate-steep priors)]]
  [(expectation result :slope) (probability-above result :slope 2.0)])
;=> ([2.467 0.919] [2.083 0.632])

;; book: solution-4
(def misleading
  (importance-estimate (enum/with-exact-proposal misspecified flag-addresses)
                       steep-ys :slope 2000 (rng/key 26)))

[(:ess misleading) (:mean misleading) (:se misleading)]
;=> [564.4 0.35 0.031] ; float64

(/ (- (:mean misleading) (first (:mean misspecified-exact))) (:se misleading))
;=> -10.9 ; float64

(defn steep-mass
  "The probability a [slope][intercept] table on the wide grid gives the
   steep lines: slopes above 1.5 with intercepts below -1.5."
  [table]
  (let [cells (midpoints (endpoints -4 4 0.25))]
    (reduce + (for [[i slope] (map-indexed vector cells)
                    [j intercept] (map-indexed vector cells)
                    :when (and (> slope 1.5) (< intercept -1.5))]
                (get-in table [i j])))))

(def prior-table
  "The misspecified prior of the line on the wide grid's cells."
  (let [cells (midpoints (endpoints -4 4 0.25))
        ps (on-cells #(array/exp (dc/log-prob (d/normal 0 (Math/sqrt 0.5)) %))
                     cells)]
    (mapv (fn [p] (mapv #(* p %) ps)) ps)))

[(steep-mass misspecified-table) (* 2000 (steep-mass prior-table))]
;=> [0.164 0.544]

(Math/pow (- 1.0 (steep-mass prior-table)) 2000)
;=> 0.58
