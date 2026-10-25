;; The code of book/022-mcmc-map.md. Each region, from a `;; book:` line to
;; the next, is included there by name (gen.book: `clojure -M:test -m
;; gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.mcmc-map
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as diag]
            [gen.inference.importance :as is]
            [gen.inference.kernel :as k]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: model
(def regression-with-outliers
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          noise (trace :noise (d/gamma 1 1))
          prob-outlier (trace :prob-outlier (d/uniform 0 1))]
      (mapv (fn [i x]
              (let [outlier? (trace [:data i :outlier?]
                                    (d/bernoulli prob-outlier))
                    line (+ (* slope x) intercept)]
                (trace [:data i :y]
                       (d/normal (array/where outlier? 0 line)
                                 (array/where outlier? 10 noise)))))
            (range) xs))))

;; book: trace-layers
(defn trace-layers
  "svg/plot's layers for a trace: its line in black, the line plus and
   minus its noise in gray, its inliers in blue and its outliers in red."
  [tr]
  (let [[xs] (:args tr)
        is (range (count xs))
        point (fn [i] [(xs i) (tr [:data i :y])])
        outlier? (fn [i] (tr [:data i :outlier?]))
        line (fn [x] (+ (* (tr :slope) x) (tr :intercept)))]
    [{:curves [#(+ (line %) (tr :noise)) #(- (line %) (tr :noise))]
      :color "gray" :width 1}
     {:curves [line] :color "black" :opacity 1}
     {:points (map point (remove outlier? is)) :color "steelblue"}
     {:points (map point (filter outlier? is)) :color "firebrick"}]))

(def window {:x-range [-5.5 5.5] :y-range [-20 20]})

;; book: prior-figure
;; figure: figures/022-prior-trace.svg
(def xs (mapv #(+ -5.0 (* 10.0 (/ % 19.0))) (range 20)))

(def prior-trace (gfi/simulate regression-with-outliers [xs] (rng/key 16)))

(svg/plot (trace-layers prior-trace) window)

;; book: prior-trace
(defn outliers
  "The indices of the points a trace classifies as outliers."
  [tr]
  (filterv #(tr [:data % :outlier?]) (range (count (first (:args tr))))))

[(prior-trace :prob-outlier) (outliers prior-trace)]
;=> [0.4356 [0 1 8 9 11 13]] ; float64

(count (filter #(< (abs (array/item (prior-trace [:data % :y]))) 20)
               (outliers prior-trace)))
;=> 5 ; float64

;; book: near-line
(defn residual
  "How far point i of trace tr lies above the trace's line."
  [tr i]
  (let [[xs] (:args tr)]
    (- (tr [:data i :y]) (+ (* (tr :slope) (xs i)) (tr :intercept)))))

(mapv #(< (abs (array/item (residual prior-trace %)))
         (array/item (prior-trace :noise)))
      (outliers prior-trace))
;=> [true false true false false true] ; float64

;; book: prior-outliers
(def outlier-counts
  (mapv #(count (outliers (gfi/simulate regression-with-outliers [xs] %)))
        (rng/split (rng/key 6) 2000)))

(let [p (/ 6.0 21.0)
      n (count outlier-counts)
      freq (/ (count (filter #(>= % 15) outlier-counts)) (double n))]
  (< (abs (- freq p)) (* 5 (Math/sqrt (/ (* p (- 1 p)) n)))))
;=> true

;; book: synthetic
(defn synthetic-dataset
  "A y for each x: with probability 0.2 an outlier, N(0, 5), and
   otherwise on the line y = -x + 2 with noise 0.5. {:ys :outliers}, host
   numbers and booleans."
  [xs key]
  (let [point (fn [x k]
                (let [[k-outlier k-y] (rng/split k)
                      outlier? (< (array/item (rng/uniform k-outlier)) 0.2)
                      z (array/item (rng/normal k-y))]
                  [outlier? (if outlier? (* 5.0 z) (+ (- 2.0 x) (* 0.5 z)))]))
        points (mapv point xs (rng/split key (count xs)))]
    {:ys (mapv second points) :outliers (mapv first points)}))

(def dataset (synthetic-dataset xs (rng/key 1)))

;; book: data
(def ys
  [6.73 -10.75 1.82 5.11 5.09 4.31 3.80 2.39 2.40 2.67
   2.07 -2.96 0.65 1.13 -0.48 -1.20 -10.62 -1.48 1.68 -2.92])

(def true-outliers [1 2 11 16 18])

[(= ys (mapv #(/ (Math/round (* 100 %)) 100.0) (:ys dataset)))
 (= true-outliers (filterv (:outliers dataset) (range 20)))]
;=> [true true] ; float64

;; book: data-figure
;; figure: figures/022-data.svg
(defn data-layers
  "svg/plot's layers for points (xs, ys): those at the indices in
   red-ones in red, the rest in black."
  [xs ys red-ones]
  (let [point (fn [i] [(xs i) (ys i)])
        is (range (count xs))]
    [{:points (map point (remove (set red-ones) is))}
     {:points (map point (filter (set red-ones) is)) :color "firebrick"}]))

(svg/plot (data-layers xs ys true-outliers) window)

;; book: observations
(defn observations
  "The choice map fixing each [:data i :y] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:data i :y] y]) (range) ys)))

;; book: record
(defn record
  "A trace's parameters, score and classifications, as host values."
  [tr]
  {:slope (array/item (tr :slope))
   :intercept (array/item (tr :intercept))
   :noise (array/item (tr :noise))
   :prob-outlier (array/item (tr :prob-outlier))
   :score (array/item (:score tr))
   :outliers (outliers tr)})

(defn misclassified
  "How many points record r classifies otherwise than the dataset drew
   them: inliers it calls outliers, and outliers it calls inliers."
  [r]
  (let [called (set (:outliers r))
        drawn (set true-outliers)]
    (+ (count (remove drawn called)) (count (remove called drawn)))))

;; book: importance
(def importance-runs
  (mapv #(record (:trace (is/importance-resampling
                          regression-with-outliers [xs] (observations ys) 2000
                          {:batched? true :key %})))
        (rng/split (rng/key 10) 100)))

;; book: batched
(def regression-with-if
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          noise (trace :noise (d/gamma 1 1))
          prob-outlier (trace :prob-outlier (d/uniform 0 1))]
      (mapv (fn [i x]
              (if (trace [:data i :outlier?] (d/bernoulli prob-outlier))
                (trace [:data i :y] (d/normal 0 10))
                (trace [:data i :y]
                       (d/normal (+ (* slope x) intercept) noise))))
            (range) xs))))

(defn stays-batched?
  "Does a batched generate of model on the data stay one batched trace?"
  [model]
  (batched/batched? (:particles (batched/generate model [xs] (observations ys)
                                                  100 (rng/key 9)))))

[(stays-batched? regression-with-outliers) (stays-batched? regression-with-if)]
;=> [true false]

;; book: mean-se
(defn mean-se
  "The mean of host numbers and its standard error."
  [vs]
  (let [n (count vs)
        m (/ (reduce + vs) n)
        var (/ (reduce + (map #(* (- % m) (- % m)) vs)) (dec n))]
    {:mean m :se (Math/sqrt (/ var n))}))

(defn mean-of
  "The mean of (f r) over records rs."
  [f rs]
  (:mean (mean-se (map f rs))))

((juxt #(mean-of :score %) #(mean-of :noise %) #(mean-of misclassified %))
 importance-runs)
;=> [-69.0 1.67 6.31] ; float64

;; book: importance-figure
;; figure: figures/022-importance.svg
(defn line-layer
  "svg/plot's layer of the lines of records rs."
  [rs]
  {:curves (map (fn [{:keys [slope intercept]}] #(+ (* slope %) intercept)) rs)
   :opacity 0.5})

(svg/plot (cons (line-layer (take 10 importance-runs))
                (data-layers xs ys true-outliers))
          window)

;; book: start
(defn start-trace
  "A trace of the model consistent with the observations ys, and with the
   choices of init (a choice map) when given."
  ([xs ys key] (start-trace xs ys cm/EMPTY key))
  ([xs ys init key]
   (:trace (gfi/generate regression-with-outliers [xs]
                         (cm/merge (observations ys) init) key))))

(def tr (start-trace xs ys (rng/key 2)))

(:score tr)
;=> -543.2 ; float64

;; book: one-move
(def moved (mh/mh tr (sel/select :prob-outlier) (rng/key 3)))

(keys moved)
;=> (:trace :accepted? :log-alpha)

;; book: block-resimulation
(def block-resimulation-update
  (k/kern [n]
    (k/mh (sel/select :slope :intercept :noise))
    (doseq [i (range n)]
      (k/mh (sel/select [:data i :outlier?])))
    (k/mh (sel/select :prob-outlier))))

(def resimulation-sweep (block-resimulation-update 20))

(count (:accepted (resimulation-sweep tr (rng/key 4))))
;=> 22

;; book: run-chain
(defn run-chain
  "n sweeps of kernel from the trace tr: the last trace, and the record of
   each sweep's trace."
  [kernel tr n key]
  (reduce (fn [chain k]
            (let [t (:trace (kernel (:trace chain) k))]
              (-> chain (assoc :trace t) (update :sweeps conj (record t)))))
          {:trace tr :sweeps []}
          (rng/split key n)))

(defn chains
  "m chains of n sweeps of kernel, each from a trace generated with the
   observations ys, on keys split from key."
  [kernel xs ys m n key]
  (mapv (fn [k]
          (let [[k-start k-chain] (rng/split k)]
            (run-chain kernel (start-trace xs ys k-start) n k-chain)))
        (rng/split key m)))

(def resimulation-chains
  (chains resimulation-sweep xs ys 100 200 (rng/key 11)))

;; book: finals
(defn finals
  "The records of the chains' last sweeps."
  [chains]
  (map #(peek (:sweeps %)) chains))

((juxt #(mean-of :score %) #(mean-of :noise %) #(mean-of misclassified %))
 (finals resimulation-chains))
;=> [-58.9 1.43 2.93] ; float64

;; book: gap
(defn gap
  "How many standard errors of their difference the mean of as lies
   above the mean of bs."
  [as bs]
  (let [a (mean-se as)
        b (mean-se bs)]
    (/ (- (:mean a) (:mean b)) (Math/hypot (:se a) (:se b)))))

(count (filter #(> (misclassified %) 10) (finals resimulation-chains)))
;=> 5 ; float64

(> (gap (map misclassified importance-runs)
        (map misclassified (finals resimulation-chains)))
   5)
;=> true

;; book: changes
(defn changes
  "The fraction of a chain's sweeps after which (f record) differed from
   the sweep before, averaged over chains."
  [chains f]
  (mean-of (fn [chain]
             (let [vs (map f (:sweeps chain))]
               (/ (count (filter true? (map not= vs (rest vs))))
                  (double (dec (count vs))))))
           chains))

[(changes resimulation-chains :slope)
 (changes resimulation-chains :outliers)]
;=> [0.037 0.81] ; float64

;; book: reference
(defn line-choices
  "The choice map of a line [slope intercept]."
  [[slope intercept]]
  (cm/choicemap :slope slope :intercept intercept))

(defn classification
  "The choice map classifying the points at the indices is as outliers,
   and the rest of n points as inliers."
  [n is]
  (apply cm/choicemap
         (mapcat (fn [i] [[:data i :outlier?] (contains? (set is) i)])
                 (range n))))

(def reference
  (start-trace xs ys (cm/merge (line-choices [-1.0 2.0])
                               (cm/choicemap :noise 0.5 :prob-outlier 0.2)
                               (classification 20 true-outliers))
               (rng/key 20)))

;; book: outlier-probability
(defn outlier-probability
  "The exact probability that point i is an outlier given the rest of a
   trace's record r: Bayes' rule on the two explanations of its y."
  [{:keys [slope intercept noise prob-outlier]} i]
  (let [y (ys i)
        line (+ (* slope (xs i)) intercept)
        as-outlier (+ (Math/log prob-outlier)
                      (dc/log-prob (d/normal 0 10) y))
        as-inlier (+ (Math/log (- 1 prob-outlier))
                     (dc/log-prob (d/normal line noise) y))]
    (array/item (array/sigmoid (- as-outlier as-inlier)))))

(residual reference 7)
;=> -0.926

(outlier-probability (record reference) 7)
;=> 0.063

;; book: block-check
(defn block-draws
  "(f trace) after each of n moves of kernel from the trace tr."
  [kernel f tr n key]
  (mapv f (rest (reductions (fn [t k] (:trace (kernel t k)))
                            tr (rng/split key n)))))

(defn within?
  "Do draws have the mean and the standard deviation sd, each within five
   Monte Carlo standard errors?"
  [draws mean sd]
  (let [s (diag/summary draws)]
    (and (< (abs (- (:mean s) mean)) (* 5 (:mcse-mean s)))
         (< (abs (- (:sd s) sd)) (* 5 (:mcse-sd s))))))

(let [p (outlier-probability (record reference) 7)]
  (within? (block-draws (k/mh (sel/select [:data 7 :outlier?]))
                        #(if (% [:data 7 :outlier?]) 1.0 0.0)
                        reference 2000 (rng/key 21))
           p (Math/sqrt (* p (- 1 p)))))
;=> true

;; book: prob-check
(let [a (+ 1 (count true-outliers))
      b (+ 1 (- 20 (count true-outliers)))]
  (within? (block-draws (k/mh (sel/select :prob-outlier))
                        #(array/item (% :prob-outlier))
                        reference 2000 (rng/key 22))
           (/ a (double (+ a b)))
           (Math/sqrt (/ (* a b) (double (* (+ a b) (+ a b) (+ a b 1)))))))
;=> true

;; book: drift
(def line-proposal
  (gen [tr]
    (trace :slope (d/normal (tr :slope) 0.5))
    (trace :intercept (d/normal (tr :intercept) 0.5))))

(:accepted? (mh/mh tr line-proposal [] (rng/key 12)))
;=> true ; float64

;; book: drift-update
(def gaussian-drift-update
  (k/kern [n]
    (k/mh line-proposal [])
    (doseq [i (range n)]
      (k/mh (sel/select [:data i :outlier?])))
    (k/mh (sel/select :prob-outlier))
    (k/mh (sel/select :noise))))

(def drift-chains
  (chains (gaussian-drift-update 20) xs ys 150 200 (rng/key 13)))

((juxt #(mean-of :score %) #(mean-of :noise %) #(mean-of misclassified %))
 (finals drift-chains))
;=> [-45.2 0.47 0.96] ; float64

(mapv #(changes drift-chains %) [:slope :noise :outliers])
;=> [0.17 0.22 0.63] ; float64

;; book: settled
(defn settled
  "The mean score of each chain's last 100 sweeps."
  [chains]
  (mapv (fn [chain] (mean-of :score (take-last 100 (:sweeps chain))))
        chains))

(> (gap (settled drift-chains) (settled resimulation-chains)) 5)
;=> true

;; book: line-posterior
(defn line-posterior
  "The exact posterior of the line given trace tr's classifications and
   noise, a Gaussian: the means and standard deviations of the slope and
   the intercept."
  [tr]
  (let [inliers (remove (set (outliers tr)) (range (count xs)))
        X (array/array (mapv (fn [i] [(xs i) 1.0]) inliers))
        Xt (array/transpose X)
        y (array/array (mapv ys inliers))
        v (* (tr :noise) (tr :noise))
        prior-precision (array/diag (array/array [0.25 0.25]))
        covariance (linalg/inv (+ prior-precision (/ (array/matmul Xt X) v)))
        [[v-slope _] [_ v-intercept]] (array/->clj covariance)]
    {:mean (array/->clj (array/matmul covariance (/ (array/matmul Xt y) v)))
     :sd [(Math/sqrt v-slope) (Math/sqrt v-intercept)]}))

(line-posterior reference)
;=> {:mean [-0.944 1.993] :sd [0.0464 0.1288]}

;; book: drift-check
(defn line-check
  "Do n moves of kernel from the trace tr, which move the line alone, draw
   slopes and intercepts with the exact posterior's means and standard
   deviations?"
  [kernel tr n key]
  (let [lines (block-draws kernel (juxt #(array/item (% :slope))
                                        #(array/item (% :intercept)))
                           tr n key)
        {[mean-slope mean-intercept] :mean [sd-slope sd-intercept] :sd}
        (line-posterior tr)]
    [(within? (mapv first lines) mean-slope sd-slope)
     (within? (mapv second lines) mean-intercept sd-intercept)]))

(line-check (k/mh line-proposal []) reference 4000 (rng/key 23))
;=> [true true]

;; book: noise-check
(defn noise-posterior
  "The mean and standard deviation of the noise given the rest of the
   trace tr, on a grid: its log density is, up to a constant, -noise -
   m log(noise) - S/(2 noise^2), for the gamma(1, 1) prior and m inliers
   whose squared residuals sum to S."
  [tr]
  (let [inliers (remove (set (outliers tr)) (range (count xs)))
        m (count inliers)
        S (reduce + (map #(let [r (array/item (residual tr %))] (* r r))
                         inliers))
        grid (mapv #(* 0.001 %) (range 1 4001))
        log-p (fn [s] (- (- s) (* m (Math/log s)) (/ S (* 2 s s))))
        top (apply max (map log-p grid))
        w (mapv #(Math/exp (- (log-p %) top)) grid)
        total (reduce + w)
        mean (/ (reduce + (map * w grid)) total)
        var (/ (reduce + (map (fn [wi s] (* wi (* (- s mean) (- s mean))))
                              w grid))
               total)]
    {:mean mean :sd (Math/sqrt var)}))

(noise-posterior reference)
;=> {:mean 0.462 :sd 0.0922}

(let [{:keys [mean sd]} (noise-posterior reference)]
  (within? (block-draws (k/mh (sel/select :noise)) #(array/item (% :noise))
                        reference 3000 (rng/key 24))
           mean sd))
;=> true

;; book: least-squares
(defn least-squares
  "The line [slope intercept] closest to the points (xs, ys) in squared
   vertical distance."
  [xs ys]
  (let [n (count xs)
        mx (/ (reduce + xs) n)
        my (/ (reduce + ys) n)
        sxy (reduce + (map (fn [x y] (* (- x mx) (- y my))) xs ys))
        sxx (reduce + (map (fn [x] (* (- x mx) (- x mx))) xs))
        slope (/ sxy sxx)]
    [slope (- my (* slope mx))]))

(least-squares [0.0 1.0 2.0] [1.0 3.0 5.0])
;=> [2.0 1.0]

;; book: ransac
(defn random-subset
  "m distinct indices below n, drawn uniformly with key."
  [n m key]
  (->> (array/->clj (rng/uniform key [n]))
       (map-indexed vector)
       (sort-by second)
       (take m)
       (mapv first)))

(defn ransac
  "RANSAC: of iters lines, each fitted to a random subset of subset-size
   points, the first with the most points within eps of it, as
   [slope intercept]."
  [xs ys {:keys [iters subset-size eps]} key]
  (let [hypothesis
        (fn [k]
          (let [idx (random-subset (count xs) subset-size k)
                [slope intercept] (least-squares (mapv xs idx) (mapv ys idx))
                near? (fn [x y] (< (abs (- y (+ (* slope x) intercept))) eps))]
            {:line [slope intercept]
             :inliers (count (filter true? (map near? xs ys)))}))]
    (:line (reduce (fn [best h] (if (> (:inliers h) (:inliers best)) h best))
                   (map hypothesis (rng/split key iters))))))

(def ransac-params {:iters 10 :subset-size 3 :eps 1.0})

(ransac xs ys ransac-params (rng/key 30))
;=> [-0.931 2.081] ; float64

;; book: ransac-proposal
(def ransac-proposal
  (gen [tr [slope-guess intercept-guess]]
    (trace :slope (d/normal slope-guess 0.1))
    (trace :intercept (d/normal intercept-guess 1.0))))

(defn ransac-jump
  "One MH move of the line to near a RANSAC guess for the points (xs, ys),
   the guess drawn with a key of its own."
  [xs ys]
  (fn [tr key]
    (let [[k-guess k-move] (rng/split key)
          guess (ransac xs ys ransac-params k-guess)]
      (mh/mh tr ransac-proposal [guess] k-move))))

;; book: ransac-check
(line-check (ransac-jump xs ys) reference 3000 (rng/key 31))
;=> [true true]

;; book: ransac-update
(def ransac-update
  (k/kern [n jump]
    jump
    (doseq [_ (range 20)]
      (k/mh (sel/select :prob-outlier))
      (k/mh (sel/select :noise))
      (k/mh line-proposal [])
      (doseq [i (range n)]
        (k/mh (sel/select [:data i :outlier?]))))))

(defn ransac-inference
  "Five RANSAC updates with the RANSAC jump jump, from a trace generated
   with the choices init: the chain of run-chain, a record per update."
  [xs ys jump init key]
  (let [[k-start k-chain] (rng/split key)]
    (run-chain (ransac-update (count xs) jump)
               (start-trace xs ys init k-start) 5 k-chain)))

(defn ransac-start
  "The choice map of a RANSAC line for the points (xs, ys)."
  [xs ys key]
  (line-choices (ransac xs ys ransac-params key)))

(def ransac-runs
  (mapv (fn [key]
          (let [[k-guess k-run] (rng/split key)]
            (ransac-inference xs ys (ransac-jump xs ys)
                              (ransac-start xs ys k-guess) k-run)))
        (rng/split (rng/key 14) 60)))

;; book: ransac-results
((juxt #(mean-of :score %) #(mean-of :noise %) #(mean-of misclassified %))
 (finals ransac-runs))
;=> [-46.0 0.503 1.12] ; float64

(> (gap (map :score (finals ransac-runs)) (settled resimulation-chains)) 5)
;=> true

;; book: scores-figure
;; figure: figures/022-scores.svg
(defn mean-scores
  "The mean score of chains after each sweep."
  [chains]
  (apply mapv (fn [& scores] (/ (reduce + scores) (count scores)))
         (map #(map :score (:sweeps %)) chains)))

(defn steps
  "The function of x that is the (ceil x)-th of values, from the first."
  [values]
  (let [n (count values)]
    (fn [x] (nth values (max 0 (min (dec n) (dec (long (Math/ceil x)))))))))

(svg/plot [{:curves [(steps (mean-scores resimulation-chains))]
            :color "darkorange" :opacity 1}
           {:curves [(steps (mean-scores drift-chains))] :opacity 1}
           {:points (map-indexed (fn [i s] [(* 20 (inc i)) s])
                                 (mean-scores ransac-runs))
            :color "seagreen" :r 4}]
          {:x-range [0 200] :y-range [-100 -35]
           :x-label "sweep" :y-label "mean score"})

;; book: outlier-map
(defn outlier-map-update
  "The trace with each point's classification set, in turn, to the more
   probable of the two given the rest."
  [tr key]
  (let [classify (fn [t [i k]]
                   (let [[k-in k-out] (rng/split k)
                         as (fn [outlier? k]
                              (:trace (gfi/update
                                       t (cm/choicemap [:data i :outlier?]
                                                       outlier?)
                                       k)))
                         inlier (as false k-in)
                         outlier (as true k-out)]
                     (if (> (array/item (:score inlier))
                            (array/item (:score outlier)))
                       inlier
                       outlier)))
        n (count xs)]
    (reduce classify tr (map vector (range n) (rng/split key n)))))

(= (outliers (outlier-map-update reference (rng/key 40)))
   (filterv #(> (outlier-probability (record reference) %) 0.5) (range 20)))
;=> true

;; book: map-optimize
(def line-selection (sel/select :slope :intercept))

(def optimized
  (nth (iterate #(mcmc/map-optimize % line-selection
                                    {:max-step-size 1.0 :min-step-size 1e-5})
                reference)
       100))

(let [{:keys [slope intercept]} (record optimized)]
  (array/allclose? (array/array [slope intercept])
                   (array/array (:mean (line-posterior reference)))
                   {:atol 2e-3}))
;=> true

;; book: map-noise-step
;; error: normal: std must be positive
#_(mcmc/map-optimize reference (sel/select :noise) {:max-step-size 1.0})

;; book: map-update
(defn outlier-fraction
  "The fraction of a trace's points classified as outliers."
  [tr]
  (/ (count (outliers tr)) (double (count xs))))

(defn map-update
  "One round of MAP optimization: gradient steps on the line and on the
   noise, the most probable classifications, and the outlier probability
   set to the fraction of outliers, kept within [0.05, 0.5]."
  [tr key]
  (let [[k-classify k-prob] (rng/split key)
        tr (-> tr
               (mcmc/map-optimize line-selection
                                  {:max-step-size 1.0 :min-step-size 1e-5})
               (mcmc/map-optimize (sel/select :noise)
                                  {:max-step-size 1e-2 :min-step-size 1e-5})
               (outlier-map-update k-classify))
        prob (min 0.5 (max 0.05 (outlier-fraction tr)))]
    (:trace (gfi/update tr (cm/choicemap :prob-outlier prob) k-prob))))

;; book: map-inference
(defn map-inference
  "20 rounds of MAP optimization from the trace tr."
  [tr key]
  (reduce map-update tr (rng/split key 20)))

(def map-runs
  (mapv #(record (map-inference (:trace %1) %2))
        ransac-runs (rng/split (rng/key 15) 60)))

[(every? true? (map > (map :score map-runs) (map :score (finals ransac-runs))))
 (> (gap (map :score map-runs) (settled drift-chains)) 5)]
;=> [true true]

(mean-of :score map-runs)
;=> -40.73 ; float64

(count (filter #(< (abs (- (:score %) -40.41)) 0.01) map-runs))
;=> 54 ; float64

(apply min (map :score map-runs))
;=> -52.85 ; float64

;; book: map-figure
;; figure: figures/022-map.svg
(def map-trace (map-inference (:trace (first ransac-runs)) (rng/key 16)))

(svg/plot (trace-layers map-trace) window)

;; book: map-answer
((juxt :slope :intercept :noise :prob-outlier :outliers) (record map-trace))
;=> [-0.9445 1.996 0.397 0.25 [1 2 11 16 18]] ; float64

;; book: borderline
(outlier-probability (record map-trace) 7)
;=> 0.11 ; float64

(defn both-ways?
  "Did the chain call point i an outlier in some of its last 100 sweeps,
   and an inlier in others?"
  [chain i]
  (let [calls (map #(contains? (set (:outliers %)) i)
                   (take-last 100 (:sweeps chain)))]
    (boolean (and (some true? calls) (some false? calls)))))

(count (filter #(both-ways? % 7) drift-chains))
;=> 148 ; float64

(some #(both-ways? % 7) drift-chains)
;=> true

;; book: solution-1
(defn noise-ransac-jump
  "ransac-jump with RANSAC's eps set to the trace's noise."
  [xs ys]
  (fn [tr key]
    (let [[k-guess k-move] (rng/split key)
          params (assoc ransac-params :eps (array/item (tr :noise)))
          guess (ransac xs ys params k-guess)]
      (mh/mh tr ransac-proposal [guess] k-move))))

(line-check (noise-ransac-jump xs ys) reference 3000 (rng/key 50))
;=> [true true]

;; book: solution-1-runs
(def noise-ransac-runs
  (mapv (fn [key]
          (let [[k-guess k-run] (rng/split key)]
            (ransac-inference xs ys (noise-ransac-jump xs ys)
                              (ransac-start xs ys k-guess) k-run)))
        (rng/split (rng/key 51) 10)))

(mapv #(mean-of :score (finals %)) [noise-ransac-runs ransac-runs])
;=> [-50.3 -46.0] ; float64

(< (gap (map :score (finals noise-ransac-runs))
        (map :score (finals ransac-runs)))
   5)
;=> true

;; book: solution-2
(def inlier-proposal
  (gen [tr xs ys]
    (let [inliers (remove (set (outliers tr)) (range (count xs)))
          [slope intercept] (least-squares (mapv xs inliers) (mapv ys inliers))]
      (trace :slope (d/normal slope 0.5))
      (trace :intercept (d/normal intercept 0.5)))))

(line-check (k/mh inlier-proposal [xs ys]) reference 3000 (rng/key 52))
;=> [true true]

;; book: solution-3
(def far
  (start-trace xs ys (cm/merge (line-choices [3.0 -2.0])
                               (cm/choicemap :noise 2.0))
               (rng/key 53)))

(def guess (ransac xs ys ransac-params (rng/key 54)))

(/ (- 3.0 (first guess)) 0.1)
;=> 39.6 ; float64

(< (:log-alpha (mh/mh far ransac-proposal [guess] (rng/key 55))) -100)
;=> true

(:weight (gfi/assess ransac-proposal [far guess] (line-choices [3.0 -2.0])))
;=> -791.0 ; float64

;; book: solution-3-runs
(def unstarted-runs
  (mapv #(ransac-inference xs ys (ransac-jump xs ys) cm/EMPTY %)
        (rng/split (rng/key 56) 10)))

(mapv #(Math/round (:score %)) (finals unstarted-runs))
;=> [-70 -47 -42 -41 -42 -47 -47 -46 -50 -42] ; float64

;; book: solution-4
(def ransac-inside-proposal
  (gen [tr xs ys]
    (let [[slope-guess intercept-guess]
          (ransac xs ys ransac-params (untraced-key))]
      (trace :slope (d/normal slope-guess 0.1))
      (trace :intercept (d/normal intercept-guess 1.0)))))

(line-check (k/mh ransac-inside-proposal [xs ys]) reference 3000
            (rng/key 57))
;=> [true true]
