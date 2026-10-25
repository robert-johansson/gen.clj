;; The code of book/021-inference-toolkit.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.inference-toolkit
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
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.kernel :as k]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: model
(def line-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

(def xs [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0])

(def ys [2.6 4.3 7.1 8.4 10.8 12.3 14.0 16.9])

(defn observations
  "The choice map fixing each [addr i] to (vs i)."
  [addr vs]
  (apply cm/choicemap (mapcat (fn [i v] [[addr i] v]) (range) vs)))

;; book: exact
(def prior-precision
  "The prior's precision matrix: the inverse of its covariance, 100 I."
  (array/diag (array/array [0.01 0.01])))

(defn design
  "The matrix of rows (x, 1)."
  [xs]
  (array/array (mapv (fn [x] [x 1.0]) xs)))

(defn posterior
  "The exact posterior of the line given the points (xs, ys), a normal of
   [slope intercept]: its precision matrix, mean and covariance."
  [xs ys]
  (let [X (design xs)
        precision (+ prior-precision (array/matmul (array/transpose X) X))]
    {:precision precision
     :mean (linalg/solve precision (array/matmul (array/transpose X)
                                                 (array/array ys)))
     :cov (linalg/inv precision)}))

(def exact (posterior xs ys))

(defn sds
  "The standard deviations of a posterior's slope and intercept."
  [{:keys [cov]}]
  (array/->clj (array/sqrt (array/diag cov))))

[(array/->clj (:mean exact)) (sds exact)]
;=> [[1.9835 0.6233] [0.1539 0.7768]]

(def slope-mean (first (array/->clj (:mean exact))))

(def slope-sd (first (sds exact)))

;; book: exact-log-ml
(defn log-ml
  "log p(ys; xs), by Bayes' rule at the line [s b]: the log joint density
   less the exact posterior's log density there."
  [xs ys [s b]]
  (let [{:keys [mean cov]} (posterior xs ys)
        line (cm/choicemap :slope s :intercept b)]
    (- (:weight (gfi/assess line-model [xs]
                            (cm/merge line (observations :y ys))))
       (dc/log-prob (d/mvnormal mean cov) (array/array [s b])))))

(def exact-log-ml (log-ml xs ys [2.0 0.5]))

[exact-log-ml (log-ml xs ys [0.0 0.0])]
;=> [-15.354 -15.354]

;; book: importance
(def n-samples 4000)

(def is-result
  (imp/importance-sampling line-model [xs] (observations :y ys) n-samples
                           {:key (rng/key 1) :batched? true}))

(sort (keys is-result))
;=> (:ess :log-ml-estimate :log-normalized-weights :log-weights :particles)

[(:ess is-result) (:log-ml-estimate is-result)]
;=> [5.80 -16.56] ; float64

;; book: efficiency
(defn efficiency
  "The fraction of importance samples from the prior that count, as their
   number grows: 1 / (the integral of p^2/q), for q the prior and p the
   exact posterior of the line given the points (xs, ys)."
  [xs ys]
  (let [{P :precision m :mean} (posterior xs ys)
        P0 prior-precision
        A (- (* 2.0 P) P0)
        log-integral (+ (array/log (linalg/det P))
                        (* -0.5 (array/log (linalg/det P0)))
                        (* -0.5 (array/log (linalg/det A)))
                        (array/sum (* (array/matmul P m)
                                      (linalg/solve A (array/matmul P0 m)))))]
    (array/item (array/exp (- log-integral)))))

(def eff (efficiency xs ys))

[eff (* eff n-samples)]
;=> [0.001063 4.25]

;; book: collapse
(defn first-points
  "The first m points, as [xs ys]."
  [m]
  [(subvec xs 0 m) (subvec ys 0 m)])

(mapv #(apply efficiency (first-points %)) [1 2 4 8])
;=> [0.098 0.0187 0.00432 0.00106]

;; book: collapse-check
(defn ess-fractions
  "ESS/n of runs of importance sampling with n samples on the first m
   points, one run per key."
  [m n ks]
  (let [[xs' ys'] (first-points m)]
    (mapv #(/ (:ess (imp/importance-sampling line-model [xs']
                                             (observations :y ys') n
                                             {:key % :batched? true}))
              n)
          ks)))

(defn sd-of
  "The sample standard deviation of the numbers vs."
  [vs]
  (let [mean (/ (reduce + vs) (count vs))]
    (Math/sqrt (/ (reduce + (map #(let [e (- % mean)] (* e e)) vs))
                  (dec (count vs))))))

(defn mean-se
  "The mean of the numbers vs, and its standard error."
  [vs]
  {:mean (/ (reduce + vs) (count vs))
   :se (/ (sd-of vs) (Math/sqrt (count vs)))})

(every? (fn [m]
          (let [{:keys [mean se]} (mean-se (ess-fractions m n-samples
                                                          (rng/split
                                                           (rng/key m) 10)))]
            (< (abs (- mean (apply efficiency (first-points m)))) (* 5 se))))
        [1 2 4])
;=> true

;; book: collapse-noise
(let [fs (ess-fractions 8 n-samples (rng/split (rng/key 8) 10))]
  [(* n-samples (apply min fs)) (* n-samples (apply max fs))])
;=> [2.15 6.36] ; float64

;; book: log-ml-se
(def relative-se (Math/sqrt (/ (- (/ 1.0 eff) 1.0) n-samples)))

relative-se
;=> 0.485

(< (abs (- (Math/exp (- (:log-ml-estimate is-result) exact-log-ml)) 1.0))
   (* 5 relative-se))
;=> true

;; book: estimate
(def slope-estimate
  (imp/estimate #(array/item (% :slope)) line-model [xs] (observations :y ys)
                n-samples {:key (rng/key 2)}))

slope-estimate
;=> {:estimate 2.048 :log-ml-estimate -15.566 :ess 5.65} ; float64

(def rule-se (/ slope-sd (Math/sqrt (* eff n-samples))))

[rule-se (< (abs (- (:estimate slope-estimate) slope-mean)) (* 5 rule-se))]
;=> [0.0747 true]

;; book: estimate-spread
(defn weighted-mean
  "The weighted mean of the choice at addr over the samples of a batched
   importance-sampling result: batched/choice reads it across them."
  [result addr]
  (array/item (array/sum (* (array/exp (array/array
                                        (:log-normalized-weights result)))
                            (batched/choice (:particles result) addr)))))

(def runs
  "Forty batched runs: each one's weighted mean of the slope and its
   estimate of p(y) over the exact p(y)."
  (mapv #(let [r (imp/importance-sampling line-model [xs] (observations :y ys)
                                          n-samples {:key % :batched? true})]
           {:slope (weighted-mean r :slope)
            :ratio (Math/exp (- (:log-ml-estimate r) exact-log-ml))})
        (rng/split (rng/key 13) 40)))

(def slope-means (mapv :slope runs))

(sd-of slope-means)
;=> 0.0569 ; float64

(let [{:keys [mean se]} (mean-se slope-means)
      sd (sd-of slope-means)]
  [(< (abs (- mean slope-mean)) (* 5 se))
   (< (abs (- sd rule-se)) (* 5 (/ sd (Math/sqrt (* 2 39)))))])
;=> [true true]

(< (abs (- (:mean (mean-se (mapv :ratio runs))) 1.0))
   (* 5 (/ relative-se (Math/sqrt 40))))
;=> true

(Math/round (/ (Math/pow (/ slope-sd 0.01) 2) eff))
;=> 222914

;; book: weights
(def toy-log-weights (mapv #(Math/log %) [10.0 6.0 3.0 1.0]))

(mapv #(Math/exp %) (:log-normalized-weights (w/normalize toy-log-weights)))
;=> [0.5 0.3 0.15 0.05]

[(w/ess toy-log-weights) (w/log-mean-exp toy-log-weights) (Math/log 5.0)]
;=> [2.7397 1.6094 1.6094]

;; book: resample
(def schemes [:multinomial :systematic :stratified :residual])

(defn counts
  "How many times each of the four samples is among the indices."
  [indices]
  (mapv (fn [i] (count (filter #{i} indices))) (range 4)))

(mapv #(counts (w/resample % toy-log-weights 10 (rng/key 9))) schemes)
;=> [[5 5 0 0] [5 3 1 1] [5 3 1 1] [5 3 1 1]] ; float64

;; book: systematic
(def expected-counts [5.0 3.0 1.5 0.5])

(def repeats
  "Each method's counts for 2000 keys."
  (into {} (map (fn [m]
                  [m (mapv #(counts (w/resample m toy-log-weights 10 %))
                           (rng/split (rng/key 10) 2000))]))
        schemes))

(every? (fn [cs]
          (every? true? (map (fn [c e] (<= (Math/floor e) c (Math/ceil e)))
                             cs expected-counts)))
        (repeats :systematic))
;=> true

;; book: unbiased
(defn unbiased?
  "Is each sample's mean count, over the repeats, within five standard
   errors of n times its probability?"
  [cs]
  (every? (fn [i]
            (let [{:keys [mean se]} (mean-se (mapv #(% i) cs))]
              (<= (abs (- mean (expected-counts i))) (* 5 se))))
          (range 4)))

(mapv #(unbiased? (repeats %)) schemes)
;=> [true true true true]

;; book: spread
(def spreads
  "Each method's standard deviation of the third sample's count."
  (mapv (fn [m] (sd-of (mapv #(% 2) (repeats m)))) schemes))

spreads
;=> [1.11 0.5 0.5 0.5] ; float64

(let [[multinomial & others] spreads]
  (every? #(< (* 2 %) multinomial) others))
;=> true

(every? (fn [m] (every? #(#{1 2} (% 2)) (repeats m))) (rest schemes))
;=> true

;; book: degeneracy
(def resampled
  (w/resample :multinomial (:log-normalized-weights is-result) 500
              (rng/key 11)))

(defn expected-distinct
  "The expected number of distinct indices among m multinomial draws, and
   a bound on its standard error: index i is drawn at least once with
   probability 1 - (1 - p_i)^m."
  [log-weights m]
  (let [hit (mapv #(- 1.0 (Math/pow (- 1.0 (Math/exp %)) m)) log-weights)]
    {:mean (reduce + hit)
     :se (Math/sqrt (reduce + (map #(* % (- 1.0 %)) hit)))}))

(def distinct-500
  (expected-distinct (:log-normalized-weights is-result) 500))

[(count (distinct resampled)) (:mean distinct-500)]
;=> [17 15.08] ; float64

(< (abs (- (count (distinct resampled)) (:mean distinct-500)))
   (* 5 (:se distinct-500)))
;=> true

;; book: sir
(def one-draw
  (imp/importance-resampling line-model [xs] (observations :y ys) 1000
                             {:key (rng/key 12) :batched? true}))

(sort (keys one-draw))
;=> (:log-ml-estimate :trace)

;; book: sir-draws
(defn sir-slopes
  "The slopes of m importance-resampling draws, each from n particles."
  [n m key]
  (mapv #(array/item ((:trace (imp/importance-resampling
                               line-model [xs] (observations :y ys) n
                               {:key % :batched? true}))
                      :slope))
        (rng/split key m)))

(def sir (into {} (map (fn [n] [n (sir-slopes n 200 (rng/key n))]))
               [10 100 1000]))

(mapv #(sd-of (sir %)) [10 100 1000])
;=> [2.28 0.652 0.241] ; float64

(let [[a b c] (mapv #(sd-of (sir %)) [10 100 1000])]
  (> a b c slope-sd))
;=> true

(let [{:keys [mean se]} (mean-se (sir 1000))]
  (< (abs (- mean slope-mean)) (* 5 se)))
;=> true

;; book: sir-figure
;; figure: figures/021-sir.svg
(let [exact-slope (d/normal slope-mean slope-sd)]
  (svg/histogram [{:values (sir 100) :label "100 samples"}
                  {:values (sir 1000) :label "1000 samples"
                   :density #(array/exp (dc/log-prob exact-slope %))}]
                 {:range [0 4] :bins 40 :x-label "slope"}))

;; book: start
(def start
  (:trace (gfi/generate line-model [xs] (observations :y ys) (rng/key 3))))

[(start :slope) (start :intercept)]
;=> [-2.474 -11.267] ; float64

;; book: kernel
(def slope-move (k/mh (sel/select :slope)))

(def moved (slope-move start (rng/key 4)))

(sort (keys moved))
;=> (:accepted? :log-alpha :trace)

;; book: random-walk
(def walk
  (gen [tr addr step]
    (trace addr (d/normal (tr addr) step))))

(defn random-walk
  "The MH kernel that proposes the choice at addr from a normal around its
   current value, of standard deviation step."
  [addr step]
  (k/mh walk [addr step]))

(sort (keys ((random-walk :slope 0.3) start (rng/key 4))))
;=> (:accepted? :log-alpha :trace)

;; book: conditional
(defn slope-given
  "The exact conditional of the slope given the intercept b and the points:
   a normal whose precision is the prior's 1/100 plus the sum of x^2."
  [b]
  (let [precision (+ 0.01 (reduce + (map * xs xs)))]
    (d/normal (/ (reduce + (map (fn [x y] (* x (- y b))) xs ys)) precision)
              (Math/sqrt (/ 1.0 precision)))))

(def conditional (slope-given (array/item (start :intercept))))

(:params conditional)
;=> {:mu 4.0817 :std 0.07001} ; float64

;; book: conditional-chains
(defn slope-draws
  "The slope's draws from a chain of the kernel from start (20100 steps,
   the first 100 dropped, then every tenth kept), with the acceptance rate."
  [kernel key]
  (k/run kernel start
         {:samples 2000 :burn 100 :thin 10 :key key :keep #(% :slope)}))

(def by-prior (slope-draws slope-move (rng/key 5)))

(def by-walk (slope-draws (random-walk :slope 0.3) (rng/key 5)))

(mapv :acceptance-rate [by-prior by-walk])
;=> [0.0100 0.273] ; float64

;; book: conditional-check
(defn near?
  "Is the mean of the draws within five Monte Carlo standard errors of mu?"
  [draws mu]
  (let [{:keys [mean mcse-mean]} (diag/summary draws)]
    (< (abs (- mean mu)) (* 5 mcse-mean))))

(mapv (fn [{:keys [draws]}]
        [(near? draws (:mu (:params conditional)))
         (near? draws slope-mean)])
      [by-prior by-walk])
;=> [[true false] [true false]]

(mapv #(diag/ess-bulk (:draws %)) [by-prior by-walk])
;=> [106.6 1838.6] ; float64

;; book: toy
(defn recorder
  "A kernel on vectors that appends label, its reversal the label primed:
   a stand-in for a move, to show what the combinators do."
  [label]
  (let [primed (keyword (str (name label) "'"))]
    (k/kernel (fn [v _] {:trace (conj v label)})
              (fn [v _] {:trace (conj v primed)}))))

(def a (recorder :a))

(def b (recorder :b))

(def c (recorder :c))

(defn applied
  "What the kernel makes from the empty vector."
  [kernel]
  (:trace (kernel [] (rng/key 0))))

(mapv applied [(k/chain a b c) (k/repeat 3 a) (k/cycle 5 [a b c])])
;=> [[:a :b :c] [:a :a :a] [:a :b :c :a :b]]

;; book: reversal
(mapv applied [(k/reversal (k/chain a b c)) (k/reversal (k/cycle 5 [a b c]))])
;=> [[:c' :b' :a'] [:b' :a' :c' :b' :a']]

(k/reversible? (k/chain a (fn [v _] {:trace v})))
;=> false

;; book: mix
(def mixed (k/mix [a b] [0.7 0.3]))

(def picks
  (frequencies (map #(first (:trace (mixed [] %)))
                    (rng/split (rng/key 6) 2000))))

picks
;=> {:a 1435 :b 565} ; float64

(< (abs (- (/ (picks :a) 2000.0) 0.7)) (* 5 (Math/sqrt (/ (* 0.7 0.3) 2000))))
;=> true

;; book: block
(def block (k/chain (random-walk :slope 0.1) (random-walk :intercept 0.5)))

(:accepted (block start (rng/key 14)))
;=> [true false] ; float64

;; book: hmc
(def hmc-move (k/hmc (sel/select :slope :intercept) {:L 10 :eps 0.1}))

(sort (keys (hmc-move start (rng/key 15))))
;=> (:accepted? :diverged? :log-alpha :trace)

(def mixed-run
  (k/run (k/chain block hmc-move) start
         {:samples 500 :burn 100 :key (rng/key 16)}))

(near? (diag/draws (:traces mixed-run) :slope) slope-mean)
;=> true

;; book: checked
;; error: The kernel changed the observation at [:y 0]
#_(def careless (k/checked (k/mh (sel/select [:y 0])) (observations :y ys)))
#_(careless start (rng/key 8))

;; book: run
(def run-opts {:samples 1000 :burn 1000 :thin 5 :key (rng/key 9)})

(def line-run (k/run block start run-opts))

(sort (keys line-run))
;=> (:acceptance-rate :divergences :traces)

[(count (:traces line-run)) (:acceptance-rate line-run)]
;=> [1000 0.601] ; float64

(= (diag/draws (:traces line-run) :slope)
   (diag/draws (:traces (k/run block start run-opts)) :slope))
;=> true

;; book: run-steps
(defn count-up
  "A kernel on numbers: one MH-like move that adds one, always accepted."
  [n _]
  {:trace (inc n) :accepted? true})

(defn stay
  "A kernel on numbers: one MH-like move that is always rejected."
  [n _]
  {:trace n :accepted? false})

(k/run (k/chain count-up stay stay) 0
       {:samples 4 :burn 2 :thin 3 :key (rng/key 0)})
;=> {:traces [5 8 11 14] :acceptance-rate 0.3333 :divergences 0}

;; book: run-check
(defn agrees?
  "Do the draws' mean and standard deviation agree with mu and sd, each
   within five of its Monte Carlo standard errors?"
  [draws mu sd]
  (let [s (diag/summary draws)]
    [(< (abs (- (:mean s) mu)) (* 5 (:mcse-mean s)))
     (< (abs (- (:sd s) sd)) (* 5 (:mcse-sd s)))]))

(let [[_ intercept-mean] (array/->clj (:mean exact))
      [_ intercept-sd] (sds exact)]
  [(agrees? (diag/draws (:traces line-run) :slope) slope-mean slope-sd)
   (agrees? (diag/draws (:traces line-run) :intercept)
            intercept-mean intercept-sd)])
;=> [[true true] [true true]]

;; book: burn-in
(def early
  (diag/draws (:traces (k/run block start {:samples 300 :key (rng/key 9)}))
              :slope))

(def hundreds (mapv #(:mean (mean-se (subvec early % (+ % 100)))) [0 100 200]))

hundreds
;=> [-0.92 1.63 1.96] ; float64

(< (first hundreds) (- slope-mean (* 10 slope-sd)))
;=> true

;; book: burn-in-figure
;; figure: figures/021-burn-in.svg
(svg/series early {:y-label "slope"})

;; book: chains
(defn run-draws
  "The draws at addrs of a chain of the kernel from start, run with opts:
   a map from address to draws, kept as the chain runs, each read as
   diag/draws reads it (a boolean as 0 or 1)."
  [kernel start opts addrs]
  (let [draw (fn [a] #(first (diag/draws [%] a)))
        readers (into {} (map (juxt identity draw)) addrs)]
    (:draws (k/run kernel start (assoc opts :keep readers)))))

(def starts
  (mapv #(:trace (gfi/generate line-model [xs] (observations :y ys) %))
        (rng/split (rng/key 20) 4)))

(mapv #(array/item (% :slope)) starts)
;=> [6.542 19.059 -0.824 -1.294] ; float64

(def four
  (mapv #(run-draws block %1 {:samples 1000 :burn 1000 :thin 20 :key %2}
                    [:slope :intercept])
        starts (rng/split (rng/key 21) 4)))

;; book: r-hat
(def slope-chains (mapv :slope four))

(def intercept-chains (mapv :intercept four))

(mapv (juxt diag/r-hat diag/ess-bulk) [slope-chains intercept-chains])
;=> [[1.0022 1538.3] [1.0014 1563.7]] ; float64

(every? #(< (diag/r-hat %) 1.01) [slope-chains intercept-chains])
;=> true

;; book: summary
(diag/summary slope-chains)

(agrees? slope-chains slope-mean slope-sd)
;=> [true true]

;; book: tail
(let [q05 (- slope-mean (* 1.6449 slope-sd))
      pooled (apply concat slope-chains)
      below (/ (count (filter #(< % q05) pooled)) (double (count pooled)))
      se (Math/sqrt (/ (* 0.05 0.95) (diag/ess-tail slope-chains)))]
  (< (abs (- below 0.05)) (* 5 se)))
;=> true

;; book: stuck
(def small-steps
  (k/chain (random-walk :slope 0.01) (random-walk :intercept 0.05)))

(def stuck
  (mapv #(:slope (run-draws small-steps %1 {:samples 1000 :key %2} [:slope]))
        starts (rng/split (rng/key 22) 4)))

[(diag/r-hat stuck) (diag/ess-bulk stuck)]
;=> [3.67 4.4] ; float64

(def stuck-rate
  (:acceptance-rate
   (k/run small-steps (first starts) {:samples 1000 :key (rng/key 23)})))

stuck-rate
;=> 0.796 ; float64

[(> (diag/r-hat stuck) 1.1)
 (< (diag/ess-bulk stuck) 20)
 (> stuck-rate (:acceptance-rate line-run))]
;=> [true true true]

;; book: stuck-figure
;; figure: figures/021-stuck.svg
(defn path
  "A chain's draws as a function of the iteration, for svg/plot's curves."
  [draws]
  (fn [x] (nth draws (min (dec (count draws)) (long x)))))

(svg/plot (mapv (fn [draws color] {:curves [(path draws)] :color color})
                stuck ["steelblue" "darkorange" "seagreen" "firebrick"])
          {:x-range [0 999] :x-label "iteration" :y-label "slope"})

;; book: groups
(def groups
  (gen [n]
    (let [p (trace :p (d/beta 2 2))]
      (dotimes [i n]
        (let [z (trace [:z i] (d/bernoulli p))]
          (trace [:y i] (d/normal (array/where z 2.0 -2.0) 1))))
      p)))

(def readings [-2.3 1.8 2.6 -0.4 0.3 2.2 -2.9 1.5])

(def n-readings (count readings))

(batched/batched? (:particles (imp/importance-sampling
                               groups [n-readings] (observations :y readings)
                               100 {:key (rng/key 36) :batched? true})))
;=> true

;; book: groups-exact
(def assignments
  "Every assignment of the readings to the groups, as choice maps of the
   [:z i]: Gen.jl's choice_vol_grid, 2^8 points."
  (mapv first (apply enum/choice-vol-grid
                     (for [i (range n-readings)] [[:z i] [false true]]))))

(defn in-group-one
  "How many readings the assignment z puts in the group around 2."
  [z]
  (count (filter true? (cm/values z))))

(defn log-joint
  "log p(z, readings) with the bias p summed out, by Bayes' rule at p = 1/2:
   given h readings in the group around 2, p's posterior is
   beta(2 + h, 2 + n - h)."
  [z]
  (let [h (in-group-one z)
        choices (cm/merge (cm/choicemap :p 0.5) z
                          (observations :y readings))]
    (- (:weight (gfi/assess groups [n-readings] choices))
       (dc/log-prob (d/beta (+ 2 h) (+ 2 (- n-readings h))) 0.5))))

(def posterior-z (w/normalize (mapv log-joint assignments)))

(defn exact-mean
  "The exact posterior mean of (f z) over the assignments z."
  [f]
  (reduce + (map (fn [z lw] (* (Math/exp lw) (f z)))
                 assignments (:log-normalized-weights posterior-z))))

(defn group-one?
  "1 when the assignment z puts reading i in the group around 2, else 0."
  [z i]
  (if (array/truthy? (cm/get-choice z [:z i])) 1.0 0.0))

(def exact-z (mapv (fn [i] (exact-mean #(group-one? % i))) (range n-readings)))

(def exact-p (exact-mean #(/ (+ 2.0 (in-group-one %)) (+ 4.0 n-readings))))

exact-z
;=> [0.0002 0.9994 1.0 0.2485 0.8126 0.9999 0.0 0.9979]

exact-p
;=> 0.5882

;; book: gibbs
(def gibbs-groups
  (enum/with-exact-proposal groups (mapv #(vector :z %) (range n-readings))))

(def group-start
  (:trace (gfi/generate gibbs-groups [n-readings] (observations :y readings)
                        (rng/key 30))))

(def gibbs-4 (k/mh (sel/select [:z 4])))

(select-keys (gibbs-4 group-start (rng/key 31)) [:accepted? :log-alpha])
;=> {:accepted? true :log-alpha 0.0}

;; book: gibbs-conditional
(defn z-given
  "The exact conditional probability that reading i is in the group around
   2, given the bias p: p N(y; 2, 1) against (1 - p) N(y; -2, 1)."
  [i p]
  (let [y (readings i)
        one (* p (array/exp (dc/log-prob (d/normal 2.0 1) y)))
        other (* (- 1.0 p) (array/exp (dc/log-prob (d/normal -2.0 1) y)))]
    (array/item (/ one (+ one other)))))

(def start-p (array/item (group-start :p)))

[start-p (z-given 4 start-p)]
;=> [0.4117 0.699] ; float64

(let [n 1000
      ones (count (filter #((:trace (gibbs-4 group-start %)) [:z 4])
                          (rng/split (rng/key 32) n)))
      q (z-given 4 start-p)]
  (< (abs (- (/ ones (double n)) q)) (* 5 (Math/sqrt (/ (* q (- 1 q)) n)))))
;=> true

;; book: sweep
(def gibbs-sweep
  (k/kern [n]
    (doseq [i (range n)]
      (k/mh (sel/select [:z i])))
    (k/mh (sel/select :p))))

(def sweep (gibbs-sweep n-readings))

(count (:accepted (sweep group-start (rng/key 33))))
;=> 9

(k/reversible? sweep)
;=> true

;; book: sweep-check
(def group-run
  (run-draws sweep group-start {:samples 1000 :burn 100 :key (rng/key 34)}
             [:p [:z 3] [:z 4]]))

[(near? (:p group-run) exact-p)
 (near? (group-run [:z 3]) (exact-z 3))
 (near? (group-run [:z 4]) (exact-z 4))]
;=> [true true true]

;; book: gibbs-only
(def z-moves
  (apply k/chain (for [i (range n-readings)] (k/mh (sel/select [:z i])))))

(:acceptance-rate (k/run z-moves group-start {:samples 50 :key (rng/key 35)}))
;=> 1.0

;; book: solution-1
(def log-p (:log-normalized-weights is-result))

(def distinct-4000 (expected-distinct log-p 4000))

(def multinomial-4000
  (count (distinct (w/resample :multinomial log-p 4000 (rng/key 40)))))

[multinomial-4000 (:mean distinct-4000)]
;=> [20 21.31] ; float64

(< (abs (- multinomial-4000 (:mean distinct-4000))) (* 5 (:se distinct-4000)))
;=> true

;; book: solution-1-systematic
(def systematic-4000
  (count (distinct (w/resample :systematic log-p 4000 (rng/key 41)))))

[systematic-4000
 (reduce + (map #(min 1.0 (* 4000 (Math/exp %))) log-p))
 (count (filter #(>= (* 4000 (Math/exp %)) 1.0) log-p))]
;=> [23 22.81 21] ; float64

;; book: solution-2
(def steps [0.02 0.3 2.0])

(def by-step (mapv #(slope-draws (random-walk :slope %) (rng/key 42)) steps))

(mapv :acceptance-rate by-step)
;=> [0.897 0.277 0.0426] ; float64

(mapv #(diag/ess-bulk (:draws %)) by-step)
;=> [107.3 1824.9 581.5] ; float64

(let [[small middle large] (mapv #(diag/ess-bulk (:draws %)) by-step)]
  [(every? #(near? (:draws %) (:mu (:params conditional))) by-step)
   (> middle (* 2 small))
   (> middle (* 2 large))])
;=> [true true true]

;; book: solution-3
(def z-run
  (run-draws z-moves group-start {:samples 1000 :key (rng/key 43)} [[:z 4]]))

[(near? (z-run [:z 4]) (z-given 4 start-p))
 (near? (z-run [:z 4]) (exact-z 4))]
;=> [true false]

;; book: solution-4
(def slope-only
  (mapv #(:slope (run-draws (random-walk :slope 0.3) start
                            {:samples 1000 :burn 100 :thin 5 :key %}
                            [:slope]))
        (rng/split (rng/key 44) 4)))

[(diag/r-hat slope-only) (diag/ess-bulk slope-only)]
;=> [1.0006 2969.8] ; float64

[(< (diag/r-hat slope-only) 1.01)
 (near? slope-only slope-mean)
 (near? slope-only (:mu (:params conditional)))]
;=> [true false true]
