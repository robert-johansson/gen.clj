;; The code of book/012-modeling.md. Each region, from a `;; book:` line to
;; the next, is included there by name (gen.book: `clojure -M:test -m
;; gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.modeling
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as is]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: helpers
(defn addresses
  "The addresses of trace t's choices."
  [t]
  (cm/addresses (:choices t)))

(defn average
  "The mean of the numbers vs."
  [vs]
  (/ (reduce + vs) (count vs)))

(defn standard-error
  "The standard error of the mean of the numbers vs."
  [vs]
  (let [m (average vs)
        n (count vs)]
    (Math/sqrt (/ (reduce + (map #(* (- % m) (- % m)) vs)) (* n (dec n))))))

(defn fraction
  "The fraction of xs for which pred holds."
  [pred xs]
  (/ (count (filter pred xs)) (double (count xs))))

(def colors ["steelblue" "darkorange" "seagreen" "firebrick"])

(defn trace-layers
  "Figure layers for traces of a model of points at xs: each trace's
   function (its return value) and its points, in a color of its own."
  [traces]
  (mapcat (fn [t color]
            (let [[xs] (:args t)]
              [{:curves [(:retval t)] :color color}
               {:points (map-indexed (fn [i x] [x (t [:y i])]) xs)
                :color color :r 2.5}]))
          traces (cycle colors)))

(defn prediction-layers
  "Figure layers: the predictions at new-xs as faint dots of a color, and
   the observed points (xs, ys) in red."
  [xs ys new-xs predictions color]
  [{:points (for [p predictions [x y] (map vector new-xs p)] [x y])
    :color color :r 1.5 :opacity 0.3}
   {:points (map vector xs ys) :color "firebrick"}])

(defn data-layer
  "A figure layer of the points (xs, ys), small and gray."
  [xs ys]
  {:points (map vector xs ys) :color "gray" :r 2 :opacity 0.6})

;; book: shifted
(def shifted
  (gen []
    (let [x (trace :initial-x (d/normal 0 1))]
      (if (array/< x 0)
        (+ x (trace :addition-to-x (d/normal 2 1)))
        x))))

(for [x [1.0 -1.0]]
  (addresses (:trace (gfi/generate shifted [] (cm/choicemap :initial-x x)
                                   (rng/key 1)))))
;=> ([[:initial-x]] [[:addition-to-x] [:initial-x]])

;; book: not-a-choice
;; error: trace at [:not-a-random-choice] needs a distribution
#_(gfi/simulate (gen [x] (trace :not-a-random-choice (array/sin x)))
              [1.0] (rng/key 3))

;; book: line-model
(def line-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 1))
          intercept (trace :intercept (d/normal 0 2))
          y (fn [x] (+ (* slope x) intercept))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (y x) 0.1)))
      y)))

(def xs [-5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0])

(fn? (line-model xs))
;=> true

;; book: simulate
(def tr (gfi/simulate line-model [xs] (rng/key 4)))

(:args tr)
;=> [[-5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0]]

(tr :slope)
;=> -0.6047 ; float64

(= (tr :slope) (cm/get-choice (:choices tr) :slope))
;=> true

[(= (:retval tr) (tr [])) (array/allclose? ((tr []) 0.0) (tr :intercept))]
;=> [true true]

;; book: line-figure
;; figure: figures/012-line-traces.svg
(svg/plot (trace-layers (mapv #(gfi/simulate line-model [xs] %)
                              (rng/split (rng/key 5) 4)))
          {:x-range [-5.5 5.5] :y-range [-5 5]})

;; book: sine-model
(defn wave
  "The sine wave of a period, an amplitude and a phase, as a function of x."
  [period amplitude phase]
  (fn [x] (* amplitude (array/sin (+ (/ (* 2 Math/PI x) period) phase)))))

(def sine-model
  (gen [xs]
    (let [period (trace :period (d/gamma 1 1))
          amplitude (trace :amplitude (d/gamma 1 1))
          phase (trace :phase (d/uniform 0 (* 2 Math/PI)))
          y (wave period amplitude phase)]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (y x) 0.1)))
      y)))

;; book: sine-figure
;; figure: figures/012-sine-traces.svg
(def sine-traces
  (mapv #(gfi/simulate sine-model [xs] %) (rng/split (rng/key 6) 3)))

(mapv #(% :period) sine-traces)
;=> [1.827 0.0031 2.211] ; float64

(svg/plot (trace-layers sine-traces) {:x-range [-5.5 5.5] :y-range [-5 5]})

;; book: data
(def ys [6.75003 6.1568 4.26414 1.84894 3.09686 1.94026 1.36411 -0.83959
         -0.976 -1.93363 -2.91303])

(defn observations
  "The choice map fixing each [:y i] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

;; book: do-inference
(defn do-inference
  "A trace of model on xs, approximately from the posterior given ys: one of
   n importance samples, drawn in proportion to its weight. opts may add
   {:batched? true}."
  [model xs ys n key & [opts]]
  (:trace (is/importance-resampling model [xs] (observations ys) n
                                    (merge {:key key} opts))))

(defn answers
  "k traces of model, each from its own inference on (xs, ys) with n
   samples. opts as do-inference's."
  [model xs ys n k key & [opts]]
  (mapv #(do-inference model xs ys n % opts) (rng/split key k)))

(def line-answers
  (answers line-model xs ys 100 10 (rng/key 7) {:batched? true}))

;; book: posterior-figure
;; figure: figures/012-line-posterior.svg
(svg/plot [{:curves (map :retval line-answers)}
           {:points (map vector xs ys)}])

;; book: exact
(defn line-posterior
  "The exact posterior of line-model's slope and intercept given ys at xs,
   for xs that sum to zero: two independent normals."
  [xs ys]
  (let [x (array/array xs)
        y (array/array ys)
        noise-precision (/ 1 (* 0.1 0.1))
        precision [(+ 1 (* noise-precision (array/sum (* x x))))
                   (+ (/ 1 4) (* noise-precision (count xs)))]
        weighted [(* noise-precision (array/sum (* x y)))
                  (* noise-precision (array/sum y))]]
    {:mean (mapv #(array/item (/ %1 %2)) weighted precision)
     :sd (mapv #(array/item (/ 1 (array/sqrt %))) precision)}))

(def exact (line-posterior xs ys))

[(:mean exact) (:sd exact)]
;=> [[-0.9409 1.7050] [0.00953 0.0301]]

;; book: spread
(defn squared-distance
  "The squared distance of trace t's line from the exact posterior mean, in
   posterior standard deviations: 2 on average for draws from the
   posterior."
  [t]
  (array/item
   (reduce + (map (fn [addr mean sd] (array/square (/ (- (t addr) mean) sd)))
                  [:slope :intercept] (:mean exact) (:sd exact)))))

(def spread (average (map squared-distance line-answers)))

spread
;=> 268.9 ; float64

(> spread (+ 2 (* 5 (/ 2 (Math/sqrt 10)))))
;=> true

;; book: predict
(defn predict-new-data
  "The ys model predicts at new-xs, with the choices at param-addrs fixed
   to their values in trace tr."
  [model tr new-xs param-addrs key]
  (let [constraints (apply cm/choicemap
                           (mapcat (fn [a] [a (tr a)]) param-addrs))
        new-tr (:trace (gfi/generate model [new-xs] constraints key))]
    (mapv #(new-tr [:y %]) (range (count new-xs)))))

;; book: infer-and-predict
(defn infer-and-predict
  "num-traces predictions of ys at new-xs, each from its own inference on
   (xs, ys) with n samples."
  [model xs ys new-xs param-addrs num-traces n key]
  (mapv (fn [k]
          (let [[k-infer k-predict] (rng/split k)
                tr (do-inference model xs ys n k-infer {:batched? true})]
            (predict-new-data model tr new-xs param-addrs k-predict)))
        (rng/split key num-traces)))

(def new-xs (mapv #(+ -5 (* 15 (/ % 99))) (range 100)))

;; book: noisy
(def ys-noisy [5.092 4.781 2.46815 1.23047 0.903318 1.11819 2.10808 1.09198
               0.0203789 -2.05068 2.66031])

(defn roughness
  "The residual standard deviation of the points (xs, ys) around their
   least-squares line."
  [xs ys]
  (let [x (array/array xs)
        y (array/array (mapv array/item ys))
        dx (- x (array/mean x))
        dy (- y (array/mean y))
        slope (/ (array/sum (* dx dy)) (array/sum (* dx dx)))
        residual (- dy (* slope dx))]
    (array/item (array/sqrt (/ (array/sum (* residual residual))
                               (- (count xs) 2))))))

[(roughness xs ys) (roughness xs ys-noisy)]
;=> [0.729 1.561]

;; book: fixed-noise
(def fixed-predictions
  (infer-and-predict line-model xs ys-noisy new-xs [:slope :intercept] 20 1000
                     (rng/key 11)))

(def fixed-roughness (mapv #(roughness new-xs %) fixed-predictions))

(< (apply max fixed-roughness) (/ (roughness xs ys-noisy) 5))
;=> true

;; book: line-model-fancy
(def line-model-fancy
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 1))
          intercept (trace :intercept (d/normal 0 2))
          noise (trace :noise (d/gamma 1 1))
          y (fn [x] (+ (* slope x) intercept))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (y x) noise)))
      y)))

(def fancy-predictions
  (infer-and-predict line-model-fancy xs ys-noisy new-xs
                     [:slope :intercept :noise] 20 10000 (rng/key 12)))

;; book: fancy-roughness
(def fancy-roughness (mapv #(roughness new-xs %) fancy-predictions))

[(average fixed-roughness) (average fancy-roughness)]
;=> [0.0969 1.726] ; float64

(let [data (roughness xs ys-noisy)
      m (average fancy-roughness)
      band (* 5 (standard-error fancy-roughness))]
  (< (/ data 2) (- m band) (+ m band) (* 2 data)))
;=> true

;; book: noisy-figure
;; figure: figures/012-predictions-noisy.svg
(svg/plot (concat (prediction-layers xs ys-noisy new-xs fancy-predictions
                                     "steelblue")
                  (prediction-layers xs ys-noisy new-xs fixed-predictions
                                     "black")))

;; book: foo-bar
(def foo (gen [] (trace :y (d/normal 0 1))))

(def bar
  (gen []
    (trace :x (d/bernoulli 0.5))
    (splice foo)))

(def bar-using-namespace
  (gen []
    (trace :x (d/bernoulli 0.5))
    (splice :z foo)))

[(addresses (gfi/simulate bar [] (rng/key 13)))
 (addresses (gfi/simulate bar-using-namespace [] (rng/key 13)))]
;=> [[[:x] [:y]] [[:x] [:z :y]]]

((gfi/simulate bar-using-namespace [] (rng/key 14)) [:z :y])
;=> 1.0212 ; float64

(def plain-call (gen [] (foo)))

(addresses (gfi/simulate plain-call [] (rng/key 16)))
;=> []

(apply = (repeatedly 2 #(:retval (gfi/simulate plain-call [] (rng/key 16)))))
;=> true

;; book: clash
;; error: Address [:y] is used twice
#_(gfi/simulate (gen [] (splice foo) (splice foo)) [] (rng/key 15))

;; book: combined-model
(def sine-model-fancy
  (gen [xs]
    (let [period (trace :period (d/gamma 5 1))
          amplitude (trace :amplitude (d/gamma 1 1))
          phase (trace :phase (d/uniform 0 (* 2 Math/PI)))
          noise (trace :noise (d/gamma 1 1))
          y (wave period amplitude phase)]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (y x) noise)))
      y)))

(def combined-model
  (gen [xs]
    (if (trace :is-line (d/bernoulli 0.5))
      (splice line-model-fancy xs)
      (splice sine-model-fancy xs))))

;; book: combined-figure
;; figure: figures/012-combined.svg
(def ys-sine [2.89 2.22 -0.612 -0.522 -2.65 -0.133 2.70 2.77 0.425 -2.11
              -2.76])

(svg/plot [{:curves (map :retval (answers combined-model xs ys-sine 2000 6
                                          (rng/key 19)))}
           {:points (map vector xs ys-sine)}]
          {:y-range [-4 4]})

;; book: probability
(defn probability
  "Importance sampling's estimate of the posterior probability that pred
   holds of model's trace given ys: the share of n samples' weight on the
   traces where it does."
  [pred model xs ys n key]
  (:estimate (is/estimate #(if (pred %) 1.0 0.0)
                          model [xs] (observations ys) n {:key key})))

(defn line?
  "Did trace t of combined-model choose the line?"
  [t]
  (array/item (t :is-line)))

(def line-probabilities
  [(probability line? combined-model xs ys 2000 (rng/key 17))
   (probability line? combined-model xs ys-sine 2000 (rng/key 18))])

line-probabilities
;=> [0.999999 0.00016] ; float64

[(> (first line-probabilities) 0.9) (< (second line-probabilities) 0.1)]
;=> [true true]

;; book: segments
(def generate-segments
  (gen [l u]
    (if (trace :isleaf (d/bernoulli 0.7))
      {:interval [l u] :value (trace :value (d/normal 0 1))}
      (let [frac (trace :frac (d/beta 2 2))
            mid (+ l (* (- u l) frac))]
        {:interval [l u]
         :left (splice :left generate-segments l mid)
         :right (splice :right generate-segments mid u)}))))

;; book: value-at
(defn value-at
  "The value of the segment of the tree node that holds x."
  [node x]
  (if-let [left (:left node)]
    (array/where (array/<= x (second (:interval left)))
                 (value-at left x)
                 (value-at (:right node) x))
    (:value node)))

;; book: segment-counts
(defn segments
  "The leaves of a tree from generate-segments, left to right."
  [node]
  (if (:left node)
    (concat (segments (:left node)) (segments (:right node)))
    [node]))

(def segment-counts
  (mapv #(count (segments (:retval (gfi/simulate generate-segments
                                                 [0.0 1.0] %))))
        (rng/split (rng/key 21) 2000)))

[(< (abs (- (fraction #{1} segment-counts) 0.7))
    (* 5 (Math/sqrt (/ (* 0.7 0.3) 2000))))
 (< (abs (- (average segment-counts) 1.75))
    (* 5 (standard-error segment-counts)))]
;=> [true true]

;; book: segment-layers
(defn segment-layers
  "Figure layers: the piecewise-constant function of each tree, in a color
   of its own."
  [trees]
  (map (fn [node color]
         {:curves [#(value-at node %)] :color color :width 3 :opacity 0.9})
       trees (cycle colors)))

;; book: segments-figure
;; figure: figures/012-segments-prior.svg
(svg/plot (segment-layers (mapv #(:retval (gfi/simulate generate-segments
                                                        [0.0 1.0] %))
                                (rng/split (rng/key 20) 4)))
          {:x-range [0 1] :y-range [-3 3]})

;; book: changepoint-model
(def changepoint-model
  (gen [xs]
    (let [node (splice :tree generate-segments (apply min xs) (apply max xs))
          noise (trace :noise (d/gamma 0.5 0.5))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (value-at node x) noise)))
      node)))

;; book: changepoint-data
(def xs-dense (mapv #(+ -5 (* 10 (/ % 49))) (range 50)))

(defn noisy
  "f at each of xs, plus normal noise with standard deviation 0.1."
  [f xs key]
  (mapv #(+ (f %1) (* 0.1 %2)) xs (array/->clj (rng/normal key [(count xs)]))))

(def ys-simple (noisy (constantly 1.0) xs-dense (rng/key 22)))

(def ys-complex
  (noisy #(if (even? (long (Math/floor (abs (/ % 3))))) 2.0 0.0)
         xs-dense (rng/key 23)))

;; book: simple-inference
(defn one-segment?
  "Did trace t of changepoint-model choose one segment?"
  [t]
  (= 1 (count (segments (:retval t)))))

(> (probability one-segment? changepoint-model xs-dense ys-simple 4000
                (rng/key 24))
   0.9)
;=> true

;; book: complex-inference
(def complex-answer
  (do-inference changepoint-model xs-dense ys-complex 20000 (rng/key 26)))

(def by-hand
  (cm/choicemap [:tree :isleaf] false [:tree :frac] 0.2
                [:tree :left :isleaf] true [:tree :left :value] 0.0
                [:tree :right :isleaf] false [:tree :right :frac] 0.75
                [:tree :right :left :isleaf] true
                [:tree :right :left :value] 2.0
                [:tree :right :right :isleaf] true
                [:tree :right :right :value] 0.0
                :noise 0.1))

(def by-hand-trace
  (:trace (gfi/generate changepoint-model [xs-dense]
                        (cm/merge by-hand (observations ys-complex))
                        (rng/key 27))))

(map :interval (segments (:retval by-hand-trace)))
;=> ([-5.0 -3.0] [-3.0 3.0] [3.0 5.0])

;; book: likelihoods
(defn log-likelihood
  "The log density of trace t's observations: how well t explains them."
  [t]
  (array/item (gfi/project t (sel/select :y))))

[(log-likelihood by-hand-trace) (log-likelihood complex-answer)]
;=> [36.3 -57.8] ; float64

(> (log-likelihood by-hand-trace) (+ (log-likelihood complex-answer) 50))
;=> true

;; book: complex-figure
;; figure: figures/012-changepoint-complex.svg
(svg/plot (cons (data-layer xs-dense ys-complex)
                (segment-layers (map :retval [complex-answer by-hand-trace])))
          {:y-range [-1 3]})

;; book: solution-1
;; error: Address [:x] is used twice
#_(gfi/simulate (gen [] (trace :x (d/normal 0 1)) (trace :x (d/normal 0 1)))
              [] (rng/key 28))

;; book: solution-2
(def errors-by-n
  (into (sorted-map)
        (for [n [1 10 100 1000 10000]]
          [n (mapv squared-distance
                   (answers line-model xs ys n 20
                            (rng/fold-in (rng/key 29) n) {:batched? true}))])))

(update-vals errors-by-n #(Math/round (double (average %))))
;=> {1 47220, 10 3214, 100 318, 1000 40, 10000 4} ; float64

(let [at-1000 (errors-by-n 1000)
      at-10000 (errors-by-n 10000)]
  [(> (average at-1000) (+ 2 (* 5 (/ 2 (Math/sqrt 20)))))
   (< (- (average at-10000) (* 5 (standard-error at-10000))) 2)])
;=> [true true]

;; book: solution-3-aliases
(let [at-xs (fn [f] (array/array (mapv f xs)))
      wave-6 (at-xs (wave 6.0 2.0 0.3))]
  [(array/allclose? wave-6 (at-xs (wave (/ 6.0 7) 2.0 0.3)))
   (array/allclose? wave-6 (at-xs (wave (/ 6.0 5) 2.0 (- Math/PI 0.3))))])
;=> [true true]

(mapv #(dc/log-prob (d/gamma 1 1) %) [6.0 (/ 6.0 7) (/ 6.0 5)])
;=> [-6.0 -0.857 -1.2]

;; book: solution-3
(def sine-model-5
  (gen [xs]
    (let [y (wave (trace :period (d/gamma 5 1))
                  (trace :amplitude (d/gamma 1 1))
                  (trace :phase (d/uniform 0 (* 2 Math/PI))))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (y x) 0.1)))
      y)))

(defn long-period-fraction
  "The fraction of 40 answers of model on ys-sine whose period is over 3."
  [model key]
  (fraction #(> (array/item (% :period)) 3)
            (answers model xs ys-sine 1000 40 key {:batched? true})))

(def fractions
  [(long-period-fraction sine-model (rng/key 30))
   (long-period-fraction sine-model-5 (rng/key 31))])

fractions
;=> [0.0 1.0] ; float64

(> (- (second fractions) (first fractions))
   (* 5 (Math/sqrt (/ (* 2 0.25) 40))))
;=> true

;; book: solution-4
(def xs-few (subvec xs 0 5))

(def ys-few (subvec ys-sine 0 5))

(def p-sine
  (fraction (complement line?)
            (answers combined-model xs-few ys-few 500 100 (rng/key 32))))

(defn log-ml
  "Importance sampling's estimate of log p(ys) under model at xs."
  [model xs ys n key]
  (:log-ml-estimate (is/importance-sampling model [xs] (observations ys) n
                                            {:key key :batched? true})))

(def p-sine-ml
  (let [line (log-ml line-model-fancy xs-few ys-few 100000 (rng/key 33))
        sine (log-ml sine-model-fancy xs-few ys-few 100000 (rng/key 34))]
    (/ 1 (+ 1 (Math/exp (- line sine))))))

[p-sine p-sine-ml]
;=> [0.4 0.3735] ; float64

(< (abs (- p-sine p-sine-ml)) (* 5 (Math/sqrt (/ 0.25 100))))
;=> true

;; book: solution-5
(def generate-segments-flat
  (gen [l u i]
    (if (trace [:isleaf i] (d/bernoulli 0.7))
      {:interval [l u] :value (trace [:value i] (d/normal 0 1))}
      (let [frac (trace [:frac i] (d/beta 2 2))
            mid (+ l (* (- u l) frac))]
        {:interval [l u]
         :left (splice generate-segments-flat l mid (+ (* 2 i) 1))
         :right (splice generate-segments-flat mid u (+ (* 2 i) 2))}))))

(take 5 (addresses (gfi/simulate generate-segments-flat [0.0 1.0 0]
                                  (rng/key 38))))
;=> [[:frac 0] [:frac 2] [:isleaf 0] [:isleaf 1] [:isleaf 2]] ; float64

;; book: solution-5-check
(defn flat-address
  "The address in generate-segments-flat of the choice at path in
   generate-segments: node i's children are 2i + 1 and 2i + 2."
  [path]
  [(peek path)
   (reduce (fn [i step] (if (= step :left) (+ (* 2 i) 1) (+ (* 2 i) 2)))
           0 (pop path))])

(defn same-score?
  "Does generate-segments-flat give trace t's choices, at their flat
   addresses, the score t has?"
  [t]
  (let [flat (apply cm/choicemap
                    (mapcat (fn [[path v]] [(flat-address path) v])
                            (cm/leaves (:choices t))))]
    (array/allclose? (:score t)
                     (:weight (gfi/assess generate-segments-flat [0.0 1.0 0]
                                          flat (rng/key 36))))))

(every? same-score? (mapv #(gfi/simulate generate-segments [0.0 1.0] %)
                          (rng/split (rng/key 37) 50)))
;=> true
