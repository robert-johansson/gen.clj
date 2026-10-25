;; The code of book/033-trace-translators.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.trace-translators
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.array.special :as special]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as diag]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.inference.transform :as tf]
            [gen.random :as rng]))

;; book: plane-models
(def polar
  (gen []
    (let [r (trace :r (d/gamma 1 1))
          theta (trace :theta (d/uniform (- Math/PI) Math/PI))]
      [r theta])))

(def cartesian
  (gen []
    (let [x (trace :x (d/normal 0 1))
          y (trace :y (d/normal 0 1))]
      (trace :obsx (d/normal x 0.5))
      (trace :obsy (d/normal y 0.5))
      [x y])))

;; book: plane-transforms
(tf/deftransform polar->cartesian
  "(r, θ) to (x, y) = (r cos θ, r sin θ)."
  [model-in aux-in model-out aux-out]
  (let [r (tf/read model-in :r :continuous)
        theta (tf/read model-in :theta :continuous)]
    (tf/write model-out :x (* r (array/cos theta)) :continuous)
    (tf/write model-out :y (* r (array/sin theta)) :continuous)))

(tf/deftransform cartesian->polar
  "(x, y) to (r, θ) = (√(x² + y²), atan2(y, x))."
  [model-in aux-in model-out aux-out]
  (let [x (tf/read model-in :x :continuous)
        y (tf/read model-in :y :continuous)]
    (tf/write model-out :r (array/sqrt (+ (* x x) (* y y))) :continuous)
    (tf/write model-out :theta (array/atan2 y x) :continuous)))

(def to-cartesian
  "polar->cartesian, paired with cartesian->polar as its inverse."
  (first (tf/pair polar->cartesian cartesian->polar)))

;; book: run
(def point (cm/choicemap :r 2.0 :theta 0.9))

(def moved (tf/run to-cartesian point nil))

(cm/to-map (:model moved))
;=> {:x 1.243 :y 1.567}

(array/allclose? (:log-abs-det moved) (Math/log 2.0))
;=> true

;; book: finite-differences
(defn log-abs-det-fd
  "log |det J| of f, a function from a vector of n numbers to another, at
   x: the Jacobian by central differences of step h."
  [f x h]
  (let [column (fn [j]
                 (let [at (fn [dx] (f (update x j + dx)))]
                   (mapv #(/ (- %1 %2) (* 2 h)) (at h) (at (- h)))))
        ;; J's columns as rows: its transpose, with the same determinant
        columns (array/array (mapv column (range (count x))))]
    (array/item (second (linalg/slogdet columns)))))

(defn polar->xy
  "The map of polar->cartesian on host numbers."
  [[r theta]]
  [(* r (Math/cos theta)) (* r (Math/sin theta))])

(< (abs (- (log-abs-det-fd polar->xy [2.0 0.9] 1e-5) (Math/log 2.0)))
   1e-6)
;=> true

;; book: round-trip
(let [back (tf/run (tf/inverse to-cartesian) (:model moved) nil)]
  [(tf/same-choices? (:model back) point)
   (array/allclose? (+ (:log-abs-det moved) (:log-abs-det back)) 0.0)])
;=> [true true]

;; book: copies
(tf/deftransform shuffle
  "(u, v, x, y) to (u, 2u - v, y, x): the thesis's example of copies."
  [model-in aux-in model-out aux-out]
  (let [u (tf/read model-in :u :continuous)
        v (tf/read model-in :v :continuous)]
    (tf/copy model-in :u model-out :u)
    (tf/write model-out :v (- (* 2 u) v) :continuous)
    (tf/copy model-in :y model-out :x)
    (tf/copy model-in :x model-out :y)))

(defn negligible?
  "Is the host number of x within 1e-5 of 0?"
  [x]
  (< (abs (array/item x)) 1e-5))

(def shuffled
  (tf/run shuffle (cm/choicemap :u 0.3 :v 1.1 :x 2.0 :y -1.0) nil))

(cm/to-map (:model shuffled))
;=> {:u 0.3 :v -0.5 :x -1.0 :y 2.0}

(negligible? (:log-abs-det shuffled))
;=> true

;; book: translator
(def observed (cm/choicemap :obsx 1.2 :obsy 1.6))

(def translate
  "The trace translator from polar to cartesian given observed."
  (tf/translator {:p-new cartesian :p-new-args [] :new-observations observed
                  :transform to-cartesian :check? true}))

(def t1 (:trace (gfi/generate polar [] point (rng/key 1))))

(def translated (translate t1 (rng/key 2)))

(cm/to-map (:choices (:trace translated)))
;=> {:x 1.243 :y 1.567 :obsx 1.2 :obsy 1.6}

;; book: translator-weight
(defn log-density
  "The log density of distribution dist at v, a host number."
  [dist v]
  (array/item (dc/log-prob dist v)))

(def weight-by-hand
  (let [[x y] [(* 2.0 (Math/cos 0.9)) (* 2.0 (Math/sin 0.9))]
        log-p2 (+ (log-density (d/normal 0 1) x) (log-density (d/normal 0 1) y)
                  (log-density (d/normal x 0.5) 1.2)
                  (log-density (d/normal y 0.5) 1.6))
        log-p1 (+ (log-density (d/gamma 1 1) 2.0)
                  (log-density (d/uniform (- Math/PI) Math/PI) 0.9))]
    (+ (- log-p2 log-p1) (Math/log 2.0))))

[(:weight translated) (array/allclose? (:weight translated) weight-by-hand)]
;=> [0.2356 true]

;; book: translator-is
(defn mean-se
  "The mean of numbers xs and its standard error."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        variance (/ (reduce + (map #(* (- % mean) (- % mean)) xs)) (dec n))]
    {:mean mean :se (Math/sqrt (/ variance n))}))

(def translations
  "[x weight] of each of 4000 traces of polar, translated."
  (mapv (fn [key]
          (let [[k1 k2] (rng/split key)
                {t :trace w :weight} (translate (gfi/simulate polar [] k1) k2)]
            [(array/item (t :x)) (array/item w)]))
        (rng/split (rng/key 3) 4000)))

(def evidence
  "p2(ρ2): obsx and obsy are independent, each N(0, √1.25)."
  (Math/exp (+ (log-density (d/normal 0 (Math/sqrt 1.25)) 1.2)
               (log-density (d/normal 0 (Math/sqrt 1.25)) 1.6))))

(let [{:keys [mean se]} (mean-se (mapv #(Math/exp (second %)) translations))]
  [evidence (< (abs (- mean evidence)) (* 5 se))])
;=> [0.0257 true]

;; book: translator-posterior
(defn weighted-mean-se
  "The self-normalized estimate Σ w·x / Σ w of [x log-w] pairs, with its
   standard error by the delta method."
  [pairs]
  (let [ws (mapv #(Math/exp (second %)) pairs)
        total (reduce + ws)
        mean (/ (reduce + (map #(* %1 (first %2)) ws pairs)) total)
        terms (map #(* %1 %1 (- (first %2) mean) (- (first %2) mean)) ws pairs)]
    {:mean mean :se (/ (Math/sqrt (reduce + terms)) total)}))

(def x-posterior
  "The exact posterior of x given obsx: N(1.2/1.25, √(0.25/1.25))."
  (d/normal 0.96 (Math/sqrt 0.2)))

(let [{:keys [mean se]} (weighted-mean-se translations)]
  (< (abs (- mean 0.96)) (* 5 se)))
;=> true

;; book: translator-figure
;; figure: figures/033-translated.svg
(svg/histogram [{:values (mapv first translations)
                 :weights (mapv #(Math/exp (second %)) translations)
                 :density #(Math/exp (log-density x-posterior %))
                 :label "translated"}]
               {:range [-1.0 3.0] :bins 30 :x-label "x"})

;; book: one-to-two
(def one-mean (gen [] (trace :s (d/normal 0 (Math/sqrt 2)))))

(def two-means (gen [] (trace :a (d/normal 0 1)) (trace :b (d/normal 0 1))))

(def difference (gen [_] (trace :d (d/normal 0 (Math/sqrt 2)))))

(tf/deftransform to-two
  "(s, d) to (a, b) = ((s + d)/2, (s - d)/2)."
  [model-in aux-in model-out aux-out]
  (let [s (tf/read model-in :s :continuous)
        dd (tf/read aux-in :d :continuous)]
    (tf/write model-out :a (/ (+ s dd) 2) :continuous)
    (tf/write model-out :b (/ (- s dd) 2) :continuous)))

(tf/deftransform to-one
  "(a, b) to (s, d) = (a + b, a - b)."
  [model-in aux-in model-out aux-out]
  (let [a (tf/read model-in :a :continuous)
        b (tf/read model-in :b :continuous)]
    (tf/write model-out :s (+ a b) :continuous)
    (tf/write aux-out :d (- a b) :continuous)))

(def splitting (tf/pair to-two to-one))

(def split-one
  (tf/translator {:p-new two-means :p-new-args [] :q-forward difference
                  :q-forward-args [] :transform (first splitting)
                  :check? true}))

(def merge-two
  (tf/translator {:p-new one-mean :p-new-args [] :q-backward difference
                  :q-backward-args [] :transform (second splitting)
                  :check? true}))

;; book: one-to-two-weights
(map (fn [key]
       (let [[k1 k2 k3] (rng/split key 3)
             {t2 :trace w :weight} (split-one (gfi/simulate one-mean [] k1) k2)]
         [(negligible? w) (negligible? (:weight (merge-two t2 k3)))]))
     (rng/split (rng/key 4) 3))
;=> [[true true] [true true] [true true]]

(:log-abs-det (tf/run (first splitting) (cm/choicemap :s 1.0)
                      (cm/choicemap :d 0.4)))
;=> -0.6931

;; book: rate-model
(def rate-model
  (gen [n]
    (let [rate (trace :rate (d/gamma 1 1))]
      (doseq [i (range n)]
        (trace [:count i] (d/poisson rate)))
      rate)))

(def counts [2 0 1])

(defn rate-trace
  "The trace of rate-model at rate, given the first n counts."
  [rate n]
  (let [observed (mapcat (fn [i c] [[:count i] c]) (range n) counts)]
    (:trace (gfi/generate rate-model [n]
                          (apply cm/choicemap :rate rate observed)
                          (rng/key 0)))))

;; book: identity
(def nothing "An auxiliary generative function of no choices." (gen [_] nil))

(tf/deftransform identity-move
  "The identity: the rate copied."
  [model-in aux-in model-out aux-out]
  (tf/copy model-in :rate model-out :rate))

(negligible? (:log-alpha (mh/involutive-mh (rate-trace 0.7 3) nothing []
                                           (tf/involution identity-move)
                                           (rng/key 5))))
;=> true

;; book: reciprocal
(tf/deftransform reciprocal
  "The rate to its reciprocal."
  [model-in aux-in model-out aux-out]
  (tf/write model-out :rate (/ 1 (tf/read model-in :rate :continuous))
            :continuous))

(defn reciprocal-weight
  "The thesis's weight of the reciprocal at x, in logs: log p(1/x) - log
   p(x) - 2 log x, p the density of Gamma(1, 1)."
  [x]
  (- (log-density (d/gamma 1 1) (/ 1 x)) (log-density (d/gamma 1 1) x)
     (* 2 (Math/log x))))

(for [x [2.0 0.3]]
  (let [{la :log-alpha} (mh/involutive-mh (rate-trace x 0) nothing []
                                          (tf/involution reciprocal)
                                          (rng/key 6))]
    [la (array/allclose? la (reciprocal-weight x))]))
;=> [[0.1137 true] [-0.6254 true]]

;; book: conditional-swap
(def exact-rate
  "The exact conditional of the rate given the three counts."
  (gen [_] (trace :rate (d/gamma 4 0.25))))

(tf/deftransform swap
  "The model's rate and the auxiliary rate exchanged."
  [model-in aux-in model-out aux-out]
  (tf/copy aux-in :rate model-out :rate)
  (tf/copy model-in :rate aux-out :rate))

(for [[x key] (map vector [0.3 1.0 2.5] (rng/split (rng/key 7) 3))]
  (negligible? (:log-alpha (mh/involutive-mh (rate-trace x 3) exact-rate []
                                             (tf/involution swap) key))))
;=> [true true true]

;; book: propose-swap
(def independent
  "A proposal for the rate that ignores the trace."
  (gen [_] (trace :rate (d/gamma 2 0.5))))

(for [key (rng/split (rng/key 8) 3)]
  (let [[k1 k2] (rng/split key)
        tr (rate-trace 0.7 3)
        swapped (mh/move tr independent [] {:involution (tf/involution swap)}
                         k1 k2)
        plain (mh/move tr independent [] {} k1 k2)
        v' (array/item ((:trace swapped) :rate))
        by-hand (+ (- (:score (:trace swapped)) (:score tr))
                   (- (log-density (d/gamma 2 0.5) 0.7)
                      (log-density (d/gamma 2 0.5) v')))]
    [(array/allclose? (:weight swapped) by-hand)
     (array/allclose? (:weight swapped) (:weight plain))]))
;=> [[true true] [true true] [true true]]

;; book: by-hand
(defn log-alpha
  "The move from trace tr with auxiliary choices u, from the interface:
   h applied to the trace and u, the trace updated with h's model choices,
   and h's auxiliary choices u' assessed under q at the new trace. Returns
   the proposed trace and log α of Eq. 3.17, a host number."
  [tr q h u key]
  (let [fwd (:weight (gfi/assess q [tr] u key))
        {model :model u' :aux ldj :log-abs-det} (tf/run h tr u)
        {tr' :trace w :weight} (gfi/update tr model key)
        bwd (:weight (gfi/assess q [tr'] u' key))]
    {:trace tr' :log-alpha (array/item (+ w ldj (- bwd fwd)))}))

(defn involutive-mh-by-hand
  "Alg. 12: u from q, then the proposed trace with probability min(1, α),
   else tr."
  [tr q h key]
  (let [[k-propose k-update k-accept] (rng/split key 3)
        u (:choices (gfi/propose q [tr] k-propose))
        {tr' :trace la :log-alpha} (log-alpha tr q h u k-update)]
    (if (< (Math/log (array/item (rng/uniform k-accept))) la) tr' tr)))

;; book: scale
(def log-step (gen [_] (trace :u (d/normal 0 0.5))))

(tf/deftransform scale
  "(rate, u) to (rate·exp(u), -u): a random walk on the rate's logarithm."
  [model-in aux-in model-out aux-out]
  (let [rate (tf/read model-in :rate :continuous)
        u (tf/read aux-in :u :continuous)]
    (tf/write model-out :rate (* rate (array/exp u)) :continuous)
    (tf/write aux-out :u (- u) :continuous)))

;; book: by-hand-check
(for [key (rng/split (rng/key 9) 4)]
  (let [[k1 k2] (rng/split key)
        tr (rate-trace 0.7 3)
        u (:choices (gfi/propose log-step [tr] k1))
        mine (log-alpha tr log-step scale u k2)
        library (mh/move tr log-step [] {:involution (tf/involution scale)}
                         k1 k2)]
    [(array/allclose? (:log-alpha mine) (:weight library))
     (tf/same-choices? (:choices (:trace mine))
                       (:choices (:trace library)))]))
;=> [[true true] [true true] [true true] [true true]]

;; book: scale-jacobian
(let [{:keys [log-abs-det]} (tf/run scale (cm/choicemap :rate 0.7)
                                    (cm/choicemap :u 0.4))
      fd (log-abs-det-fd (fn [[rate u]] [(* rate (Math/exp u)) (- u)])
                         [0.7 0.4] 1e-6)]
  [(array/allclose? log-abs-det 0.4) (< (abs (- fd 0.4)) 1e-6)])
;=> [true true]

;; book: scale-chain
(def rates
  "The rate along 4000 moves of the scale move, from rate 1."
  (mapv #(array/item (% :rate))
        (reductions #(involutive-mh-by-hand %1 log-step scale %2)
                    (rate-trace 1.0 3)
                    (rng/split (rng/key 10) 4000))))

(def rate-summary (diag/summary rates))

(< (abs (- (:mean rate-summary) 1.0)) (* 5 (:mcse-mean rate-summary)))
;=> true

;; book: levels
(def levels
  (gen []
    (let [k (trace :k (d/uniform-discrete 0 5))]
      (trace :y (d/normal k 1))
      k)))

(def up-probabilities
  "The probability of a step up from each level: always from 0, never
   from 5."
  (array/array [1.0 0.5 0.5 0.5 0.5 0.0]))

(def direction
  (gen [tr]
    (trace :up? (d/bernoulli (array/take up-probabilities (tr :k))))))

(tf/deftransform step
  "A step up or down, and the direction reversed."
  [model-in aux-in model-out aux-out]
  (let [k (tf/read model-in :k :discrete)
        up? (tf/read aux-in :up? :discrete)]
    (tf/write model-out :k (if up? (inc k) (dec k)) :discrete)
    (tf/write aux-out :up? (not up?) :discrete)))

;; book: enumerate
(def level-traces
  "The trace at each level, given y = 3.4."
  (mapv #(:trace (gfi/generate levels [] (cm/choicemap :k % :y 3.4)
                               (rng/key 0)))
        (range 6)))

(def posterior-levels
  "The exact posterior of the level, by enumeration."
  (let [ps (mapv #(Math/exp (array/item (:score %))) level-traces)
        total (reduce + ps)]
    (mapv #(/ % total) ps)))

(defn probability
  "q(u; tr): the probability that q, given trace tr, chooses u."
  [q tr u]
  (Math/exp (array/item (:weight (gfi/assess q [tr] u (rng/key 0))))))

(defn transitions
  "The kernel's transition matrix: row i holds the probabilities of a move
   from level i to each level, summed over the auxiliary choices q can
   make there, each move accepted with probability min(1, α)."
  [q h]
  (vec (for [[i tr] (map-indexed vector level-traces)]
         (reduce (fn [row u]
                   (let [p (probability q tr u)]
                     (if (zero? p)
                       row
                       (let [{tr' :trace la :log-alpha}
                             (log-alpha tr q h u (rng/key 0))
                             a (min 1.0 (Math/exp la))]
                         (-> row
                             (update (long (array/item (tr' :k))) + (* p a))
                             (update i + (* p (- 1 a))))))))
                 (vec (repeat 6 0.0))
                 [(cm/choicemap :up? true) (cm/choicemap :up? false)]))))

;; book: detailed-balance
(def walk (transitions direction step))

(defn balanced?
  "Does π(i) K(i, j) = π(j) K(j, i) hold for every pair of levels?"
  [pi kernel]
  (every? true? (for [i (range 6) j (range 6)]
                  (< (abs (- (* (pi i) (get-in kernel [i j]))
                             (* (pi j) (get-in kernel [j i]))))
                     1e-5))))

(defn stationary?
  "Is π K = π?"
  [pi kernel]
  (every? true? (for [j (range 6)]
                  (< (abs (- (reduce + (map #(* (pi %) (get-in kernel [% j]))
                                            (range 6)))
                             (pi j)))
                     1e-5))))

[(balanced? posterior-levels walk) (stationary? posterior-levels walk)]
;=> [true true]

;; book: library-agrees
(every? true?
        (for [tr level-traces
              key (rng/split (rng/key 11) 3)]
          (let [[k1 k2] (rng/split key)
                {tr' :trace w :weight}
                (mh/move tr direction [] {:involution (tf/involution step)}
                         k1 k2)
                u (cm/choicemap :up? (> (array/item (tr' :k))
                                        (array/item (tr :k))))]
            (array/allclose? w (:log-alpha (log-alpha tr direction step u
                                                      (rng/key 0)))))))
;=> true

;; book: boundary
(defn step-up
  "log α of the step up from level k."
  [k]
  (:log-alpha (log-alpha (level-traces k) direction step
                         (cm/choicemap :up? true) (rng/key 0))))

(defn log-odds
  "log π(j)/π(i)."
  [i j]
  (Math/log (/ (posterior-levels j) (posterior-levels i))))

[(array/allclose? (step-up 2) (log-odds 2 3))
 (array/allclose? (step-up 0) (+ (log-odds 0 1) (Math/log 0.5)))]
;=> [true true]

;; book: flows-figure
;; figure: figures/033-flows.svg
(svg/heatmap (range 6) (range 6)
             (vec (for [i (range 6)]
                    (vec (for [j (range 6)]
                           (if (= i j)
                             0.0
                             (* (posterior-levels i) (get-in walk [i j])))))))
             {:labels? true :x-label "to level" :y-label "from level"})

;; book: no-flip
;; error: The inverse transform does not give back the trace
#_(tf/deftransform step-no-flip
  "A step up or down, with the direction kept: not an involution."
  [model-in aux-in model-out aux-out]
  (let [k (tf/read model-in :k :discrete)
        up? (tf/read aux-in :up? :discrete)]
    (tf/write model-out :k (if up? (inc k) (dec k)) :discrete)
    (tf/write aux-out :up? up? :discrete)))

#_(def checked-no-flip (tf/involution step-no-flip {:check? true}))

#_(checked-no-flip (level-traces 2) (cm/choicemap :up? true) nil [] (rng/key 0))

;; book: branching-model
(def branching-model
  (gen []
    (let [[m1 m2] (if (trace :z (d/bernoulli 0.5))
                    [(trace :m1 (d/gamma 1 1)) (trace :m2 (d/gamma 1 1))]
                    (let [m (trace :m (d/gamma 1 1))] [m m]))]
      (trace :y1 (d/normal m1 0.1))
      (trace :y2 (d/normal m2 0.1))
      [m1 m2])))

(def data (cm/choicemap :y1 1.0 :y2 1.3))

;; book: exact-two
(defn normal-cdf
  "Φ(x), the standard normal's distribution function."
  [x]
  (* 0.5 (special/erfc (/ (- x) (Math/sqrt 2)))))

(defn gamma-normal
  "The integral of exp(-m) N(m; mu, s) over m > 0, which is
   exp(s²/2 - mu) Φ((mu - s²)/s)."
  [mu s]
  (* (Math/exp (- (/ (* s s) 2) mu)) (normal-cdf (/ (- mu (* s s)) s))))

(defn p-two-of
  "P(z | y1 = 1.0, y2 = 1.3) under observation noise sigma: each branch's
   marginal likelihood, normalized (the prior on z is even)."
  [sigma]
  (let [two (* (gamma-normal 1.0 sigma) (gamma-normal 1.3 sigma))
        one (* (Math/exp (log-density (d/normal 0 (* (Math/sqrt 2) sigma))
                                      -0.3))
               (gamma-normal 1.15 (/ sigma (Math/sqrt 2))))]
    (/ two (+ two one))))

(def p-two (p-two-of 0.1))

[p-two (p-two-of 0.4)]
;=> [0.5176 0.3635]

;; book: quadrature
(defn integrate
  "The integral of f over [lo, hi], by the midpoint rule with n intervals."
  [f lo hi n]
  (let [h (/ (- hi lo) n)]
    (* h (reduce + (map #(f (+ lo (* h (+ % 0.5)))) (range n))))))

(let [prior #(Math/exp (- %))
      likelihood (fn [m y] (Math/exp (log-density (d/normal m 0.1) y)))
      two (* (integrate #(* (prior %) (likelihood % 1.0)) 0 5 20000)
             (integrate #(* (prior %) (likelihood % 1.3)) 0 5 20000))
      one (integrate #(* (prior %) (likelihood % 1.0) (likelihood % 1.3))
                     0 5 20000)]
  (< (abs (- p-two (/ two (+ two one)))) 1e-6))
;=> true

;; book: split-merge
(def branch-proposal
  (gen [tr]
    (when-not (tr :z)
      (trace :dof (d/uniform 0 1)))))

(defn merge-means
  "[m dof]: the geometric mean of m1 and m2, and m1's share of their sum."
  [m1 m2]
  [(array/sqrt (* m1 m2)) (/ m1 (+ m1 m2))])

(defn split-mean
  "[m1 m2] of geometric mean m and share dof: merge-means' inverse."
  [m dof]
  [(* m (array/sqrt (/ dof (- 1 dof)))) (* m (array/sqrt (/ (- 1 dof) dof)))])

(tf/deftransform split-merge
  "Two means merged into one, or one mean split into two."
  [model-in aux-in model-out aux-out]
  (if (tf/read model-in :z :discrete)
    (let [[m dof] (merge-means (tf/read model-in :m1 :continuous)
                               (tf/read model-in :m2 :continuous))]
      (tf/write model-out :z false :discrete)
      (tf/write model-out :m m :continuous)
      (tf/write aux-out :dof dof :continuous))
    (let [[m1 m2] (split-mean (tf/read model-in :m :continuous)
                              (tf/read aux-in :dof :continuous))]
      (tf/write model-out :z true :discrete)
      (tf/write model-out :m1 m1 :continuous)
      (tf/write model-out :m2 m2 :continuous))))

;; book: split-jacobian
(defn branch-trace
  "The trace of branching-model with the data and these latent choices,
   address and value in turn."
  [& choices]
  (:trace (gfi/generate branching-model []
                        (cm/merge data (apply cm/choicemap choices))
                        (rng/key 0))))

(def one-trace (branch-trace :z false :m 1.15))

(def split (tf/run split-merge one-trace (cm/choicemap :dof 0.4)))

(cm/to-map (:model split))
;=> {:z true :m1 0.939 :m2 1.408}

(def split-log-jacobian (Math/log (/ 1.15 (* 0.4 0.6))))

[(array/allclose? (:log-abs-det split) split-log-jacobian)
 (< (abs (- (log-abs-det-fd #(apply split-mean %) [1.15 0.4] 1e-6)
            split-log-jacobian))
    1e-6)]
;=> [true true]

;; book: split-twice
(let [two-trace (:trace (gfi/update one-trace (:model split) (rng/key 0)))
      merged (tf/run split-merge two-trace (:aux split))]
  [(tf/same-choices? (:model merged) (cm/choicemap :z false :m 1.15))
   (tf/same-choices? (:aux merged) (cm/choicemap :dof 0.4))
   (negligible? (+ (:log-abs-det split) (:log-abs-det merged)))])
;=> [true true true]

;; book: four-pieces
(def split-pieces
  (let [u (cm/choicemap :dof 0.4)
        fwd (:weight (gfi/assess branch-proposal [one-trace] u (rng/key 0)))
        {two-trace :trace w :weight} (gfi/update one-trace (:model split)
                                                 (rng/key 0))
        bwd (:weight (gfi/assess branch-proposal [two-trace] (:aux split)
                                 (rng/key 0)))]
    {:update w :jacobian (:log-abs-det split) :forward fwd :backward bwd}))

split-pieces
;=> {:update 0.2782 :jacobian 1.5669 :forward 0.0 :backward 0.0}

;; book: rj-formula
(defn log-likelihood
  "The observations' log density given the means: log l(m1, m2)."
  [m1 m2]
  (+ (log-density (d/normal m1 0.1) 1.0) (log-density (d/normal m2 0.1) 1.3)))

(defn log-prior
  "The log density of means under their Gamma(1, 1) priors."
  [& ms]
  (reduce + (map #(log-density (d/gamma 1 1) %) ms)))

(let [[m1 m2] (split-mean 1.15 0.4)
      reversible-jump (+ (- (Math/log 0.5) (Math/log 0.5))
                         (- (log-prior m1 m2) (log-prior 1.15))
                         (- (log-likelihood m1 m2) (log-likelihood 1.15 1.15))
                         split-log-jacobian)
      {la :log-alpha} (log-alpha one-trace branch-proposal split-merge
                                 (cm/choicemap :dof 0.4) (rng/key 0))]
  [(array/allclose? reversible-jump la)
   (array/allclose? la (reduce + (vals split-pieces)))])
;=> [true true]

;; book: rj-chain
(def within
  "A random walk on the means of the trace's branch."
  (gen [tr]
    (if (tr :z)
      (do (trace :m1 (d/normal (tr :m1) 0.1))
          (trace :m2 (d/normal (tr :m2) 0.1)))
      (trace :m (d/normal (tr :m) 0.1)))))

(defn jump-sweep
  "A sweep: an involutive move of auxiliary q and transform h, then a
   random walk within the branch."
  [q h]
  (let [involution (tf/involution h)]
    (fn [tr key]
      (let [[k1 k2] (rng/split key)
            jumped (mh/involutive-mh tr q [] involution k1)]
        (:trace (mh/mh (:trace jumped) within [] k2))))))

(def rj-sweep (jump-sweep branch-proposal split-merge))

(def start
  "The thesis's starting trace: one mean, at 1."
  (branch-trace :z false :m 1.0))

(defn run-sweeps
  "[z m1 m2] after each of n sweeps from start, as host values (m1 = m2 = m
   in the branch of one mean)."
  [sweep n key]
  (mapv (fn [tr] (into [(tr :z)] (map array/item (:retval tr))))
        (rest (reductions sweep start (rng/split key n)))))

(defn two-fraction
  "The summary of the indicator of two means along a run."
  [run]
  (diag/summary (mapv #(if (first %) 1.0 0.0) run)))

(def rj-run (run-sweeps rj-sweep 4000 (rng/key 12)))

(def rj-two (two-fraction rj-run))

(< (abs (- (:mean rj-two) p-two)) (* 5 (:mcse-mean rj-two)))
;=> true

;; book: mh-switch
(def switch
  "The other branch, its means drawn from their prior."
  (gen [tr]
    (if (trace :z (d/bernoulli (if (tr :z) 0.0 1.0)))
      (do (trace :m1 (d/gamma 1 1))
          (trace :m2 (d/gamma 1 1)))
      (trace :m (d/gamma 1 1)))))

(defn mh-sweep
  "An MH move to the other branch, then a random walk within the branch."
  [tr key]
  (let [[k1 k2] (rng/split key)]
    (:trace (mh/mh (:trace (mh/mh tr switch [] k1)) within [] k2))))

(def mh-run (run-sweeps mh-sweep 4000 (rng/key 12)))

(defn crossings
  "The number of sweeps of a run that changed the branch."
  [run]
  (count (filter true? (map not= (map first run) (rest (map first run))))))

[(crossings rj-run) (crossings mh-run)]
;=> [610 90] ; float64

;; book: crossing-rates
(let [n 3999
      rate #(/ (crossings %) (double n))
      [a b] [(rate rj-run) (rate mh-run)]
      se (Math/sqrt (+ (/ (* a (- 1 a)) n) (/ (* b (- 1 b)) n)))]
  (> (- a b) (* 5 se)))
;=> true

(def mh-two (two-fraction mh-run))

[(Math/round (:ess-bulk rj-two)) (Math/round (:ess-bulk mh-two))]
;=> [525 70] ; float64

(< (abs (- (:mean mh-two) p-two)) (* 5 (:mcse-mean mh-two)))
;=> true

;; book: means-figure
;; figure: figures/033-split-merge.svg
(defn mean-paths
  "The means along a run, as svg/paths layers: m in blue, m1 in orange and
   m2 in green, each path a stay in one branch."
  [run]
  (let [stays (partition-by (comp first second) (map-indexed vector run))
        path (fn [stay i] (mapv (fn [[t v]] [t (v i)]) stay))
        two? (fn [stay] (first (second (first stay))))]
    [{:paths (for [s stays :when (not (two? s))] (path s 1))
      :color "steelblue" :r 1.5}
     {:paths (for [s stays :when (two? s)] (path s 1))
      :color "darkorange" :r 1.5}
     {:paths (for [s stays :when (two? s)] (path s 2))
      :color "seagreen" :r 1.5}]))

(svg/paths (mean-paths (subvec rj-run 0 200))
           {:x-label "sweep" :y-label "mean"})

;; book: running-figure
;; figure: figures/033-two-means.svg
(defn running-fraction
  "The fraction of a run's first t sweeps in the branch of two means, for
   each t, as points [t fraction]."
  [run]
  (map-indexed (fn [t total] [t (/ total (inc t))])
               (reductions + (map #(if (first %) 1.0 0.0) run))))

(svg/paths [{:paths [(running-fraction rj-run)] :color "steelblue"}
            {:paths [(running-fraction mh-run)] :color "darkorange"}
            {:paths [[[0 p-two] [4000 p-two]]] :color "black" :width 1}]
           {:x-label "sweep" :y-label "P(z | y)" :y-range [-0.05 1.05]})

;; book: smcp3-model
(def mean-model
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (doseq [t (range n)]
        (trace [:y t] (d/normal mu 1)))
      mu)))

(def ys [0.9 1.7 0.4 1.2 1.5])

(defn posterior-after
  "[m s]: the exact posterior N(m, s) of mu given the first t observations."
  [t]
  [(/ (reduce + (take t ys)) (+ 1 t)) (Math/sqrt (/ 1 (+ 1 t)))])

;; book: smcp3-step
(def standard-normal (gen [_] (trace :u (d/normal 0 1))))

(defn refit
  "The transform of step t: mu' = m' + s'·u, from the auxiliary u, and
   u' = (mu - m)/s for the backward kernel, (m, s) and (m', s') the exact
   posteriors before observation t and after it."
  [t]
  (let [[m s] (posterior-after t)
        [m' s'] (posterior-after (inc t))]
    (tf/transform [model-in aux-in model-out aux-out]
      (let [mu (tf/read model-in :mu :continuous)
            u (tf/read aux-in :u :continuous)]
        (tf/write model-out :mu (+ m' (* s' u)) :continuous)
        (tf/write aux-out :u (/ (- mu m) s) :continuous)))))

(defn smcp3-step
  "The SMCP3 step that adds observation t."
  [t]
  (tf/updating-translator
   {:p-new-args [(inc t)] :new-observations (cm/choicemap [:y t] (ys t))
    :q-forward standard-normal :q-backward standard-normal
    :transform (refit t)}))

;; book: smcp3-weights
(defn log-predictive
  "log p(y_t | y_0 … y_t-1): normal with variance s² + 1 around m."
  [t]
  (let [[m s] (posterior-after t)]
    (log-density (d/normal m (Math/sqrt (+ (* s s) 1))) (ys t))))

(let [prior (pf/initialize mean-model [0] cm/EMPTY 5 {:key (rng/key 13)})
      stepped (pf/move-reweight prior (smcp3-step 0) (rng/key 14))]
  (every? #(array/allclose? % (log-predictive 0)) (pf/log-weights stepped)))
;=> true

;; book: smcp3-run
(defn run-smcp3
  "n particles of mean-model from its prior, moved through the five
   observations by the steps (step t), with move-reweight."
  [step n key]
  (let [[k-init k-steps] (rng/split key)]
    (reduce (fn [state [t k]] (pf/move-reweight state (step t) k))
            (pf/initialize mean-model [0] cm/EMPTY n {:key k-init})
            (map vector (range 5) (rng/split k-steps 5)))))

(def smcp3-state (run-smcp3 smcp3-step 100 (rng/key 15)))

(def exact-log-ml (reduce + (map log-predictive (range 5))))

[(array/allclose? (pf/log-ml-estimate smcp3-state) exact-log-ml)
 (every? #(array/allclose? % exact-log-ml) (pf/log-weights smcp3-state))]
;=> [true true]

;; book: smcp3-particles
(let [[m s] (posterior-after 5)
      mus (mapv #(array/item (% :mu)) (pf/traces smcp3-state))
      sd (* (:se (mean-se mus)) 10)]
  [m s
   (< (abs (- (/ (reduce + mus) 100) m)) (* 5 (/ s 10)))
   (< (abs (- sd s)) (* 5 (/ s (Math/sqrt 198))))])
;=> [0.95 0.4082 true true]

;; book: smcp3-no-backward
;; error: updating-translator: there are backward choices but no :q-backward
#_(def no-backward
  (tf/updating-translator {:p-new-args [1]
                           :new-observations (cm/choicemap [:y 0] 0.9)
                           :q-forward standard-normal :transform (refit 0)}))

#_(no-backward (gfi/simulate mean-model [0] (rng/key 17)) (rng/key 18))

;; book: solution-1
(def offset-proposal
  (gen [tr]
    (when-not (tr :z)
      (trace :u (d/normal 0 0.2)))))

(tf/deftransform offset-split-merge
  "Two means merged into their mean and half their difference, or one mean
   split into m - u and m + u."
  [model-in aux-in model-out aux-out]
  (if (tf/read model-in :z :discrete)
    (let [m1 (tf/read model-in :m1 :continuous)
          m2 (tf/read model-in :m2 :continuous)]
      (tf/write model-out :z false :discrete)
      (tf/write model-out :m (/ (+ m1 m2) 2) :continuous)
      (tf/write aux-out :u (/ (- m2 m1) 2) :continuous))
    (let [m (tf/read model-in :m :continuous)
          u (tf/read aux-in :u :continuous)]
      (tf/write model-out :z true :discrete)
      (tf/write model-out :m1 (- m u) :continuous)
      (tf/write model-out :m2 (+ m u) :continuous))))

(:log-abs-det (tf/run offset-split-merge one-trace (cm/choicemap :u 0.15)))
;=> 0.6931

;; book: solution-1-chain
(def offset-run
  (run-sweeps (jump-sweep offset-proposal offset-split-merge) 4000
              (rng/key 19)))

(def offset-two (two-fraction offset-run))

(< (abs (- (:mean offset-two) p-two)) (* 5 (:mcse-mean offset-two)))
;=> true

[(crossings offset-run) (crossings rj-run)]
;=> [1680 610] ; float64

;; book: solution-2
(defn scale-without-jacobian
  "The scale move as an involution function, with its Jacobian left out."
  [tr u _ _ key]
  (let [step (cm/get-choice u :u)
        moved (cm/choicemap :rate (* (tr :rate) (array/exp step)))
        {tr' :trace w :weight} (gfi/update tr moved key)]
    {:trace tr' :bwd-choices (cm/choicemap :u (- step)) :weight w}))

(def biased-rates
  (mapv #(array/item (% :rate))
        (reductions #(:trace (mh/involutive-mh %1 log-step []
                                               scale-without-jacobian %2))
                    (rate-trace 1.0 3)
                    (rng/split (rng/key 20) 4000))))

(let [{:keys [mean mcse-mean]} (diag/summary biased-rates)]
  [(< (abs (- mean 0.75)) (* 5 mcse-mean))
   (> (abs (- mean 1.0)) (* 5 mcse-mean))])
;=> [true true]

;; book: solution-3
(tf/deftransform ratio-split-merge
  "split-merge with the wrong reverse move: a merge writes m1/m2 as dof."
  [model-in aux-in model-out aux-out]
  (if (tf/read model-in :z :discrete)
    (let [m1 (tf/read model-in :m1 :continuous)
          m2 (tf/read model-in :m2 :continuous)]
      (tf/write model-out :z false :discrete)
      (tf/write model-out :m (array/sqrt (* m1 m2)) :continuous)
      (tf/write aux-out :dof (/ m1 m2) :continuous))
    (let [[m1 m2] (split-mean (tf/read model-in :m :continuous)
                              (tf/read aux-in :dof :continuous))]
      (tf/write model-out :z true :discrete)
      (tf/write model-out :m1 m1 :continuous)
      (tf/write model-out :m2 m2 :continuous))))

;; book: solution-3-check
;; error: The inverse transform does not give back the forward choices
#_(def checked-ratio (tf/involution ratio-split-merge {:check? true}))

#_(checked-ratio one-trace (cm/choicemap :dof 0.4) nil [] (rng/key 0))

;; book: solution-3-chain
;; error: The backward proposal gives the move's reverse choices density 0
#_(run-sweeps (jump-sweep branch-proposal ratio-split-merge) 4000
            (rng/key 12))

;; book: solution-3-jacobian
(let [two-trace (branch-trace :z true :m1 1.0 :m2 1.3)
      log-jacobian #(:log-abs-det (tf/run % two-trace cm/EMPTY))]
  (Math/exp (- (log-jacobian ratio-split-merge) (log-jacobian split-merge))))
;=> 3.13

;; book: solution-4
(def drifting-probabilities
  "Up from level k with probability (5 - k)/5."
  (array/array [1.0 0.8 0.6 0.4 0.2 0.0]))

(def drifting
  (gen [tr]
    (trace :up? (d/bernoulli (array/take drifting-probabilities (tr :k))))))

(let [kernel (transitions drifting step)]
  [(balanced? posterior-levels kernel) (stationary? posterior-levels kernel)])
;=> [true true]

(mapv #(:log-alpha (log-alpha (level-traces 1) % step (cm/choicemap :up? true)
                              (rng/key 0)))
      [direction drifting])
;=> [1.9 1.2069]

;; book: solution-5
(defn wide-refit
  "refit with a forward kernel twice as wide: mu' = m' + 2s'·u."
  [t]
  (let [[m s] (posterior-after t)
        [m' s'] (posterior-after (inc t))]
    (tf/transform [model-in aux-in model-out aux-out]
      (let [mu (tf/read model-in :mu :continuous)
            u (tf/read aux-in :u :continuous)]
        (tf/write model-out :mu (+ m' (* 2 s' u)) :continuous)
        (tf/write aux-out :u (/ (- mu m) s) :continuous)))))

(defn wide-step
  "smcp3-step with wide-refit."
  [t]
  (tf/updating-translator
   {:p-new-args [(inc t)] :new-observations (cm/choicemap [:y t] (ys t))
    :q-forward standard-normal :q-backward standard-normal
    :transform (wide-refit t)}))

(def wide-estimates
  (mapv #(pf/log-ml-estimate (run-smcp3 wide-step 20 %))
        (rng/split (rng/key 21) 40)))

(let [{:keys [mean se]} (mean-se (mapv #(Math/exp (- % exact-log-ml))
                                       wide-estimates))
      weights (pf/log-weights (run-smcp3 wide-step 20 (rng/key 22)))]
  [(< (abs (- mean 1.0)) (* 5 se))
   (> (- (apply max weights) (apply min weights)) 0.1)])
;=> [true true]
