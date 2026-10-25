;; The code of book/025-smc.md. Each region, from a `;; book:` line to the
;; next, is included there by name (gen.book: `clojure -M:test -m gen.book
;; sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.smc
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as kernel]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

;; book: constants
(def measurement-noise
  "The standard deviation of a bearing, in radians."
  0.005)

(def velocity-sd
  "The standard deviation of a velocity's change in one step (Gen.jl's
   velocity_var is its square, 1e-6)."
  1e-3)

(defn bearing
  "The angle between the positive x axis and the ray from the origin to
   (x, y), between -pi and pi: Gen.jl's atan(y, x)."
  [x y]
  (array/atan2 y x))

;; book: kernel
(def move
  (gen [t {:keys [x y vx vy]} velocity-sd measurement-noise]
    (let [vx (trace :vx (d/normal vx velocity-sd))
          vy (trace :vy (d/normal vy velocity-sd))
          x (+ x vx)
          y (+ y vy)]
      (trace :z (d/normal (bearing x y) measurement-noise))
      {:x x :y y :vx vx :vy vy})))

(def moves (comb/unfold move))

;; book: model
(defn tracker
  "The tracking model, a velocity changing by velocity-sd in one step and
   a bearing measured with noise measurement-noise."
  [velocity-sd measurement-noise]
  (gen [n]
    (let [x (trace :x0 (d/normal 0.01 0.01))
          y (trace :y0 (d/normal 0.95 0.01))
          vx (trace :vx0 (d/normal 0.002 0.01))
          vy (trace :vy0 (d/normal -0.013 0.01))
          start {:x x :y y :vx vx :vy vy}]
      (trace :z0 (d/normal (bearing x y) measurement-noise))
      (into [start]
            (splice :moves moves n start velocity-sd measurement-noise)))))

(def model (tracker velocity-sd measurement-noise))

;; book: small-trace
(def small (gfi/simulate model [2] (rng/key 1)))

(count (cm/addresses (:choices small)))
;=> 11

(take 4 (cm/addresses (:choices small)))
;=> [[:moves 0 :vx] [:moves 0 :vy] [:moves 0 :z] [:moves 1 :vx]]

(count (:retval small))
;=> 3

;; book: data
(def T 50)

(def start-choices
  (cm/choicemap :x0 0.01 :y0 0.95 :vx0 0.002 :vy0 -0.013))

(def truth (:trace (gfi/generate model [T] start-choices (rng/key 2))))

(def zs
  "The observed bearings: z0, then each move's."
  (mapv array/item
        (into [(truth :z0)] (map #(truth [:moves % :z]) (range T)))))

(count zs)
;=> 51

(subvec zs 0 3)
;=> [1.5597 1.5581 1.5507] ; float64

;; book: render
(defn positions
  "The positions (x, y) of trace tr's states."
  [tr]
  (mapv (fn [{:keys [x y]}] [x y]) (:retval tr)))

(defn rays
  "A layer of rays from the origin, one along each bearing of zs."
  [zs]
  {:paths (mapv (fn [z] [[0.0 0.0] [(Math/cos z) (Math/sin z)]]) zs)
   :color "firebrick" :opacity 0.3 :width 1})

(def view
  "The part of the plane the figures show."
  {:x-range [-0.05 0.65] :y-range [0.0 1.0]})

(defn render
  "The bearings zs as rays, and the path of each trace of trs."
  [zs trs]
  (svg/paths [(rays zs) {:paths (mapv positions trs) :width 0.8 :opacity 0.4}]
             view))

;; book: data-figure
;; figure: figures/025-bearings.svg
(svg/paths [(rays zs) {:paths [(positions truth)] :r 2 :width 0}] view)

;; book: observation
(defn observation
  "The choice map of bearing t of zs: z0 at t = 0, and move t - 1's after
   that."
  [zs t]
  (if (zero? t)
    (cm/choicemap :z0 (zs 0))
    (cm/choicemap [:moves (dec t) :z] (zs t))))

;; book: particle-filter
(defn particle-filter
  "A particle filter of n particles of model over the bearings zs,
   revising the state with (prepare state t key) before step t: {:state,
   the state after the last bearing, :ess, the effective sample size after
   each step}. opts go to pf/initialize."
  [model zs n prepare opts key]
  (let [[k-init & ks] (rng/split key (count zs))
        init (pf/initialize model [0] (observation zs 0) n
                            (assoc opts :key k-init))]
    (reduce (fn [{:keys [state ess]} [t k]]
              (let [[k-prepare k-step] (rng/split k)
                    state (pf/step (prepare state t k-prepare)
                                   [t] [diff/unknown-change]
                                   (observation zs t) {:key k-step})]
                {:state state :ess (conj ess (pf/ess state))}))
            {:state init :ess [(pf/ess init)]}
            (map vector (range 1 (count zs)) ks))))

(defn resample
  "Resample when the effective sample size is below half the number of
   particles."
  [state _ key]
  (pf/maybe-resample state {:key key}))

;; book: run-filter
(def filtered
  (particle-filter model zs 5000 resample {:batched? true} (rng/key 3)))

(def samples (pf/sample-unweighted-traces (:state filtered) 100 (rng/key 4)))

;; book: filter-figure
;; figure: figures/025-filter.svg
(render zs samples)

;; book: posterior-shape
(defn final-bearing
  "The bearing of trace tr's last position."
  [tr]
  (let [{:keys [x y]} (peek (:retval tr))] (array/item (bearing x y))))

(defn final-range
  "The distance from the origin of trace tr's last position."
  [tr]
  (let [{:keys [x y]} (peek (:retval tr))]
    (array/item (array/sqrt (+ (* x x) (* y y))))))

(defn spread
  "The mean and standard deviation of (f tr) over the traces trs."
  [f trs]
  (let [xs (mapv f trs)
        m (/ (reduce + xs) (count xs))]
    {:mean m
     :sd (Math/sqrt (/ (reduce + (map #(* (- % m) (- % m)) xs))
                       (dec (count xs))))}))

(def bearing-spread (spread final-bearing samples))

(def range-spread (spread final-range samples))

[bearing-spread range-spread]
;=> [{:mean 0.5233 :sd 0.0042} {:mean 0.556 :sd 0.098}] ; float64

;; book: across-the-ray
(let [across (* (:mean range-spread) (:sd bearing-spread))]
  [(< (:sd bearing-spread) (* 2 measurement-noise))
   (> (:sd range-spread) (* 10 across))])
;=> [true true]

(< (abs (- (:mean range-spread) (final-range truth)))
   (* 5 (:sd range-spread)))
;=> true

;; book: one-step
(def before (pf/initialize model [0] (observation zs 0) 3 {:key (rng/key 5)}))

(def after
  (pf/step before [1] [diff/unknown-change] (observation zs 1)
           {:key (rng/key 6)}))

(defn bearing-density
  "log p(z | tr): the density of bearing z at trace tr's last position."
  [tr z]
  (dc/log-prob (d/normal (final-bearing tr) measurement-noise) z))

(array/allclose?
 (array/array (mapv - (:log-weights after) (:log-weights before)))
 (array/array (mapv #(bearing-density % (zs 1)) (pf/traces after))))
;=> true

(defn starts
  "The starting x of each particle of a state."
  [state]
  (mapv #(array/item (% :x0)) (pf/traces state)))

(= (starts before) (starts after))
;=> true

;; book: step-error
;; error: A particle filter step changed or removed choices
#_(pf/step after [1] [diff/no-change] (cm/choicemap :z0 1.5)
         {:key (rng/key 7)})

;; book: no-resampling
(defn no-resampling
  "The state as it is."
  [state _ _]
  state)

(def unresampled
  (particle-filter model zs 5000 no-resampling {:batched? true} (rng/key 3)))

(peek (:ess unresampled))
;=> 1.0 ; float64

(< (peek (:ess unresampled)) 10)
;=> true

;; book: collapse
(def unresampled-samples
  (pf/sample-unweighted-traces (:state unresampled) 100 (rng/key 4)))

(count (distinct (map final-range unresampled-samples)))
;=> 1 ; float64

(< (count (distinct (map final-range unresampled-samples))) 5)
;=> true

;; book: collapse-figure
;; figure: figures/025-no-resampling.svg
(render zs unresampled-samples)

;; book: ess-figure
;; figure: figures/025-ess.svg
(defn log10-path
  "The base-10 logarithm of each value of xs, against its index."
  [xs]
  (vec (map-indexed (fn [i x] [i (Math/log10 x)]) xs)))

(svg/paths [{:paths [(log10-path (:ess filtered))] :r 2}
            {:paths [(log10-path (:ess unresampled))] :r 2
             :color "darkorange"}]
           {:x-label "step" :y-label "log10 ESS"})

;; book: values-at
(defn values-at
  "Every particle's choice at addr, as host numbers."
  [state addr]
  (if-let [ps (:particles state)]
    (array/->clj (batched/choice ps addr))
    (mapv #(array/item (% addr)) (pf/traces state))))

(defn distinct-at
  "How many distinct values the particles hold at addr."
  [state addr]
  (count (distinct (values-at state addr))))

;; book: degeneracy
(distinct-at (:state filtered) :x0)
;=> 35 ; float64

(< (distinct-at (:state filtered) :x0) 500)
;=> true

;; book: resimulate
(def initial-choices (sel/select :x0 :y0 :vx0 :vy0))

(defn resimulate-start
  "One MH move on the initial choices of every particle of a batched
   filter, proposed from their prior, then resampling as before."
  [state t key]
  (let [[k-move k-resample] (rng/split key)
        moved (mh/mh-particles (:particles state) initial-choices k-move)]
    (resample (assoc state :particles (:particles moved)) t k-resample)))

(def resimulated
  (particle-filter model zs 5000 resimulate-start {:batched? true}
                   (rng/key 8)))

(distinct-at (:state resimulated) :x0)
;=> 36 ; float64

;; book: resimulate-acceptance
(def final-resimulation
  (mh/mh-particles (:particles (:state resimulated)) initial-choices
                   (rng/key 9)))

(def resimulation-rate
  (/ (count (filter true? (:accepted final-resimulation))) 5000.0))

resimulation-rate
;=> 0.0008 ; float64

(< resimulation-rate 0.01)
;=> true

;; book: perturbation
(def perturbation
  (gen [tr a b]
    (doseq [t (range a (inc b))]
      (trace [:moves t :vx] (d/normal (tr [:moves t :vx]) 1e-3))
      (trace [:moves t :vy] (d/normal (tr [:moves t :vy]) 1e-3)))))

(defn recent
  "The first and last of the last five moves of a trace with n moves."
  [n]
  [(max 0 (- n 5)) (dec n)])

(defn perturb-recent
  "One random-walk MH move on the velocities of every particle's last five
   moves, then resampling as before."
  [state t key]
  (let [[k-move k-resample] (rng/split key)
        n (dec t)
        moved (if (zero? n)
                state
                (pf/rejuvenate state (kernel/mh perturbation (recent n))
                               k-move))]
    (resample moved t k-resample)))

;; book: perturbed
(def plain
  (particle-filter model zs 1000 resample {:batched? true} (rng/key 10)))

(def perturbed
  (particle-filter model zs 1000 perturb-recent {:batched? true}
                   (rng/key 10)))

;; book: perturb-acceptance
(def perturbation-rate
  (let [trs (pf/traces (:state perturbed))
        moved (map #(mh/mh %1 perturbation (recent T) %2)
                   trs (rng/split (rng/key 11) (count trs)))]
    (/ (count (filter :accepted? moved)) (double (count trs)))))

perturbation-rate
;=> 0.176 ; float64

(> perturbation-rate (* 10 resimulation-rate))
;=> true

;; book: diversity
(defn diversity
  "The number of distinct values of each move's :vx among the particles."
  [state]
  (mapv #(distinct-at state [:moves % :vx]) (range T)))

(def plain-diversity (diversity (:state plain)))

(def perturbed-diversity (diversity (:state perturbed)))

[(plain-diversity 44) (perturbed-diversity 44)]
;=> [94 222] ; float64

;; book: diversity-figure
;; figure: figures/025-diversity.svg
(svg/paths [{:paths [(log10-path plain-diversity)] :r 2}
            {:paths [(log10-path perturbed-diversity)] :r 2
             :color "darkorange"}]
           {:x-label "move" :y-label "log10 distinct values of vx"})

;; book: early-moves
[(plain-diversity 0) (perturbed-diversity 0)]
;=> [9 7] ; float64

(< (max (plain-diversity 0) (perturbed-diversity 0)) 50)
;=> true

;; book: loop-model
(def loop-model
  (gen [n]
    (let [x (trace :x0 (d/normal 0.01 0.01))
          y (trace :y0 (d/normal 0.95 0.01))
          vx (trace :vx0 (d/normal 0.002 0.01))
          vy (trace :vy0 (d/normal -0.013 0.01))]
      (trace :z0 (d/normal (bearing x y) measurement-noise))
      (loop [t 0
             states [{:x x :y y :vx vx :vy vy}]]
        (if (< t n)
          (let [{:keys [x y vx vy]} (peek states)
                vx (trace [:moves t :vx] (d/normal vx velocity-sd))
                vy (trace [:moves t :vy] (d/normal vy velocity-sd))
                x (+ x vx)
                y (+ y vy)]
            (trace [:moves t :z] (d/normal (bearing x y) measurement-noise))
            (recur (inc t) (conj states {:x x :y y :vx vx :vy vy})))
          states)))))

;; book: same-model
(array/allclose? (:weight (gfi/assess loop-model [T] (:choices truth)))
                 (:score truth))
;=> true

(def loop-truth
  (:trace (gfi/generate loop-model [T] (:choices truth) (rng/key 12))))

(def next-move
  (cm/choicemap [:moves T :vx] 0.004 [:moves T :vy] -0.012
                [:moves T :z] 0.5))

(defn extend-by-one
  "Trace tr of n moves updated to n + 1, with the choices cs."
  [tr cs]
  (gfi/update tr [(inc (first (:args tr)))] [diff/unknown-change] cs
              (rng/key 13)))

(array/allclose? (:weight (extend-by-one truth next-move))
                 (:weight (extend-by-one loop-truth next-move)))
;=> true

;; book: revisited
(defn revisited
  "The moves whose subtraces differ between the traces a and b of model."
  [a b]
  (vec (remove #(identical? (trace/subtrace a [:moves %])
                            (trace/subtrace b [:moves %]))
               (range (first (:args a))))))

(revisited truth (:trace (extend-by-one truth next-move)))
;=> [0]

;; book: timing
(defn millis
  "How long (f) takes to run, in milliseconds."
  [f]
  (let [start (System/nanoTime)]
    (f)
    (/ (- (System/nanoTime) start) 1e6)))

(defn cost-ratio
  "The median, over reps runs, of the time the loop model takes to extend a
   trace of n moves by one, over the time the Unfold takes."
  [n reps]
  (let [tr (gfi/simulate model [n] (rng/key 14))
        loop-tr (:trace (gfi/generate loop-model [n] (:choices tr)
                                      (rng/key 14)))
        bearing-n (cm/choicemap [:moves n :z] 0.5)
        ratio #(/ (millis (fn [] (extend-by-one loop-tr bearing-n)))
                  (millis (fn [] (extend-by-one tr bearing-n))))]
    (nth (sort (repeatedly reps ratio)) (quot reps 2))))

(def cost-50 (cost-ratio 50 101))

(def cost-200 (cost-ratio 200 101))

[(> cost-200 5) (> cost-200 cost-50)]
;=> [true true]

;; book: solution-1
(defn log-ml
  "The estimate of log p(zs) of a filter of n particles of model."
  [model n key]
  (pf/log-ml-estimate
   (:state (particle-filter model zs n resample {:batched? true} key))))

(def few (mapv #(log-ml model 50 %) (rng/split (rng/key 20) 30)))

(def many (mapv #(log-ml model 800 %) (rng/split (rng/key 21) 10)))

(spread identity many)
;=> {:mean 172.05 :sd 0.65} ; float64

(count (filter #(< % (- (:mean (spread identity many)) 10)) few))
;=> 10 ; float64

;; book: solution-1-check
(let [{:keys [mean sd]} (spread identity many)]
  [(every? #(< (abs (- % mean)) (* 5 sd)) many)
   (pos? (count (filter #(< % (- mean (* 5 sd))) few)))])
;=> [true true]

;; book: solution-2
(def steadiness
  "The estimates of log p(zs) of five filters under each velocity-sd."
  (into {}
        (for [v [1e-4 1e-3 1e-2]]
          [v (mapv #(log-ml (tracker v measurement-noise) 500 %)
                   (rng/split (rng/key 22) 5))])))

(defn mean-and-se
  "The mean of xs and its standard error."
  [xs]
  (let [{:keys [mean sd]} (spread identity xs)]
    {:mean mean :se (/ sd (Math/sqrt (count xs)))}))

(update-vals steadiness #(:mean (mean-and-se %)))
;=> {1.0E-4 -7964.3 0.001 171.98 0.01 159.36} ; float64

;; book: solution-2-check
(defn exceeds?
  "Does the estimate a exceed the estimate b by more than five standard
   errors of their difference?"
  [a b]
  (> (- (:mean a) (:mean b))
     (* 5 (Math/sqrt (+ (Math/pow (:se a) 2) (Math/pow (:se b) 2))))))

[(exceeds? (mean-and-se (steadiness 1e-3)) (mean-and-se (steadiness 1e-2)))
 (< (apply max (steadiness 1e-4)) (apply min (steadiness 1e-3)))]
;=> [true true]

;; book: solution-3
(def walk-start
  (gen [tr]
    (doseq [a [:x0 :y0 :vx0 :vy0]]
      (trace a (d/normal (tr a) 0.001)))))

(defn acceptance
  "The fraction of the traces trs that the MH move (move tr key) moves."
  [move trs key]
  (/ (count (filter :accepted? (map move trs (rng/split key (count trs)))))
     (double (count trs))))

(def final-traces (pf/traces (:state perturbed)))

(def resimulation-acceptance
  (acceptance #(mh/mh %1 initial-choices %2) final-traces (rng/key 23)))

(def walk-acceptance
  (acceptance #(mh/mh %1 walk-start [] %2) final-traces (rng/key 23)))

[resimulation-acceptance walk-acceptance]
;=> [0.003 0.329] ; float64

;; book: solution-3-check
(let [n (count final-traces)
      variance #(/ (* % (- 1 %)) n)]
  (> (- walk-acceptance resimulation-acceptance)
     (* 5 (Math/sqrt (+ (variance walk-acceptance)
                        (variance resimulation-acceptance))))))
;=> true
