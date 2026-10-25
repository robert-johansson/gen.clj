;; The code of book/052-reversible-jump.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.reversible-jump
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.book :as book]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dist.dsl :refer [dist]]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as diag]
            [gen.inference.mh :as mh]
            [gen.inference.transform :as tf]
            [gen.random :as rng]
            [gen.selection :as sel]))


;; book: poisson-plus-one
(dist poisson-plus-one
  "A count of at least 1: a Poisson count plus one."
  [rate]
  (+ (d/poisson rate) 1))

[(dc/log-prob (poisson-plus-one 1) 3) (dc/log-prob (d/poisson 1) 2)]
;=> [-1.6931 -1.6931]

;; book: segment-index
(defn segment-index
  "The segment that holds each of xs, when segments of the given fractions
   divide the range of xs: the number of segment ends left of x, not
   counting the last end, which is the range's."
  [xs fractions]
  (let [xs (array/array xs)
        lo (array/min xs)
        u (/ (- xs lo) (- (array/max xs) lo))
        k (first (array/shape fractions))
        ends (array/slice (array/cumsum fractions) 0 0 (dec k))]
    (array/sum (array/< (array/expand-dims ends 0) (array/expand-dims u 1))
               {:axis 1})))

(segment-index (vec (range 8)) (array/array [0.3 0.5 0.2]))
;=> [0 0 0 1 1 1 2 2]

;; book: model
(defn host-count
  "A count drawn in a body, as the host integer that range and array shapes
   need: on MLX a draw is a 0-d array."
  [k]
  (long (array/item k)))

(def piecewise-constant
  (gen [xs]
    (let [k (host-count (trace :segment-count (poisson-plus-one 1)))
          fractions (trace :fractions (d/dirichlet (array/ones [k])))
          values (mapv #(trace [:segments %] (d/normal 0 1)) (range k))
          noise (trace :noise (d/gamma 1 1))
          means (array/take (array/stack values) (segment-index xs fractions))]
      (trace :ys (d/normal means noise))
      values)))

;; book: prior-traces
(def xs-dense (mapv #(+ -5.0 (* 10.0 (/ % 49.0))) (range 50)))

(def prior-traces
  (mapv #(gfi/simulate piecewise-constant [xs-dense] %)
        (rng/split (rng/key 1) 4)))

(mapv #(% :segment-count) prior-traces)
;=> [1 2 1 2] ; float64

;; book: segment-layers
(defn segment-value
  "The value of trace tr's piecewise-constant function at x."
  [tr x]
  (let [[xs] (:args tr)]
    (array/item (array/take (array/stack (:retval tr))
                            (array/take (segment-index (conj xs x)
                                                       (tr :fractions))
                                        (count xs))))))

(def colors ["steelblue" "darkorange" "seagreen" "firebrick"])

(defn segment-layers
  "Figure layers: each trace's observations as dots, and its function as a
   curve, in a color of its own."
  [traces]
  (mapcat (fn [tr color]
            [{:points (map vector xs-dense (array/->clj (tr :ys)))
              :color color :r 2 :opacity 0.6}
             {:curves [#(segment-value tr %)] :color color :width 2.5
              :opacity 0.9}])
          traces colors))

;; book: prior-figure
;; figure: figures/052-prior.svg
(svg/plot (segment-layers prior-traces) {:x-range [-5 5]})


;; book: split-merge-proposal
(defn segment-values
  "The values of trace tr's segments, in order."
  [tr]
  (mapv #(tr [:segments %]) (range (first (array/shape (tr :fractions))))))

(def split-merge-proposal
  (gen [tr share]
    (let [values (segment-values tr)
          k (count values)]
      (if (trace :split? (d/bernoulli (if (= k 1) 1.0 0.3)))
        (let [v (nth values (trace :index (d/uniform-discrete 0 (dec k))))]
          (trace :split-percentage share)
          (trace :new-value-1 (d/normal v 0.1))
          (trace :new-value-2 (d/normal v 0.1)))
        (let [i (trace :index (d/uniform-discrete 0 (- k 2)))
              mean (/ (+ (nth values i) (nth values (inc i))) 2)]
          (trace :new-value (d/normal mean 0.1)))))))

;; book: fractions
(defn split-fractions
  "fractions with the i-th split in two, its share s and the rest."
  [fractions i s]
  (let [f (array/take fractions i)]
    (array/concatenate [(array/slice fractions 0 0 i)
                        (array/stack [(* f s) (* f (- 1 s))])
                        (array/slice fractions 0 (inc i) nil)])))

(defn merge-fractions
  "[fractions' share]: fractions with the i-th and the next merged, and the
   i-th's share of the two."
  [fractions i]
  (let [a (array/take fractions i)
        b (array/take fractions (inc i))]
    [(array/concatenate [(array/slice fractions 0 0 i)
                         (array/stack [(+ a b)])
                         (array/slice fractions 0 (+ i 2) nil)])
     (/ a (+ a b))]))

(split-fractions (array/array [0.3 0.5 0.2]) 1 0.4)
;=> [0.3 0.2 0.3 0.2]

(merge-fractions (array/array [0.3 0.2 0.3 0.2]) 1)
;=> [[0.3 0.5 0.2] 0.4]

;; book: split-merge
(tf/deftransform split-merge
  "Segment i split in two, or segments i and i + 1 merged into one: the
   segments after them move up or down an index."
  [model-in aux-in model-out aux-out]
  (let [k (tf/read model-in :segment-count :discrete)
        split? (tf/read aux-in :split? :discrete)
        i (tf/read aux-in :index :discrete)
        fractions (tf/read model-in :fractions :continuous)
        value (fn [j] (tf/read model-in [:segments j] :continuous))]
    (tf/write aux-out :split? (not split?) :discrete)
    (tf/write aux-out :index i :discrete)
    (if split?
      (let [s (tf/read aux-in :split-percentage :continuous)]
        (tf/write model-out :segment-count (inc k) :discrete)
        (tf/write model-out :fractions (split-fractions fractions i s)
                  :continuous)
        (tf/write aux-out :new-value (value i) :continuous)
        (tf/write model-out [:segments i]
                  (tf/read aux-in :new-value-1 :continuous) :continuous)
        (tf/write model-out [:segments (inc i)]
                  (tf/read aux-in :new-value-2 :continuous) :continuous)
        (doseq [j (range (inc i) k)]
          (tf/copy model-in [:segments j] model-out [:segments (inc j)])))
      (let [[merged share] (merge-fractions fractions i)]
        (tf/write model-out :segment-count (dec k) :discrete)
        (tf/write model-out :fractions merged :continuous)
        (tf/write aux-out :split-percentage share :continuous)
        (tf/write aux-out :new-value-1 (value i) :continuous)
        (tf/write aux-out :new-value-2 (value (inc i)) :continuous)
        (tf/write model-out [:segments i]
                  (tf/read aux-in :new-value :continuous) :continuous)
        (doseq [j (range (+ i 2) k)]
          (tf/copy model-in [:segments j] model-out [:segments (dec j)]))))))

;; book: three-segments
(def xs-small [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0])

(def ys-small [0.0 0.3 -0.2 1.1 0.9 1.2 0.8 1.0])

(defn observations
  "The choice map fixing the observations to ys."
  [ys]
  (cm/choicemap :ys (array/array ys)))

(defn segments-trace
  "The trace of piecewise-constant given ys-small with these latent
   choices, address and value in turn."
  [& choices]
  (:trace (gfi/generate piecewise-constant [xs-small]
                        (cm/merge (observations ys-small)
                                  (apply cm/choicemap choices))
                        (rng/key 0))))

(def three
  (segments-trace :segment-count 3 :fractions (array/array [0.3 0.5 0.2])
                  [:segments 0] 0.0 [:segments 1] 1.0 [:segments 2] 0.9
                  :noise 0.2))

(def split-choices
  (cm/choicemap :split? true :index 1 :split-percentage 0.4
                :new-value-1 0.95 :new-value-2 1.05))

(def split (tf/run split-merge three split-choices))

(def split-model (cm/to-map (:model split)))

[(:segment-count split-model) (:fractions split-model)]
;=> [4 [0.3 0.2 0.3 0.2]]

(:segments split-model)
;=> {1 0.95 2 1.05 3 0.9}

(cm/to-map (:aux split))
;=> {:split? false :index 1 :new-value 1.0}

;; book: split-jacobian
[(:log-abs-det split) (Math/log 0.5)]
;=> [-0.6931 -0.6931]

;; book: split-twice
(def split-merge-move (tf/involution split-merge {:check? true}))

(def four (:trace (split-merge-move three split-choices (rng/key 0))))

(let [merged (tf/run split-merge four (:aux split))]
  [(tf/same-choices? (:aux merged) split-choices)
   (array/allclose? (+ (:log-abs-det split) (:log-abs-det merged)) 0.0)])
;=> [true true]

;; book: notebook-split-merge
(tf/deftransform notebook-split-merge
  "split-merge as the notebook writes it: a split puts the second new value
   first."
  [model-in aux-in model-out aux-out]
  (let [k (tf/read model-in :segment-count :discrete)
        split? (tf/read aux-in :split? :discrete)
        i (tf/read aux-in :index :discrete)
        fractions (tf/read model-in :fractions :continuous)
        value (fn [j] (tf/read model-in [:segments j] :continuous))]
    (tf/write aux-out :split? (not split?) :discrete)
    (tf/write aux-out :index i :discrete)
    (if split?
      (let [s (tf/read aux-in :split-percentage :continuous)]
        (tf/write model-out :segment-count (inc k) :discrete)
        (tf/write model-out :fractions (split-fractions fractions i s)
                  :continuous)
        (tf/write aux-out :new-value (value i) :continuous)
        (tf/write model-out [:segments i]
                  (tf/read aux-in :new-value-2 :continuous) :continuous)
        (tf/write model-out [:segments (inc i)]
                  (tf/read aux-in :new-value-1 :continuous) :continuous)
        (doseq [j (range (inc i) k)]
          (tf/copy model-in [:segments j] model-out [:segments (inc j)])))
      (let [[merged share] (merge-fractions fractions i)]
        (tf/write model-out :segment-count (dec k) :discrete)
        (tf/write model-out :fractions merged :continuous)
        (tf/write aux-out :split-percentage share :continuous)
        (tf/write aux-out :new-value-1 (value i) :continuous)
        (tf/write aux-out :new-value-2 (value (inc i)) :continuous)
        (tf/write model-out [:segments i]
                  (tf/read aux-in :new-value :continuous) :continuous)
        (doseq [j (range (+ i 2) k)]
          (tf/copy model-in [:segments j] model-out [:segments (dec j)]))))))

;; book: notebook-check
;; error: The involution applied twice does not give back the forward choices
#_(def notebook-move (tf/involution notebook-split-merge {:check? true}))

#_(notebook-move three split-choices (rng/key 0))


;; book: subsets
(defn subsets
  "Every subset of xs, each a vector."
  [xs]
  (reduce (fn [acc x] (into acc (map #(conj % x) acc))) [[]] xs))

(defn occupied-probability
  "The probability that m points, each uniform on [0, 1], fall in exactly
   the gaps S (at least one in each) of gaps of the given lengths: by
   inclusion and exclusion over the subsets T of S, each holding all m
   points with probability (Σ_T lengths)^m."
  [lengths S m]
  (reduce + (for [T (subsets S)]
              (* (if (even? (- (count S) (count T))) 1.0 -1.0)
                 (Math/pow (reduce + 0.0 (map lengths T)) m)))))

(def gap-lengths
  "The lengths of the gaps between consecutive xs-small, on [0, 1]."
  (vec (repeat 7 (/ 1.0 7))))

(map (fn [m] (reduce + (map #(occupied-probability gap-lengths % m)
                            (subsets (range 7)))))
     [0 1 2 5])
;=> [1.0 1.0 1.0 1.0]

;; book: blocks
(defn blocks
  "The indices 0 … n-1 of the data in runs that the occupied gaps S (gap j
   lies between points j and j + 1) separate: the data of each segment."
  [n S]
  (let [cut? (set S)]
    (reduce (fn [runs j]
              (if (cut? (dec j))
                (conj runs [j])
                (update runs (dec (count runs)) conj j)))
            [[0]] (range 1 n))))

(blocks 8 [2 5])
;=> [[0 1 2] [3 4 5] [6 7]]

;; book: log-block
(defn log-block
  "The log marginal density of the ys of a block, one segment's data: its
   value v ~ N(0, 1) integrated out of Π_i N(y_i; v, sigma). Their
   covariance σ²I + 11ᵀ has determinant σ^2(m-1) (σ² + m)."
  [ys block sigma]
  (let [m (count block)
        vs (map ys block)
        mean (/ (reduce + vs) m)
        ss (reduce + (map #(* (- % mean) (- % mean)) vs))
        s2 (* sigma sigma)]
    (- (* -0.5 m (Math/log (* 2 Math/PI)))
       (* (dec m) (Math/log sigma))
       (* 0.5 (Math/log (+ s2 m)))
       (/ ss (* 2 s2))
       (/ (* m mean mean) (* 2 (+ s2 m))))))

(let [block [3 4 5 6 7]
      sigma 0.3
      cov (+ (* sigma sigma (array/eye 5)) (array/ones [5 5]))]
  (array/allclose? (log-block ys-small block sigma)
                   (dc/log-prob (d/mvnormal (array/zeros [5]) cov)
                                (array/array (mapv ys-small block)))))
;=> true

;; book: exact-posterior
(defn log-sum-exp
  "log Σ exp(xs), stable."
  [xs]
  (let [top (apply max xs)]
    (+ top (Math/log (reduce + (map #(Math/exp (- % top)) xs))))))

(defn log-evidence
  "log p(ys | the segments' data are the blocks bs): each block's marginal
   density, the noise integrated out against its Gamma(1, 1) prior, whose
   density is exp(-σ), by the midpoint rule with n intervals on (0, 4]."
  [ys bs n]
  (let [h (/ 4.0 n)]
    (+ (Math/log h)
       (log-sum-exp (for [i (range n)
                          :let [sigma (* h (+ i 0.5))]]
                      (- (reduce + (map #(log-block ys % sigma) bs))
                         sigma))))))

(defn segment-count-evidence
  "log p(ys | k) for k = 1 … k-max: Σ_S P(S | k) p(ys | S) over the sets
   S of gaps that hold the k - 1 segment ends."
  [xs ys k-max]
  (let [n (count xs)
        span (- (apply max xs) (apply min xs))
        lengths (mapv #(/ (- %2 %1) span) xs (rest xs))
        gap-sets (subsets (range (dec n)))
        evidence (mapv #(log-evidence ys (blocks n %) 2000) gap-sets)]
    (vec (for [k (range 1 (inc k-max))]
           (log-sum-exp
            (for [[S e] (map vector gap-sets evidence)
                  :let [p (occupied-probability lengths S (dec k))]
                  :when (pos? p)]
              (+ (Math/log p) e)))))))

(def evidence (segment-count-evidence xs-small ys-small 12))

(def exact
  "P(k | ys) for k = 1 … 12: the evidence times k's prior, Poisson(1) at
   k - 1, normalized."
  (let [log-prior (fn [k] (- -1.0 (reduce + 0.0 (map #(Math/log %)
                                                     (range 1 k)))))
        joint (map-indexed #(+ (log-prior (inc %1)) %2) evidence)
        total (log-sum-exp joint)]
    (mapv #(Math/exp (- % total)) joint)))

(subvec exact 0 5)
;=> [0.0387 0.6944 0.2148 0.0440 0.0070]

;; book: monte-carlo
(defn mean-se
  "The mean of numbers xs and its standard error."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        variance (/ (reduce + (map #(* (- % mean) (- % mean)) xs)) (dec n))]
    {:mean mean :se (Math/sqrt (/ variance n))}))

(defn evidence-estimate
  "A Monte Carlo estimate of p(ys | k) and its standard error, from n
   draws of the fractions and the noise from their priors, the segments'
   values integrated out."
  [k n key]
  (let [[k1 k2] (rng/split key)
        fractions (dc/sample (d/dirichlet (array/ones [k])) k1 [n])
        sigmas (array/->clj (dc/sample (d/gamma 1 1) k2 [n]))
        likelihood
        (fn [i sigma]
          (let [segment (array/->clj (segment-index
                                      xs-small (array/take fractions i)))
                runs (vals (group-by segment (range (count xs-small))))]
            (Math/exp (reduce + (map #(log-block ys-small (vec %) sigma)
                                     runs)))))]
    (mean-se (mapv likelihood (range n) sigmas))))

(let [{:keys [mean se]} (evidence-estimate 3 20000 (rng/key 12))]
  (< (abs (- mean (Math/exp (evidence 2)))) (* 5 se)))
;=> true


;; book: segment-proposal
(def segment-proposal
  (gen [tr xs ys i]
    (let [in? (array/== (segment-index xs (tr :fractions)) i)
          n (array/sum in?)
          mean (/ (array/sum (array/where in? ys 0.0)) (array/maximum n 1))]
      (if (array/> n 0)
        (trace [:segments i] (d/normal mean 0.3))
        (trace [:segments i] (d/normal 0 1))))))

;; book: sweep
(defn rj-sweep
  "A sweep: the split/merge move (the involution move, splits drawing their
   share from share), a data-driven move of each segment's value, and moves
   of the fractions and the noise from their priors."
  [move share xs ys]
  (let [ys (array/array ys)]
    (fn [tr key]
      (let [[k1 k2 k3 k4] (rng/split key 4)
            tr (:trace (mh/involutive-mh tr split-merge-proposal [share] move
                                         k1))
            k (first (array/shape (tr :fractions)))
            tr (reduce (fn [tr [i k]]
                         (:trace (mh/mh tr segment-proposal [xs ys i] k)))
                       tr (map vector (range k) (rng/split k2 k)))
            tr (:trace (mh/mh tr (sel/select :fractions) k3))]
        (:trace (mh/mh tr (sel/select :noise) k4))))))

(defn segment-counts
  "The number of segments after each of n sweeps from one segment."
  [sweep n key]
  (let [[k-start k-chain] (rng/split key)
        start (:trace (gfi/generate piecewise-constant [xs-small]
                                    (cm/merge (observations ys-small)
                                              (cm/choicemap :segment-count 1))
                                    k-start))]
    (mapv #(first (array/shape (% :fractions)))
          (rest (reductions sweep start (rng/split k-chain n))))))

(def uniform-share "The notebook's share of a split." (d/uniform 0 1))

(def counts
  (segment-counts (rj-sweep (tf/involution split-merge) uniform-share
                            xs-small ys-small)
                  4000 (rng/key 2)))

;; book: chain-check
(defn indicator-summary
  "The summary of the indicator of k segments along a run of counts."
  [counts k]
  (diag/summary (mapv #(if (= % k) 1.0 0.0) counts)))

(defn agrees?
  "Does each P(k | ys), k = 1 … 4, estimated from counts, agree with the
   exact one within five Monte Carlo standard errors?"
  [counts]
  (every? (fn [k]
            (let [{:keys [mean mcse-mean]} (indicator-summary counts k)]
              (< (abs (- mean (exact (dec k)))) (* 5 mcse-mean))))
          [1 2 3 4]))

(agrees? counts)
;=> true

;; book: without-jacobian
(defn without-jacobian
  "The involution of transform t with log |det J| taken out of its weight:
   the notebook's weight, the update's alone."
  [t]
  (let [move (tf/involution t)]
    (fn [tr u retval args key]
      (let [{:keys [log-abs-det] :as moved} (move tr u retval args key)]
        ;; a move outside the support is not weighed, and has no Jacobian
        (cond-> moved log-abs-det (update :weight - log-abs-det))))))

(def notebook-counts
  (segment-counts (rj-sweep (without-jacobian split-merge) uniform-share
                            xs-small ys-small)
                  1500 (rng/key 2)))

(let [{:keys [mean mcse-mean]} (indicator-summary notebook-counts 2)]
  [(agrees? notebook-counts)
   (> (- (exact 1) mean) (* 5 mcse-mean))])
;=> [false true]

;; book: counts-figure
;; figure: figures/052-segment-counts.svg
(svg/histogram [{:values counts :label "split/merge"
                 :density #(get exact (dec (Math/round (double %))) 0.0)}
                {:values notebook-counts :label "no Jacobian"}]
               {:range [0.5 24.5] :bins 24 :x-label "segments"
                :y-label "probability"})


;; book: covariance
(defn primitive?
  "Is kernel a primitive, not a :plus or a :times of two kernels?"
  [kernel]
  (not (#{:plus :times} (first kernel))))

(defn covariance
  "k(x1, x2) of a kernel program: its value at host numbers, and, by
   broadcasting, the covariance matrix at a column and a row of points."
  [[kind a b] x1 x2]
  (case kind
    :constant (array/full-like (- x1 x2) a)
    :linear (* (- x1 a) (- x2 a))
    :squared-exponential (array/exp (/ (* -0.5 (array/square (- x1 x2))) a))
    :periodic (array/exp (/ (- (array/square
                                (array/sin (* (/ (* 2 Math/PI) b)
                                              (array/abs (- x1 x2))))))
                            a))
    :plus (+ (covariance a x1 x2) (covariance b x1 x2))
    :times (* (covariance a x1 x2) (covariance b x1 x2))))

(defn covariance-matrix
  "The covariance matrix of kernel at xs: covariance at a column and a row."
  [kernel xs]
  (let [x (array/array xs)]
    (covariance kernel (array/expand-dims x 1) (array/expand-dims x 0))))

;; book: hand-values
(mapv #(covariance % 0.1 0.6)
      [[:constant 0.3] [:linear 0.2] [:squared-exponential 0.5]
       [:periodic 0.5 0.4]])
;=> [0.3 -0.04 0.7788 0.1353]

(let [k [:times [:linear 0.2] [:plus [:constant 0.3] [:periodic 0.5 0.4]]]]
  (array/allclose? (covariance k 0.1 0.6) (* -0.04 (+ 0.3 0.1353353))))
;=> true

;; book: half-period
(mapv #(covariance [:periodic 0.5 0.4] 0.0 %) [0.1 0.2 0.4])
;=> [0.1353 1.0 1.0]

;; book: matrix
(let [xs [0.0 0.25 0.5 0.75]
      k [:plus [:linear 0.2] [:periodic 0.5 0.4]]
      m (covariance-matrix k xs)]
  [(array/shape m)
   (array/allclose? (array/take (array/take m 1) 3)
                    (covariance k 0.25 0.75))])
;=> [[4 4] true]


;; book: grammar
(def kernel-kinds
  [:constant :linear :squared-exponential :periodic :plus :times])

(dist kernel-kind
  "A kernel node's kind: each primitive with probability 0.2, :plus and
   :times with 0.1."
  []
  (nth kernel-kinds (d/categorical [0.2 0.2 0.2 0.2 0.1 0.1])))

(def kernel-grammar
  (gen [period]
    (let [kind (trace :kernel-type (kernel-kind))]
      (case kind
        (:plus :times) [kind (splice :left kernel-grammar period)
                        (splice :right kernel-grammar period)]
        :periodic [kind (trace :scale (d/uniform 0 1)) (trace :period period)]
        [kind (trace :param (d/uniform 0 1))]))))

(def uniform-period "The prior of a period." (d/uniform 0 1))

;; book: prior-programs
(def prior-programs
  (mapv #(:retval (gfi/simulate kernel-grammar [uniform-period] %))
        (rng/split (rng/key 3) 4000)))

(mapv first (take 5 prior-programs))
;=> [:periodic :plus :plus :squared-exponential :constant] ; float64

;; book: prior-sizes
(defn size
  "The number of nodes of a kernel program."
  [kernel]
  (if (primitive? kernel)
    1
    (+ 1 (size (second kernel)) (size (nth kernel 2)))))

(defn within-5-se?
  "Is the mean of xs within five standard errors of expected?"
  [xs expected]
  (let [{:keys [mean se]} (mean-se xs)]
    (< (abs (- mean expected)) (* 5 se))))

[(within-5-se? (mapv #(if (primitive? %) 0.0 1.0) prior-programs) 0.2)
 (within-5-se? (mapv size prior-programs) (/ 5.0 3))]
;=> [true true]

;; book: gp-model
(dist gamma-bounded-below
  "A Gamma(shape, scale) draw plus bound."
  [shape scale bound]
  (+ (d/gamma shape scale) bound))

(def gp-model
  (gen [xs]
    (let [kernel (splice :tree kernel-grammar uniform-period)
          noise (trace :noise (gamma-bounded-below 1 1 0.01))
          n (count xs)
          cov (+ (covariance-matrix kernel xs) (* noise (array/eye n)))]
      (trace :ys (d/mvnormal (array/zeros [n]) cov))
      kernel)))

;; book: float32
(def unit-xs (mapv #(/ % 71.0) (range 72)))

(defn float32-mvnormal
  "mvnormal with mean 0 and covariance cov, factored in float32."
  [cov]
  (d/mvnormal (array/zeros [(first (array/shape cov))])
              (array/astype cov :float32)))

(def rank-one (covariance-matrix [:linear 0.5] unit-xs))

;; book: float32-error
;; error: mvnormal: cov must be positive definite, and not singular at float32
#_(float32-mvnormal (+ rank-one (* 1e-8 (array/eye 72))))

;; book: smooth-error
;; error: mvnormal: cov must be positive definite
#_(d/mvnormal (array/zeros [72])
            (covariance-matrix [:squared-exponential 0.5] unit-xs))

;; book: float32-jitter
(def suggested
  (try (float32-mvnormal (+ rank-one (* 1e-7 (array/eye 72))))
       (catch clojure.lang.ExceptionInfo e (:suggested-jitter (ex-data e)))))

suggested
;=> 1.0E-6 ; float64

(some? (float32-mvnormal (+ rank-one (* 0.01 (array/eye 72)))))
;=> true

;; book: float32-prior
(every? #(some? (float32-mvnormal (+ (covariance-matrix % unit-xs)
                                     (* 0.01 (array/eye 72)))))
        (take 500 prior-programs))
;=> true


;; book: random-node-path
(def random-node-path
  (gen [kernel]
    (if (trace :stop (d/bernoulli (if (primitive? kernel) 1.0 0.5)))
      []
      (let [left? (trace :left (d/bernoulli 0.5))
            [_ l r] kernel]
        (into [(if left? :left :right)]
              (splice :rest-of-path random-node-path (if left? l r)))))))

(def example-kernel
  [:plus [:linear 0.72] [:times [:constant 0.3] [:periodic 0.5 0.4]]])

(mapv #(:retval (gfi/simulate random-node-path [example-kernel] %))
      (rng/split (rng/key 8) 6))
;=> [[] [:right :left] [:right] [:left] [:right :left] [:right]] ; float64

;; book: regenerate-subtree
(def regenerate-subtree
  (gen [tr period]
    (let [path (splice :path random-node-path (:retval tr))]
      (splice :new-subtree kernel-grammar period)
      path)))

;; book: subtree-swap
(tf/deftransform subtree-swap
  "The subtree at the picked node exchanged with the new subtree. The path
   is kept, so that the move back picks the same node."
  [model-in aux-in model-out aux-out]
  (let [path (into [:tree] (tf/retval aux-in))]
    (tf/copy aux-in :path aux-out :path)
    (tf/copy aux-in :new-subtree model-out path)
    (tf/copy model-in path aux-out :new-subtree)))

;; book: subtree-example
(def kernel-only (gen [] (splice :tree kernel-grammar uniform-period)))

(def example-trace
  (:trace (gfi/generate kernel-only []
                        (cm/choicemap [:tree :kernel-type] :plus
                                      [:tree :left :kernel-type] :linear
                                      [:tree :left :param] 0.72
                                      [:tree :right :kernel-type] :constant
                                      [:tree :right :param] 0.3)
                        (rng/key 0))))

(def regrow
  (cm/choicemap [:path :stop] false [:path :left] false
                [:path :rest-of-path :stop] true
                [:new-subtree :kernel-type] :periodic
                [:new-subtree :scale] 0.5 [:new-subtree :period] 0.4))

(def regrow-path
  "The path the proposal returns for regrow's choices."
  (:retval (gfi/assess regenerate-subtree [example-trace uniform-period]
                       regrow (rng/key 0))))

regrow-path
;=> [:right]

(def swapped
  ((tf/involution subtree-swap {:check? true :proposal regenerate-subtree})
   example-trace regrow regrow-path [uniform-period] (rng/key 0)))

(:retval (:trace swapped))
;=> [:plus [:linear 0.72] [:periodic 0.5 0.4]]

(cm/to-map (cm/get-submap (:bwd-choices swapped) :new-subtree))
;=> {:kernel-type :constant :param 0.3}

;; book: prior-invariance
(def subtree-move (tf/involution subtree-swap))

(defn subtree-chain
  "The kernel programs along n subtree-regeneration moves on traces of
   kernel-only, new periods drawn from period."
  [period n key]
  (let [[k-start k-chain] (rng/split key)]
    (mapv :retval
          (reductions (fn [tr k]
                        (:trace (mh/involutive-mh tr regenerate-subtree
                                                  [period] subtree-move k)))
                      (gfi/simulate kernel-only [] k-start)
                      (rng/split k-chain n)))))

(defn within-5-mcse?
  "Is a chain's average of the numbers xs within five Monte Carlo standard
   errors of expected?"
  [xs expected]
  (let [{:keys [mean mcse-mean]} (diag/summary (mapv double xs))]
    (< (abs (- mean expected)) (* 5 mcse-mean))))

(def prior-chain (subtree-chain uniform-period 4000 (rng/key 5)))

[(within-5-mcse? (mapv #(if (primitive? %) 0 1) prior-chain) 0.2)
 (within-5-mcse? (mapv size prior-chain) (/ 5.0 3))]
;=> [true true]


;; book: airline
(def airline
  "The AirPassengers series: international airline passengers each month,
   in thousands, January 1949 to December 1960."
  (edn/read-string (slurp (io/file book/dir "data" "airpassengers.edn"))))

(def passengers (:passengers airline))

[(:start airline) (count passengers) (first passengers) (apply max passengers)]
;=> ["1949-01" 144 112 622]

(/ (reduce + passengers) (count passengers))
;=> 280.2986

;; book: windows
(def train-months "1954 to 1959, the months the model sees." (range 60 132))

(def test-months "1960, held out." (range 132 144))

(defn month-x
  "Month m (0 is January 1949) as the notebook scales time: 0 to 1."
  [m]
  (/ m 143.0))

(def log-train (mapv #(Math/log (passengers %)) train-months))

(def center (/ (reduce + log-train) (count log-train)))

(def stretch (/ 4.0 (- (apply max log-train) (apply min log-train))))

(defn month-y
  "The log count of month m, centered and stretched as the training months
   need to span 4."
  [m]
  (* stretch (- (Math/log (passengers m)) center)))

(defn y-passengers
  "The count, in thousands, that y stands for."
  [y]
  (Math/exp (+ center (/ y stretch))))

(def train-xs (mapv month-x train-months))
(def train-ys (mapv month-y train-months))
(def test-xs (mapv month-x test-months))
(def test-ys (mapv month-y test-months))

[(- (apply max train-ys) (apply min train-ys)) (y-passengers (first test-ys))]
;=> [4.0 417.0]

;; book: autocorrelation
(defn autocorrelation-peak
  "The lag, from 2 to 30 months, at which the month-to-month changes of ys
   correlate most with themselves."
  [ys]
  (let [changes (mapv - (rest ys) ys)
        m (/ (reduce + changes) (count changes))
        at-lag (fn [lag] (reduce + (map #(* (- %1 m) (- %2 m))
                                        changes (drop lag changes))))]
    (apply max-key at-lag (range 2 31))))

(autocorrelation-peak train-ys)
;=> 12

;; book: year-period
(def year-period
  "The period of a kernel that repeats every 12 months: sin² repeats at
   half the period."
  (/ (* 2 12) 143.0))

[year-period (covariance [:periodic 0.5 year-period] (month-x 0) (month-x 12))]
;=> [0.1678 1.0]

(* 0.02 year-period)
;=> 0.0034

(defn periodic-leaves
  "The number of kernel's periodic leaves."
  [kernel]
  (count (filter #(= :periodic (first %))
                 (tree-seq (complement primitive?) rest kernel))))

[(within-5-se? (mapv periodic-leaves prior-programs) (/ 1.0 3))
 (* (/ 1.0 3) 0.02 year-period)]
;=> [true 0.0011]

;; book: period-proposal
(defn period-proposal
  "The period of new periodic nodes: half the time uniform on (0, 1), as the
   prior, and half the time within 1% of the period that repeats every lag
   months."
  [lag]
  (let [p (/ (* 2 lag) 143.0)
        a (* 0.99 p)
        b (* 1.01 p)]
    (d/piecewise-uniform (array/array [0.0 a b 1.0])
                         (array/array [(* 0.5 a) (+ 0.5 (* 0.5 (- b a)))
                                       (* 0.5 (- 1 b))]))))

(def data-period (period-proposal (autocorrelation-peak train-ys)))


;; book: walks
(def noise-walk
  (gen [tr]
    (trace :noise (d/truncated-normal (tr :noise) (* 0.2 (tr :noise))
                                      0.01 ##Inf))))

(defn leaf-paths
  "The model addresses of kernel's leaves, the kernel at address path."
  [kernel path]
  (if (primitive? kernel)
    [path]
    (into (leaf-paths (second kernel) (conj path :left))
          (leaf-paths (nth kernel 2) (conj path :right)))))

(def leaf-walk
  (gen [tr path]
    (let [at #(conj path %)
          walk (fn [a step] (trace (at a) (d/truncated-normal (tr (at a))
                                                              step 0 1)))]
      (if (= :periodic (cm/get-choice (:choices tr) (at :kernel-type)))
        (do (walk :scale 0.05) (walk :period 0.002))
        (walk :param 0.05)))))

;; book: synthesis-sweep
(defn synthesis-sweep
  "A sweep: subtree regeneration, its new periods from period; a random walk
   on the noise; and one on the parameters of a leaf picked uniformly."
  [period]
  (fn [tr key]
    (let [[k1 k2 k3 k4] (rng/split key 4)
          tr (:trace (mh/involutive-mh tr regenerate-subtree [period]
                                       subtree-move k1))
          tr (:trace (mh/mh tr noise-walk [] k2))
          leaves (leaf-paths (:retval tr) [:tree])
          leaf (nth leaves (array/item (rng/randint k3 0 (count leaves))))]
      (:trace (mh/mh tr leaf-walk [leaf] k4)))))

;; book: synthesize
(defn synthesize
  "The last trace of n sweeps from a trace of gp-model given the training
   months."
  [sweep n key]
  (let [[k-start k-chain] (rng/split key)
        start (:trace (gfi/generate gp-model [train-xs]
                                    (cm/choicemap :ys (array/array train-ys))
                                    k-start))]
    (reduce sweep start (rng/split k-chain n))))

(def synthesized
  (mapv #(synthesize (synthesis-sweep data-period) 500 %)
        (rng/split (rng/key 6) 8)))

;; book: structure
(defn kinds
  "The kinds of kernel's leaves."
  [kernel]
  (set (map first (filter primitive? (tree-seq (complement primitive?) rest
                                               kernel)))))

(defn periods
  "The periods of kernel's periodic leaves, as host numbers."
  [kernel]
  (for [leaf (tree-seq (complement primitive?) rest kernel)
        :when (= :periodic (first leaf))]
    (array/item (nth leaf 2))))

(defn yearly?
  "Has kernel a periodic leaf whose period is within 1% of a year's?"
  [kernel]
  (boolean (some #(< (abs (- (/ % year-period) 1)) 0.01) (periods kernel))))

(every? #(and (yearly? (:retval %))
              (contains? (kinds (:retval %)) :squared-exponential))
        synthesized)
;=> true

;; book: best
(def best (apply max-key #(array/item (:score %)) synthesized))


;; book: slope
(count (filter #(contains? (kinds (:retval %)) :linear) synthesized))
;=> 2 ; float64

(defn least-squares-slope
  "The slope of the least-squares line through (xs, ys)."
  [xs ys]
  (let [mx (/ (reduce + xs) (count xs))
        my (/ (reduce + ys) (count ys))]
    (/ (reduce + (map #(* (- %1 mx) (- %2 my)) xs ys))
       (reduce + (map #(* (- % mx) (- % mx)) xs)))))

(least-squares-slope train-xs train-ys)
;=> 4.87

;; book: predict
(defn predict
  "The predictive means and standard deviations of y at new-xs, given ys at
   xs, under kernel and noise: the Gaussian process conditioned on the
   data, through the Cholesky factor of their covariance."
  [kernel noise xs ys new-xs]
  (let [column #(array/expand-dims (array/array %) 1)
        row #(array/expand-dims (array/array %) 0)
        l (linalg/cholesky (+ (covariance kernel (column xs) (row xs))
                              (* noise (array/eye (count xs)))))
        a (linalg/solve-triangular l (column ys))
        v (linalg/solve-triangular l (covariance kernel (column xs)
                                                 (row new-xs)))
        x* (array/array new-xs)]
    {:mean (array/->clj (array/squeeze (array/matmul (array/transpose v) a)
                                       1))
     :sd (mapv #(Math/sqrt %)
               (array/->clj (- (+ (covariance kernel x* x*) noise)
                               (array/sum (* v v) {:axis 0}))))}))

(defn forecast
  "The predictive means and standard deviations of trace tr's kernel and
   noise at the held-out months."
  [tr]
  (predict (:retval tr) (tr :noise) train-xs train-ys test-xs))

;; book: errors
(def held-out (mapv passengers test-months))

(defn rmse
  "The root mean square error of counts against the held-out months'."
  [counts]
  (Math/sqrt (/ (reduce + (map #(* (- %1 %2) (- %1 %2)) counts held-out))
                (count held-out))))

(def errors (mapv #(rmse (map y-passengers (:mean (forecast %)))) synthesized))

(def last-year (rmse (map passengers (range 120 132))))

last-year
;=> 50.7

(mean-se errors)
;=> {:mean 24.5 :se 2.6} ; float64

(let [{:keys [mean se]} (mean-se errors)]
  (> (- last-year mean) (* 5 se)))
;=> true

;; book: growth
(def growth
  "1959's passengers over 1958's."
  (/ (reduce + (map passengers (range 120 132)))
     (reduce + (map passengers (range 108 120)))))

(rmse (map #(* growth (passengers %)) (range 120 132)))
;=> 16.8

;; book: coverage
(defn coverage
  "The fraction of the held-out months within two predictive standard
   deviations of trace tr's forecast."
  [tr]
  (let [{:keys [mean sd]} (forecast tr)]
    (/ (count (filter true? (map #(< (abs (- %1 %3)) (* 2 %2))
                                 mean sd test-ys)))
       (double (count test-ys)))))

(mapv coverage synthesized)
;=> [0.9167 1.0 0.9167 1.0 0.9167 1.0 0.9167 0.9167] ; float64

;; book: forecast-figure
;; figure: figures/052-forecast.svg
(defn year-of
  "Month m as a year: 1949 + m/12."
  [m]
  (+ 1949 (/ m 12.0)))

(defn forecast-layers
  "Figure layers: the training months (black) and the held-out ones
   (orange) as dots, each trace's forecast as a line, and the best trace's
   two-standard-deviation band."
  [traces best]
  (let [months-at (fn [ms ys] (map #(vector (year-of %1) %2) ms ys))
        {:keys [mean sd]} (forecast best)
        band (fn [sign] (months-at test-months
                                   (map #(y-passengers (+ %1 (* sign 2 %2)))
                                        mean sd)))]
    (concat
     [{:paths [(months-at train-months (map passengers train-months))]
       :color "black" :width 0 :r 1.5 :opacity 0.8}
      {:paths [(months-at test-months held-out)]
       :color "darkorange" :width 0 :r 2.5 :opacity 1}]
     (for [tr traces]
       {:paths [(months-at test-months (map y-passengers
                                            (:mean (forecast tr))))]
        :color "steelblue" :width 1 :opacity 0.6})
     [{:paths [(band 1) (band -1)] :color "steelblue" :width 1
       :opacity 0.3}])))

(svg/paths (forecast-layers synthesized best)
           {:x-range [1957 1961] :y-range [250 720] :x-label "year"
            :y-label "passengers (thousands)"})

;; book: kernel-figure
;; figure: figures/052-kernel.svg
(defn kernel-label
  "A node's label: a composite's operator, or a primitive's kind and its
   parameters to three digits."
  [[kind a b]]
  (let [digits #(format "%.3f" (array/item %))]
    (case kind
      :plus "+"
      :times "*"
      :constant (str "Constant(" (digits a) ")")
      :linear (str "Linear(" (digits a) ")")
      :squared-exponential (str "SE(" (digits a) ")")
      :periodic (str "Periodic(" (digits a) ", " (digits b) ")"))))

(defn kernel-paths
  "The paths from kernel's root to each of its leaves, each node a symbol
   of its label."
  [kernel]
  (let [node (symbol (kernel-label kernel))]
    (if (primitive? kernel)
      [[node]]
      (mapv #(into [node] %) (concat (kernel-paths (second kernel))
                                     (kernel-paths (nth kernel 2)))))))

(svg/tree (map rest (kernel-paths (:retval best)))
          {:root (kernel-label (:retval best))})


;; book: solution-1
(def beta-counts
  (segment-counts (rj-sweep (tf/involution split-merge) (d/beta 2 2)
                            xs-small ys-small)
                  3000 (rng/key 7)))

(agrees? beta-counts)
;=> true

;; book: solution-2
(def middle-share "Splits in the middle half of a segment only."
  (d/uniform 0.25 0.75))

(def lopsided
  (segments-trace :segment-count 2 :fractions (array/array [0.1 0.9])
                  [:segments 0] 0.0 [:segments 1] 0.9 :noise 0.2))

(let [{merged :trace back :bwd-choices}
      (split-merge-move lopsided
                        (cm/choicemap :split? false :index 0 :new-value 0.5)
                        nil [] (rng/key 0))]
  [(cm/get-choice back :split-percentage)
   (:weight (gfi/assess split-merge-proposal [merged middle-share] back
                        (rng/key 0)))])
;=> [0.1 ##-Inf]

;; book: solution-2-chain
;; error: The backward proposal gives the move's reverse choices density 0
#_(segment-counts (rj-sweep (tf/involution split-merge) middle-share
                          xs-small ys-small)
                3000 (rng/key 8))

;; book: solution-3
(defn periodic-with-pi
  "The periodic kernel's value with π in place of 2π."
  [scale period x1 x2]
  (array/exp (/ (- (array/square (array/sin (* (/ Math/PI period)
                                               (array/abs (- x1 x2))))))
                scale)))

(mapv #(periodic-with-pi 0.5 0.4 0.0 %) [0.1 0.2 0.4])
;=> [0.3679 0.1353 1.0]

(periodic-with-pi 0.5 (/ 12 143.0) (month-x 0) (month-x 12))
;=> 1.0

;; book: solution-4
(dist bushy-kind
  "A kernel node's kind: each primitive with probability 0.15, :plus and
   :times with 0.2."
  []
  (nth kernel-kinds (d/categorical [0.15 0.15 0.15 0.15 0.2 0.2])))

(def bushy-grammar
  (gen []
    (let [kind (trace :kernel-type (bushy-kind))]
      (case kind
        (:plus :times) [kind (splice :left bushy-grammar)
                        (splice :right bushy-grammar)]
        :periodic [kind (trace :scale (d/uniform 0 1))
                   (trace :period (d/uniform 0 1))]
        [kind (trace :param (d/uniform 0 1))]))))

(def bushy-sizes
  (mapv #(size (:retval (gfi/simulate bushy-grammar [] %)))
        (rng/split (rng/key 9) 4000)))

(within-5-se? bushy-sizes 5.0)
;=> true

;; book: solution-5
(def data-driven-chain (subtree-chain data-period 4000 (rng/key 10)))

(defn short-periodic?
  "Is kernel a periodic leaf with a period below 0.5?"
  [kernel]
  (and (= :periodic (first kernel)) (< (array/item (nth kernel 2)) 0.5)))

[(within-5-mcse? (mapv size data-driven-chain) (/ 5.0 3))
 (within-5-mcse? (mapv #(if (short-periodic? %) 1 0) data-driven-chain) 0.1)]
;=> [true true]
