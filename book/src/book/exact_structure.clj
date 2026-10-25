;; The code of book/042-exact-structure.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.exact-structure
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.conjugate :as conj]
            [gen.inference.diagnostics :as diag]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.inference.weights :as weights]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.state-space :as ss]
            [gen.trace :as trace]))

;; book: model
(def ys [1.8 2.2 1.5 2.5 2.0 1.9])

(def noisy-mean
  (gen [n]
    (let [mu (trace :mu (d/normal 0 3))]
      (doseq [i (range n)]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(defn observations
  "The choice map fixing each [:y i] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

(def obs (observations ys))

;; book: evidence
(defn log-evidence
  "log p(ys) for noisy-mean with the noise's standard deviation s (1 by
   default): the ys are jointly normal, with mean 0 and covariance
   9J + s^2 I (J all ones), the prior's variance shared by every pair."
  ([ys] (log-evidence ys 1.0))
  ([ys s]
   (let [n (count ys)]
     (dc/log-prob (d/mvnormal (array/zeros [n])
                              (+ (* 9.0 (array/ones [n n]))
                                 (* s s (array/eye n))))
                  (array/array ys)))))

(log-evidence ys)
;=> -8.02603

;; book: plain-weights
(mapv #(:weight (gfi/generate noisy-mean [6] obs (rng/key %))) [1 2 3])
;=> [-5.889 -79.476 -28.095] ; float64

;; book: conjugate
(def conjugate-mean
  (conj/with-conjugate-proposal noisy-mean :mu (sel/select :y)))

(def runs (mapv #(gfi/generate conjugate-mean [6] obs (rng/key %)) [1 2 3]))

(mapv #(array/allclose? (:weight %) (log-evidence ys)) runs)
;=> [true true true]

(mapv #((:trace %) :mu) runs)
;=> [2.237 2.883 1.847] ; float64

;; book: posterior
(defn posterior
  "mu's exact posterior in noisy-mean given ys: precision 1/9 + n, mean
   Σy over the precision. {:mean :sd}."
  [ys]
  (let [precision (+ (/ 1.0 9.0) (count ys))]
    {:mean (/ (reduce + ys) precision) :sd (Math/sqrt (/ 1.0 precision))}))

(posterior ys)
;=> {:mean 1.9473 :sd 0.4045}

;; book: stats
(defn mean-and-se
  "The mean of the host numbers xs and its standard error."
  [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)
        v (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))]
    [m (Math/sqrt (/ v n))]))

(defn variance-and-se
  "The sample variance of the host numbers xs and its standard error,
   sqrt((m4 - v^2) / n), m4 their fourth central moment."
  [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)
        sq (mapv #(* (- % m) (- % m)) xs)
        v (/ (reduce + sq) (dec n))
        m4 (/ (reduce + (map #(* % %) sq)) n)]
    [v (Math/sqrt (/ (- m4 (* v v)) n))]))

;; book: posterior-draws
(def mus
  (let [{ps :particles} (batched/generate conjugate-mean [6] obs 4000
                                          (rng/key 4))]
    (array/->clj (batched/choice ps :mu))))

(let [{:keys [mean sd]} (posterior ys)
      [m se] (mean-and-se mus)
      [v v-se] (variance-and-se mus)]
  [(< (abs (- m mean)) (* 5 se))
   (< (abs (- v (* sd sd))) (* 5 v-se))])
;=> [true true]

;; book: one-sample
(def one-sample
  (imp/importance-sampling conjugate-mean [6] obs 1 {:key (rng/key 5)}))

(array/allclose? (:log-ml-estimate one-sample) (log-evidence ys))
;=> true

;; book: gibbs-move
(let [tr (:trace (first runs))
      {moved :trace w :weight} (gfi/regenerate tr (sel/select :mu) (rng/key 6))]
  [(array/allclose? w 0.0 {:atol 1e-4})
   (array/allclose? (moved [:y 0]) (tr [:y 0]))
   (array/allclose? (moved :mu) (tr :mu))])
;=> [true true false]

;; book: log-ml-runs
(defn log-ml-runs
  "r independent importance sampling estimates of log p(observations),
   each from n samples: one batched generate of r·n samples, its weights
   split into r runs. Host numbers."
  [model args observations r n key]
  (let [{ws :weights} (batched/generate model args observations (* r n) key)]
    (mapv weights/log-mean-exp (partition n (array/->clj ws)))))

;; book: variance-full
(def plain-runs (log-ml-runs noisy-mean [6] obs 1000 100 (rng/key 7)))

(def conjugate-runs (log-ml-runs conjugate-mean [6] obs 1000 100 (rng/key 7)))

(variance-and-se plain-runs)
;=> [0.0562 0.0026] ; float64

[(let [[v se] (variance-and-se plain-runs)] (> v (* 5 se)))
 (every? #(array/allclose? % (log-evidence ys)) conjugate-runs)]
;=> [true true]

;; book: noisy-scale
(def noisy-scale
  (gen [n]
    (let [s (trace :s (d/gamma 4 0.25))
          mu (trace :mu (d/normal 0 3))]
      (doseq [i (range n)]
        (trace [:y i] (d/normal mu s)))
      mu)))

(def conjugate-scale
  (conj/with-conjugate-proposal noisy-scale :mu (sel/select :y)))

;; book: scale-evidence
(def s-grid
  "Midpoints of 2000 intervals of width 0.0025 covering s in (0, 5]."
  (mapv #(* 0.0025 (+ % 0.5)) (range 2000)))

(def s-log-posterior
  "log p(s) + log p(ys | s) at each point of s-grid: mu summed out by
   log-evidence."
  (mapv #(array/item (+ (dc/log-prob (d/gamma 4 0.25) %) (log-evidence ys %)))
        s-grid))

(def scale-evidence
  (+ (Math/log 0.0025) (weights/log-sum-exp s-log-posterior)))

scale-evidence
;=> -6.5276

;; book: variance-partial
(def plain-scale-runs
  (log-ml-runs noisy-scale [6] obs 200 100 (rng/key 8)))

(def conjugate-scale-runs
  (log-ml-runs conjugate-scale [6] obs 200 100 (rng/key 8)))

(mapv (comp first variance-and-se) [plain-scale-runs conjugate-scale-runs])
;=> [0.558 0.0198] ; float64

(let [[v1 se1] (variance-and-se plain-scale-runs)
      [v2 se2] (variance-and-se conjugate-scale-runs)]
  (> (- v1 v2) (* 5 (Math/sqrt (+ (* se1 se1) (* se2 se2))))))
;=> true

;; book: unbiased
(defn ratio-check
  "Is the mean of exp(estimate - exact) over the runs within 5 standard
   errors of 1, as it is for unbiased estimates of exp(exact)?"
  [runs exact]
  (let [[m se] (mean-and-se (mapv #(Math/exp (- % exact)) runs))]
    (< (abs (- m 1.0)) (* 5 se))))

[(ratio-check plain-scale-runs scale-evidence)
 (ratio-check conjugate-scale-runs scale-evidence)]
;=> [true true]

;; book: variance-figure
;; figure: figures/042-variance.svg
(svg/histogram [{:values plain-scale-runs :label "plain"}
                {:values conjugate-scale-runs :label "conjugate"}]
               {:bins 50 :x-label "estimate of log p(y), 100 samples each"})

;; book: coin
(def coin
  (gen [n]
    (let [p (trace :p (d/beta 2 2))]
      (doseq [i (range n)]
        (trace [:x i] (d/bernoulli p)))
      p)))

(def flips (cm/choicemap [:x 0] true [:x 1] true [:x 2] false))

(defn urn-log-evidence
  "log p(cs) of categories cs drawn from a dirichlet(alpha) probability
   vector summed out: each draw has probability (its category's alpha +
   its count so far) / (Σ alpha + draws so far), the Pólya urn."
  [alpha cs]
  (loop [counts (vec alpha) cs cs total 0.0]
    (if-let [[c & more] (seq cs)]
      (recur (update counts c + 1)
             more
             (+ total (Math/log (/ (counts c) (reduce + counts)))))
      total)))

(urn-log-evidence [2 2] [0 0 1])
;=> -2.302585

(def conjugate-coin (conj/with-conjugate-proposal coin :p (sel/select :x)))

(array/allclose? (:weight (gfi/generate conjugate-coin [3] flips (rng/key 9)))
                 (urn-log-evidence [2 2] [0 0 1]))
;=> true

;; book: counts
(def counts
  (gen [n]
    (let [rate (trace :rate (d/gamma 3 1))]
      (doseq [i (range n)]
        (trace [:k i] (d/poisson rate)))
      rate)))

(defn lgamma
  "log Γ(x) of a host number."
  [x]
  (array/item (array/lgamma (double x))))

(defn negative-binomial-log-evidence
  "log p(ks) of poisson counts ks with a gamma(shape, scale) rate summed
   out: each count is negative binomial given the ones before, and adds
   itself to the shape and 1 to the rate 1/scale."
  [shape scale ks]
  (loop [a (double shape) b (/ 1.0 scale) ks ks total 0.0]
    (if-let [[k & more] (seq ks)]
      (recur (+ a k) (+ b 1.0) more
             (+ total (lgamma (+ a k)) (- (lgamma a)) (- (lgamma (inc k)))
                (* a (Math/log (/ b (+ b 1.0))))
                (- (* k (Math/log (+ b 1.0))))))
      total)))

(negative-binomial-log-evidence 3 1 [2 4])
;=> -3.847256

(def conjugate-counts
  (conj/with-conjugate-proposal counts :rate (sel/select :k)))

(array/allclose? (:weight (gfi/generate conjugate-counts [2]
                                        (cm/choicemap [:k 0] 2 [:k 1] 4)
                                        (rng/key 10)))
                 (negative-binomial-log-evidence 3 1 [2 4]))
;=> true

;; book: categories
(def categories
  (gen [n]
    (let [probs (trace :probs (d/dirichlet [1.0 2.0 0.5]))]
      (doseq [i (range n)]
        (trace [:c i] (d/categorical probs)))
      probs)))

(def drawn [0 2 1 1 0 1])

(def conjugate-categories
  (conj/with-conjugate-proposal categories :probs (sel/select :c)))

(def drawn-observations
  (apply cm/choicemap (mapcat (fn [i c] [[:c i] c]) (range) drawn)))

(array/allclose? (:weight (gfi/generate conjugate-categories [6]
                                        drawn-observations (rng/key 11)))
                 (urn-log-evidence [1.0 2.0 0.5] drawn))
;=> true

;; book: affine-error
;; error: with-conjugate-proposal: the mu of the child at [:y] is not the latent
#_(def doubled
  (gen []
    (let [mu (trace :mu (d/normal 0 3))]
      (trace :y (d/normal (* 2 mu) 1)))))

#_(gfi/generate (conj/with-conjugate-proposal doubled :mu [:y])
              [] (cm/choicemap :y 1.0) (rng/key 12))

;; book: family-error
;; error: with-conjugate-proposal: the child at [:z] is a laplace
#_(def heavy-tailed
  (gen []
    (let [mu (trace :mu (d/normal 0 3))]
      (trace :y (d/normal mu 1))
      (trace :z (d/laplace mu 1)))))

#_(gfi/generate (conj/with-conjugate-proposal heavy-tailed :mu [:y :z])
              [] (cm/choicemap :y 1.0 :z 0.5) (rng/key 13))

;; book: gibbs-chain
(defn sweeps
  "num-iters sweeps of MH on :s from its prior, then on :mu, from a trace
   of model given obs: each sweep's s, mu and whether the move on mu was
   accepted, as host values."
  [model num-iters key]
  (let [[k-start k-chain] (rng/split key)
        start (:trace (gfi/generate model [6] obs k-start))]
    (loop [tr start ks (rng/split k-chain num-iters) out []]
      (if-let [[k & more] (seq ks)]
        (let [[k-s k-mu] (rng/split k)
              tr (:trace (mh/mh tr (sel/select :s) k-s))
              {tr :trace accepted? :accepted?} (mh/mh tr (sel/select :mu) k-mu)]
          (recur tr more (conj out {:s (array/item (tr :s))
                                    :mu (array/item (tr :mu))
                                    :accepted? accepted?})))
        out))))

(def gibbs-sweeps (sweeps conjugate-scale 2000 (rng/key 14)))

(def plain-sweeps (sweeps noisy-scale 2000 (rng/key 14)))

(defn accepted
  "The fraction of sweeps whose move on mu was accepted."
  [sweeps]
  (/ (count (filter :accepted? sweeps)) (double (count sweeps))))

[(accepted gibbs-sweeps) (accepted plain-sweeps)]
;=> [1.0 0.076] ; float64

[(every? :accepted? gibbs-sweeps) (< (accepted plain-sweeps) 0.2)]
;=> [true true]

;; book: gibbs-exact
(def s-posterior
  "The posterior probabilities of s-grid's points."
  (mapv #(Math/exp %) (:log-weights (weights/normalize s-log-posterior))))

(def exact-means
  "E[s | ys], and E[mu | ys]: E[mu | ys, s], Σy/s^2 over the precision
   1/9 + n/s^2, averaged over s's posterior."
  {:s (reduce + (map * s-posterior s-grid))
   :mu (reduce + (map (fn [p s]
                        (let [precision (+ (/ 1.0 9.0) (/ (count ys) (* s s)))]
                          (* p (/ (reduce + ys) (* s s) precision))))
                      s-posterior s-grid))})

exact-means
;=> {:s 0.5249 :mu 1.9718}

(defn chain-check
  "The summaries of the sweeps' s and mu after the first 200, and whether
   each mean is within 5 MCSE of the exact one."
  [sweeps]
  (let [summary (fn [k] (diag/summary [(mapv k (subvec sweeps 200))]))
        summaries {:s (summary :s) :mu (summary :mu)}]
    [(update-vals summaries #(select-keys % [:mean :mcse-mean :ess-bulk]))
     (every? (fn [[k {:keys [mean mcse-mean]}]]
               (< (abs (- mean (exact-means k))) (* 5 mcse-mean)))
             summaries)]))

(:mu (first (chain-check gibbs-sweeps)))
;=> {:mean 1.9748 :mcse-mean 0.0062 :ess-bulk 1577.7} ; float64

(:mu (first (chain-check plain-sweeps)))
;=> {:mean 1.9446 :mcse-mean 0.0231 :ess-bulk 111.4} ; float64

(mapv (comp second chain-check) [gibbs-sweeps plain-sweeps])
;=> [true true]

;; book: ar-model
(def ar-step
  (gen [t prev]
    (let [z (trace :z (if (zero? t)
                        (d/normal 0 1)
                        (d/normal (* 0.9 prev) 0.5)))]
      (trace :y (d/normal z 0.3))
      z)))

(def ar-chain (comb/unfold ar-step))

(def kys [0.4 0.7 1.1])

;; book: ar-kalman
(def ar-prior {:mean 0.0 :cov 1.0})

(def ar-params
  {:transition 0.9 :transition-cov 0.25 :emission 1.0 :emission-cov 0.09})

(defn series
  "The choice map of a state-space generative function's observations ys:
   (ys t) at address t."
  [ys]
  (apply cm/choicemap (interleave (range) ys)))

(def ar-exact
  (:weight (gfi/assess ss/kalman [3 ar-prior ar-params] (series kys))))

ar-exact
;=> -2.501852

;; book: ar-joint
(defn ar-covariance
  "The covariance of z_0 … z_n-1 when Var z_0 = v0 and z_t = a z_t-1 plus
   noise of standard deviation q: Var z_t = a^2 Var z_t-1 + q^2, and
   Cov(z_s, z_t) = a^|t-s| Var z_min(s,t)."
  [n a q v0]
  (let [variances (vec (reductions (fn [v _] (+ (* a a v) (* q q)))
                                   v0 (range (dec n))))]
    (vec (for [s (range n)]
           (vec (for [t (range n)]
                  (* (Math/pow a (abs (- t s))) (variances (min s t)))))))))

(defn ar-log-evidence
  "log p(ys) of the chain: the ys are jointly normal, with the states'
   covariance plus r^2 on the diagonal."
  [ys a q r]
  (let [n (count ys)]
    (dc/log-prob (d/mvnormal (array/zeros [n])
                             (+ (array/array (ar-covariance n a q 1.0))
                                (* r r (array/eye n))))
                 (array/array ys))))

(array/allclose? (ar-log-evidence kys 0.9 0.5 0.3) ar-exact)
;=> true

;; book: ar-importance
(def ar-observations
  (apply cm/choicemap (mapcat (fn [t y] [[t :y] y]) (range) kys)))

(def ar-is
  (imp/importance-sampling ar-chain [3 0.0] ar-observations 3000
                           {:key (rng/key 15) :batched? true}))

(defn log-ml-se
  "The standard error of the log marginal likelihood estimate of n
   weighted samples with effective sample size ess: about
   sqrt(1/ess - 1/n)."
  [ess n]
  (Math/sqrt (- (/ 1.0 ess) (/ 1.0 n))))

[(:log-ml-estimate ar-is) (:ess ar-is)]
;=> [-2.403 278.3] ; float64

(< (abs (- (:log-ml-estimate ar-is) ar-exact))
   (* 5 (log-ml-se (:ess ar-is) 3000)))
;=> true

;; book: ar-update
(def ar-trace (:trace (gfi/generate ss/kalman [3 ar-prior ar-params]
                                    (series kys) (rng/key 16))))

(def extended
  (gfi/update ar-trace [4 ar-prior ar-params]
              [diff/unknown-change diff/no-change diff/no-change]
              (cm/choicemap 3 1.6) (rng/key 17)))

(array/allclose? (:weight extended)
                 (- (ar-log-evidence (conj kys 1.6) 0.9 0.5 0.3) ar-exact))
;=> true

(every? #(identical? (trace/subtrace ar-trace %)
                     (trace/subtrace (:trace extended) %))
        (range 3))
;=> true

;; book: ar-beliefs
(mapv (fn [{:keys [mean cov]}] [mean (Math/sqrt (array/item cov))])
      (:retval ar-trace))
;=> [[0.367 0.2873] [0.6182 0.2647] [0.9767 0.2638]]

;; book: regimes
(def coefficients [0.5 0.9 1.0])

(def regime-step
  (gen [t prev a]
    (let [z (trace :z (if (zero? t)
                        (d/normal 0 1)
                        (d/normal (* a prev) 0.5)))]
      (trace :y (d/normal z 0.3))
      z)))

(def sampled-states
  (gen [n]
    (let [k (trace :a (d/uniform-discrete 0 2))
          a (array/take (array/array coefficients) k)]
      (splice :ys (comb/unfold regime-step) n 0.0 a))))

(def summed-states
  (gen [n]
    (let [k (trace :a (d/uniform-discrete 0 2))
          a (array/take (array/array coefficients) k)]
      (splice :ys ss/kalman n ar-prior (assoc ar-params :transition a)))))

;; book: drift
(def drift [0.3 0.9 1.4 1.2 1.9 2.5 2.2 2.8 3.1 2.6 3.3 3.9])

(def drift-log-likelihoods
  "log p(drift | a) for each coefficient a, by the Kalman filter."
  (mapv (fn [a]
          (array/item (:weight (gfi/assess ss/kalman
                                           [(count drift) ar-prior
                                            (assoc ar-params :transition a)]
                                           (series drift)))))
        coefficients))

(def drift-evidence
  (- (weights/log-sum-exp drift-log-likelihoods) (Math/log 3)))

drift-evidence
;=> -11.4555

(mapv #(Math/exp %) (:log-weights (weights/normalize drift-log-likelihoods)))
;=> [0.0 0.0364 0.9636]

;; book: filters
(defn run-filter
  "A bootstrap particle filter of n particles on model through drift,
   batched, resampling when the ESS falls below n/2; observation is the
   choice map of step t's observation. Its state."
  [model observation n key]
  (let [steps (count drift)
        ks (rng/split key (* 2 steps))]
    (reduce (fn [state t]
              (-> state
                  (pf/maybe-resample {:key (ks (+ steps t))})
                  (pf/step [(inc t)] [diff/unknown-change] (observation t)
                           {:key (ks t)})))
            (pf/initialize model [1] (observation 0) n
                           {:batched? true :key (ks 0)})
            (range 1 steps))))

(defn sampled-observation [t] (cm/choicemap [:ys t :y] (drift t)))

(defn summed-observation [t] (cm/choicemap [:ys t] (drift t)))

(defn filter-runs
  "The log evidence estimates of r runs of a filter of n particles."
  [model observation n r key]
  (mapv #(pf/log-ml-estimate (run-filter model observation n %))
        (rng/split key r)))

;; book: filter-variance
(def bootstrap-runs
  (filter-runs sampled-states sampled-observation 400 250 (rng/key 18)))

(def rao-blackwell-runs
  (filter-runs summed-states summed-observation 400 250 (rng/key 18)))

(mapv (comp first variance-and-se) [bootstrap-runs rao-blackwell-runs])
;=> [0.157 0.0046] ; float64

(let [[v1 se1] (variance-and-se bootstrap-runs)
      [v2 se2] (variance-and-se rao-blackwell-runs)]
  (> (- v1 v2) (* 5 (Math/sqrt (+ (* se1 se1) (* se2 se2))))))
;=> true

[(ratio-check bootstrap-runs drift-evidence)
 (ratio-check rao-blackwell-runs drift-evidence)]
;=> [true true]

;; book: casino
(def dice
  "The two dice's probabilities of faces 0 … 5, 5 standing for a six:
   the fair die's, then the loaded one's."
  (array/array [(vec (repeat 6 (/ 1.0 6))) [0.1 0.1 0.1 0.1 0.1 0.5]]))

(def switches
  "The probabilities of the next die given this one: row 0 the fair's."
  (array/array [[0.95 0.05] [0.1 0.9]]))

(def casino {:transition switches :emission (d/categorical dice)})

(def start (array/array [0.5 0.5]))

(def rolls [5 5 2 0 1 3 2 5 0 1 5 2 0 5 3 5 4 5 0 3 0 1 3 4 5 2 4 4 3 2
            3 2 3 5 5 0 1 3 3 4 4 4 5 5 5 5 5 2 3 5 0 5 5 5 0 1 5 5 3 5])

(def first-rolls (subvec rolls 0 10))

(def casino-exact
  (:weight (gfi/assess ss/hmm [10 start casino] (series first-rolls))))

casino-exact
;=> -17.488016

;; book: casino-enumeration
(def casino-step
  (gen [t prev]
    (let [die (trace :die (d/categorical (if (zero? t)
                                           start
                                           (array/take switches prev 0))))]
      (trace :roll (d/categorical (array/take dice die 0)))
      die)))

(def every-path
  (apply enum/choice-vol-grid (map (fn [t] [[t :die] [0 1]]) (range 10))))

(def enumerated
  (enum/enumerative-inference
   (comb/unfold casino-step) [10 0]
   (apply cm/choicemap (mapcat (fn [t r] [[t :roll] r]) (range) first-rolls))
   every-path {:batched? true :key (rng/key 20)}))

[(count every-path)
 (array/allclose? (:log-ml-estimate enumerated) casino-exact)]
;=> [1024 true]

;; book: casino-smoothing
(def smoothed
  "P(die t is loaded | the ten rolls), by enumeration."
  (mapv #(get (enum/marginal enumerated [% :die]) 1 0.0) (range 10)))

smoothed
;=> [0.730 0.660 0.430 0.305 0.242 0.217 0.222 0.259 0.217 0.207]

(def casino-trace
  (:trace (gfi/generate ss/hmm [10 start casino] (series first-rolls)
                        (rng/key 21))))

(def die-paths
  (mapv #(mapv array/item (ss/sample-states casino-trace %))
        (rng/split (rng/key 22) 2000)))

(defn loaded-share
  "The share of paths whose die t is loaded."
  [paths t]
  (/ (count (filter #(= 1 (% t)) paths)) (double (count paths))))

(every? (fn [t]
          (let [p (smoothed t)]
            (< (abs (- (loaded-share die-paths t) p))
               (* 5 (Math/sqrt (/ (* p (- 1 p)) (count die-paths)))))))
        (range 10))
;=> true

;; book: casino-figure
;; figure: figures/042-casino.svg
(let [tr (:trace (gfi/generate ss/hmm [60 start casino] (series rolls)
                               (rng/key 23)))
      paths (mapv #(mapv array/item (ss/sample-states tr %))
                  (rng/split (rng/key 24) 500))]
  (svg/paths [{:paths [(mapv (fn [t] [t (loaded-share paths t)]) (range 60))]}
              {:paths [(keep (fn [t] (when (= 5 (rolls t)) [t 1.05]))
                             (range 60))]
               :width 0 :r 2.5 :color "firebrick"}]
             {:x-label "roll t" :y-label "P(loaded at t | rolls)"
              :y-range [0.0 1.1]}))

;; book: six-model
(defn casino-with
  "The casino's parameters with the loaded die's probability of a six p6,
   its other faces sharing the rest."
  [p6]
  (let [others (array/full [5] (/ (- 1.0 p6) 5.0))]
    {:transition switches
     :emission (d/categorical
                (array/stack [(array/full [6] (/ 1.0 6))
                              (array/concatenate [others
                                                  (array/reshape p6 [1])])]))}))

(def unknown-six
  (gen [n]
    (let [p6 (trace :p6 (d/uniform 0 1))]
      (splice :rolls ss/hmm n start (casino-with p6))
      p6)))

(def roll-observations
  (apply cm/choicemap (mapcat (fn [t r] [[:rolls t] r]) (range) rolls)))

;; book: six-grid
(def p6-grid
  "Midpoints of 200 intervals covering [0, 1]."
  (mapv #(* 0.005 (+ % 0.5)) (range 200)))

(def p6-posterior
  "The posterior probabilities of p6-grid's points: the prior is uniform,
   and the HMM gives each point's likelihood exactly."
  (let [lls (mapv #(array/item
                    (:weight (gfi/assess ss/hmm [60 start (casino-with %)]
                                         (series rolls))))
                  p6-grid)]
    (mapv #(Math/exp %) (:log-weights (weights/normalize lls)))))

(def p6-mean (reduce + (map * p6-posterior p6-grid)))

p6-mean
;=> 0.6135

;; book: six-chain
(def six-chain
  (let [[k-start k-chain] (rng/split (rng/key 25))
        start-tr (:trace (gfi/generate unknown-six [60] roll-observations
                                       k-start))]
    (loop [tr start-tr ks (rng/split k-chain 1500) out []]
      (if-let [[k & more] (seq ks)]
        (let [tr (:trace (mh/mh tr (sel/select :p6) k))]
          (recur tr more (conj out (array/item (tr :p6)))))
        out))))

(def six-summary (diag/summary [(subvec six-chain 100)]))

(select-keys six-summary [:mean :mcse-mean :ess-bulk])
;=> {:mean 0.6178 :mcse-mean 0.0058 :ess-bulk 538.3} ; float64

(< (abs (- (:mean six-summary) p6-mean)) (* 5 (:mcse-mean six-summary)))
;=> true

;; book: six-states
(def six-trace
  (:trace (gfi/generate unknown-six [60]
                        (cm/set-choice roll-observations :p6 (peek six-chain))
                        (rng/key 31))))

(def dice-path (ss/sample-states (trace/subtrace six-trace :rolls)
                                 (rng/key 32)))

(count dice-path)
;=> 60

;; book: six-figure
;; figure: figures/042-six.svg
(svg/histogram [{:values (subvec six-chain 100)
                 :density (fn [x]
                            (let [i (min 199 (long (/ x 0.005)))]
                              (/ (p6-posterior i) 0.005)))}]
               {:bins 30 :range [0.0 1.0]
                :x-label "p6, the loaded die's probability of a six"})

;; book: two-means
(def two-means
  (gen [n m]
    (let [mu (trace :mu (d/normal 0 3))
          nu (trace :nu (d/normal 0 3))]
      (doseq [i (range n)]
        (trace [:y i] (d/normal mu 1)))
      (doseq [i (range m)]
        (trace [:z i] (d/normal nu 1))))))

(def zs [-0.4 0.3 -1.1])

(def both-observed
  (cm/merge obs (apply cm/choicemap (mapcat (fn [i z] [[:z i] z]) (range) zs))))

(def mu-only (conj/with-conjugate-proposal two-means :mu (sel/select :y)))

(mapv #(:weight (gfi/generate mu-only [6 3] both-observed (rng/key %))) [1 2])
;=> [-31.313 -46.765] ; float64

;; book: nested-error
;; error: with-conjugate-proposal needs a dynamic generative function
#_(conj/with-conjugate-proposal mu-only :nu (sel/select :z))

;; book: one-each
(def grouped
  (gen [n m]
    (splice :a conjugate-mean n)
    (splice :b conjugate-mean m)))

(def grouped-observations
  (apply cm/choicemap (concat (mapcat (fn [i y] [[:a :y i] y]) (range) ys)
                              (mapcat (fn [i z] [[:b :y i] z]) (range) zs))))

(mapv #(array/allclose? (:weight (gfi/generate grouped [6 3]
                                               grouped-observations
                                               (rng/key %)))
                        (+ (log-evidence ys) (log-evidence zs)))
      [1 2])
;=> [true true]

;; book: gap
(def walk-args [12 ar-prior (assoc ar-params :transition 1.0)])

(def gapped
  "drift's observations without step 5's."
  (apply cm/choicemap (mapcat (fn [t y] (when (not= t 5) [t y]))
                              (range) drift)))

(def gap-weights
  (array/->clj (:weights (batched/generate ss/kalman walk-args gapped 4000
                                           (rng/key 26)))))

(- (apply max gap-weights) (apply min gap-weights))
;=> 8.08 ; float64

;; book: gap-exact
(def gap-evidence
  "log p of drift but step 5, from the joint Gaussian of the walk's
   observations with step 5's row and column left out."
  (let [kept (vec (remove #{5} (range 12)))
        full (ar-covariance 12 1.0 0.5 1.0)
        cov (mapv (fn [s] (mapv (fn [t] (+ (get-in full [s t])
                                           (if (= s t) 0.09 0.0)))
                                kept))
                  kept)]
    (array/item (dc/log-prob (d/mvnormal (array/zeros [11]) (array/array cov))
                             (array/array (mapv drift kept))))))

gap-evidence
;=> -9.75597

(ratio-check gap-weights gap-evidence)
;=> true

;; book: gap-workaround
(defn blind-at
  "Kalman parameters that, at step gap, observe nothing of the state: an
   emission of 0, its noise 1."
  [params gap]
  (fn [t] (if (= t gap) (assoc params :emission 0.0 :emission-cov 1.0) params)))

(let [[n prior params] walk-args
      w (:weight (gfi/assess ss/kalman [n prior (blind-at params 5)]
                             (cm/set-choice gapped 5 0.0)))]
  (array/allclose? (- w (dc/log-prob (d/normal 0 1) 0.0)) gap-evidence))
;=> true

;; book: solution-1-model
(def waits
  (gen [n]
    (let [rate (trace :rate (d/gamma 2 1))]
      (doseq [i (range n)]
        (trace [:t i] (d/exponential rate)))
      rate)))

(def waiting (cm/choicemap [:t 0] 0.5 [:t 1] 1.5))

;; book: solution-1-error
;; error: with-conjugate-proposal: the child at [:t 0] is a
#_(gfi/generate (conj/with-conjugate-proposal waits :rate (sel/select :t))
              [2] waiting (rng/key 27))

;; book: solution-1
(defn lomax-log-evidence
  "log p(ts) of exponential waits ts with a gamma(shape, scale) rate
   summed out: each wait is Lomax given the ones before, a b^a / (b +
   t)^(a+1) for shape a and rate b = 1/scale, and adds 1 to the shape and
   itself to the rate."
  [shape scale ts]
  (loop [a (double shape) b (/ 1.0 scale) ts ts total 0.0]
    (if-let [[t & more] (seq ts)]
      (recur (+ a 1.0) (+ b t) more
             (+ total (Math/log a) (* a (Math/log b))
                (- (* (+ a 1.0) (Math/log (+ b t))))))
      total)))

(lomax-log-evidence 2 1 [0.5 1.5])
;=> -2.60269

(def waits-is
  (imp/importance-sampling waits [2] waiting 10000
                           {:key (rng/key 28) :batched? true}))

(:ess waits-is)
;=> 7143.3 ; float64

(< (abs (- (:log-ml-estimate waits-is) (lomax-log-evidence 2 1 [0.5 1.5])))
   (* 5 (log-ml-se (:ess waits-is) 10000)))
;=> true

;; book: solution-2
(defn hand-kalman
  "log p(ys) of the chain z_0 ~ N(0, 1), z_t ~ N(a z_t-1, q), y_t ~ N(z_t,
   r), by the Kalman recursion on host numbers: predict the state, score
   y_t by its predictive N(m, P + r^2), then correct. ahead is the
   predicted variance as a function of the last one, a^2 P + q^2."
  ([ys a q r] (hand-kalman ys a q r (fn [p] (+ (* a a p) (* q q)))))
  ([ys a q r ahead]
   (loop [t 0 m 0.0 p 1.0 total 0.0]
     (if (= t (count ys))
       total
       (let [[m p] (if (zero? t) [m p] [(* a m) (ahead p)])
             s (+ p (* r r))
             e (- (ys t) m)
             gain (/ p s)]
         (recur (inc t) (+ m (* gain e)) (* (- 1.0 gain) p)
                (- total (* 0.5 (+ (Math/log (* 2 Math/PI s))
                                   (/ (* e e) s))))))))))

(array/allclose? (hand-kalman kys 0.9 0.5 0.3) ar-exact)
;=> true

(hand-kalman kys 0.9 0.5 0.3 (fn [_] (* 0.5 0.5)))
;=> -2.4535

;; book: solution-3
(def runs-400 (log-ml-runs noisy-mean [6] obs 500 400 (rng/key 29)))

(let [[v1 se1] (variance-and-se plain-runs)
      [v2 se2] (variance-and-se runs-400)]
  [(* 100 v1) (* 400 v2)])
;=> [5.62 4.87] ; float64

(let [[v1 se1] (variance-and-se plain-runs)
      [v2 se2] (variance-and-se runs-400)]
  (< (abs (- (* 100 v1) (* 400 v2)))
     (* 5 (Math/sqrt (+ (* 100 se1 100 se1) (* 400 se2 400 se2))))))
;=> true

;; book: solution-4
(defn ar-smoothing
  "The mean and standard deviation of each z_t given ys, by conditioning
   the joint Gaussian of the states and the observations: with Σ the
   states' covariance and C = Σ + r^2 I the observations', E[z | ys] = Σ
   C^-1 ys and Cov[z | ys] = Σ - Σ C^-1 Σ."
  [ys a q r]
  (let [n (count ys)
        sigma (array/array (ar-covariance n a q 1.0))
        c-inv-sigma (linalg/solve (+ sigma (* r r (array/eye n))) sigma)
        cov (- sigma (array/matmul sigma c-inv-sigma))]
    {:mean (array/->clj (array/matmul (array/transpose c-inv-sigma)
                                      (array/array ys)))
     :sd (mapv #(Math/sqrt %) (array/->clj (array/diag cov)))}))

(def ar-smoothed (ar-smoothing kys 0.9 0.5 0.3))

ar-smoothed
;=> {:mean [0.4548 0.7046 0.9767] :sd [0.2616 0.2451 0.2638]}

(def state-draws
  (mapv #(mapv array/item (ss/sample-states ar-trace %))
        (rng/split (rng/key 30) 2000)))

(every? (fn [t]
          (let [[m se] (mean-and-se (mapv #(% t) state-draws))]
            (< (abs (- m (get-in ar-smoothed [:mean t])))
               (* 5 (/ (get-in ar-smoothed [:sd t]) (Math/sqrt 2000))))))
        (range 3))
;=> true
