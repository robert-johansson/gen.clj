;; The code of book/027-vi.md. Each region, from a `;; book:` line to the
;; next, is included there by name (gen.book: `clojure -M:test -m gen.book
;; sync` writes the Markdown's copies; gen.book-test checks them).
;;
;; Bands. The fits' bands are measured spreads over keys (the iterates of
;; a stochastic optimizer have no closed-form spread): the mean plus five
;; standard deviations of the quantity, or five around the mean, over
;; (rng/key k) for k in 100-115 (Gen.jl's settings, both estimators),
;; 11-30 (the line's mean-field and structured fits, the two modes), 11-20
;; (the stronger ridge) and 21-30 (VIMCO's widths, each ratio at least
;; 2.1). Every other statistical check has a band of five standard errors,
;; from the draws themselves, or an exact value.
;;
;; Planted bugs (2026-10-24), each failing its check: the normal KL without
;; its -0.5, or KL(p || q) for KL(q || p) (the ELBO identity); the guide's
;; sd 5% off the posterior's (equal log weights); 0.0005 for Gen.jl's step,
;; or readings offset by 0.2 (its fit's band); no baseline (the baseline's
;; fit); 3.6 for 4 in the exact gradient (the estimators' means); pathwise
;; estimates from one draw (the variance ratio); prob-a 0.4 (foo's fit);
;; the marginal sds as the mean-field optimum's (its bands); the gap
;; without its ½ (the excess KL); the coupling dropped (the structured
;; KL); the structured guide's log weight without log q (the ELBO gain);
;; the mean log weight for the K-sample bound (the bound's checks); the
;; ELBO's fit for VIMCO's (the widths, the bounds); 10 draws for 20 (the
;; landing's sds); Adam at 0.05 on the ridge (its excess); the mixture's
;; KL without log 2 (two modes); no baseline (the growth of the variance).

;; book: ns
(ns book.vi
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.vi :as vi]
            [gen.inference.weights :as w]
            [gen.learning :as learn]
            [gen.random :as rng]))

;; book: target
(def target
  (gen []
    (trace :x (d/normal -1 (array/exp 0.5)))))

;; book: approx
(def approx
  (gen []
    (trace :x (d/normal (param :x-mu) (array/exp (param :x-log-std))))))

;; book: fit-target
(def start {:x-mu 0.0 :x-log-std 0.0})

(def to-target
  (vi/black-box-vi target [] cm/EMPTY approx [] start
                   (learn/gradient-descent {:step-size-init 1.0
                                            :step-size-beta 1000})
                   {:iters 200 :samples 100 :batched? true
                    :key (rng/key 1)}))

(:store to-target)
;=> {:x-mu -1.0 :x-log-std 0.5}

(let [{:keys [x-mu x-log-std]} (:store to-target)]
  [(array/allclose? x-mu -1.0) (array/allclose? x-log-std 0.5)])
;=> [true true]

;; book: fixed-point
(vi/elbo-gradient target [] cm/EMPTY approx [] {:x-mu -1.0 :x-log-std 0.5}
                  100 {:key (rng/key 2)})
;=> {:objective 0.0 :grads {:x-mu 0.0 :x-log-std 0.0}}

;; book: sample-guide
(def fitted-approx (dyn/with-params approx (:store to-target)))

((gfi/simulate fitted-approx [] (rng/key 3)) :x)
;=> -1.4079 ; float64

;; book: model
(def model
  (gen [n]
    (let [x (trace :x (d/normal 0 1))]
      (dotimes [i n]
        (trace [:y i] (d/normal x 0.5)))
      x)))

(def ys [3.12 2.25 2.21 1.55 2.15 1.06])

(defn observations
  "The choice map fixing each [:y i] to (ys i)."
  [ys]
  (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

(def observed (observations ys))

;; book: exact
(defn posterior
  "The exact posterior of x given ys: a normal with mean 4·Σy/(1 + 4n)
   and standard deviation 1/sqrt(1 + 4n)."
  [ys]
  (let [precision (+ 1.0 (* 4.0 (count ys)))]
    {:mu (/ (* 4.0 (reduce + ys)) precision)
     :sd (Math/sqrt (/ 1.0 precision))}))

(def exact (posterior ys))

exact
;=> {:mu 1.9744 :sd 0.2}

(defn log-evidence-at
  "log p(ys) by Bayes' rule at x: log p(x, ys) - log p(x | ys)."
  [x]
  (array/item
   (- (:weight (gfi/assess model [6] (cm/set-choice observed :x x)))
      (dc/log-prob (d/normal (:mu exact) (:sd exact)) x))))

[(log-evidence-at 0.0) (log-evidence-at 2.0)]
;=> [-9.8952 -9.8952]

(def log-evidence (log-evidence-at 0.0))

;; book: log-weight
(defn log-weight
  "log p(observations, z) - log q(z) for one draw z of the guide under the
   store: the model assessed at the observations and the guide's choices."
  [model args observations guide store key]
  (let [q (gfi/simulate (dyn/with-params guide store) [] key)
        joint (cm/merge observations (:choices q))]
    (array/item (- (:weight (gfi/assess model args joint)) (:score q)))))

(defn log-weights
  "The log weights of n draws of the guide, from n keys split from key."
  [model args observations guide store n key]
  (mapv #(log-weight model args observations guide store %)
        (rng/split key n)))

(defn estimate
  "The mean of the numbers xs and its standard error: {:mean :se}."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        sq (fn [x] (let [e (- x mean)] (* e e)))]
    {:mean mean :se (Math/sqrt (/ (reduce + (map sq xs)) (* n (dec n))))}))

;; book: elbo-identity
(defn kl-normal
  "KL(N(m1, s1) || N(m2, s2)), in closed form."
  [m1 s1 m2 s2]
  (+ (Math/log (/ s2 s1))
     (/ (+ (* s1 s1) (* (- m1 m2) (- m1 m2))) (* 2.0 s2 s2))
     -0.5))

(defn kl-from-posterior
  "KL(q || p(x | ys)) for approx under the store."
  [{:keys [x-mu x-log-std]}]
  (kl-normal (array/item x-mu) (Math/exp (array/item x-log-std))
             (:mu exact) (:sd exact)))

(def q {:x-mu 1.8 :x-log-std -1.4})

(- log-evidence (kl-from-posterior q))
;=> -10.3261

(def elbo-estimate
  (estimate (log-weights model [6] observed approx q 2000 (rng/key 4))))

elbo-estimate
;=> {:mean -10.3221 :se 0.0246} ; float64

(< (abs (- (:mean elbo-estimate) (- log-evidence (kl-from-posterior q))))
   (* 5 (:se elbo-estimate)))
;=> true

;; book: at-posterior
(def at-posterior {:x-mu (:mu exact) :x-log-std (Math/log (:sd exact))})

(kl-from-posterior at-posterior)
;=> 0.0

(every? #(array/allclose? % log-evidence)
        (log-weights model [6] observed approx at-posterior 100 (rng/key 5)))
;=> true

;; book: genjl-fit
(def genjl
  (vi/black-box-vi model [6] observed approx [] start
                   (learn/gradient-descent {:step-size-init 0.01
                                            :step-size-beta 1000})
                   {:iters 500 :samples 200 :batched? true
                    :key (rng/key 6)}))

(mapv (:history genjl) [0 49 99 199 299 399 499])
;=> [-66.92 -9.96 -10.22 -9.89 -9.91 -9.90 -9.91] ; float64

;; book: elbo-figure
;; figure: figures/027-elbo-history.svg
(svg/series (:history genjl) {:y-label "ELBO estimate"})

;; book: genjl-result
(:store genjl)
;=> {:x-mu 1.9273 :x-log-std -1.5971} ; float64

(kl-from-posterior (:store genjl))
;=> 0.0279 ; float64

(< (kl-from-posterior (:store genjl)) 0.2)
;=> true

;; book: noisy-at-posterior
(defn gradient-at
  "elbo-gradient's estimate of the ELBO's gradient for approx under the
   store, from 200 draws, with the options given."
  [store opts]
  (:grads (vi/elbo-gradient model [6] observed approx [] store 200
                            (assoc opts :key (rng/key 7)))))

(gradient-at at-posterior {})
;=> {:x-mu 2.2954 :x-log-std 1.175} ; float64

(gradient-at at-posterior {:baseline :leave-one-out})
;=> {:x-mu 0.0 :x-log-std 0.0}

;; book: baseline-fit
(def with-baseline
  (vi/black-box-vi model [6] observed approx [] start
                   (learn/gradient-descent {:step-size-init 0.01
                                            :step-size-beta 1000})
                   {:iters 500 :samples 200 :batched? true
                    :baseline :leave-one-out :key (rng/key 6)}))

(:store with-baseline)
;=> {:x-mu 1.974 :x-log-std -1.609}

(< (kl-from-posterior (:store with-baseline)) 1e-6)
;=> true

(last (:history with-baseline))
;=> -9.895

;; book: exact-gradient
(defn exact-gradient
  "The ELBO's gradient for approx under the store, in closed form:
   4·Σy - (1 + 4n)·mu in x-mu, 1 - (1 + 4n)·sd² in x-log-std."
  [{:keys [x-mu x-log-std]}]
  (let [precision (+ 1.0 (* 4.0 (count ys)))
        sd (Math/exp x-log-std)]
    {:x-mu (- (* 4.0 (reduce + ys)) (* precision x-mu))
     :x-log-std (- 1.0 (* precision sd sd))}))

(exact-gradient start)
;=> {:x-mu 49.36 :x-log-std -24.0}

(exact-gradient at-posterior)
;=> {:x-mu 0.0 :x-log-std 0.0}

;; book: gradient-draws
(def estimators
  {:score {}
   :leave-one-out {:baseline :leave-one-out}
   :pathwise {:estimator :reparam}})

(defn gradient-draws
  "The x-mu components of reps estimates of the ELBO's gradient for approx
   under the store, each from n draws, with elbo-gradient's options."
  [store opts reps n key]
  (mapv (fn [k]
          (let [g (vi/elbo-gradient model [6] observed approx [] store n
                                    (assoc opts :key k))]
            (array/item (:x-mu (:grads g)))))
        (rng/split key reps)))

(def draws
  (update-vals estimators #(gradient-draws start % 500 20 (rng/key 8))))

(update-vals draws (comp :mean estimate))
;=> {:score 50.39 :leave-one-out 50.55 :pathwise 49.33} ; float64

(every? (fn [xs]
          (let [{:keys [mean se]} (estimate xs)]
            (< (abs (- mean (:x-mu (exact-gradient start)))) (* 5 se))))
        (vals draws))
;=> true

;; book: variances
(defn variance
  "The sample variance of the numbers xs."
  [xs]
  (let [{:keys [se]} (estimate xs)] (* se se (count xs))))

(update-vals draws #(Math/sqrt (variance %)))
;=> {:score 27.35 :leave-one-out 18.42 :pathwise 5.54} ; float64

(< (* 4 (variance (:pathwise draws))) (variance (:leave-one-out draws)))
;=> true

;; book: gradients-figure
;; figure: figures/027-gradients.svg
(svg/histogram (for [[k xs] draws] {:values xs :label (name k)})
               {:range [-20 160] :bins 60 :x-label "gradient in x-mu"})

;; book: pathwise-families
(mapv dc/pathwise? [(d/normal 0 1) (d/uniform 0 1) (d/exponential 1)
                    (d/gamma 2 1) (d/beta 2 2) (d/bernoulli 0.5)])
;=> [true true true false false false]

;; book: foo
(def foo
  (gen [prob-a]
    (let [val (if (trace :a (d/bernoulli prob-a))
                (trace :b (d/bernoulli 0.6))
                true)]
      (and (trace :c (d/bernoulli (if val 0.9 0.2))) val))))

(def foo-guide
  (gen []
    (when (trace :a (d/bernoulli (array/sigmoid (param :a-logit))))
      (trace :b (d/bernoulli (array/sigmoid (param :b-logit)))))))

(def not-c (cm/choicemap :c false))

(def logits {:a-logit 0.0 :b-logit 0.0})

;; book: foo-reparam
;; error: The reparameterization estimator needs every choice of q
#_(vi/elbo-gradient foo [0.3] not-c foo-guide [] logits 10
                    {:estimator :reparam :key (rng/key 9)})

;; book: foo-fit
(def foo-fit
  (vi/black-box-vi foo [0.3] not-c foo-guide [] logits
                   (learn/gradient-descent {:step-size-init 1.0
                                            :step-size-beta 1000})
                   {:iters 300 :samples 20 :baseline :leave-one-out
                    :key (rng/key 10)}))

(let [{:keys [a-logit b-logit]} (:store foo-fit)]
  [(array/sigmoid a-logit) (array/sigmoid b-logit)])
;=> [0.6196 0.1579]

[(/ 0.114 0.184) (/ 0.018 0.114)]
;=> [0.6196 0.1579]

(last (:history foo-fit))
;=> -1.6928

(Math/log 0.184)
;=> -1.6928

;; book: line-model
(def line-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

(def xs [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0])

(def line-ys [8.23 5.87 3.99 2.59 0.23 -0.66 -3.53 -6.91 -7.24 -9.90])

(def line-observed (observations line-ys))

;; book: line-exact
(defn line-posterior
  "The exact posterior of [slope intercept] given points (xs, ys): its
   precision, the prior's diag(1/2², 1/10²) plus the design's Gram
   matrix, its covariance and its mean."
  [xs ys]
  (let [X (array/array (mapv (fn [x] [x 1.0]) xs))
        Xt (array/transpose X)
        precision (+ (array/diag (array/array [0.25 0.01]))
                     (array/matmul Xt X))
        covariance (linalg/inv precision)]
    {:precision precision
     :covariance covariance
     :mean (array/matmul covariance (array/matmul Xt (array/array ys)))}))

(def line (line-posterior xs line-ys))

(:mean line)
;=> [-1.979 10.143]

(defn sds-and-correlation
  "The standard deviations and the correlation of a 2 × 2 covariance."
  [covariance]
  (let [[[a c] [_ b]] (array/->clj covariance)]
    {:sds [(Math/sqrt a) (Math/sqrt b)]
     :correlation (/ c (Math/sqrt (* a b)))}))

(sds-and-correlation (:covariance line))
;=> {:sds [0.1097 0.6807] :correlation -0.886}

;; book: line-evidence
(def line-log-evidence
  "log p(ys) by Bayes' rule at slope -2 and intercept 9."
  (let [at (cm/choicemap :slope -2.0 :intercept 9.0)]
    (array/item
     (- (:weight (gfi/assess line-model [xs] (cm/merge line-observed at)))
        (dc/log-prob (d/mvnormal (:mean line) (:covariance line))
                     (array/array [-2.0 9.0]))))))

line-log-evidence
;=> -18.1505

;; book: mean-field
(def mean-field
  (gen []
    (trace :slope (d/normal (param :slope-mu)
                            (array/exp (param :slope-log-sd))))
    (trace :intercept (d/normal (param :intercept-mu)
                                (array/exp (param :intercept-log-sd))))))

(def mean-field-start
  {:slope-mu 0.0 :slope-log-sd 0.0 :intercept-mu 0.0 :intercept-log-sd 0.0})

;; book: mean-field-theory
(defn mean-field-optimum
  "The best mean-field guide for a Gaussian posterior: the posterior's
   means, standard deviations one over the root of the precision's
   diagonal, and its KL from the posterior, -½ log(1 - r²) for two
   coordinates of correlation r."
  [{:keys [precision covariance]}]
  (let [r (:correlation (sds-and-correlation covariance))]
    {:sds (mapv #(Math/sqrt (/ 1.0 %))
                (array/->clj (linalg/diagonal precision)))
     :gap (* -0.5 (Math/log (- 1.0 (* r r))))}))

(mean-field-optimum line)
;=> {:sds [0.0509 0.3161] :gap 0.767}

;; book: advi
(defn advi
  "The guide's store fit by ADVI from start: Adam at the rate lr (0.05)
   for 1000 iterations, then at 0.005 for 1000 more to settle, each with
   10 pathwise draws."
  [model args observations guide start {:keys [lr key] :or {lr 0.05}}]
  (let [run (fn [store lr k]
              (:store (vi/black-box-vi model args observations guide []
                                       store (learn/adam {:lr lr})
                                       {:iters 1000 :samples 10
                                        :estimator :reparam :key k})))
        [k1 k2] (rng/split key)]
    (-> start (run lr k1) (run 0.005 k2))))

(def mean-field-fit
  (advi line-model [xs] line-observed mean-field mean-field-start
        {:key (rng/key 11)}))

;; book: mean-field-result
(defn sds
  "A guide's standard deviations of the slope and the intercept under the
   store."
  [{:keys [slope-log-sd intercept-log-sd]}]
  (mapv #(Math/exp (array/item %)) [slope-log-sd intercept-log-sd]))

(defn means
  "A guide's means of the slope and the intercept under the store."
  [{:keys [slope-mu intercept-mu]}]
  (mapv array/item [slope-mu intercept-mu]))

[(means mean-field-fit) (sds mean-field-fit)]
;=> [[-1.9762 10.1354] [0.0523 0.3206]] ; float64

(defn within?
  "Is each of xs within its band of the corresponding target?"
  [xs targets bands]
  (every? true? (map #(< (abs (- %1 %2)) %3) xs targets bands)))

[(within? (means mean-field-fit) (array/->clj (:mean line)) [0.035 0.04])
 (within? (sds mean-field-fit) (:sds (mean-field-optimum line))
          [0.007 0.05])]
;=> [true true]

;; book: gaussian-kl
(defn moments
  "The mean and covariance of a guide's [slope intercept] under the store:
   the mean-field guide's, or, given a :coupling c, the structured
   guide's, whose intercept is c·(slope - its mean) plus independent
   noise."
  [{:keys [coupling] :or {coupling 0.0} :as store}]
  (let [[v1 v2] (mapv #(* % %) (sds store))
        c (array/item coupling)]
    {:mean (array/array (means store))
     :covariance (array/array [[v1 (* c v1)]
                               [(* c v1) (+ (* c c v1) v2)]])}))

(defn kl-from-gaussian
  "KL(q || p) for a guide's moments m and S, against a Gaussian posterior
   of mean u, precision Λ and covariance Σ, in closed form:
   ½[tr(ΛS) + (m - u)·Λ(m - u) - 2 + log det Σ - log det S]."
  [{:keys [precision covariance mean]} {m :mean S :covariance}]
  (let [dm (- m mean)]
    (array/item
     (* 0.5 (+ (array/sum (* precision S))
               (array/sum (* dm (array/matmul precision dm)))
               -2.0
               (array/log (linalg/det covariance))
               (- (array/log (linalg/det S))))))))

(def mean-field-kl (kl-from-gaussian line (moments mean-field-fit)))

mean-field-kl
;=> 0.7689 ; float64

(< 0.0 (- mean-field-kl (:gap (mean-field-optimum line))) 0.075)
;=> true

;; book: draws-of
(defn draws-of
  "n draws of [slope intercept] from the guide under the store, as host
   numbers."
  [guide store n key]
  (mapv (fn [k]
          (let [t (gfi/simulate (dyn/with-params guide store) [] k)]
            [(array/item (t :slope)) (array/item (t :intercept))]))
        (rng/split key n)))

(def posterior-draws
  (array/->clj (dc/sample (d/mvnormal (:mean line) (:covariance line))
                          (rng/key 12) [400])))

(defn ridge-figure
  "The posterior's draws (black) and a guide's (orange), on one box."
  [guide store key]
  (svg/plot [{:points posterior-draws :r 1.5 :opacity 0.35}
             {:points (draws-of guide store 400 key)
              :color "darkorange" :r 1.5 :opacity 0.6}]
            {:x-range [-2.4 -1.55] :y-range [7.6 12.7]
             :x-label "slope" :y-label "intercept"}))

;; book: mean-field-figure
;; figure: figures/027-mean-field.svg
(ridge-figure mean-field mean-field-fit (rng/key 13))

;; book: structured
(def structured
  (gen []
    (let [slope (trace :slope (d/normal (param :slope-mu)
                                        (array/exp (param :slope-log-sd))))]
      (trace :intercept
             (d/normal (+ (param :intercept-mu)
                          (* (param :coupling) (- slope (param :slope-mu))))
                       (array/exp (param :intercept-log-sd)))))))

(def structured-fit
  (advi line-model [xs] line-observed structured
        (assoc mean-field-start :coupling 0.0) {:key (rng/key 14)}))

;; book: structured-result
(sds-and-correlation (:covariance (moments structured-fit)))
;=> {:sds [0.1099 0.675] :correlation -0.8777} ; float64

(def structured-kl (kl-from-gaussian line (moments structured-fit)))

structured-kl
;=> 0.002 ; float64

(< structured-kl 0.055)
;=> true

;; book: compare-elbos
(defn difference
  "The difference of two estimates, a - b, with its standard error."
  [a b]
  {:mean (- (:mean a) (:mean b))
   :se (Math/sqrt (+ (* (:se a) (:se a)) (* (:se b) (:se b))))})

(defn line-elbo
  "The ELBO of the guide on the line under the store, from 2000 draws."
  [guide store key]
  (estimate (log-weights line-model [xs] line-observed guide store 2000 key)))

(def elbo-gain
  (difference (line-elbo structured structured-fit (rng/key 15))
              (line-elbo mean-field mean-field-fit (rng/key 16))))

elbo-gain
;=> {:mean 0.7505 :se 0.0199} ; float64

(< (abs (- (:mean elbo-gain) (- mean-field-kl structured-kl)))
   (* 5 (:se elbo-gain)))
;=> true

;; book: bound
(defn bound-draws
  "reps draws of the K-sample bound's estimate for the guide on the line
   under the store: each the log of the mean weight of k draws."
  [guide store k reps key]
  (mapv #(w/log-mean-exp
          (log-weights line-model [xs] line-observed guide store k %))
        (rng/split key reps)))

(defn bounds
  "Estimates of the K-sample bound of the guide on the line under the
   store, for K = 1, 10, 100 and 1000: {K {:mean :se}}."
  [guide store key]
  (into (sorted-map)
        (map (fn [[k reps] key]
               [k (estimate (bound-draws guide store k reps key))])
             [[1 2000] [10 800] [100 400] [1000 40]]
             (rng/split key 4))))

(def mean-field-bounds (bounds mean-field mean-field-fit (rng/key 17)))

(update-vals mean-field-bounds :mean)
;=> {1 -18.9 10 -18.57 100 -18.41 1000 -18.29} ; float64

;; book: bound-checks
(defn above?
  "Is the estimate a above b by more than five standard errors of their
   difference?"
  [a b]
  (let [{:keys [mean se]} (difference a b)] (> mean (* 5 se))))

(let [b mean-field-bounds
      {:keys [mean se]} (b 1)]
  [(< (abs (- mean (- line-log-evidence mean-field-kl))) (* 5 se))
   (above? (b 10) (b 1))
   (above? (b 100) (b 10))
   (every? (fn [{:keys [mean se]}] (< mean (+ line-log-evidence (* 5 se))))
           (vals b))])
;=> [true true true true]

;; book: vimco
(def vimco-fit
  (:store (vi/black-box-vimco line-model [xs] line-observed mean-field []
                              mean-field-fit (learn/adam {:lr 0.02})
                              {:iters 2000 :samples 10 :batched? true
                               :key (rng/key 18)})))

(sds vimco-fit)
;=> [0.1294 0.8165] ; float64

(map #(> (/ %1 %2) 1.5) (sds vimco-fit) (sds mean-field-fit))
;=> [true true]

;; book: vimco-figure
;; figure: figures/027-vimco.svg
(ridge-figure mean-field vimco-fit (rng/key 19))

;; book: vimco-bounds
(def vimco-bounds (bounds mean-field vimco-fit (rng/key 20)))

(update-vals vimco-bounds :mean)
;=> {1 -22.91 10 -18.25 100 -18.16 1000 -18.16} ; float64

(let [a mean-field-bounds
      b vimco-bounds
      {:keys [mean se]} (b 1000)]
  [(above? (a 1) (b 1)) (above? (b 10) (a 10)) (above? (b 100) (a 100))
   (< (abs (- mean line-log-evidence)) (* 5 se))])
;=> [true true true true]

;; book: solution-1
(def landing
  (update-vals estimators
               #(gradient-draws at-posterior % 500 20 (rng/key 21))))

(update-vals landing #(Math/sqrt (variance %)))
;=> {:score 11.01 :leave-one-out 0.0 :pathwise 1.113} ; float64

(defn sd-agrees?
  "Is the sample standard deviation of normal draws xs within five of its
   standard errors, sd/sqrt(2(n - 1)), of the expected sd?"
  [xs expected]
  (< (abs (- (Math/sqrt (variance xs)) expected))
     (* 5 expected (Math/sqrt (/ 1.0 (* 2 (dec (count xs))))))))

(let [per-draw (/ 1.0 (* (:sd exact) (Math/sqrt 20)))]
  [(sd-agrees? (:score landing) (* (abs log-evidence) per-draw))
   (every? #(< (abs %) 1e-3) (:leave-one-out landing))
   (sd-agrees? (:pathwise landing) per-draw)])
;=> [true true true]

;; book: solution-2
(def far-xs (mapv #(+ % 10.0) xs))

(def far-line (line-posterior far-xs line-ys))

(sds-and-correlation (:covariance far-line))
;=> {:sds [0.108 1.71] :correlation -0.983}

(mean-field-optimum far-line)
;=> {:sds [0.0201 0.3161] :gap 1.69}

(def far-fit
  (advi line-model [far-xs] line-observed mean-field mean-field-start
        {:lr 0.1 :key (rng/key 22)}))

(sds far-fit)
;=> [0.0203 0.3201] ; float64

(- (kl-from-gaussian far-line (moments far-fit))
   (:gap (mean-field-optimum far-line)))
;=> 0.0172 ; float64

(< 0.0 (- (kl-from-gaussian far-line (moments far-fit))
          (:gap (mean-field-optimum far-line)))
   0.3)
;=> true

;; book: solution-3
(def twin
  (gen []
    (let [x (trace :x (d/normal 0 3))]
      (trace :y (d/normal (array/abs x) 0.2))
      x)))

(def y-is-3 (cm/choicemap :y 3.0))

(def mode
  "The posterior's mode at positive x: the normal N(0, 3) prior times the
   N(3; x, 0.2) likelihood, a normal of precision 1/9 + 25."
  (let [precision (+ (/ 1.0 9.0) 25.0)]
    {:mu (/ (* 25.0 3.0) precision) :sd (Math/sqrt (/ 1.0 precision))}))

mode
;=> {:mu 2.9867 :sd 0.1996}

(def twin-log-evidence
  "log p(y = 3): twice the mass of one mode, log 2 + log N(3; 0, √9.04)."
  (+ (Math/log 2.0)
     (array/item (dc/log-prob (d/normal 0 (Math/sqrt 9.04)) 3.0))))

(def twin-fit
  (advi twin [] y-is-3 approx {:x-mu 0.5 :x-log-std 0.0} {:key (rng/key 23)}))

twin-fit
;=> {:x-mu 2.9845 :x-log-std -1.6094} ; float64

(let [{:keys [x-mu x-log-std]} twin-fit
      kl-to-mode (kl-normal (array/item x-mu) (Math/exp (array/item x-log-std))
                            (:mu mode) (:sd mode))
      {:keys [mean se]}
      (estimate (log-weights twin [] y-is-3 approx twin-fit 2000
                             (rng/key 24)))]
  [(< kl-to-mode 0.012)
   (< (abs (- mean (- twin-log-evidence (Math/log 2.0) kl-to-mode)))
      (* 5 se))])
;=> [true true]

;; book: solution-4
(def latents
  (gen [d]
    (dotimes [i d]
      (trace [:z i] (d/normal 0 1)))))

(def shifted
  (gen [d]
    (dotimes [i d]
      (trace [:z i] (d/normal (param :m) 1)))))

(defn shift-gradients
  "300 estimates of the ELBO's gradient in m at m = 1, from 10 draws each,
   for d latents."
  [d opts key]
  (mapv (fn [k]
          (let [g (vi/elbo-gradient latents [d] cm/EMPTY shifted [d] {:m 1.0}
                                    10 (assoc opts :key k))]
            (array/item (:m (:grads g)))))
        (rng/split key 300)))

(defn log-variance
  "The log of the sample variance of xs, with its standard error, from
   their kurtosis: sqrt((m4/v² - 1)/n), where v is their variance and m4
   their fourth central moment."
  [xs]
  (let [n (count xs)
        mean (/ (reduce + xs) n)
        moment (fn [p] (/ (reduce + (map #(Math/pow (- % mean) p) xs)) n))
        v (moment 2)]
    {:mean (Math/log v)
     :se (Math/sqrt (/ (- (/ (moment 4) (* v v)) 1.0) n))}))

(defn growth
  "How much the estimator's log variance grows from 5 latents to 50, with
   its standard error."
  [opts key]
  (let [[k1 k2] (rng/split key)]
    (difference (log-variance (shift-gradients 50 opts k2))
                (log-variance (shift-gradients 5 opts k1)))))

(def score-growth
  (growth {:baseline :leave-one-out :batched? true} (rng/key 25)))

(def pathwise-growth (growth {:estimator :reparam} (rng/key 26)))

(mapv #(Math/exp (:mean %)) [score-growth pathwise-growth])
;=> [101.86 7.99] ; float64

(mapv (fn [{:keys [mean se]} factor]
        (< (abs (- mean (Math/log factor))) (* 5 se)))
      [score-growth pathwise-growth] [100.0 10.0])
;=> [true true]

