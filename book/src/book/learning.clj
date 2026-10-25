;; The code of book/028-learning.md. Each region, from a `;; book:` line to
;; the next, is included there by name (gen.book: `clojure -M:test -m
;; gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.learning
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.array.linalg :as linalg]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.custom :as custom]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.diagnostics :as diag]
            [gen.inference.proposal :as prop]
            [gen.learning :as learn]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: model
(def model
  (gen []
    (let [x (trace :x (d/normal (param :x-mu) 1))]
      (trace :y (d/normal (+ (* (param :a) x) (param :b)) 1)))))

;; book: store
(def truth {:x-mu 1.5 :a -0.8 :b 2.0})

(def tr (gfi/simulate (dyn/with-params model truth) [] (rng/key 1)))

(cm/to-map (:choices tr))
;=> {:x 2.216 :y -0.320} ; float64

;; book: score
(defn log-density
  "log p(x, y) of a choice map under the store, by hand."
  [{:keys [x-mu a b]} choices]
  (let [x (cm/get-choice choices :x)]
    (+ (dc/log-prob (d/normal x-mu 1) x)
       (dc/log-prob (d/normal (+ (* a x) b) 1) (cm/get-choice choices :y)))))

(array/allclose? (:score tr) (log-density truth (:choices tr)))
;=> true

;; book: trace-store
(gfi/stores (:gen-fn tr))
;=> {:x-mu 1.5 :a -0.8 :b 2.0}

;; book: no-store
;; error: The parameter :x-mu has no value
#_(gfi/simulate model [] (rng/key 1))

;; book: choice-gradients
(def by-choice (grad/choice-gradients tr (sel/select :x :y)))

(cm/to-map (:gradients by-choice))
;=> {:x -0.278 :y 0.547} ; float64

;; book: choice-gradients-check
(let [{:keys [x-mu a b]} truth
      x (tr :x)
      r (- (tr :y) (+ (* a x) b))
      g (:gradients by-choice)]
  [(array/allclose? (cm/get-choice g :x) (+ (- x-mu x) (* a r)))
   (array/allclose? (cm/get-choice g :y) (- r))])
;=> [true true]

;; book: param-gradients
(def by-param (grad/param-gradients tr))

by-param
;=> {:x-mu 0.716 :a -1.213 :b -0.547} ; float64

;; book: param-gradients-check
(let [{:keys [x-mu a b]} truth
      x (tr :x)
      r (- (tr :y) (+ (* a x) b))]
  (mapv array/allclose? [(:x-mu by-param) (:a by-param) (:b by-param)]
        [(- x x-mu) (* r x) r]))
;=> [true true true]

;; book: dataset
(def dataset
  (mapv #(:choices (gfi/simulate (dyn/with-params model truth) [] %))
        (rng/split (rng/key 2) 100)))

;; book: mle
(defn column
  "The values at addr across data, an array."
  [data addr]
  (array/array (mapv #(array/item (cm/get-choice % addr)) data)))

(defn mle
  "The maximum-likelihood store for complete data: x-mu the mean of the
   x's, and a and b the least-squares line of the y's on the x's."
  [data]
  (let [x (column data :x)
        y (column data :y)
        dx (- x (array/mean x))
        a (/ (array/sum (* dx (- y (array/mean y)))) (array/sum (* dx dx)))]
    {:x-mu (array/mean x) :a a :b (- (array/mean y) (* a (array/mean x)))}))

(def exact (mle dataset))

exact
;=> {:x-mu 1.451 :a -0.662 :b 1.700} ; float64

;; book: traces-under
(defn traces-under
  "Traces of model under store, one for each example of data, generated
   with every choice constrained: each trace's score is its example's log
   density."
  [store data key]
  (mapv (fn [choices k]
          (:trace (gfi/generate (dyn/with-params model store) [] choices k)))
        data
        (rng/split key (count data))))

(defn objective
  "The mean score of traces: the log likelihood per example."
  [traces]
  (/ (reduce + (map #(array/item (:score %)) traces)) (count traces)))

(def start {:x-mu 0.0 :a 0.0 :b 0.0})

(objective (traces-under start dataset (rng/key 3)))
;=> -4.140 ; float64

;; book: gradient
(defn complete-gradient
  "The gradient of the objective on data at store: the mean of its
   traces' parameter gradients."
  [data]
  (fn [store key]
    (grad/mean (map grad/param-gradients (traces-under store data key)))))

(def g0 ((complete-gradient dataset) start (rng/key 4)))

(let [x (column dataset :x)
      y (column dataset :y)]
  (mapv array/allclose? [(:x-mu g0) (:a g0) (:b g0)]
        [(array/mean x) (array/mean (* x y)) (array/mean y)]))
;=> [true true true]

;; book: ascend
(defn ascend
  "The stores along steps of optimizer from store, each step along
   (gradient store key): a lazy sequence of steps + 1 stores."
  [optimizer store steps gradient key]
  (map first
       (reductions (fn [[store state] k]
                     (learn/step optimizer state store (gradient store k)))
                   [store (learn/init optimizer store)]
                   (rng/split key steps))))

;; book: fit
(def sgd-path
  (vec (ascend (learn/sgd {:lr 0.4}) start 150
               (complete-gradient dataset) (rng/key 5))))

(def fitted (peek sgd-path))

(defn near?
  "Is every parameter of store within tol of target's?"
  [store target tol]
  (every? (fn [[k v]] (< (abs (- (array/item (store k)) (array/item v))) tol))
          target))

(near? fitted exact 1e-4)
;=> true

(objective (traces-under fitted dataset (rng/key 6)))
;=> -2.623 ; float64

;; book: fit-figure
;; figure: figures/028-complete-fit.svg
(svg/plot [{:points (mapv vector
                          (array/->clj (column dataset :x))
                          (array/->clj (column dataset :y)))}
           {:curves [#(+ (* (:a truth) %) (:b truth))] :color "gray"}
           {:curves [#(+ (* (:a fitted) %) (:b fitted))]}])

;; book: one-step
(def s0 {:a 0.0 :b 0.0})

(def g {:a 1.0 :b 0.01})

(defn one-step
  "The store after one step of optimizer from s0 along g."
  [optimizer]
  (first (learn/step optimizer (learn/init optimizer s0) s0 g)))

(one-step (learn/sgd {:lr 0.1}))
;=> {:a 0.1 :b 0.001}

(one-step (learn/adam {:lr 0.1}))
;=> {:a 0.1 :b 0.0999999}

;; book: gradient-descent
(let [opt (learn/gradient-descent {:step-size-init 0.1 :step-size-beta 10})
      [s1 state] (learn/step opt (learn/init opt s0) s0 g)
      [s2 _] (learn/step opt state s1 g)]
  [(:a s1) (- (:a s2) (:a s1))])
;=> [0.1 0.0917]

;; book: curvature
(defn curvatures
  "The eigenvalues of the objective's curvature in (a, b), the matrix
   [[mean x², mean x] [mean x, 1]], the largest first (x-mu's is 1)."
  [data]
  (let [x (column data :x)
        xx (array/item (array/mean (* x x)))
        t (+ xx 1.0)
        det (- xx (array/item (array/square (array/mean x))))
        largest (/ (+ t (Math/sqrt (- (* t t) (* 4.0 det)))) 2.0)]
    [largest (/ det largest)]))

(curvatures dataset)
;=> [3.747 0.234] ; float64

(def lr-max (/ 2.0 (first (curvatures dataset))))

lr-max
;=> 0.534 ; float64

;; book: threshold
(defn error-after
  "How far a is from the MLE after steps of SGD at lr."
  [lr steps]
  (let [store (last (ascend (learn/sgd {:lr lr}) start steps
                            (complete-gradient dataset) (rng/key 7)))]
    (abs (- (array/item (:a store)) (array/item (:a exact))))))

[(< (error-after (* 0.95 lr-max) 200) 1e-3)
 (> (error-after (* 1.05 lr-max) 200) 1e3)]
;=> [true true]

;; book: adam
(def adam-path
  (vec (ascend (learn/adam {:lr 0.1}) start 150
               (complete-gradient dataset) (rng/key 8))))

(near? (peek adam-path) exact 1e-3)
;=> true

;; book: paths-figure
;; figure: figures/028-paths.svg
(let [ab (fn [path] (mapv (fn [s] [(:a s) (:b s)]) path))]
  (svg/plot [{:points (ab adam-path) :color "darkorange" :r 2}
             {:points (ab sgd-path) :color "steelblue" :r 2}
             {:points [[(:a exact) (:b exact)]] :r 4}]
            {:x-label "a" :y-label "b"}))

;; book: disc-model
(def disc-model
  (gen [x]
    (trace :y (d/normal (+ (* (param :a) x) (param :b)) 1))))

;; book: example
(defn example
  "One example of the dataset, drawn uniformly with key: [args
   constraints], as train's data-fn returns them."
  [key]
  (let [choices (dataset (array/item (rng/randint key 0 (count dataset))))]
    [[(cm/get-choice choices :x)]
     (cm/choicemap :y (cm/get-choice choices :y))]))

;; book: train
(def epochs
  (vec (reductions
        (fn [{:keys [store state]} key]
          (learn/train disc-model store (learn/sgd {:lr 0.05})
                       example grad/param-gradients
                       {:epochs 1 :epoch-size 100 :batches-per-epoch 10
                        :batch-size 10 :state state :key key}))
        {:store {:a 0.0 :b 0.0}}
        (rng/split (rng/key 9) 400))))

(:history (peek epochs))
;=> [-1.263] ; float64

;; book: history
(defn conditional-objective
  "The mean log density of the dataset's y's given their x's under store:
   what the history estimates, since the held-out examples are draws from
   the dataset."
  [{:keys [a b]}]
  (let [x (column dataset :x)]
    (array/item (array/mean (dc/log-prob (d/normal (+ (* a x) b) 1)
                                         (column dataset :y))))))

(def gaps
  "Each epoch's history less the objective at its store."
  (mapv (fn [{[h] :history store :store}] (- h (conditional-objective store)))
        (rest epochs)))

(let [{:keys [mean mcse-mean]} (diag/summary gaps)]
  (< (abs mean) (* 5 mcse-mean)))
;=> true

;; book: polyak
(defn settled
  "The parameter k's values after each epoch from the 100th on."
  [k]
  (mapv #(array/item (get-in % [:store k])) (subvec epochs 100)))

(def a-summary (diag/summary (settled :a)))

(def b-summary (diag/summary (settled :b)))

[(:sd a-summary) (:sd b-summary)]
;=> [0.052 0.054] ; float64

(mapv (fn [s k]
        (< (abs (- (:mean s) (array/item (k exact)))) (* 5 (:mcse-mean s))))
      [a-summary b-summary] [:a :b])
;=> [true true]

;; book: epochs-figure
;; figure: figures/028-epochs.svg
(svg/series (map #(get-in % [:store :a]) epochs)
            {:x-label "epoch" :y-label "a"})

;; book: mixture
(def point
  (gen []
    (let [z (trace :z (d/bernoulli (array/sigmoid (param :logit))))]
      (trace :y (d/normal (array/where z (param :mu1) (param :mu0)) 1)))))

(def mixture-truth {:mu0 -1.0 :mu1 2.0 :logit (Math/log (/ 0.3 0.7))})

(def ys
  (mapv #(:retval (gfi/simulate (dyn/with-params point mixture-truth) [] %))
        (rng/split (rng/key 10) 200)))

;; book: marginal
(defn log-marginal
  "log p(y) = log Σ_z p(z, y) under store: the two completions' log
   densities, assessed, and summed in exp space."
  [store y]
  (let [m (dyn/with-params point store)
        joint (fn [z] (:weight (gfi/assess m [] (cm/choicemap :z z :y y)
                                           (rng/key 0))))]
    (array/logaddexp (joint false) (joint true))))

(def store0 {:mu0 0.0 :mu1 1.0 :logit 0.0})

(log-marginal store0 (first ys))
;=> -1.171 ; float64

;; book: completions
(defn completions
  "Both completions of y under store, as traces, with their posterior
   probabilities p(z | y)."
  [store y key]
  (let [m (dyn/with-params point store)
        traces (mapv (fn [z k]
                       (:trace (gfi/generate m [] (cm/choicemap :z z :y y) k)))
                     [false true]
                     (rng/split key 2))
        scores (mapv #(array/item (:score %)) traces)
        total (array/item (array/logaddexp (scores 0) (scores 1)))]
    {:traces traces :probs (mapv #(Math/exp (- % total)) scores)}))

;; book: fisher
(let [y (first ys)
      {:keys [traces probs]} (completions store0 y (rng/key 11))
      fisher (grad/weighted-sum probs (map grad/param-gradients traces))
      direct ((array/grad #(log-marginal % y)) store0)]
  (every? #(array/allclose? (% fisher) (% direct)) [:mu0 :mu1 :logit]))
;=> true

;; book: weighted-gradient
(defn weighted-gradient
  "The mean over points of each point's completions' parameter gradients,
   weighted by the completions' normalized importance weights. cs holds k
   completions (choice maps) of each point in turn, generated by m as one
   batch."
  [m cs k key]
  (let [n (quot (count cs) k)
        {:keys [particles weights]}
        (batched/generate m [] cm/EMPTY (count cs) key {:per-particle cs})
        probs (array/softmax (array/reshape weights [n k]))
        grads (batched/map-particles particles
                                     (fn [t _] (grad/param-gradients t)) nil)]
    (array/tree-map #(/ (array/sum (* probs (array/reshape % [n k]))) n)
                    grads)))

(defn em-gradient
  "The gradient of the mixture at store along k completions of each point
   of cs (weighted-gradient)."
  [cs k]
  (fn [store key]
    (weighted-gradient (dyn/with-params point store) cs k key)))

(defn enumerated
  "Both completions of each of ys: :z false, then true."
  [ys]
  (vec (for [y ys z [false true]] (cm/choicemap :z z :y y))))

(defn marginal-gradient
  "The gradient of the mean log marginal likelihood of ys at store."
  [store]
  (grad/mean (map (fn [y] ((array/grad #(log-marginal % y)) store)) ys)))

(let [batch ((em-gradient (enumerated ys) 2) store0 (rng/key 12))
      direct (marginal-gradient store0)]
  (every? #(array/allclose? (% batch) (% direct)) [:mu0 :mu1 :logit]))
;=> true

;; book: em
(def em-path
  (vec (ascend (learn/sgd {:lr 2.0}) store0 300
               (em-gradient (enumerated ys) 2) (rng/key 13))))

(def em (peek em-path))

em
;=> {:mu0 -1.134 :mu1 1.913 :logit -0.670} ; float64

(every? #(< (abs (array/item %)) 1e-5) (vals (marginal-gradient em)))
;=> true

;; book: classic-em
(defn classic-em-step
  "One step of the textbook EM for the mixture, in closed form: each
   point's responsibility r = p(z = 1 | y), then the means weighted by
   them and their mean."
  [{:keys [mu0 mu1 logit]}]
  (let [y (array/array ys)
        r (array/sigmoid (+ logit (* 0.5 (- (array/square (- y mu0))
                                              (array/square (- y mu1))))))
        pi (array/mean r)]
    {:mu0 (array/item (/ (array/sum (* (- 1.0 r) y)) (array/sum (- 1.0 r))))
     :mu1 (array/item (/ (array/sum (* r y)) (array/sum r)))
     :logit (array/item (array/log (/ pi (- 1.0 pi))))}))

(def mle-mixture (nth (iterate classic-em-step store0) 500))

(near? em mle-mixture 1e-4)
;=> true

;; book: mixture-figure
;; figure: figures/028-mixture.svg
(svg/histogram [{:values ys :density #(array/exp (log-marginal em %))}]
               {:bins 30 :x-label "y"})

;; book: sampled
(defn sampled
  "k copies of each of ys's observation, completions whose z the batch's
   internal proposal, the prior, fills in."
  [ys k]
  (vec (for [y ys _ (range k)] (cm/choicemap :y y))))

(defn mc-em
  "Monte Carlo EM: the logit along steps of SGD, each along the gradient
   weighted over k sampled completions of every point."
  [k steps key]
  (mapv #(array/item (:logit %))
        (ascend (learn/sgd {:lr 2.0}) store0 steps
                (em-gradient (sampled ys k) k) key)))

(def mc-20-runs (mapv #(mc-em 20 300 %) (rng/split (rng/key 14) 4)))

(def mc-20 (first mc-20-runs))

(def mc-20-summary (diag/summary (mapv #(subvec % 100) mc-20-runs)))

[(:mean mc-20-summary) (:mcse-mean mc-20-summary) (:logit mle-mixture)]
;=> [-0.6906 0.00175 -0.6702] ; float64

(let [{:keys [mean mcse-mean]} mc-20-summary]
  (< mean (- (:logit mle-mixture) (* 5 mcse-mean))))
;=> true

;; book: mc-em-5
(def mc-5 (mc-em 5 300 (rng/key 15)))

;; the last logit's spread over ten runs (keys 101 to 110): 0.20
(< (peek mc-5) (- (:logit mle-mixture) (* 5 0.20)))
;=> true

;; book: em-figure
;; figure: figures/028-em.svg
(let [dots (fn [logits color]
             {:points (map-indexed vector logits) :color color :r 1.5})]
  (svg/plot [(dots mc-5 "steelblue")
             (dots mc-20 "darkorange")
             (dots (map :logit em-path) "black")]
            {:x-label "step" :y-label "logit"}))

;; book: no-sample-rate
(let [pi (array/sigmoid (:logit mle-mixture))]
  (mapv #(array/pow (- 1.0 pi) %) [20 5]))
;=> [0.00026 0.127] ; float64

;; book: bias
(defn mean-and-error
  "The mean of independent draws xs, and its standard error."
  [xs]
  (let [{:keys [mean sd]} (diag/summary xs)]
    [mean (/ sd (Math/sqrt (count xs)))]))

(def at-mle
  (let [gradient (em-gradient (sampled ys 5) 5)]
    (mapv #(array/item (:logit (gradient mle-mixture %)))
          (rng/split (rng/key 16) 100))))

(mean-and-error at-mle)
;=> [-0.0326 0.0014] ; float64

(let [[mean se] (mean-and-error at-mle)]
  (< mean (* -5 se)))
;=> true

;; book: mlp
(defn mlp
  "A network of one hidden layer of tanh units, applied to each element of
   the 1-d array xs: w1 and b1 the hidden layer's weights and biases, [h]
   each, and w2 ([h]) and b2 the output's."
  [{:keys [w1 b1 w2 b2]} xs]
  (+ (array/matmul (array/tanh (+ (array/outer xs w1) b1)) w2) b2))

(def net
  (custom/deterministic
   {:params [:w1 :b1 :w2 :b2]
    :apply (fn [[xs] params] (mlp params xs))}))

;; book: curve
(def curve
  (gen [xs]
    (let [mean (splice :net net xs)]
      (trace :ys (d/normal mean (array/exp (param :log-sigma)))))))

;; book: net-data
(def grid (array/linspace -3.0 3.0 200))

(def noisy (+ (array/sin grid) (* 0.1 (rng/normal (rng/key 17) [200]))))

(defn initial-store
  "Random weights for h hidden units (the output's scaled by 1/√h), zero
   biases, and log-sigma 0."
  [h key]
  (let [[k1 k2] (rng/split key)]
    {:w1 (rng/normal k1 [h]) :b1 (array/zeros [h])
     :w2 (/ (rng/normal k2 [h]) (Math/sqrt h)) :b2 0.0
     :log-sigma 0.0}))

;; book: net-train
(defn curve-gradient
  "The gradient of the log likelihood of the noisy data at store."
  [store key]
  (let [m (dyn/with-params curve store)]
    (grad/param-gradients
     (:trace (gfi/generate m [grid] (cm/choicemap :ys noisy) key)))))

(def trained
  (last (ascend (learn/adam {:lr 0.01}) (initial-store 16 (rng/key 18)) 3000
                curve-gradient (rng/key 19))))

;; book: net-check
(def fitted-mean (mlp trained grid))

(def rms (array/sqrt (array/mean (array/square (- noisy fitted-mean)))))

(def noise-rms
  (array/sqrt (array/mean (array/square (- noisy (array/sin grid))))))

[(array/exp (:log-sigma trained)) rms noise-rms]
;=> [0.0866 0.0871 0.0887] ; float64

;; book: net-check-bands
(def fit-error
  (array/sqrt (array/mean (array/square (- fitted-mean (array/sin grid))))))

fit-error
;=> 0.0191 ; float64

;; spreads over ten trainings, each with its own noise (keys 201 to 210)
;; and initial weights (keys 101 to 110): 0.00033 of sigma - rms, 0.0066
;; of sigma, 0.00073 of rms - noise-rms, 0.0060 of noise-rms - fit-error
(let [sigma (array/item (array/exp (:log-sigma trained)))]
  [(< (abs (- sigma (array/item rms))) (* 5 0.00033))
   (< (abs (- sigma 0.1)) (* 5 0.0066))
   (< (array/item (- rms noise-rms)) (* 5 0.00073))
   (> (array/item (- noise-rms fit-error)) (* 5 0.0060))])
;=> [true true true true]

;; book: net-figure
;; figure: figures/028-network.svg
(svg/plot [{:points (mapv vector (array/->clj grid) (array/->clj noisy))
            :r 2 :opacity 0.3}
           {:curves [#(mlp trained (array/array [%]))] :width 2}
           {:curves [#(Math/sin %)] :color "darkorange" :width 1}])

;; book: net-trace
(let [weights (select-keys trained [:w1 :b1 :w2 :b2])
      t (gfi/simulate (custom/with-params net weights) [grid] (rng/key 20))]
  [(cm/empty? (:choices t))
   (:score t)
   (array/allclose? (:retval t) fitted-mean)])
;=> [true 0.0 true]

;; book: kepler
(defn eccentric-anomaly
  "E with E - e·sin E = m (Kepler's equation), by Newton's method on host
   doubles."
  [m e]
  (loop [E m i 0]
    (if (< i 20)
      (recur (- E (/ (- E (* e (Math/sin E)) m) (- 1.0 (* e (Math/cos E)))))
             (inc i))
      E)))

(eccentric-anomaly 1.2 0.5)
;=> 1.6961

;; book: anomaly-plain
(def anomaly-plain
  (custom/deterministic
   {:apply (fn [[m]] (eccentric-anomaly (array/item m) 0.5))}))

(defn orbit
  "A mean anomaly m, its eccentric anomaly by the function anomaly, and a
   noisy observation of it."
  [anomaly]
  (gen []
    (let [m (trace :m (d/normal 1.0 0.5))]
      (trace :obs (d/normal (splice :E anomaly m) 0.1)))))

(defn m-gradient
  "The derivative in m of the log density of orbit's trace at m = 1.2 and
   obs = 1.9."
  [anomaly]
  (let [t (:trace (gfi/generate (orbit anomaly) []
                                (cm/choicemap :m 1.2 :obs 1.9) (rng/key 21)))]
    (cm/get-choice (:gradients (grad/choice-gradients t (sel/select :m)))
                   :m)))

;; book: anomaly-plain-error
;; error: Cannot read a value being differentiated on the host
#_(m-gradient anomaly-plain)

;; book: anomaly
(def anomaly
  (custom/deterministic
   {:apply (fn [[m]] (eccentric-anomaly (array/item m) 0.5))
    :gradient (fn [_ E retgrad]
                [(/ retgrad (- 1.0 (* 0.5 (array/cos E))))])}))

(let [E (eccentric-anomaly 1.2 0.5)
      dE (/ 1.0 (- 1.0 (* 0.5 (Math/cos E))))]
  (array/allclose? (m-gradient anomaly)
                   (+ (/ (- 1.0 1.2) 0.25) (* (/ (- 1.9 E) 0.01) dE))))
;=> true

;; book: anomaly-fd
(defn orbit-density
  "The log density of orbit at m and obs = 1.9."
  [m]
  (array/item (:weight (gfi/assess (orbit anomaly) []
                                   (cm/choicemap :m m :obs 1.9) (rng/key 0)))))

(let [h 1e-3]
  (/ (- (orbit-density (+ 1.2 h)) (orbit-density (- 1.2 h))) (* 2 h)))
;=> 18.39

;; book: proposal
(def proposal
  (gen [constraints]
    (when-not (cm/has-value? (cm/get-submap constraints :z))
      (let [y (cm/get-choice constraints :y)]
        (trace :z (d/bernoulli (array/sigmoid (+ (* (param :w) y)
                                                  (param :c)))))))))

(defn guided
  "The mixture under model-store, generating with its z drawn by the
   proposal under q-store (the internal-proposal override)."
  [model-store q-store]
  (prop/with-internal-proposal (dyn/with-params point model-store)
                               (dyn/with-params proposal q-store)))

;; book: posterior-logit
(defn posterior-logit
  "The exact posterior's logit, w·y + c, under the mixture's store: log
   p(z = 1, y) - log p(z = 0, y)."
  [{:keys [mu0 mu1 logit]}]
  {:w (- mu1 mu0) :c (- logit (/ (- (* mu1 mu1) (* mu0 mu0)) 2.0))})

(def best (posterior-logit mle-mixture))

best
;=> {:w 3.047 :c -1.857} ; float64

(let [y (first ys)
      {:keys [w c]} best]
  (array/allclose? (second (:probs (completions mle-mixture y (rng/key 22))))
                   (array/sigmoid (+ (* w y) c))))
;=> true

;; book: sleep
(def learned-model (dyn/with-params point mle-mixture))

(defn q-args
  "The proposal's arguments for a trace of the model: its observation, as
   constraints."
  [t]
  [(cm/choicemap :y (t :y))])

(defn sleep-gradient
  "The gradient at a proposal's store of its mean log density of the
   latent choices of 20 of model's own traces (lecture-batched)."
  [model]
  (fn [store key]
    (:grads (learn/lecture-batched model [] proposal store q-args 20 key))))

(def sleep-path
  (mapv (fn [s] [(array/item (:w s)) (array/item (:c s))])
        (ascend (learn/adam {:lr 0.1}) {:w 0.0 :c 0.0} 1500
                (sleep-gradient learned-model) (rng/key 23))))

;; book: sleep-check
(def trained-q
  (let [tail (subvec sleep-path 750)]
    {:w (:mean (diag/summary (mapv first tail)))
     :c (:mean (diag/summary (mapv second tail)))}))

trained-q
;=> {:w 3.10 :c -1.86} ; float64

;; the largest spread of the proposal's probability at a point of ys, over
;; ten trainings (keys 101 to 110): 0.0078
(let [y (array/array ys)
      prob (fn [{:keys [w c]}] (array/sigmoid (+ (* w y) c)))]
  (< (array/item (array/max (array/abs (- (prob trained-q) (prob best)))))
     (* 5 0.0078)))
;=> true

;; book: amortized
(defn ess-fractions
  "The effective sample size, as a fraction of k, of k importance samples
   of m given each of ys, generated as one batch: a fraction a point."
  [m ys k key]
  (let [n (count ys)
        {:keys [weights]} (batched/generate m [] cm/EMPTY (* n k) key
                                            {:per-particle (sampled ys k)})
        p (array/softmax (array/reshape weights [n k]))]
    (array/->clj (/ 1.0 (* k (array/sum (* p p) {:axis 1}))))))

(def new-ys
  (mapv #(:retval (gfi/simulate learned-model [] %))
        (rng/split (rng/key 24) 100)))

(def ess-prior
  (mean-and-error (ess-fractions learned-model new-ys 20 (rng/key 25))))

(def ess-trained
  (mean-and-error (ess-fractions (guided mle-mixture trained-q) new-ys 20
                                 (rng/key 26))))

[ess-prior ess-trained]
;=> [[0.619 0.02] [0.9999 2.3E-5]] ; float64

;; each point's fraction is an independent draw (its own y and samples),
;; so the standard error is mean-and-error's; the prior's means over 140
;; such sets of draws (keys 301 to 440, each split in three: the points,
;; the prior's samples, the proposal's) spread 0.022, against its 0.020
(let [[[prior prior-se] [trained trained-se]] [ess-prior ess-trained]]
  [(< (+ prior (* 5 prior-se)) 0.8) (> (- trained (* 5 trained-se)) 0.99)])
;=> [true true]

;; book: amortized-bias
(def at-mle-trained
  (let [m (guided mle-mixture trained-q)
        cs (sampled ys 5)]
    (mapv #(array/item (:logit (weighted-gradient m cs 5 %)))
          (rng/split (rng/key 16) 100))))

(let [[mean se] (mean-and-error at-mle-trained)]
  (< (abs mean) (* 5 se)))
;=> true

;; book: rws
(defn rws-gradient
  "Reweighted wake-sleep's gradients at the stores {:model :proposal}: the
   model's along k completions of every point drawn by the proposal (the
   wake phase), the proposal's along the model's own traces (the sleep
   phase)."
  [k]
  (let [cs (sampled ys k)]
    (fn [{model :model q :proposal} key]
      (let [[k-wake k-sleep] (rng/split key)
            sleep (sleep-gradient (dyn/with-params point model))]
        {:model (weighted-gradient (guided model q) cs k k-wake)
         :proposal (sleep q k-sleep)}))))

(defn wake-sleep
  "The model's stores along steps of Adam on both stores, from store0 and
   a proposal that knows nothing, with k completions of every point."
  [k steps key]
  (mapv :model
        (ascend (learn/adam {:lr 0.05})
                {:model store0 :proposal {:w 0.0 :c 0.0}}
                steps (rws-gradient k) key)))

(def rws (wake-sleep 5 600 (rng/key 27)))

;; book: rws-check
(defn second-half
  "The averages of the model's parameters over path's second half."
  [path]
  (let [tail (subvec path (quot (count path) 2))]
    (into {} (for [k [:mu0 :mu1 :logit]]
               [k (/ (reduce + (map #(array/item (k %)) tail))
                     (count tail))]))))

(second-half rws)
;=> {:mu0 -1.133 :mu1 1.911 :logit -0.671} ; float64

;; the spreads of these averages over ten runs (keys 101 to 110)
(def rws-spreads {:mu0 0.0045 :mu1 0.0092 :logit 0.0090})

(let [averages (second-half rws)]
  (every? (fn [[k spread]]
            (< (abs (- (averages k) (k mle-mixture))) (* 5 spread)))
          rws-spreads))
;=> true

;; book: solution-1
(def centered
  (let [m (array/mean (column dataset :x))]
    (mapv #(cm/choicemap :x (- (cm/get-choice % :x) m)
                         :y (cm/get-choice % :y))
          dataset)))

(curvatures centered)
;=> [1.0 0.876] ; float64

(let [path (ascend (learn/sgd {:lr 1.0}) start 10
                   (complete-gradient centered) (rng/key 28))]
  [(near? (last path) (mle centered) 1e-4)
   (near? (nth sgd-path 10) exact 1e-4)])
;=> [true false]

;; book: solution-2
(def bayes-line
  (gen [xs]
    (let [a (trace :a (d/normal 0 1))
          b (trace :b (d/normal 0 1))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* a x) b) 1))))))

(def few (subvec dataset 0 10))

(def few-ys
  (apply cm/choicemap
         (mapcat (fn [i c] [[:y i] (cm/get-choice c :y)]) (range) few)))

(defn map-step
  "One step of SGD on the choices a and b, along their gradient."
  [tr key]
  (let [{:keys [values gradients]}
        (grad/choice-gradients tr (sel/select :a :b))
        [moved _] (learn/step (learn/sgd {:lr 0.02}) nil
                              (cm/to-map values) (cm/to-map gradients))]
    (:trace (gfi/update tr (cm/from-map moved) key))))

(def map-trace
  (let [xs (mapv #(cm/get-choice % :x) few)
        start (cm/merge few-ys (cm/choicemap :a 0.0 :b 0.0))]
    (reduce map-step
            (:trace (gfi/generate bayes-line [xs] start (rng/key 29)))
            (rng/split (rng/key 30) 300))))

;; book: solution-2-check
(defn ridge
  "The MAP line under the N(0, 1) priors: (X^T X + I)^-1 X^T y, X's rows
   (x, 1)."
  [data]
  (let [X (array/array (mapv (fn [c] [(array/item (cm/get-choice c :x)) 1.0])
                             data))
        Xt (array/transpose X)
        precision (+ (array/eye 2) (array/matmul Xt X))]
    (array/->clj (array/matmul (linalg/inv precision)
                               (array/matmul Xt (column data :y))))))

(let [[a b] (ridge few)]
  [(near? {:a (map-trace :a) :b (map-trace :b)} {:a a :b b} 1e-4)
   (> (abs (- a (array/item (:a (mle few))))) 0.01)])
;=> [true true]

;; book: solution-3
(def untrained (initial-store 16 (rng/key 18)))

(def noise-only
  (last (ascend (learn/adam {:lr 0.05}) untrained 500
                #(select-keys (curve-gradient %1 %2) [:log-sigma])
                (rng/key 31))))

(def misfit
  (array/sqrt (array/mean (array/square (- noisy (mlp untrained grid))))))

[(array/exp (:log-sigma noise-only)) misfit]
;=> [0.757 0.757] ; float64

[(< (abs (array/item (- (array/exp (:log-sigma noise-only)) misfit))) 1e-3)
 (identical? (:w1 noise-only) (:w1 untrained))]
;=> [true true]

;; book: solution-4
(def coin
  (gen [constraints]
    (when-not (cm/has-value? (cm/get-submap constraints :z))
      (trace :z (d/bernoulli 0.5)))))

(def coin-em
  (let [cs (sampled ys 5)]
    (vec (ascend (learn/sgd {:lr 2.0}) store0 300
                 (fn [store key]
                   (weighted-gradient
                    (prop/with-internal-proposal (dyn/with-params point store)
                                                 coin)
                    cs 5 key))
                 (rng/key 32)))))

(def coin-gaps
  (let [averages (second-half coin-em)]
    (into {} (for [[k v] averages] [k (- v (k mle-mixture))]))))

coin-gaps
;=> {:mu0 -0.014 :mu1 -0.357 :logit 0.209} ; float64

;; the spread of either gap over ten runs (keys 101 to 110): 0.014
[(> (:logit coin-gaps) (* 5 0.014)) (< (:mu1 coin-gaps) (* -5 0.014))]
;=> [true true]
