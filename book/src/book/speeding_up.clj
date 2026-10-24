;; The code of book/029-speeding-up.md. Each region, from a `;; book:` line
;; to the next, is included there by name (gen.book: `clojure -M:test -m
;; gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.speeding-up
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as batched]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.dynamic.static :as static]
            [gen.gfi :as gfi]
            [gen.inference.importance :as imp]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

;; book: loop-model
(def loop-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          noise (trace :noise (d/gamma 1 1))
          prob-outlier (trace :prob-outlier (d/uniform 0 1))]
      (vec (map-indexed
            (fn [i x]
              (let [[mu sd] (if (trace [:data i :is-outlier]
                                       (d/bernoulli prob-outlier))
                              [0.0 10.0]
                              [(+ (* x slope) intercept) noise])]
                (trace [:data i :y] (d/normal mu sd))))
            xs)))))

;; book: block-update
(defn sweep
  "MH over tr, one move for each selection of blocks in turn, each
   resimulating the selected choices from their prior."
  [tr blocks key]
  (reduce (fn [tr [block k]] (:trace (mh/mh tr block k)))
          tr
          (map vector blocks (rng/split key (count blocks)))))

(defn block-resimulation-update
  "One sweep of block resimulation over tr: the line's parameters, then
   each point's outlier flag in turn, then the outlier probability."
  [tr key]
  (let [[xs] (:args tr)]
    (sweep tr
           (concat [(sel/select :noise :slope :intercept)]
                   (for [i (range (count xs))]
                     (sel/select [:data i :is-outlier]))
                   [(sel/select :prob-outlier)])
           key)))

;; book: inference
(defn observations
  "The choice map of the y values ys."
  [ys]
  (reduce (fn [c [i y]] (cm/set-choice c [:data i :y] y))
          cm/EMPTY
          (map-indexed vector ys)))

(defn block-resimulation-inference
  "A trace of model on xs given ys, after iters sweeps of block
   resimulation."
  [model xs ys iters key]
  (let [[k-init k-sweeps] (rng/split key)
        tr (:trace (gfi/generate model [xs] (observations ys) k-init))]
    (reduce (fn [tr k] (gfi/eval-trace! (block-resimulation-update tr k)))
            tr
            (rng/split k-sweeps iters))))

;; book: counted
(defn counted
  "gf with its regenerate transition wrapped to count, in the atom visits,
   the points it visits: the sites at a path ending in :y. The transition
   it wraps is the handler's own, so gf means what it meant."
  [gf visits]
  (let [regenerate (dyn/site-transitions :regenerate)]
    (dyn/with-transitions gf
      {:regenerate (fn [state p dist k]
                     (when (= :y (peek p)) (swap! visits inc))
                     (regenerate state p dist k))})))

(defn sweep-visits
  "The points one sweep visits, on n random points, of the model that
   (make visits) builds. The sweep is block-resimulation-update, or the
   function of a trace and a key given."
  [make n key & [sweep-fn]]
  (let [visits (atom 0)
        [k-x k-y k-init k-sweep] (rng/split key 4)
        xs (array/->clj (rng/uniform k-x [n]))
        ys (array/->clj (rng/uniform k-y [n]))
        tr (:trace (gfi/generate (make visits) [xs] (observations ys) k-init))]
    ((or sweep-fn block-resimulation-update) tr k-sweep)
    @visits))

;; book: loop-visits
(def sizes [1 3 7 10 30 70 100])

(def loop-visits
  (mapv #(sweep-visits (fn [v] (counted loop-model v)) % (rng/key 1)) sizes))

loop-visits
;=> [3 15 63 120 960 5040 10200]

(= loop-visits (mapv #(* % (+ % 2)) sizes))
;=> true

;; book: point
(def point
  (gen [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          [mu sd] (if outlier? [0.0 10.0] [(+ (* x slope) intercept) noise])]
      (trace :y (d/normal mu sd)))))

;; book: with-map
(defn with-map
  "The robust regression with its points made by a Map of kernel."
  [kernel]
  (let [points (comb/map kernel)]
    (gen [xs]
      (let [slope (trace :slope (d/normal 0 2))
            intercept (trace :intercept (d/normal 0 2))
            noise (trace :noise (d/gamma 1 1))
            prob-outlier (trace :prob-outlier (d/uniform 0 1))
            n (count xs)]
        (splice :data points xs (repeat n prob-outlier) (repeat n noise)
                (repeat n slope) (repeat n intercept))))))

(def model-with-map (with-map point))

;; book: same-model
(def xs5 [0.0 1.0 2.0 3.0 4.0])

(def five (gfi/simulate loop-model [xs5] (rng/key 2)))

(take 4 (cm/addresses (:choices five)))
;=> [[:data 0 :is-outlier] [:data 0 :y] [:data 1 :is-outlier] [:data 1 :y]]

(= (cm/addresses (:choices five))
   (cm/addresses (:choices (gfi/simulate model-with-map [xs5] (rng/key 2)))))
;=> true

(array/allclose?
 (:weight (gfi/assess loop-model [xs5] (:choices five) (rng/key 3)))
 (:weight (gfi/assess model-with-map [xs5] (:choices five) (rng/key 3))))
;=> true

;; book: map-visits
(def map-visits
  (mapv #(sweep-visits (fn [v] (with-map (counted point v))) % (rng/key 1))
        sizes))

(= map-visits loop-visits)
;=> true

;; book: static-model
(defn static-with-map
  "with-map's model in the static language: the same body, flagged."
  [kernel]
  (let [points (comb/map kernel)]
    (gen ^:static [xs]
      (let [slope (trace :slope (d/normal 0 2))
            intercept (trace :intercept (d/normal 0 2))
            noise (trace :noise (d/gamma 1 1))
            prob-outlier (trace :prob-outlier (d/uniform 0 1))
            n (count xs)]
        (splice :data points xs (repeat n prob-outlier) (repeat n noise)
                (repeat n slope) (repeat n intercept))))))

(def static-model-with-map (static-with-map point))

;; book: graph
(def graph (static/ir static-model-with-map))

(mapv :kind (:nodes graph))
;=> [:trace :trace :trace :trace :compute :splice]

(mapv :inputs (:nodes graph))
;=> [[] [] [] [] [xs] [xs n prob-outlier noise slope intercept]]

;; book: drawing
(defn drawing
  "The nodes and edges of a static generative function's graph, for
   svg/dag: an argument as text, a choice as an ellipse, a call as a box, a
   computation as a diamond and the return value as a circle, each node
   named by its address or by the name it binds."
  [gf]
  (let [{:keys [args nodes return]} (static/ir gf)
        shapes {:trace :ellipse :splice :box :compute :diamond}
        id (fn [n] (or (:addr n) (first (:names n))))
        binder (into {} (for [n nodes s (:names n)] [s (id n)]))
        source #(get binder % %)]
    {:nodes (concat (for [a (apply concat (:fixed args))] {:id a :shape :text})
                    (for [n nodes] {:id (id n) :shape (shapes (:kind n))})
                    [{:id 'return :shape :circle}])
     :edges (concat (for [n nodes s (:inputs n)] [(source s) (id n)])
                    (for [s (:inputs return)] [(source s) 'return]))}))

(str/includes? (static/dot static-model-with-map) "\":noise\" -> \":data\"")
;=> true

;; book: graph-figure
;; figure: figures/029-graph.svg
(let [{:keys [nodes edges]} (drawing static-model-with-map)]
  (svg/dag nodes edges))

;; book: not-static
;; error: Syntax error macroexpanding gen
#_(gen ^:static [xs]
    (let [slope (trace :slope (d/normal 0 2))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (* slope x) 1)))
      slope))

;; book: lint
(static/could-be-static model-with-map)
;=> {:static? true}

(let [{:keys [static? reason]} (static/could-be-static loop-model)]
  [static? (str/includes? reason "calls trace, which may run any number")])
;=> [false true]

;; book: static-visits
(def static-visits
  (mapv #(sweep-visits (fn [v] (static-with-map (counted point v))) %
                       (rng/key 1))
        sizes))

static-visits
;=> [3 9 21 30 90 210 300]

(= static-visits (mapv #(* 3 %) sizes))
;=> true

;; book: visits-figure
;; figure: figures/029-visits.svg
(svg/plot [{:curves [#(* % (+ % 2)) #(* 3 %)] :color "gray" :width 1}
           {:points (map vector sizes loop-visits) :color "darkorange"}
           {:points (map vector sizes static-visits) :color "steelblue"}]
          {:x-range [0 102] :x-label "number of data points"
           :y-label "points visited in one sweep"})

;; book: revisited
(defn revisited
  "The indices of the points whose subtraces differ between traces a and b
   of a model whose points are a Map at :data."
  [a b]
  (let [pa (trace/subtrace a :data)
        pb (trace/subtrace b :data)]
    (vec (remove #(identical? (trace/subtrace pa %) (trace/subtrace pb %))
                 (range (count (:retval pa)))))))

(defn five-points
  "A trace of model on xs5, given five y values."
  [model]
  (:trace (gfi/generate model [xs5] (observations [1 2 3 4 5]) (rng/key 4))))

(defn flag-moved
  "The points that regenerating point 2's outlier flag revisits, in a
   trace of model."
  [model]
  (let [tr (five-points model)
        moved (gfi/regenerate tr (sel/select [:data 2 :is-outlier])
                              (rng/key 5))]
    (revisited tr (:trace moved))))

[(flag-moved model-with-map) (flag-moved static-model-with-map)]
;=> [[0 1 2 3 4] [2]]

;; book: same-chain
(defn same-choices?
  "Do traces a and b hold the same choices, exactly, and scores that agree
   up to rounding?"
  [a b]
  (let [ca (:choices a)
        cb (:choices b)]
    (and (= (cm/addresses ca) (cm/addresses cb))
         (every? true? (map array/equal? (cm/values ca) (cm/values cb)))
         (array/allclose? (:score a) (:score b)))))

(def xs10 (array/->clj (rng/uniform (rng/key 6) [10])))

(def ys10 (mapv #(+ (* 2.0 %) 0.5) xs10))

(defn chain
  "model's trace after 20 sweeps on (xs10, ys10), from one key."
  [model]
  (block-resimulation-inference model xs10 ys10 20 (rng/key 7)))

(same-choices? (chain model-with-map) (chain static-model-with-map))
;=> true

;; book: static-point
(static/could-be-static point)
;=> {:static? true}

(def static-point
  (gen ^:static [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          [mu sd] (if outlier? [0.0 10.0] [(+ (* x slope) intercept) noise])]
      (trace :y (d/normal mu sd)))))

(def fully-static-model (static-with-map static-point))

[(sweep-visits (fn [v] (static-with-map (counted static-point v))) 30
               (rng/key 1))
 (same-choices? (chain static-model-with-map) (chain fully-static-model))]
;=> [90 true]

;; book: static-assess
(let [args [1.0 0.1 0.5 -1.0 2.0]
      choices (cm/choicemap :is-outlier false :y 1.2)]
  (array/allclose? (:weight (gfi/assess static-point args choices (rng/key 0)))
                   (:weight (gfi/assess point args choices (rng/key 0)))))
;=> true

;; book: synthetic
(def xs (mapv #(- (* 10.0 (/ % 49.0)) 5.0) (range 50)))

(def true-line
  (cm/choicemap :slope -1.0 :intercept 2.0 :noise 0.5 :prob-outlier 0.1))

(def ys
  (-> (:retval (:trace (gfi/generate loop-model [xs] true-line (rng/key 8))))
      (assoc 44 13.0 46 14.0)))

;; book: fit
(def sweeps
  (let [[k-init k-sweeps] (rng/split (rng/key 9))
        start (:trace (gfi/generate static-model-with-map [xs]
                                    (observations ys) k-init))]
    (reductions (fn [tr k] (gfi/eval-trace! (block-resimulation-update tr k)))
                start
                (rng/split k-sweeps 500))))

(def fitted (last sweeps))

[(fitted :slope) (fitted :intercept)]
;=> [-0.944 1.998] ; float64

[(fitted [:data 44 :is-outlier]) (fitted [:data 46 :is-outlier])]
;=> [true true] ; float64

(count (filter true? (map #(not= (%1 :slope) (%2 :slope))
                          sweeps (rest sweeps))))
;=> 7 ; float64

;; book: fit-figure
;; figure: figures/029-fit.svg
(let [outlier? #(fitted [:data % :is-outlier])
      at (fn [i] [(xs i) (ys i)])
      line #(+ (* (fitted :slope) %) (fitted :intercept))]
  (svg/plot [{:curves [line] :color "black" :width 1}
             {:points (map at (remove outlier? (range 50)))}
             {:points (map at (filter outlier? (range 50)))
              :color "darkorange"}]
            {:x-range [-5.5 5.5] :y-range [-5 16]}))

;; book: batched-prior
(def normal-mean
  (gen [n]
    (let [mu (trace :mu (d/normal 0 2))]
      (dotimes [i n]
        (trace [:y i] (d/normal mu 1)))
      mu)))

(def prior-particles (batched/simulate normal-mean [1] 20000 (rng/key 10)))

[(batched/batched? prior-particles)
 (array/shape (batched/choice prior-particles :mu))
 (array/shape (batched/scores prior-particles))]
;=> [true [20000] [20000]]

;; book: prior-moments
(defn moments
  "The mean and the variance of the host numbers vs."
  [vs]
  (let [n (count vs)
        mean (/ (reduce + vs) n)]
    [mean (/ (reduce + (map #(* (- % mean) (- % mean)) vs)) (dec n))]))

(let [n 20000
      [m-mu v-mu] (moments (array/->clj (batched/choice prior-particles :mu)))
      [_ v-y] (moments (array/->clj (batched/choice prior-particles [:y 0])))
      ;; five standard errors of a mean, and of a normal sample's variance
      band-mean (fn [v] (* 5 (Math/sqrt (/ v n))))
      band-var (fn [v] (* 5 v (Math/sqrt (/ 2.0 (dec n)))))]
  [(< (abs m-mu) (band-mean 4.0))
   (< (abs (- v-mu 4.0)) (band-var 4.0))
   (< (abs (- v-y 5.0)) (band-var 5.0))])
;=> [true true true]

;; book: particle-is-trace
(let [t (first (batched/traces (batched/take prior-particles [7])))
      mus (batched/choice prior-particles :mu)
      scores (batched/scores prior-particles)]
  [(array/allclose? (t :mu) (array/take mus 7))
   (array/allclose? (:score t) (array/take scores 7))])
;=> [true true]

;; book: exact
(def ys5 [1.2 0.4 2.1 1.7 0.9])

(def observed
  (reduce (fn [c [i y]] (cm/set-choice c [:y i] y))
          cm/EMPTY
          (map-indexed vector ys5)))

(defn exact
  "normal-mean's posterior of mu given the observations ys, normal with
   this mean and sd, and its log marginal likelihood: the ys are normal
   with mean 0 and covariance I + 4J, J a matrix of ones."
  [ys]
  (let [n (count ys)
        s (reduce + ys)
        ss (reduce + (map #(* % %) ys))
        precision (+ 0.25 n)]
    {:mean (/ s precision)
     :sd (Math/sqrt (/ 1.0 precision))
     :log-ml (- (* -0.5 n (Math/log (* 2.0 Math/PI)))
                (* 0.5 (Math/log (+ 1.0 (* 4.0 n))))
                (* 0.5 (- ss (/ (* 4.0 s s) (+ 1.0 (* 4.0 n))))))}))

(def posterior (exact ys5))

posterior
;=> {:mean 1.2 :sd 0.4364 :log-ml -7.192}

;; book: batched-is
(defn agrees?
  "Are importance sampling's estimates of mu's posterior mean and of the
   log marginal likelihood, from n samples, within five standard errors of
   the exact ones?"
  [{:keys [estimate log-ml-estimate ess]} n]
  (let [{:keys [mean sd log-ml]} posterior]
    [(< (abs (- estimate mean)) (* 5 (/ sd (Math/sqrt ess))))
     (< (abs (- log-ml-estimate log-ml))
        (* 5 (Math/sqrt (- (/ 1.0 ess) (/ 1.0 n)))))]))

(defn sampled
  "Importance sampling's estimate of mu's posterior mean from 10000
   samples, with opts beside the key."
  [opts]
  (imp/estimate #(% :mu) normal-mean [5] observed 10000
                (merge {:key (rng/key 11)} opts)))

(def one-at-a-time (sampled {}))

(def all-at-once (sampled {:batched? true}))

[(agrees? one-at-a-time 10000) (agrees? all-at-once 10000)]
;=> [[true true] [true true]]

(== (:estimate one-at-a-time) (:estimate all-at-once))
;=> false

;; book: per-particle
(def grid [0.0 0.5 1.0 1.5 2.0])

(def scored
  (batched/generate normal-mean [5] observed 5 (rng/key 12)
                    {:per-particle (mapv #(cm/choicemap :mu %) grid)}))

(defn joint
  "The log density of normal-mean's observations with mu, by the scalar
   handler."
  [mu]
  (:weight (gfi/generate normal-mean [5]
                         (cm/merge observed (cm/choicemap :mu mu))
                         (rng/key 0))))

(array/allclose? (:weights scored) (array/array (mapv joint grid)))
;=> true

;; book: fallback
(def flag-particles (batched/simulate model-with-map [xs5] 100 (rng/key 13)))

[(batched/batched? flag-particles)
 (:message (batched/fallback-reason flag-particles))]
;=> [false "Cannot branch on a batched value: it differs across particles"]

;; book: where-point
(def where-point
  (gen [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          mu (array/where outlier? 0.0 (+ (* x slope) intercept))
          sd (array/where outlier? 10.0 noise)]
      (trace :y (d/normal mu sd)))))

(def where-model (static-with-map where-point))

(def where-particles (batched/simulate where-model [xs5] 100 (rng/key 13)))

(batched/batched? where-particles)
;=> true

(array/allclose? (:score five)
                 (:weight (gfi/assess where-model [xs5] (:choices five)
                                      (rng/key 3))))
;=> true

;; book: wrong
(defn wrong
  "The number of particles of ps whose batched score is not the scalar
   handler's score of their own choices."
  [ps]
  (count (remove true? (map array/allclose?
                            (map :score (batched/traces ps))
                            (array/->clj (batched/scores ps))))))

(wrong where-particles)
;=> 0

;; book: unread
(defn inlier-spread
  "The standard deviation of a point, given whether it is an inlier: a
   branch outside gen, unguarded."
  [inlier? noise]
  (if inlier? noise 10.0))

(def unread-point
  (gen [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          mu (array/where outlier? 0.0 (+ (* x slope) intercept))
          sd (inlier-spread (array/logical-not outlier?) noise)]
      (trace :y (d/normal mu sd)))))

(let [ps (batched/simulate (static-with-map unread-point) [xs5] 100
                           (rng/key 13))]
  [(batched/batched? ps)
   (str/starts-with? (:message (batched/fallback-reason ps))
                     "A batched comparison was never read")])
;=> [false true]

;; book: helper
(defn spread
  "The standard deviation of a point: a branch outside gen, unguarded."
  [outlier? noise]
  (if outlier? 10.0 noise))

(def helper-point
  (gen [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          mu (array/where outlier? 0.0 (+ (* x slope) intercept))]
      (trace :y (d/normal mu (spread outlier? noise))))))

(def helper-model (static-with-map helper-point))

(def helper-particles (batched/simulate helper-model [xs5] 100 (rng/key 13)))

batched/*spot-checks*
;=> 8

[(batched/batched? helper-particles)
 (str/starts-with? (:message (batched/fallback-reason helper-particles))
                   "The batched run disagrees with the scalar handler")]
;=> [false true]

;; book: unchecked
(def unchecked
  (binding [batched/*spot-checks* 0]
    (batched/simulate helper-model [xs5] 100 (rng/key 13))))

[(batched/batched? unchecked) (wrong unchecked)]
;=> [true 81] ; float64

;; book: guarded
(dyn/defn-guarded guarded-spread
  "spread, with a gen body's guards."
  [outlier? noise]
  (if outlier? 10.0 noise))

(def guarded-point
  (gen [x prob-outlier noise slope intercept]
    (let [outlier? (trace :is-outlier (d/bernoulli prob-outlier))
          mu (array/where outlier? 0.0 (+ (* x slope) intercept))]
      (trace :y (d/normal mu (guarded-spread outlier? noise))))))

(let [ps (binding [batched/*spot-checks* 0]
           (batched/simulate (static-with-map guarded-point) [xs5] 100
                             (rng/key 13)))]
  [(batched/batched? ps) (:message (batched/fallback-reason ps))])
;=> [false "Cannot branch on a batched value: it differs across particles"]

;; book: many-chains
(def starts
  (let [{:keys [mean sd]} posterior]
    (mapv #(+ mean (* sd %)) (array/->clj (rng/normal (rng/key 14) [2000])))))

(def chains
  (:particles (batched/generate normal-mean [5] observed 2000 (rng/key 15)
                                {:per-particle (mapv #(cm/choicemap :mu %)
                                                     starts)})))

(def moved
  (reduce (fn [ps k] (:particles (mh/mh-particles ps (sel/select :mu) k)))
          chains
          (rng/split (rng/key 16) 20)))

(batched/batched? moved)
;=> true

;; book: stationary
(defn posterior-spread?
  "Are the particles' values of mu distributed as the posterior: their
   mean and sd within five standard errors of the exact ones?"
  [ps]
  (let [k (batched/size ps)
        {:keys [mean sd]} posterior
        [m v] (moments (array/->clj (batched/choice ps :mu)))]
    [(< (abs (- m mean)) (* 5 (/ sd (Math/sqrt k))))
     (< (abs (- (Math/sqrt v) sd)) (* 5 (/ sd (Math/sqrt (* 2 (dec k))))))]))

(posterior-spread? moved)
;=> [true true]

;; book: chains-figure
;; figure: figures/029-chains.svg
(let [{:keys [mean sd]} posterior]
  (svg/histogram [{:values (array/->clj (batched/choice moved :mu))
                   :density #(array/exp (dc/log-prob (d/normal mean sd) %))}]
                 {:x-label "mu"}))

;; book: blind
(def blind
  (reduce (fn [ps k]
            (:particles (batched/regenerate ps [5] [diff/no-change]
                                            (sel/select :mu) k)))
          chains
          (rng/split (rng/key 16) 20)))

(posterior-spread? blind)
;=> [false false]

;; book: compile
(defn log-joint
  "The log density of normal-mean's observations with mu, as a function of
   mu alone, which compile takes."
  [mu]
  (:weight (gfi/assess normal-mean [5] (cm/merge observed (cm/choicemap :mu mu))
                       (rng/key 0))))

(def compiled-log-joint (array/compile log-joint))

(array/allclose? (compiled-log-joint 1.0) (log-joint 1.0))
;=> true

;; book: compile-key
;; error: compile: the arguments must be arrays
#_((array/compile (fn [k] (rng/normal k []))) (rng/key 17))

;; book: compiled-hmc
(def start (:trace (gfi/generate normal-mean [5] observed (rng/key 18))))

(defn hmc-move
  "One HMC move of start's mu, L = 10 steps of 0.1, with opts."
  [opts]
  (mcmc/hmc start (sel/select :mu) (merge {:L 10 :eps 0.1} opts) (rng/key 19)))

(def plain (hmc-move {}))

(def fused (hmc-move {:compile? true}))

(:accepted? plain)
;=> true ; float64

[(= (:accepted? plain) (:accepted? fused))
 (array/allclose? ((:trace plain) :mu) ((:trace fused) :mu))]
;=> [true true]

;; book: solution-1
(defn split-update
  "block-resimulation-update with the line's three parameters moved one at
   a time."
  [tr key]
  (let [[xs] (:args tr)]
    (sweep tr
           (concat (map sel/select [:noise :slope :intercept])
                   (for [i (range (count xs))]
                     (sel/select [:data i :is-outlier]))
                   [(sel/select :prob-outlier)])
           key)))

[(sweep-visits (fn [v] (counted loop-model v)) 10 (rng/key 1) split-update)
 (sweep-visits (fn [v] (static-with-map (counted point v))) 10 (rng/key 1)
               split-update)]
;=> [140 50]

;; book: solution-2
(let [tr (five-points static-model-with-map)
      after #(:trace (gfi/regenerate tr % (rng/key 5)))]
  (mapv #(revisited tr (after %))
        [(sel/select :noise :slope :intercept)
         (sel/select [:data 2 :is-outlier])
         (sel/select :prob-outlier)]))
;=> [[0 1 2 3 4] [2] [0 1 2 3 4]]

(let [tr (five-points static-model-with-map)]
  (revisited tr (:trace (gfi/update tr (cm/choicemap [:data 2 :y] 3.5)
                                    (rng/key 5)))))
;=> [2]

;; book: solution-3
(def where-loop-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 2))
          noise (trace :noise (d/gamma 1 1))
          prob-outlier (trace :prob-outlier (d/uniform 0 1))]
      (vec (map-indexed
            (fn [i x]
              (let [outlier? (trace [:data i :is-outlier]
                                    (d/bernoulli prob-outlier))
                    mu (array/where outlier? 0.0 (+ (* x slope) intercept))
                    sd (array/where outlier? 10.0 noise)]
                (trace [:data i :y] (d/normal mu sd))))
            xs)))))

(let [batched? #(batched/batched? (batched/simulate % [xs5] 100 (rng/key 20)))]
  [(batched? loop-model)
   (batched? where-loop-model)
   (array/allclose? (:score five)
                    (:weight (gfi/assess where-loop-model [xs5] (:choices five)
                                         (rng/key 3))))])
;=> [false true true]

;; book: solution-4
(defn rare
  "helper-model's 100 particles with the outlier probability given as
   0.998, under checks spot checks."
  [checks]
  (binding [batched/*spot-checks* checks]
    (:particles (batched/generate helper-model [xs5]
                                  (cm/choicemap :prob-outlier 0.998)
                                  100 (rng/key 20)))))

(let [sampled (rare 8)
      every (rare :all)]
  [(batched/batched? sampled) (wrong sampled) (batched/batched? every)])
;=> [true 1 false] ; float64
