(ns gen.inference.enumerative-test
  "Enumerative inference (Gen.jl's enumerative_inference, choice_vol_grid)
   and exact proposals, against answers derived by hand: a discrete
   posterior, a conjugate normal, and a state-space model whose discrete
   latent is summed out step by step (GenGDT's hand enumeration). Batched,
   both are their scalar forms point by point."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.enumerative :as enum]
            [gen.inference.importance :as imp]
            [gen.inference.particle-filter :as pf]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

(defn- log ^double [x] (Math/log (double x)))

(defn- close-to [x y] (< (ts/abs (- (ts/number x) (ts/number y))) (ts/tol 1e-12)))

;; ---------------------------------------------------------------------------
;; A discrete posterior: a ~ Bern(0.3), b | a ~ Cat, obs | b ~ Bern
;; ---------------------------------------------------------------------------

(def model
  (gen []
    (let [a (trace :a (d/bernoulli 0.3))
          b (trace :b (d/categorical (if a [0.6 0.3 0.1] [0.2 0.3 0.5])))]
      (trace :obs (d/bernoulli (nth [0.1 0.5 0.9] b))))))

(def observations (cm/choicemap :obs true))

;; p(a, b, obs = true), by hand: a = false is 0.7 · [0.2 0.3 0.5] · [0.1 0.5 0.9]
(def joint {[false 0] 0.014 [false 1] 0.105 [false 2] 0.315
            [true 0] 0.018 [true 1] 0.045 [true 2] 0.027})
(def evidence 0.524)

(deftest enumerative-inference-is-exact-on-a-discrete-grid
  (let [grid (enum/choice-vol-grid [:a [false true]] [:b [0 1 2]])
        {:keys [traces log-normalized-weights log-ml-estimate] :as result}
        (enum/enumerative-inference model [] observations grid {:key (rng/key 1)})]
    (is (= 6 (count grid) (count traces)))
    (is (= [[false 0] [false 1] [false 2] [true 0] [true 1] [true 2]]
           (mapv #(ts/host [(% :a) (% :b)]) traces))
        "row-major: the last spec varies fastest")
    (is (every? zero? (map second grid)) "discrete points have log-volume 0")
    (is (close-to log-ml-estimate (log evidence)))
    (is (every? true? (map (fn [tr lw] (close-to lw (log (/ (joint (ts/host [(tr :a) (tr :b)])) evidence))))
                           traces log-normalized-weights)))
    (testing "marginals"
      (let [pa (enum/marginal result :a) pb (enum/marginal result :b)]
        (is (close-to (pa true) (/ 0.09 evidence)))
        (is (close-to (pa false) (/ 0.434 evidence)))
        (is (= #{0 1 2} (set (keys pb))))
        (is (close-to (pb 2) (/ 0.342 evidence)))))
    (testing "an observed choice in the grid is an error"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"sets the observed choice"
                            (enum/enumerative-inference model [] observations
                                                        (enum/choice-vol-grid [:obs [true]] [:a [true]])))))
    (testing "an impossible observation is an error, as importance sampling's"
      (let [never (gen [] (trace :a (d/bernoulli 0.5)) (trace :obs (d/bernoulli 0.0)))]
        (is (thrown? clojure.lang.ExceptionInfo
                     (enum/enumerative-inference never [] observations (enum/choice-vol-grid [:a [false true]]))))))))

(deftest grids-follow-gen-jl
  (let [values (fn [grid addr] (mapv #(ts/host (cm/get-choice (first %) addr)) grid))
        vols (fn [grid] (mapv second grid))]
    (testing "a continuous choice: the intervals between the endpoints, anchored"
      (let [g (enum/choice-vol-grid [:x [0.0 1.0 3.0] :continuous])]
        (is (= [0.5 2.0] (values g :x)))
        (is (= [0.0 (log 2.0)] (vols g))))
      (is (= [0.0 1.0] (values (enum/choice-vol-grid [:x [0.0 1.0 3.0] :continuous] {:anchor :left}) :x)))
      (is (= [1.0 3.0] (values (enum/choice-vol-grid [:x [0.0 1.0 3.0] :continuous] {:anchor :right}) :x))))
    (testing "several dimensions: the boxes of the endpoints' product, as arrays"
      (let [g (enum/choice-vol-grid [:z [[0.0 1.0 2.0] [0.0 2.0]] :continuous 2])]
        (is (= [[0.5 1.0] [1.5 1.0]] (values g :z)))
        (is (= [(log 2.0) (log 2.0)] (vols g)))))
    (testing "mixed specs: volumes add, the last spec varies fastest, paths work"
      (let [g (enum/choice-vol-grid [[:p 0] [1 2]] [:x [0.0 0.5 2.0] :continuous])]
        (is (= [1 1 2 2] (values g [:p 0])))
        (is (= [0.25 1.25 0.25 1.25] (values g :x)))
        (is (= [(log 0.5) (log 1.5) (log 0.5) (log 1.5)] (vols g)))))
    (testing "bad specs are errors"
      (doseq [specs [[[:x [1.0 0.0] :continuous]]
                     [[:x [0.0] :continuous]]
                     [[:x [[0.0 1.0]] :continuous 2]]
                     [[:x [1 2] :ordinal]]
                     [[:x []]]
                     [[:x [1]] [:x [2]]]
                     [[:x [0.0 1.0] :continuous] {:anchor :centre}]
                     []]]
        (is (thrown? clojure.lang.ExceptionInfo (apply enum/choice-vol-grid specs)) (pr-str specs))))))

(deftest a-continuous-grid-approaches-the-conjugate-answer
  ;; mu ~ N(0, 1), y ~ N(mu, 0.5), y = 1: the posterior is N(0.8, 0.2) and
  ;; p(y) = N(1; 0, √1.25). The midpoint rule's relative error is about
  ;; h²/24 · |f''/f| ≈ 1e-4/24 · 25 ≈ 1e-4 at h = 0.01 (the tails past ±4
  ;; are 7 posterior sds away), so 1e-4 on f64 and 1e-3 on f32 backends
  (let [m (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/normal mu 0.5))))
        grid (enum/choice-vol-grid [:mu (mapv #(- (* 0.01 %) 4.0) (range 801)) :continuous])
        {:keys [traces log-normalized-weights log-ml-estimate]}
        (enum/enumerative-inference m [] (cm/choicemap :y 1.0) grid {:key (rng/key 2)})
        tol (if (ts/f32?) 1e-3 1e-4)
        mean (reduce + (map #(* (Math/exp %2) (ts/number (%1 :mu))) traces log-normalized-weights))]
    (is (< (ts/abs (- log-ml-estimate (ts/number (dc/log-prob (d/normal 0 (Math/sqrt 1.25)) 1.0)))) tol))
    (is (< (ts/abs (- mean 0.8)) tol))))

;; ---------------------------------------------------------------------------
;; Batched enumeration: the grid's points as the particles of one run
;; ---------------------------------------------------------------------------

(def batchable
  "model, with where and take for its if and nth, so that its particles
   run batched"
  (gen []
    (let [a (trace :a (d/bernoulli 0.3))
          b (trace :b (d/categorical (array/where a (array/array [0.6 0.3 0.1]) (array/array [0.2 0.3 0.5]))))]
      (trace :obs (d/bernoulli (array/take (array/array [0.1 0.5 0.9]) b))))))

(def discrete-grid (enum/choice-vol-grid [:a [false true]] [:b [0 1 2]]))

(defn- same-choices?
  "Do the traces t and u hold the same choices (to the backend's
   precision)?"
  [t u]
  (let [ps (cm/addresses (:choices t))]
    (and (= ps (cm/addresses (:choices u))) (every? #(ts/close? (t %) (u %)) ps))))

(defn- same-results?
  "Are two enumerative-inference results the same point by point: the
   traces' choices, the normalized weights, and log-ml-estimate?"
  [r s]
  (and (= (count (:log-normalized-weights r)) (count (:log-normalized-weights s)))
       (every? true? (map same-choices? (enum/traces r) (enum/traces s)))
       (every? true? (map laws/close? (:log-normalized-weights r) (:log-normalized-weights s)))
       (laws/close? (:log-ml-estimate r) (:log-ml-estimate s))))

(def kernel (gen [m] (trace :y (d/normal m 1))))

(def doubled (gen [m] (trace :y (d/normal (* 2.0 m) 1))))

(def drift-step (gen [t s v] (let [x (trace :x (d/normal (+ s v) 1))] (trace :y (d/normal x 0.5)) x)))

(def mu-grid (enum/choice-vol-grid [:mu (mapv #(- (* 0.1 %) 3.0) (range 61)) :continuous]))

(def enumeration-cases
  "[label model args observations grid batched?], batched? whether the
   particles run batched (nil: either)."
  [[:discrete batchable [] observations discrete-grid true]
   [:branching model [] observations discrete-grid false]
   [:continuous (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/normal mu 0.5)))) [] (cm/choicemap :y 1.0)
    mu-grid true]
   [:boxes (gen [] (let [z (trace :z (d/mvnormal [0.0 0.0] [[1.0 0.5] [0.5 1.0]]))] (trace :y (d/normal (array/sum z) 0.5))))
    [] (cm/choicemap :y 0.8)
    (enum/choice-vol-grid [:z [(mapv #(- (* 0.5 %) 2.0) (range 9)) (mapv #(- (* 0.5 %) 2.0) (range 9))] :continuous 2]) true]
   [:map (gen [n] (let [mu (trace :mu (d/normal 0 1))] (splice :obs (comb/map kernel) (vec (repeat n mu))))) [3]
    (cm/choicemap [:obs 0 :y] 0.3 [:obs 1 :y] 1.2 [:obs 2 :y] -0.4) mu-grid true]
   [:unfold (gen [n] (let [v (trace :drift (d/normal 0 1))] (splice :path (comb/unfold drift-step) n 0.0 v))) [3]
    (cm/choicemap [:path 0 :x] 0.5 [:path 0 :y] 0.4 [:path 1 :x] 1.1 [:path 1 :y] 1.3 [:path 2 :x] 1.4 [:path 2 :y] 1.2)
    (enum/choice-vol-grid [:drift (mapv #(- (* 0.1 %) 3.0) (range 61)) :continuous]) true]
   [:switch (gen [] (let [mu (trace :mu (d/normal 0 1))] (splice :s (comb/switch kernel doubled) 1 mu)))
    [] (cm/choicemap [:s :y] 0.7) mu-grid true]
   ;; the Switch's index a grid value, so per particle: each branch runs on
   ;; its own points, the shared constraint in each (by partition,
   ;; genclj-7tqw)
   [:switch-index (gen [] (let [mu (trace :mu (d/normal 0 1))
                                k (trace :k (d/categorical [0.5 0.5]))]
                            (splice :s (comb/switch kernel doubled) k mu)))
    [] (cm/choicemap [:s :y] 0.7)
    (enum/choice-vol-grid [:mu (mapv #(- (* 0.1 %) 3.0) (range 61)) :continuous] [:k [0 1]]) true]])

(deftest batched-enumeration-is-the-scalar-loop
  ;; every choice given, so point by point the same: a model that branches
  ;; on a grid value falls back to one point at a time
  (doseq [[label m args obs grid batched?] enumeration-cases]
    (let [scalar (enum/enumerative-inference m args obs grid {:key (rng/key 20)})
          batched (enum/enumerative-inference m args obs grid {:key (rng/key 20) :batched? true})]
      (is (= [:batches :log-ml-estimate :log-normalized-weights] (sort (keys batched))) (str label))
      (is (same-results? scalar batched) (str label))
      (when (some? batched?)
        (is (= batched? (every? b/batched? (:batches batched))) (str label))))))

(deftest batched-enumeration-is-exact-on-a-discrete-grid
  (let [{:keys [batches log-normalized-weights log-ml-estimate] :as result}
        (enum/enumerative-inference batchable [] observations discrete-grid {:key (rng/key 1) :batched? true})
        traces (enum/traces result)]
    (is (every? b/batched? batches))
    (is (= [[false 0] [false 1] [false 2] [true 0] [true 1] [true 2]] (mapv #(ts/host [(% :a) (% :b)]) traces)))
    (is (close-to log-ml-estimate (log evidence)))
    (is (every? true? (map (fn [tr lw] (close-to lw (log (/ (joint (ts/host [(tr :a) (tr :b)])) evidence))))
                           traces log-normalized-weights)))
    (let [pa (enum/marginal result :a) pb (enum/marginal result :b)]
      (is (close-to (pa true) (/ 0.09 evidence)))
      (is (= #{0 1 2} (set (keys pb))))
      (is (close-to (pb 2) (/ 0.342 evidence))))))

(deftest large-grids-go-in-chunks
  ;; every choice given, chunks change nothing (bit for bit on the JVM
  ;; backend, whose kernels act per element)
  (let [same? (fn [x y] (if (ts/jvm?) (= x y) (laws/close? x y)))
        check (fn [m obs grid one k]
                (let [r (enum/enumerative-inference m [] obs grid {:key (rng/key 21) :batched? true :chunk-size k})]
                  (is (= (long (Math/ceil (/ (count grid) k))) (count (:batches r))) (str k))
                  (is (every? true? (map same? (:log-normalized-weights one) (:log-normalized-weights r))) (str k))
                  (is (same? (:log-ml-estimate one) (:log-ml-estimate r)) (str k))
                  (is (every? true? (map same-choices? (enum/traces one) (enum/traces r))) (str k))))]
    (let [one (enum/enumerative-inference batchable [] observations discrete-grid {:key (rng/key 21) :batched? true})]
      (doseq [k [1 2 4 5 6 7]] (check batchable observations discrete-grid one k)))
    (testing "on a continuous grid"
      (let [[_ m _ obs grid] (nth enumeration-cases 2)
            one (enum/enumerative-inference m [] obs grid {:key (rng/key 21) :batched? true})]
        (doseq [k [10 32]] (check m obs grid one k))))))

(deftest batched-enumeration-draws-the-choices-the-grid-leaves-out
  ;; the grid over :a alone: each point's :b is drawn, and its weight is
  ;; the density of :a and of the observation given the :b drawn
  (doseq [k [nil 1]]
    (let [{:keys [log-normalized-weights log-ml-estimate] :as r}
          (enum/enumerative-inference batchable [] observations (enum/choice-vol-grid [:a [false true]])
                                      {:key (rng/key 22) :batched? true :chunk-size k})]
      (is (every? true? (map (fn [tr lw] (laws/close? (+ lw log-ml-estimate) (gfi/project tr (sel/select :a :obs))))
                             (enum/traces r) log-normalized-weights)))))
  (testing "each chunk with a key of its own"
    (let [m (gen [] (let [k (trace :k (d/categorical [0.25 0.25 0.25 0.25]))] (trace :x (d/normal 0 1)) k))
          r (enum/enumerative-inference m [] cm/EMPTY (enum/choice-vol-grid [:k [0 1 2 3]])
                                        {:key (rng/key 23) :batched? true :chunk-size 1})]
      (is (= 4 (count (distinct (map #(ts/host (% :x)) (enum/traces r)))))))))

(deftest batched-enumeration-is-loud-too
  (doseq [k [0 -1 1.5 :all]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #":chunk-size"
                          (enum/enumerative-inference batchable [] observations discrete-grid {:batched? true :chunk-size k}))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"sets the observed choice"
                        (enum/enumerative-inference batchable [] observations (enum/choice-vol-grid [:obs [true]] [:a [true]])
                                                    {:batched? true}))))

;; ---------------------------------------------------------------------------
;; Exact proposals
;; ---------------------------------------------------------------------------

(def exact (enum/with-exact-proposal model [:a :b]))

(defn- chi-square
  "Pearson's statistic of observed states (a seq) against probabilities (a
   map from state)."
  [states probabilities]
  (let [n (count states) observed (frequencies states)]
    (reduce + (map (fn [[s p]] (let [e (* n p) o (get observed s 0)] (/ (* (- o e) (- o e)) e))) probabilities))))

;; P(χ² > 30) ≈ 1.5e-5 for 5 degrees of freedom
(def chi-square-5 30.0)

(deftest the-generate-weight-is-the-exact-marginal-likelihood
  (doseq [k (range 5)]
    (let [{tr :trace w :weight} (gfi/generate exact [] observations (rng/key k))]
      (is (close-to w (log evidence)) "the same weight whatever is drawn")
      (is (= true (ts/host (tr :obs))))
      (is (close-to (:score tr) (log (joint (ts/host [(tr :a) (tr :b)])))) "the model's own density")))
  (testing "inside importance sampling, every weight is exact, so one sample suffices"
    (is (close-to (:log-ml-estimate (imp/importance-sampling exact [] observations 1 {:key (rng/key 3)}))
                  (log evidence)))))

(deftest calls-are-the-model's
  (let [inner (gen [] (trace :x (d/normal 0 1)))
        egf (enum/with-exact-proposal (gen [] (let [a (trace :a (d/bernoulli 0.3))] (splice :inner inner) a)) [:a])
        tr (gfi/simulate egf [] (rng/key 4))]
    (is (= (ts/host (tr [:inner :x])) (ts/host (tr :inner))))
    (is (= inner (:gen-fn (trace/subtrace tr :inner))))))

(deftest unobserved-choices-keep-the-forward-proposal
  ;; enumerating :b alone, with :a drawn from its prior: the weight is
  ;; log p(obs | a) = log Σ_b p(b | a) p(obs | b), 0.62 for a = false and
  ;; 0.30 for a = true, whose prior mean is p(obs) (unbiased)
  (let [exact-b (enum/with-exact-proposal model [:b])]
    (doseq [k (range 8)]
      (let [{tr :trace w :weight} (gfi/generate exact-b [] observations (rng/key k))]
        (is (close-to w (log (if (ts/host (tr :a)) 0.30 0.62))))))
    (is (close-to (+ (* 0.7 0.62) (* 0.3 0.30)) evidence)))
  (testing "a continuous unobserved choice: z ~ Bern(0.3), x ~ N(±1, 1), y ~ N(x, 0.5)"
    ;; the weight is log p(x, y) - log p(x) = log N(y; x, 0.5), whatever z is drawn
    (let [m (gen [] (let [z (trace :z (d/bernoulli 0.3))
                          x (trace :x (d/normal (if z 1.0 -1.0) 1.0))]
                      (trace :y (d/normal x 0.5))))
          g (enum/with-exact-proposal m [:z])]
      (doseq [k (range 8)]
        (let [{tr :trace w :weight} (gfi/generate g [] (cm/choicemap :y 0.4) (rng/key k))]
          (is (close-to w (dc/log-prob (d/normal (tr :x) 0.5) 0.4))))))))

(deftest regenerating-the-enumerated-choices-is-gibbs
  (let [tr (:trace (gfi/generate exact [] observations (rng/key 4)))]
    (doseq [k (range 5)]
      (let [{w :weight} (gfi/regenerate tr (sel/select :a :b) (rng/key k))]
        (is (close-to w 0.0))))
    (testing "project of the observation is the exact log marginal"
      (is (close-to (gfi/project tr (sel/select :obs)) (log evidence)))
      (is (close-to (gfi/project tr sel/all) (:score tr)))
      (is (close-to (gfi/project tr sel/none) 0.0)))))

(deftest batched-runs-are-the-same-proposal
  ;; the same code under the batch tracer: each particle's weight is exact
  (let [ws (array/->clj (:weights (b/generate exact [] observations 50 (rng/key 12))))]
    (is (every? #(close-to % (log evidence)) ws)))
  (let [m (gen [] (let [z (trace :z (d/bernoulli 0.3))
                        x (trace :x (d/normal (array/where z 1.0 -1.0) 1.0))]
                    (trace :y (d/normal x 0.5))))
        n 2000
        {ps :particles ws :weights} (b/generate (enum/with-exact-proposal m [:z]) [] (cm/choicemap :y 0.4) n (rng/key 13))
        trs (b/traces ps)
        ;; P(z | x) = 0.3 N(x; 1, 1) / (0.3 N(x; 1, 1) + 0.7 N(x; -1, 1))
        pz (mapv #(let [x (ts/number (% :x))
                        l1 (* 0.3 (Math/exp (* -0.5 (* (- x 1.0) (- x 1.0)))))
                        l0 (* 0.7 (Math/exp (* -0.5 (* (+ x 1.0) (+ x 1.0)))))]
                    (/ l1 (+ l1 l0)))
                 trs)
        excess (reduce + (map #(- (if (ts/host (%1 :z)) 1.0 0.0) %2) trs pz))]
    (is (every? true? (map #(close-to %2 (dc/log-prob (d/normal (%1 :x) 0.5) 0.4)) trs (array/->clj ws))))
    (is (< (ts/abs excess) (* 5 (Math/sqrt (reduce + (map #(* % (- 1.0 %)) pz)))))
        "z is drawn from its exact conditional, within 5 standard errors")))

(deftest ^:slow exact-proposals-draw-from-the-posterior
  (let [posterior (update-vals joint #(/ % evidence))
        tr (:trace (gfi/generate exact [] observations (rng/key 4)))]
    (testing "generate draws the enumerated choices from the posterior"
      (let [states (map #(let [t (:trace (gfi/generate exact [] observations %))] (ts/host [(t :a) (t :b)]))
                        (rng/split (rng/key 10) 4000))]
        (is (< (chi-square states posterior) chi-square-5))))
    (testing "so regenerate on them is a Gibbs move to the posterior"
      (let [states (map #(let [t (:trace (gfi/regenerate tr (sel/select :a :b) %))] (ts/host [(t :a) (t :b)]))
                        (rng/split (rng/key 11) 4000))]
        (is (< (chi-square states posterior) chi-square-5))))))

;; ---------------------------------------------------------------------------
;; A state-space model with a discrete latent per step (GenGDT's case)
;; ---------------------------------------------------------------------------

;; Each step, an option comes to mind (h ~ Bern(q)); without it, action 1
;; is impossible. y reports the action and the state; the state moves with
;; the action alone, so h can be summed out step by step.
(def step
  (gen [t s q]
    (let [h (trace :h (d/bernoulli q))
          act (trace :a (d/categorical (if h [0.4 0.6] [1.0 0.0])))]
      (trace :y (d/normal (+ act (* 0.3 s)) 1.0))
      (+ s act))))

(def chain (comb/unfold (enum/with-exact-proposal step [:h])))

(defn- step-observations [tr n]
  (apply cm/choicemap (mapcat (fn [t] [[t :a] (ts/host (tr [t :a])) [t :y] (ts/host (tr [t :y]))]) (range n))))

(defn- exact-log-likelihood
  "log p(a, y for every step | q), h summed out by hand."
  [tr n q]
  (loop [t 0 s 0.0 ll 0.0]
    (if (= t n)
      ll
      (let [act (long (ts/host (tr [t :a]))) y (ts/number (tr [t :y]))
            p-a (+ (* q (nth [0.4 0.6] act)) (* (- 1.0 q) (nth [1.0 0.0] act)))]
        (recur (inc t) (+ s act)
               (+ ll (log p-a) (ts/number (dc/log-prob (d/normal (+ act (* 0.3 s)) 1.0) y))))))))

(deftest a-state-space-likelihood-is-exact
  (let [n 12 q 0.6
        tr (gfi/simulate (comb/unfold step) [n 0.0 q] (rng/key 5))
        obs (step-observations tr n)
        exact-ll (exact-log-likelihood tr n q)]
    (is (some #(= 1 (ts/host (tr [% :a]))) (range n)) "some steps take the action that needs h")
    (doseq [k (range 3)]
      (is (close-to (:weight (gfi/generate chain [n 0.0 q] obs (rng/key k))) exact-ll)))
    (testing "a particle filter's estimate is exact with any number of particles"
      (let [obs-at (fn [t] (cm/choicemap [t :a] (ts/host (tr [t :a])) [t :y] (ts/host (tr [t :y]))))
            run (fn [n-particles key]
                  (let [ks (rng/split key n)]
                    (pf/log-ml-estimate
                     (reduce (fn [state t]
                               (pf/step state [(inc t) 0.0 q] [diff/unknown-change diff/no-change diff/no-change]
                                        (obs-at t) {:key (ks t)}))
                             (pf/initialize chain [1 0.0 q] (obs-at 0) n-particles {:key (ks 0)})
                             (range 1 n)))))]
        (is (close-to (run 1 (rng/key 6)) exact-ll))
        (is (close-to (run 7 (rng/key 7)) exact-ll))))
    (testing "the likelihood as a function of q, as GenGDT's grid over parameters"
      (doseq [q' [0.2 0.9]]
        (is (close-to (:weight (gfi/generate chain [n 0.0 q'] obs (rng/key 8))) (exact-log-likelihood tr n q')))))))

(def learning-step
  "step, but the state moves only when the option came to mind: the next
   state depends on the enumerated choice."
  (gen [t s q]
    (let [h (trace :h (d/bernoulli q))
          act (trace :a (d/categorical (if h [0.4 0.6] [1.0 0.0])))]
      (trace :y (d/normal (+ act (* 0.3 s)) 1.0))
      (if h (+ s 1.0) s))))

(deftest an-unfold-is-exact-only-when-the-state-is-free-of-the-latents
  (let [n 12 q 0.6
        tr (gfi/simulate (comb/unfold learning-step) [n 0.0 q] (rng/key 5))
        obs (step-observations tr n)
        weights (fn [gf] (set (map #(ts/number (:weight (gfi/generate gf [n 0.0 q] obs (rng/key %)))) (range 4))))]
    (testing "when the state depends on h, later steps condition on the h drawn: an estimate, which varies with the key"
      (is (< 1 (count (weights (comb/unfold (enum/with-exact-proposal learning-step [:h])))))))
    (testing ":check-retval? makes that an error"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"return value depends on the enumerated choices"
                            (gfi/generate (comb/unfold (enum/with-exact-proposal learning-step [:h] {:check-retval? true}))
                                          [n 0.0 q] obs (rng/key 1)))))
    (testing "and passes a state built from observed choices, whose weight is exact under every key"
      (let [tr (gfi/simulate (comb/unfold step) [n 0.0 q] (rng/key 5))
            ws (set (map #(ts/number (:weight (gfi/generate (comb/unfold (enum/with-exact-proposal step [:h] {:check-retval? true}))
                                                             [n 0.0 q] (step-observations tr n) (rng/key %))))
                         (range 4)))]
        (is (= 1 (count ws)))
        (is (close-to (first ws) (exact-log-likelihood tr n q)))))))

;; ---------------------------------------------------------------------------
;; Batched exact proposals: the enumeration in one batched run
;; ---------------------------------------------------------------------------

(def continuous-latent
  (gen [m] (let [z (trace :z (d/bernoulli 0.3))
                 x (trace :x (d/normal (array/where z m (- m)) 1.0))]
             (trace :y (d/normal x 0.5)))))

(def digits
  "Three digits, observed through their number: 1000 assignments."
  (gen []
    (let [ds (mapv #(trace [:d %] (d/categorical (array/array (vec (repeat 10 0.1))))) (range 3))]
      (trace :y (d/normal (reduce + (map * ds [100.0 10.0 1.0])) 50.0)))))

(deftest batched-enumerations-are-the-scalar-ones
  (doseq [[label m args addrs obs] [[:discrete batchable [] [:a :b] observations]
                                    [:branching model [] [:a :b] observations]
                                    [:continuous-latent continuous-latent [1.0] [:z] (cm/choicemap :y 0.4)]
                                    [:digits digits [] [[:d 0] [:d 1] [:d 2]] (cm/choicemap :y 427.0)]]
          :let [scalar (enum/with-exact-proposal m addrs)
                batched (enum/with-exact-proposal m addrs {:batched? true})]
          k (range 3)]
    (let [{t :trace w :weight} (gfi/generate scalar args obs (rng/key k))
          {t' :trace w' :weight} (gfi/generate batched args obs (rng/key k))
          s (sel/from-choicemap obs)]
      (is (identical? batched (:gen-fn t')) (str label))
      (is (laws/close? w w') (str label))
      (is (same-choices? t t') (str label))
      (is (laws/close? (gfi/project t s) (gfi/project t' s)) (str label " project"))
      (is (laws/close? 0.0 (:weight (gfi/regenerate t' (apply sel/select addrs) (rng/key 9)))) (str label " regenerate")))))

(deftest a-batched-enumeration-inside-a-batched-run-is-the-run-s
  ;; one assignment at a time under the run's tracer: the failure of a
  ;; nested batched run must not make the run fall back
  (let [{ps :particles ws :weights}
        (b/generate (enum/with-exact-proposal batchable [:a :b] {:batched? true}) [] observations 50 (rng/key 12))]
    (is (b/batched? ps))
    (is (every? #(close-to % (log evidence)) (array/->clj ws)))))

(deftest batched-enumerations-are-as-loud
  (let [obs (cm/choicemap :y 0.0)
        gen-err (fn [m addrs re] (is (thrown-with-msg? clojure.lang.ExceptionInfo re
                                                       (gfi/generate (enum/with-exact-proposal m addrs {:batched? true})
                                                                     [] obs (rng/key 1)))))]
    ;; the forward run's z false, so that true makes a choice more
    (gen-err (gen [] (when (trace :z (d/bernoulli 1e-9)) (trace :w (d/normal 0 1))) (trace :y (d/normal 0 1)))
             [:z] #"structure depends")
    ;; a choice the other assignment leaves unvisited
    (gen-err (gen [] (let [z (trace :z (d/bernoulli 0.5))] (trace (if z :w :v) (d/normal 0 1)) (trace :y (d/normal 0 1))))
             [:z] #"structure depends")
    (gen-err (gen [] (let [n (trace :n (d/uniform-discrete 1 3))] (trace :k (d/uniform-discrete 0 n)) (trace :y (d/normal 0 1))))
             [:n :k] #"support of the choice at \[:k\] depends"))
  (testing ":check-retval? enumerates one assignment at a time, to read the return values"
    ;; learning-step with where for its ifs, so that its enumeration batches
    (let [n 12 q 0.6
          learning (gen [t s q]
                     (let [h (trace :h (d/bernoulli q))
                           act (trace :a (d/categorical (array/where h (array/array [0.4 0.6]) (array/array [1.0 0.0]))))]
                       (trace :y (d/normal (+ act (* 0.3 s)) 1.0))
                       (array/where h (+ s 1.0) s)))
          tr (gfi/simulate (comb/unfold learning) [n 0.0 q] (rng/key 5))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"return value depends on the enumerated choices"
                            (gfi/generate (comb/unfold (enum/with-exact-proposal learning [:h] {:check-retval? true :batched? true}))
                                          [n 0.0 q] (step-observations tr n) (rng/key 1)))))))

;; ---------------------------------------------------------------------------
;; The GFI laws (the general set: the internal proposal is not forward sampling)
;; ---------------------------------------------------------------------------

(defn- law-models-with
  "The law models, their exact proposals made with opts."
  [opts]
  [[:discrete (enum/with-exact-proposal model [:a :b] opts) [] []]
   [:discrete-partly (enum/with-exact-proposal model [:b] opts) [] []]
   [:continuous-latent (enum/with-exact-proposal
                         (gen [m] (let [z (trace :z (d/bernoulli 0.3))
                                        x (trace :x (d/normal (if z m (- m)) 1.0))]
                                    (trace :y (d/normal x 0.5))))
                         [:z] opts)
    [1.0] [[2.0]]]
   [:unfold (comb/unfold (enum/with-exact-proposal step [:h] opts)) [3 0.0 0.6] [[4 0.0 0.6] [2 0.0 0.3]]]
   ;; an update to more arguments makes fresh choices, which the exact
   ;; proposal's generate samples given the enumerated one (genclj-nqsm:
   ;; update-under, shared with with-internal-proposal; fewer arguments
   ;; would leave an unselected choice to regenerate unvisited)
   [:fresh-choices-beside-an-enumerated-one
    (enum/with-exact-proposal
      (gen [n] (let [k (trace :k (d/bernoulli 0.3))]
                 (doseq [i (range n)] (trace (keyword (str "x" i)) (d/normal (if k 1.0 -1.0) 1.0)))))
      [:k] opts)
    [1] [[2] [3]]]])

(def law-models (law-models-with nil))

(def batched-law-models
  "law-models enumerating in one batched run (their models branch with if,
   so it falls back to one assignment at a time), and models whose
   enumerations run batched."
  (into (law-models-with {:batched? true})
        [[:discrete-batched (enum/with-exact-proposal batchable [:a :b] {:batched? true}) [] []]
         [:continuous-latent-batched (enum/with-exact-proposal continuous-latent [:z] {:batched? true}) [1.0] [[2.0]]]]))

(laws/deflaws exact-proposal law-models 60 laws/general)
(laws/deflaws batched-exact-proposal batched-law-models 60 laws/general)

(deftest updates-stay-the-wrapper-s
  ;; genclj-nqsm: update-under keeps the trace the wrapper's, with fresh
  ;; choices or without (planted: the trace left as gf's)
  (let [m (second (last law-models))
        tr (gfi/simulate m [1] (rng/key 3))]
    (is (identical? m (:gen-fn (:trace (gfi/update tr [1] [diff/no-change] (cm/choicemap :x0 0.3) (rng/key 4))))))
    (is (identical? m (:gen-fn (:trace (gfi/update tr [2] [diff/unknown-change] cm/EMPTY (rng/key 5))))))))
(laws/deflaws exact-proposal law-models 60 laws/fixed-support)
(laws/deflaws batched-exact-proposal batched-law-models 60 laws/fixed-support)

;; ---------------------------------------------------------------------------
;; Loud errors
;; ---------------------------------------------------------------------------

(deftest unsupported-uses-are-loud
  (let [obs (cm/choicemap :y 0.0)
        gen-err (fn [gf re] (is (thrown-with-msg? clojure.lang.ExceptionInfo re (gfi/generate gf [] obs (rng/key 1)))))]
    (testing "a continuous choice"
      (gen-err (enum/with-exact-proposal (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x 1)))) [:x])
               #"not finite"))
    (testing "a choice made inside a callee at an address"
      (let [inner (gen [] (trace :z (d/bernoulli 0.5)))]
        (gen-err (enum/with-exact-proposal (gen [] (splice :in inner) (trace :y (d/normal 0 1))) [[:in :z]])
                 #"wrap that one")))
    (testing "an address the model does not visit"
      (gen-err (enum/with-exact-proposal (gen [] (trace :y (d/normal 0 1))) [:nope]) #"not visited"))
    (testing "a structure that depends on the enumerated choice"
      (gen-err (enum/with-exact-proposal (gen [] (when (trace :z (d/bernoulli 0.5)) (trace :w (d/normal 0 1)))
                                           (trace :y (d/normal 0 1)))
                                         [:z])
               #"structure depends"))
    (testing "a support that depends on another enumerated choice"
      (gen-err (enum/with-exact-proposal (gen [] (let [n (trace :n (d/uniform-discrete 1 3))]
                                                   (trace :k (d/uniform-discrete 0 n))
                                                   (trace :y (d/normal 0 1))))
                                         [:n :k])
               #"support of the choice at \[:k\] depends"))
    (testing "too many assignments"
      (gen-err (enum/with-exact-proposal (gen [] (doseq [i (range 3)] (trace [:b i] (d/bernoulli 0.5)))
                                           (trace :y (d/normal 0 1)))
                                         [[:b 0] [:b 1] [:b 2]] {:max-assignments 7})
               #"8 assignments"))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"dynamic generative function"
                          (enum/with-exact-proposal (comb/unfold step) [[0 :h]])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs the addresses" (enum/with-exact-proposal model [])))))

(deftest regenerating-a-structure-deciding-choice-keeps-the-rest
  ;; z decides whether :d exists; the thesis's gap A (law audit 2026-10-20)
  (let [gated (gen [] (let [z (trace :z (d/bernoulli 0.5))]
                        (when z (trace :d (d/normal 0 1)))
                        (trace :y (d/normal 0.3 1))))
        egf (enum/with-exact-proposal gated [:z])
        tr (:trace (gfi/generate egf [] (cm/choicemap :z true :d 0.4 :y 1.0) (rng/key 1)))
        outcomes (map #(try (let [{t' :trace} (gfi/regenerate tr (gen.selection/select :z) %)] [(t' :z) (t' :d)])
                            (catch clojure.lang.ExceptionInfo e (ex-message e)))
                      (rng/split (rng/key 2) 12))]
    ;; enumerating z needs a structure that does not depend on it, and the
    ;; unselected :d must stay: every outcome is a loud error, none a silent
    ;; resample of :d
    (is (empty? (filter vector? outcomes)))
    (is (every? #(re-find #"leaves an unselected choice unvisited|structure depends on the enumerated" %) outcomes))))
