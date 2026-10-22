(ns gen.inference.enumerative-test
  "Enumerative inference (Gen.jl's enumerative_inference, choice_vol_grid)
   and exact proposals, against answers derived by hand: a discrete
   posterior, a conjugate normal, and a state-space model whose discrete
   latent is summed out step by step (GenGDT's hand enumeration)."
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
;; The GFI laws (the general set: the internal proposal is not forward sampling)
;; ---------------------------------------------------------------------------

(def law-models
  [[:discrete exact [] []]
   [:discrete-partly (enum/with-exact-proposal model [:b]) [] []]
   [:continuous-latent (enum/with-exact-proposal
                         (gen [m] (let [z (trace :z (d/bernoulli 0.3))
                                        x (trace :x (d/normal (if z m (- m)) 1.0))]
                                    (trace :y (d/normal x 0.5))))
                         [:z])
    [1.0] [[2.0]]]
   [:unfold chain [3 0.0 0.6] [[4 0.0 0.6] [2 0.0 0.3]]]
   ;; an update to more arguments makes fresh choices, which the exact
   ;; proposal's generate samples given the enumerated one (genclj-nqsm:
   ;; update-under, shared with with-internal-proposal; fewer arguments
   ;; would leave an unselected choice to regenerate unvisited)
   [:fresh-choices-beside-an-enumerated-one
    (enum/with-exact-proposal
      (gen [n] (let [k (trace :k (d/bernoulli 0.3))]
                 (doseq [i (range n)] (trace (keyword (str "x" i)) (d/normal (if k 1.0 -1.0) 1.0)))))
      [:k])
    [1] [[2] [3]]]])

(laws/deflaws exact-proposal law-models 60 laws/general)

(deftest updates-stay-the-wrapper-s
  ;; genclj-nqsm: update-under keeps the trace the wrapper's, with fresh
  ;; choices or without (planted: the trace left as gf's)
  (let [m (second (last law-models))
        tr (gfi/simulate m [1] (rng/key 3))]
    (is (identical? m (:gen-fn (:trace (gfi/update tr [1] [diff/no-change] (cm/choicemap :x0 0.3) (rng/key 4))))))
    (is (identical? m (:gen-fn (:trace (gfi/update tr [2] [diff/unknown-change] cm/EMPTY (rng/key 5))))))))
(laws/deflaws exact-proposal law-models 60 laws/fixed-support)

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
