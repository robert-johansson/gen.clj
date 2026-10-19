(ns gen.gradients-test
  "Choice, argument and parameter gradients against finite differences of the
   assessed log density (thesis Eqs 2.10-2.12), and parameter learning."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as g]
            [gen.learning :as learn]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def h 1e-6)

;; Finite differences run on the JVM backend in float64 whatever the backend
;; under test (a float32 backend's would be noise)
(def ^:private reference (array/backend-named :jvm))

(defn- on-reference [v]
  (if (array/host? v) v (array/with-backend reference (array/array (array/->clj v)))))

(defn- reference-choices [choices]
  (reduce (fn [c p] (cm/set-choice c p (on-reference (cm/get-choice choices p)))) choices (cm/addresses choices)))

(defn- objective [gf args choices retgrad]
  (array/with-backend reference
    (let [{w :weight rv :retval} (gfi/assess gf args (reference-choices choices))]
      (+ (double (array/item w))
         (if retgrad (reduce + (map #(double (array/item (array/sum (* %1 %2)))) (array/tree-leaves retgrad) (array/tree-leaves rv))) 0.0)))))

(defn- grad-close?
  "A gradient (of the backend under test) and a float64 finite difference."
  [g fd]
  (< (ts/abs (- (ts/number g) fd)) (if (ts/f32?) (* 1e-3 (max 1.0 (Math/abs (double fd)))) 1e-5)))

(defn- fd-choice
  "The central difference of the objective in element i of the choice at p."
  [{gf :gen-fn args :args choices :choices} p i retgrad]
  (let [v (on-reference (cm/get-choice choices p))
        bump (fn [d] (if (array/host? v)
                       (+ v d)
                       (array/with-backend reference
                         (array/reshape (array/array (update (vec (flatten (vector (array/->clj v)))) i + d)) (array/shape v)))))
        at (fn [d] (objective gf args (cm/set-choice choices p (bump d)) retgrad))]
    (/ (- (at h) (at (- h))) (* 2 h))))

(defn- grads-match? [tr selection & [retgrad]]
  (let [{:keys [values gradients]} (g/choice-gradients tr selection retgrad)]
    (every? true?
            (for [p (cm/addresses values)
                  :let [gv (cm/get-choice gradients p)]
                  i (range (array/size gv))]
              (grad-close? (array/item (if (array/host? gv) gv (array/take (array/flatten gv) i)))
                           (fd-choice tr p i retgrad))))))

(def linreg
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      slope)))

(def observation (gen [mu] (trace :x (d/normal mu 1))))

(def hierarchical
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (mapv #(splice [:obs %] observation mu) (range n)))))

(def step
  (gen [t state]
    (let [s (trace :s (d/normal (* 0.8 state) 1))]
      (trace :obs (d/normal s 0.5))
      s)))

(def mv (gen [] (let [z (trace :z (d/mvnormal [0.0 1.0] [[2.0 0.3] [0.3 1.0]]))] (trace :y (d/normal (array/sum z) 1)))))

(defn- sim [gf args seed] (gfi/simulate gf args (rng/key seed)))

(deftest choice-gradients-match-finite-differences
  (is (grads-match? (sim linreg [[1.0 2.0 3.0]] 1) (sel/select :slope :intercept)))
  (is (grads-match? (sim linreg [[1.0 2.0 3.0]] 1) sel/all) "observations too")
  (is (grads-match? (sim hierarchical [3] 2) (sel/select :mu [:obs 1 :x])) "nested addresses")
  (is (grads-match? (sim (comb/unfold step) [4 0.0] 3) (sel/select [1 :s] [2 :s])) "inside an Unfold")
  (is (grads-match? (sim (comb/map observation) [[0.5 1.5]] 4) (sel/select [0 :x])) "inside a Map")
  (is (grads-match? (sim mv [] 5) (sel/select :z)) "an array-valued choice")
  (testing "with a return-value gradient"
    (is (grads-match? (sim linreg [[1.0 2.0]] 6) (sel/select :slope) 2.5))
    (is (grads-match? (sim (comb/unfold step) [3 0.0] 7) (sel/select [0 :s]) [0.5 -1.0 2.0]))))

(deftest argument-gradients
  (let [tr (sim linreg [[1.0 2.0 3.0]] 8)
        {[xs-grad] :arg-grads} (g/choice-gradients tr sel/none nil {:argnums [0]})
        fd (fn [i] (let [at (fn [d] (objective linreg [(update [1.0 2.0 3.0] i + d)] (:choices tr) nil))]
                     (/ (- (at h) (at (- h))) (* 2 h))))]
    (is (every? true? (map #(grad-close? %1 (fd %2)) (ts/host xs-grad) (range 3)))))
  (is (= [nil] (:arg-grads (g/choice-gradients (sim linreg [[1.0 2.0]] 9) (sel/select :slope))))
      "arguments are not differentiated unless asked")
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not a tree of floats"
                        (g/choice-gradients (sim hierarchical [2] 9) (sel/select :mu) nil {:argnums [0]})))
  (testing "host math on an argument is fine when it is not differentiated"
    (let [m (gen [n] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal (* x (clojure.core/* 2.0 n)) 1))))
          tr (sim m [1.5] 10)]
      (is (map? (g/choice-gradients tr (sel/select :x)))))))

(deftest discrete-choices-are-rejected
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not continuous"
                        (g/choice-gradients (sim (gen [] (trace :b (d/bernoulli 0.5))) [] 1) (sel/select :b)))))

(deftest the-jacobian-log-determinant
  (is (< (ts/abs (- (g/log-abs-det-jacobian #(* % (array/array [2.0 3.0])) (array/array [1.0 1.0])) (Math/log 6.0))) (ts/tol 1e-12)))
  (testing "a nonlinear map: (x, u) ↦ (x·e^u, -u) has |det J| = e^u"
    (is (< (ts/abs (- (g/log-abs-det-jacobian (fn [v] (let [x (array/take v 0) u (array/take v 1)]
                                                          (array/stack [(* x (array/exp u)) (array/negate u)])))
                                                (array/array [1.5 0.3]))
                        0.3))
           (ts/tol 1e-12)))))

;; ---------------------------------------------------------------------------
;; Parameters
;; ---------------------------------------------------------------------------

(def regression
  "y_i ~ N(w·x_i + b, 1), with trainable w and b."
  (gen [xs]
    (doseq [[i x] (map-indexed vector xs)]
      (trace [:y i] (d/normal (+ (* (param :w) x) (param :b)) 1)))))

(def xs [-1.0 0.0 1.0 2.0])
(def ys [-1.1 0.9 3.2 4.8])
(def data (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

(deftest parameter-gradients
  (let [store {:w 0.5 :b -0.2}
        tr (:trace (gfi/generate (dyn/with-params regression store) [xs] data (rng/key 1)))
        grads (g/param-gradients tr)
        fd (fn [k] (let [at (fn [d] (objective (dyn/with-params regression (update store k + d)) [xs] (:choices tr) nil))]
                     (/ (- (at h) (at (- h))) (* 2 h))))]
    (is (grad-close? (:w grads) (fd :w)))
    (is (grad-close? (:b grads) (fd :b))))
  (testing "parameters reach nested calls"
    (let [inner (gen [] (trace :x (d/normal (param :mu) 1)))
          outer (dyn/with-params (gen [] (splice :in inner)) {:mu 2.0})
          tr (:trace (gfi/generate outer [] (cm/choicemap [:in :x] 3.0) (rng/key 2)))]
      (is (< (ts/abs (- 1.0 (:mu (g/param-gradients tr)))) (ts/tol 1e-12)))))
  (testing "a missing parameter is a loud error"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"has no value" (gfi/simulate regression [xs] (rng/key 3))))))

(defn- fit [optimizer steps]
  (loop [i 0 store {:w 0.0 :b 0.0} state (learn/init optimizer {:w 0.0 :b 0.0})]
    (if (< i steps)
      (let [tr (:trace (gfi/generate (dyn/with-params regression store) [xs] data (rng/key i)))
            [store state] (learn/step optimizer state store (g/param-gradients tr))]
        (recur (inc i) store state))
      store)))

(deftest learning-finds-the-maximum-likelihood
  ;; least squares: w = cov(x, y)/var(x), b = ȳ - w·x̄
  (let [mx (/ (reduce + xs) 4) my (/ (reduce + ys) 4)
        w (/ (reduce + (map #(* (- %1 mx) (- %2 my)) xs ys)) (reduce + (map #(* (- % mx) (- % mx)) xs)))
        b (- my (* w mx))]
    (doseq [[label opt steps] [[:sgd (learn/sgd {:lr 0.05}) 400] [:adam (learn/adam {:lr 0.05}) 1500]]]
      (let [store (fit opt steps)]
        (is (< (ts/abs (- (:w store) w)) 1e-3) (str label " w"))
        (is (< (ts/abs (- (:b store) b)) 1e-3) (str label " b"))))))

(deftest training-on-generated-data
  ;; Gen.jl's train!: data drawn from a generator, averaged parameter
  ;; gradients, a decaying-step optimizer
  (let [data-fn (fn [_] [[xs] data])
        {:keys [store history]} (learn/train regression {:w 0.0 :b 0.0}
                                             (learn/gradient-descent {:step-size-init 0.05 :step-size-beta 1000})
                                             data-fn g/param-gradients
                                             {:epochs 20 :batches-per-epoch 20 :batch-size 2 :key (rng/key 9)})
        mx (/ (reduce + xs) 4) my (/ (reduce + ys) 4)
        w (/ (reduce + (map #(* (- %1 mx) (- %2 my)) xs ys)) (reduce + (map #(* (- % mx) (- % mx)) xs)))]
    (is (= 20 (count history)))
    (is (< (first history) (last history)) "the log density rises")
    (is (< (ts/abs (- (:w store) w)) 1e-3))
    (is (< (ts/abs (- (:b store) (- my (* w mx)))) 1e-3))))

(def ^:private conjugate
  (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/normal mu 1)) mu)))

(def ^:private amortized
  "A proposal for mu given y: N(a·y + b, exp(log-s))."
  (gen [y] (trace :mu (d/normal (+ (* (param :a) y) (param :b)) (array/exp (param :log-s))))))

(deftest lecture-trains-a-proposal-toward-the-posterior
  ;; mu | y ~ N(y/2, √½): the sleep phase's optimum is a = 1/2, b = 0,
  ;; log-s = log √½
  (let [opt (learn/adam {:lr 0.02})
        y-of (fn [tr] [(tr :y)])
        store (loop [i 0 store {:a 0.0 :b 0.3 :log-s 0.5} state (learn/init opt {:a 0.0 :b 0.3 :log-s 0.5}) key (rng/key 21)]
                (if (< i 1500)
                  (let [[k k-next] (rng/split key)
                        {:keys [grads]} (learn/lecture-batched conjugate [] amortized store y-of 16 k)
                        [store state] (learn/step opt state store grads)]
                    (recur (inc i) store state k-next))
                  store))]
    (is (< (ts/abs (- (:a store) 0.5)) 0.05) (str store))
    (is (< (ts/abs (:b store)) 0.05) (str store))
    (is (< (ts/abs (- (:log-s store) (Math/log (Math/sqrt 0.5)))) 0.05) (str store)))
  (testing "one lecture: the proposal's score on the model's own latent choice"
    (let [{:keys [score grads]} (learn/lecture conjugate [] amortized {:a 0.5 :b 0.0 :log-s 0.0} (fn [tr] [(tr :y)]) (rng/key 3))]
      (is (ts/finite? score))
      (is (= #{:a :b :log-s} (set (keys grads)))))))

(deftest one-step-updates-several-stores
  ;; Gen.jl's ParamUpdate over gf => params pairs: stores nested in a map
  (let [opt (learn/sgd {:lr 0.1})
        [stores _] (learn/step opt nil {:model {:w 1.0} :proposal {:a 2.0}} {:model {:w 0.5} :proposal {:a -1.0}})]
    (is (ts/close? 1.05 (get-in stores [:model :w])))
    (is (ts/close? 1.9 (get-in stores [:proposal :a])))))
