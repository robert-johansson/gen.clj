(ns gen.gradients-test
  "Choice, argument and parameter gradients against finite differences of the
   assessed log density (thesis Eqs 2.10-2.12), and parameter learning."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as a]
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

(defn- objective [gf args choices retgrad]
  (let [{w :weight rv :retval} (gfi/assess gf args choices)]
    (+ (double (a/item w))
       (if retgrad (reduce + (map #(double (a/item (a/sum (a/* %1 %2)))) (a/tree-leaves retgrad) (a/tree-leaves rv))) 0.0))))

(defn- fd-choice
  "The central difference of the objective in element i of the choice at p."
  [{gf :gen-fn args :args choices :choices} p i retgrad]
  (let [v (cm/get-choice choices p)
        bump (fn [d] (if (a/host? v)
                       (+ v d)
                       (a/reshape (a/array (update (vec (flatten (vector (a/->clj v)))) i + d)) (a/shape v))))
        at (fn [d] (objective gf args (cm/set-choice choices p (bump d)) retgrad))]
    (/ (- (at h) (at (- h))) (* 2 h))))

(defn- grads-match? [tr selection & [retgrad]]
  (let [{:keys [values gradients]} (g/choice-gradients tr selection retgrad)]
    (every? true?
            (for [p (cm/addresses values)
                  :let [gv (cm/get-choice gradients p)]
                  i (range (a/size gv))]
              (< (Math/abs (- (double (a/item (if (a/host? gv) gv (a/take (a/flatten gv) i))))
                              (fd-choice tr p i retgrad)))
                 1e-5)))))

(def linreg
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (a/+ (a/* slope x) intercept) 1)))
      slope)))

(def observation (gen [mu] (trace :x (d/normal mu 1))))

(def hierarchical
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))]
      (mapv #(splice [:obs %] observation mu) (range n)))))

(def step
  (gen [t state]
    (let [s (trace :s (d/normal (a/* 0.8 state) 1))]
      (trace :obs (d/normal s 0.5))
      s)))

(def mv (gen [] (let [z (trace :z (d/mvnormal [0.0 1.0] [[2.0 0.3] [0.3 1.0]]))] (trace :y (d/normal (a/sum z) 1)))))

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
        {[xs-grad] :arg-grads} (g/choice-gradients tr sel/none)
        fd (fn [i] (let [at (fn [d] (objective linreg [(update [1.0 2.0 3.0] i + d)] (:choices tr) nil))]
                     (/ (- (at h) (at (- h))) (* 2 h))))]
    (is (every? true? (map #(< (Math/abs (- %1 (fd %2))) 1e-5) xs-grad (range 3)))))
  (is (= [nil] (:arg-grads (g/choice-gradients (sim hierarchical [2] 9) (sel/select :mu))))
      "an integer argument has no gradient"))

(deftest discrete-choices-are-rejected
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not continuous"
                        (g/choice-gradients (sim (gen [] (trace :b (d/bernoulli 0.5))) [] 1) (sel/select :b)))))

(deftest the-jacobian-log-determinant
  (is (< (Math/abs (- (g/log-abs-det-jacobian #(a/* % (a/array [2.0 3.0])) (a/array [1.0 1.0])) (Math/log 6.0))) 1e-12))
  (testing "a nonlinear map: (x, u) ↦ (x·e^u, -u) has |det J| = e^u"
    (is (< (Math/abs (- (g/log-abs-det-jacobian (fn [v] (let [x (a/take v 0) u (a/take v 1)]
                                                          (a/stack [(a/* x (a/exp u)) (a/negate u)])))
                                                (a/array [1.5 0.3]))
                        0.3))
           1e-12))))

;; ---------------------------------------------------------------------------
;; Parameters
;; ---------------------------------------------------------------------------

(def regression
  "y_i ~ N(w·x_i + b, 1), with trainable w and b."
  (gen [xs]
    (doseq [[i x] (map-indexed vector xs)]
      (trace [:y i] (d/normal (a/+ (a/* (param :w) x) (param :b)) 1)))))

(def xs [-1.0 0.0 1.0 2.0])
(def ys [-1.1 0.9 3.2 4.8])
(def data (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys)))

(deftest parameter-gradients
  (let [store {:w 0.5 :b -0.2}
        tr (:trace (gfi/generate (dyn/with-params regression store) [xs] data (rng/key 1)))
        grads (g/param-gradients tr)
        fd (fn [k] (let [at (fn [d] (objective (dyn/with-params regression (update store k + d)) [xs] (:choices tr) nil))]
                     (/ (- (at h) (at (- h))) (* 2 h))))]
    (is (< (Math/abs (- (:w grads) (fd :w))) 1e-5))
    (is (< (Math/abs (- (:b grads) (fd :b))) 1e-5)))
  (testing "parameters reach nested calls"
    (let [inner (gen [] (trace :x (d/normal (param :mu) 1)))
          outer (dyn/with-params (gen [] (splice :in inner)) {:mu 2.0})
          tr (:trace (gfi/generate outer [] (cm/choicemap [:in :x] 3.0) (rng/key 2)))]
      (is (< (Math/abs (- 1.0 (:mu (g/param-gradients tr)))) 1e-12))))
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
        (is (< (Math/abs (- (:w store) w)) 1e-3) (str label " w"))
        (is (< (Math/abs (- (:b store) b)) 1e-3) (str label " b"))))))

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
    (is (< (Math/abs (- (:w store) w)) 1e-3))
    (is (< (Math/abs (- (:b store) (- my (* w mx)))) 1e-3))))
