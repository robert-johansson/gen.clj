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
            [gen.dist.core :as dc]
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

;; ---------------------------------------------------------------------------
;; Parameters a combinator's callees hold (DESIGN §9, genclj-umww)
;; ---------------------------------------------------------------------------

(defn- store-paths
  "The paths to the parameters in a tree of stores."
  [stores]
  (mapcat (fn [[k v]] (if (map? v) (map #(into [k] %) (store-paths v)) [[k]])) stores))

(defn- fd-store
  "The central difference of the objective in the parameter at path of the
   stores the trace's generative function holds."
  [{gf :gen-fn args :args choices :choices} path retgrad]
  (let [stores (gfi/stores gf)
        at (fn [d] (objective (gfi/with-stores gf (update-in stores path + d)) args choices retgrad))]
    (/ (- (at h) (at (- h))) (* 2 h))))

(defn- store-gradients-match?
  "param-gradients of tr, in the shape of its stores, against finite
   differences, parameter by parameter."
  [tr & [retgrad]]
  (let [stores (gfi/stores (:gen-fn tr))
        grads (g/param-gradients tr retgrad)]
    (and (= (set (store-paths stores)) (set (store-paths grads)))
         (every? #(grad-close? (get-in grads %) (fd-store tr % retgrad)) (store-paths stores)))))

(def ^:private scaled
  "z ~ N(mu·x, s), holding its parameters."
  (dyn/with-params (gen [x] (trace :z (d/normal (* (param :mu) x) (param :s)))) {:mu 0.5 :s 1.5}))

(def ^:private ar1
  "An AR(1) step, x ~ N(a·x_prev, 1), holding a."
  (dyn/with-params (gen [t prev] (trace :x (d/normal (* (param :a) prev) 1))) {:a 0.8}))

(def ^:private shifted
  "z ~ N(x + shift, 1), holding shift."
  (dyn/with-params (gen [x] (trace :z (d/normal (+ x (param :shift)) 1))) {:shift 0.3}))

(def ^:private plain
  "z ~ N(x, 1), holding nothing."
  (gen [x] (trace :z (d/normal x 1))))

(def ^:private counting-down
  "A Recurse whose production holds m and aggregation c: node n draws u ~
   N(m, 1) and has child n - 1 while n > 0; its value y ~ N(c + the child's
   y, 1)."
  (comb/recurse (dyn/with-params (gen [n] (trace :u (d/normal (param :m) 1)) {:value n :children (if (pos? n) [(dec n)] [])})
                                 {:m 0.2})
                (dyn/with-params (gen [_ kids] (trace :y (d/normal (reduce + (param :c) kids) 1))) {:c -0.4})
                1))

(deftest parameter-gradients-through-combinators
  (testing "Map: the kernel's store, accumulated over the elements"
    (let [xs [1.0 2.0 3.0]
          zs [0.4 1.7 2.2]
          tr (:trace (gfi/generate (comb/map scaled) [xs] (apply cm/choicemap (mapcat (fn [i z] [[i :z] z]) (range) zs))
                                   (rng/key 30)))
          per-element (map #(g/param-gradients (:trace (gfi/generate scaled [%1] (cm/choicemap :z %2) (rng/key 31)))) xs zs)
          summed (g/weighted-sum (repeat 1.0) per-element)]
      (is (= {:mu 0.5 :s 1.5} (gfi/stores (:gen-fn tr))))
      (is (store-gradients-match? tr))
      (is (and (ts/close? (:mu summed) (:mu (g/param-gradients tr))) (ts/close? (:s summed) (:s (g/param-gradients tr))))
          "the elements' gradients, summed")))
  (testing "Unfold: the kernel's"
    (is (store-gradients-match? (:trace (gfi/generate (comb/unfold ar1) [3 1.0] (cm/choicemap [0 :x] 0.9 [1 :x] 0.5 [2 :x] 0.7)
                                                      (rng/key 32))))))
  (testing "Switch: by branch, the inactive one's zero"
    (let [sw (comb/switch scaled shifted plain)]
      (is (= {0 {:mu 0.5 :s 1.5} 1 {:shift 0.3}} (gfi/stores sw)) "the branches that hold one")
      (doseq [[b branch] [[0 scaled] [1 shifted]]]
        (let [tr (:trace (gfi/generate sw [b 2.0] (cm/choicemap :z 1.1) (rng/key 33)))
              own (g/param-gradients (:trace (gfi/generate branch [2.0] (cm/choicemap :z 1.1) (rng/key 33))))]
          (is (store-gradients-match? tr) (str "branch " b))
          (is (every? (fn [[k v]] (ts/close? v (get-in (g/param-gradients tr) [b k]))) own)
              (str "branch " b "'s, as its own trace's"))
          (is (ts/close? 0.0 (get-in (g/param-gradients tr) (if (= b 0) [1 :shift] [0 :mu]))) (str "branch " b))))))
  (testing "Switch with branch keys"
    (is (store-gradients-match? (:trace (gfi/generate (comb/switch {:a scaled :b shifted}) [:b 2.0] (cm/choicemap :z 1.1)
                                                      (rng/key 34))))))
  (testing "Recurse: the production's and the aggregation's"
    (let [choices (cm/choicemap [0 :production :u] 0.1 [1 :production :u] -0.3 [2 :production :u] 0.6
                                [0 :aggregation :y] -1.0 [1 :aggregation :y] -0.6 [2 :aggregation :y] 0.2)
          tr (:trace (gfi/generate counting-down [2] choices (rng/key 35)))]
      (is (= {:production {:m 0.2} :aggregation {:c -0.4}} (gfi/stores counting-down)))
      (is (store-gradients-match? tr))
      ;; d/dm Σ log N(u; m, 1) = Σ (u - m); d/dc Σ log N(y; c + y_child, 1) = Σ (y - c - y_child)
      (is (ts/close? (- (+ 0.1 -0.3 0.6) (* 3 0.2)) (get-in (g/param-gradients tr) [:production :m])))
      (is (ts/close? (+ (- -1.0 -0.4 -0.6) (- -0.6 -0.4 0.2) (- 0.2 -0.4)) (get-in (g/param-gradients tr) [:aggregation :c])))))
  (testing "nested: a Map of Switches, with a return-value gradient"
    (let [tr (:trace (gfi/generate (comb/map (comb/switch scaled shifted)) [[0 1] [2.0 -1.0]]
                                   (cm/choicemap [0 :z] 0.8 [1 :z] -0.2) (rng/key 36)))]
      (is (store-gradients-match? tr [0.5 -2.0]))))
  (testing "no store is loud"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"holds no parameter store"
                          (g/param-gradients (gfi/simulate (comb/map plain) [[1.0]] (rng/key 37)))))))

(deftest learning-steps-a-combinator-s-stores
  ;; gen.learning steps the tree gfi/stores gives, and gfi/with-stores gives
  ;; it back: the Map's kernel's mu to its maximum likelihood, Σx·z/Σx², s
  ;; fixed
  (let [xs [1.0 2.0 3.0]
        zs [2.1 3.9 6.2]
        kernel (dyn/with-params (gen [x] (trace :z (d/normal (* (param :mu) x) 1))) {:mu 0.5})
        data (apply cm/choicemap (mapcat (fn [i z] [[i :z] z]) (range) zs))
        opt (learn/sgd {:lr 0.02})
        trained (reduce (fn [gf i]
                          (let [tr (:trace (gfi/generate gf [xs] data (rng/key i)))
                                [stores _] (learn/step opt nil (gfi/stores gf) (g/param-gradients tr))]
                            (gfi/with-stores gf stores)))
                        (comb/map kernel)
                        (range 300))]
    (is (< (ts/abs (- (:mu (gfi/stores trained)) (/ (reduce + (map * xs zs)) (reduce + (map * xs xs))))) 1e-4))))

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
    ;; five standard deviations of the final values over 16 keys (JVM,
    ;; measured 2026-10-20: 0.029, 0.043 and 0.048; Adam's iterates keep
    ;; moving at the optimum). The band of 0.05 before was under two, and
    ;; failed on float32 backends
    (is (< (ts/abs (- (:a store) 0.5)) 0.15) (str store))
    (is (< (ts/abs (:b store)) 0.22) (str store))
    (is (< (ts/abs (- (:log-s store) (Math/log (Math/sqrt 0.5)))) 0.25) (str store)))
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

(deftest a-step-updates-the-parameters-it-is-given
  (doseq [opt [(learn/sgd {:lr 0.1}) (learn/adam {:lr 0.1}) (learn/gradient-descent {:step-size-init 0.1 :step-size-beta 1.0})]]
    (let [store {:w 1.0 :b 2.0}
          [store' _] (learn/step opt (learn/init opt store) store {:w 0.5})]
      (is (= 2.0 (:b store')) (str (:kind opt) ": :b has no gradient and stays"))
      (is (< 1.0 (ts/number (:w store'))) (str (:kind opt) ": :w ascends"))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not have"
                        (learn/step (learn/sgd {:lr 0.1}) nil {:w 1.0} {:v 0.5}))))

(deftest argument-gradients-with-a-return-value-gradient
  ;; Eq. 2.11 (law audit gap K): log N(z; x, 1) + retgrad·(x·z) has
  ;; ∂/∂x = (z - x) + retgrad·z = 0.5 + 1.7·1.2 = 2.54 at x = 0.7, z = 1.2
  (let [model (gen [x] (let [z (trace :z (d/normal x 1))] (* x z)))
        tr (:trace (gfi/generate model [0.7] (cm/choicemap :z 1.2) (rng/key 1)))
        {[xg] :arg-grads} (g/choice-gradients tr sel/none 1.7 {:argnums [0]})
        f (fn [x] (+ (ts/number (:weight (gfi/assess model [x] (cm/choicemap :z 1.2)))) (* 1.7 x 1.2)))
        h 1e-5]
    (is (ts/close? 2.54 xg {:rtol (ts/tol 1e-9) :atol (ts/tol 1e-9)}))
    (when (ts/jvm?)
      (is (ts/close? (/ (- (f (+ 0.7 h)) (f (- 0.7 h))) (* 2 h)) xg {:rtol 1e-6 :atol 1e-6}) "and the finite difference"))))

(deftest lecture-is-exact
  ;; Eqs. 3.8-3.9 (law audit gap M): the score is log q(mu; w·y, e^log-s) of
  ;; the model's own mu, the gradient its closed form
  (let [model (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/normal mu 1))))
        proposal (gen [y] (trace :mu (d/normal (* (param :w) y) (array/exp (param :log-s)))))
        store {:w 0.3 :log-s 0.1}
        key (rng/key 21)
        {:keys [score grads]} (learn/lecture model [] proposal store (fn [tr] [(tr :y)]) key)
        tr (gfi/simulate model [] (first (rng/split key)))
        mu (ts/number (tr :mu)) y (ts/number (tr :y))
        s (Math/exp 0.1) z (/ (- mu (* 0.3 y)) s)]
    (is (ts/close? (ts/number (dc/log-prob (d/normal (* 0.3 y) s) mu)) score))
    (is (ts/close? (* (/ z s) y) (:w grads)))
    (is (ts/close? (- (* z z) 1.0) (:log-s grads)))))

(deftest gradient-trees-add-leaf-by-leaf
  ;; the one sum behind train, lecture-batched and VI's estimators: nested
  ;; trees add leaf by leaf, and a name only some runs read keeps theirs
  (let [g1 {:w 1.0 :layer {:b [1.0 2.0]}}
        g2 {:w 3.0 :layer {:b [5.0 6.0]} :only 4.0}]
    (is (= {:w 7.0 :layer {:b [11.0 14.0]} :only 8.0} (ts/host (g/weighted-sum [1.0 2.0] [g1 g2]))))
    (is (= {:w 2.0 :layer {:b [3.0 4.0]} :only 2.0} (ts/host (g/mean [g1 g2]))))))
