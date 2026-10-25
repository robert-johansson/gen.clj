(ns gen.gradients-test
  "Choice, argument and parameter gradients against finite differences of the
   assessed log density (thesis Eqs 2.10-2.12), every particle's at once
   against each trace's, and parameter learning."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as array :refer [+ - * /]]
            [gen.array.batch :as batch]
            [gen.array.dtype :as dt]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.custom :as custom]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as g]
            [gen.inference.enumerative :as enum]
            [gen.inference.mcmc :as mcmc]
            [gen.inference.proposal :as prop]
            [gen.inference.vi :as vi]
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

(defn- inner
  "⟨retgrad, retval⟩ on the host, pairing maps by key and vectors by
   position."
  [g v]
  (cond (nil? g) 0.0
        (map? g) (reduce + 0.0 (map (fn [[k gk]] (inner gk (get v k))) g))
        (sequential? g) (reduce + 0.0 (map inner g v))
        :else (double (array/item (array/sum (* g v))))))

(defn- objective [gf args choices retgrad]
  (array/with-backend reference
    (let [{w :weight rv :retval} (gfi/assess gf args (reference-choices choices))]
      (+ (double (array/item w)) (inner retgrad rv)))))

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

(defn- choices-close?
  "Do choice maps a and b hold the same addresses, with close values?"
  [a b]
  (and (= (cm/addresses a) (cm/addresses b))
       (every? #(ts/close? (cm/get-choice a %) (cm/get-choice b %)) (cm/addresses a))))

(defn- stacked
  "Trees of one structure (vectors of arrays, or nil), one per particle,
   stacked leaf by leaf."
  [ts]
  (cond (nil? (first ts)) nil
        (vector? (first ts)) (apply mapv (fn [& xs] (stacked xs)) ts)
        :else (array/stack (vec ts))))

(defn- batched-only
  "f's value, where gen.batched/map-particles falling back to one trace at a
   time is an error: so that a batched collection's result is checked as
   the batched run made it, not as a fallback made it again."
  [f]
  (with-redefs [b/traces (fn [_] (throw (ex-info "map-particles fell back to one trace at a time" {})))]
    (f)))

(defn- every-particle-s?
  "Is choice-gradients of the collection ps each of its traces', stacked:
   values, gradients and argument gradients? A batched collection's in one
   batched run."
  [ps selection retgrad opts]
  (let [run #(g/choice-gradients ps selection retgrad opts)
        r (if (b/batched? ps) (batched-only run) (run))
        each (mapv #(g/choice-gradients % selection retgrad opts) (b/traces ps))
        arg-grads (stacked (mapv :arg-grads each))]
    (and (choices-close? (:values r) (cm/stack (mapv :values each)))
         (choices-close? (:gradients r) (cm/stack (mapv :gradients each)))
         (= (map nil? (:arg-grads r)) (map nil? arg-grads))
         (every? true? (map ts/close? (remove nil? (array/tree-leaves (:arg-grads r))) (remove nil? (array/tree-leaves arg-grads)))))))

(defn- particles-of [gf args observations n seed]
  (:particles (b/generate gf args observations n (rng/key seed))))

(deftest every-particle-s-choice-gradients-at-once
  ;; one batched gradient for all the particles: each particle's its own
  ;; trace's (gen.array.batch's vmap semantics)
  (let [obs (cm/choicemap [:y 0] 2.1 [:y 1] 3.9 [:y 2] 6.2)
        lr (particles-of linreg [[1.0 2.0 3.0]] obs 5 11)]
    (is (b/batched? lr))
    (is (every-particle-s? lr (sel/select :slope :intercept) nil nil))
    (is (every-particle-s? lr sel/all nil nil) "the observations too, shared by every particle")
    (is (every-particle-s? lr (sel/select :slope) 2.5 nil) "with a return-value gradient")
    (is (every-particle-s? lr (sel/select :slope) nil {:argnums [0]}) "an argument's, per particle")
    (is (every-particle-s? (particles-of hierarchical [3] cm/EMPTY 4 12) (sel/select :mu [:obs 1 :x]) nil nil)
        "nested addresses")
    (is (every-particle-s? (particles-of (comb/unfold step) [4 0.0] cm/EMPTY 4 13) (sel/select [1 :s] [2 :s]) nil nil)
        "inside an Unfold")
    (is (every-particle-s? (particles-of (comb/map observation) [[0.5 1.5]] cm/EMPTY 4 14) (sel/select [0 :x]) nil nil)
        "inside a Map")
    (is (every-particle-s? (particles-of mv [] cm/EMPTY 4 15) (sel/select :z) nil nil) "an array-valued choice")
    (testing "a collection that fell back, one trace at a time"
      (let [ps (particles-of (gen [] (let [k (trace :k (d/categorical [0.5 0.5]))]
                                       (trace :x (d/normal (if (= k 0) 0.0 3.0) 1))))
                             [] cm/EMPTY 6 16)]
        (is (not (b/batched? ps)))
        (is (every-particle-s? ps (sel/select :x) nil nil))))))

;; ---------------------------------------------------------------------------
;; Reading a value being differentiated on the host (genclj-de4g): its
;; gradient would be lost, so it is an error, not a silent zero
;; ---------------------------------------------------------------------------

(defmacro ^:private doubled-mean
  "m ~ N(0, 1), y ~ N(mean, 1), mean computed from m by the form: d/dm of
   the log density at m = 0.8, y = 3 is -0.8 + 2·(3 - 1.6) = 2.0 when the
   mean is 2m and its gradient kept, -0.8 (the prior's term) when lost."
  [m form]
  `(gen [] (let [~m (~'trace :m (d/normal 0 1))] (~'trace :y (d/normal ~form 1)))))

(def ^:private by-array (doubled-mean m (* 2.0 m)))
(def ^:private by-item (doubled-mean m (* 2.0 (array/item m))))
(def ^:private by-clj (doubled-mean m (* 2.0 (array/->clj m))))
(def ^:private by-stop-gradient (doubled-mean m (* 2.0 (array/item (array/stop-gradient m)))))
(def ^:private by-host-math (doubled-mean m (clojure.core/* 2.0 m)))
(def ^:private by-comparisons
  ;; a body's = reads m on the host (gen.dynamic.core/host-value), and a
  ;; comparison is a bool: control flow keeps the branch's gradient
  (doubled-mean m (if (and (array/> m 0) (array/item (array/> m 0)) (not= m 0.0)) (* 2.0 m) m)))

(def ^:private doubled (custom/deterministic {:apply (fn [[m]] (* 2.0 (array/item m)))}))
(def ^:private doubled-with-gradient
  (custom/deterministic {:apply (fn [[m]] (* 2.0 (array/item m))) :gradient (fn [_ _ g] [(* 2.0 g)])}))
(def ^:private by-deterministic (doubled-mean m (splice :f doubled m)))
(def ^:private by-custom-gradient (doubled-mean m (splice :f doubled-with-gradient m)))

(defn- gradient-at-m [model]
  (let [tr (:trace (gfi/generate model [] (cm/choicemap :m 0.8 :y 3.0) (rng/key 0)))]
    (cm/get-choice (:gradients (g/choice-gradients tr (sel/select :m))) :m)))

(defn- read-error
  "The ex-info thrown by (f), or nil."
  [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e e)))

(deftest host-reads-of-differentiated-values-are-loud
  ;; Planted bugs (2026-10-24): the read returning the primal again (the
  ;; tape's throw removed, the old behaviour) fails 16 assertions and errors
  ;; 3 here; without value-and-grad's catch, host arithmetic is a raw
  ;; ClassCastException (an error); a body's = reading m without
  ;; stop-gradient (gen.dynamic.core/host-value) errors by-comparisons
  (testing "gen.array's operators, a custom gradient and control flow keep the gradient"
    (is (ts/close? 2.0 (gradient-at-m by-array)))
    (is (ts/close? 2.0 (gradient-at-m by-custom-gradient)))
    (is (ts/close? 2.0 (gradient-at-m by-comparisons))))
  (testing "a read of the value through stop-gradient drops its gradient on purpose"
    (is (ts/close? -0.8 (gradient-at-m by-stop-gradient))))
  (when-not (ts/native-grad?)
    (testing "item and ->clj of a float being differentiated throw, naming the ways out"
      (doseq [model [by-item by-clj by-deterministic]
              :let [e (read-error #(gradient-at-m model))]]
        (is (some? e))
        (is (re-find #"gradient would be lost" (str (ex-message e))))
        (is (re-find #"stop-gradient" (str (ex-message e))))
        (is (re-find #"custom gradient" (str (ex-message e))))
        (is (= [] (:shape (ex-data e))))
        (is (= :float (dt/kind (:dtype (ex-data e)))))))
    (testing "an array's read reports its shape"
      (let [e (read-error #((array/grad (fn [x] (array/sum (array/array (array/->clj x))))) (array/array [1.0 2.0])))]
        (is (= [2] (:shape (ex-data e))))))
    (testing "bools and ints have no gradient: they read"
      (is (ts/close? 2.0 ((array/grad (fn [x] (if (array/item (array/> x 0)) (* 2.0 x) x))) 1.5)))
      (is (ts/close? 3.0 ((array/grad (fn [x] (* x (array/item (array/astype x :int32))))) 3.5)))
      (is (ts/close? 1.0 ((array/grad (fn [x] (+ x (array/item (array/floor x))))) 1.5)) "floor's value is untracked"))
    (testing "host arithmetic on a value being differentiated is loud, not a ClassCastException"
      (let [e (read-error #((array/grad (fn [x] (clojure.core/* 2.0 x))) 3.0))]
        (is (re-find #"Host arithmetic" (str (ex-message e))))
        (is (re-find #"stop-gradient" (str (ex-message e))))
        (is (instance? ClassCastException (ex-cause e))))
      (when (ts/jvm?)
        ;; on MLX a body's 0-d values are arrays, so host arithmetic fails
        ;; on them outside gradients too
        (is (re-find #"Host arithmetic" (str (ex-message (read-error #(gradient-at-m by-host-math))))))))))

(def ^:private validated
  "Parameters the distributions validate on the host, each a function of
   the differentiated :x: positive (normal's std, gamma's scale, beta's,
   dirichlet's), a probability (bernoulli's), probabilities (categorical's),
   ordered bounds (uniform's, truncated-normal's) and a covariance
   (mvnormal's). Their checks read comparisons, which have no gradient, so
   gradients through them run."
  (gen []
    (let [x (trace :x (d/normal 0 1))
          s (array/exp (* 0.5 x))
          p (array/sigmoid x)]
      (trace :a (d/normal x s))
      (trace :b (d/bernoulli p))
      (trace :c (d/categorical (array/stack [p (- 1.0 p)])))
      (trace :g (d/gamma 2.0 s))
      (trace :be (d/beta s 2.0))
      (trace :di (d/dirichlet (array/stack [s 1.0])))
      (trace :u (d/uniform (- x 5.0) (+ x 5.0)))
      (trace :t (d/truncated-normal x s -3.0 3.0))
      (trace :z (d/mvnormal (array/stack [x x]) (* s (array/eye 2))))
      x)))

(def ^:private validated-observations
  (cm/choicemap :a 0.3 :b true :c 1 :g 1.5 :be 0.4 :di (array/array [0.3 0.7]) :u 0.5 :t 0.2
                :z (array/array [0.2 0.4])))

(def ^:private validated-family (gen [] (trace :x (d/normal (param :m) (array/exp (param :log-s))))))

(deftest gradients-through-validated-parameters
  ;; Planted bugs (2026-10-24): a check reading its parameter's floats
  ;; (check-positive as (pos? (array/item (array/min v)))) errors every case
  ;; of the first part with the host-read error; mvnormal's jitter
  ;; suggestion reading the covariance being differentiated fails the
  ;; second's two assertions, and a distribution's printed parameters read
  ;; without stop-gradient error the third
  (let [start (:trace (gfi/generate validated [] (cm/merge validated-observations (cm/choicemap :x 0.1)) (rng/key 0)))
        x (sel/select :x)
        moved? (fn [step] (some #(:accepted? (step %)) (rng/split (rng/key 1) 10)))]
    (testing "choice gradients"
      (is (grad-close? (cm/get-choice (:gradients (g/choice-gradients start x)) :x) (fd-choice start :x 0 nil))))
    (testing "HMC, MALA and MAP move"
      (is (moved? #(mcmc/hmc start x {:L 5 :eps 0.05} %)))
      (is (moved? #(mcmc/mala start x 0.05 %)))
      (is (not= (ts/number (start :x)) (ts/number ((:trace (mcmc/map-optimize start x)) :x)))))
    (testing "VI's reparameterized ELBO gradient, through the model's checks"
      (let [{:keys [objective grads]} (vi/elbo-gradient validated [] validated-observations validated-family []
                                                        {:m 0.0 :log-s -1.0} 4 {:estimator :reparam :key (rng/key 2)})]
        (is (Double/isFinite (ts/number objective)))
        (is (every? #(Double/isFinite (ts/number %)) (vals grads))))))
  (when (array/concrete? (array/ones [2 2]))
    (testing "a covariance failing its check under a gradient is the check's error, with its jitter"
      ;; its suggestion reads the covariance's diagonal: on purpose, by stop-gradient
      (let [e (read-error #((array/grad (fn [v] (dc/log-prob (d/mvnormal [0.0 0.0] (* v (array/ones [2 2]))) (array/zeros [2]))))
                            2.0))]
        (is (re-find #"positive definite" (str (ex-message e))))
        (is (:suggested-jitter (ex-data e))))))
  (testing "printing a distribution whose parameters are being differentiated shows their values"
    (let [printed (atom nil)]
      ((array/grad (fn [v] (let [dist (d/normal (* v (array/array [1.0 2.0])) v)]
                             (reset! printed (pr-str dist))
                             (array/sum (dc/log-prob dist (array/zeros [2]))))))
       1.5)
      (is (re-find #"normal" @printed))
      (is (re-find #"1\.5" @printed)))))

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
           (ts/tol 1e-12))))
  (testing "a non-square Jacobian, the trace kernels' and transforms' rows: (a, b) ↦ (ab, a + b, b²)"
    (is (ts/close? (array/array [[3.0 2.0] [1.0 1.0] [0.0 6.0]])
                   (g/jacobian (fn [v] (let [a (array/take v 0) b (array/take v 1)] (array/stack [(* a b) (+ a b) (* b b)])))
                               (array/array [2.0 3.0]) 3))))
  (testing "values flattened for a Jacobian, and back in their shapes"
    (let [values [0.5 (array/array [1.0 2.0]) (array/array [[3.0 4.0] [5.0 6.0]])]
          [x unflatten] (g/flattened values)]
      (is (= [7] (array/shape x)))
      (is (= (ts/host values) (ts/host (unflatten x))))
      (is (= [[] [2] [2 2]] (mapv array/shape (unflatten (* 2.0 x))))))))

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

(def ^:private scan-ar1
  "An AR(1) Scan step, x ~ N(a·carry, 1), holding a; its carry and output x."
  (dyn/with-params (gen [prev _] (let [x (trace :x (d/normal (* (param :a) prev) 1))] [x x])) {:a 0.8}))

(deftest parameter-gradients-through-scan-mix-and-dimap
  ;; each against finite differences and its closed form: d/da Σ log N(x_t;
  ;; a·x_t-1, 1) = Σ (x_t - a·x_t-1)·x_t-1, and d/dmu log N(z; mu·x, s) =
  ;; (z - mu·x)·x/s². Planted bugs (2026-10-23): Scan's with-stores keeping
  ;; the kernel's store fails the first testing; dimap's assess calling the
  ;; inner on the arguments unmapped, the third
  (let [dmu (/ (* (- 1.1 (* 0.5 2.0)) 2.0) (* 1.5 1.5))]
    (testing "Scan: the kernel's, accumulated over the steps"
      (let [tr (:trace (gfi/generate (comb/scan scan-ar1) [1.0 [nil nil nil]] (cm/choicemap [0 :x] 0.9 [1 :x] 0.5 [2 :x] 0.7)
                                     (rng/key 38)))]
        (is (store-gradients-match? tr))
        (is (ts/close? (+ (* (- 0.9 (* 0.8 1.0)) 1.0) (* (- 0.5 (* 0.8 0.9)) 0.9) (* (- 0.7 (* 0.8 0.5)) 0.5))
                       (:a (g/param-gradients tr))))))
    (testing "Mix: by component, as a Switch's"
      (let [m (comb/mix scaled shifted plain)]
        (is (= {0 {:mu 0.5 :s 1.5} 1 {:shift 0.3}} (gfi/stores m)))
        (doseq [i [0 1]]
          (is (store-gradients-match? (:trace (gfi/generate m [[0.2 0.5 0.3] 2.0] (cm/choicemap :index i [:component :z] 1.1)
                                                            (rng/key 39))))
              (str "component " i)))
        (is (ts/close? dmu (get-in (g/param-gradients (:trace (gfi/generate m [[0.2 0.5 0.3] 2.0]
                                                                            (cm/choicemap :index 0 [:component :z] 1.1)
                                                                            (rng/key 39))))
                                   [0 :mu])))))
    (testing "dimap: the inner's, on the mapped arguments"
      (let [tr (:trace (gfi/generate (comb/dimap scaled (fn [x] [(* 2.0 x)]) -) [1.0] (cm/choicemap :z 1.1) (rng/key 40)))]
        (is (store-gradients-match? tr))
        (is (ts/close? dmu (:mu (g/param-gradients tr))))))))

;; with-internal-proposal and with-exact-proposal keep their model's density,
;; so their stores and parameter gradients are the model's
(deftest a-wrapper-s-stores-are-its-model-s
  (let [model (dyn/with-params (gen [] (let [k (trace :k (d/bernoulli 0.3))]
                                          (trace :x (d/normal (+ (param :mu) (array/where k 1.0 0.0)) 1))))
                               {:mu 0.5})
        choices (cm/choicemap :k true :x 1.2)
        own (g/param-gradients (:trace (gfi/generate model [] choices (rng/key 40))))]
    (doseq [[label wrapped] [["an internal proposal" (prop/with-internal-proposal
                                                      model
                                                      (gen [cs] (when-not (cm/has-value? (cm/get-submap cs :k))
                                                                  (trace :k (d/bernoulli 0.5)))))]
                             ["an exact proposal" (enum/with-exact-proposal model [:k])]]]
      (testing label
        (let [tr (:trace (gfi/generate wrapped [] choices (rng/key 41)))
              moved (gfi/with-stores wrapped {:mu 2.0})]
          (is (= {:mu 0.5} (gfi/stores wrapped)))
          (is (ts/close? (:mu own) (:mu (g/param-gradients tr))) "the model's gradient, d/dmu = 1.2 - 1.5")
          (is (= {:mu 2.0} (gfi/stores moved)))
          ;; d/dmu log N(1.2; mu + 1, 1) at mu = 2
          (is (ts/close? -1.8 (:mu (g/param-gradients (:trace (gfi/generate moved [] choices (rng/key 42))))))
              "the stores given back reach the model"))))))

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

(defn- evaluated?
  "Is x a value rather than a graph: always on an eager backend, and on MLX
   when MLX holds its data."
  [x]
  (or (not (array/array? x)) (array/concrete? x) ((requiring-resolve 'gen.backend.mlx/evaluated?) x)))

(deftest a-step-is-evaluated
  ;; a step is an inference loop's boundary (rule 3): on a lazy backend its
  ;; store and state are values, so a loop of steps holds no graph from one
  ;; to the next. Without it, train-test's lecture loop grew each store into
  ;; a graph over every step before (genclj-9e9e: 270 KB a lecture on Metal)
  (doseq [opt [(learn/sgd {:lr 0.1}) (learn/adam {:lr 0.1}) (learn/gradient-descent {:step-size-init 0.1 :step-size-beta 1.0})]]
    (let [store {:w (array/array [1.0 2.0]) :b (array/array 0.5)}
          grads {:w (* 2.0 (array/array [0.5 -0.5])) :b (* 3.0 (array/array 0.1))}
          stepped (learn/step opt (learn/init opt store) store grads)]
      (is (every? evaluated? (array/tree-leaves stepped)) (str (:kind opt)))))
  (testing "a step inside a gradient (tape values), and under the batch tracer, is the same step"
    (let [lr 0.1
          w-after #(:w (first (learn/step (learn/sgd {:lr lr}) nil {:w %} {:w (* 2.0 %)})))]
      (is (ts/close? (+ 1.0 (* 2.0 lr)) ((array/grad w-after) 1.5)) "d/dw of w + lr·2w")
      (let [ws (array/array [1.0 2.0 3.0])]
        (is (ts/close? (* (+ 1.0 (* 2.0 lr)) ws) (batch/unwrap (batch/batched [3] (w-after (batch/wrap ws [3]))) [3])))))))

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

(deftest a-retgrad-pairs-with-the-return-value-by-structure
  ;; the post-M5 review's P3-03: maps by key, vectors by position, nil for
  ;; no cotangent, and any other structure loud. Planted bug (2026-10-22):
  ;; the leaf-order pairing it replaced gives 1.5 for {:a 1 :b 0}, 0.5 for
  ;; {:z 1 :y 0}, and takes the short and scalar cotangents silently
  (let [model (gen [] (let [x (trace :x (d/normal 0 1))] {:b (* 2.0 x) :a x}))
        tr (:trace (gfi/generate model [] (cm/choicemap :x 0.5) (rng/key 1)))
        dx (fn [retgrad] (ts/number (cm/get-choice (:gradients (g/choice-gradients tr (sel/select :x) retgrad)) :x)))]
    (testing "log N(x; 0, 1) + a·x + b·2x has ∂/∂x = -x + a + 2b at x = 0.5, in either key order"
      (is (ts/close? 0.5 (dx {:a 1.0 :b 0.0})))
      (is (ts/close? 0.5 (dx {:b 0.0 :a 1.0})))
      (is (ts/close? 1.5 (dx {:a 0.0 :b 1.0})))
      (is (ts/close? 0.5 (dx {:a 1.0 :b nil})) "nil: no cotangent")))
  (testing "through param-gradients: {:y w·x, :z 3w} at x = 0.5 has ∂/∂w = 0.5 y' + 3 z'"
    (let [model (dyn/with-params (gen [] (let [x (trace :x (d/normal 0 1)) w (param :w)] {:y (* w x) :z (* 3.0 w)}))
                  {:w 1.0})
          tr (:trace (gfi/generate model [] (cm/choicemap :x 0.5) (rng/key 1)))
          dw (fn [retgrad] (ts/number (:w (g/param-gradients tr retgrad))))]
      (is (ts/close? 3.0 (dw {:z 1.0 :y 0.0})))
      (is (ts/close? 0.5 (dw {:y 1.0 :z 0.0})))
      (testing "a key in one tree only is loud"
        (is (= {:only-retgrad [:q] :only-retval [:z]}
               (select-keys (ts/thrown-data (dw {:q 1.0 :y 1.0})) [:only-retgrad :only-retval])))
        (is (= [:z] (:only-retval (ts/thrown-data (dw {:y 1.0}))))))))
  (testing "vectors pair by position, nested paths named; a short or scalar cotangent is loud"
    (let [model (gen [] (let [x (trace :x (d/normal 0 1))] [x {:s (* 3.0 x)}]))
          tr (:trace (gfi/generate model [] (cm/choicemap :x 0.5) (rng/key 1)))
          dx (fn [retgrad] (ts/number (cm/get-choice (:gradients (g/choice-gradients tr (sel/select :x) retgrad)) :x)))]
      (is (ts/close? 3.5 (dx [1.0 {:s 1.0}])) "-x + 1 + 3")
      (is (ts/close? 0.5 (dx [1.0 nil])))
      (is (= [] (:path (ts/thrown-data (dx [1.0])))) "one element against two")
      (is (= [] (:path (ts/thrown-data (dx 1.0)))) "a leaf against a sequence")
      (is (= [1] (:path (ts/thrown-data (dx [1.0 2.0])))) "a leaf against a map")
      (is (= [1 :s] (:path (ts/thrown-data (dx [1.0 {:s [1.0]}])))) "a sequence against a leaf"))))

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

(deftest misnamed-optimizers-are-loud
  ;; the post-M5 review's I3-14: an unknown :kind names the kinds. Planted
  ;; bug (2026-10-22): the case without a default throws a bare
  ;; IllegalArgumentException, "No matching clause". The kinds are open
  ;; now: gen.learning-test adds one
  (is (every? (set (:kinds (ts/thrown-data (learn/init {:kind :rmsprob} {:w 1.0})))) [:sgd :gradient-descent :adam]))
  (is (= :rmsprob (:kind (ts/thrown-data (learn/step {:kind :rmsprob} nil {:w 1.0} {:w 0.1}))))))

(deftest gradient-trees-add-leaf-by-leaf
  ;; the one sum behind train, lecture-batched and VI's estimators: nested
  ;; trees add leaf by leaf, and a name only some runs read keeps theirs
  (let [g1 {:w 1.0 :layer {:b [1.0 2.0]}}
        g2 {:w 3.0 :layer {:b [5.0 6.0]} :only 4.0}]
    (is (= {:w 7.0 :layer {:b [11.0 14.0]} :only 8.0} (ts/host (g/weighted-sum [1.0 2.0] [g1 g2]))))
    (is (= {:w 2.0 :layer {:b [3.0 4.0]} :only 2.0} (ts/host (g/mean [g1 g2]))))))
