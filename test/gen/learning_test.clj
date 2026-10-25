(ns gen.learning-test
  "gen.learning's optimizers against reference trajectories, and its open
   interface.

   The references are PyTorch's update rules (torch.optim's documented
   algorithms; optax's clip_by_global_norm), written here in host doubles
   in their descent form, on the loss L = -f of a quadratic f with
   curvatures from 0.2 to 30, so that each rule's per-parameter scaling
   shows. gen.learning ascends f on a nested store of arrays; each step of
   the path must agree. Neither PyTorch nor optax is installed beside
   JAX in the fixtures' Python (2026-10-25), so the rules are transcribed,
   not run. Each rule's power was checked by a planted bug in its -step
   (2026-10-25), noted at its test."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core :as c]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gradients :as grad]
            [gen.inference.vi :as vi]
            [gen.learning :as learn]
            [gen.random :as rng]
            [gen.test-support :as ts]))

;; ---------------------------------------------------------------------------
;; Reference trajectories on a quadratic
;; ---------------------------------------------------------------------------

;; f(θ) = -½ Σ a_i (θ_i - c_i)², four parameters: three in :w and one in
;; [:more :b], a nested store
(def ^:private curvature [1.0 30.0 0.2 5.0])
(def ^:private center [3.0 0.5 -1.0 -2.0])
(def ^:private start [1.0 -2.0 0.5 4.0])

(defn- store-of [[w0 w1 w2 b]] {:w (array/array [w0 w1 w2]) :more {:b (array/array [b])}})

(defn- params-of [store] (vec (concat (ts/host (:w store)) (ts/host (get-in store [:more :b])))))

(defn- ascent-gradient
  "∇f at the store, a tree of its shape."
  [store]
  (let [[a0 a1 a2 a3] curvature [c0 c1 c2 c3] center]
    {:w (* (array/array [(c/- a0) (c/- a1) (c/- a2)]) (- (:w store) (array/array [c0 c1 c2])))
     :more {:b (* (array/array [(c/- a3)]) (- (get-in store [:more :b]) (array/array [c3])))}}))

(defn- loss-gradient
  "∇L = a ⊙ (θ - c) at θ, host doubles."
  [theta]
  (mapv (fn [a t ct] (c/* a (c/- t ct))) curvature theta center))

(defn- path
  "The parameters along n steps of optimizer from start, each along f's
   gradient at the store."
  [optimizer n]
  (let [store (store-of start)]
    (loop [i 0 store store state (learn/init optimizer store) out [start]]
      (if (c/< i n)
        (let [[store state] (learn/step optimizer state store (ascent-gradient store))]
          (recur (inc i) store state (conj out (params-of store))))
        out))))

(defn- reference-path
  "The parameters along n steps of a reference rule from start: (rule t θ
   d state) → [θ' state'], t the step from 1 and d ∇L at θ."
  [rule n]
  (loop [t 1 theta start state nil out [start]]
    (if (c/> t n)
      out
      (let [[theta state] (rule t theta (loss-gradient theta) state)]
        (recur (inc t) theta state (conj out theta))))))

(def ^:private steps 40)

(defn- agrees-with?
  "Do optimizer's path and the rule's agree at every step?"
  [optimizer rule]
  (let [ours (path optimizer steps)
        theirs (reference-path rule steps)
        tol (if (ts/f32?) {:rtol 1e-4 :atol 1e-4} {:rtol 1e-12 :atol 1e-12})]
    (every? true? (map #(ts/close? (array/array %1) (array/array %2) tol) ours theirs))))

(defn- zeros [] (vec (repeat (count start) 0.0)))

(defn- sgd-rule [lr]
  (fn [_ theta d _] [(mapv #(c/- %1 (c/* lr %2)) theta d) nil]))

(defn- momentum-rule
  "torch.optim.SGD with momentum μ, dampening 0: b = d at the first step,
   b = μb + d after it; θ - lr·b, or with nesterov θ - lr·(d + μb)."
  [lr mu nesterov?]
  (fn [t theta d b]
    (let [b (if (= t 1) d (mapv #(c/+ (c/* mu %1) %2) b d))
          direction (if nesterov? (mapv #(c/+ %1 (c/* mu %2)) d b) b)]
      [(mapv #(c/- %1 (c/* lr %2)) theta direction) b])))

(defn- rmsprop-rule
  "torch.optim.RMSprop, not centered, no momentum: v = αv + (1 - α)d², then
   θ - lr·d / (√v + eps)."
  [lr alpha eps]
  (fn [_ theta d v]
    (let [v (mapv #(c/+ (c/* alpha %1) (c/* (c/- 1.0 alpha) %2 %2)) (or v (zeros)) d)]
      [(mapv (fn [th di vi] (c/- th (c/* lr (c// di (c/+ (Math/sqrt vi) eps))))) theta d v) v])))

(defn- adagrad-rule
  "torch.optim.Adagrad, no decay, accumulator from 0: s = s + d², then
   θ - lr·d / (√s + eps)."
  [lr eps]
  (fn [_ theta d s]
    (let [s (mapv #(c/+ %1 (c/* %2 %2)) (or s (zeros)) d)]
      [(mapv (fn [th di si] (c/- th (c/* lr (c// di (c/+ (Math/sqrt si) eps))))) theta d s) s])))

(defn- adam-rule
  "torch.optim.Adam (AdamW with decay λ, applied first: θ(1 - lr·λ)), in
   its own arithmetic: m.lerp(d, 1 - β1), v = β2 v + (1 - β2)d², step size
   lr / (1 - β1^t), denominator √v / √(1 - β2^t) + eps."
  [lr beta1 beta2 eps decay]
  (fn [t theta d state]
    (let [{:keys [m v]} (or state {:m (zeros) :v (zeros)})
          m (mapv #(c/+ %1 (c/* (c/- 1.0 beta1) (c/- %2 %1))) m d)
          v (mapv #(c/+ (c/* beta2 %1) (c/* (c/- 1.0 beta2) %2 %2)) v d)
          step-size (c// lr (c/- 1.0 (Math/pow beta1 t)))
          root-c2 (Math/sqrt (c/- 1.0 (Math/pow beta2 t)))
          theta (mapv #(c/* % (c/- 1.0 (c/* lr decay))) theta)]
      [(mapv (fn [th mi vi] (c/- th (c/* step-size (c// mi (c/+ (c// (Math/sqrt vi) root-c2) eps))))) theta m v)
       {:m m :v v}])))

(defn- clipped-rule
  "optax.clip_by_global_norm then rule: d where ‖d‖ < max-norm, else
   d / ‖d‖ · max-norm."
  [max-norm rule]
  (fn [t theta d state]
    (let [norm (Math/sqrt (reduce c/+ (map #(c/* % %) d)))
          d (if (c/< norm max-norm) d (mapv #(c/* (c// % norm) max-norm) d))]
      (rule t theta d state))))

;; Planted bugs (2026-10-25), each in gen.learning's -step and each failing
;; its line: sgd stepping along -g; momentum's velocity βv + (1-β)g;
;; Nesterov's step without the look-ahead (lr·v'); rmsprop with eps inside
;; the root; adagrad accumulating |g| for g²; Adam without the second
;; moment's bias correction; AdamW's decay without lr (θ(1 - λ)); clipping
;; each leaf by its own norm.
(deftest optimizers-follow-their-reference-rules
  (is (agrees-with? (learn/sgd {:lr 0.02}) (sgd-rule 0.02)) "sgd")
  (is (agrees-with? (learn/momentum {:lr 0.01 :momentum 0.8}) (momentum-rule 0.01 0.8 false)) "momentum")
  (is (agrees-with? (learn/momentum {:lr 0.01 :momentum 0.8 :nesterov? true}) (momentum-rule 0.01 0.8 true)) "nesterov")
  (is (agrees-with? (learn/rmsprop {:lr 0.05 :decay 0.9 :eps 1e-3}) (rmsprop-rule 0.05 0.9 1e-3)) "rmsprop")
  (is (agrees-with? (learn/adagrad {:lr 0.5 :eps 1e-3}) (adagrad-rule 0.5 1e-3)) "adagrad")
  (is (agrees-with? (learn/adam {:lr 0.1 :beta1 0.8 :beta2 0.95 :eps 1e-3}) (adam-rule 0.1 0.8 0.95 1e-3 0.0)) "adam")
  (is (agrees-with? (learn/adamw {:lr 0.1 :beta1 0.85 :beta2 0.99 :eps 1e-3 :weight-decay 0.5})
                    (adam-rule 0.1 0.85 0.99 1e-3 0.5))
      "adamw")
  (testing "clipping, active for the first steps (‖∇L‖ = 81 at the start) and not after"
    (is (agrees-with? (learn/clip-norm (learn/sgd {:lr 0.02}) {:max-norm 5.0}) (clipped-rule 5.0 (sgd-rule 0.02))) "sgd")
    (is (agrees-with? (learn/clip-norm (learn/adam {:lr 0.1 :beta1 0.8 :beta2 0.95 :eps 1e-3}) {:max-norm 5.0})
                      (clipped-rule 5.0 (adam-rule 0.1 0.8 0.95 1e-3 0.0)))
        "adam")))

(deftest closed-forms
  ;; independent of the rules above: SGD on a quadratic contracts each
  ;; coordinate by 1 - lr·a per step, and Adam along a constant gradient g
  ;; steps lr·g / (|g| + eps) exactly, its bias corrections undoing the
  ;; moments' start at 0. Planted bug (2026-10-25): Adam's first moment
  ;; without its bias correction fails the second
  (let [n 25
        theta (peek (path (learn/sgd {:lr 0.02}) n))]
    (is (ts/close? (array/array theta)
                   (array/array (mapv (fn [a t0 ct] (c/+ ct (c/* (Math/pow (c/- 1.0 (c/* 0.02 a)) n) (c/- t0 ct))))
                                      curvature start center))
                   {:rtol (ts/tol 1e-12) :atol (ts/tol 1e-12)})))
  (let [g {:w (array/array [2.0 -0.5]) :b 3.0}
        opt (learn/adam {:lr 0.1 :eps 1e-3})
        store (loop [i 0 store {:w (array/array [0.0 0.0]) :b 0.0} state (learn/init opt store)]
                (if (c/< i 30)
                  (let [[store state] (learn/step opt state store g)] (recur (inc i) store state))
                  store))]
    (is (ts/close? (:w store) (array/array [(c/* 30 0.1 (c// 2.0 2.001)) (c/* 30 0.1 (c// -0.5 0.501))])
                   {:rtol (ts/tol 1e-10) :atol (ts/tol 1e-10)}))
    (is (ts/close? (:b store) (c/* 30 0.1 (c// 3.0 3.001)) {:rtol (ts/tol 1e-10) :atol (ts/tol 1e-10)}))))

(deftest defaults-are-pytorchs
  (is (= {:kind :momentum :lr 0.1 :momentum 0.9 :nesterov? false} (learn/momentum {:lr 0.1})))
  (is (= {:kind :rmsprop :lr 0.01 :decay 0.99 :eps 1e-8} (learn/rmsprop {})))
  (is (= {:kind :adagrad :lr 0.01 :eps 1e-10} (learn/adagrad {})))
  (is (= {:kind :adam :lr 0.001 :beta1 0.9 :beta2 0.999 :eps 1e-8} (learn/adam {})))
  (is (= {:kind :adamw :lr 0.001 :beta1 0.9 :beta2 0.999 :eps 1e-8 :weight-decay 0.01} (learn/adamw {}))))

(deftest optimizer-options-are-checked
  ;; loud at construction, where a missing rate used to fail at the first
  ;; step with a NullPointerException
  (is (= {:option :lr :value nil} (ts/thrown-data (learn/sgd {}))))
  (is (= {:option :lr :value nil} (ts/thrown-data (learn/momentum {}))))
  (is (= {:option :momentum :value 1.0} (ts/thrown-data (learn/momentum {:lr 0.1 :momentum 1.0}))))
  (is (= {:option :decay :value 1.5} (ts/thrown-data (learn/rmsprop {:decay 1.5}))))
  (is (= {:option :eps :value 0.0} (ts/thrown-data (learn/adagrad {:eps 0.0}))) "0/0 at a zero gradient")
  (is (= {:option :beta2 :value 1.0} (ts/thrown-data (learn/adam {:beta2 1.0}))))
  (is (= {:option :weight-decay :value -0.1} (ts/thrown-data (learn/adamw {:weight-decay -0.1}))))
  (is (= {:option :max-norm :value 0.0} (ts/thrown-data (learn/clip-norm (learn/sgd {:lr 0.1}) {:max-norm 0.0}))))
  (is (= {:option :step-size-init :value nil} (ts/thrown-data (learn/gradient-descent {:step-size-beta 1.0})))))

(deftest every-optimizer-steps-the-parameters-it-is-given
  (doseq [opt [(learn/momentum {:lr 0.1}) (learn/momentum {:lr 0.1 :nesterov? true}) (learn/rmsprop {})
               (learn/adagrad {}) (learn/adamw {}) (learn/clip-norm (learn/adam {}) {:max-norm 1.0})]]
    (let [store {:w 1.0 :nested {:b 2.0 :c 3.0}}
          [store' _] (learn/step opt (learn/init opt store) store {:w 0.5 :nested {:c -0.5}})]
      (is (= 2.0 (get-in store' [:nested :b])) (str (:kind opt) ": :b has no gradient and stays"))
      (is (c/< 1.0 (ts/number (:w store'))) (str (:kind opt) ": :w ascends"))
      (is (c/> 3.0 (ts/number (get-in store' [:nested :c]))) (str (:kind opt) ": :c ascends, downhill")))))

;; ---------------------------------------------------------------------------
;; The open interface
;; ---------------------------------------------------------------------------

;; momentum written outside gen.learning, as a user would: it must step as
;; the built-in one does, through step, train and VI
(defmethod learn/-init ::heavy-ball [_ store] (array/tree-map array/zeros-like store))

(defmethod learn/-step ::heavy-ball [{:keys [lr mu]} velocity store grads]
  (let [v (learn/update-given #(+ (* mu %1) %2) velocity grads)]
    [(learn/update-given (fn [theta _ vi] (+ theta (* lr vi))) store grads v) v]))

(defmethod learn/-init ::broken [_ _] nil)

(defmethod learn/-step ::broken [_ _ store _] store)

(def ^:private line (gen [x] (trace :y (d/normal (+ (* (param :a) x) (param :b)) 1))))

(defn- line-data [key]
  (let [[kx ky] (rng/split key)
        x (rng/normal kx)]
    [[x] (cm/choicemap :y (+ (* 2.0 x) 1.0 (* 0.3 (rng/normal ky))))]))

(deftest a-users-optimizer-steps-through-train-and-vi
  ;; planted bug (2026-10-25): train calling the built-in kinds directly
  ;; (a case on :kind) fails the first
  (let [ours {:kind ::heavy-ball :lr 0.05 :mu 0.8}
        builtin (learn/momentum {:lr 0.05 :momentum 0.8})
        fit #(learn/train line {:a 0.0 :b 0.0} % line-data grad/param-gradients
                          {:epochs 5 :batches-per-epoch 4 :batch-size 5 :key (rng/key 1)})]
    (is (= (ts/host (:store (fit builtin))) (ts/host (:store (fit ours)))) "train")
    (is (= (:history (fit builtin)) (:history (fit ours)))))
  (let [model (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/normal mu 1))))
        q (gen [] (trace :mu (d/normal (param :m) (array/exp (param :log-s)))))
        vi-fit #(vi/black-box-vi model [] (cm/choicemap :y 1.3) q [] {:m 0.0 :log-s 0.0} %
                                 {:iters 30 :samples 5 :key (rng/key 2)})]
    (is (= (ts/host (:store (vi-fit (learn/momentum {:lr 0.02 :momentum 0.5}))))
           (ts/host (:store (vi-fit {:kind ::heavy-ball :lr 0.02 :mu 0.5}))))
        "VI"))
  (testing "a step that is not [store' state'] is loud"
    (is (= ::broken (:kind (ts/thrown-data (learn/step {:kind ::broken} nil {:w 1.0} {:w 0.1}))))))
  (testing "an unknown kind names the kinds, a user's among them"
    (let [data (ts/thrown-data (learn/init {:kind :rmsprob} {:w 1.0}))]
      (is (= :rmsprob (:kind data)))
      (is (every? (set (:kinds data)) [:sgd :gradient-descent :adam :adamw :momentum :rmsprop :adagrad :clip-norm ::heavy-ball])))
    (is (= :rmsprob (:kind (ts/thrown-data (learn/step {:kind :rmsprob} nil {:w 1.0} {:w 0.1})))))
    (is (= :rmsprob (:kind (ts/thrown-data (learn/init (learn/clip-norm {:kind :rmsprob} {:max-norm 1.0}) {:w 1.0}))))
        "a wrapped one too")))
