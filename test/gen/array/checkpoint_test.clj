(ns gen.array.checkpoint-test
  "gen.array/checkpoint (experimental): a function whose gradient
   recomputes it in the backward pass. On the backend under test, and its
   gradients (the tape, or MLX's native ones with -Dgen.mlx.grad=native):
   1. a stack of checkpointed residual layers (trees of weights, with an
      int array and a host number passed through) has the unchecked
      stack's value and gradients: exactly on the JVM backend (the same
      ops in the same order), within float32's band on MLX; nested
      checkpoints too;
   2. under value-and-grad f runs twice, the forward and the recompute,
      and once elsewhere (outside a gradient, inside array/compile under
      native gradients);
   3. under the batch tracer (vmap of a gradient, and of the loss) the
      same per-particle values and gradients;
   4. misuse is loud: f returning no float array, and on the tape f
      closing over a value being differentiated.
   Planted bugs, each failing 1 (2026-11-07, the JVM backend; they run on
   the tape): the recompute reading the layer's output for its input (the
   gradients off by 4.9), and the recompute's backward seeded with ones,
   the cotangent dropped (off by 12.7)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.tape :as tape]
            [gen.test-support :as ts]))

(defn- host [x] (ts/host x))

(def ^:private d 6)

(defn- weights
  "Layer i's weights: a matrix and a bias (floats, differentiated), a
   permutation of the features (an int array) and a scale (a host number),
   passed through."
  [i]
  {:w (array/array (vec (for [r (range d)] (vec (for [c (range d)] (* 0.4 (Math/sin (+ (* 1.7 i) (* 0.9 r) (* 0.31 c)))))))))
   :b (array/array (vec (for [c (range d)] (* 0.1 (Math/cos (+ i c))))))
   :perm (array/array (vec (for [c (range d)] (mod (+ c i) d))) {:dtype (ts/int-dtype)})
   :scale (+ 0.5 (* 0.25 i))})

(defn- layer
  "x [n d] through a residual layer: x + scale·tanh(x·w + b), its
   features permuted."
  [x {:keys [w b perm scale]}]
  (array/+ x (array/* scale (array/take (array/tanh (array/+ (array/matmul x w) b)) perm 1))))

(def ^:private ws
  "The layers' weights differentiated: each layer's matrix and bias."
  (mapv #(select-keys (weights %) [:w :b]) (range 4)))

(def ^:private constants
  "The layers' leaves passed through: a permutation and a scale each."
  (mapv #(select-keys (weights %) [:perm :scale]) (range 4)))

(def ^:private x0
  (array/array (vec (for [r (range 3)] (vec (for [c (range d)] (Math/sin (+ (* 2.3 r) c))))))))

(defn- loss
  "The stack's loss: Σ (x after the layers, block-of each)²."
  [block-of]
  (fn [ws x] (array/sum (array/square (reduce block-of x (map merge ws constants))))))

(defn- value-and-grads [block-of]
  ((array/value-and-grad (loss block-of) {:argnums [0 1]}) ws x0))

(defn- gap
  "The largest |Δ| between two results' values and gradients."
  [a b]
  (reduce max (map (fn [x y] (Math/abs (- (double x) (double y))))
                   (flatten (host (array/tree-leaves a))) (flatten (host (array/tree-leaves b))))))

(def ^:private band
  "On the JVM backend the same ops in the same order: exactly. On MLX,
   a few float32 ulps of gradients up to about 20 (measured 0 on fenrir's
   CUDA and CPU, native and tape gradients, 2026-11-07; not yet on
   Metal)."
  (if (ts/jvm?) 0.0 1e-5))

(deftest checkpointed-gradients-are-the-unchecked-ones
  (let [plain (value-and-grads layer)]
    (is (<= (gap plain (value-and-grads (array/checkpoint layer))) band) "each layer checkpointed")
    (is (<= (gap plain (value-and-grads (array/checkpoint #((array/checkpoint layer) %1 %2)))) band) "nested")
    (testing "two layers in one checkpoint, and a layer of host-number inputs"
      (let [two (array/checkpoint (fn [x [w1 w2]] (layer (layer x w1) w2)))
            ck (fn [ws x] (let [ws (mapv merge ws constants)]
                            (array/sum (array/square (two (two x (subvec ws 0 2)) (subvec ws 2 4))))))]
        (is (<= (gap plain ((array/value-and-grad ck {:argnums [0 1]}) ws x0)) band))
        (is (ts/close? ((array/grad (array/checkpoint (fn [a b] (array/* a (array/sin b))))) 2.0 0.5)
                       (Math/sin 0.5)))))
    (when (ts/native-grad?)
      (testing "MLX's own mlx_checkpoint, for the memory bench: the same gradients"
        (with-bindings {(requiring-resolve 'gen.backend.mlx.transforms/*checkpoint-order*) :plain}
          (is (<= (gap plain (value-and-grads (array/checkpoint layer))) band)))))
    (testing "the forward's value is f's"
      (let [ws' (mapv merge ws constants)]
        (is (<= (gap [(reduce layer x0 ws')] [(reduce (array/checkpoint layer) x0 ws')]) band))))))

(deftest a-checkpoint-recomputes-in-the-backward-pass
  (let [calls (atom 0)
        counted (array/checkpoint (fn [x w] (swap! calls inc) (layer x w)))]
    ((array/value-and-grad (loss counted) {:argnums [0 1]}) ws x0)
    (is (= 8 @calls) "4 layers, each run forward and recomputed")
    (reset! calls 0)
    (array/eval! (reduce counted x0 (map merge ws constants)))
    (is (= 4 @calls) "outside a gradient, once")
    (when (ts/native-grad?)
      (testing "inside array/compile under native gradients, f itself: the same gradient"
        (reset! calls 0)
        (let [g (array/compile (fn [ws x] ((array/grad (loss counted)) ws x)))]
          (is (<= (gap ((array/grad (loss layer)) ws x0) (g ws x0)) band))
          (is (= 4 @calls) "traced once, f called once a layer"))))))

(deftest batched-gradients-are-per-particle
  (let [xs (array/stack [x0 (array/* 0.5 x0) (array/- x0 0.25)] 0)
        per-row (fn [block-of] ((array/vmap (fn [x] ((array/grad (loss block-of) {:argnums [0 1]}) ws x))) xs))
        forward (fn [block-of] ((array/vmap (fn [x] ((loss block-of) ws x))) xs))]
    (is (<= (gap (per-row layer) (per-row (array/checkpoint layer))) band) "vmap of a gradient")
    (is (<= (gap [(forward layer)] [(forward (array/checkpoint layer))]) band) "vmap of the loss")))

(deftest misuse-is-loud
  (is (= :invalid-result (:reason (ts/thrown-data ((array/checkpoint (fn [x] [x x])) x0)))) "a vector")
  (is (= :invalid-result (:reason (ts/thrown-data ((array/checkpoint (fn [x] (array/> x 0))) x0)))) "a bool array")
  (when-not (ts/native-grad?)
    (testing "on the tape, f closing over a value being differentiated (under native gradients its gradient is lost)"
      (is (= :closes-over (:reason (ts/thrown-data ((array/grad (fn [w x] (array/sum ((array/checkpoint (fn [x] (array/* x w))) x)))
                                                                {:argnums [0 1]})
                                                    (array/ones [d]) x0))))))))

(deftest planted-bugs-fail-it
  (when-not (ts/native-grad?)
    (let [recompute @#'tape/recompute-vjp
          plain (value-and-grads layer)
          off (fn [redefs] (with-redefs-fn redefs #(gap plain (value-and-grads (array/checkpoint layer)))))]
      (is (< 0.1 (off {#'tape/recompute-vjp (fn [inner f ps g need] (recompute inner f (assoc ps 0 (f ps)) g need))}))
          "the recompute reads the layer's output for its input")
      (is (< 0.1 (off {#'tape/recompute-vjp (fn [inner f ps g need] (recompute inner f ps (array/ones-like g) need))}))
          "the cotangent dropped: the recompute's backward seeded with ones"))))
