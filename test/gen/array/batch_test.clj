(ns gen.array.batch-test
  "The batch tracer against its definition (vmap's): an op run under the
   tracer on batched values equals the op run on each particle's values,
   stacked. Shared (unbatched) operands are the same for every particle."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array.protocols :as p]
            [gen.test-support :as ts]
            [gen.array :as array]
            [gen.array.batch :as batch]
            [gen.array.linalg :as la]
            [gen.random :as rng]))

(def n 4)

(defn- particle [x i] (array/take x i 0))

(defn- vmap-matches?
  "f on batched args (the indices in batched-args) under the tracer, against
   f on each particle's slice, stacked."
  [f args batched-args]
  (let [batched-in (map-indexed (fn [i x] (if (batched-args i) (batch/wrap x [n]) x)) args)
        out (batch/batched [n] (apply f batched-in))
        got (batch/unwrap out [n])
        expected (array/stack (mapv (fn [p] (apply f (map-indexed (fn [i x] (if (batched-args i) (particle x p) x)) args)))
                                    (range n)))]
    (and (= (array/shape expected) (array/shape got))
         (ts/close? got expected))))

(defn- rand-array [seed shape] (rng/normal (rng/key seed) shape))

(def b3 (rand-array 1 [n 3]))      ; a per-particle vector of 3
(def b23 (rand-array 2 [n 2 3]))   ; a per-particle 2×3 matrix
(def bs (rand-array 3 [n]))        ; a per-particle scalar
(def s3 (rand-array 4 [3]))        ; a shared vector
(def s33 (rand-array 5 [3 3]))
(def spd (array/+ (array/matmul s33 (array/transpose s33)) (array/* 3.0 (array/eye 3))))
(def bspd (array/+ (array/* (array/reshape (array/abs bs) [n 1 1]) (array/eye 3)) spd))

(deftest elementwise
  (is (vmap-matches? array/+ [bs s3] #{0}) "a per-particle scalar meets a shared vector")
  (is (vmap-matches? array/* [b3 s3] #{0}))
  (is (vmap-matches? array/- [b3 b3] #{0 1}))
  (is (vmap-matches? array/exp [b23] #{0}))
  (is (vmap-matches? #(array/where (array/> %1 0) %1 %2) [b3 s3] #{0}))
  (is (vmap-matches? array/pow [(array/abs b3) 2.0] #{0}))
  (is (vmap-matches? array/atan2 [b3 s3] #{0}) "atan2: a per-particle y, a shared x")
  (is (vmap-matches? array/asin [(array/tanh b23)] #{0}))
  (is (vmap-matches? #(array/+ % 1.0) [bs] #{0})))

(deftest a-batched-op-takes-its-particles-implementation
  ;; the tracer reports the wrapped backend's native ops, so an op with a
  ;; native form takes it batched as unbatched (genclj-bk62): on MLX,
  ;; logaddexp took MLX's op unbatched and the composite batched, whose
  ;; gradient at ties is 0 under MLX's autodiff where the op's is 1/2.
  ;; Planted (2026-10-22): the tracer's #{} again fails both, on MLX
  ;; (the second with -Dgen.mlx.grad=native)
  (let [inner array/*backend*
        xs (array/array [1.0 2.5 -3.0])
        traced (atom nil)]
    (batch/batched [3] (reset! traced (p/-native-ops (array/backend (batch/wrap xs [3])))))
    (is (= (p/-native-ops inner) @traced))
    (testing "logaddexp's gradient at ties, batched as unbatched"
      (let [grad-x (fn [f] (ts/host (first ((array/grad f {:argnums [0 1]}) xs xs))))]
        (is (= (grad-x (fn [x y] (array/sum (array/logaddexp x y))))
               (grad-x (fn [x y] (array/sum (batch/unwrap (batch/batched [3] (array/logaddexp (batch/wrap x [3]) (batch/wrap y [3]))) [3]))))))))))

(deftest reductions-and-scans
  (doseq [[label f] [[:sum array/sum] [:sum-0 #(array/sum % {:axis 0})] [:sum-keep #(array/sum % {:axis 1 :keep-dims? true})]
                     [:max #(array/max % {:axis -1})] [:argmax #(array/astype (array/argmax % {:axis 1}) (ts/float-dtype))]
                     [:prod array/prod] [:mean array/mean] [:logsumexp #(array/logsumexp % {:axis -1})]
                     [:softmax array/softmax] [:cumsum #(array/cumsum % {:axis 0})] [:any #(array/astype (array/any (array/> % 0)) (ts/float-dtype))]]]
    (is (vmap-matches? f [b23] #{0}) (str label))))

(deftest shapes
  (doseq [[label f] [[:reshape #(array/reshape % [3 2])] [:transpose array/transpose] [:flatten array/flatten]
                     [:broadcast #(array/broadcast-to % [5 2 3])] [:expand #(array/expand-dims % 0)]
                     [:slice #(array/slice % 1 0 2)] [:slice-step #(array/slice % 1 nil nil -1)]]]
    (is (vmap-matches? f [b23] #{0}) (str label)))
  (is (vmap-matches? #(array/concatenate [%1 %2] 0) [b3 s3] #{0}) "concatenate with a shared operand")
  (is (vmap-matches? #(array/stack [%1 %2]) [bs bs] #{0 1})))

(deftest gathers
  (let [bi (array/array (mapv #(vector (mod % 3) (mod (inc %) 3)) (range n)) {:dtype :int64})
        si (array/array [2 0] {:dtype :int64})
        bk (array/array (mapv #(mod % 3) (range n)) {:dtype :int64})]
    (is (vmap-matches? #(array/take %1 %2 1) [b23 si] #{0}) "batched values, shared indices")
    (is (vmap-matches? #(array/take %1 %2 0) [s3 bi] #{1}) "shared values, batched indices")
    (is (vmap-matches? #(array/take %1 %2 1) [b23 bi] #{0 1}) "both batched")
    (is (vmap-matches? #(array/pick %1 %2) [b3 bk] #{0 1}) "pick: a categorical log-prob")
    (is (vmap-matches? #(array/pick %1 %2) [s3 bk] #{1}))
    (is (vmap-matches? #(array/take-along-axis %1 (array/reshape %2 [1]) 0) [b3 bk] #{0 1}))
    (is (vmap-matches? #(array/scatter-add (array/zeros [3]) (array/reshape %1 [1]) (array/reshape %2 [1]) 0) [bk bs] #{0 1}))))

(deftest matrices
  (is (vmap-matches? array/matmul [b23 s3] #{0}) "a per-particle matrix times a shared vector")
  (is (vmap-matches? array/matmul [s3 b3] #{1}) "vector · vector")
  (is (vmap-matches? array/matmul [b3 s33] #{0}) "a per-particle vector times a shared matrix")
  (is (vmap-matches? #(array/matmul %1 (array/transpose %1)) [b23] #{0}))
  (is (vmap-matches? la/cholesky [bspd] #{0}))
  (is (vmap-matches? #(la/solve %1 %2) [bspd s3] #{0}) "a per-particle matrix, a shared right-hand side")
  (is (vmap-matches? #(la/solve %1 %2) [spd b3] #{1}) "a shared matrix, a per-particle right-hand side")
  (is (vmap-matches? #(la/solve-triangular (la/cholesky %1) %2) [bspd b3] #{0 1}))
  (is (vmap-matches? #(second (la/slogdet %)) [bspd] #{0})))

(deftest draws-are-per-particle
  (let [x (batch/unwrap (batch/batched [n] (rng/normal (rng/key 9) [3])) [n])]
    (is (= [n 3] (array/shape x)))
    (is (= n (count (distinct (map first (array/->clj x))))) "independent draws, not one shared draw")))

(deftest all-particles-reads-across-the-batch
  ;; genclj-z6iy: gen.dist read a batch's particles through the backend
  ;; protocol; array/all-particles is the public reach, and all-true? and
  ;; any-true? read through it (planted: a batched value as it is)
  (let [x (batch/wrap b3 [n])]
    (is (ts/close? b3 (batch/batched [n] (array/all-particles x))))
    (is (= [n 3] (array/shape (batch/batched [n] (array/all-particles x)))))
    (is (identical? s3 (array/all-particles s3)) "an unbatched value as it is")
    (is (batch/batched [n] (array/all-true? (array/< x 100.0))))
    (is (not (batch/batched [n] (array/all-true? (array/< x (array/max b3))))) "one particle's element fails")))

(deftest host-reads-are-the-fallback-signal
  (let [x (batch/wrap bs [n])]
    (doseq [[label f] [[:item #(array/item x)] [:->clj #(array/->clj x)] [:truthy-of-a-bool #(array/truthy? (array/> x 0))]
                       [:= #(= x 0)] [:hash #(hash x)]]]
      (is (:gen.array/batched-branch (ex-data (try (batch/batched [n] (f)) nil (catch clojure.lang.ExceptionInfo e e))))
          (str label)))
    (is (true? (batch/batched [n] (array/truthy? x))) "a batch of numbers is truthy, as each number is")))

(deftest third-review-ops
  (testing "B4: linear algebra with different per-particle batch ranks"
    (let [b233 (array/+ (array/* 3.0 (array/eye 3)) (array/* 0.1 (rand-array 6 [n 2 3 3])))]
      (is (vmap-matches? #(la/solve %1 %2) [b233 s3] #{0}) "per-particle [2 3 3] against a shared [3]")
      (is (vmap-matches? #(la/solve %1 %2) [b233 b3] #{0 1}) "against a per-particle [3]")))
  (testing "B9: scans and arg-reductions of per-particle scalars"
    (is (vmap-matches? array/cumsum [bs] #{0}))
    (is (vmap-matches? #(array/astype (array/argmax %) (ts/float-dtype)) [bs] #{0})))
  (testing "a per-particle fill value"
    (is (vmap-matches? #(array/full [2] %) [bs] #{0})))
  (testing "a gradient of unbatched leaves inside a batched run"
    (is (= 6.0 (ts/host (batch/batched [n] ((array/grad (fn [x] (array/* x x))) 3.0)))))))
