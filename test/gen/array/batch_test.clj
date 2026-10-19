(ns gen.array.batch-test
  "The batch tracer against its definition (vmap's): an op run under the
   tracer on batched values equals the op run on each particle's values,
   stacked. Shared (unbatched) operands are the same for every particle."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as a]
            [gen.array.batch :as batch]
            [gen.array.linalg :as la]
            [gen.random :as rng]))

(def n 4)

(defn- particle [x i] (a/take x i 0))

(defn- vmap-matches?
  "f on batched args (the indices in batched-args) under the tracer, against
   f on each particle's slice, stacked."
  [f args batched-args]
  (let [batched-in (map-indexed (fn [i x] (if (batched-args i) (batch/wrap x [n]) x)) args)
        out (batch/batched [n] (apply f batched-in))
        got (batch/unwrap out [n])
        expected (a/stack (mapv (fn [p] (apply f (map-indexed (fn [i x] (if (batched-args i) (particle x p) x)) args)))
                                (range n)))]
    (and (= (a/shape expected) (a/shape got))
         (a/allclose? got expected {:rtol 1e-12 :atol 1e-12}))))

(defn- rand-array [seed shape] (rng/normal (rng/key seed) shape))

(def b3 (rand-array 1 [n 3]))      ; a per-particle vector of 3
(def b23 (rand-array 2 [n 2 3]))   ; a per-particle 2×3 matrix
(def bs (rand-array 3 [n]))        ; a per-particle scalar
(def s3 (rand-array 4 [3]))        ; a shared vector
(def s33 (rand-array 5 [3 3]))
(def spd (a/+ (a/matmul s33 (a/transpose s33)) (a/* 3.0 (a/eye 3))))
(def bspd (a/+ (a/* (a/reshape (a/abs bs) [n 1 1]) (a/eye 3)) spd))

(deftest elementwise
  (is (vmap-matches? a/+ [bs s3] #{0}) "a per-particle scalar meets a shared vector")
  (is (vmap-matches? a/* [b3 s3] #{0}))
  (is (vmap-matches? a/- [b3 b3] #{0 1}))
  (is (vmap-matches? a/exp [b23] #{0}))
  (is (vmap-matches? #(a/where (a/> %1 0) %1 %2) [b3 s3] #{0}))
  (is (vmap-matches? a/pow [(a/abs b3) 2.0] #{0}))
  (is (vmap-matches? #(a/+ % 1.0) [bs] #{0})))

(deftest reductions-and-scans
  (doseq [[label f] [[:sum a/sum] [:sum-0 #(a/sum % {:axis 0})] [:sum-keep #(a/sum % {:axis 1 :keep-dims? true})]
                     [:max #(a/max % {:axis -1})] [:argmax #(a/astype (a/argmax % {:axis 1}) :float64)]
                     [:prod a/prod] [:mean a/mean] [:logsumexp #(a/logsumexp % {:axis -1})]
                     [:softmax a/softmax] [:cumsum #(a/cumsum % {:axis 0})] [:any #(a/astype (a/any (a/> % 0)) :float64)]]]
    (is (vmap-matches? f [b23] #{0}) (str label))))

(deftest shapes
  (doseq [[label f] [[:reshape #(a/reshape % [3 2])] [:transpose a/transpose] [:flatten a/flatten]
                     [:broadcast #(a/broadcast-to % [5 2 3])] [:expand #(a/expand-dims % 0)]
                     [:slice #(a/slice % 1 0 2)] [:slice-step #(a/slice % 1 nil nil -1)]]]
    (is (vmap-matches? f [b23] #{0}) (str label)))
  (is (vmap-matches? #(a/concatenate [%1 %2] 0) [b3 s3] #{0}) "concatenate with a shared operand")
  (is (vmap-matches? #(a/stack [%1 %2]) [bs bs] #{0 1})))

(deftest gathers
  (let [bi (a/array (mapv #(vector (mod % 3) (mod (inc %) 3)) (range n)) {:dtype :int64})
        si (a/array [2 0] {:dtype :int64})
        bk (a/array (mapv #(mod % 3) (range n)) {:dtype :int64})]
    (is (vmap-matches? #(a/take %1 %2 1) [b23 si] #{0}) "batched values, shared indices")
    (is (vmap-matches? #(a/take %1 %2 0) [s3 bi] #{1}) "shared values, batched indices")
    (is (vmap-matches? #(a/take %1 %2 1) [b23 bi] #{0 1}) "both batched")
    (is (vmap-matches? #(a/pick %1 %2) [b3 bk] #{0 1}) "pick: a categorical log-prob")
    (is (vmap-matches? #(a/pick %1 %2) [s3 bk] #{1}))
    (is (vmap-matches? #(a/take-along-axis %1 (a/reshape %2 [1]) 0) [b3 bk] #{0 1}))
    (is (vmap-matches? #(a/scatter-add (a/zeros [3]) (a/reshape %1 [1]) (a/reshape %2 [1]) 0) [bk bs] #{0 1}))))

(deftest matrices
  (is (vmap-matches? a/matmul [b23 s3] #{0}) "a per-particle matrix times a shared vector")
  (is (vmap-matches? a/matmul [s3 b3] #{1}) "vector · vector")
  (is (vmap-matches? a/matmul [b3 s33] #{0}) "a per-particle vector times a shared matrix")
  (is (vmap-matches? #(a/matmul %1 (a/transpose %1)) [b23] #{0}))
  (is (vmap-matches? la/cholesky [bspd] #{0}))
  (is (vmap-matches? #(la/solve %1 %2) [bspd s3] #{0}) "a per-particle matrix, a shared right-hand side")
  (is (vmap-matches? #(la/solve %1 %2) [spd b3] #{1}) "a shared matrix, a per-particle right-hand side")
  (is (vmap-matches? #(la/solve-triangular (la/cholesky %1) %2) [bspd b3] #{0 1}))
  (is (vmap-matches? #(second (la/slogdet %)) [bspd] #{0})))

(deftest draws-are-per-particle
  (let [x (batch/unwrap (batch/batched [n] (rng/normal (rng/key 9) [3])) [n])]
    (is (= [n 3] (a/shape x)))
    (is (= n (count (distinct (map first (a/->clj x))))) "independent draws, not one shared draw")))

(deftest host-reads-are-the-fallback-signal
  (let [x (batch/wrap bs [n])]
    (doseq [[label f] [[:item #(a/item x)] [:->clj #(a/->clj x)] [:truthy-of-a-bool #(a/truthy? (a/> x 0))]
                       [:= #(= x 0)] [:hash #(hash x)]]]
      (is (:gen.array/batched-branch (ex-data (try (batch/batched [n] (f)) nil (catch clojure.lang.ExceptionInfo e e))))
          (str label)))
    (is (true? (batch/batched [n] (a/truthy? x))) "a batch of numbers is truthy, as each number is")))

(deftest third-review-ops
  (testing "B4: linear algebra with different per-particle batch ranks"
    (let [b233 (a/+ (a/* 3.0 (a/eye 3)) (a/* 0.1 (rand-array 6 [n 2 3 3])))]
      (is (vmap-matches? #(la/solve %1 %2) [b233 s3] #{0}) "per-particle [2 3 3] against a shared [3]")
      (is (vmap-matches? #(la/solve %1 %2) [b233 b3] #{0 1}) "against a per-particle [3]")))
  (testing "B9: scans and arg-reductions of per-particle scalars"
    (is (vmap-matches? a/cumsum [bs] #{0}))
    (is (vmap-matches? #(a/astype (a/argmax %) :float64) [bs] #{0})))
  (testing "a per-particle fill value"
    (is (vmap-matches? #(a/full [2] %) [bs] #{0})))
  (testing "a gradient of unbatched leaves inside a batched run"
    (is (= 6.0 (ts/host (batch/batched [n] ((a/grad (fn [x] (a/* x x))) 3.0)))))))
