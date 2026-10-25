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

(deftest shape-errors-name-the-model-s-shapes
  ;; gen.array and gen.array.linalg check shapes before they call a backend
  ;; (genclj-q45d, the review's I1-02), so under the tracer a mismatch is the
  ;; error the model would get unbatched, naming its per-particle shapes.
  ;; The backends checked, on the lifted shapes: [4 1 3] against [4 2 2],
  ;; and a take-along-axis with indices of lower rank ran, lifted to the
  ;; array's rank. Planted (2026-10-22): each check taken out of gen.array
  ;; or gen.array.linalg fails its cases, and the tracer's own check of
  ;; elementwise operands (gen.array leaves those to the backends, the hot
  ;; path) fails elementwise and where
  (let [error (fn [f] (try (f) :no-error (catch clojure.lang.ExceptionInfo e [(ex-message e) (ex-data e)])))
        ints #(array/astype % :int32)
        broadcast (fn [a b] [(str "Shapes " a " and " b " do not broadcast") {:shapes [a b]}])
        square (fn [op shape] [(str (name op) " needs square matrices, got shape " shape) {:op op :shape shape}])
        rhs (fn [op shapes] [(str (name op) ": matrix and right-hand side do not match") {:shapes shapes}])]
    (doseq [[label f shapes want]
            [["elementwise" array/+ [[3] [2 2]] (broadcast [3] [2 2])]
             ["where" #(array/where (array/> %1 0) %2 %3) [[3] [2 2] [2]] (broadcast [3] [2 2])]
             ["broadcast-to" #(array/broadcast-to % [2 2]) [[3]] (broadcast [3] [2 2])]
             ["concatenate" #(array/concatenate [%1 %2] 0) [[3 2] [3 4]]
              ["Arrays must agree on every axis but the concatenation axis" {:shapes [[3 2] [3 4]] :axis 0}]]
             ["take-along-axis, ranks" #(array/take-along-axis %1 (ints %2) 0) [[3 2] [2]]
              ["take-along-axis needs indices of the same rank as the array" {:shape [3 2] :indices-shape [2]}]]
             ["take-along-axis, other axes" #(array/take-along-axis %1 (ints %2) 0) [[3 2] [2 3]] (broadcast [2] [3])]
             ["pick" #(array/pick %1 (ints %2)) [[2 3] [3]] (broadcast [2] [3])]
             ["scatter-add" #(array/scatter-add %1 (ints %2) %3 0) [[3 2] [2 3] [2 3]]
              ["scatter-add needs indices and updates of one shape, equal to the array's but on axis"
               {:shape [3 2] :indices-shape [2 3] :updates-shape [2 3] :axis 0}]]
             ["matmul" array/matmul [[2 3] [2 3]] ["matmul: inner dimensions differ (3 vs 2)" {:shapes [[2 3] [2 3]]}]]
             ["matmul of a scalar" array/matmul [[] [3]] ["matmul needs arrays of rank >= 1" {:shapes [[] [3]]}]]
             ["cholesky" la/cholesky [[2 3]] (square :cholesky [2 3])]
             ["slogdet" la/slogdet [[3]] (square :slogdet [3])]
             ["inv" la/inv [[2 3]] (square :inv [2 3])]
             ["solve, a vector matrix" la/solve [[3] [3]] (square :solve [3])]
             ["solve, rows" la/solve [[3 3] [2]] (rhs :solve [[3 3] [2]])]
             ["solve-triangular, rows" la/solve-triangular [[3 3] [2 1]] (rhs :solve-triangular [[3 3] [2 1]])]
             ["solve, batch axes" la/solve [[2 3 3] [5 3 1]] (broadcast [2] [5])]]]
      (is (= want (error #(apply f (map array/zeros shapes)))) (str label ", unbatched"))
      (is (= want (error #(batch/batched [n] (apply f (map (fn [s] (batch/wrap (array/zeros (into [n] s)) [n])) shapes)))))
          (str label ", batched")))))

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

(deftest particles-of-a-batched-value
  ;; array/batched?, batch-shape, particle-backend, particle and particles:
  ;; the public reach of a batched value's particles, as a host function
  ;; spot-checked inside a batched run reads them (memo.clj's memo.lower,
  ;; genclj-9efy). Planted (2026-10-25), each failing its case: batched? not
  ;; seeing through the tape; a tracked particle read as its primal; a
  ;; batch's axes counted column-major; particles' indices sorted; no index
  ;; check; particle-backend giving *backend* (the tracer, in a run)
  (let [x (batch/wrap b3 [n])
        y (batch/batched [n] (array/* x 2.0))]
    (testing "batched? and batch-shape"
      (is (array/batched? x))
      (is (array/batched? y) "a value the run computed")
      (is (= [n] (array/batch-shape y)))
      (is (not-any? array/batched? [s3 1.5 true :k nil (array/array [1.0])]))
      (is (nil? (array/batch-shape s3))))
    (testing "particle i: its own value, unbatched, of its own shape"
      (doseq [i (range n)]
        (let [p (array/particle y i)]
          (is (not (array/batched? p)))
          (is (= [3] (array/shape p)))
          (is (ts/close? (array/* (array/take b3 i 0) 2.0) p) (str "particle " i))))
      (is (identical? s3 (array/particle s3 2)) "a shared value as it is")
      (is (= :k (array/particle :k 0)) "any value"))
    (testing "particles: their values stacked, in their order, repeats allowed"
      (is (ts/close? (array/take b3 (array/array [3 0 3]) 0) (array/particles x [3 0 3])))
      (is (= [0 3] (array/shape (array/particles x []))))
      (is (ts/close? (array/stack [s3 s3]) (array/particles s3 [1 2])) "a shared value broadcast"))
    (testing "inside the run, where reading the batched value is the batched-branch error"
      (is (= (ts/host (array/take b3 1 0)) (batch/batched [n] (array/->clj (array/particle (array/+ x 0.0) 1)))))
      (is (= (ts/host (array/take b3 (array/array [2 1]) 0))
             (batch/batched [n] (array/->clj (array/particles (array/+ x 0.0) [2 1]))))))
    (testing "particle-backend: the backend the batch wraps, where code apart from the run draws unbatched"
      (is (identical? (array/backend b3) (array/particle-backend y)))
      (is (nil? (array/particle-backend s3)))
      (batch/batched [n]
        (is (array/batched? (rng/normal (rng/key 1) [2])) "in the run, a draw is per particle")
        (is (not (array/batched? (array/with-backend (array/particle-backend x) (rng/normal (rng/key 1) [2])))))))
    (testing "a batch of several axes counts its particles row-major"
      (let [a (rand-array 7 [2 3 4])
            z (batch/wrap a [2 3])]
        (is (= [2 3] (array/batch-shape z)))
        (doseq [i (range 6)]
          (is (ts/close? (array/take (array/take a (quot i 3) 0) (rem i 3) 0) (array/particle z i)) (str "particle " i)))
        (is (= [2 4] (array/shape (array/particles z [5 1]))))))
    (testing "a lazily gathered value (resampling's)"
      (let [g (batch/gather x (array/array [2 0]) [2])]
        (is (= [2] (array/batch-shape g)))
        (is (ts/close? (array/take b3 2 0) (array/particle g 0)))))
    (testing "loud: an index that is no particle"
      (doseq [i [n -1 1.0]]
        (is (= {:op :particle :index i :particles n} (ts/thrown-data (array/particle x i))) (pr-str i)))
      (is (= {:op :particles :index n :particles n} (ts/thrown-data (array/particles x [0 n]))))
      (is (= {:op :particles :index -1 :particles nil} (ts/thrown-data (array/particles s3 [-1]))) "a shared value's too")
      (is (= {:op :particles :value :k} (ts/thrown-data (array/particles :k [0]))) "a shared value that is no array"))
    (when-not (ts/native-grad?)
      (testing "a value being differentiated: batched when its primal is, its particles refused"
        (let [seen (volatile! nil)]
          (batch/batched [n]
            ((array/grad (fn [t]
                           (let [v (array/* t x)]
                             (vreset! seen {:tracked? (array/tracked? v)
                                            :batched? (array/batched? v)
                                            :batch-shape (array/batch-shape v)
                                            :particle (:op (ts/thrown-data (array/particle v 0)))
                                            :particles (:op (ts/thrown-data (array/particles v [0])))
                                            :primal (array/->clj (array/particle (array/stop-gradient v) 1))
                                            :shared (identical? t (array/particle t 0))})
                             (array/sum v))))
             1.5))
          (is (= {:tracked? true :batched? true :batch-shape [n] :particle :particle :particles :particles
                  :primal (ts/host (array/* 1.5 (array/take b3 1 0))) :shared true}
                 @seen)))))))

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
