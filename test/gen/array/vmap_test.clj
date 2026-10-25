(ns gen.array.vmap-test
  "array/vmap against its definition: f batched over the leading axis of
   its batched arguments equals f run on each row of them (the shared ones
   as they are), stacked. On every backend.

   Planted (2026-10-25, genclj-t3ur), each failing the tests named:
   - in-axes ignored, every argument batched: shared-arguments;
   - a shared leaf of the value returned unbroadcast: trees-of-results;
   - no check that the batch sizes agree: mismatches-are-loud;
   - the unread-comparison guard dropped: branching-on-a-value-is-loud;
   - a caught batched-branch error not rethrown: the same;
   - f run off the batch backend (draws unbatched): draws-are-per-particle;
   - the nested-run refusal dropped: nested-vmap-is-refused;
   - the tracked-argument refusal dropped, and the tape's mixed-backend
     error left unnamed: a-gradient-through-a-vmap-is-refused."
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * / < > ==]]
            [gen.array.linalg :as la]
            [gen.batched :as gb]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(def n 5)

(defn- rand-array [seed shape] (rng/normal (rng/key seed) shape))

(def b3 (rand-array 1 [n 3]))      ; a per-particle vector of 3
(def b23 (rand-array 2 [n 2 3]))   ; a per-particle 2×3 matrix
(def bs (rand-array 3 [n]))        ; a per-particle scalar
(def s3 (rand-array 4 [3]))        ; a shared vector
(def s33 (array/+ (array/* 4.0 (array/eye 3)) (rand-array 5 [3 3])))

(defn- per-row
  "f on row i of each batched argument (in-axis 0) and the shared ones as
   they are, for every row, stacked: what vmap computes."
  [f in-axes args]
  (let [rows (first (keep (fn [[x axis]] (when axis (first (array/shape x)))) (map vector args in-axes)))]
    (array/stack (mapv (fn [i] (apply f (map (fn [x axis] (if axis (array/take x i 0) x)) args in-axes)))
                       (range rows)))))

(defn- matches?
  "((vmap f {:in-axes in-axes}) args) against per-row, in shape and value."
  [f in-axes args]
  (let [got (apply (array/vmap f {:in-axes in-axes}) args)
        expected (per-row f in-axes args)]
    (and (= (array/shape expected) (array/shape got))
         (ts/close? got expected))))

(deftest elementwise-and-reductions-match-a-per-row-loop
  (doseq [[label f x] [[:exp array/exp b3]
                       [:affine #(+ (* 2.0 %) 1.0) b23]
                       [:sum array/sum b23]
                       [:sum-axis #(array/sum % {:axis 0}) b23]
                       [:max-last #(array/max % {:axis -1}) b23]
                       [:logsumexp array/logsumexp b3]
                       [:softmax array/softmax b3]
                       [:cumsum array/cumsum b3]
                       [:norm #(array/sqrt (array/sum (array/square %))) b3]
                       [:matmul-shared #(array/matmul s33 %) b3]
                       [:solve-shared #(la/solve s33 %) b3]
                       [:log-density #(dc/log-prob (d/normal 0 1) %) bs]]]
    (is (matches? f [0] [x]) (str label)))
  (testing "the default batches every argument"
    (is (ts/close? (per-row * [0 0] [b3 b3]) ((array/vmap *) b3 b3))))
  (testing "nothing is evaluated: on a lazy backend the value is lazy"
    (is (= (array/concrete? b3) (array/concrete? ((array/vmap array/sum) b3))))))

(deftest shared-arguments
  (testing "a per-particle scalar meets a shared vector by its per-particle axes"
    ;; n = 3 and a shared vector of 3: batching by shape would add them element by element
    (let [x (array/array [0.0 1.0 2.0])
          shared (array/array [10.0 20.0 30.0])]
      (is (= [[10.0 20.0 30.0] [11.0 21.0 31.0] [12.0 22.0 32.0]]
             (ts/host ((array/vmap + {:in-axes [0 nil]}) x shared))))))
  (is (matches? (fn [x v] (+ x v)) [0 nil] [bs s3]))
  (is (matches? (fn [v x] (array/matmul v x)) [nil 0] [s3 (rand-array 6 [n 3 2])]))
  (testing "a shared argument is any value: a host number, options, a function"
    (is (matches? (fn [x c] (* x c)) [0 nil] [b3 2.5]))
    (is (matches? (fn [x opts] (array/sum x opts)) [0 nil] [b23 {:axis 1}]))
    (is (matches? (fn [g x] (g x)) [nil 0] [array/tanh b3])))
  (testing "a batched argument may be a tree of arrays, every leaf of it batched"
    (is (ts/close? (per-row (fn [mu sigma s] (+ mu (* sigma (array/sum s)))) [0 0 0] [bs b3 b23])
                   ((array/vmap (fn [{:keys [mu sigma]} s] (+ mu (* sigma (array/sum s)))))
                    {:mu bs :sigma b3} b23)))
    (is (ts/close? (per-row (fn [x y] (+ (array/sum x) y)) [0 0] [b3 bs])
                   ((array/vmap (fn [[x y]] (+ (array/sum x) y))) [b3 bs])))))

(deftest trees-of-results
  (let [{:keys [s v k shared flag]} ((array/vmap (fn [x] {:s (array/sum x) :v [x (* 2.0 x)] :k 1.5 :shared s3 :flag true}))
                                     b3)]
    (testing "batched leaves: the particles' values"
      (is (ts/close? (per-row array/sum [0] [b3]) s))
      (is (= [[n 3] [n 3]] (mapv array/shape v)))
      (is (ts/close? (* 2.0 b3) (second v))))
    (testing "shared leaves, a host number among them, broadcast to every particle"
      (is (= [n] (array/shape k)))
      (is (= (ts/float-dtype) (array/dtype k)) "a host float, weakly typed")
      (is (ts/close? (array/full [n] 1.5) k))
      (is (= [n 3] (array/shape shared)))
      (is (ts/close? (array/broadcast-to s3 [n 3]) shared))
      (is (= (vec (repeat n true)) (ts/host flag)))))
  (testing "a host integer takes the default int"
    (is (= (ts/int-dtype) (array/dtype ((array/vmap (fn [_] 2)) bs)))))
  (testing "a batched comparison handed back is a result, not a branch"
    (is (= (mapv neg? (ts/host bs)) (ts/host ((array/vmap #(< % 0.0)) bs)))))
  (testing "an empty batch"
    (is (= [0] (array/shape ((array/vmap array/sum) (array/zeros [0 3])))))
    (is (= [0 3] (array/shape ((array/vmap (fn [_] s3)) (array/zeros [0]))))))
  (testing "a value that is no tree of arrays is loud"
    (doseq [v [:k nil "s" {:a [1.0 :b]}]]
      (is (= :vmap (:op (ts/thrown-data ((array/vmap (fn [_] v)) bs)))) (pr-str v)))))

(deftest mismatches-are-loud
  (testing "batched arguments disagreeing on the batch size"
    (is (= {:op :vmap :sizes [[0 n] [1 3]]} (ts/thrown-data ((array/vmap +) b3 (rand-array 7 [3 3])))))
    (is (= {:op :vmap :sizes [[0 n] [0 2]]} (ts/thrown-data ((array/vmap (fn [m] m)) {:a b3 :b (rand-array 8 [2])})))
        "within one argument's tree"))
  (testing "a batched leaf without a leading axis"
    (is (= {:op :vmap :argument 1 :value 1.0} (ts/thrown-data ((array/vmap +) b3 1.0))))
    (is (= [:vmap 0] ((juxt :op :argument) (ts/thrown-data ((array/vmap array/exp) (array/sum b3)))))
        "a 0-d array (on the JVM backend, a host number)")
    (is (= {:op :vmap :argument 0 :value :k} (ts/thrown-data ((array/vmap (fn [m] (:x m))) {:x b3 :tag :k})))))
  (testing "no batched argument"
    (is (= {:op :vmap :in-axes [nil]} (ts/thrown-data ((array/vmap array/exp {:in-axes [nil]}) b3))))
    (is (= {:op :vmap :in-axes []} (ts/thrown-data ((array/vmap (fn [] 1.0))))))
    (is (= {:op :vmap :in-axes [0]} (ts/thrown-data ((array/vmap (fn [m] m)) {})))))
  (testing "in-axes of another length than the arguments"
    (is (= {:op :vmap :in-axes [0 nil] :arguments 1} (ts/thrown-data ((array/vmap array/exp {:in-axes [0 nil]}) b3)))))
  (testing "in-axes other than 0 and nil, options that are not vmap's, f that is no function"
    (doseq [axes [[1] [-1] [:x] 0 '(0)]]
      (is (= {:op :vmap :in-axes axes} (ts/thrown-data (array/vmap array/exp {:in-axes axes}))) (pr-str axes)))
    (is (= {:op :vmap :unknown [:out-axes]} (ts/thrown-data (array/vmap array/exp {:out-axes 0}))))
    (is (= {:op :vmap :value [0]} (ts/thrown-data (array/vmap array/exp [0]))))
    (is (= {:op :vmap :value 3} (ts/thrown-data (array/vmap 3))))))

(deftest branching-on-a-value-is-loud
  (let [branch? #(:gen.array/batched-branch (ts/thrown-data (%)))]
    (testing "reading a batched value on the host: the batched-branch error, as in gen bodies"
      (doseq [[label f] [[:item array/item] [:->clj array/->clj] [:truthy #(array/truthy? (> % 0.0))]
                         [:= #(= % 0.0)] [:if-truthy #(if (array/truthy? (< % 0.0)) (- %) %)]]]
        (is (branch? #((array/vmap f) bs)) (str label))))
    (testing "a raw if on a comparison takes one branch for every particle: the unread comparison"
      (let [data (ts/thrown-data ((array/vmap (fn [x] (if (< x 0.0) (- x) x))) bs))]
        (is (:gen.array/batched-branch data))
        (is (:gen.array.batch/unread-comparison data))))
    (testing "a batched-branch error f caught still means the particles differ"
      (is (branch? #((array/vmap (fn [x] (try (array/item x) (catch clojure.lang.ExceptionInfo _ 0.0)))) bs))))
    (testing "where picks per particle"
      (is (matches? (fn [x] (array/where (< x 0.0) (- x) x)) [0] [bs])))))

(deftest draws-are-per-particle
  (let [draw (array/vmap (fn [k x] (+ x (rng/normal k []))) {:in-axes [nil 0]})
        xs (array/zeros [n])
        a (ts/host (draw (rng/key 1) xs))]
    (is (= n (count (distinct a))) "independent draws, not one shared draw")
    (is (= a (ts/host (draw (rng/key 1) xs))) "the same key, the same draws")))

(deftest gradients-per-particle
  (let [g (fn [x] (array/sum (* (array/sin x) x)))
        density (fn [mu x] (dc/log-prob (d/normal mu 2.0) x))]
    (testing "a vmap of a gradient: each row's own"
      (is (matches? (array/grad g) [0] [b3]))
      (let [[vs gs] ((array/vmap (array/value-and-grad g)) b3)]
        (is (ts/close? (per-row g [0] [b3]) vs))
        (is (ts/close? (per-row (array/grad g) [0] [b3]) gs))))
    (testing "a shared argument's gradient is per particle, and sums to the total's"
      (let [mu 0.7
            per (array/vmap (array/grad density) {:in-axes [nil 0]})
            slopes (per mu bs)]
        (is (= [n] (array/shape slopes)))
        (is (ts/close? (per-row (array/grad density) [nil 0] [mu bs]) slopes))
        (is (ts/close? ((array/grad #(array/sum (density % bs))) mu) (array/sum slopes)))))
    (testing "several argnums, one batched and one shared"
      (let [[gm gx] ((array/vmap (array/grad density {:argnums [0 1]}) {:in-axes [nil 0]}) 0.7 bs)]
        (is (ts/close? (per-row (array/grad density {:argnums 0}) [nil 0] [0.7 bs]) gm))
        (is (ts/close? (per-row (array/grad density {:argnums 1}) [nil 0] [0.7 bs]) gx))))))

(deftest a-gradient-through-a-vmap-is-refused
  ;; the tape outside the tracer (docs/array-api.md §8b); under MLX's native
  ;; gradients no value is tracked, and MLX differentiates through
  (when-not (ts/native-grad?)
    (testing "a batched argument being differentiated"
      (is (= {:op :vmap :argument 0}
             (ts/thrown-data ((array/grad (fn [t] (array/sum ((array/vmap array/square) t)))) b3)))))
    (testing "a shared one"
      (is (= {:op :vmap :argument 1}
             (ts/thrown-data ((array/grad (fn [t] (array/sum ((array/vmap * {:in-axes [0 nil]}) bs t)))) 1.5)))))
    (testing "one f closes over: the op that meets it is named"
      (let [e (try ((array/grad (fn [t] (array/sum ((array/vmap #(* t %)) bs)))) 1.5) nil
                   (catch clojure.lang.ExceptionInfo e e))]
        (is (= {:op :vmap} (ex-data e)))
        (is (re-find #"gradient through a vmap" (ex-message e)))))))

(deftest nested-vmap-is-refused
  (testing "a vmap inside a vmap's function: the batched-branch error"
    (let [data (ts/thrown-data ((array/vmap (fn [x] ((array/vmap array/sum) x))) b23))]
      (is (= {:op :vmap :outer [n] :gen.array/batched-branch true} data))))
  (testing "a vmap given a batched run's value"
    (let [escaped (atom nil)]
      ((array/vmap (fn [x] (reset! escaped x) x)) b3)
      (is (= {:op :vmap :argument 0 :gen.array/batched-branch true}
             (ts/thrown-data ((array/vmap array/exp) @escaped))))
      (is (= {:op :vmap :argument 1 :gen.array/batched-branch true}
             (ts/thrown-data ((array/vmap + {:in-axes [0 nil]}) b3 @escaped)))
          "a shared one too")))
  (testing "in a batched generative function, the run falls back to one particle at a time, where vmap runs"
    (let [model (gen []
                  (let [s (trace :s (d/normal 0 1))]
                    (array/sum ((array/vmap #(* % s)) (array/array [1.0 2.0 3.0])))))
          ps (gb/simulate model [] 4 (rng/key 3))]
      (is (re-find #"vmap inside a batched run" (:message (gb/fallback-reason ps))))
      (is (ts/close? (* 6.0 (gb/choice ps :s)) (array/stack (mapv :retval (gb/traces ps))))))))
