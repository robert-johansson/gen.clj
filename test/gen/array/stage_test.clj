(ns gen.array.stage-test
  "gen.array.stage (experimental): staging is observational. A program
   staged from gen.clj's own code and replayed on the JVM backend
   (stage/evaluate) equals the code run directly, bit for bit in float64:
   arithmetic, structure, gathers, reductions, matrix products, draws
   (Threefry on staged keys), loops (array/scan), and the handler's
   generate on the law tests' models and random ones. Host reads of staged
   values are loud. array/scan's host loop is the staged loop's semantics."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array]
            [gen.array.stage :as st]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :as dyn]
            [gen.gfi :as gfi]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.random-static-models :as rsm]
            [gen.test-support :as ts]))

(defn- same?
  "a and b equal leaf for leaf, values and dtypes (NaN equal to NaN)."
  [a b]
  (let [la (array/tree-leaves a) lb (array/tree-leaves b)]
    (and (= (count la) (count lb))
         (every? true? (map (fn [x y] (and (= (array/dtype x) (array/dtype y))
                                           (= (pr-str (array/->clj x)) (pr-str (array/->clj y)))))
                            la lb)))))

(defn- replays?
  "f staged on specs and replayed on inputs equals f on inputs."
  [f specs inputs]
  (same? (st/evaluate (apply st/stage f specs) inputs) (apply f inputs)))

(deftest a-replay-is-the-direct-run
  (if-not (ts/jvm?)
    (println "gen.array.stage-test: replays are compared on the JVM backend")
    (binding [st/*float* :float64]
      (let [x (array/array [0.5 -1.25 3.0 0.0])
            m (array/array [[1.0 2.0] [3.0 -4.0] [0.5 0.25]])
            v (array/array [2.0 -1.0])]
        (testing "elementwise ops, unary and binary, with host numbers"
          (is (replays? (fn [x] (array/+ (array/exp (array/* x 2.0)) (array/log1p (array/abs x)))) [[[4] :float64]] [x]))
          (is (replays? (fn [x] (array/where (array/> x 0.0) (array/sqrt (array/abs x)) (array/- x))) [[[4] :float64]] [x])))
        (testing "reductions, argmax, logsumexp"
          (is (replays? (fn [m] {:s (array/sum m) :r (array/max m {:axis 0}) :a (array/argmax m {:axis 1}) :l (array/logsumexp m)})
                        [[[3 2] :float64]] [m])))
        (testing "structure: reshape, transpose, slice, concatenate, broadcast"
          (is (replays? (fn [m] (array/concatenate [(array/transpose m) (array/broadcast-to (array/reshape (array/slice m 0 0 1) [2 1]) [2 3])] 1))
                        [[[3 2] :float64]] [m])))
        (testing "a matrix product"
          (is (replays? (fn [m v] (array/matmul m v)) [[[3 2] :float64] [[2] :float64]] [m v])))
        (testing "gathers at host and staged indices"
          (is (replays? (fn [x] (array/take x (array/array [3 0 2] {:dtype :int32}))) [[[4] :float64]] [x]))
          (is (replays? (fn [m i] (array/take-along-axis m i 1)) [[[3 2] :float64] [[3 1] :int32]] [m (array/array [[1] [0] [1]] {:dtype :int32})])))))))

(deftest staged-draws-are-the-hosts
  (if-not (ts/jvm?)
    (println "gen.array.stage-test: draws are compared on the JVM backend")
    (doseq [fl [:float32 :float64]]
      (testing fl
        (binding [st/*float* fl]
          (let [key (rng/key 9)
                p (st/stage (fn [k] (let [[k1 k2] (rng/split k 2)]
                                      {:u (rng/uniform k1 [5] {:dtype fl}) :n (rng/normal (rng/fold-in k2 7) [3] {:dtype fl})}))
                            (st/key-spec))
                r (st/evaluate p [key])
                [k1 k2] (rng/split key 2)]
            (is (same? (:u r) (rng/uniform k1 [5] {:dtype fl})))
            (is (same? (:n r) (rng/normal (rng/fold-in k2 7) [3] {:dtype fl})))))))))

(deftest array-scan-is-one-loop-staged-and-a-host-loop-eagerly
  (let [step (fn [c x] [(array/+ (array/* c 1.5) x) (array/* c 2.0)])
        xs (array/array [1.0 2.0 3.0])]
    (testing "a host loop: the last carry and the stacked ys"
      (let [[c ys] (array/scan step 0.5 xs 3)]
        (is (ts/close? 9.9375 c))
        (is (ts/close? (array/array [1.0 3.5 9.25]) ys))))
    (testing "with xs nil, x is the step's index"
      (let [[c ys] (array/scan (fn [c t] [(array/+ c t) c]) 0 nil 4)]
        (is (= 6 (array/->clj c)))
        (is (= [0 0 1 3] (array/->clj ys)))))
    (testing "staged, one :scan node, its replay the host loop's"
      (when (ts/jvm?)
        (binding [st/*float* :float64]
          (let [p (st/stage (fn [c0 xs] (array/scan step c0 xs 3)) [[] :float64] [[3] :float64])]
            (is (= 1 (count (filter #(= :scan (:op %)) (:graph p)))))
            (is (same? (st/evaluate p [0.5 xs]) (array/scan step 0.5 xs 3)))))))))

(def ^:private regression
  (dyn/gen [xs]
    (let [s (trace :s (d/normal 0 10))
          i (trace :i (d/normal 0 10))]
      (trace :y (d/normal (array/+ (array/* s xs) i) 1.0)))))

(def ^:private indexed-mean
  (dyn/gen []
    (let [k (trace :k (d/categorical [0.2 0.3 0.5]))]
      (trace :y (d/normal (array/take (array/array [-1.0 0.0 2.0]) k) 1.0)))))

(defn- generate-replays?
  "gf's generate on args and observations, staged and replayed from key,
   equals the handler's: weight and score."
  [gf args obs key]
  (let [p (st/stage (fn [k] (let [{:keys [trace weight]} (gfi/generate gf (st/constants args) (st/constants obs) k)]
                              {:w weight :s (:score trace)}))
                    (st/key-spec))
        r (st/evaluate p [key])
        {:keys [trace weight]} (gfi/generate gf args obs key)]
    (and (== (:w r) weight) (== (:s r) (:score trace)))))

(deftest the-handlers-generate-staged-is-the-handlers
  (if-not (ts/jvm?)
    (println "gen.array.stage-test: generate is compared on the JVM backend")
    (binding [st/*float* :float64]
      (doseq [s (range 10)]
        (is (generate-replays? regression [(array/array [0.0 1.0 2.0 3.0])] (cm/choicemap :y (array/array [0.1 2.2 3.9 6.1])) (rng/key s)))
        (is (generate-replays? indexed-mean [] cm/EMPTY (rng/key s)))
        (is (generate-replays? indexed-mean [] (cm/choicemap :k 2) (rng/key s)))))))

(defspec random-static-models-stage-and-replay-as-the-handler 40
  (prop/for-all [[_ gf args] rsm/gen-model
                 seed laws/seeds]
    (or (not (ts/jvm?))
        (binding [st/*float* :float64]
          (let [key (rng/key seed)
                p (try (st/stage (fn [k] {:s (:score (gfi/simulate gf (st/constants args) k))}) (st/key-spec))
                       (catch clojure.lang.ExceptionInfo _ nil))]
            ;; a model that does not stage says nothing here (the census counts them)
            (or (nil? p)
                (== (:s (st/evaluate p [key])) (:score (gfi/simulate gf args key)))))))))

(deftest staged-scalars-and-keys-at-the-edges
  ;; genclj-kdau
  (testing "a staged 0-d value fills an array, as a host number does (it threw a ClassCastException)"
    (when (ts/jvm?)
      (binding [st/*float* :float64]
        (is (replays? (fn [x] (array/full [3] x)) [[[] :float64]] [2.5]))
        (is (replays? (fn [x] (array/full [2 2] (array/* x 2.0) {:dtype :int32})) [[[] :float64]] [1.75]) "cast to the dtype"))))
  (testing "rng/key-data of a staged key is loud (a ClassCastException, though key? accepts it)"
    (is (= :staged-key (:reason (try (st/stage (fn [k] (rng/key-data k) 1.0) (st/key-spec)) nil
                                     (catch clojure.lang.ExceptionInfo e (ex-data e)))))))
  (testing "a staged scan's step keeps the carry's structure, as the host loop's"
    (is (= :invalid-carry (:reason (try (st/stage (fn [c] (array/scan (fn [c _] [[c c] c]) c nil 2)) [[] :float32]) nil
                                        (catch clojure.lang.ExceptionInfo e (ex-data e))))))))

(deftest host-reads-of-staged-values-are-loud
  (testing "a branch on a staged value"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Cannot read a staged value"
                          (st/stage (fn [x] (if (array/truthy? (array/> x 0.0)) 1.0 2.0)) [[] :float32]))))
  (testing "clojure.core/= with a number"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"clojure.core/= on a staged value"
                          (st/stage (fn [x] (if (= x 0.0) 1.0 2.0)) [[] :float32]))))
  (testing "= with a keyword or nil is false, as on the JVM backend"
    (is (some? (st/stage (fn [x] (if (or (= x :missing) (= x nil)) 1.0 (array/+ x 1.0))) [[] :float32])))))
