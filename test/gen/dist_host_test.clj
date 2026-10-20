(ns gen.dist-host-test
  "Distributions backed by a host library (gen.dist.host), with families
   written in plain Clojure: draws as quantiles of the key's uniforms, log
   densities, discrete families and supports, the GFI laws, the batched
   fallback, loud gradients and misuse."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dist.host :as host]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(def expo
  "The exponential distribution with rate r, by its quantile."
  (host/family {:name "expo"
                :make (fn [r] (when-not (pos? r) (throw (IllegalArgumentException. "the rate must be positive"))) r)
                :quantile (fn [r u] (clojure.core// (- (Math/log1p (- u))) r))
                :log-density (fn [r x] (if (neg? x) ##-Inf (clojure.core/- (Math/log r) (clojure.core/* r x))))}))

(def geom
  "The number of failures before the first success, success probability p,
   truncated at n (a finite support)."
  (host/family {:name "geom"
                :discrete? true
                :make (fn [p n] {:p p :n (long n)
                                 :z (reduce clojure.core/+ (map #(Math/pow (clojure.core/- 1.0 p) %) (range (inc (long n)))))})
                :quantile (fn [{:keys [p n z]} u]
                            (loop [k 0 c 0.0]
                              (let [c (clojure.core/+ c (clojure.core// (clojure.core/* p (Math/pow (clojure.core/- 1.0 p) k)) (clojure.core/* p z)))]
                                (if (or (>= c u) (== k n)) k (recur (inc k) c)))))
                :log-density (fn [{:keys [p n z]} x]
                               (if (<= 0 x n) (clojure.core/- (clojure.core/* x (Math/log1p (- p))) (Math/log z)) ##-Inf))
                :support (fn [{:keys [n]}] (range (inc n)))}))

(defn- mean-sd [xs]
  (let [n (count xs)
        m (clojure.core// (reduce clojure.core/+ xs) n)]
    [m (Math/sqrt (clojure.core// (reduce clojure.core/+ (map #(let [e (clojure.core/- % m)] (clojure.core/* e e)) xs)) (dec n)))]))

(deftest draws-are-quantiles-of-the-key-s-uniforms
  (let [e (expo 2.0)]
    (testing "a pure function of the key, element i of a batch the same as alone"
      (is (= (dc/sample e (rng/key 1)) (dc/sample e (rng/key 1))))
      (is (not= (dc/sample e (rng/key 1)) (dc/sample e (rng/key 2))))
      ;; (a batch is the backend's float dtype; a lone draw a host double)
      (is (ts/close? (dc/sample e (rng/key 3)) (ts/number (array/take (dc/sample e (rng/key 3) [6]) 0))))
      (is (= [2 3] (array/shape (dc/sample e (rng/key 4) [2 3])))))
    (testing "the mean and the variance of 4000 draws (1/r, 1/r²), within five standard errors"
      (let [xs (mapv ts/number (array/->clj (dc/sample e (rng/key 5) [4000])))
            [m s] (mean-sd xs)]
        (is (< (Math/abs (clojure.core/- m 0.5)) (clojure.core/* 5 (clojure.core// 0.5 (Math/sqrt 4000)))))
        ;; the sample variance's sd is about √((μ4 - σ⁴)/n), μ4 = 9/r⁴ for the exponential
        (is (< (Math/abs (clojure.core/- (clojure.core/* s s) 0.25)) (clojure.core/* 5 (Math/sqrt (clojure.core// (clojure.core/- (clojure.core// 9.0 16) 0.0625) 4000)))))))
    (testing "log densities, as gen.dist's exponential's, elementwise over arrays"
      (is (ts/close? (dc/log-prob (d/exponential 2.0) 0.3) (dc/log-prob e 0.3)))
      (is (ts/close? (dc/log-prob (d/exponential 2.0) (array/array [0.1 0.7])) (dc/log-prob e (array/array [0.1 0.7]))))
      (is (= ##-Inf (dc/log-prob e -1.0))))))

(deftest a-discrete-family
  (let [g (geom 0.4 5)]
    (is (dc/discrete? g))
    (is (= (range 6) (dc/support g)))
    (is (every? #(<= 0 % 5) (map long (array/->clj (dc/sample g (rng/key 6) [200])))))
    (is (= ##-Inf (dc/log-prob g 1.5)) "a value that is not an integer")
    (testing "the probabilities add up to one, and the draws follow them"
      (let [ps (mapv #(Math/exp (dc/log-prob g %)) (range 6))
            draws (frequencies (map long (array/->clj (dc/sample g (rng/key 7) [4000]))))]
        (is (ts/close? 1.0 (reduce clojure.core/+ ps)))
        (doseq [k (range 6)]
          (let [f (clojure.core// (get draws k 0) 4000.0) p (ps k)]
            (is (< (Math/abs (clojure.core/- f p)) (clojure.core/* 5 (Math/sqrt (clojure.core// (clojure.core/* p (clojure.core/- 1 p)) 4000)))) (str k))))))))

(def model
  "A host family's parameter from a traced choice, and a discrete one."
  (gen [] (let [r (trace :r (d/gamma 2 1))
                t (trace :t (expo r))
                k (trace :k (geom 0.3 4))]
            (trace :y (d/normal (+ t k) 1)))))

(laws/deflaws host [[:model model [] []]] 100)

(deftest a-batched-run-falls-back-to-particles
  (let [k (rng/key 8)
        ps (batched/simulate model [] 5 k)]
    (is (re-find #"host library" (:message (batched/fallback-reason ps))))
    (is (= (mapv #(ts/host (% :t)) (batched/traces ps))
           (mapv #(ts/host ((gfi/simulate model [] %) :t)) (rng/split k 5))))))

(deftest a-draw-in-a-batched-run-falls-back
  ;; constant parameters: the one host draw would be every particle's (review 4, R1)
  (let [model (gen [] (trace :t (expo 2.0)))
        k (rng/key 18)
        ps (batched/simulate model [] 6 k)]
    (is (re-find #"host library" (:message (batched/fallback-reason ps))))
    (is (= 6 (count (distinct (map #(ts/host (% :t)) (batched/traces ps))))))))

(deftest a-host-distribution-prints-its-family
  (is (= "#gen/dist[host [expo 2.0]]" (pr-str (expo 2.0)))))

(deftest a-gradient-through-a-host-library-is-loud
  (let [tr (gfi/simulate model [] (rng/key 9))]
    (testing "of its value, or of a choice its parameters read"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no gradient" (grad/choice-gradients tr (sel/select :t))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no gradient" (grad/choice-gradients tr (sel/select :r)))))
    (testing "of choices it does not read: as without it"
      (let [{g :gradients} (grad/choice-gradients tr (sel/select :y))]
        (is (ts/close? (- (+ (tr :t) (tr :k)) (tr :y)) (cm/get-choice g :y)))))))

(deftest misuse-is-loud
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs :quantile" (host/family {:name "x" :make identity :log-density (fn [_ _] 0.0)})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid parameters.*positive" (expo -1.0)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"NaN" (expo ##NaN)))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be a number" (expo [1.0 2.0])))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no enumerable support" (dc/support (expo 1.0)))))
