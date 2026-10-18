(ns gen.inference.mh-test
  "MH moves against an exactly enumerable posterior: each kernel must leave
   the posterior invariant (law 10) and converge to it from a fixed start."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.mh :as mh]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def model
  (gen []
    (let [a (trace :a (d/bernoulli 0.3))
          b (trace :b (d/categorical (if a [0.6 0.3 0.1] [0.2 0.3 0.5])))]
      (trace :obs (d/bernoulli (nth [0.1 0.5 0.9] b))))))

(def observations (cm/choicemap :obs true))

(def states (vec (for [a [false true] b [0 1 2]] [a b])))

(defn- constraints [[a b]] (cm/merge observations (cm/choicemap :a a :b b)))

(def posterior
  (let [lps (mapv #(:weight (gfi/assess model [] (constraints %))) states)]
    (mapv #(Math/exp %) (:log-weights (w/normalize lps)))))

(defn- state [tr] [(tr :a) (tr :b)])

(defn- trace-at [s] (:trace (gfi/generate model [] (constraints s) (rng/key 0))))

(defn- chi-square
  "Pearson's statistic of traces' states against the posterior."
  [traces]
  (let [n (count traces)
        observed (frequencies (map state traces))]
    (reduce + (map (fn [s p] (let [e (* n p) o (get observed s 0)] (/ (* (- o e) (- o e)) e)))
                   states posterior))))

;; five degrees of freedom: P(χ² > 30) ≈ 1.5e-5 (the keys are fixed, so each
;; test is deterministic)
(def threshold 30.0)

(defn- stationary? [step]
  (let [n 10000
        [k-init k-step] (rng/split (rng/key 11))
        starts (map #(trace-at (states %)) (w/multinomial (map #(Math/log %) posterior) n k-init))
        moved (map step starts (rng/split k-step n))]
    (chi-square moved)))

(defn- converged? [step]
  (let [chains 1000 steps 40
        start (trace-at [true 0])
        run (fn [key] (reduce step start (rng/split key steps)))]
    (chi-square (map run (rng/split (rng/key 12) chains)))))

;; ---------------------------------------------------------------------------
;; The moves
;; ---------------------------------------------------------------------------

(defn- gibbs-like [tr key]
  (let [[k1 k2] (rng/split key)]
    (-> tr (mh/mh (sel/select :a) k1) :trace (mh/mh (sel/select :b) k2) :trace)))

(defn- joint [tr key] (:trace (mh/mh tr (sel/select :a :b) key)))

(def b-proposal
  "Proposes :b from a distribution that depends on the current :a, so the
   forward and backward proposal densities differ."
  (gen [tr] (trace :b (d/categorical (if (tr :a) [0.7 0.2 0.1] [0.1 0.1 0.8])))))

(def a-proposal (gen [tr] (trace :a (d/bernoulli 0.8))))

(defn- custom [tr key]
  (let [[k1 k2] (rng/split key)]
    (-> tr (mh/mh a-proposal [] k1) :trace (mh/mh b-proposal [] k2) :trace)))

(def direction (gen [tr] (trace :up (d/bernoulli 0.7))))

(defn- step-b
  "An involution on (b, up): move b one step around the circle in the
   direction up, and reverse the direction."
  [tr fwd _ _]
  (let [up? (cm/get-choice fwd :up)
        {t :trace weight :weight} (gfi/update tr (cm/choicemap :b (mod (+ (tr :b) (if up? 1 -1)) 3)))]
    {:trace t :bwd-choices (cm/choicemap :up (not up?)) :weight weight}))

(defn- involutive [tr key]
  (let [[k1 k2] (rng/split key)]
    (-> tr (mh/mh (sel/select :a) k1) :trace (mh/involutive-mh direction [] step-b k2) :trace)))

(deftest kernels-leave-the-posterior-invariant
  (doseq [[label step] [[:gibbs-like gibbs-like] [:joint joint] [:custom custom] [:involutive involutive]]]
    (is (< (stationary? step) threshold) (str label " is not stationary"))))

(deftest kernels-converge-to-the-posterior
  (doseq [[label step] [[:gibbs-like gibbs-like] [:joint joint] [:custom custom] [:involutive involutive]]]
    (is (< (converged? step) threshold) (str label " does not converge"))))

(deftest a-proposal-that-ignores-the-backward-move-is-caught
  ;; the same custom move, but accepting with the update weight alone: the
  ;; test above must be able to tell
  (let [wrong (fn [tr key]
                (let [[k1 k2 k3] (rng/split key 3)
                      {fwd :choices} (gfi/propose b-proposal [tr] k1)
                      {t :trace weight :weight} (gfi/update tr fwd k2)]
                  (if (< (Math/log (rng/uniform k3)) weight) t tr)))]
    (is (> (stationary? wrong) threshold))))

(deftest results-and-errors
  (let [tr (trace-at [true 0])
        r (mh/mh tr (sel/select :a) (rng/key 3))]
    (is (= #{:trace :accepted? :log-alpha} (set (keys r))))
    (is (= r (mh/mh tr (sel/select :a) (rng/key 3))) "the same key gives the same move")
    (is (or (:accepted? r) (identical? tr (:trace r)))))
  (testing "a rejected move returns the old trace"
    ;; from the most likely state, so that some moves are rejected
    (let [tr (trace-at [true 2])
          rs (map #(mh/mh tr b-proposal [] %) (rng/split (rng/key 4) 50))]
      (is (every? #(or (:accepted? %) (identical? tr (:trace %))) rs))
      (is (some (complement :accepted?) rs))))
  (testing "proposal arguments must be a vector"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"vector" (mh/mh (trace-at [true 0]) b-proposal (rng/key 1)))))
  (testing "a NaN acceptance ratio is a loud error"
    (let [impossible (gen [] (trace :x (d/normal 0 1)) (trace :obs (d/bernoulli 0.0)))
          tr (:trace (gfi/generate impossible [] (cm/choicemap :obs true) (rng/key 0)))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"NaN" (mh/mh tr (sel/select :x) (rng/key 1)))))))
