(ns gen.genjl.particle-filter-test
  "Gen.jl 0.4.8's test/inference/particle_filter.jl, ported: the test's HMM
   forward algorithm on a hand case, and particle filters with a custom
   proposal and with the internal one (gen.inference.particle-filter),
   resampling at every step. States, emissions and Unfold's steps are
   0-based (D2); Gen.jl's column-stochastic matrices are their transposes
   here, a row per state.

   gen.clj's step returns the state, so a step's incremental weights are
   the state's log weights, one per particle. The ports assert their count
   with 100 particles; Gen.jl's 20,000 serve its log-ML band, which is the
   statistical port's (pending, below)."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.genjl.support :as genjl]
            [gen.inference.particle-filter :as pf]
            [gen.random :as rng]))

(defn- hmm-forward
  "particle_filter.jl:1: p(emissions) of an HMM, by the forward algorithm.
   emission[z][x] = p(x | z), transition[z][z'] = p(z' | z)."
  [prior emission transition emissions]
  (let [states (range (count prior))]
    (loop [alpha prior [x & more] emissions marg-lik 1.0]
      (let [joint (mapv #(* (alpha %) (get-in emission [% x])) states)
            denom (reduce + joint)]
        (if (empty? more)
          (* marg-lik denom)
          (let [posterior (mapv #(/ % denom) joint)]
            (recur (mapv (fn [z'] (reduce + (map #(* (posterior %) (get-in transition [% z'])) states))) states)
                   more
                   (* marg-lik denom))))))))

(deftest hmm-forward-algorithm
  ;; particle_filter.jl:29: against the four paths' sum, by hand
  (let [prior [0.4 0.6]
        emission [[0.1 0.9] [0.7 0.3]]
        transition [[0.5 0.5] [0.2 0.8]]
        [x0 x1] [1 0]
        path (fn [z0 z1] (* (prior z0) (get-in transition [z0 z1])
                                         (get-in emission [z0 x0]) (get-in emission [z1 x1])))
        expected (+ (path 0 0) (path 0 1) (path 1 0) (path 1 1))]
    (is (genjl/approx? expected (hmm-forward prior emission transition [x0 x1])))))

;; ---------------------------------------------------------------------------
;; particle_filter.jl:50, particle filtering
;; ---------------------------------------------------------------------------

(def ^:private prior [0.2 0.3 0.5])

(def ^:private emission [[0.1 0.2 0.7] [0.2 0.7 0.1] [0.7 0.2 0.1]])

(def ^:private transition [[0.4 0.4 0.2] [0.2 0.3 0.5] [0.9 0.05 0.05]])

(def ^:private kernel
  (gen [_t prev-z _params]
    (let [z (trace :z (d/categorical (array/take (array/array transition) prev-z)))]
      (trace :x (d/categorical (array/take (array/array emission) z)))
      z)))

(def ^:private chain (comb/unfold kernel))

(def ^:private model
  (gen [num-steps]
    (let [z-init (trace :z-init (d/categorical (array/array prior)))]
      (trace :x-init (d/categorical (array/take (array/array emission) z-init)))
      (splice :chain chain (dec num-steps) z-init nil))))

(def ^:private obs-x [0 0 1 2])

(defn- normalized [w] (/ w (array/sum w)))

(defn- p-emission
  "p(x | z) for each state z."
  [x]
  (array/take (array/transpose (array/array emission)) x))

(def ^:private init-proposal
  ;; particle_filter.jl:104
  (gen [x] (trace :z-init (d/categorical (normalized (* (array/array prior) (p-emission x)))))))

(def ^:private step-proposal
  ;; particle_filter.jl:117: the step-(T - 1) state, given the previous
  ;; one and its emission
  (gen [prev-trace T x]
    (let [prev-z (if (> T 2) (prev-trace [:chain (- T 3) :z]) (prev-trace :z-init))]
      (trace [:chain (- T 2) :z]
             (d/categorical (normalized (* (array/take (array/array transition) prev-z) (p-emission x))))))))

(defn- filtered
  "The filter's states over the steps of obs-x, n particles, resampling
   at every step (an ess threshold of n), with the given step options."
  [n init-opts step-opts key]
  (let [[k-init k-steps] (rng/split key)
        init (pf/initialize model [1] (cm/choicemap :x-init (obs-x 0)) n (assoc init-opts :key k-init))]
    (reductions (fn [state T]
                  (let [[k-resample k-step] (rng/split (rng/fold-in k-steps T))]
                    (pf/step (pf/maybe-resample state {:ess-threshold n :key k-resample})
                             [T] [diff/unknown-change]
                             (cm/choicemap [:chain (- T 2) :x] (obs-x (dec T)))
                             (cond-> (assoc step-opts :key k-step)
                               (:proposal step-opts) (assoc :proposal-args [T (obs-x (dec T))])))))
                init
                (range 2 (inc (count obs-x))))))

(def ^:private num-particles 100)

(deftest particle-filtering
  (testing "a custom proposal"
    ;; particle_filter.jl:96
    (doseq [state (rest (filtered num-particles {:proposal init-proposal :proposal-args [(obs-x 0)]}
                                  {:proposal step-proposal} (rng/key 1)))]
      (is (= num-particles (count (:log-weights state))))))
  (testing "the internal proposal"
    ;; particle_filter.jl:146
    (doseq [state (rest (filtered num-particles {} {} (rng/key 2)))]
      (is (= num-particles (count (:log-weights state)))))))

;; Pending:
;; - particle_filter.jl:143 and :169: log-ml-estimate within 0.02 of the
;;   forward algorithm's, with 20,000 particles (statistical, genclj-zgkq).
;; n/a: particle_filter.jl:172, the state's copy, ==, hash and mutation
;; independence (8): gen.clj's states are persistent values.
