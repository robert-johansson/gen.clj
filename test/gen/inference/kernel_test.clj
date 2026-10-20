(ns gen.inference.kernel-test
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as k]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; Toy kernels on vectors that record which steps ran, with reversals that
;; record a primed step.
(defn- step [label] (fn [t _] {:trace (conj t label)}))
(defn- toy [label] (k/kernel (step label) (step (keyword (str (name label) "'")))))

(def a (toy :a))
(def b (toy :b))
(def c (toy :c))

(defn- run1 [kernel] (:trace (kernel [] (rng/key 0))))

(deftest reversals
  (is (= [:a] (run1 a)))
  (is (= [:a'] (run1 (k/reversal a))))
  (is (= [:a] (run1 (k/reversal (k/reversal a)))) "the reversal of the reversal is the kernel")
  (testing "the reversal of a chain is the reversed chain of reversals"
    (is (= [:a :b :c] (run1 (k/chain a b c))))
    (is (= [:c' :b' :a'] (run1 (k/reversal (k/chain a b c))))))
  (testing "cycles and repeats, including a cycle that stops mid-round"
    (is (= [:a :b :c :a] (run1 (k/cycle 4 [a b c]))))
    (is (= [:a' :c' :b' :a'] (run1 (k/reversal (k/cycle 4 [a b c])))))
    (is (= [:a :b] (run1 (k/cycle [a b]))))
    (is (= [:b' :b' :b'] (run1 (k/reversal (k/repeat 3 b))))))
  (testing "a mixture's reversal mixes the reversals with the same weights"
    (is (= [:b] (run1 (k/mix [a b] [0 1]))))
    (is (= [:b'] (run1 (k/reversal (k/mix [a b] [0 1]))))))
  (testing "a kernel without a declared reversal"
    (is (not (k/reversible? (step :x))))
    (is (not (k/reversible? (k/chain a (step :x)))))
    (is (= [:a :x] (run1 (k/chain a (step :x)))) "still runs")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no declared reversal" (k/reversal (k/chain a (step :x)))))))

(deftest every-step-gets-its-own-key
  (let [record-key (fn [t key] {:trace (conj t key)})
        keys (run1 (k/repeat 5 record-key))]
    (is (= 5 (count (set keys))))))

(deftest running-a-chain
  (let [count-up (fn [n _] {:trace (inc n)})]
    (is (= {:traces [5 8 11 14] :acceptance-rate nil}
           (k/run count-up 0 {:samples 4 :burn 2 :thin 3 :key (rng/key 1)})))
    (is (= {:traces [] :acceptance-rate nil} (k/run count-up 0 {:samples 0})))
    (is (thrown? clojure.lang.ExceptionInfo (k/run count-up 0 {:samples 3 :thin 0}))))
  (testing "combinators report their moves' accept decisions, and run counts every move"
    (let [accept (fn [t _] {:trace t :accepted? true})
          reject (fn [t _] {:trace t :accepted? false})]
      (is (= [true false false] (:accepted ((k/chain accept reject reject) 0 (rng/key 2)))))
      (is (= [true false true false] (:accepted ((k/cycle 4 [accept reject]) 0 (rng/key 2)))))
      (is (= [true false true false] (:accepted ((k/repeat 2 (k/chain accept reject)) 0 (rng/key 2)))) "nested")
      (is (nil? (:accepted ((k/chain accept (fn [t _] {:trace t})) 0 (rng/key 2)))) "unless a move reports none")
      (is (= 0.25 (:acceptance-rate (k/run (k/chain accept reject reject reject) 0 {:samples 5 :key (rng/key 3)})))))))

(def model
  (gen []
    (let [a (trace :a (d/bernoulli 0.3))
          b (trace :b (d/categorical (if a [0.6 0.3 0.1] [0.2 0.3 0.5])))]
      (trace :obs (d/bernoulli (nth [0.1 0.5 0.9] b))))))

(def observations (cm/choicemap :obs true))

(deftest ^:heavy an-mh-cycle-targets-the-posterior
  (let [start (:trace (gfi/generate model [] (cm/choicemap :a true :b 0 :obs true) (rng/key 0)))
        kernel (k/checked (k/cycle [(k/mh (sel/select :a)) (k/mh (sel/select :b))]) observations)
        ends (map #(peek (:traces (k/run kernel start {:samples 1 :burn 30 :key %})))
                  (rng/split (rng/key 2) 2000))
        p-a (/ (count (filter #(ts/host (% :a)) ends)) 2000.0)
        ;; exact: p(a | obs) by enumeration over b
        joint (fn [a] (reduce + (for [b [0 1 2]]
                                  (Math/exp (ts/number (:weight (gfi/assess model [] (cm/choicemap :a a :b b :obs true))))))))
        exact (/ (joint true) (+ (joint true) (joint false)))]
    (is (< (ts/abs (- p-a exact)) (* 5 (Math/sqrt (/ (* exact (- 1 exact)) 2000)))))
    (is (k/reversible? kernel))
    (let [{:keys [acceptance-rate]} (k/run (k/mh (sel/select :b)) start {:samples 100 :key (rng/key 3)})]
      (is (< 0.0 acceptance-rate 1.0)))))

(deftest checked-kernels-catch-changed-observations
  (let [start (:trace (gfi/generate model [] (cm/choicemap :obs true) (rng/key 0)))
        bad (k/checked (k/mh (sel/select :obs)) observations)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"changed the observation"
                          (dotimes [i 20] (bad start (rng/key i)))))))

(deftest checked-kernels-catch-a-proposal-that-writes-an-observation
  ;; the custom-proposal requirement of Alg 5 (p. 92): never propose an observed address
  (let [start (:trace (gfi/generate model [] (cm/choicemap :obs true) (rng/key 0)))
        writes-obs (gen [tr] (trace :obs (d/bernoulli 0.5)))
        bad (k/checked (k/mh writes-obs []) observations)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"changed the observation"
                          (dotimes [i 20] (bad start (rng/key i)))))))

(deftest ^:heavy a-mixture-of-kernels-targets-the-posterior
  (let [start (:trace (gfi/generate model [] (cm/choicemap :a true :b 0 :obs true) (rng/key 0)))
        kernel (k/mix [(k/mh (sel/select :a)) (k/mh (sel/select :b)) (k/mh (sel/select :a :b))] [0.3 0.3 0.4])
        ends (map #(peek (:traces (k/run kernel start {:samples 1 :burn 60 :key %}))) (rng/split (rng/key 5) 2000))
        p-a (/ (count (filter #(ts/host (% :a)) ends)) 2000.0)
        joint (fn [a] (reduce + (for [b [0 1 2]]
                                  (Math/exp (ts/number (:weight (gfi/assess model [] (cm/choicemap :a a :b b :obs true))))))))
        exact (/ (joint true) (+ (joint true) (joint false)))]
    (is (< (ts/abs (- p-a exact)) (* 5 (Math/sqrt (/ (* exact (- 1 exact)) 2000)))))))

(deftest mixture-weights-are-validated
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"non-negative" (k/mix [a b] [-1 1]))))

;; ---------------------------------------------------------------------------
;; kern
;; ---------------------------------------------------------------------------

(defn- recording
  "A kernel that leaves the trace alone and logs label."
  [log label]
  (k/symmetric (fn [tr _key] (swap! log conj label) {:trace tr})))

(deftest kern-builds-a-chain-and-its-reversal
  (let [log (atom [])
        kernel (k/kern [n extra?]
                 (recording log :first)
                 (doseq [i (range n)] (recording log [:loop i]))
                 (when extra? (recording log :extra))
                 (let [label :last] (recording log label)))
        tr (gfi/simulate model [] (rng/key 1))]
    (testing "the statements in order"
      ((kernel 2 true) tr (rng/key 2))
      (is (= [:first [:loop 0] [:loop 1] :extra :last] @log)))
    (testing "the reversal: every statement reversed, in reverse order"
      (reset! log [])
      ((k/reversal (kernel 2 true)) tr (rng/key 3))
      (is (= [:last :extra [:loop 1] [:loop 0] :first] @log)))
    (testing "branches and loops follow the arguments"
      (reset! log [])
      ((kernel 0 false) tr (rng/key 4))
      (is (= [:first :last] @log))))
  (testing "a kern of MH steps moves as the chain written by hand"
    ;; a loop is a chain of its iterations, each a chain of its statements
    (let [step #(k/chain (k/mh (sel/select :a)) (k/mh (sel/select :b)))
          by-hand (k/chain (step) (step))
          kerned ((k/kern [] (doseq [_ (range 2)] (k/mh (sel/select :a)) (k/mh (sel/select :b)))))
          tr (:trace (gfi/generate model [] observations (rng/key 5)))]
      (doseq [key (rng/split (rng/key 6) 6)]
        (let [a (by-hand tr key) b (kerned tr key)]
          (is (= (:choices (:trace a)) (:choices (:trace b))))
          (is (= (:accepted a) (:accepted b)) "the same decisions, so the same keys")))))
  (testing "a statement that is not a kernel is loud"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not a kernel"
                          ((k/kern [] (k/mh (sel/select :a)) (when-not true (k/mh (sel/select :b)))))))))

;; ---------------------------------------------------------------------------
;; Kernels that read the trace (kern ^:trace; the owner, after the final law
;; audit: Gen.jl's @kern reads the trace in its control flow)
;; ---------------------------------------------------------------------------

(def ^:private counted
  "A model whose :n decides how many [:z i] it makes."
  (gen [] (let [n (trace :n (d/categorical [0.2 0.3 0.5]))]
            (doseq [i (range (ts/number n))] (trace [:z i] (d/normal 0 1)))
            (trace :x (d/normal 0 1)))))

(defn- host-of [tr addr] (ts/host (tr addr)))

(defn- set-n [tr key] {:trace (:trace (gfi/update tr (cm/choicemap :n 2) key))})

(defn- stay [tr _key] {:trace tr})

(deftest a-kern-that-reads-the-trace
  ;; Planted bugs (2026-10-20), each failing this test: no check; the trace
  ;; bound once, before the first statement; the reversal in forward order,
  ;; or applying the kernels themselves; one key for every application;
  ;; loops that apply no kernels (the confirmation read's K1); the check
  ;; against the final trace, of every construct at once (7357b4f's)
  (let [log (atom [])
        kernel (k/kern ^:trace [tr extra?]
                 (recording log :first)
                 (doseq [i (range (host-of tr :n))] (recording log [:loop i]))
                 (when (and extra? (pos? (host-of tr :n))) (recording log :extra))
                 (let [[n] [(host-of tr :n)] label [:last n]] (recording log label)))
        tr (:trace (gfi/generate counted [] (cm/choicemap :n 2) (rng/key 1)))]
    (testing "the statements in order, loops and branches read from the trace"
      ((kernel true) tr (rng/key 2))
      (is (= [:first [:loop 0] [:loop 1] :extra [:last 2]] @log)))
    (testing "the reversal: every statement reversed, in reverse order"
      (reset! log [])
      ((k/reversal (kernel true)) tr (rng/key 3))
      (is (= [[:last 2] :extra [:loop 1] [:loop 0] :first] @log)))
    (testing "another trace, another structure"
      (reset! log [])
      ((kernel true) (:trace (gfi/generate counted [] (cm/choicemap :n 0) (rng/key 4))) (rng/key 5))
      (is (= [:first [:last 0]] @log))))
  (testing "a kernel's arguments see the trace as the statements before it left it"
    (let [seen (atom [])
          set-x (k/symmetric (fn [tr key] {:trace (:trace (gfi/update tr (cm/choicemap :x 5.0) key))}))
          kernel ((k/kern ^:trace [tr] set-x (recording seen (host-of tr :x)) set-x))
          tr (:trace (gfi/generate counted [] (cm/choicemap :n 1 :x 0.5) (rng/key 6)))]
      (kernel tr (rng/key 7))
      (is (= [5.0] @seen))))
  (let [tr (:trace (gfi/generate counted [] (cm/choicemap :n 1) (rng/key 8)))]
    (testing "a body that changes what its construct read is loud (the reversal would not undo it)"
      (doseq [kernel [((k/kern ^:trace [tr] (doseq [_ (range (host-of tr :n))] (k/symmetric set-n))))
                      ((k/kern ^:trace [tr] (when (= 1 (host-of tr :n)) (k/symmetric set-n))))
                      ((k/kern ^:trace [tr] (if (= 1 (host-of tr :n)) (k/symmetric set-n) (k/chain))))
                      ((k/kern ^:trace [tr] (let [n (host-of tr :n)] (k/symmetric set-n))))
                      (k/reversal ((k/kern ^:trace [tr] (doseq [_ (range (host-of tr :n))] (k/kernel stay set-n)))))]]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"changed what it reads" (kernel tr (rng/key 9))))))
    (testing "a later statement may change what an earlier one read, as the thesis's p. 102 kernel does"
      (let [log (atom [])
            kernel ((k/kern ^:trace [tr]
                      (doseq [i (range (host-of tr :n))] (recording log [:loop i]))
                      (k/symmetric set-n)
                      (recording log [:n (host-of tr :n)])))]
        (is (= 2 (host-of (:trace (kernel tr (rng/key 10))) :n)))
        (is (= [[:loop 0] [:n 2]] @log))
        (reset! log [])
        ((k/reversal kernel) tr (rng/key 11))
        (is (= [[:n 1] [:loop 1] [:loop 0]] @log) "the reversal's loop reads the trace its first statement left"))))
  (testing "every application gets its own key, also in loops and in the reversal"
    (let [keys-seen (atom [])
          record-key (k/symmetric (fn [tr key] (swap! keys-seen conj key) {:trace tr}))
          kernel ((k/kern ^:trace [tr] record-key (doseq [_ (range (host-of tr :n))] record-key) record-key))
          tr (:trace (gfi/generate counted [] (cm/choicemap :n 2) (rng/key 14)))]
      (kernel tr (rng/key 15))
      ((k/reversal kernel) tr (rng/key 15))
      (is (= 8 (count @keys-seen)))
      (is (= 4 (count (set (take 4 @keys-seen)))))
      (is (= 4 (count (set (drop 4 @keys-seen)))))))
  (testing "a statement without a reversal makes the reversal loud"
    (let [kernel ((k/kern ^:trace [_tr] (k/kernel stay)))
          tr (gfi/simulate counted [] (rng/key 12))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no declared reversal" ((k/reversal kernel) tr (rng/key 13)))))))

;; ---------------------------------------------------------------------------
;; The thesis's p. 102 kernel
;; ---------------------------------------------------------------------------

(def ^:private polynomial
  "The polynomial regression of the thesis's p. 102 kernel, small enough to
   enumerate: a degree of 0 or 1, coefficients in {-1, 0, 1} (categorical,
   less 1) and a noise level, the sd 0.8 or 1.6."
  (gen [xs]
    (let [degree (trace :degree (d/bernoulli 0.5))
          c0 (trace [:coeff 0] (d/categorical [0.3 0.4 0.3]))
          c1 (if degree (trace [:coeff 1] (d/categorical [0.3 0.4 0.3])) 1)
          noise (trace :noise (d/categorical [0.5 0.5]))]
      (doseq [[j x] (map-indexed vector xs)]
        (trace [:y j] (d/normal (+ (- c0 1) (* (- c1 1) x)) (* 0.8 (+ 1 noise)))))
      degree)))

(defn- coefficients
  "0:trace[:degree], the indices of tr's coefficients."
  [tr]
  (range (if (host-of tr :degree) 2 1)))

(def ^:private choose (gen [_tr addr probs] (trace addr (d/categorical probs))))

(defn- gibbs
  "A Gibbs move on the categorical choice at addr, of n values: MH that
   proposes it from its conditional given tr's other choices, enumerated,
   and so always accepts."
  [tr addr n]
  (let [scores (mapv #(ts/number (:score (:trace (gfi/update tr (cm/choicemap addr %) (rng/key 0))))) (range n))
        ws (mapv #(Math/exp (clojure.core/- % (apply max scores))) scores)]
    (k/mh choose [addr (mapv #(clojure.core// % (reduce clojure.core/+ ws)) ws)])))

(def ^:private polynomial-kernel
  ;; composite_kernel (p. 102) in its order: a loop over the coefficients, a
  ;; structure move with probability 0.1, a loop over the coefficients the
  ;; trace then has, and a Gibbs move on the noise (the thesis's variance).
  ;; The structure move changes the :degree the first loop read, which only
  ;; the first loop's own body must leave alone. The coefficient moves are
  ;; this model's: Gibbs in the first loop and resampling from the prior in
  ;; the second, where the thesis has a random walk, then a data-driven move
  (k/kern ^:trace [tr]
    (doseq [i (coefficients tr)] (gibbs tr [:coeff i] 3))
    (k/mix [(k/mh (sel/select :degree)) (k/chain)] [0.1 0.9])
    (doseq [i (coefficients tr)] (k/mh (sel/select [:coeff i])))
    (gibbs tr :noise 2)))

(deftest ^:heavy the-thesis-s-degree-kernel-leaves-the-posterior-invariant
  ;; Exact posterior draws (by enumeration), with the choice that the first
  ;; move draws forgotten (drawn again from its prior): that Gibbs move
  ;; returns the draws to the posterior exactly, and the moves after it must
  ;; leave it there. Each cell (degree, c0, noise) is within 5 standard
  ;; errors (binomial) of its posterior probability, the smallest 0.023, so
  ;; 46 draws of 2000. Planted bugs (2026-10-20): 7357b4f's check, of
  ;; every construct against the final trace, is loud whenever the
  ;; structure move is accepted (21 and 13 of 400 applications, forward and
  ;; reversed); with loops that apply no kernels (the confirmation read's
  ;; K1), c0 keeps its prior in the kernel's case, P(c0 = 1) = 0.40 against
  ;; 0.79, and 11 cells fail; one key for every application fails 4
  (let [xs [-1.0 0.0 1.0]
        obs (cm/choicemap [:y 0] -0.7 [:y 1] 0.0 [:y 2] 0.7)
        cells (vec (for [degree [false true] c0 [0 1 2] c1 (if degree [0 1 2] [nil]) noise [0 1]]
                     (cond-> (cm/choicemap :degree degree [:coeff 0] c0 :noise noise) c1 (cm/set-choice [:coeff 1] c1))))
        weights (mapv #(Math/exp (ts/number (:weight (gfi/assess polynomial [xs] (cm/merge % obs) (rng/key 0))))) cells)
        total (reduce clojure.core/+ weights)
        cumulative (vec (reductions clojure.core/+ weights))
        n 2000
        draw (fn [k forgotten]
               (let [[k-cell k-forget] (rng/split k)
                     u (clojure.core/* total (ts/number (:retval (gfi/simulate (d/uniform 0 1) [] k-cell))))
                     i (min (count (take-while #(clojure.core/< % u) cumulative)) (dec (count cells)))
                     tr (:trace (gfi/generate polynomial [xs] (cm/merge (cells i) obs) (rng/key 0)))]
                 (:trace (gfi/regenerate tr (sel/select forgotten) k-forget))))
        cell (fn [choices] (mapv #(ts/host (cm/get-choice choices %)) [:degree [:coeff 0] :noise]))
        exact (reduce (fn [m [c w]] (update m (cell c) (fnil clojure.core/+ 0.0) (clojure.core// w total)))
                      {} (map vector cells weights))
        at-the-posterior (fn [kernel forgotten seed]
                           (let [ends (map #(let [[k1 k2] (rng/split %)] (:trace (kernel (draw k1 forgotten) k2)))
                                           (rng/split (rng/key seed) n))
                                 observed (frequencies (map (comp cell :choices) ends))]
                             (doseq [[c p] exact]
                               (let [f (clojure.core// (get observed c 0) (double n))]
                                 (is (clojure.core/< (Math/abs (clojure.core/- f p))
                                                     (clojure.core/* 5 (Math/sqrt (clojure.core// (clojure.core/* p (clojure.core/- 1 p)) n))))
                                     (str c ": " f " against " p))))))]
    (testing "the kernel, from the posterior with c0 forgotten"
      (at-the-posterior (polynomial-kernel) [:coeff 0] 12))
    (testing "its reversal, from the posterior with the noise forgotten"
      (at-the-posterior (k/reversal (polynomial-kernel)) :noise 13))))
