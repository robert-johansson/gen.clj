(ns gen.inference.mh-test
  "MH moves against an exactly enumerable posterior: each kernel must leave
   the posterior invariant (law 10) and converge to it from a fixed start."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.test-support :as ts]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.mh :as mh]
            [gen.inference.weights :as w]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.state-space :as ss]))

(def model
  (gen []
    (let [a (trace :a (d/bernoulli 0.3))
          b (trace :b (d/categorical (if a [0.6 0.3 0.1] [0.2 0.3 0.5])))]
      (trace :obs (d/bernoulli (nth [0.1 0.5 0.9] b))))))

(def observations (cm/choicemap :obs true))

(defn- enumerated
  "An exactly enumerable target: the latent states (choice maps) of a model
   given observations, with their posterior probabilities."
  [model observations states]
  (let [lps (mapv #(:weight (gfi/assess model [] (cm/merge observations %))) states)]
    {:model model :observations observations :states states
     :posterior (mapv #(Math/exp %) (:log-weights (w/normalize lps)))}))

(def main (enumerated model observations (vec (for [a [false true] b [0 1 2]] (cm/choicemap :a a :b b)))))

(defn- latent [{:keys [observations]} tr]
  (sel/get-selected (:choices tr) (sel/complement (sel/from-choicemap observations))))

(defn- state [tr] [(tr :a) (tr :b)])

(defn- trace-at [{:keys [model observations]} s]
  (:trace (gfi/generate model [] (cm/merge observations s) (rng/key 0))))

(defn- chi-square
  "Pearson's statistic of traces' latent states against the posterior."
  [{:keys [states posterior] :as target} traces]
  (let [n (count traces)
        ;; states as host values (MLX's are arrays)
        observed (frequencies (map #(ts/host (latent target %)) traces))]
    (reduce + (map (fn [s p] (let [e (* n p) o (get observed (ts/host s) 0)] (/ (* (- o e) (- o e)) e)))
                   states posterior))))

(defn- threshold
  "P(χ² > threshold) ≈ 1e-5 or less for the target's degrees of freedom (the
   keys are fixed, so each test is deterministic)."
  [{:keys [states]}]
  (case (dec (count states)) 2 25.0 5 30.0))

(defn- stationary? [{:keys [states posterior] :as target} step]
  (let [n 10000
        [k-init k-step] (rng/split (rng/key 11))
        starts (map #(trace-at target (states %)) (w/multinomial (map #(Math/log %) posterior) n k-init))
        moved (map step starts (rng/split k-step n))]
    (chi-square target moved)))

(defn- converged? [target step start]
  (let [chains 1000 steps 40
        start (trace-at target start)
        run (fn [key] (reduce step start (rng/split key steps)))]
    (chi-square target (map run (rng/split (rng/key 12) chains)))))

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
  [tr fwd _ _ key]
  ;; a plain function (not a gen body): it reads the values it branches on
  (let [up? (ts/host (cm/get-choice fwd :up))
        {t :trace weight :weight} (gfi/update tr (cm/choicemap :b (mod (+ (ts/host (tr :b)) (if up? 1 -1)) 3)) key)]
    {:trace t :bwd-choices (cm/choicemap :up (not up?)) :weight weight}))

(defn- involutive [tr key]
  (let [[k1 k2] (rng/split key)]
    (-> tr (mh/mh (sel/select :a) k1) :trace (mh/involutive-mh direction [] step-b k2) :trace)))

(deftest ^:heavy kernels-leave-the-posterior-invariant
  (doseq [[label step] [[:gibbs-like gibbs-like] [:joint joint] [:custom custom] [:involutive involutive]]]
    (is (< (stationary? main step) (threshold main)) (str label " is not stationary"))))

(deftest ^:slow kernels-converge-to-the-posterior
  (doseq [[label step] [[:gibbs-like gibbs-like] [:joint joint] [:custom custom] [:involutive involutive]]]
    (is (< (converged? main step (cm/choicemap :a true :b 0)) (threshold main)) (str label " does not converge"))))

(deftest a-proposal-that-ignores-the-backward-move-is-caught
  ;; the same custom move, but accepting with the update weight alone: the
  ;; test above must be able to tell
  (let [wrong (fn [tr key]
                (let [[k1 k2 k3] (rng/split key 3)
                      {fwd :choices} (gfi/propose b-proposal [tr] k1)
                      {t :trace weight :weight} (gfi/update tr fwd k2)]
                  (if (< (Math/log (ts/number (rng/uniform k3))) (ts/number weight)) t tr)))]
    (is (> (stationary? main wrong) (threshold main)))))

(deftest results-and-errors
  (let [tr (trace-at main (cm/choicemap :a true :b 0))
        r (mh/mh tr (sel/select :a) (rng/key 3))]
    (is (= #{:trace :accepted? :log-alpha} (set (keys r))))
    (is (= r (mh/mh tr (sel/select :a) (rng/key 3))) "the same key gives the same move")
    (is (or (:accepted? r) (identical? tr (:trace r)))))
  (testing "a rejected move returns the old trace"
    ;; from the most likely state, so that some moves are rejected
    (let [tr (trace-at main (cm/choicemap :a true :b 2))
          rs (map #(mh/mh tr b-proposal [] %) (rng/split (rng/key 4) 50))]
      (is (every? #(or (:accepted? %) (identical? tr (:trace %))) rs))
      (is (some (complement :accepted?) rs))))
  (testing "proposal arguments must be a vector"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"vector" (mh/mh (trace-at main (cm/choicemap :a true :b 0)) b-proposal (rng/key 1)))))
  (testing "a keyword is neither a selection nor a proposal"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"takes a selection"
                          (mh/mh (trace-at main (cm/choicemap :a true :b 0)) :a (rng/key 1)))))
  (testing "a NaN acceptance ratio is a loud error"
    (let [impossible (gen [] (trace :x (d/normal 0 1)) (trace :obs (d/bernoulli 0.0)))
          tr (:trace (gfi/generate impossible [] (cm/choicemap :obs true) (rng/key 0)))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"NaN" (mh/mh tr (sel/select :x) (rng/key 1)))))))

(deftest the-involution-is-an-involution
  ;; law 17: applying it twice gives back the trace and the auxiliary choices
  (let [tr (trace-at main (cm/choicemap :a true :b 0))]
    (doseq [up? [true false]]
      (let [fwd (cm/choicemap :up up?)
            {t1 :trace u1 :bwd-choices} (step-b tr fwd nil [] (rng/key 0))
            {t2 :trace u2 :bwd-choices w2 :weight} (step-b t1 u1 nil [] (rng/key 0))]
        (is (= (:choices tr) (:choices t2)))
        (is (= fwd u2))
        (is (< (ts/abs (+ w2 (:weight (step-b tr fwd nil [] (rng/key 0))))) 1e-12) "the weights are opposite")))))

;; ---------------------------------------------------------------------------
;; A stochastic-structure target: the thesis's p_mh (pp. 92-93)
;; ---------------------------------------------------------------------------

(def p-mh
  (gen []
    (let [z (trace :a (d/bernoulli 0.5))
          z (and z (trace :b (d/bernoulli 0.5)))]
      (trace :c (d/bernoulli (if z 0.9 0.1))))))

(def p-mh-target
  (enumerated p-mh (cm/choicemap :c true)
              [(cm/choicemap :a false) (cm/choicemap :a true :b false) (cm/choicemap :a true :b true)]))

(def q4-mh
  "The thesis's valid proposal: switch branches with probability 0.9; on
   switching to a = T, sample b afresh."
  (gen [t]
    (if (t :a)
      (trace :a (d/bernoulli 0.1))
      (when (trace :a (d/bernoulli 0.9))
        (trace :b (d/bernoulli 0.5))))))

(deftest the-thesis-posterior
  (is (every? true? (map #(< (ts/abs (- %1 %2)) 1e-12) (:posterior p-mh-target) [(/ 1 6.0) (/ 1 12.0) 0.75]))))

(deftest ^:slow structure-changing-moves-target-the-posterior
  (doseq [[label step] [[:selection-a (fn [tr key] (:trace (mh/mh tr (sel/select :a) key)))]
                        [:q4-mh (fn [tr key] (:trace (mh/mh tr q4-mh [] key)))]]]
    (is (< (stationary? p-mh-target step) (threshold p-mh-target)) (str label " is not stationary"))
    (is (< (converged? p-mh-target step (cm/choicemap :a false)) (threshold p-mh-target))
        (str label " does not converge"))))

(def q2-mh (gen [_] (trace :a (d/bernoulli 0.5))))
(def q3-mh (gen [_] (trace :a (d/bernoulli 0.0))))

(deftest the-thesis-invalid-proposals-are-loud
  ;; Alg. 5, requirement 3 (pp. 92-93; law audit gap J): q2 and q3 cannot
  ;; propose b's value back, so a move from a = T to a = F, which discards
  ;; b, cannot be weighed: a loud error, never a silent move. q2's backward
  ;; assess leaves b unvisited. q3's gives the reverse choice a = T density
  ;; 0 at a possible trace: the proposal cannot reverse its own move
  ;; (gfi/assess-backward; DESIGN.md §9, the owner's second ruling on the
  ;; ten decisions' fifth), where the stop at a had made every such move a
  ;; quiet rejection (9277423). On a lazy backend the assess does not stop
  ;; at a, and b unvisited is the error. Planted bug (2026-10-25):
  ;; assess-backward without its check, q3 never moves and is never loud.
  (let [two (:trace (gfi/generate p-mh [] (cm/choicemap :a true :b false :c true) (rng/key 1)))
        outcome (fn [q key] (try (let [r (mh/mh two q [] key)] (if ((:trace r) :a) :stayed :moved))
                                 (catch clojure.lang.ExceptionInfo e e)))]
    (doseq [[label q] [[:q2-mh q2-mh] [:q3-mh q3-mh]]]
      (let [outcomes (map #(outcome q %) (rng/split (rng/key 2) 12))]
        (is (not-any? #{:moved} outcomes) (str label " " (frequencies outcomes)))
        (is (some #(instance? clojure.lang.ExceptionInfo %) outcomes) (str label " " (frequencies outcomes)))))
    (testing "q3 always proposes a = F: every move is loud, naming q3 and the reverse choices"
      (doseq [key (rng/split (rng/key 3) 6)]
        (let [e (outcome q3-mh key)]
          (is (instance? clojure.lang.ExceptionInfo e) (str e))
          (when (ts/jvm?)
            (is (re-find #"cannot reverse its own move" (str (ex-message e))))
            (is (identical? q3-mh (:proposal (ex-data e))))
            (is (= {:a true :b false} (:reverse-choices (ex-data e))))))))))

(def unit-a (gen [] (let [a (trace :a (d/uniform 0 1))] (trace :y (d/normal a 0.2)))))

(def upward
  "Moves a up, never down: from a, a' ~ U(a, 1), so the way back from a'
   has density 0 (§3.4.2's third requirement broken), wherever a' is."
  (gen [tr] (trace :a (d/uniform (tr :a) 1.0))))

(def wide
  "A valid proposal that leaves [0, 1] at times: from a in it, a' ~
   U(a - 0.5, a + 0.5), whose way back has density 1. From an a outside
   [0, 1], an impossible trace, a' ~ U(5, 6), so the way back from a move
   there has density 0: the third requirement holds where p(t') > 0, and
   asks nothing of a move to an impossible trace."
  (gen [tr] (let [a (tr :a)]
              (if (array/<= (array/abs (- a 0.5)) 0.5)
                (trace :a (d/uniform (- a 0.5) (+ a 0.5)))
                (trace :a (d/uniform 5.0 6.0))))))

;; Planted bugs (2026-10-25), gfi/assess-backward: without its check, the
;; first testing fails (upward's moves rejected, never loud), as do q3's
;; above and transform-test's and trace-kernel-test's; throwing whatever
;; the new trace, the third fails on every backend (and the second on a
;; lazy one, whose update does not stop before the backward assess);
;; throwing whatever the old trace, the fourth fails, its NaN error
;; replaced by this one.
(deftest a-proposal-that-cannot-reverse-its-move-is-loud
  (let [tr (:trace (gfi/generate unit-a [] (cm/choicemap :a 0.5 :y 0.6) (rng/key 0)))]
    (testing "every move of upward is to a possible trace, and none can be reversed"
      (doseq [key (rng/split (rng/key 1) 6)]
        (let [e (try (mh/mh tr upward [] key) (catch clojure.lang.ExceptionInfo e e))]
          (is (instance? clojure.lang.ExceptionInfo e) (str e))
          (is (re-find #"cannot reverse its own move" (str (ex-message e))))
          (is (identical? upward (:proposal (ex-data e))))
          (is (ts/close? 0.5 (get-in (ex-data e) [:reverse-choices :a]))))))
    (testing "a valid proposal's moves outside the support are rejected quietly"
      (let [near-one (:trace (gfi/generate unit-a [] (cm/choicemap :a 0.9 :y 0.6) (rng/key 2)))
            rs (map #(mh/mh near-one wide [] %) (rng/split (rng/key 3) 30))]
        (is (some #(= ##-Inf (:log-alpha %)) rs) "some proposals left [0, 1]")
        (is (every? #(or (not= ##-Inf (:log-alpha %)) (not (:accepted? %))) rs))
        (is (some :accepted? rs))))
    (testing "gfi/assess-backward is quiet for a move to an impossible trace"
      (let [outside (:trace (gfi/generate unit-a [] (cm/choicemap :a 5.5 :y 0.6) (rng/key 4)))]
        (is (= ##-Inf (ts/number (:weight (gfi/assess-backward tr outside wide [] (cm/choicemap :a 0.5) (rng/key 5))))))))
    (testing "a chain at an impossible trace keeps MH's NaN error (the owner's first ruling)"
      ;; from a = 1.5 to a' in [0, 1] weighs +Inf, and U(0, 1) gives the way
      ;; back density 0: +Inf - Inf is NaN
      (let [unit (gen [] (trace :a (d/uniform 0 1)))
            impossible (:trace (gfi/generate unit [] (cm/choicemap :a 1.5) (rng/key 6)))
            anywhere (gen [_] (trace :a (d/uniform 0 1)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"NaN" (mh/mh impossible anywhere [] (rng/key 7))))))))

(deftest the-thesis-s-exact-acceptances
  (testing "p. 148: from {a T b T c T}, regenerating {a} weighs 0 whichever a it draws"
    (let [model (gen [] (let [a (trace :a (d/bernoulli 0.3))
                              v (if a (trace :b (d/bernoulli 0.6)) true)]
                          (trace :c (d/bernoulli (if v 0.9 0.2)))))
          tr (:trace (gfi/generate model [] (cm/choicemap :a true :b true :c true) (rng/key 0)))
          rs (map #(gfi/regenerate tr (sel/select :a) %) (rng/split (rng/key 98) 16))]
      (is (some #(false? (ts/host ((:trace %) :a))) rs) "both outcomes drawn")
      (is (every? #(ts/close? 0.0 (:weight %)) rs))))
  (testing "p. 95: MH with the exact conditional as its proposal always accepts (log alpha = 0)"
    (let [model (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/normal mu 1)) mu))
          exact (gen [_] (trace :mu (d/normal 0.65 (Math/sqrt 0.5))))]
      (doseq [k (rng/split (rng/key 99) 8)]
        (let [[k0 k1] (rng/split k)
              tr (:trace (gfi/generate model [] (cm/choicemap :y 1.3) k0))]
          (is (ts/close? 0.0 (:log-alpha (mh/mh tr exact [] k1)))))))))

;; ---------------------------------------------------------------------------
;; A move outside the support is rejected (DESIGN.md §9, the ten decisions'
;; fifth; genclj-q7cq)
;; ---------------------------------------------------------------------------

(def unknown-six
  "Chapter 042's casino with the loaded die's probability of a six p6
   unknown, over ten rolls: p6 outside [0, 1] makes the HMM's emission
   probabilities invalid, which categorical refuses."
  (let [switches (array/array [[0.95 0.05] [0.1 0.9]])
        start (array/array [0.5 0.5])]
    (gen [n]
      (let [p6 (trace :p6 (d/uniform 0 1))
            others (array/full [5] (/ (- 1.0 p6) 5.0))]
        (splice :rolls ss/hmm n start
                {:transition switches
                 :emission (d/categorical (array/stack [(array/full [6] (/ 1.0 6))
                                                        (array/concatenate [others (array/reshape p6 [1])])]))})
        p6))))

(def rolls (apply cm/choicemap (mapcat (fn [t r] [[:rolls t] r]) (range) [5 5 2 0 1 3 2 5 0 5])))

(def p6-walk (gen [tr] (trace :p6 (d/normal (tr :p6) 0.3))))

;; Planted bugs (2026-10-25): the handler not stopping (weigh never setting
;; ::stopped) fails this test, its update throwing "categorical: probs must
;; be probabilities" (and q3's moves above are loud again); MH's move
;; weighing the backward proposal on such a trace fails it too: the
;; stopped update's discard holds the rolls it did not reach, which the
;; walk's assess leaves unvisited.
(deftest a-move-outside-the-support-is-rejected
  (let [tr (:trace (gfi/generate unknown-six [10] (cm/set-choice rolls :p6 0.9) (rng/key 0)))]
    (testing "the update to p6 = 1.2 weighs -Inf, the HMM never reached"
      (let [{t2 :trace w :weight} (gfi/update tr (cm/choicemap :p6 1.2) (rng/key 1))]
        (is (= ##-Inf (ts/number w)))
        ;; (a lazy backend's run goes on, its weight -Inf all the same)
        (when (ts/jvm?) (is (= [[:p6]] (cm/addresses (:choices t2)))))))
    (testing "random-walk MH on p6 rejects every proposal outside [0, 1]"
      (let [steps (reductions (fn [{tr :trace} k] (mh/mh tr p6-walk [] k)) {:trace tr} (rng/split (rng/key 2) 60))
            rs (rest steps)
            outside (filter #(= ##-Inf (:log-alpha %)) rs)]
        (is (seq outside) "some proposals left [0, 1]")
        (is (every? (complement :accepted?) outside))
        (is (every? #(<= 0.0 (ts/number ((:trace %) :p6)) 1.0) rs))
        (is (some :accepted? rs))))))

(deftest a-backward-proposal-is-not-weighed-on-a-trace-outside-the-support
  ;; the update stops at a, so the new trace lacks b, which the proposal
  ;; reads: MH's move rejects it unweighed (gen.inference.mh/move). Planted
  ;; bug (2026-10-25): the backward assess run on that trace throws "no
  ;; value at [:b]".
  (let [m (gen [] (let [a (trace :a (d/uniform 0 1))] (trace :b (d/normal a 0.2))))
        near-b (gen [tr] (trace :a (d/normal (tr :b) 1.0)))
        tr (:trace (gfi/generate m [] (cm/choicemap :a 0.5 :b 0.5) (rng/key 0)))
        rs (map #(mh/mh tr near-b [] %) (rng/split (rng/key 1) 30))]
    (is (some #(= ##-Inf (:log-alpha %)) rs) "some proposals left [0, 1]")
    (is (every? #(or (not= ##-Inf (:log-alpha %)) (identical? tr (:trace %))) rs))))
