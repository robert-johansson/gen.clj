(ns gen.inference.trace-kernel-test
  "Trace kernels (GenTraceKernelDSL): the transform tests' moves written as
   kernels. Exact weights against closed-form Jacobians; MH and a reversible
   jump keep the exact posterior over one step; an SMCP3 step weighs a
   reparameterized proposal exactly as the optimal one; loud misuse."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.custom :as custom]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as k]
            [gen.inference.particle-filter :as pf]
            [gen.inference.trace-kernel :as tk]
            [gen.inference.transform-test :as tt]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(defn- lp [d v] (ts/number (dc/log-prob d v)))

(def split-merge
  "GenTraceKernelDSL's example, with u ~ N(0, 1)."
  (gen [tr]
    (if (tr :two?)
      [(cm/choicemap :two? false :mu (/ (+ (tr :mu1) (tr :mu2)) 2.0))
       (cm/choicemap :u (/ (- (tr :mu2) (tr :mu1)) 2.0))]
      (let [u (trace :u (d/normal 0 1))]
        [(cm/choicemap :two? true :mu1 (- (tr :mu) u) :mu2 (+ (tr :mu) u)) (cm/choicemap)]))))

(deftest a-split-and-a-merge-weigh-exactly
  (let [one (:trace (gfi/generate tt/two-means [] (cm/choicemap :two? false :mu 0.7 :y1 tt/y1 :y2 tt/y2) (rng/key 1)))
        [k-propose k-update] (rng/split (rng/key 2) 3)
        u (cm/get-choice (:choices (gfi/propose split-merge [one] k-propose)) :u)
        {la :log-alpha} (tk/mh one split-merge [] (rng/key 2))]
    ;; split: the score ratio - log N(u; 0, 1) + log 2 (no backward choices)
    (let [split' (:trace (gfi/update one (cm/choicemap :two? true :mu1 (- 0.7 u) :mu2 (+ 0.7 u)) k-update))]
      (is (ts/close? (+ (- (:score split') (:score one)) (- (lp (d/normal 0 1) u)) (Math/log 2.0)) la)))
    (testing "a merge: the score ratio + log N(u; 0, 1) - log 2"
      (let [two (:trace (gfi/generate tt/two-means [] (cm/choicemap :two? true :mu1 0.5 :mu2 0.9 :y1 tt/y1 :y2 tt/y2) (rng/key 3)))
            {merged :trace la :log-alpha} (tk/mh two split-merge [] (rng/key 4))
            merged' (:trace (gfi/update two (cm/choicemap :two? false :mu 0.7) (rng/key 5)))]
        (is (ts/close? (+ (- (:score merged') (:score two)) (lp (d/normal 0 1) 0.2) (- (Math/log 2.0))) la))))))

(deftest a-reversible-jump-kernel-keeps-the-posterior
  (let [n 8000
        move (tk/mh-kernel split-merge [])
        after (mapv (fn [key] (let [[k1 k2] (rng/split key)] (:trace (move (tt/exact-posterior-trace k1) k2))))
                    (rng/split (rng/key 6) n))
        frac (/ (count (filter #(% :two?) after)) (double n))]
    (is (< (Math/abs (- frac tt/p-two)) (* 5 (Math/sqrt (/ (* tt/p-two (- 1 tt/p-two)) n)))) (str frac " against " tt/p-two))))

(def scale
  "x → x·e^u, u → -u: |det J| = e^u."
  (gen [tr] (let [u (trace :u (d/normal 0 0.5))]
              [(cm/choicemap :x (* (tr :x) (array/exp u))) (cm/choicemap :u (array/negate u))])))

(deftest ^:slow a-kernel-mh-keeps-the-posterior
  (let [n 4000
        xs (tt/normals (rng/key 7) n 0.64 (Math/sqrt 0.2))
        stepped (mapv (fn [x key]
                        (tk/mh (:trace (gfi/generate tt/scaled [] (cm/choicemap :x x :y 0.8) (rng/key 0))) scale [] key))
                      xs (rng/split (rng/key 8) n))
        [m s] (tt/mean-sd (mapv #(ts/number ((:trace %) :x)) stepped))]
    (is (< 0.2 (/ (count (filter :accepted? stepped)) n) 0.95))
    (is (< (Math/abs (- m 0.64)) (* 5 (/ (Math/sqrt 0.2) (Math/sqrt n)))) (str m))
    (is (< (Math/abs (- (* s s) 0.2)) (* 5 0.2 (Math/sqrt (/ 2.0 (dec n))))) (str (* s s)))))

(deftest a-read-the-update-keeps-is-an-output
  ;; (x, y, u) → (x, y·e^{xu}, -u): |det J| = e^{xu}
  (let [kern (gen [tr] (let [u (trace :u (d/normal 0 1))]
                         [(cm/choicemap :y (* (tr :y) (array/exp (* (tr :x) u)))) (cm/choicemap :u (array/negate u))]))
        tr (:trace (gfi/generate tt/two-latents [] (cm/choicemap :x 0.5 :y 2.0) (rng/key 9)))
        [k-propose k-update] (rng/split (rng/key 10) 3)
        u (cm/get-choice (:choices (gfi/propose kern [tr] k-propose)) :u)
        {la :log-alpha t' :trace} (tk/mh tr kern [] (rng/key 10))
        t'' (:trace (gfi/update tr (cm/choicemap :y (* 2.0 (Math/exp (* 0.5 (ts/number u))))) k-update))]
    (is (ts/close? (+ (- (:score t'') (:score tr)) (- (lp (d/normal 0 1) u)) (lp (d/normal 0 1) (- u)) (* 0.5 u)) la))))

;; SMCP3 on a linear-Gaussian state-space step: x_1 ~ N(0.8 x_0, 1),
;; y_1 ~ N(x_1, 0.5)

(def ssm (gen [n] (doseq [t (range n)]
                    (let [x (trace [t :x] (d/normal (if (zero? t) 0.0 0.0) 1))]
                      (trace [t :y] (d/normal x 0.5))))))

(def y-1 0.9)
(def v (/ 1.0 (+ 1.0 4.0)))
(defn- moments [] [(* v (* 4.0 y-1)) (Math/sqrt v)])

(def forward
  (gen [tr] (let [z (trace :z (d/normal 0 1)) [m s] (moments)]
              [(cm/choicemap [1 :x] (+ m (* s z))) (cm/choicemap)])))

(def backward
  (gen [tr] (let [[m s] (moments)] [(cm/choicemap) (cm/choicemap :z (/ (- (tr [1 :x]) m) s))])))

(deftest an-smcp3-step-weighs-a-reparameterized-proposal-exactly
  (let [tr (:trace (gfi/generate ssm [1] (cm/choicemap [0 :y] 0.3) (rng/key 11)))
        step (tk/smcp3 {:p-new-args [2] :new-observations (cm/choicemap [1 :y] y-1) :k forward :l backward :check? true})
        state (pf/initialize ssm [1] (cm/choicemap [0 :y] 0.3) 6 {:key (rng/key 12)})
        moved (pf/move-reweight state step (rng/key 13))
        [m s] (moments)]
    ;; z ~ N(0, 1) mapped to x = m + s·z: the weight is the score change less log N(x; m, s)
    (doseq [[tr tr' w w'] (map vector (:traces state) (:traces moved) (:log-weights state) (:log-weights moved))]
      (is (ts/close? (- w' w) (- (- (:score tr') (:score tr)) (lp (d/normal m s) (tr' [1 :x]))))))
    (testing "the round-trip check catches a backward kernel that is not the inverse"
      (let [wrong (gen [tr] [(cm/choicemap) (cm/choicemap :z (tr [1 :x]))])]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not give back"
                              ((tk/smcp3 {:p-new-args [2] :new-observations (cm/choicemap [1 :y] y-1) :k forward :l wrong
                                          :check? true})
                               tr (rng/key 14))))))))

(def walk-forward
  "Moves x_0 by a random walk and proposes x_1; hands the old x_0 back."
  (gen [tr] (let [w (trace :w (d/normal (tr [0 :x]) 0.3)) z (trace :z (d/normal 0 1)) [m s] (moments)]
              [(cm/choicemap [0 :x] w [1 :x] (+ m (* s z))) (cm/choicemap :back (tr [0 :x]))])))

(def walk-backward
  "Draws the old x_0 by the same walk; gives back the forward choices."
  (gen [tr] (let [back (trace :back (d/normal (tr [0 :x]) 0.3)) [m s] (moments)]
              [(cm/choicemap [0 :x] back) (cm/choicemap :w (tr [0 :x]) :z (/ (- (tr [1 :x]) m) s))])))

(deftest an-smcp3-step-with-a-backward-kernel
  ;; the walk's forward and backward densities cancel (it is symmetric), so
  ;; the weight is again the score change less log N(x_1; m, s)
  (let [state (pf/initialize ssm [1] (cm/choicemap [0 :y] 0.3) 6 {:key (rng/key 19)})
        moved (pf/move-reweight state (tk/smcp3 {:p-new-args [2] :new-observations (cm/choicemap [1 :y] y-1)
                                                 :k walk-forward :l walk-backward :check? true})
                                (rng/key 20))
        [m s] (moments)]
    (doseq [[tr tr' w w'] (map vector (:traces state) (:traces moved) (:log-weights state) (:log-weights moved))]
      (is (not= (tr [0 :x]) (tr' [0 :x])))
      (is (ts/close? (- w' w) (- (- (:score tr') (:score tr)) (lp (d/normal m s) (tr' [1 :x]))))))))

(deftest misuse-is-loud
  (let [tr (:trace (gfi/generate tt/two-latents [] (cm/choicemap :x 0.5 :y 2.0) (rng/key 15)))]
    (testing "a move that is not a bijection: (x, y, u) to (-x, y + u)"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"maps 3 dimensions to 2"
                            (tk/mh tr (gen [tr] (if (array/> (tr :x) 0)
                                                  (let [u (trace :u (d/normal 0 1))]
                                                    [(cm/choicemap :x (array/negate (tr :x)) :y (+ (tr :y) u)) (cm/choicemap)])
                                                  [(cm/choicemap :x (array/negate (tr :x))) (cm/choicemap)]))
                                   [] (rng/key 16)))))
    (testing "a constant output and randomness thrown away: (x, u) to (x' = 0, u' = x) (law audit)"
      (let [one (:trace (gfi/generate tt/two-latents [] (cm/choicemap :x 1.5 :y 2.0) (rng/key 21)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"singular|dimensions"
                              (tk/mh one (gen [tr] (trace :u (d/normal 0 1))
                                           [(cm/choicemap :x 0.0) (cm/choicemap :u (tr :x))])
                                     [] (rng/key 22))))))
    (testing "a kernel that is not its own reverse fails on the way back"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no value for the choice"
                            (tk/mh tr (gen [tr] (let [u (trace :u (d/normal 0 1))]
                                                  [(cm/choicemap :y (+ (tr :y) u)) (cm/choicemap)]))
                                   [] (rng/key 16)))))
    (testing "an update that leaves a new choice unwritten"
      (let [one (:trace (gfi/generate tt/two-means [] (cm/choicemap :two? false :mu 0.1 :y1 tt/y1 :y2 tt/y2) (rng/key 17)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unwritten"
                              (tk/mh one (gen [tr] [(cm/choicemap :two? true) (cm/choicemap)]) [] (rng/key 18))))))))

(defn- ldj-is?
  "Is the kernel's log |det J| at its proposal (f u), u its choice?"
  [f kernel tr key]
  (let [{kc :choices ret :retval} (gfi/propose kernel [tr] key)]
    (ts/close? (f (ts/number (cm/get-choice kc :u))) (tk/log-abs-det kernel tr [] kc ret (constantly true)))))

(deftest the-jacobian-sees-through-a-call-s-value
  ;; the post-M5 review's P3-04: the kernel runs on the trace its
  ;; substituted choices make, so a call's value is differentiated too.
  ;; Planted bug (2026-10-22): the trace with its choices substituted and
  ;; its calls' values old, as before, throws "maps 1 dimensions to 2" in
  ;; the first case and gives -0.554 (log 1.3 too little) and 0.0 in the
  ;; others
  (let [inner (gen [] (trace :x (d/normal 0 1)))
        model (gen [] (let [x (splice :sub inner)] (trace :y (d/normal x 1)) x))
        tr (:trace (gfi/generate model [] (cm/choicemap [:sub :x] 0.7 :y 0.2) (rng/key 30)))
        scaling (gen [tr] (let [u (trace :u (d/normal 0 0.5))]
                            [(cm/set-choice (cm/choicemap) [:sub :x] (* (tr :sub) (array/exp u))) (cm/choicemap :u (array/negate u))]))]
    (testing "x' = x·e^u, x read as the call's value: log |det J| = u"
      (is (ldj-is? identity scaling tr (rng/key 31)))
      (is (boolean? (:accepted? (tk/mh tr scaling [] (rng/key 32)))))))
  (testing "x' = x·s·e^u with s = e^x the call's value: log |det J| = log(1 + x) + x + u"
    (let [inner (gen [] (array/exp (trace :x (d/normal 0 1))))
          model (gen [] (let [s (splice :sub inner)] (trace :y (d/normal s 1)) s))
          tr (:trace (gfi/generate model [] (cm/choicemap [:sub :x] 0.3 :y 1.1) (rng/key 33)))
          kernel (gen [tr] (let [u (trace :u (d/normal 0 1))]
                             [(cm/set-choice (cm/choicemap) [:sub :x] (* (* (tr [:sub :x]) (tr :sub)) (array/exp u)))
                              (cm/choicemap :u (array/negate u))]))]
      (is (ldj-is? #(+ (Math/log 1.3) 0.3 %) kernel tr (rng/key 34)))))
  (testing "x' = x + s·u with s = x² + 1 the call's value (the verifier's): log |det J| = log |1 + 2xu|"
    (let [inner (gen [] (let [x (trace :x (d/normal 0 1))] (+ (* x x) 1.0)))
          model (gen [] (let [s (splice :sub inner)] (trace :y (d/normal s 1)) s))
          tr (:trace (gfi/generate model [] (cm/choicemap [:sub :x] 0.4 :y 1.5) (rng/key 35)))
          kernel (gen [tr] (let [u (trace :u (d/normal 0 1))]
                             [(cm/set-choice (cm/choicemap) [:sub :x] (+ (tr [:sub :x]) (* (tr :sub) u)))
                              (cm/choicemap :u (array/negate u))]))]
      (is (ldj-is? #(Math/log (Math/abs (+ 1.0 (* 2.0 0.4 %)))) kernel tr (rng/key 36))))))

(def ^:private posterior-mu
  "mu ~ N(0, 1), y ~ N(mu, 1)."
  (gen [] (let [mu (trace :mu (d/normal 0 1))] (trace :y (d/normal mu 1)))))

(deftest inputs-a-move-overwrites-or-discards-count
  ;; the post-M5 review's P3-07: a trace choice the update rewrites or
  ;; discards is an input, so a move that loses it is loud, except an
  ;; observation SMCP3's :new-observations replace, which is data. Planted
  ;; bugs (2026-10-22): those choices left out when no output depends on
  ;; them, and the early return (0.0 without continuous outputs), as
  ;; before, let the first three moves through silently (the forgetting
  ;; kernel's E[exp w] is 7 where 1 is right); counting the replaced
  ;; observation makes the last one loud
  (let [tr (:trace (gfi/generate posterior-mu [] (cm/choicemap :mu 0.3 :y 0.8) (rng/key 40)))
        none (gen [_tr] [(cm/choicemap) (cm/choicemap)])]
    (testing "mu := u, the old mu kept nowhere"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"maps 2 dimensions to 1"
                            ((tk/smcp3 {:k (gen [_tr] (let [u (trace :u (d/normal 0 1))] [(cm/choicemap :mu u) (cm/choicemap)]))
                                        :l none})
                             tr (rng/key 41)))))
    (testing "u thrown away, only a discrete output"
      (let [flagged (gen [] (trace :flag (d/bernoulli 0.5)) (trace :mu (d/normal 0 1)))
            tr (:trace (gfi/generate flagged [] (cm/choicemap :flag true) (rng/key 42)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"maps 1 dimensions to 0"
                              ((tk/smcp3 {:k (gen [_tr] [(cm/choicemap :flag (array/> (trace :u (d/normal 0 1)) 0.0)) (cm/choicemap)])
                                          :l none})
                               tr (rng/key 43))))))
    (testing "z discarded by a structure change (the verifier's)"
      (let [model (gen [] (let [flag (trace :flag (d/bernoulli 0.5)) m (if flag (trace :z (d/uniform 0 1)) 0.0)]
                            (trace :y (d/normal m 1))))
            tr (:trace (gfi/generate model [] (cm/choicemap :flag true :z 0.3 :y 1.0) (rng/key 44)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"maps 1 dimensions to 0"
                              ((tk/smcp3 {:k (gen [tr] (if (tr :flag) [(cm/choicemap :flag false) (cm/choicemap)] [nil nil]))
                                          :l none})
                               tr (rng/key 45))))))
    (testing "a new observation replaces y: the swap (mu, u) to (u, mu) weighs log N(0.9; u, 1) - log N(0.8; mu, 1)"
      (let [swap (gen [tr] (let [u (trace :u (d/normal 0 1))] [(cm/choicemap :mu u) (cm/choicemap :u (tr :mu))]))
            {t' :trace w :weight} ((tk/smcp3 {:k swap :l swap :new-observations (cm/choicemap :y 0.9)}) tr (rng/key 46))
            u (ts/number (t' :mu))]
        (is (ts/close? (- (lp (d/normal u 1) 0.9) (lp (d/normal 0.3 1) 0.8)) w))))))

(deftest a-model-s-encapsulated-randomness-is-allowed-where-the-kernel-does-not-read-it
  ;; m3-grammar's cross-review of P3-04's fix: the model's generate runs
  ;; outside with-exact-density (only the kernel needs exact densities), so
  ;; a pseudo-marginal walk works as before the fix; a kernel whose outputs
  ;; read the model's ω (its score here) fails the check at the trace's own
  ;; choices, loudly. Planted bugs (2026-10-22): the generate under
  ;; with-exact-density, as in d828198, throws "needs exact densities" in
  ;; the first two cases; without the check the third passes silently,
  ;; differentiated at another ω
  (let [lik (custom/estimated {:addresses [:y]
                               :estimate (fn [[mu] choices key]
                                           (+ (dc/log-prob (d/normal mu 1) (cm/get-choice choices :y))
                                              (- (* 0.1 (rng/normal key [])) 0.005)))})
        model (gen [] (let [mu (trace :mu (d/normal 0 1))] (splice :obs lik mu)))
        tr (:trace (gfi/generate model [] (cm/choicemap :mu 0.3 [:obs :y] 0.8) (rng/key 50)))
        walk (gen [tr] (let [u (trace :u (d/normal 0 0.5))] [(cm/choicemap :mu (+ (tr :mu) u)) (cm/choicemap :u (array/negate u))]))
        walked (fn [key] (let [[k-propose k-update] (rng/split key 3)
                               u (cm/get-choice (:choices (gfi/propose walk [tr] k-propose)) :u)]
                           (:weight (gfi/update tr (cm/choicemap :mu (+ 0.3 u)) k-update))))]
    (testing "MH: a walk's ratio is the update's weight (|det J| = 1)"
      (is (ts/close? (walked (rng/key 51)) (:log-alpha (tk/mh tr walk [] (rng/key 51))))))
    (testing "SMCP3: the same weight"
      (is (ts/close? (walked (rng/key 52)) (:weight ((tk/smcp3 {:k walk :l walk}) tr (rng/key 52))))))
    (testing "a kernel that reads the score, which depends on ω"
      (let [by-score (gen [tr] (let [u (trace :u (d/normal 0 0.5))]
                                 [(cm/choicemap :mu (+ (tr :mu) (* u (array/exp (* 0.01 (:score tr)))))) (cm/choicemap :u (array/negate u))]))
            {kc :choices ret :retval} (gfi/propose by-score [tr] (rng/key 53))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"encapsulated randomness"
                              (tk/log-abs-det by-score tr [] kc ret (constantly true))))))))
