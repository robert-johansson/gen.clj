(ns gen.inference.transform-test
  "Trace transforms and translators: Jacobians against closed forms;
   translators whose weights are exactly 0 (a pushforward, a change of
   variables) or exactly the evidence (an extension by the conditional);
   involutive MCMC, a reversible jump included, one step from the exact
   posterior leaves it as it was; the round-trip checks; loud misuse."
  (:refer-clojure :exclude [read + - * /])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dist.dsl :refer [dist]]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.kernel :as k]
            [gen.inference.mh :as mh]
            [gen.inference.particle-filter :as pf]
            [gen.inference.transform :as tf :refer [deftransform read write copy]]
            [gen.random :as rng]
            [gen.test-support :as ts]))

(defn- lp [d v] (ts/number (dc/log-prob d v)))

;; ---------------------------------------------------------------------------
;; Jacobians
;; ---------------------------------------------------------------------------

(deftransform scale-move
  "(x, u) → (x·e^u, -u): |det J| = e^u, and it is its own inverse."
  [model-in aux-in model-out aux-out]
  (let [x (read model-in :x :continuous)
        u (read aux-in :u :continuous)]
    (write model-out :x (* x (array/exp u)) :continuous)
    (write aux-out :u (array/negate u) :continuous)))

(deftest jacobians-match-the-closed-forms
  (let [{:keys [model aux log-abs-det]} (tf/run scale-move (cm/choicemap :x 2.0) (cm/choicemap :u 0.3))]
    (is (ts/close? (* 2.0 (Math/exp 0.3)) (cm/get-choice model :x)))
    (is (ts/close? -0.3 (cm/get-choice aux :u)))
    (is (ts/close? 0.3 log-abs-det)))
  (testing "an array-valued choice"
    (let [t (tf/transform [mi ai mo ao] (write mo :v (* (array/array [2.0 3.0]) (read mi :v :continuous)) :continuous))]
      (is (ts/close? (Math/log 6.0) (:log-abs-det (tf/run t (cm/choicemap :v (array/array [1.0 -1.0])) nil))))))
  (testing "copies of unread choices are left out"
    (let [t (tf/transform [mi ai mo ao]
              (write mo :x (* 3.0 (read mi :x :continuous)) :continuous)
              (copy mi :z mo :z))]
      (is (ts/close? (Math/log 3.0) (:log-abs-det (tf/run t (cm/choicemap :x 1.0 :z 5.0) nil))))))
  (testing "a copy of a continuous read is an output: (x, u) → (x·u, x) has |det| = |x|"
    (let [t (tf/transform [mi ai mo ao]
              (let [x (read mi :x :continuous) u (read ai :u :continuous)]
                (write mo :x (* x u) :continuous)
                (copy mi :x ao :x)))]
      (is (ts/close? (Math/log 2.0) (:log-abs-det (tf/run t (cm/choicemap :x 2.0) (cm/choicemap :u 3.0)))))))
  (testing "discrete reads and writes do not count"
    (let [t (tf/transform [mi ai mo ao]
              (write mo :k (inc (read mi :k :discrete)) :discrete)
              (write mo :x (* 0.5 (read mi :x :continuous)) :continuous))]
      (is (ts/close? (Math/log 0.5) (:log-abs-det (tf/run t (cm/choicemap :k 1 :x 1.0) nil)))))))

;; ---------------------------------------------------------------------------
;; Translators
;; ---------------------------------------------------------------------------

(dist log-normal [mu sd] (array/exp (d/normal mu sd)))

(def p-normal (gen [] (trace :x (d/normal 0.3 0.8))))
(def p-log-normal (gen [] (trace :z (log-normal 0.3 0.8))))

(deftransform exp-of-x [model-in aux-in model-out aux-out]
  (write model-out :z (array/exp (read model-in :x :continuous)) :continuous))

(deftest a-pushforward-translates-with-weight-zero
  ;; z = e^x is exactly the log-normal: p(z)·|dz/dx| = p(x)
  (let [translate (tf/deterministic-translator {:p-new p-log-normal :p-new-args [] :transform exp-of-x})]
    (doseq [key (rng/split (rng/key 1) 5)]
      (let [tr (gfi/simulate p-normal [] key)
            {t' :trace w :weight} (translate tr (rng/key 2))]
        (is (ts/close? (array/exp (tr :x)) (t' :z)))
        (is (ts/close? 0.0 w {:atol (ts/tol 1e-9)}))))))

(deftest a-deterministic-translator-has-no-proposals
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no auxiliary choices"
                        (tf/deterministic-translator {:p-new p-log-normal :p-new-args [] :transform exp-of-x
                                                      :q-forward p-normal}))))

(def p-sum (gen [] (trace :s (d/normal 0 (Math/sqrt 2)))))
(def p-pair (gen [] (trace :a (d/normal 0 1)) (trace :b (d/normal 0 1))))
(def difference (gen [_] (trace :d (d/normal 0 (Math/sqrt 2)))))

(def to-pair
  (tf/transform [model-in aux-in model-out aux-out]
    (let [s (read model-in :s :continuous) dd (read aux-in :d :continuous)]
      (write model-out :a (/ (+ s dd) 2.0) :continuous)
      (write model-out :b (/ (- s dd) 2.0) :continuous))))

(def to-sum
  (tf/transform [model-in aux-in model-out aux-out]
    (let [a (read model-in :a :continuous) b (read model-in :b :continuous)]
      (write model-out :s (+ a b) :continuous)
      (write aux-out :d (- a b) :continuous))))

(deftest pairing-is-symmetric
  (let [[f g] (tf/pair to-pair to-sum)]
    (is (= (:body to-sum) (:body (tf/inverse f))))
    (is (= (:body to-pair) (:body (tf/inverse (tf/inverse f)))) "the inverse's inverse")
    (is (= (:body to-pair) (:body (tf/inverse g))))))

(deftest a-change-of-variables-translates-with-weight-zero
  ;; a, b iid N(0, 1) is s = a + b and d = a - b iid N(0, 2), Jacobian 1/2
  (let [[to-pair to-sum] (tf/pair to-pair to-sum)
        forward (tf/translator {:p-new p-pair :q-forward difference :transform to-pair :check? true})
        backward (tf/translator {:p-new p-sum :q-backward difference :transform to-sum :check? true})]
    (doseq [key (rng/split (rng/key 3) 5)]
      (let [tr (gfi/simulate p-sum [] key)
            {t' :trace w :weight} (forward tr (rng/key 4))]
        (is (ts/close? (tr :s) (+ (t' :a) (t' :b))))
        (is (ts/close? 0.0 w {:atol (ts/tol 1e-9)}) "forward")
        (is (ts/close? 0.0 (:weight (backward t' (rng/key 5))) {:atol (ts/tol 1e-9)}) "backward")))
    (testing "the round-trip check catches a wrong inverse"
      (let [[f _] (tf/pair to-pair (tf/transform [mi ai mo ao]
                                     (write mo :s (+ (read mi :a :continuous) (read mi :b :continuous)) :continuous)
                                     (write ao :d (* 2.0 (- (read mi :a :continuous) (read mi :b :continuous))) :continuous)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not give back the auxiliary"
                              ((tf/translator {:p-new p-pair :q-forward difference :transform f :check? true})
                               (gfi/simulate p-sum [] (rng/key 6)) (rng/key 7))))))))

(def growing
  (gen [n] (doseq [i (range n)] (let [x (trace [:x i] (d/normal 0 1))] (trace [:y i] (d/normal x 1))))))

(def conditional
  "The exact p(x_1 | y_1)."
  (gen [_ y] (trace [:x 1] (d/normal (/ y 2.0) (Math/sqrt 0.5)))))

(deftest extending-by-the-conditional-weighs-the-evidence
  (let [y 1.1
        extend (tf/extending-translator {:p-new-args [2] :p-argdiffs [gen.diff/unknown-change]
                                         :new-observations (cm/choicemap [:y 1] y)
                                         :q-forward conditional :q-forward-args [y]})
        tr (:trace (gfi/generate growing [1] (cm/choicemap [:y 0] 0.4) (rng/key 8)))]
    (doseq [key (rng/split (rng/key 9) 4)]
      (let [{t' :trace w :weight} (extend tr key)]
        (is (= 4 (count (cm/addresses (:choices t')))))
        (is (ts/close? (lp (d/normal 0 (Math/sqrt 2)) y) w) "log p(y_1), whatever x_1")))
    (testing "an extension may not change choices"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"may only add"
                            ((tf/extending-translator {:p-new-args [2] :p-argdiffs [gen.diff/unknown-change]
                                                       :new-observations (cm/choicemap [:y 0] 9.0)})
                             tr (rng/key 10)))))))

(deftest a-translator-reweighs-particles
  (let [state (pf/initialize p-normal [] cm/EMPTY 10 {:key (rng/key 11)})
        moved (pf/move-reweight state (tf/deterministic-translator {:p-new p-log-normal :p-new-args [] :transform exp-of-x}) (rng/key 12))]
    (is (every? #(contains? (set (cm/addresses (:choices %))) [:z]) (pf/traces moved)))
    (is (every? true? (map #(ts/close? %1 %2 {:atol (ts/tol 1e-9)}) (:log-weights state) (:log-weights moved))))))

;; ---------------------------------------------------------------------------
;; Involutive MCMC
;; ---------------------------------------------------------------------------

(defn normals [key n mu sd] (mapv #(+ mu (* sd %)) (ts/host (dc/sample (d/normal 0 1) key [n]))))

(defn mean-sd [xs]
  (let [n (count xs) m (/ (reduce clojure.core/+ xs) n)]
    [m (Math/sqrt (/ (reduce clojure.core/+ (map #(Math/pow (- % m) 2) xs)) (dec n)))]))

(def scaled
  "x ~ N(0, 1), y ~ N(x, 0.5) with y = 0.8: x | y ~ N(0.64, √0.2)."
  (gen [] (let [x (trace :x (d/normal 0 1))] (trace :y (d/normal x 0.5)))))

(def log-scale (gen [_] (trace :u (d/normal 0 0.5))))

(deftest ^:slow involutive-mh-with-a-transform-keeps-the-posterior
  (let [n 4000
        xs (normals (rng/key 13) n 0.64 (Math/sqrt 0.2))
        move (tf/involution (tf/involutive scale-move))
        stepped (mapv (fn [x key]
                        (let [tr (:trace (gfi/generate scaled [] (cm/choicemap :x x :y 0.8) (rng/key 0)))]
                          (mh/involutive-mh tr log-scale [] move key)))
                      xs (rng/split (rng/key 14) n))
        xs' (mapv #(ts/number ((:trace %) :x)) stepped)
        accepted (count (filter :accepted? stepped))
        [m s] (mean-sd xs')]
    (is (< 0.2 (/ accepted n) 0.95) (str accepted " accepted"))
    ;; mean 0.64, sd √0.2: five standard errors of each over n chains
    (is (< (Math/abs (- m 0.64)) (* 5 (/ (Math/sqrt 0.2) (Math/sqrt n)))) (str m))
    (is (< (Math/abs (- (* s s) 0.2)) (* 5 0.2 (Math/sqrt (/ 2.0 (dec n))))) (str (* s s)))))

(def two-means
  "One mean for both observations, or one each (prior 1/2): a jump between
   dimensions."
  (gen []
    (if (trace :two? (d/bernoulli 0.5))
      (let [mu1 (trace :mu1 (d/normal 0 1)) mu2 (trace :mu2 (d/normal 0 1))]
        (trace :y1 (d/normal mu1 1)) (trace :y2 (d/normal mu2 1)))
      (let [mu (trace :mu (d/normal 0 1))]
        (trace :y1 (d/normal mu 1)) (trace :y2 (d/normal mu 1))))))

(def split-proposal (gen [tr] (when-not (tr :two?) (trace :u (d/normal 0 1)))))

(deftransform split-merge [model-in aux-in model-out aux-out]
  (if (read model-in :two? :discrete)
    (let [mu1 (read model-in :mu1 :continuous) mu2 (read model-in :mu2 :continuous)]
      (write model-out :two? false :discrete)
      (write model-out :mu (/ (+ mu1 mu2) 2.0) :continuous)
      (write aux-out :u (/ (- mu2 mu1) 2.0) :continuous))
    (let [mu (read model-in :mu :continuous) u (read aux-in :u :continuous)]
      (write model-out :two? true :discrete)
      (write model-out :mu1 (- mu u) :continuous)
      (write model-out :mu2 (+ mu u) :continuous))))

(def y1 0.3)
(def y2 1.9)

(def p-two
  "P(two? | y): p(y | one) = N(y; 0, [[2 1] [1 2]]), p(y | two) = N(y1; 0, √2) N(y2; 0, √2)."
  (let [one (- (- (Math/log (* 2 Math/PI))) (* 0.5 (Math/log 3.0))
               (/ (+ (* 2 y1 y1) (* -2 y1 y2) (* 2 y2 y2)) 6.0))
        two (+ (lp (d/normal 0 (Math/sqrt 2)) y1) (lp (d/normal 0 (Math/sqrt 2)) y2))]
    (/ 1.0 (+ 1.0 (Math/exp (- one two))))))

(defn exact-posterior-trace
  "A trace drawn from the exact posterior: two? by p-two, then the means from
   their Gaussian conditionals (precision 3 for the shared one, 2 for each)."
  [key]
  (let [[k1 k2 k3] (rng/split key 3)
        two? (< (ts/number (rng/uniform k1)) p-two)
        z (ts/host (dc/sample (d/normal 0 1) k2 [2]))
        obs (cm/choicemap :y1 y1 :y2 y2)
        choices (if two?
                  (cm/merge obs (cm/choicemap :two? true :mu1 (+ (/ y1 2.0) (* (Math/sqrt 0.5) (z 0)))
                                              :mu2 (+ (/ y2 2.0) (* (Math/sqrt 0.5) (z 1)))))
                  (cm/merge obs (cm/choicemap :two? false :mu (+ (/ (+ y1 y2) 3.0) (* (Math/sqrt (/ 1.0 3)) (z 0))))))]
    (:trace (gfi/generate two-means [] choices k3))))

(deftest a-reversible-jump-keeps-the-posterior
  (testing "the jump's weight: the score ratio and log 2 to split, -log 2 to merge"
    (let [one (:trace (gfi/generate two-means [] (cm/choicemap :two? false :mu 0.7 :y1 y1 :y2 y2) (rng/key 15)))
          {split :trace w :weight bwd :bwd-choices} ((tf/involution split-merge) one (cm/choicemap :u 0.2) nil nil (rng/key 0))]
      (is (ts/close? 0.5 (split :mu1)))
      (is (ts/close? 0.9 (split :mu2)))
      (is (cm/empty? bwd))
      (is (ts/close? (+ (- (:score split) (:score one)) (Math/log 2.0)) w))
      (let [{merged :trace w' :weight bwd' :bwd-choices} ((tf/involution split-merge) split cm/EMPTY nil nil (rng/key 0))]
        (is (ts/close? 0.7 (merged :mu)))
        (is (ts/close? 0.2 (cm/get-choice bwd' :u)))
        (is (ts/close? (- (- (:score merged) (:score split)) (Math/log 2.0)) w')))))
  (let [n 8000
        move (tf/involution split-merge)
        after (mapv (fn [key] (let [[k1 k2] (rng/split key)]
                                (:trace (mh/involutive-mh (exact-posterior-trace k1) split-proposal [] move k2))))
                    (rng/split (rng/key 17) n))
        frac (/ (count (filter #(% :two?) after)) (double n))]
    (is (< (Math/abs (- frac p-two)) (* 5 (Math/sqrt (/ (* p-two (- 1 p-two)) n)))) (str frac " against " p-two))))

(def two-latents (gen [] (trace :x (d/normal 0 1)) (trace :y (d/normal 0 1))))

(deftest a-read-the-update-keeps-is-an-output
  ;; (x, y, u) → (x, y·e^{xu}, -u), x read and kept: |det J| = e^{xu}
  (let [t (tf/transform [model-in aux-in model-out aux-out]
            (let [x (read model-in :x :continuous) y (read model-in :y :continuous) u (read aux-in :u :continuous)]
              (write model-out :y (* y (array/exp (* x u))) :continuous)
              (write aux-out :u (array/negate u) :continuous)))
        tr (:trace (gfi/generate two-latents [] (cm/choicemap :x 0.5 :y 2.0) (rng/key 20)))
        {t' :trace w :weight} ((tf/involution t) tr (cm/choicemap :u 0.6) nil nil (rng/key 0))]
    (is (ts/close? 0.5 (t' :x)))
    (is (ts/close? (+ (- (:score t') (:score tr)) (* 0.5 0.6)) w))))

(deftest the-involution-check
  (let [tr (exact-posterior-trace (rng/key 18))
        u (if (tr :two?) cm/EMPTY (cm/choicemap :u 0.4))]
    (is (map? ((tf/involution split-merge {:check? true}) tr u nil nil (rng/key 0))))
    (let [broken (tf/transform [model-in aux-in model-out aux-out]
                   (if (read model-in :two? :discrete)
                     (let [mu1 (read model-in :mu1 :continuous) mu2 (read model-in :mu2 :continuous)]
                       (write model-out :two? false :discrete)
                       (write model-out :mu (/ (+ mu1 mu2) 2.0) :continuous)
                       (write aux-out :u (- mu2 mu1) :continuous))
                     (let [mu (read model-in :mu :continuous) u (read aux-in :u :continuous)]
                       (write model-out :two? true :discrete)
                       (write model-out :mu1 (- mu u) :continuous)
                       (write model-out :mu2 (+ mu u) :continuous))))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not give back"
                            ((tf/involution broken {:check? true}) tr u nil nil (rng/key 0)))))))

;; ---------------------------------------------------------------------------
;; Misuse
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [x1 (cm/choicemap :x 1.0)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"maps 2 dimensions to 1"
                          (tf/run (tf/transform [mi ai mo ao]
                                    (let [x (read mi :x :continuous) u (read ai :u :continuous)]
                                      (write mo :x (+ x u) :continuous)))
                                  x1 (cm/choicemap :u 0.1))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"written twice"
                          (tf/run (tf/transform [mi ai mo ao] (write mo :x 1.0 :discrete) (write mo :x 2.0 :discrete)) x1 nil)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no choice at"
                          (tf/run (tf/transform [mi ai mo ao] (read mi :nope :discrete)) x1 nil)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"read as"
                          (tf/run (tf/transform [mi ai mo ao] (read mi :x :discrete) (read mi :x :continuous)) x1 nil)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"is not one of"
                          (tf/run (tf/transform [mi ai mo ao] (write mi :x 1.0 :discrete)) x1 nil)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"the tag is"
                          (tf/run (tf/transform [mi ai mo ao] (read mi :x :real)) x1 nil)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"inside a transform" (read :model-in :x :discrete)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no inverse" (tf/inverse scale-move)))
    (testing "a lazy loop that writes is an error at definition"
      (is (re-find #"is lazy"
                   (try (eval '(gen.inference.transform/transform [mi ai mo ao]
                                   (for [i (range 2)] (gen.inference.transform/write mo [:x i] 1.0 :discrete))))
                        ""
                        (catch Exception e (ex-message (or (ex-cause e) e))))))
      (let [t (tf/transform [mi ai mo ao] (doseq [i (range 2)] (write mo [:x i] (* 2.0 (read mi [:x i] :continuous)) :continuous)))]
        (is (ts/close? (Math/log 4.0) (:log-abs-det (tf/run t (cm/choicemap [:x 0] 1.0 [:x 1] 2.0) nil))) "doseq runs")))
    (testing "copying a subtree copies the reads under it: (x) to (2x, x) is no bijection"
      (let [t (tf/transform [mi ai mo ao]
                (write mo :y (* 2.0 (read mi [:a :x] :continuous)) :continuous)
                (copy mi :a mo :a))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"maps 1 dimensions to 2"
                              (tf/run t (cm/choicemap [:a :x] 1.0) nil)))))
    (testing "an extending transform may not write auxiliary choices"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"writes auxiliary choices"
                            ((tf/extending-translator
                              {:p-new-args [] :q-forward (gen [_] (trace :z (d/normal 0 1)))
                               :transform (tf/transform [mi ai mo ao]
                                            (write mo :x (read mi :z :continuous) :continuous)
                                            (write ao :d 1.0 :continuous))})
                             (gfi/simulate p-normal [] (rng/key 30)) (rng/key 31)))))
    (testing "inputs are traces or choice maps"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"from-map" (tf/run scale-move {:x 1.0} nil))))
    (testing "every translator checks its transform"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs a transform" (tf/involution :nope)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"needs a transform" (tf/updating-translator {:transform :nope}))))
    (testing "an involution must write every new choice"
      (let [half (tf/transform [model-in aux-in model-out aux-out]
                   (read model-in :mu :continuous) (read aux-in :u :continuous)
                   (write model-out :two? true :discrete))
            tr (:trace (gfi/generate two-means [] (cm/choicemap :two? false :mu 0.1 :y1 y1 :y2 y2) (rng/key 19)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"did not write"
                              ((tf/involution half) tr (cm/choicemap :u 0.2) nil nil (rng/key 0))))))))

(def walk-u (gen [tr] (trace :u (d/normal (tr :x) 0.5))))

(def swap
  "x and the proposed u exchanged: |det J| = 1."
  (tf/transform [model-in aux-in model-out aux-out]
    (write model-out :x (read aux-in :u :continuous) :continuous)
    (write aux-out :u (read model-in :x :continuous) :continuous)))

(deftest a-swap-involution-is-plain-mh
  ;; Eq. 3.16 (law audit gap J): involutive MH with a swap is MH with the
  ;; proposal: log α = log p(x') - log p(x) - log q(x'; x) + log q(x; x')
  (doseq [key (rng/split (rng/key 40) 6)]
    (let [tr (:trace (gfi/generate scaled [] (cm/choicemap :x 0.3 :y 0.8) (rng/key 41)))
          {t' :trace la :log-alpha} (mh/involutive-mh tr walk-u [] (tf/involution swap) key)
          [k-propose] (rng/split key)
          x' (cm/get-choice (:choices (gfi/propose walk-u [tr] k-propose)) :u)
          moved (:trace (gfi/update tr (cm/choicemap :x x') (rng/key 0)))]
      (is (ts/close? (+ (- (:score moved) (:score tr)) (- (lp (d/normal 0.3 0.5) x')) (lp (d/normal x' 0.5) 0.3)) la)))))

(deftest a-checked-involutive-kernel-catches-a-write-to-an-observation
  ;; Alg. 11 (law audit gap J): an involution that moves y, a bijection, so
  ;; only the check stops it once a move is accepted
  (let [writes-y (tf/transform [model-in aux-in model-out aux-out]
                   (write model-out :x (read aux-in :u :continuous) :continuous)
                   (write model-out :y (+ (read model-in :y :continuous) 0.1) :continuous)
                   (write aux-out :u (read model-in :x :continuous) :continuous))
        tr (:trace (gfi/generate scaled [] (cm/choicemap :x 0.3 :y 0.8) (rng/key 42)))
        kernel (k/checked (k/involutive-mh walk-u [] (tf/involution writes-y)) (cm/choicemap :y 0.8))
        outcomes (map #(try ((:trace (kernel tr %)) :y) (catch clojure.lang.ExceptionInfo e (ex-message e)))
                      (rng/split (rng/key 43) 10))]
    (is (some string? outcomes))
    (is (every? #(or (= 0.8 %) (re-find #"observ" %)) outcomes) (str outcomes))))

(def p-sum-observed (gen [] (let [s (trace :s (d/normal 0 (Math/sqrt 2)))] (trace :o (d/normal s 1)))))
(def p-pair-observed
  (gen [] (let [a (trace :a (d/normal 0 1)) b (trace :b (d/normal 0 1))]
            (trace :o (d/normal (+ a b) 1))
            (trace :r (d/normal (- a b) 1)))))

(deftest ^:slow ^:heavy translating-particles-between-models-keeps-the-estimate-unbiased
  ;; Eq. 3.13, Alg. 9 (law audit gap H): particles of P1 given o = 0.5,
  ;; translated to P2 given o and r = 0.3; a + b and a - b are independent
  ;; N(0, 2), so p2(o, r) = N(.5; 0, √3) N(.3; 0, √3)
  (let [[forward _] (tf/pair to-pair to-sum)
        translate (tf/translator {:p-new p-pair-observed :q-forward difference :transform forward
                                  :new-observations (cm/choicemap :o 0.5 :r 0.3)})
        evidence (+ (lp (d/normal 0 (Math/sqrt 3)) 0.5) (lp (d/normal 0 (Math/sqrt 3)) 0.3))
        runs 400
        ratios (mapv (fn [key]
                       (let [[k1 k2] (rng/split key)
                             state (pf/initialize p-sum-observed [] (cm/choicemap :o 0.5) 20 {:key k1})]
                         (Math/exp (- (pf/log-ml-estimate (pf/move-reweight state translate k2)) evidence))))
                     (rng/split (rng/key 50) runs))
        [m sd] (mean-sd ratios)]
    (is (< (Math/abs (- m 1.0)) (* 5 (/ sd (Math/sqrt runs)))) (str m " ± " (/ sd (Math/sqrt runs))))))

(def merge-or-walk
  "From two means a fair coin picks a merge or a random walk on mu1; from
   one, a split always: move probabilities that depend on the state."
  (gen [tr] (if (tr :two?)
              (when-not (trace :merge? (d/bernoulli 0.5)) (trace :w (d/normal 0 0.3)))
              (trace :u (d/normal 0 1)))))

(deftransform one-or-two [model-in aux-in model-out aux-out]
  (if (read model-in :two? :discrete)
    (if (read aux-in :merge? :discrete)
      (let [mu1 (read model-in :mu1 :continuous) mu2 (read model-in :mu2 :continuous)]
        (write model-out :two? false :discrete)
        (write model-out :mu (/ (+ mu1 mu2) 2.0) :continuous)
        (write aux-out :u (/ (- mu2 mu1) 2.0) :continuous))
      (let [mu1 (read model-in :mu1 :continuous) w (read aux-in :w :continuous)]
        (write model-out :mu1 (+ mu1 w) :continuous)
        (write aux-out :merge? false :discrete)
        (write aux-out :w (array/negate w) :continuous)))
    (let [mu (read model-in :mu :continuous) u (read aux-in :u :continuous)]
      (write model-out :two? true :discrete)
      (write model-out :mu1 (- mu u) :continuous)
      (write model-out :mu2 (+ mu u) :continuous)
      (write aux-out :merge? true :discrete))))

(deftest ^:slow state-dependent-move-probabilities-keep-the-posterior
  ;; §3.7.5, pp. 133-135 (law audit gap I): the coin's 1/2 enters the ratio
  ;; of a split and a merge, and must not be dropped
  (let [n 8000
        move (tf/involution one-or-two {:check? true})
        after (mapv (fn [key] (let [[k1 k2] (rng/split key)]
                                (:trace (mh/involutive-mh (exact-posterior-trace k1) merge-or-walk [] move k2))))
                    (rng/split (rng/key 60) n))
        frac (/ (count (filter #(% :two?) after)) (double n))]
    (is (< (Math/abs (- frac p-two)) (* 5 (Math/sqrt (/ (* p-two (- 1 p-two)) n)))) (str frac " against " p-two))))
