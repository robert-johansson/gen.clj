(ns gen.genjl.unfold-test
  "Gen.jl 0.4.8's test/modeling_library/unfold.jl, on the checks it shares
   with map.jl (gen.genjl.indexed-combinators). Unfold's steps are 0-based
   (D2): Gen.jl's step i is step i - 1 here. The kernel is static, so
   Gen.jl's retdiffs are as precise as gen.clj's."
  (:refer-clojure :exclude [+ - * update])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :refer [+ - *]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.genjl.finite-diff :refer [logpdf-grad]]
            [gen.genjl.indexed-combinators :as indexed]
            [gen.genjl.support :as support :refer [approx?]]
            [gen.gfi :as gfi]
            [gen.gradients :as g]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def ^:private std 1.0)

(def kernel
  "unfold.jl:3: x ~ normal(x_prev * alpha + beta, std)."
  (gen ^:static [t x-prev alpha beta]
    (let [x (trace :x (d/normal (+ (* x-prev alpha) beta) std))] x)))

(def foo (comb/unfold kernel))

(def ^:private check (partial indexed/check :x))
(def ^:private x (partial indexed/choice :x))

(def ^:private x-init 0.1)
(def ^:private alpha 0.2)
(def ^:private alpha' 0.5)
(def ^:private beta 0.3)
(def ^:private x1 1.1)
(def ^:private x2 1.2)

(defn- lp
  "logpdf(normal, x, prev * alpha + beta, std)"
  [x prev alpha]
  (support/logpdf d/normal x (+ (* prev alpha) beta) std))

(defn- xs
  "The steps' choices of a result, in order."
  [r n]
  (mapv #(x r %) (range n)))

(defn- chain-score
  "The log density of the states xs from x-init."
  [xs alpha]
  (reduce + (map #(lp %2 %1 alpha) (cons x-init xs) xs)))

;; unfold.jl:12
(deftest untraced-call
  (is (= 5 (count (foo 5 0.0 1.0 1.0)))))

;; unfold.jl:16
(deftest simulate
  (let [tr (gfi/simulate foo [3 x-init alpha beta] (rng/key 1))]
    (check tr {:shallow {:values 0 :submaps 3}
               :score (chain-score (xs tr 3) alpha)
               :retval (xs tr 3)})))

;; unfold.jl:41
(deftest generate
  (let [x3 1.3
        r (gfi/generate foo [3 x-init alpha beta] (cm/choicemap [0 :x] x1 [2 :x] x3) (rng/key 2))
        x2 (x r 1)]
    (check r {:choices {0 x1 2 x3}
              :shallow {:values 0 :submaps 3}
              :weight (+ (lp x1 x-init alpha) (lp x3 x2 alpha))
              :score (chain-score [x1 x2 x3] alpha)
              :retval [x1 x2 x3]})))

;; unfold.jl:72
(deftest propose
  (let [r (gfi/propose foo [3 x-init alpha beta] (rng/key 3))]
    (check r {:shallow {:values 0 :submaps 3}
              :weight (chain-score (xs r 3) alpha)
              :retval (xs r 3)})))

;; unfold.jl:92
(deftest assess
  (let [x3 1.3]
    (check (gfi/assess foo [3 x-init alpha beta] (cm/choicemap [0 :x] x1 [1 :x] x2 [2 :x] x3) (rng/key 4))
           {:weight (chain-score [x1 x2 x3] alpha)
            :retval [x1 x2 x3]})))

(defn- initial-trace
  "get_initial_trace: two steps, at x1 and x2."
  []
  (:trace (gfi/generate foo [2 x-init alpha beta] (cm/choicemap [0 :x] x1 [1 :x] x2) (rng/key 5))))

(def ^:private unknown (vec (repeat 4 diff/unknown-change)))
(def ^:private unchanged (diff/no-changes 4))
(def ^:private init-changed [diff/no-change diff/unknown-change diff/no-change diff/no-change])
(def ^:private params-changed [diff/no-change diff/no-change diff/unknown-change diff/unknown-change])

;; unfold.jl:114
(deftest update
  (let [tr (initial-trace)]
    (testing "unknown change to args, 2 to 3 steps, steps 1 and 2 constrained, alpha changed" ; unfold.jl:129
      (let [x2' 1.3 x3' 1.4]
        (check (indexed/update tr [3 x-init alpha' beta] unknown (cm/choicemap [1 :x] x2' [2 :x] x3'))
               {:args [3 x-init alpha' beta]
                :choices {0 x1 1 x2' 2 x3'}
                :discard {1 x2}
                :score (chain-score [x1 x2' x3'] alpha')
                :weight (- (+ (lp x3' x2' alpha') (lp x2' x1 alpha') (lp x1 x-init alpha'))
                           (lp x2 x1 alpha) (lp x1 x-init alpha))
                :retval [x1 x2' x3']
                :retdiff :vector-diff
                :not-updated [0 2]
                :updated {1 diff/unknown-change}})))
    (testing "unknown change to args, 2 to 1 step, step 0 constrained, alpha changed" ; unfold.jl:166
      (let [x1' 1.3]
        (check (indexed/update tr [1 x-init alpha' beta] unknown (cm/choicemap [0 :x] x1'))
               {:args [1 x-init alpha' beta]
                :absent [1 2]
                :choices {0 x1'}
                :discard {0 x1 1 x2}
                :score (lp x1' x-init alpha')
                :weight (- (lp x1' x-init alpha') (lp x1 x-init alpha) (lp x2 x1 alpha))
                :retval [x1']
                :retdiff :vector-diff
                :updated {0 diff/unknown-change}
                :not-updated [1]})))
    (testing "2 to 4 steps, step 3 constrained, step 2 from the prior, alpha changed" ; unfold.jl:194
      (let [x4' 1.3
            r (indexed/update tr [4 x-init alpha' beta] unknown (cm/choicemap [3 :x] x4'))
            x3' (x r 2)]
        (check r {:discard {}
                  :args [4 x-init alpha' beta]
                  :choices {0 x1 1 x2 3 x4'}
                  :score (chain-score [x1 x2 x3' x4'] alpha')
                  :weight (+ (- (lp x1 x-init alpha') (lp x1 x-init alpha))
                             (- (lp x2 x1 alpha') (lp x2 x1 alpha))
                             (lp x4' x3' alpha'))
                  :retval [x1 x2 x3' x4']
                  :retdiff :vector-diff
                  :not-updated [0 1 2 3]})))
    (testing "no change to args, nothing constrained" ; unfold.jl:234
      (check (indexed/update tr [2 x-init alpha beta] unchanged cm/EMPTY)
             {:args [2 x-init alpha beta]
              :choices {0 x1 1 x2}
              :discard {}
              :score (chain-score [x1 x2] alpha)
              :weight 0.0
              :retdiff :no-change}))
    (testing "no change to args, step 1 constrained" ; unfold.jl:251
      (let [x2' 3.3]
        (check (indexed/update tr [2 x-init alpha beta] unchanged (cm/choicemap [1 :x] x2'))
               {:args [2 x-init alpha beta]
                :choices {0 x1 1 x2'}
                :discard {1 x2}
                :score (chain-score [x1 x2'] alpha)
                :weight (- (lp x2' x1 alpha) (lp x2 x1 alpha))
                :retval [x1 x2']
                :retdiff :vector-diff
                :not-updated [0]
                :updated {1 diff/unknown-change}})))
    (testing "the initial state changed (to the same value), nothing constrained" ; unfold.jl:278
      (let [x-init' 0.1]
        (check (indexed/update tr [2 x-init' alpha beta] init-changed cm/EMPTY)
               {:args [2 x-init' alpha beta]
                :choices {0 x1 1 x2}
                :discard {}
                :score (+ (lp x1 x-init' alpha) (lp x2 x1 alpha))
                :weight (- (lp x1 x-init' alpha) (lp x1 x-init alpha))
                :retval [x1 x2]
                :retdiff :no-change})))
    (testing "the parameters changed, nothing constrained" ; unfold.jl:301
      (check (indexed/update tr [2 x-init alpha' beta] params-changed cm/EMPTY)
             {:args [2 x-init alpha' beta]
              :choices {0 x1 1 x2}
              :discard {}
              :score (chain-score [x1 x2] alpha')
              :weight (- (chain-score [x1 x2] alpha') (chain-score [x1 x2] alpha))
              :retval [x1 x2]
              :retdiff :no-change}))))

;; unfold.jl:327
(deftest regenerate
  (let [tr (initial-trace)]
    (testing "unknown change to args, 2 to 3 steps, step 1 selected, alpha changed" ; unfold.jl:342
      (let [r (indexed/regenerate tr [3 x-init alpha' beta] unknown (sel/select [1 :x]))
            [_ x2' x3'] (xs r 3)]
        (check r {:args [3 x-init alpha' beta]
                  :choices {0 x1}
                  :score (chain-score [x1 x2' x3'] alpha')
                  :weight (- (lp x1 x-init alpha') (lp x1 x-init alpha))
                  :retval [x1 x2' x3']
                  :retdiff :vector-diff
                  :not-updated [0 2]
                  :updated {1 diff/unknown-change}})))
    (testing "unknown change to args, 2 to 1 step, step 0 selected, alpha changed" ; unfold.jl:371
      (let [r (indexed/regenerate tr [1 x-init alpha' beta] unknown (sel/select [0 :x]))
            x1' (x r 0)]
        (check r {:args [1 x-init alpha' beta]
                  :absent [1 2]
                  :score (lp x1' x-init alpha')
                  :weight 0.0
                  :retval [x1']
                  :retdiff :vector-diff
                  :updated {0 diff/unknown-change}
                  :not-updated [1]})))
    (testing "no change to args, nothing selected" ; unfold.jl:392
      (check (indexed/regenerate tr [2 x-init alpha beta] unchanged sel/none)
             {:args [2 x-init alpha beta]
              :choices {0 x1 1 x2}
              :score (chain-score [x1 x2] alpha)
              :weight 0.0
              :retval [x1 x2]
              :retdiff :no-change}))
    (testing "no change to args, step 1 selected" ; unfold.jl:411
      (let [r (indexed/regenerate tr [2 x-init alpha beta] unchanged (sel/select [1 :x]))
            x2' (x r 1)]
        (check r {:args [2 x-init alpha beta]
                  :choices {0 x1}
                  :score (chain-score [x1 x2'] alpha)
                  :weight 0.0
                  :retval [x1 x2']
                  :retdiff :vector-diff
                  :not-updated [0]
                  :updated {1 diff/unknown-change}})))
    (testing "the initial state changed, nothing selected" ; unfold.jl:434
      (let [x-init' -0.1]
        (check (indexed/regenerate tr [2 x-init' alpha beta] init-changed sel/none)
               {:args [2 x-init' alpha beta]
                :choices {0 x1 1 x2}
                :score (+ (lp x1 x-init' alpha) (lp x2 x1 alpha))
                :weight (- (lp x1 x-init' alpha) (lp x1 x-init alpha))
                :retval [x1 x2]
                :retdiff :no-change})))
    (testing "the parameters changed, nothing selected" ; unfold.jl:456
      (check (indexed/regenerate tr [2 x-init alpha' beta] params-changed sel/none)
             {:args [2 x-init alpha' beta]
              :choices {0 x1 1 x2}
              :score (chain-score [x1 x2] alpha')
              :weight (- (chain-score [x1 x2] alpha') (chain-score [x1 x2] alpha))
              :retval [x1 x2]
              :retdiff :no-change}))))

(defn- dmean
  "logpdf_grad(normal, x, mean, sd)[2]"
  [x mean sd]
  (second (logpdf-grad d/normal x mean sd)))

(def dynamic-kernel
  "unfold.jl:483: the kernel with std a parameter."
  (gen [t x-prev alpha beta] (trace :x (d/normal (+ (* x-prev alpha) beta) (param :std)))))

(def ^:private chain
  "Unfold(kernel), the kernel holding its std, as set_param! sets it."
  (comb/unfold (dyn/with-params dynamic-kernel {:std 1.0})))

;; unfold.jl:481: argument gradients (D7: choice-gradients' :argnums, the
;; length and the initial state not differentiated, as in Gen.jl) and the
;; parameter's, from the Unfold's trace (its kernel's store, DESIGN §9)
(deftest accumulate-param-gradients-of-a-dynamic-kernel
  (let [tr (:trace (gfi/generate chain [2 x-init alpha beta] (cm/choicemap [0 :x] x1 [1 :x] x2) (rng/key 6)))
        [dn dinit dalpha dbeta] (:arg-grads (g/choice-gradients tr sel/none nil {:argnums [2 3]}))
        m1 (+ (* x-init alpha) beta)
        m2 (+ (* x1 alpha) beta)
        [_ _ dstd1] (logpdf-grad d/normal x1 m1 std)
        [_ _ dstd2] (logpdf-grad d/normal x2 m2 std)]
    (is (nil? dn) "the length")
    (is (nil? dinit) "the initial state")
    (is (approx? (+ (* (dmean x1 m1 std) x-init) (* (dmean x2 m2 std) x1)) dalpha))
    (is (approx? (+ (dmean x1 m1 std) (dmean x2 m2 std)) dbeta))
    (is (approx? (+ dstd1 dstd2) (:std (g/param-gradients tr))))))

(def ker
  "unfold.jl:519: 2x, no choices."
  (gen [t x] (* 2.0 x)))

;; unfold.jl:523: Gen.jl's nothing retgrads are zeros here
(deftest accumulate-param-gradients-through-ten-steps
  (let [tr (gfi/simulate (comb/unfold ker) [10 1.0] (rng/key 7))
        retgrad (conj (vec (repeat 9 0.0)) 1.0)]
    (is (approx? 1024.0 (second (:arg-grads (g/choice-gradients tr sel/none retgrad {:argnums [1]})))))))

(def gradient-kernel
  "unfold.jl:530: x ~ normal(x_prev * alpha + beta, 1)."
  (gen [t x-prev alpha beta] (trace :x (d/normal (+ (* x-prev alpha) beta) 1.0))))

;; unfold.jl:538: a choice's gradient is its own log density's plus the next
;; step's, whose mean it is (alpha 1, beta 0), plus its return value's
(deftest choice-gradients
  (let [tr (gfi/simulate (comb/unfold gradient-kernel) [10 0.0 1.0 0.0] (rng/key 8))
        chm (:choices tr)
        xi (fn [i] (if (neg? i) 0.0 (cm/get-choice chm [i :x])))
        expected (fn [i] (+ (first (logpdf-grad d/normal (xi i) (xi (dec i)) 1.0))
                            (if (= i 9) 0.0 (second (logpdf-grad d/normal (xi (inc i)) (xi i) 1.0)))))
        gradient (fn [i retgrad] (cm/get-choice (:gradients (g/choice-gradients tr (sel/select [i :x]) retgrad)) [i :x]))]
    (testing "propagation over the state"
      (doseq [i (range 10)]
        (is (approx? (expected i) (gradient i nil)) (str "step " i))))
    (testing "with a return-value gradient at the step"
      (doseq [i (range 10)]
        (is (approx? (+ (expected i) 1.0) (gradient i (assoc (vec (repeat 10 0.0)) i 1.0))) (str "step " i))))))
