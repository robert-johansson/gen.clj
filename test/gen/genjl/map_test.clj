(ns gen.genjl.map-test
  "Gen.jl 0.4.8's test/modeling_library/map.jl. Map's addresses are 0-based
   (D2): Gen.jl's element i is element i - 1 here. The retdiffs asserted are
   gen.clj's, which are more precise than Gen.jl's (D1, DESIGN.md §9): an
   element whose value did not change is absent from a vector-diff, and a
   Map whose values all stayed returns no-change. Every retdiff is also
   checked with diff/consistent?. Gen.jl's isapprox is approx?, its ==
   close?."
  (:refer-clojure :exclude [+ - update])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :refer [+ -]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.genjl.finite-diff :refer [logpdf-grad]]
            [gen.genjl.support :as support :refer [approx?]]
            [gen.gfi :as gfi]
            [gen.gradients :as g]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :refer [close?]]))

(def foo
  "map.jl:3: z ~ normal(x + y, std), std a parameter."
  (gen [x y] (trace :z (d/normal (+ x y) (param :std)))))

(def bar
  "map.jl:11, foo's std 1 (map.jl:9's set_param!)."
  (comb/map (dyn/with-params foo {:std 1.0})))

(def ^:private xs [1.0 2.0 3.0 4.0])
(def ^:private ys [3.0 4.0 5.0 6.0])

(defn- args
  "(xs[1:n], ys[1:n])"
  [n]
  [(subvec xs 0 n) (subvec ys 0 n)])

(defn- lp
  "logpdf(normal, z, mu, 1.)"
  [z mu]
  (support/logpdf d/normal z mu 1.0))

(defn- z
  "choices[i => :z], 0-based"
  [c i]
  (cm/get-choice c [i :z]))

(def ^:private z1 1.1)
(def ^:private z2 2.2)

(def ^:private both-constrained (cm/choicemap [0 :z] z1 [1 :z] z2))

(defn- initial-trace
  "get_initial_trace: two elements, at z1 and z2."
  []
  (:trace (gfi/generate bar (args 2) both-constrained (rng/key 1))))

(def ^:private unknown [diff/unknown-change diff/unknown-change])
(def ^:private unchanged [diff/no-change diff/no-change])

(def ^:private first-changed
  "Gen.jl's custom argdiff: xs's first element changed, ys unchanged."
  [(diff/vector-diff 2 2 {0 diff/unknown-change}) diff/no-change])

(def ^:private xs-first-changed
  "(xs_new[1:2], ys[1:2]), xs_new[1] = -1"
  (assoc (args 2) 0 [-1.0 2.0]))

;; map.jl:15
(deftest untraced-call
  (is (= 4 (count (bar xs ys)))))

;; map.jl:19
(deftest simulate
  (let [tr (gfi/simulate bar (args 2) (rng/key 2))]
    (doseq [i [0 1]]
      (is (close? (tr [i :z]) (nth (:retval tr) i)) "a random choice"))
    (is (approx? (+ (lp (tr [0 :z]) 4.0) (lp (tr [1 :z]) 6.0)) (:score tr)))))

;; map.jl:30
(deftest generate
  (let [{tr :trace w :weight} (gfi/generate bar (args 2) (cm/choicemap [0 :z] z1) (rng/key 3))]
    (is (close? z1 (tr [0 :z])))
    (is (approx? (lp z1 4.0) w))))

;; map.jl:41
(deftest propose
  (let [{c :choices w :weight} (gfi/propose bar (args 2) (rng/key 4))]
    (is (approx? (+ (lp (z c 0) 4.0) (lp (z c 1) 6.0)) w))))

;; map.jl:49
(deftest assess
  (let [{w :weight rv :retval} (gfi/assess bar (args 2) both-constrained (rng/key 5))]
    (is (= 2 (count rv)))
    (is (approx? (+ (lp z1 4.0) (lp z2 6.0)) w))))

;; map.jl:60
(deftest update
  (let [tr (initial-trace)]
    ;; map.jl:71
    (testing "unknown change to args, 2 to 3 elements, elements 1 and 2 constrained"
      (let [z2' 3.3
            z3' 4.4
            {t :trace w :weight rd :retdiff disc :discard}
            (gfi/update tr (args 3) unknown (cm/choicemap [1 :z] z2' [2 :z] z3') (rng/key 6))
            c (:choices t)
            [r0 r1 r2 :as rv] (:retval t)]
        (is (= (args 3) (:args t)))
        (is (close? z1 (z c 0)))
        (is (close? z2' (z c 1)))
        (is (close? z3' (z c 2)))
        (is (close? z2 (z disc 1)))
        (is (approx? (+ (lp z1 4.0) (lp z2' 6.0) (lp z3' 8.0)) (:score t)))
        (is (approx? (- (+ (lp z3' 8.0) (lp z2' 6.0)) (lp z2 6.0)) w))
        (is (= 3 (count rv)))
        (is (close? z1 r0))
        (is (close? z2' r1))
        (is (close? z3' r2))
        (is (diff/vector-diff? rd))
        (is (not (contains? (:updated rd) 0)) "D1: element 0 kept its value")
        (is (= diff/unknown-change (get (:updated rd) 1)))
        (is (not (contains? (:updated rd) 2)) "new, not retained")
        (is (diff/consistent? (:retval tr) (:retval t) rd))))
    ;; map.jl:104
    (testing "unknown change to args, 2 to 4 elements, element 3 constrained and element 2 from the prior"
      (let [z4 4.4
            {t :trace w :weight rd :retdiff disc :discard}
            (gfi/update tr (args 4) unknown (cm/choicemap [3 :z] z4) (rng/key 7))
            c (:choices t)
            z3 (z c 2)]
        (is (cm/empty? disc))
        (is (= (args 4) (:args t)))
        (is (close? z1 (z c 0)))
        (is (close? z2 (z c 1)))
        (is (close? z4 (z c 3)))
        (is (approx? (+ (lp z1 4.0) (lp z2 6.0) (lp z3 8.0) (lp z4 10.0)) (:score t)))
        (is (approx? (lp z4 10.0) w))
        (is (diff/vector-diff? rd))
        (is (= 2 (:prev-length rd)))
        (is (= 4 (:new-length rd)))
        (is (not (contains? (:updated rd) 0)) "D1: element 0 kept its value")
        (is (not (contains? (:updated rd) 1)) "D1: element 1 kept its value")
        (is (diff/consistent? (:retval tr) (:retval t) rd))))
    ;; map.jl:132
    (testing "unknown change to args, 2 to 1 element, element 0 constrained"
      (let [z1' 3.3
            {t :trace w :weight rd :retdiff disc :discard}
            (gfi/update tr (args 1) unknown (cm/choicemap [0 :z] z1') (rng/key 8))
            c (:choices t)
            [r0 :as rv] (:retval t)]
        (is (= (args 1) (:args t)))
        (is (not (support/has-value? c [1 :z])))
        (is (not (support/has-value? c [2 :z])))
        (is (close? z1' (z c 0)))
        (is (close? z1 (z disc 0)))
        (is (close? z2 (z disc 1)))
        (is (approx? (lp z1' 4.0) (:score t)))
        (is (approx? (- (lp z1' 4.0) (lp z1 4.0) (lp z2 6.0)) w))
        (is (= 1 (count rv)))
        (is (close? z1' r0))
        (is (diff/vector-diff? rd))
        (is (= diff/unknown-change (get (:updated rd) 0)))
        (is (not (contains? (:updated rd) 1)) "deleted, not retained")
        (is (diff/consistent? (:retval tr) (:retval t) rd))))
    ;; map.jl:158
    (testing "no change to args, nothing constrained"
      (let [{t :trace w :weight rd :retdiff disc :discard} (gfi/update tr (args 2) unchanged cm/EMPTY (rng/key 9))
            c (:choices t)
            [r0 r1 :as rv] (:retval t)]
        (is (= (args 2) (:args t)))
        (is (close? z1 (z c 0)))
        (is (close? z2 (z c 1)))
        (is (cm/empty? disc))
        (is (approx? (+ (lp z1 4.0) (lp z2 6.0)) (:score t)))
        (is (approx? 0.0 w))
        (is (= 2 (count rv)))
        (is (close? z1 r0))
        (is (close? z2 r1))
        (is (diff/no-change? rd))
        (is (diff/consistent? (:retval tr) (:retval t) rd))))
    ;; map.jl:178
    (testing "no change to args, element 1 constrained"
      (let [z2' 3.3
            {t :trace w :weight rd :retdiff disc :discard}
            (gfi/update tr (args 2) unchanged (cm/choicemap [1 :z] z2') (rng/key 10))
            c (:choices t)
            [r0 r1 :as rv] (:retval t)]
        (is (= (args 2) (:args t)))
        (is (close? z1 (z c 0)))
        (is (close? z2' (z c 1)))
        (is (close? z2 (z disc 1)))
        (is (approx? (+ (lp z1 4.0) (lp z2' 6.0)) (:score t)))
        (is (approx? (- (lp z2' 6.0) (lp z2 6.0)) w))
        (is (= 2 (count rv)))
        (is (close? z1 r0))
        (is (close? z2' r1))
        (is (diff/vector-diff? rd))
        (is (not (contains? (:updated rd) 0)))
        (is (= diff/unknown-change (get (:updated rd) 1)))
        (is (diff/consistent? (:retval tr) (:retval t) rd))))
    ;; map.jl:204
    (testing "a custom argdiff, xs's element 0 changed, nothing constrained"
      (let [{t :trace w :weight rd :retdiff disc :discard}
            (gfi/update tr xs-first-changed first-changed cm/EMPTY (rng/key 11))
            c (:choices t)
            [r0 r1 :as rv] (:retval t)]
        (is (= xs-first-changed (:args t)))
        (is (close? z1 (z c 0)))
        (is (close? z2 (z c 1)))
        (is (cm/empty? disc))
        (is (approx? (+ (lp z1 2.0) (lp z2 6.0)) (:score t)))
        (is (approx? (- (lp z1 2.0) (lp z1 4.0)) w))
        (is (= 2 (count rv)))
        (is (close? z1 r0))
        (is (close? z2 r1))
        ;; D1: Gen.jl's VectorDiff naming element 0; its value did not change
        (is (diff/no-change? rd))
        (is (not (diff/element-changed? rd 0)))
        (is (not (diff/element-changed? rd 1)))
        (is (diff/consistent? (:retval tr) (:retval t) rd))))))

;; map.jl:232
(deftest regenerate
  (let [tr (initial-trace)]
    ;; map.jl:243
    (testing "unknown change to args, 2 to 3 elements, element 1 selected"
      (let [{t :trace w :weight rd :retdiff} (gfi/regenerate tr (args 3) unknown (sel/select [1 :z]) (rng/key 12))
            c (:choices t)]
        (is (= (args 3) (:args t)))
        (is (close? z1 (z c 0)))
        (is (approx? (+ (lp z1 4.0) (lp (z c 1) 6.0) (lp (z c 2) 8.0)) (:score t)))
        (is (approx? 0.0 w))
        (is (diff/vector-diff? rd))
        (is (not (contains? (:updated rd) 0)) "D1: element 0 kept its value")
        (is (= diff/unknown-change (get (:updated rd) 1)))
        (is (not (contains? (:updated rd) 2)) "new, not retained")
        (is (diff/consistent? (:retval tr) (:retval t) rd))))
    ;; map.jl:263
    (testing "unknown change to args, 2 to 1 element, element 0 selected"
      (let [{t :trace w :weight rd :retdiff} (gfi/regenerate tr (args 1) unknown (sel/select [0 :z]) (rng/key 13))
            c (:choices t)]
        (is (= (args 1) (:args t)))
        (is (not (support/has-value? c [1 :z])))
        (is (not (support/has-value? c [2 :z])))
        (is (approx? (lp (z c 0) 4.0) (:score t)))
        (is (approx? 0.0 w))
        (is (diff/vector-diff? rd))
        (is (= diff/unknown-change (get (:updated rd) 0)))
        (is (not (contains? (:updated rd) 1)) "deleted, not retained")
        (is (diff/consistent? (:retval tr) (:retval t) rd))))
    ;; map.jl:279
    (testing "no change to args, nothing selected"
      (let [{t :trace w :weight rd :retdiff} (gfi/regenerate tr (args 2) unchanged sel/none (rng/key 14))
            c (:choices t)
            [r0 :as rv] (:retval t)]
        (is (= (args 2) (:args t)))
        (is (close? z1 (z c 0)))
        (is (close? z2 (z c 1)))
        (is (approx? (+ (lp z1 4.0) (lp z2 6.0)) (:score t)))
        (is (approx? 0.0 w))
        (is (= 2 (count rv)))
        (is (close? z1 r0))
        (is (diff/no-change? rd))
        (is (diff/consistent? (:retval tr) (:retval t) rd))))
    ;; map.jl:297
    (testing "no change to args, element 1 selected"
      (let [{t :trace w :weight rd :retdiff} (gfi/regenerate tr (args 2) unchanged (sel/select [1 :z]) (rng/key 15))
            c (:choices t)
            [r0 r1 :as rv] (:retval t)]
        (is (= (args 2) (:args t)))
        (is (close? z1 (z c 0)))
        (is (approx? (+ (lp z1 4.0) (lp (z c 1) 6.0)) (:score t)))
        (is (approx? 0.0 w))
        (is (= 2 (count rv)))
        (is (close? z1 r0))
        (is (close? (z c 1) r1))
        (is (diff/vector-diff? rd))
        (is (not (contains? (:updated rd) 0)))
        (is (= diff/unknown-change (get (:updated rd) 1)))
        (is (diff/consistent? (:retval tr) (:retval t) rd))))
    ;; map.jl:318
    (testing "a custom argdiff, xs's element 0 changed, nothing selected"
      (let [{t :trace w :weight rd :retdiff} (gfi/regenerate tr xs-first-changed first-changed sel/none (rng/key 16))
            c (:choices t)
            [r0 r1 :as rv] (:retval t)]
        (is (= xs-first-changed (:args t)))
        (is (close? z1 (z c 0)))
        (is (close? z2 (z c 1)))
        (is (approx? (+ (lp z1 2.0) (lp z2 6.0)) (:score t)))
        (is (approx? (- (lp z1 2.0) (lp z1 4.0)) w))
        (is (= 2 (count rv)))
        (is (close? z1 r0))
        (is (close? z2 r1))
        ;; D1: Gen.jl's VectorDiff naming element 0; its value did not change
        (is (diff/no-change? rd))
        (is (not (diff/element-changed? rd 0)))
        (is (not (diff/element-changed? rd 1)))
        (is (diff/consistent? (:retval tr) (:retval t) rd))))))

(def ^:private retgrad
  "Gen.jl's retval_grad, rand(2): any values serve."
  [0.3 0.7])

(defn- lp-grad
  "logpdf_grad(normal, z, mu, 1.): [dz dmu dstd]"
  [z mu]
  (logpdf-grad d/normal z mu 1.0))

(def ^:private mean-grads
  "expected_xs_grad, and expected_ys_grad: z's mean is x + y, so both are
   the gradients in the means."
  [(second (lp-grad z1 4.0)) (second (lp-grad z2 6.0))])

;; map.jl:345
(deftest choice-gradients
  (let [{:keys [values gradients] [gxs gys] :arg-grads}
        (g/choice-gradients (initial-trace) (sel/select [1 :z]) retgrad {:argnums [0 1]})]
    (is (approx? mean-grads gxs))
    (is (approx? mean-grads gys))
    (is (not (support/has-value? values [0 :z])))
    (is (close? z2 (z values 1)))
    (is (not (support/has-value? gradients [0 :z])))
    (is (approx? (+ (first (lp-grad z2 6.0)) (retgrad 1)) (z gradients 1)))))

(def ^:private model
  "bar as a model holding foo's std (D7: one parameter store per model). A
   Map has no store of its own, so a dynamic function splicing it holds it,
   as set_param! sets foo's."
  (dyn/with-params (gen [xs ys] (splice :map (comb/map foo) xs ys)) {:std 1.0}))

;; map.jl:376
(deftest accumulate-param-gradients
  (let [tr (:trace (gfi/generate model (args 2) (cm/set-submap cm/EMPTY :map both-constrained) (rng/key 17)))
        [gxs gys] (:arg-grads (g/choice-gradients tr sel/none retgrad {:argnums [0 1]}))
        [_ _ dstd1] (lp-grad z1 4.0)
        [_ _ dstd2] (lp-grad z2 6.0)]
    (is (approx? mean-grads gxs))
    (is (approx? mean-grads gys))
    (is (approx? (+ dstd1 dstd2) (:std (g/param-gradients tr retgrad))))))

;; Pending
;; - map.jl:25, get_retval(trace)[i] == trace[i]: a call's return value at
;;   its address (G1), genclj-cbec.
