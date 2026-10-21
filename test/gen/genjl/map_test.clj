(ns gen.genjl.map-test
  "Gen.jl 0.4.8's test/modeling_library/map.jl, on the checks it shares
   with unfold.jl (gen.genjl.indexed-combinators). Map's addresses are
   0-based (D2): Gen.jl's element i is element i - 1 here. The retdiffs
   asserted are gen.clj's, which are more precise than Gen.jl's (D1,
   DESIGN.md §9): an element whose value did not change is absent from a
   vector-diff, and a Map whose values all stayed returns no-change."
  (:refer-clojure :exclude [+ - update])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :refer [+ -]]
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
            [gen.selection :as sel]
            [gen.test-support :refer [close?]]))

(def foo
  "map.jl:3: z ~ normal(x + y, std), std a parameter."
  (gen [x y] (trace :z (d/normal (+ x y) (param :std)))))

(def bar
  "map.jl:11, foo's std 1 (map.jl:9's set_param!)."
  (comb/map (dyn/with-params foo {:std 1.0})))

(def ^:private check (partial indexed/check :z))
(def ^:private z (partial indexed/choice :z))

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
    (check tr {:retval [(z tr 0) (z tr 1)]
               :score (+ (lp (z tr 0) 4.0) (lp (z tr 1) 6.0))})
    ;; map.jl:25, trace[i]: element i's return value
    (check tr {:retval [(tr 0) (tr 1)]})))

;; map.jl:30
(deftest generate
  (check (gfi/generate bar (args 2) (cm/choicemap [0 :z] z1) (rng/key 3))
         {:choices {0 z1} :weight (lp z1 4.0)}))

;; map.jl:41
(deftest propose
  (let [r (gfi/propose bar (args 2) (rng/key 4))]
    (check r {:weight (+ (lp (z r 0) 4.0) (lp (z r 1) 6.0))})))

;; map.jl:49
(deftest assess
  (let [r (gfi/assess bar (args 2) both-constrained (rng/key 5))]
    (is (= 2 (count (:retval r))))
    (check r {:weight (+ (lp z1 4.0) (lp z2 6.0))})))

;; map.jl:60
(deftest update
  (let [tr (initial-trace)]
    (testing "unknown change to args, 2 to 3 elements, elements 1 and 2 constrained" ; map.jl:71
      (let [z2' 3.3 z3' 4.4]
        (check (indexed/update tr (args 3) unknown (cm/choicemap [1 :z] z2' [2 :z] z3'))
               {:args (args 3)
                :choices {0 z1 1 z2' 2 z3'}
                :discard {1 z2}
                :score (+ (lp z1 4.0) (lp z2' 6.0) (lp z3' 8.0))
                :weight (- (+ (lp z3' 8.0) (lp z2' 6.0)) (lp z2 6.0))
                :retval [z1 z2' z3']
                :retdiff :vector-diff
                ;; D1: element 0 kept its value (Gen.jl: updated[1] == UnknownChange())
                :not-updated [0 2]
                :updated {1 diff/unknown-change}})))
    (testing "unknown change to args, 2 to 4 elements, element 3 constrained, 2 from the prior" ; map.jl:104
      (let [z4 4.4
            r (indexed/update tr (args 4) unknown (cm/choicemap [3 :z] z4))]
        (check r {:discard {}
                  :args (args 4)
                  :choices {0 z1 1 z2 3 z4}
                  :score (+ (lp z1 4.0) (lp z2 6.0) (lp (z r 2) 8.0) (lp z4 10.0))
                  :weight (lp z4 10.0)
                  :retdiff :vector-diff
                  :lengths [2 4]
                  ;; D1: elements 0 and 1 kept their values (Gen.jl: updated[1], [2] == UnknownChange())
                  :not-updated [0 1]})))
    (testing "unknown change to args, 2 to 1 element, element 0 constrained" ; map.jl:132
      (let [z1' 3.3]
        (check (indexed/update tr (args 1) unknown (cm/choicemap [0 :z] z1'))
               {:args (args 1)
                :absent [1 2]
                :choices {0 z1'}
                :discard {0 z1 1 z2}
                :score (lp z1' 4.0)
                :weight (- (lp z1' 4.0) (lp z1 4.0) (lp z2 6.0))
                :retval [z1']
                :retdiff :vector-diff
                :updated {0 diff/unknown-change}
                :not-updated [1]})))
    (testing "no change to args, nothing constrained" ; map.jl:158
      (check (indexed/update tr (args 2) unchanged cm/EMPTY)
             {:args (args 2)
              :choices {0 z1 1 z2}
              :discard {}
              :score (+ (lp z1 4.0) (lp z2 6.0))
              :weight 0.0
              :retval [z1 z2]
              :retdiff :no-change}))
    (testing "no change to args, element 1 constrained" ; map.jl:178
      (let [z2' 3.3]
        (check (indexed/update tr (args 2) unchanged (cm/choicemap [1 :z] z2'))
               {:args (args 2)
                :choices {0 z1 1 z2'}
                :discard {1 z2}
                :score (+ (lp z1 4.0) (lp z2' 6.0))
                :weight (- (lp z2' 6.0) (lp z2 6.0))
                :retval [z1 z2']
                :retdiff :vector-diff
                :not-updated [0]
                :updated {1 diff/unknown-change}})))
    (testing "a custom argdiff, xs's element 0 changed, nothing constrained" ; map.jl:204
      (check (indexed/update tr xs-first-changed first-changed cm/EMPTY)
             {:args xs-first-changed
              :choices {0 z1 1 z2}
              :discard {}
              :score (+ (lp z1 2.0) (lp z2 6.0))
              :weight (- (lp z1 2.0) (lp z1 4.0))
              :retval [z1 z2]
              ;; D1: no value changed (Gen.jl: a VectorDiff, updated[1] == UnknownChange())
              :retdiff :no-change
              :not-updated [0 1]}))))

;; map.jl:232
(deftest regenerate
  (let [tr (initial-trace)]
    (testing "unknown change to args, 2 to 3 elements, element 1 selected" ; map.jl:243
      (let [r (indexed/regenerate tr (args 3) unknown (sel/select [1 :z]))]
        (check r {:args (args 3)
                  :choices {0 z1}
                  :score (+ (lp z1 4.0) (lp (z r 1) 6.0) (lp (z r 2) 8.0))
                  :weight 0.0
                  :retdiff :vector-diff
                  ;; D1: element 0 kept its value (Gen.jl: updated[1] == UnknownChange())
                  :not-updated [0 2]
                  :updated {1 diff/unknown-change}})))
    (testing "unknown change to args, 2 to 1 element, element 0 selected" ; map.jl:263
      (let [r (indexed/regenerate tr (args 1) unknown (sel/select [0 :z]))]
        (check r {:args (args 1)
                  :absent [1 2]
                  :score (lp (z r 0) 4.0)
                  :weight 0.0
                  :retdiff :vector-diff
                  :updated {0 diff/unknown-change}
                  :not-updated [1]})))
    (testing "no change to args, nothing selected" ; map.jl:279
      (check (indexed/regenerate tr (args 2) unchanged sel/none)
             {:args (args 2)
              :choices {0 z1 1 z2}
              :score (+ (lp z1 4.0) (lp z2 6.0))
              :weight 0.0
              :retval [z1 z2]
              :retdiff :no-change}))
    (testing "no change to args, element 1 selected" ; map.jl:297
      (let [r (indexed/regenerate tr (args 2) unchanged (sel/select [1 :z]))]
        (check r {:args (args 2)
                  :choices {0 z1}
                  :score (+ (lp z1 4.0) (lp (z r 1) 6.0))
                  :weight 0.0
                  :retval [z1 (z r 1)]
                  :retdiff :vector-diff
                  :not-updated [0]
                  :updated {1 diff/unknown-change}})))
    (testing "a custom argdiff, xs's element 0 changed, nothing selected" ; map.jl:318
      (check (indexed/regenerate tr xs-first-changed first-changed sel/none)
             {:args xs-first-changed
              :choices {0 z1 1 z2}
              :score (+ (lp z1 2.0) (lp z2 6.0))
              :weight (- (lp z1 2.0) (lp z1 4.0))
              :retval [z1 z2]
              ;; D1: no value changed (Gen.jl: a VectorDiff, updated[1] == UnknownChange())
              :retdiff :no-change
              :not-updated [0 1]}))))

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
    (is (close? z2 (cm/get-choice values [1 :z])))
    (is (not (support/has-value? gradients [0 :z])))
    (is (approx? (+ (first (lp-grad z2 6.0)) (retgrad 1)) (cm/get-choice gradients [1 :z])))))

(def ^:private model
  "bar holding foo's std, as set_param! does (D7)."
  (indexed/holding :map (comb/map foo) {:std 1.0}))

;; map.jl:376
(deftest accumulate-param-gradients
  (let [tr (:trace (gfi/generate model (args 2) (cm/set-submap cm/EMPTY :map both-constrained) (rng/key 6)))
        [gxs gys] (:arg-grads (g/choice-gradients tr sel/none retgrad {:argnums [0 1]}))
        [_ _ dstd1] (lp-grad z1 4.0)
        [_ _ dstd2] (lp-grad z2 6.0)]
    (is (approx? mean-grads gxs))
    (is (approx? mean-grads gys))
    (is (approx? (+ dstd1 dstd2) (:std (g/param-gradients tr retgrad))))))
