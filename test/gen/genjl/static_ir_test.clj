(ns gen.genjl.static-ir-test
  "Gen.jl 0.4.8's test/static_ir/static_ir.jl. Gen.jl builds its functions
   with StaticIRBuilder, which is internal; the ports write the bodies its
   comments give, in gen ^:static. A counter counts how often a
   computation runs, so the tests see what update and regenerate rerun.

   gen.clj's static language keeps every computation's value and tracks
   diffs, Gen.jl's (static, diffs) and cache_julia_nodes together (D10): a
   count that assumes Gen.jl's nojuliacache, and the untracked variant's
   own retdiff, are not ported. A splice passes no-change for an argument
   identical to its previous call's (D1, DESIGN.md §9), so where Gen.jl
   reruns a callee given the same b with UnknownChange, gen.clj does not:
   those assertions are gen.clj's."
  (:refer-clojure :exclude [+ - *])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :refer [+ - *]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.genjl.expectations :as expectations]
            [gen.genjl.finite-diff :as fd]
            [gen.genjl.support :as support :refer [approx?]]
            [gen.gfi :as gfi]
            [gen.gradients :as g]
            [gen.inference.mh :as mh]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :refer [close?]]
            [gen.trace :refer [trace?]]))

(def ^:private runs
  "How often the counted computations ran: Gen.jl's counter."
  (atom 0))

(defn- counted
  "x, counting a run."
  [x]
  (swap! runs inc)
  x)

(def bar
  "static_ir.jl:7: z = (y + b) + v, the sum counted."
  (gen ^:static [y b]
    (let [u (counted (+ y b))
          v (trace :v (d/normal 0 1))
          z (+ u v)]
      z)))

(def ^:private theta-val
  "Gen.jl's rand(): any value serves."
  0.37)

(def foo
  "static_ir.jl:27, theta set (set_param!)."
  (dyn/with-params
    (gen ^:static [a b]
      (let [x (trace :x (d/normal 0 1))
            y (trace :y (d/normal x a))
            z (splice :z bar y b)
            w (+ z 1 a (param :theta))]
        w))
    {:theta theta-val}))

(def const-fn
  "static_ir.jl:54 (its builder's constant is 2)"
  (gen ^:static [] (let [] 2)))

;; static_ir.jl:65
(deftest julia-call
  (is (= 2 (const-fn))))

(defn- lp [x mu sd] (support/logpdf d/normal x mu sd))

(defn- expected-score [a b x y v]
  (+ (lp x 0 1) (lp y x a) (lp v 0 1)))

;; static_ir.jl:75
(deftest simulate
  (let [[a b] [1 2]
        tr (gfi/simulate foo [a b] (rng/key 1))
        [x y v] (map tr [:x :y [:z :v]])
        z (+ y b v)]
    (is (approx? (expected-score a b x y v) (:score tr)))
    (is (approx? (+ z 1 a theta-val) (:retval tr)))))

;; static_ir.jl:89
(deftest generate
  (let [[a b] [1 2]
        [y v] [1.123 2.345]
        {tr :trace w :weight} (gfi/generate foo [a b] (cm/choicemap :y y [:z :v] v) (rng/key 2))
        x (tr :x)]
    (is (close? y (tr :y)))
    (is (close? v (tr [:z :v])))
    (is (approx? (expected-score a b x y v) (:score tr)))
    (is (approx? (- (:score tr) (lp x 0 1)) w) "the score less x's proposal")))

;; static_ir.jl:108
(deftest project
  (let [[a b] [1 2]
        [y v x] [1.123 2.345 3.456]
        tr (:trace (gfi/generate foo [a b] (cm/choicemap :y y [:z :v] v :x x) (rng/key 3)))]
    (is (approx? (+ (lp y x a) (lp v 0 1)) (gfi/project tr (sel/select :y [:z :v]))))))

(defn- revise
  "update (constraints) or regenerate (a selection) of tr, with :runs, the
   counted computations' runs."
  [op tr args argdiffs x]
  (reset! runs 0)
  (let [r ((case op :update gfi/update :regenerate gfi/regenerate) tr args argdiffs x (rng/key 4))]
    (assoc r :runs @runs)))

(def ^:private checks
  "Each key's assertions, from a revision (with :runs) and the expectation."
  {:choices (fn [{t :trace} want] (doseq [[addr v] want] (is (close? v (t addr)) (str "choices[" addr "]"))))
   :changed (fn [{t :trace} want] (doseq [[addr old] want] (is (not (close? old (t addr))) (str addr " != its old value"))))
   :retval (fn [{t :trace} want] (is (approx? want (:retval t)) "get_retval(new_trace)"))
   :weight (fn [r want] (is (approx? want (:weight r)) "weight"))
   :retdiff (fn [r want] (is (= want (expectations/diff-kind (:retdiff r))) "retdiff"))
   :discard (fn [{d :discard} want]
              (if (empty? want)
                (is (cm/empty? d) "isempty(discard)")
                (doseq [[addr v] want] (is (close? v (cm/get-choice d addr)) (str "discard[" addr "]")))))
   :runs (fn [r want] (is (= want (:runs r)) "counter"))
   :score (fn [{t :trace} want] (is (approx? want (:score t)) "get_score(new_trace)"))})

(defn- check
  "Gen.jl's assertions of a revision, a key per kind: :choices {addr v},
   :changed {addr old} (a resampled value differs from its old one),
   :retval, :weight, :retdiff (:no-change or :unknown-change), :discard
   ({} isempty, else {addr v}), :runs (the counter) and :score."
  [result expected]
  (expectations/check checks result expected))

(def ^:private unchanged [diff/no-change diff/no-change])
(def ^:private a-changed [diff/unknown-change diff/no-change])
(def ^:private b-changed [diff/no-change diff/unknown-change])

;; static_ir.jl:120
(deftest update-without-tracked-diffs
  (let [[a b] [1 2]
        tr (gfi/simulate foo [a b] (rng/key 5))
        [x y v] (map tr [:x :y [:z :v]])
        w (:retval tr)
        same {:x x :y y [:z :v] v}]
    (testing "no change, nothing constrained" ; static_ir.jl:131
      (check (revise :update tr [a b] unchanged cm/EMPTY)
             {:choices same :retval w :weight 0.0 :retdiff :no-change :discard {} :runs 0
              :score (expected-score a b x y v)}))
    (testing "a changed, nothing constrained" ; static_ir.jl:144
      (let [a' 3]
        (check (revise :update tr [a' b] a-changed cm/EMPTY)
               {:choices same :retval (+ w (- a' a)) :weight (- (lp y x a') (lp y x a))
                :retdiff :unknown-change :discard {} :runs 0 :score (expected-score a' b x y v)})))
    (testing "b changed, nothing constrained" ; static_ir.jl:158
      ;; D1: b is passed to bar again, identical, so bar gets no-change and
      ;; reruns nothing (Gen.jl: counter 1, retdiff UnknownChange())
      (check (revise :update tr [a b] b-changed cm/EMPTY)
             {:choices same :retval w :weight 0.0 :retdiff :no-change :discard {} :runs 0
              :score (expected-score a b x y v)}))
    (testing "y constrained" ; static_ir.jl:171
      (let [y' 1.123]
        (check (revise :update tr [a b] unchanged (cm/choicemap :y y'))
               {:choices (assoc same :y y') :retval (+ w (- y' y)) :weight (- (lp y' x a) (lp y x a))
                :retdiff :unknown-change :discard {:y y} :runs 1 :score (expected-score a b x y' v)})))
    (testing ":z => :v constrained" ; static_ir.jl:185
      ;; no :runs: bar's sum is kept (D10), where Gen.jl's nojuliacache reruns it
      (let [v' 2.345]
        (check (revise :update tr [a b] unchanged (cm/choicemap [:z :v] v'))
               {:choices (assoc same [:z :v] v') :retval (+ w (- v' v)) :weight (- (lp v' 0 1) (lp v 0 1))
                :retdiff :unknown-change :discard {[:z :v] v} :score (expected-score a b x y v')})))))

;; static_ir.jl:200
(deftest regenerate-without-tracked-diffs
  (let [[a b] [1 2]
        tr (gfi/simulate foo [a b] (rng/key 5))
        [x y v] (map tr [:x :y [:z :v]])
        w (:retval tr)
        same {:x x :y y [:z :v] v}]
    (testing "no change, nothing selected" ; static_ir.jl:211
      (check (revise :regenerate tr [a b] unchanged sel/none)
             {:choices same :retval w :weight 0.0 :retdiff :no-change :runs 0 :score (expected-score a b x y v)}))
    (testing "a changed, nothing selected" ; static_ir.jl:223
      (let [a' 3]
        (check (revise :regenerate tr [a' b] a-changed sel/none)
               {:choices same :retval (+ w (- a' a)) :weight (- (lp y x a') (lp y x a))
                :retdiff :unknown-change :runs 0 :score (expected-score a' b x y v)})))
    (testing "b changed, nothing selected" ; static_ir.jl:236
      ;; D1, as in update's third case (Gen.jl: counter 1, retdiff UnknownChange())
      (check (revise :regenerate tr [a b] b-changed sel/none)
             {:choices same :retval w :weight 0.0 :retdiff :no-change :runs 0 :score (expected-score a b x y v)}))
    (testing "y selected" ; static_ir.jl:249
      (let [r (revise :regenerate tr [a b] unchanged (sel/select :y))
            y' ((:trace r) :y)]
        (check r {:changed {:y y} :choices (dissoc same :y) :retval (+ w (- y' y)) :weight 0.0
                  :retdiff :unknown-change :runs 1 :score (expected-score a b x y' v)})))
    (testing ":z => :v selected" ; static_ir.jl:262
      ;; no :runs, as in update's last case (D10)
      (let [r (revise :regenerate tr [a b] unchanged (sel/select [:z :v]))
            v' ((:trace r) [:z :v])]
        (check r {:changed {[:z :v] v} :choices (dissoc same [:z :v]) :retval (+ w (- v' v)) :weight 0.0
                  :retdiff :unknown-change :score (expected-score a b x y v')})))))

(def bar-b
  "static_ir.jl:279: z + mu_z, z ~ normal(mu_z, 1)."
  (gen ^:static [mu-z] (let [z (trace :z (d/normal mu-z 1))] (+ z mu-z))))

(def ^:private theta -0.5)

(def foo-b
  "static_ir.jl:294, theta initialized (init_param!)."
  (dyn/with-params
    (gen ^:static [mu-a]
      (let [theta (param :theta)
            a (trace :a (d/normal mu-a 1))
            b (trace :b (d/normal a 1))
            bar-val (splice :bar bar-b a)
            c (* a b bar-val theta)
            out (trace :out (d/normal c 1))]
        out))
    {:theta theta}))

(defn- f
  "static_ir.jl:319: foo's log density, plus the return value times its
   gradient, 2."
  [mu-a theta a b z out]
  (let [mu-z a
        c (* a b (+ z mu-z) theta)]
    (+ (lp z mu-z 1) (lp a mu-a 1) (lp b a 1) (lp out c 1) (* 2 out))))

;; static_ir.jl:277: choice and parameter gradients against f's central
;; differences. Gradients are values here (D7): get_param_grad's zeros
;; before and after zero_param_grad! (340, 378) have no counterpart.
(deftest backprop
  (let [[mu-a a b z out] [1.0 2.0 3.0 4.0 5.0]
        fd-at (fn [i] (fd/finite-diff f [mu-a theta a b z out] i))
        tr (:trace (gfi/generate foo-b [mu-a] (cm/choicemap :a a :b b :out out [:bar :z] z) (rng/key 6)))
        param (fn [] (:theta (:params (:gen-fn tr))))]
    (is (= theta (param)))
    (testing "choice_gradients" ; static_ir.jl:350
      (let [{:keys [values gradients] [mu-a-grad] :arg-grads}
            (g/choice-gradients tr (sel/select [:bar :z] :a :out) 2.0 {:argnums [0]})]
        (is (fd/grad-close? mu-a-grad (fd-at 0)))
        (is (close? a (cm/get-choice values :a)))
        (is (close? out (cm/get-choice values :out)))
        (is (close? z (cm/get-choice values [:bar :z])))
        (is (not (support/has-value? values :b)) "not selected")
        (is (= {:values 2 :submaps 1} (support/shallow values)))
        (is (= {:values 2 :submaps 1} (support/shallow gradients)))
        (is (not (support/has-value? gradients :b)) "not selected")
        (is (fd/grad-close? (cm/get-choice gradients :a) (fd-at 2)))
        (is (fd/grad-close? (cm/get-choice gradients :out) (fd-at 5)))
        (is (fd/grad-close? (cm/get-choice gradients [:bar :z]) (fd-at 4)))))
    (is (= theta (param)) "the gradients leave the parameter")
    (testing "accumulate_param_gradients!" ; static_ir.jl:380
      (is (fd/grad-close? (first (:arg-grads (g/choice-gradients tr sel/none 2.0 {:argnums [0]}))) (fd-at 0)))
      (is (fd/grad-close? (:theta (g/param-gradients tr 2.0)) (fd-at 1))))))

(def bar-t
  "static_ir.jl:398: 0, counted, so its retdiff is always no-change."
  (gen ^:static [a] (let [x (counted 0)] x)))

(def foo-t
  "static_ir.jl:413: bar's retdiff makes its own."
  (gen ^:static [a] (let [x (splice :x bar-t a) y (+ x 1)] y)))

(def foo-without-tracked-diffs
  "static_ir.jl:430: gen.clj's static language always tracks diffs (D10),
   so the untracked variant is the same function."
  foo-t)

(defn- tracked-diffs
  "The testsets of static_ir.jl:432 and 468, for update or regenerate (op)
   with nothing constrained or selected (x). The counter of a run with a
   (unknown, but identical) is Gen.jl's under nojuliacache, and the
   untracked variant's retdiff there assumes untracked diffs: not ported
   (D10)."
  [op x]
  (let [tr (gfi/simulate foo-t [1] (rng/key 7))]
    (testing "no change: bar is not run"
      (check (revise op tr [1] [diff/no-change] x) {:runs 0 :retdiff :no-change}))
    (testing "a unknown: bar runs, and its no-change retdiff is foo's"
      (check (revise op tr [1] [diff/unknown-change] x) {:retdiff :no-change})))
  (let [tr (gfi/simulate foo-without-tracked-diffs [1] (rng/key 7))]
    (testing "the untracked variant, no change: bar is not run"
      (check (revise op tr [1] [diff/no-change] x) {:runs 0 :retdiff :no-change}))))

;; static_ir.jl:432
(deftest update-with-tracked-diffs (tracked-diffs :update cm/EMPTY))

;; static_ir.jl:468
(deftest regenerate-with-tracked-diffs (tracked-diffs :regenerate sel/none))

(def foo-c
  "static_ir.jl:505: x ~ normal(a + b, 1), the sum counted."
  (gen ^:static [a b] (let [u (counted (+ a b)) x (trace :x (d/normal u 1))] x)))

;; static_ir.jl:521: the sum runs in simulate, and again only when a or b
;; may have changed
(deftest cached-julia-nodes
  (reset! runs 0)
  (let [tr (gfi/simulate foo-c [1 2] (rng/key 8))
        x1 (cm/choicemap :x 1.0)]
    (is (= 1 @runs) "simulate")
    (doseq [[op args argdiffs x n] [[:update [1 2] unchanged x1 0]
                                    [:update [1 2] unchanged x1 0]
                                    [:regenerate [1 2] unchanged (sel/select :x) 0]
                                    [:update [1 3] b-changed x1 1]
                                    [:update [1 3] b-changed x1 1]
                                    [:regenerate [1 3] b-changed (sel/select :x) 1]]]
      (is (= n (:runs (revise op tr args argdiffs x))) (str op " to " args)))))

;; static_ir.jl:552: mh with a selection on a static model ran into a
;; crash (Gen.jl's issue 168)
(deftest gen-issue-168
  (let [model (gen ^:static [var] (let [mean (trace :mean (d/normal 0 1)) T (trace :T (d/normal mean var))] T))
        tr (:trace (gfi/generate model [1] cm/EMPTY (rng/key 9)))]
    (is (trace? (:trace (mh/mh tr (sel/select :mean) (rng/key 10)))))))

;; Not ported
;; - static_ir.jl:25, 49 and 63, the IR's text/plain show (Julia's).
;; - D10, as the namespace doc says: update's and regenerate's counts
;;   for :z => :v (196, 272), and the tracked-diffs testsets' counts for a
;;   unknown (447, 463, 483, 499) and untracked retdiffs (464, 500).
;; - D7: get_param_grad before and after zero_param_grad! (340, 378).
