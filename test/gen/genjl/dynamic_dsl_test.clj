(ns gen.genjl.dynamic-dsl-test
  "Gen.jl 0.4.8's test/dsl/dynamic_dsl.jl: the dynamic modeling language's
   untraced calls, simulate, update, regenerate, gradients, parameter
   steps, multi-component addresses, project and trace lookups
   (docs/research/genjl-test-suite.md §2).

   As mapped there: Gen.jl's :x => 1 is the path [:x 0] (0-based,
   DESIGN.md §9), trace[] is the trace's :retval, retdiffs are gen.clj's
   more precise ones (D1), checked with diff/consistent?, and one
   parameter store serves a model and its callees, with argument gradients
   from choice-gradients' :argnums (D7)."
  (:refer-clojure :exclude [+ - *])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - *]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.genjl.finite-diff :refer [finite-diff grad-close? logpdf-grad]]
            [gen.genjl.support :refer [approx? has-value? logpdf shallow]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.learning :as learn]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

;; ---------------------------------------------------------------------------
;; Calls
;; ---------------------------------------------------------------------------

(deftest julia-call-syntax
  ;; dynamic_dsl.jl:8. bar() and oneliner(5.0) take default arguments,
  ;; which gen.clj does not have (G3, n/a: DESIGN.md §9); baz's x::Int is a
  ;; primitive hint, and outer's addressless @trace an addressless splice
  (let [foo (gen [] 1)
        baz (gen [^long x] x)
        inner (gen [] 5.0)
        outer (gen [] (splice inner))]
    (is (= 1 (foo)))
    (is (= 5 (baz 5)))
    (is (= 5.0 (outer)))))

(deftest simulate-a-bernoulli
  ;; dynamic_dsl.jl:73
  (let [foo (gen [p] (trace :x (d/bernoulli p)))
        p 0.4
        tr (gfi/simulate foo [p] (rng/key 1))]
    (is (= (ts/host (tr :x)) (ts/host (:retval tr))))
    (is (= [p] (:args tr)))
    (is (ts/close? (:score tr) (if (ts/host (tr :x)) (array/log p) (array/log (- 1 p)))))))

;; ---------------------------------------------------------------------------
;; Update
;; ---------------------------------------------------------------------------

(deftest update-switches-branch
  ;; dynamic_dsl.jl:86, the branch switch (lines 87-151)
  (let [bar (gen [] (trace :a (d/normal 0 1)))
        baz (gen [] (trace :b (d/normal 0 1)))
        foo (gen []
              (if (trace :branch (d/bernoulli 0.4))
                (do (trace :x (d/normal 0 1)) (splice :u bar))
                (do (trace :y (d/normal 0 1)) (splice :v baz))))
        ;; a trace which follows the first branch, forced to the second
        tr (:trace (gfi/generate foo [] (cm/choicemap :branch true) (rng/key 1)))
        [x a] [(tr :x) (tr [:u :a])]
        [y b] [1.123 -2.1]
        {new-tr :trace w :weight rd :retdiff discard :discard}
        (gfi/update tr [] [] (cm/choicemap :branch false :y y [:v :b] b) (rng/key 2))
        choices (:choices new-tr)]
    (testing "discard"
      (is (true? (ts/host (cm/get-choice discard :branch))))
      (is (= (ts/host x) (ts/host (cm/get-choice discard :x))))
      (is (= (ts/host a) (ts/host (cm/get-choice discard [:u :a]))))
      (is (= {:values 2 :submaps 1} (shallow discard))))
    (testing "new trace"
      (is (false? (ts/host (cm/get-choice choices :branch))))
      (is (= y (ts/host (cm/get-choice choices :y))))
      (is (= b (ts/host (cm/get-choice choices [:v :b]))))
      (is (= {:values 2 :submaps 1} (shallow choices))))
    (testing "score and weight"
      (let [prev-score (+ (logpdf d/bernoulli true 0.4) (logpdf d/normal x 0 1) (logpdf d/normal a 0 1))
            new-score (+ (logpdf d/bernoulli false 0.4) (logpdf d/normal y 0 1) (logpdf d/normal b 0 1))]
        (is (approx? new-score (:score new-tr)))
        (is (approx? (- new-score prev-score) w))))
    (testing "retdiff: the branch returns another value"
      (is (diff/unknown-change? rd))
      (is (diff/consistent? (:retval tr) (:retval new-tr) rd)))))

(deftest update-a-loop
  ;; dynamic_dsl.jl:86, the loop (lines 153-183): addresses under :data are
  ;; visited, but nothing there is discarded. The body returns nil both
  ;; times, so gen.clj's retdiff is no-change where Gen.jl's is
  ;; UnknownChange (D1)
  (let [loopy (gen []
                (let [a (trace :a (d/normal 0 1))]
                  (doseq [i (range 5)]
                    (trace [:data i] (d/normal a 1)))))
        constraints (reduce #(cm/set-choice %1 [:data %2] 0.0) (cm/choicemap :a 0.0) (range 5))
        tr (:trace (gfi/generate loopy [] constraints (rng/key 1)))
        {new-tr :trace w :weight rd :retdiff discard :discard}
        (gfi/update tr [] [] (cm/choicemap :a 1.0) (rng/key 2))
        prev-score (* 6 (logpdf d/normal 0.0 0 1))
        new-score (+ (logpdf d/normal 1.0 0 1) (* 5 (logpdf d/normal 0.0 1 1)))]
    (is (= 0.0 (ts/host (cm/get-choice discard :a))))
    (is (approx? new-score (:score new-tr)))
    (is (approx? (- new-score prev-score) w))
    (is (diff/no-change? rd))
    (is (diff/consistent? (:retval tr) (:retval new-tr) rd))))

(deftest update-discards-a-shrinking-loop
  ;; dynamic_dsl.jl:86, the hierarchical discard (lines 185-196): k from 3
  ;; to 1 drops Gen.jl's :value => 2 and 3, here [:value 1] and [:value 2]
  (let [hierarchical-update (gen []
                              (let [k (trace :k (d/poisson 5))]
                                (doseq [i (range k)]
                                  (trace [:value i] (d/uniform 0 1)))))
        tr (:trace (gfi/generate hierarchical-update [] (cm/choicemap :k 3) (rng/key 1)))
        {discard :discard} (gfi/update tr (cm/choicemap :k 1) (rng/key 2))]
    (is (has-value? discard [:value 1]))
    (is (has-value? discard [:value 2]))))

;; ---------------------------------------------------------------------------
;; Regenerate
;; ---------------------------------------------------------------------------

(deftest regenerate-the-branch
  ;; dynamic_dsl.jl:199. Ten regenerations of :branch with a changed mu,
  ;; Gen.jl's rand() fixed here; the keys give both a stay and a switch
  ;; (Gen.jl's "likely"). When the branch stays its two choices are kept,
  ;; so the return value is the previous one and gen.clj's retdiff is
  ;; no-change, where Gen.jl's is UnknownChange (D1)
  (let [bar (gen [mu] (trace :a (d/normal mu 1)))
        baz (gen [mu] (trace :b (d/normal mu 1)))
        foo (gen [mu]
              (if (trace :branch (d/bernoulli 0.4))
                (do (trace :x (d/normal mu 1)) (splice :u bar mu))
                (do (trace :y (d/normal mu 1)) (splice :v baz mu))))
        tr (:trace (gfi/generate foo [0.123] (cm/choicemap :branch true) (rng/key 1)))
        mus [0.912 0.245 0.637 0.081 0.558 0.734 0.329 0.476 0.893 0.164]
        steps (rest (reductions (fn [{prev :trace} [mu k]]
                                  (assoc (gfi/regenerate prev [mu] [diff/unknown-change] (sel/select :branch) (rng/key k))
                                         :prev prev))
                                {:trace tr}
                                (map vector mus (range 2 12))))
        stays (map #(= (ts/host ((:prev %) :branch)) (ts/host ((:trace %) :branch))) steps)]
    (is (= #{true false} (set stays)))
    (doseq [[{tr :trace prev :prev w :weight rd :retdiff} stay?] (map vector steps stays)
            :let [[prev-mu] (:args prev)
                  [mu] (:args tr)
                  branch (ts/host (tr :branch))
                  ;; the branch's choice and its call's
                  [here call sub] (if branch [:x :u :a] [:y :v :b])
                  density (fn [m] (+ (logpdf d/normal (tr here) m 1) (logpdf d/normal (tr [call sub]) m 1)))
                  choices (:choices tr)]]
      (testing (str "mu " mu)
        (is (approx? (+ (density mu) (logpdf d/bernoulli branch 0.4)) (:score tr)))
        (is (has-value? choices here))
        (is (not (cm/empty? (cm/get-submap choices call))))
        (is (= {:values 2 :submaps 1} (shallow choices)))
        (is (approx? (if stay? (- (density mu) (density prev-mu)) 0.0) w))
        (is (if stay? (diff/no-change? rd) (diff/unknown-change? rd)))
        (is (diff/consistent? (:retval prev) (:retval tr) rd))))))

;; ---------------------------------------------------------------------------
;; Gradients and parameter steps
;; ---------------------------------------------------------------------------

(def ^:private grad-bar
  (gen [mu-z]
    (let [z (trace :z (d/normal (+ mu-z (param :theta1)) 1))]
      (+ z mu-z))))

(def ^:private grad-foo
  ;; one store for foo and the bar it calls (D7)
  (dyn/with-params
    (gen [mu-a]
      (let [a (trace :a (d/normal mu-a 1))
            b (trace :b (d/normal a 1))
            c (* a b (splice :bar grad-bar a))]
        (+ (trace :out (d/normal c 1)) (* (param :theta2) 3))))
    {:theta1 0.0 :theta2 0.0}))

(defn- grad-objective
  "dynamic_dsl.jl:314's f, choice-gradients' objective: grad-foo's log
   density at its choices plus its return value (out, the parameters
   being 0) times the return value's gradient 2."
  [mu-a a b z out]
  (let [mu-z a
        c (* a b (+ z mu-z))]
    (+ (logpdf d/normal z mu-z 1) (logpdf d/normal a mu-a 1) (logpdf d/normal b a 1)
       (logpdf d/normal out c 1) (* 2 out))))

(deftest choice-gradients-and-param-gradients
  ;; dynamic_dsl.jl:293. Gen.jl's accumulate_param_gradients! also returns
  ;; mu_a's gradient, here choice-gradients' :argnums of no choices (D7).
  ;; The central differences are within 2.6e-9 of the exact gradients
  ;; (relative), at least 5 times inside isapprox's 1.5e-8 on float64
  (let [at [1.0 2.0 3.0 4.0 5.0]
        [mu-a a b z out] at
        fd (fn [i] (finite-diff grad-objective at i))
        tr (:trace (gfi/generate grad-foo [mu-a] (cm/choicemap :a a :b b :out out [:bar :z] z) (rng/key 1)))
        retval-grad 2.0
        {:keys [values gradients arg-grads]}
        (grad/choice-gradients tr (sel/select [:bar :z] :a :out) retval-grad {:argnums [0]})]
    (testing "choice-gradients: the argument's gradient"
      (is (grad-close? (first arg-grads) (fd 0))))
    (testing "choice-gradients: the values"
      (is (= a (ts/host (cm/get-choice values :a))))
      (is (= out (ts/host (cm/get-choice values :out))))
      (is (= z (ts/host (cm/get-choice values [:bar :z]))))
      (is (not (has-value? values :b)))
      (is (= {:values 2 :submaps 1} (shallow values))))
    (testing "choice-gradients: the gradients"
      (is (= {:values 2 :submaps 1} (shallow gradients)))
      (is (not (has-value? gradients :b)))
      (is (grad-close? (cm/get-choice gradients [:bar :z]) (fd 3)))
      (is (grad-close? (cm/get-choice gradients :out) (fd 4))))
    (testing "param-gradients"
      (let [{[mu-a-grad] :arg-grads} (grad/choice-gradients tr sel/none retval-grad {:argnums [0]})
            {:keys [theta1 theta2]} (grad/param-gradients tr retval-grad)]
        (is (grad-close? mu-a-grad (fd 0)))
        ;; Gen.jl's logpdf_grad(normal, z, a, 1)[2], the gradient in the mean
        (is (approx? theta1 (nth (logpdf-grad d/normal z a 1) 1)))
        (is (approx? theta2 (* 3 2)))))))

(deftest backprop-params-with-splice
  ;; dynamic_dsl.jl:381: a parameter read by an addressless callee
  (let [baz (gen [] (param :theta))
        foo (dyn/with-params (gen [] (splice baz)) {:theta 0.0})
        tr (:trace (gfi/generate foo [] nil (rng/key 1)))
        retval-grad 2.0]
    (is (approx? (:theta (grad/param-gradients tr retval-grad)) retval-grad))))

(defn- theta-after-one-step
  "theta after one step of optimizer from 0, along the gradient of a body
   that returns it, with the return value's gradient 1."
  [optimizer]
  (let [store {:theta 0.0}
        foo (dyn/with-params (gen [] (param :theta)) store)
        tr (:trace (gfi/generate foo [] nil (rng/key 1)))
        [store'] (learn/step optimizer (learn/init optimizer store) store (grad/param-gradients tr 1.0))]
    (:theta store')))

;; Gen.jl's second assertion in each, get_param_grad after apply_update!, is
;; the accumulator's reset: n/a, since gradients are values (D7)

(deftest gradient-descent-with-fixed-step-size
  ;; dynamic_dsl.jl:400
  (is (approx? (theta-after-one-step (learn/sgd {:lr 0.001})) 0.001)))

(deftest gradient-descent-with-shrinking-step-size
  ;; dynamic_dsl.jl:415
  (is (approx? (theta-after-one-step (learn/gradient-descent {:step-size-init 0.001 :step-size-beta 1000}))
               0.001)))

;; ---------------------------------------------------------------------------
;; Addresses, project and lookups
;; ---------------------------------------------------------------------------

(deftest multi-component-addresses
  ;; dynamic_dsl.jl:430. :x => 1, 2, 3 are [:x 0], [:x 1], [:x 2]
  (let [bar (gen [] (trace :z (d/normal 0 1)))
        foo (gen []
              (trace :y (d/normal 0 1))
              (trace [:x 0] (d/normal 0 1))
              (trace [:x 1] (d/normal 0 1))
              (splice [:x 2] bar))
        tr (:trace (gfi/generate foo [] (cm/choicemap [:x 0] 1.0 [:x 1] 2.0 [:x 2 :z] 3.0) (rng/key 1)))
        choices (:choices tr)
        submap (cm/get-submap choices :x)]
    (testing "the trace"
      (is (= 1.0 (ts/host (tr [:x 0]))))
      (is (= 2.0 (ts/host (tr [:x 1]))))
      (is (= 3.0 (ts/host (tr [:x 2 :z])))))
    (testing "its choices"
      (is (= 1.0 (ts/host (cm/get-choice choices [:x 0]))))
      (is (= 2.0 (ts/host (cm/get-choice choices [:x 1]))))
      (is (= 3.0 (ts/host (cm/get-choice choices [:x 2 :z]))))
      (is (= {:values 1 :submaps 1} (shallow choices))))
    (testing "the submap at :x"
      (is (= 1.0 (ts/host (cm/get-choice submap 0))))
      (is (= 2.0 (ts/host (cm/get-choice submap 1))))
      (is (= 3.0 (ts/host (cm/get-choice submap [2 :z]))))
      (is (= {:values 2 :submaps 1} (shallow submap)))
      (is (= 3.0 (ts/host (cm/get-choice (cm/get-submap submap 2) :z)))))))

(deftest project-test
  ;; dynamic_dsl.jl:465
  (let [bar (gen [] (trace :x (d/normal 0 1)))
        foo (gen [] (trace :y (d/normal 0 2)) (splice :z bar))
        tr (gfi/simulate foo [] (rng/key 1))
        [x y] [(tr [:z :x]) (tr :y)]]
    (is (approx? (gfi/project tr (sel/select)) 0.0))
    (is (approx? (gfi/project tr (sel/select :y)) (logpdf d/normal y 0 2)))
    (is (approx? (gfi/project tr (sel/select [:z :x])) (logpdf d/normal x 0 1)))
    (is (approx? (gfi/project tr (sel/select [:z :x] :y)) (+ (logpdf d/normal x 0 1) (logpdf d/normal y 0 2))))))

(deftest getindex-trace
  ;; dynamic_dsl.jl:487: choices and calls' return values by address,
  ;; trace[], and loud lookups where nothing is (D4)
  (let [bar (gen [r] (trace :a (d/normal 0 1)) r)
        foo (gen []
              (splice :x bar 1)
              (splice [:y :z] bar 2)
              (trace :u (d/normal 0 1))
              (trace [:v :w] (d/normal 0 1))
              7)
        tr (:trace (gfi/generate foo [] (cm/choicemap :u 1.1 [:v :w] 1.2 [:x :a] 1.3 [:y :z :a] 1.4) (rng/key 1)))]
    (testing "random choices"
      (is (= 1.1 (ts/host (tr :u))))
      (is (= 1.2 (ts/host (tr [:v :w]))))
      (is (= 1.3 (ts/host (tr [:x :a]))))
      (is (= 1.4 (ts/host (tr [:y :z :a])))))
    (testing "auxiliary state: a call's return value"
      (is (= 1 (tr :x)))
      (is (= 2 (tr [:y :z]))))
    (testing "return value"
      (is (= 7 (:retval tr))))
    (testing "an address that does not exist"
      (doseq [addr [:absent [:absent :x]]]
        (is (= :absent (:reason (ts/thrown-data (tr addr)))) (str addr))))))

;; Not ported (n/a, genjl-test-suite.md §2):
;; - dynamic_dsl.jl:20 and :30, bar() and oneliner(5.0): default arguments
;;   (G3; DESIGN.md §9, no default positional arguments).
;; - dynamic_dsl.jl:38, get_return_type: Julia's return-type annotations.
;; - dynamic_dsl.jl:49, AddressVisitor and all_visited: internal; their
;;   behavior is generate's loud unvisited check.
;; - dynamic_dsl.jl:412 and :427, get_param_grad after apply_update!: the
;;   accumulator's reset (D7).
;; - dynamic_dsl.jl:535, @doc of a @gen function: Julia's docstrings.
