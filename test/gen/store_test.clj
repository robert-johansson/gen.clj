(ns gen.store-test
  "The parameter store and when a generative function may keep its
   previous trace (gen.dynamic/kept?, same-gen-fn?): a store that a choice
   gives a callee reaches every kind of callee (dynamic, combinators, the
   internal and exact proposal overrides, estimated densities) in update
   and regenerate; a deterministic or estimated callee picked by a choice
   runs as the new one; a kernel a body builds again is the same kernel
   unless it closes over another value; combinators under an unchanged
   store skip their unchanged calls; a batched store value runs batched.
   Each weight is checked against the assessed densities, each trace's
   score against its own."
  (:refer-clojure :exclude [+ - *])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - *]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.custom :as custom]
            [gen.dist :as d]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.enumerative :as enum]
            [gen.inference.proposal :as prop]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(defn- assessed [model tr] (ts/number (:weight (gfi/assess model (:args tr) (:choices tr) (rng/key 7)))))

(defn- consistent? [model tr] (ts/close? (:score tr) (assessed model tr)))

;; ---------------------------------------------------------------------------
;; A store a choice gives the callee (review 4, R2)
;; ---------------------------------------------------------------------------

(def ^:private inner (gen [z] (trace :x (d/normal (+ z (param :w)) 1))))
(def ^:private q (gen [cs _z] (when-not (cm/has-value? (cm/get-submap cs :x)) (trace :x (d/normal 0 2)))))
(def ^:private inner-b (gen [_z] (let [b (trace :b (d/bernoulli 0.5))] (trace :x (d/normal (+ (param :w) (if b 1.0 0.0)) 1)))))
(def ^:private kernel (gen [x] (trace :y (d/normal (+ x (param :w)) 1))))
(def ^:private kq (gen [cs x] (when-not (cm/has-value? (cm/get-submap cs :y)) (trace :y (d/normal x 2)))))
(def ^:private inner0 (gen [] (trace :x (d/normal (param :w) 1))))
(def ^:private q0 (gen [cs] (when-not (cm/has-value? (cm/get-submap cs :x)) (trace :x (d/normal 0 2)))))

(def ^:private estimated
  "An estimator that assesses a model reading the store."
  (custom/estimated {:addresses [:x]
                     :estimate (fn [[z] cs k] (:weight (gfi/assess inner [z] cs k)))
                     :simulate (fn [[z] k] (let [tr (gfi/simulate inner [z] k)] {:choices (:choices tr) :log-estimate (:score tr)}))}))

(defn- holder [callee & args] (gen [] (splice :p callee args)))

(defn- given-a-store [callee]
  (gen [] (let [a (trace :a (d/uniform 0 1))] (splice :c (dyn/with-params callee {:w (* 10.0 a)})))))

(def ^:private store-cases
  {"a dynamic callee" (given-a-store (gen [] (splice :p inner 0.0)))
   "an internal proposal" (given-a-store (gen [] (splice :p (prop/with-internal-proposal inner q) 0.0)))
   "an internal proposal of no arguments" (given-a-store (gen [] (splice :p (prop/with-internal-proposal inner0 q0))))
   "an exact proposal" (given-a-store (gen [] (splice :p (enum/with-exact-proposal inner-b [:b]) 0.0)))
   "a Map of an internal-proposal kernel" (given-a-store (gen [] (splice :m (comb/map (prop/with-internal-proposal kernel kq)) [0.0 0.0])))
   "an estimated density" (given-a-store (gen [] (splice :p estimated 0.0)))})

(deftest a-store-from-a-choice-reaches-every-callee
  ;; :a's prior is uniform on (0, 1), so selection MH's weight is the change
  ;; in the joint density
  (doseq [[label model] store-cases]
    (testing label
      (let [tr (gfi/simulate model [] (rng/key 5))
            {u :trace uw :weight} (gfi/update tr (cm/choicemap :a 0.9) (rng/key 6))
            {r :trace rw :weight} (gfi/regenerate tr (sel/select :a) (rng/key 8))]
        (is (consistent? model u) "update")
        (is (ts/close? (- (assessed model u) (ts/number (:score tr))) uw) "update's weight")
        (is (consistent? model r) "regenerate")
        (is (ts/close? (- (assessed model r) (ts/number (:score tr))) rw) "regenerate's weight")))))

(deftest a-changed-call-is-weighed-against-its-old-store
  ;; the store at a call changes (the whole model is given another): the
  ;; call runs again, and the old trace is weighed under the store it ran
  ;; under, which an exact proposal's projection needs (it reruns its model)
  (let [exact (enum/with-exact-proposal inner-b [:b])
        model (gen [] (splice :p exact 0.0))
        before (dyn/with-params model {:w 1.0})
        after (dyn/with-params model {:w 5.0})
        tr (gfi/simulate before [] (rng/key 16))
        {t :trace w :weight} (gfi/regenerate (assoc tr :gen-fn after) sel/none (rng/key 17))]
    (is (consistent? after t))
    ;; nothing selected: the projection ratio on every choice (Eq. 4.1),
    ;; each projection under its own model's store
    (is (ts/close? (- (gfi/project t sel/all) (gfi/project tr sel/all)) w))))

;; ---------------------------------------------------------------------------
;; Another generative function at the address (review 4, R4)
;; ---------------------------------------------------------------------------

(def ^:private plus (custom/deterministic {:apply (fn [[x]] (+ x 10.0))}))
(def ^:private minus (custom/deterministic {:apply (fn [[x]] (- x 10.0))}))

(defn- normal-estimator [mu]
  (custom/estimated {:addresses [:y]
                     :estimate (fn [_ cs _k] (:weight (gfi/assess (d/normal mu 1) [] (cm/get-submap cs :y) (rng/key 0))))
                     :simulate (fn [_ k] (let [tr (gfi/simulate (d/normal mu 1) [] k)]
                                           {:choices (cm/choicemap :y (:retval tr)) :log-estimate (:score tr)}))}))

(def ^:private est-a (normal-estimator 0.0))
(def ^:private est-b (normal-estimator 5.0))

(deftest a-callee-picked-by-a-choice-runs-as-the-new-one
  (doseq [[label model] {"a deterministic callee" (gen [] (let [b (trace :b (d/bernoulli 0.5))
                                                                v (splice :d (if b plus minus) 0.0)]
                                                            (trace :z (d/normal v 1))))
                         "a Map's deterministic kernel" (gen [] (let [b (trace :b (d/bernoulli 0.5))
                                                                      vs (splice :m (comb/map (if b plus minus)) [0.0 0.0])]
                                                                  (trace :z (d/normal (nth vs 0) 1))))
                         "an estimated callee" (gen [] (let [b (trace :b (d/bernoulli 0.5))]
                                                         (splice :e (if b est-a est-b))))}]
    (testing label
      (let [tr (gfi/simulate model [] (rng/key 3))
            {u :trace w :weight} (gfi/update tr (cm/choicemap :b (not (ts/host (tr :b)))) (rng/key 4))]
        (is (consistent? model u))
        (is (ts/close? (- (assessed model u) (ts/number (:score tr))) w))))))

;; ---------------------------------------------------------------------------
;; A kernel built again (review 4, R9)
;; ---------------------------------------------------------------------------

(deftest a-kernel-built-again-is-the-same-kernel
  ;; (the data is one vector, as data passed in is: a vector the body built
  ;; again would be an unknown change, and rerun every element rightly)
  (let [runs (atom 0)
        xs (vec (repeat 8 0.0))
        counted (fn [model tr cs] (reset! runs 0) [(gfi/update tr cs (rng/key 11)) @runs])]
    (testing "an inline kernel closing over nothing reruns only the element constrained"
      (let [model (gen [] (splice :m (comb/map (gen [x] (swap! runs inc) (trace :y (d/normal x 1)))) xs))
            tr (gfi/simulate model [] (rng/key 10))]
        (is (= 1 (second (counted model tr (cm/choicemap [:m 3 :y] 0.5)))))))
    (testing "and one given an equal store"
      (let [k (gen [x] (swap! runs inc) (trace :y (d/normal (+ x (param :w)) 1)))
            model (gen [] (splice :m (comb/map (dyn/with-params k {:w 1.0})) xs))
            tr (gfi/simulate model [] (rng/key 10))]
        (is (= 1 (second (counted model tr (cm/choicemap [:m 3 :y] 0.5)))))))
    (testing "but one closing over a changed choice reruns, and weighs right"
      (let [model (gen [] (let [a (trace :a (d/normal 0 1))]
                            (splice :m (comb/map (gen [x] (swap! runs inc) (trace :y (d/normal (+ x (* 10.0 a)) 1)))) [0.0 0.0])))
            tr (gfi/simulate model [] (rng/key 12))
            [{u :trace w :weight} n] (counted model tr (cm/choicemap :a 1.5))]
        (is (= 2 n))
        (is (consistent? model u))
        (is (ts/close? (- (assessed model u) (ts/number (:score tr))) w))))))

;; ---------------------------------------------------------------------------
;; Combinators under a store (review 4, C5)
;; ---------------------------------------------------------------------------

(deftest combinators-under-an-unchanged-store-skip-their-calls
  ;; an identity update runs no kernel (a store not recorded would rerun all)
  (let [runs (atom 0)
        step (gen [_t x] (swap! runs inc) (trace :x (d/normal (* (param :w) x) 1)))
        production (gen [x] (swap! runs inc)
                     (let [v (trace :v (d/normal (* (param :w) x) 1))]
                       {:value v :children (if (clojure.core/< x 2) [(clojure.core/+ x 1)] [])}))
        aggregation (gen [v ws] (+ v (reduce + 0.0 ws)))]
    (doseq [[label model] {"Map" (gen [] (splice :m (comb/map (gen [x] (swap! runs inc) (trace :y (d/normal (+ x (param :w)) 1)))) [0.0 0.0 0.0]))
                           "Unfold" (gen [] (splice :u (comb/unfold step) 3 1.0))
                           "Recurse" (gen [] (splice :r (comb/recurse production aggregation 1) 1))}]
      (testing label
        (let [gf (dyn/with-params model {:w 0.5})
              tr (gfi/simulate gf [] (rng/key 13))]
          (reset! runs 0)
          (gfi/update tr cm/EMPTY (rng/key 14))
          (is (= 0 @runs)))))))

;; ---------------------------------------------------------------------------
;; A batched store value (review 4, R13)
;; ---------------------------------------------------------------------------

(deftest a-batched-store-value-runs-batched
  (let [model (given-a-store (gen [] (splice :p inner 0.0)))]
    (is (nil? (batched/fallback-reason (batched/simulate model [] 6 (rng/key 15)))))))
