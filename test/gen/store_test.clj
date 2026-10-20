(ns gen.store-test
  "The parameter store, and what an update or regenerate may keep of a
   trace (gfi/revise, gfi/same-gen-fn?, dyn/run-call): a store that a
   choice gives a callee reaches every kind of callee (dynamic,
   combinators, the internal and exact proposal overrides, estimated
   densities) in update and regenerate; a deterministic or estimated
   callee picked by a choice runs as the new one; a trace given another
   gen-fn at the top level is revised, of every kind; a store that changes
   a callee's structure drops the old choices it leaves; a generative
   function a body builds again is the same one unless it closes over
   another value, so its calls stay incremental; combinators under an
   unchanged store skip their unchanged calls; a batched store value runs
   batched. Each weight is checked against the assessed densities (or the
   handler's rule), each trace's score against its own."
  (:refer-clojure :exclude [+ - * >])
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array :refer [+ - * >]]
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
            [gen.test-support :as ts]
            [gen.trace :as trace]))

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

;; ---------------------------------------------------------------------------
;; Truths for a revise (review 5)
;; ---------------------------------------------------------------------------

(defn- update-truth
  "log p(new) - log p(old) - log q(fresh), the fresh choices (neither old
   nor constrained) drawn from the prior: a model of forward-sampling kinds."
  [model tr u constraints]
  (let [old (set (cm/addresses (:choices tr)))
        given (set (cm/addresses constraints))
        fresh (remove #(or (old %) (given %)) (cm/addresses (:choices u)))]
    (- (assessed model u) (ts/number (:score tr)) (ts/number (gfi/project u (sel/from-paths fresh))))))

(defn- regenerate-truth
  "The handler's rule: the old unselected choices the new trace kept,
   rescored (a choice no longer visited cancels against the reverse move)."
  [tr r s]
  (let [kept (sel/from-paths (filter #(cm/has-value? (cm/get-submap (:choices r) %))
                                     (cm/addresses (sel/get-selected (:choices tr) (sel/complement s)))))]
    (- (ts/number (gfi/project r kept)) (ts/number (gfi/project tr kept)))))

(defn- discard-ok?
  "Is d every old choice the update did not keep: constrained, or gone?"
  [tr u constraints d]
  (= (set (cm/addresses d))
     (set (filter #(or (not (cm/empty? (cm/get-submap constraints %))) (not (cm/has-value? (cm/get-submap (:choices u) %))))
                  (cm/addresses (:choices tr))))))

;; ---------------------------------------------------------------------------
;; A trace given another gen-fn at the top level (review 5, R5-1)
;; ---------------------------------------------------------------------------

(def ^:private step (gen [_t s] (trace :y (d/normal (+ s (param :w)) 1))))
(def ^:private static-model (gen ^:static [] (let [x (trace :x (d/normal (param :w) 1))] x)))

(def ^:private top-swaps
  "[before after args]: a generative function under two stores."
  (let [ws (fn [f args] [(f {:w 1.0}) (f {:w 5.0}) args])]
    {"an internal proposal of a with-params model" (ws #(prop/with-internal-proposal (dyn/with-params inner %) q) [0.0])
     "an exact proposal of a with-params model" (ws #(enum/with-exact-proposal (dyn/with-params inner-b %) [:b]) [0.0])
     "a Map of exact proposals of a with-params model"
     (ws #(comb/map (enum/with-exact-proposal (dyn/with-params inner-b %) [:b])) [[0.0 1.0]])
     "a Map of a with-params kernel" (ws #(comb/map (dyn/with-params kernel %)) [[0.0 1.0]])
     "an Unfold of a with-params kernel" (ws #(comb/unfold (dyn/with-params step %)) [3 0.0])
     "a Switch of a with-params branch" (ws #(comb/switch (dyn/with-params inner %)) [0 0.0])
     "a static model" (ws #(dyn/with-params static-model %) [])
     "with-params outside an exact proposal" (ws #(dyn/with-params (gen [] (splice :p (enum/with-exact-proposal inner-b [:b]) 0.0)) %) [])}))

(deftest a-trace-given-another-gen-fn-is-revised
  ;; update (nothing constrained): assessed(new) - score(old); regenerate
  ;; (nothing selected): the projection ratio on every choice (Eq. 4.1),
  ;; each projection by its own gen-fn. The internal and exact proposals
  ;; kept the stale trace with weight 0 before (R5-1), a Map of exact
  ;; proposals too without the gen-fn check (mutation M12).
  (doseq [[label [before after args]] top-swaps]
    (testing label
      (let [tr (gfi/simulate before args (rng/key 16))
            swapped (assoc tr :gen-fn after)
            {u :trace uw :weight} (gfi/update swapped cm/EMPTY (rng/key 18))
            {r :trace rw :weight} (gfi/regenerate swapped sel/none (rng/key 17))]
        (is (not (ts/close? 0.0 uw)) "the stores differ")
        (is (consistent? after u))
        (is (ts/close? (- (assessed after u) (ts/number (:score tr))) uw))
        (is (consistent? after r))
        (is (ts/close? (- (ts/number (gfi/project r sel/all)) (ts/number (gfi/project tr sel/all))) rw))))))

(deftest a-custom-trace-given-another-gen-fn-is-revised
  (testing "deterministic"
    (let [tr (gfi/simulate plus [1.0] (rng/key 1))]
      (is (= (minus 1.0) (:retval (:trace (gfi/update (assoc tr :gen-fn minus) cm/EMPTY (rng/key 2))))))))
  (testing "estimated"
    (let [tr (gfi/simulate est-a [] (rng/key 1))
          {w :weight} (gfi/update (assoc tr :gen-fn est-b) cm/EMPTY (rng/key 2))]
      (is (ts/close? (- (ts/number (:weight (gfi/assess est-b [] (:choices tr) (rng/key 0)))) (ts/number (:score tr))) w)))))

(deftest a-trace-made-under-middleware-is-rescored-without-it
  ;; a static body skips nothing for a trace another gen-fn made: the same
  ;; graph under other middleware (here a distribution swapped in simulate)
  (let [shifted (dyn/with-transitions static-model
                  {:simulate (fn [state p _dist k] ((dyn/site-transitions :simulate) state p (d/normal 5 1) k))})
        plain (dyn/with-params static-model {:w 0.0})
        tr (gfi/simulate (dyn/with-params shifted {:w 0.0}) [] (rng/key 1))
        {u :trace w :weight} (gfi/update (assoc tr :gen-fn plain) cm/EMPTY (rng/key 2))]
    (is (consistent? plain u))
    (is (ts/close? (- (assessed plain u) (ts/number (:score tr))) w))))

(def ^:private untraced-model
  "An untraced draw: a batched run falls back to particles."
  (gen [] (let [u (untraced (d/normal 0 1))] (trace :x (d/normal (+ (param :w) (* 0.0 u)) 1)))))

(deftest a-batched-trace-given-another-gen-fn-is-revised
  ;; batched, per-particle from the start, and batched then falling back
  ;; (review 6, R6-2: the last two ran the old gen-fn, or rebuilt the old
  ;; traces under the new one, with weight 0)
  (doseq [[label before after batched-update?]
          [["batched" (dyn/with-params static-model {:w 1.0}) (dyn/with-params static-model {:w 5.0}) true]
           ["per-particle" (dyn/with-params untraced-model {:w 1.0}) (dyn/with-params untraced-model {:w 5.0}) false]
           ["batched, then falling back" (dyn/with-params static-model {:w 1.0}) (dyn/with-params untraced-model {:w 5.0}) false]]]
    (testing label
      (let [ps (batched/simulate before [] 4 (rng/key 1))
            {ps2 :particles ws :weights} (batched/update (assoc ps :gen-fn after) [] [] cm/EMPTY (rng/key 2))]
        (is (= batched-update? (batched/batched? ps2)))
        (doseq [[t0 t1 w] (map vector (batched/traces ps) (batched/traces ps2) (ts/host ws))]
          (is (identical? after (:gen-fn t1)))
          (is (consistent? after t1))
          (is (ts/close? (- (assessed after t1) (ts/number (:score t0))) w)))))))

;; ---------------------------------------------------------------------------
;; A store that decides a callee's structure (review 5, R5-2, mutation M7b)
;; ---------------------------------------------------------------------------

(def ^:private forked (gen [] (if (> (param :w) 5.0) (trace :x (d/normal 0 1)) (trace :y (d/normal 0 1)))))
(def ^:private forked1 (gen [_z] (if (> (param :w) 5.0) (trace :x (d/normal 0 1)) (trace :y (d/normal 0 1)))))
(def ^:private forked-step (gen [_t s] (if (> (param :w) 5.0) (trace :x (d/normal s 1)) (trace :y (d/normal s 1)))))
(def ^:private forked-b (gen [] (let [b (trace :b (d/bernoulli 0.5))]
                                  (if (> (param :w) 5.0) (trace :x (d/normal (if b 1.0 0.0) 1)) (trace :y (d/normal 0 1))))))

(def ^:private structure-cases
  "Callees under a store of 10a: :a below 0.5 makes :y, above makes :x."
  {"a dynamic callee of no arguments" (given-a-store (gen [] (splice :p forked)))
   "an internal proposal of no arguments, def'd" (given-a-store (gen [] (splice :p (prop/with-internal-proposal forked (gen [_cs] nil)))))
   "a Map of a forked kernel" (given-a-store (gen [] (splice :p (comb/map forked1) [0.0 1.0 2.0])))
   "a Switch of a forked branch" (given-a-store (gen [] (splice :p (comb/switch forked1 forked1) 1 0.0)))
   "an Unfold of a forked step" (given-a-store (gen [] (splice :p (comb/unfold forked-step) 3 0.0)))})

(defn- moved? [tr] (boolean (some #(= :x (peek %)) (cm/addresses (:choices tr)))))

;; ---------------------------------------------------------------------------
;; Values a closure can tell apart (review 6, R6-1, R6-5, R6-7)
;; ---------------------------------------------------------------------------

(deftest same-value-tells-apart-what-code-could
  (let [same? gfi/same-value?]
    (testing "different: an array-map's order, sorted against hash, vector against list, metadata, -0.0, a class"
      (is (not (same? (array-map :p 0.0 :q 10.0) (array-map :q 10.0 :p 0.0))))
      (is (not (same? (sorted-map :a 1 :g 2) (hash-map :a 1 :g 2))))
      (is (not (same? (sorted-set 10 20) (hash-set 10 20))))
      (is (not (same? [1.0 2.0] (list 1.0 2.0))))
      (is (not (same? (with-meta [1.0] {:w 1.0}) (with-meta [1.0] {:w 100.0}))))
      (is (not (same? 0.0 -0.0)))
      (is (not (same? (float 1.5) 1.5)))
      (is (not (same? 1 1.0))))
    (testing "different: sorted collections of other comparators, which conj tells apart (review 7, R7-6)"
      (is (not (same? (sorted-set-by clojure.core/< 5) (sorted-set-by clojure.core/> 5))))
      (is (not (same? (sorted-set-by clojure.core/<) (sorted-set-by clojure.core/>))))
      (is (not (same? (sorted-map-by clojure.core/< 5 :a) (sorted-map-by clojure.core/> 5 :a)))))
    (testing "the same: equal values built again, NaN, a set of paths (an estimated density's addresses)"
      (is (same? (array-map :p 0.0 :q 10.0) (array-map :p 0.0 :q 10.0)))
      (is (same? {:w [1.0 2.0]} {:w (vector 1.0 (+ 1.0 1.0))}))
      (is (same? ##NaN ##NaN))
      (is (same? #{[:y] [:z 0]} (set [[:y] [:z 0]])))
      (is (same? (sorted-map :a 1 :b 2) (sorted-map :a 1 :b 2))))
    (testing "the same: maps of 300 entries built again from identical keys and values, which spend no budget (review 7, R7-2: an entry spent one)"
      (let [ks (mapv #(keyword (str "p" %)) (range 300))
            vs (mapv double (range 300))]
        (is (same? (zipmap ks vs) (zipmap ks vs)))))
    (testing "arrays: by identity, also as map keys (no hashing, which reads them on MLX)"
      (let [x (array/array [1.0 2.0])]
        (is (same? {x 1} {x 1}))
        (is (not (same? {x 1} {(array/array [1.0 2.0]) 1})))))))

(defn- in-order
  "An array-map whose order :b decides: a callee reading (first (vals m)) sees it."
  [b]
  (if b (array-map :p 0.0 :q 10.0) (array-map :q 10.0 :p 0.0)))

(deftest a-callee-over-a-value-only-equal-to-the-old-one-runs-again
  ;; (review 6, R6-1: = called these the same, and the callee kept a stale
  ;; trace with weight 0)
  (doseq [[label model]
          [["a deterministic callee over an array-map"
            (gen [] (let [b (trace :b (d/bernoulli 0.5)) m (in-order b)
                          v (splice :d (custom/deterministic {:apply (fn [[x]] (+ x (first (vals m))))}) 1.0)]
                      (trace :z (d/normal v 1))))]
           ["a static callee over an array-map"
            (gen [] (let [b (trace :b (d/bernoulli 0.5)) m (in-order b)]
                      (splice :c (gen ^:static [] (let [x (trace :x (d/normal (first (vals m)) 1))] x)))))]
           ["a Map kernel over an array-map"
            (gen [] (let [b (trace :b (d/bernoulli 0.5)) m (in-order b)]
                      (splice :m (comb/map (gen [i] (trace :y (d/normal (+ i (first (vals m))) 1)))) [0.0 1.0])))]
           ["a Map kernel over a vector or a list"
            (gen [] (let [b (trace :b (d/bernoulli 0.5)) xs (if b [0.0 10.0] (list 0.0 10.0))]
                      (splice :m (comb/map (gen [i] (trace :y (d/normal (+ i (first (conj xs 5.0))) 1)))) [0.0 1.0])))]
           ;; one element, so only the comparator differs (review 7, R7-6)
           ["a Map kernel over a sorted set of either order"
            (gen [] (let [b (trace :b (d/bernoulli 0.5)) s (sorted-set-by (if b clojure.core/< clojure.core/>) 5.0)]
                      (splice :m (comb/map (gen [i] (trace :y (d/normal (+ i (first (conj s 9.0))) 1)))) [0.0 1.0])))]
           ["a deterministic callee over a sorted set of either order"
            (gen [] (let [b (trace :b (d/bernoulli 0.5)) s (sorted-set-by (if b clojure.core/< clojure.core/>) 5.0)
                          v (splice :d (custom/deterministic {:apply (fn [[x]] (+ x (first (conj s 9.0))))}) 0.0)]
                      (trace :z (d/normal v 1))))]]]
    (testing label
      (let [tr (:trace (gfi/generate model [] (cm/choicemap :b true) (rng/key 1)))
            {u :trace w :weight} (gfi/update tr (cm/choicemap :b false) (rng/key 2))]
        (is (consistent? model u))
        (is (ts/close? (- (assessed model u) (ts/number (:score tr))) w)))))
  (testing "a top-level swap of a Map whose kernel closes over another order"
    (let [kernel-over (fn [m] (gen [i] (trace :y (d/normal (+ i (first (vals m))) 1))))
          before (comb/map (kernel-over (in-order true)))
          after (comb/map (kernel-over (in-order false)))
          tr (gfi/simulate before [[0.0 1.0]] (rng/key 3))
          {u :trace w :weight} (gfi/update (assoc tr :gen-fn after) cm/EMPTY (rng/key 4))]
      (is (consistent? after u))
      (is (ts/close? (- (assessed after u) (ts/number (:score tr))) w)))))

(deftest a-store-that-changes-a-callees-structure-drops-what-it-leaves
  (doseq [[label model] structure-cases]
    (testing label
      (let [tr (:trace (gfi/generate model [] (cm/choicemap :a 0.1) (rng/key 1)))
            cs (cm/choicemap :a 0.9)
            {u :trace w :weight dsc :discard} (gfi/update tr cs (rng/key 2))]
        (is (and (not (moved? tr)) (moved? u)) "the structure changed")
        (is (consistent? model u))
        (is (ts/close? (update-truth model tr u cs) w))
        (is (discard-ok? tr u cs dsc))
        (let [regenerated (for [k (range 4)] (:trace (gfi/regenerate tr (sel/select :a) (rng/key (+ 10 k)))))]
          (is (some moved? regenerated) "a key that changes the structure")
          (doseq [k (range 4)]
            (let [{r :trace rw :weight} (gfi/regenerate tr (sel/select :a) (rng/key (+ 10 k)))]
              (is (consistent? model r))
              (is (ts/close? (regenerate-truth tr r (sel/select :a)) rw)))))))))

(deftest an-exact-proposal-whose-structure-the-store-decides
  ;; def'd with no arguments, so no argdiff shows the change (R5-2: the
  ;; trace stayed stale, with weight 0)
  (let [model (given-a-store (gen [] (splice :p (enum/with-exact-proposal forked-b [:b]))))
        tr (:trace (gfi/generate model [] (cm/choicemap :a 0.1) (rng/key 1)))
        {u :trace} (gfi/update tr (cm/choicemap :a 0.9) (rng/key 2))]
    (is (and (not (moved? tr)) (moved? u)))
    (is (consistent? model u))))

(deftest an-estimators-own-unvisited-constraint-stays-loud
  ;; the estimator's model stops observing :o under the new store: its
  ;; error, not a choice of the estimated function to drop (R5-2)
  (let [obs (gen [] (when-not (> (param :w) 5.0) (trace :o (d/normal 0 1))))
        est (custom/estimated {:addresses [:o]
                               :estimate (fn [_ cs k] (:weight (gfi/assess obs [] cs k)))
                               :simulate (fn [_ k] (let [tr (gfi/simulate (gen [] (trace :o (d/normal 0 1))) [] k)]
                                                     {:choices (:choices tr) :log-estimate (:score tr)}))})
        model (given-a-store (gen [] (splice :p est)))
        tr (:trace (gfi/generate model [] (cm/choicemap :a 0.1) (rng/key 1)))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"the constraint at \[:o\] was not visited"
                          (gfi/update tr (cm/choicemap :a 0.9) (rng/key 2))))
    ;; regenerate: loud whenever the new :a moves the store over 5
    (is (some #(re-find #"the constraint at \[:o\] was not visited" %)
              (for [k (range 4)]
                (try (gfi/regenerate tr (sel/select :a) (rng/key (+ 10 k))) ""
                     (catch clojure.lang.ExceptionInfo e (ex-message e))))))))

(deftest a-structure-change-costs-each-call-one-more-run
  ;; each element drops what it leaves where it leaves it: 2n kernel runs,
  ;; not one revise of the whole Map per element (n(n+1)/2)
  (let [runs (atom 0)
        k (gen [_i] (swap! runs inc) (if (> (param :w) 5.0) (trace :x (d/normal 0 1)) (trace :y (d/normal 0 1))))
        model (given-a-store (gen [] (splice :p (comb/map k) (vec (range 20)))))
        tr (:trace (gfi/generate model [] (cm/choicemap :a 0.1) (rng/key 1)))]
    (reset! runs 0)
    (let [{u :trace} (gfi/update tr (cm/choicemap :a 0.9) (rng/key 2))]
      (is (moved? u))
      (is (= 40 @runs))))
  (testing "inside an internal proposal's model: one more run of the override, without the drops, so that q sees the final constraints (review 6, R6-4: n(n+1)/2 + n)"
    (let [runs (atom 0)
          k (gen [_i] (swap! runs inc) (if (> (param :w) 5.0) (trace :x (d/normal 0 1)) (trace :y (d/normal 0 1))))
          model (given-a-store (gen [] (splice :p (prop/with-internal-proposal (gen [] (splice :m (comb/map k) (vec (range 20))))
                                                                             (gen [_cs] nil)))))
          tr (:trace (gfi/generate model [] (cm/choicemap :a 0.1) (rng/key 1)))
          cs (cm/choicemap :a 0.9)]
      (reset! runs 0)
      (let [{u :trace w :weight} (gfi/update tr cs (rng/key 2))]
        (is (moved? u))
        (is (= 60 @runs))
        (is (consistent? model u))
        (is (ts/close? (update-truth model tr u cs) w)))))
  (testing "inside an exact proposal's model: its forward run drops and repeats once (2n + n), then it enumerates :b (2n): 5n, not quadratic"
    (let [runs (atom 0)
          k (gen [_i] (swap! runs inc) (if (> (param :w) 5.0) (trace :x (d/normal 0 1)) (trace :y (d/normal 0 1))))
          model (given-a-store (gen [] (splice :p (enum/with-exact-proposal
                                                     (gen [] (let [b (trace :b (d/bernoulli 0.5))] (splice :m (comb/map k) (vec (range 20))) b))
                                                     [:b]))))
          tr (:trace (gfi/generate model [] (cm/choicemap :a 0.1) (rng/key 1)))]
      (reset! runs 0)
      (let [{u :trace} (gfi/update tr (cm/choicemap :a 0.9) (rng/key 2))]
        (is (moved? u))
        (is (= 100 @runs))
        (is (consistent? model u))))))

(deftest a-constraint-the-revised-run-leaves-is-loud
  ;; a revise drops old choices, never constraints: in a dynamic callee's
  ;; own update, and in a generate from the old choices (a Map's, an
  ;; internal proposal's)
  (doseq [[label callee p] [["a dynamic callee" (gen [] (splice :p forked)) [:c :p :y]]
                            ["a Map" (gen [] (splice :p (comb/map forked1) [0.0 1.0])) [:c :p 1 :y]]
                            ["an internal proposal" (gen [] (splice :p (prop/with-internal-proposal forked (gen [_cs] nil)))) [:c :p :y]]]]
    (testing label
      (let [model (given-a-store callee)
            tr (:trace (gfi/generate model [] (cm/choicemap :a 0.1) (rng/key 1)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo
                              (re-pattern (java.util.regex.Pattern/quote (str "the constraint at " p " was not visited")))
                              (gfi/update tr (cm/choicemap :a 0.9 p 0.5) (rng/key 2)))))))
  (testing "a proposal that reads its constraints, which would visit the address once the constraint is dropped (review 6, R6-3)"
    ;; the model makes :x only when :b; q proposes :b from whether :x is given
    (let [model (gen [_i] (let [b (trace :b (d/bernoulli 0.5))] (when b (trace :x (d/normal 0 1)))))
          q (gen [cs _i] (when-not (cm/has-value? (cm/get-submap cs :b))
                           (trace :b (d/bernoulli (if (cm/has-value? (cm/get-submap cs :x)) 0.01 0.99)))))
          after (prop/with-internal-proposal model q)
          other (gen [_i] (trace :y (d/normal 0 1)))]
      (testing "at the top"
        (let [tr (gfi/simulate other [0] (rng/key 1))]
          (doseq [k (range 4)]
            (is (thrown-with-msg? clojure.lang.ExceptionInfo #"the constraint at \[:x\] was not visited"
                                  (gfi/update (assoc tr :gen-fn after) (cm/choicemap :x 1.0) (rng/key (+ 10 k))))))))
      (testing "inside a call, where the constraint is known by its path from the revise (each call extends the prefix)"
        (let [tr (gfi/simulate (comb/map other) [[0]] (rng/key 1))]
          (doseq [k (range 4)]
            (is (thrown-with-msg? clojure.lang.ExceptionInfo #"the constraint at \[0 :x\] was not visited"
                                  (gfi/update (assoc tr :gen-fn (comb/map after)) (cm/choicemap [0 :x] 1.0) (rng/key (+ 20 k))))))))
      (testing "two calls deep in a Map's kernel, where only the prefix of every call names it (review 7: a reversed or unextended prefix, G1 and G2, replaced it for 3 of 4 keys)"
        (let [two-deep (fn [g] (comb/map (gen [i] (splice :c (gen [] (splice :p g i))))))
              tr (gfi/simulate (two-deep other) [[0]] (rng/key 1))
              deep (two-deep after)]
          (doseq [k (range 4)]
            (is (thrown-with-msg? clojure.lang.ExceptionInfo #"the constraint at \[0 :c :p :x\] was not visited"
                                  (gfi/update (assoc tr :gen-fn deep) (cm/choicemap [0 :c :p :x] 1.0) (rng/key (+ 20 k))))))))))
  (testing "a proposal's own choice that the model's new structure leaves (review 7, R7-1: dropped as an old choice, its log q kept in the weight)"
    ;; q is written for w > 5, where sub makes :x; the store moves w to 1.
    ;; sub stops a revise that runs on (a drop that removes nothing)
    (let [runs (atom 0)
          sub (gen [] (when (< 50 (swap! runs inc)) (throw (ex-info "the revise runs on" {})))
                (if (> (param :w) 5.0) (trace :x (d/normal 0 1)) (trace :y (d/normal 0 1))))
          q (gen [cs] (when-not (cm/has-value? (cm/get-submap cs [:s :x])) (trace [:s :x] (d/normal 3 1))))
          model (given-a-store (gen [] (splice :p (prop/with-internal-proposal (gen [] (splice :s sub)) q))))]
      (doseq [k (range 4)]
        (let [tr (:trace (gfi/generate model [] (cm/choicemap :a 0.9) (rng/key (+ 1 k))))]
          (reset! runs 0)
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"the constraint at \[:c :p :s :x\] was not visited"
                                (gfi/update tr (cm/choicemap :a 0.1) (rng/key (+ 20 k))))))))))

(deftest user-code-in-a-revised-call-keeps-its-errors
  ;; an estimator runs generate itself (or gen.batched's), with a constraint
  ;; its model leaves unvisited: loud, though the call around it is being
  ;; revised (a revise drops old choices only in its own run,
  ;; gfi/outside-revise)
  (let [sub (dyn/with-params (gen [] (splice :c (gen [] (trace :x (d/normal (param :w) 1))))) {:w 1.0})
        cs (cm/choicemap [:c :x] 0.0 [:c :typo] 1.0)]
    (doseq [[label estimate] {"generate" (fn [_ _cs k] (:weight (gfi/generate sub [] cs k)))
                              "gen.batched's generate" (fn [_ _cs k] (ts/number (array/sum (:weights (batched/generate sub [] cs 2 k)))))}]
      (testing label
        (let [est (custom/estimated {:addresses [:o]
                                     :estimate estimate
                                     :simulate (fn [_ _k] {:choices (cm/choicemap :o 0.0) :log-estimate 0.0})})
              model (given-a-store (gen [] (splice :p est)))
              tr (gfi/simulate model [] (rng/key 1))]
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"the constraint at \[:c :typo\] was not visited"
                                (gfi/update tr (cm/choicemap :a 0.9) (rng/key 2)))))))))

;; ---------------------------------------------------------------------------
;; The discard of a value replaced by nested choices (review 5, R5-4)
;; ---------------------------------------------------------------------------

(deftest an-old-value-replaced-by-nested-choices-is-discarded
  (let [leafy (gen [] (trace :x (d/normal (param :w) 1)))
        nested (gen [] (splice :x leafy))
        model (given-a-store (gen [] (splice :p (gen [] (if (> (param :w) 5.0) (trace :v (d/normal 0 1)) (splice :v nested))))))
        tr (:trace (gfi/generate model [] (cm/choicemap :a 0.9) (rng/key 1)))
        cs (cm/choicemap :a 0.1 [:c :p :v :x :x] 1.5)
        {u :trace w :weight dsc :discard} (gfi/update tr cs (rng/key 6))]
    (is (cm/has-value? (cm/get-submap (:choices tr) [:c :p :v])))
    (is (consistent? model u))
    (is (ts/close? (update-truth model tr u cs) w))
    (is (= (ts/host (tr [:c :p :v])) (ts/host (cm/get-choice dsc [:c :p :v]))))
    (is (discard-ok? tr u cs dsc))))

;; ---------------------------------------------------------------------------
;; Middleware under a changed store (review 5, R5-7)
;; ---------------------------------------------------------------------------

(deftest update-middleware-sees-an-update-under-a-changed-store
  (let [seen (atom 0)
        watched (dyn/with-transitions inner {:update (fn [state p dist k] (swap! seen inc) ((dyn/site-transitions :update) state p dist k))})
        model (given-a-store (gen [] (splice :p watched 0.0)))
        tr (gfi/simulate model [] (rng/key 5))]
    (reset! seen 0)
    (let [{u :trace w :weight} (gfi/update tr (cm/choicemap :a 0.9) (rng/key 6))]
      (is (= 1 @seen))
      (is (ts/close? (- (assessed model u) (ts/number (:score tr))) w)))))

;; ---------------------------------------------------------------------------
;; Generative functions built again stay incremental (review 5, R5-3;
;; mutations M10, M11, M14)
;; ---------------------------------------------------------------------------

(deftest calls-built-again-stay-incremental
  ;; kernel runs for an update of one element of an 8-element Map: 1 when
  ;; nothing it depends on changed (each case reran all 8 before)
  (let [runs (atom 0)
        mk (gen [x] (swap! runs inc) (trace :y (d/normal x 1)))
        xs (vec (repeat 8 0.0))
        holds-map (gen [] (splice :m (comb/map mk) xs))
        store-255 (zipmap (map #(keyword (str "p" %)) (range 255)) (repeatedly #(array/array [1.0 2.0])))
        count-update (fn [model cs]
                       (let [tr (gfi/simulate model [] (rng/key 1))]
                         (reset! runs 0)
                         (gfi/update tr cs (rng/key 2))
                         @runs))]
    (doseq [[label model cs]
            [["a Switch built in the body" (gen [] (splice :s (comb/switch (comb/map mk) (comb/map mk)) 0 xs)) (cm/choicemap [:s 3 :y] 1.0)]
             ["a keyed Switch built in the body" (gen [] (splice :s (comb/switch {:a (comb/map mk) :b (comb/map mk)}) :a xs))
              (cm/choicemap [:s 3 :y] 1.0)]
             ["a static parent of a Switch built in the body"
              (gen ^:static [] (let [r (splice :s (comb/switch (comb/map mk) (comb/map mk)) 0 xs)] r)) (cm/choicemap [:s 3 :y] 1.0)]
             ["a callee closing over a computed number"
              (gen [] (let [a (trace :a (d/normal 0 1)) m (+ a 1.0)] (splice :c (gen [] (splice :m (comb/map mk) xs) m))))
              (cm/choicemap [:c :m 3 :y] 1.0)]
             ["a callee closing over a gen built in the body"
              (gen [] (let [g (comb/map mk)] (splice :c (gen [] (splice :m g xs))))) (cm/choicemap [:c :m 3 :y] 1.0)]
             ["a callee closing over a vector built in the body"
              (gen [] (let [v (mapv identity xs)] (splice :c (gen [] (count v) (splice :m (comb/map mk) xs)))))
              (cm/choicemap [:c :m 3 :y] 1.0)]
             ["call-at of a Map, both built in the body" (gen [] (splice :c (comb/call-at (comb/map mk)) xs :k))
              (cm/choicemap [:c :k 3 :y] 1.0)]
             ["with-transitions built in the body" (gen [] (splice :c (dyn/with-transitions holds-map {:update (dyn/site-transitions :update)})))
              (cm/choicemap [:c :m 3 :y] 1.0)]
             ["a store holding a vector built in the body" (gen [] (splice :c (dyn/with-params holds-map {:w (vector 1.0 (+ 1.0 1.0))})))
              (cm/choicemap [:c :m 3 :y] 1.0)]
             ["an outer store merged with an inner one (M11)"
              (dyn/with-params (gen [] (splice :c (dyn/with-params holds-map {:v 1.0}))) {:w 2.0}) (cm/choicemap [:c :m 3 :y] 1.0)]
             ["an outer store of 255 arrays merged with an inner one (review 7, R7-2: each entry spent the budget, so the merge was never the same)"
              (dyn/with-params (gen [] (splice :c (dyn/with-params holds-map {:v 1.0}))) store-255) (cm/choicemap [:c :m 3 :y] 1.0)]
             ["a Map simulated under a store, then updated (M14)" (dyn/with-params holds-map {:w 2.0}) (cm/choicemap [:m 3 :y] 1.0)]]]
      (testing label
        (is (= 1 (count-update model cs)))))
    (testing "a Switch built in the body, an unrelated choice regenerated"
      (let [model (gen [] (splice :s (comb/switch (comb/map mk) (comb/map mk)) 0 xs) (trace :z (d/normal 0 1)))
            tr (gfi/simulate model [] (rng/key 1))]
        (reset! runs 0)
        (gfi/regenerate tr (sel/select :z) (rng/key 3))
        (is (= 0 @runs))))
    (testing "after a store change, the next update reruns only its element (the marks are the new store's, M10)"
      (let [model (given-a-store (gen [] (splice :m (comb/map (gen [x] (swap! runs inc) (trace :y (d/normal (+ x (param :w)) 1)))) xs)))
            tr (gfi/simulate model [] (rng/key 1))
            {u :trace} (gfi/update tr (cm/choicemap :a 0.9) (rng/key 2))]
        (reset! runs 0)
        (gfi/update u (cm/choicemap [:c :m 3 :y] 1.0) (rng/key 3))
        (is (= 1 @runs))))
    (testing "a deterministic built in the body, an unrelated choice: it does not run (its closures are the same, N14)"
      (let [applied (atom 0)
            model (gen [] (let [v (splice :d (custom/deterministic {:apply (fn [[x]] (swap! applied inc) (* 2.0 x))}) 1.0)]
                            (trace :z (d/normal v 1))))
            tr (gfi/simulate model [] (rng/key 1))]
        (reset! applied 0)
        (gfi/update tr (cm/choicemap :z 0.5) (rng/key 2))
        (is (= 0 @applied))))
    (testing "an internal proposal of a static model skips its unchanged calls (the model made its trace, N19)"
      (let [static (gen ^:static [] (let [a (trace :a (d/normal 0 1)) c (splice :c (gen [] (swap! runs inc) (trace :x (d/normal 0 1))))] a))
            model (prop/with-internal-proposal static (gen [_cs] nil))
            tr (gfi/simulate model [] (rng/key 1))]
        (reset! runs 0)
        (gfi/update tr (cm/choicemap :a 0.5) (rng/key 2))
        (is (= 0 @runs))))
    (testing "an estimated density built in the body: an identity update weighs 0, and nothing reruns it (review 6, R6-5)"
      (let [estimates (atom 0)
            likelihood (fn [theta]
                         (custom/estimated {:addresses [:y]
                                            :estimate (fn [_ cs k] (swap! estimates inc)
                                                        (:weight (gfi/assess (d/normal theta 1) [] (cm/get-submap cs :y) k)))
                                            :simulate (fn [_ k] (let [t (gfi/simulate (d/normal theta 1) [] k)]
                                                                  {:choices (cm/choicemap :y (:retval t)) :log-estimate (:score t)}))}))
            model (gen [] (let [theta (trace :theta (d/normal 0 1)) z (trace :z (d/normal 0 1))]
                            (splice :obs (likelihood theta))
                            z))
            tr (:trace (gfi/generate model [] (cm/choicemap [:obs :y] 0.5) (rng/key 1)))]
        (reset! estimates 0)
        (is (== 0.0 (ts/number (:weight (gfi/update tr cm/EMPTY (rng/key 2))))))
        (gfi/update tr (cm/choicemap :z 0.1) (rng/key 3))
        (is (= 0 @estimates))))
    (testing "an internal proposal built in the body, an unrelated choice: its proposal does not run"
      (let [q-runs (atom 0)
            kq' (gen [cs x] (swap! q-runs inc) (when-not (cm/has-value? (cm/get-submap cs :y)) (trace :y (d/normal x 2))))
            model (gen [] (splice :e (prop/with-internal-proposal mk kq') 0.0) (trace :z (d/normal 0 1)))
            tr (gfi/simulate model [] (rng/key 1))]
        (reset! runs 0)
        (reset! q-runs 0)
        (gfi/update tr (cm/choicemap :z 0.5) (rng/key 2))
        (is (= [1 0] [@runs @q-runs]))))))

;; ---------------------------------------------------------------------------
;; Unvisited constraints are named by their full paths; with-params needs
;; a body
;; ---------------------------------------------------------------------------

(deftest an-unvisited-constraint-is-named-by-its-full-path
  ;; in generate, and in assess (review 6, R6-7), through a dynamic and a
  ;; static parent
  (doseq [[label model] [["dynamic" (gen [] (splice :c (gen [] (splice :m (comb/map (gen [x] (trace :y (d/normal x 1)))) [0.0 0.0]))))]
                         ["static" (gen ^:static [] (let [r (splice :c (gen [] (splice :m (comb/map (gen [x] (trace :y (d/normal x 1)))) [0.0 0.0])))] r))]]
          [op run] {:generate #(gfi/generate %1 [] %2 (rng/key 1))
                    :assess #(gfi/assess %1 [] (cm/merge (cm/choicemap [:c :m 0 :y] 0.5 [:c :m 1 :y] 0.5) %2) (rng/key 1))}
          [p cs] {[:c :m 1 :z] (cm/choicemap [:c :m 1 :z] 1.0)
                  [:c :m 5] (cm/choicemap [:c :m 5 :y] 1.0)}]
    (testing [label op p]
      (let [e (try (run model cs) nil (catch clojure.lang.ExceptionInfo e e))]
        (is (= p (:path (ex-data e))))
        (is (:unvisited (ex-data e)))
        (is (re-find (re-pattern (java.util.regex.Pattern/quote (str p " was not visited"))) (ex-message e))))))
  (testing "every unvisited path of an error is named, not only the first"
    (let [model (gen [] (splice :c (gen [] (trace :x (d/normal 0 1)))))
          e (try (gfi/generate model [] (cm/choicemap [:c :y1] 1.0 [:c :y2] 2.0) (rng/key 1)) nil
                 (catch clojure.lang.ExceptionInfo e e))]
      (is (= #{[:c :y1] [:c :y2]} (set (:paths (ex-data e)))))))
  (testing "a user function's own error keeps its own path"
    (let [inner (gen [] (trace :x (d/normal 0 1)))
          model (gen [] (splice :d (custom/deterministic {:apply (fn [_] (:weight (gfi/generate inner [] (cm/choicemap :typo 1.0) (rng/key 2))))})))
          e (try (gfi/simulate model [] (rng/key 1)) nil (catch clojure.lang.ExceptionInfo e e))]
      (is (= [:typo] (:path (ex-data e))))
      (is (not (:unvisited (ex-data e)))))))

;; ---------------------------------------------------------------------------
;; Probes of the mutations review 6 found surviving (R6-6)
;; ---------------------------------------------------------------------------

(def ^:private ret-w (gen [] (let [x (trace :x (d/normal 0 1))] (+ x (param :w)))))

(defn- cycle-of [v] (letfn [(f [] (+ v (g))) (g [] (if (pos? v) 0 (f)))] f))

(deftest probes-of-review-6s-survivors
  (doseq [[label model before cs]
          [["a def'd static callee under an outer store from a choice (the static skip compares the store)"
            (gen [] (let [a (trace :a (d/uniform 0 1))]
                      (splice :c (dyn/with-params (gen [] (splice :s static-model)) {:w (* 10.0 a)}))))
            (cm/choicemap :a 0.1) (cm/choicemap :a 0.9)]
           ["a kernel over a vector whose length a choice decides (sequences compare lengths)"
            (gen [] (let [n (trace :n (d/categorical [0.5 0.5]))
                          v (vec (repeat (clojure.core/+ 2 (long n)) 1.0))]
                      (splice :m (comb/map (gen [x] (trace :y (d/normal (+ x (reduce + 0.0 v)) 1)))) [0.0 1.0])))
            (cm/choicemap :n 1) (cm/choicemap :n 0)]
           ["a kernel over a primitive double local (primitive closure fields compare)"
            (gen [] (let [s (trace :s (d/normal 0 1))
                          p (double (* 10.0 (Math/signum (double s))))]
                      (splice :m (comb/map (gen [x] (trace :y (d/normal (+ x p) 1)))) [0.0 1.0])))
            (cm/choicemap :s 0.5) (cm/choicemap :s -0.5)]
           ["a static parent after a revised deterministic callee (generate-again's retdiff is unknown-change)"
            (gen ^:static [] (let [a (trace :a (d/normal 0 1))
                                   v (splice :d (custom/deterministic {:apply (fn [[x]] (+ x (* 10.0 a)))}) 1.0)
                                   z (trace :z (d/normal v 1))]
                               z))
            (cm/choicemap :a 0.2) (cm/choicemap :a 1.5)]
           ["a static parent after a revised dynamic callee (the handler revise's retdiff is the handler's)"
            (gen ^:static [] (let [a (trace :a (d/normal 0 1))
                                   v (splice :c (dyn/with-params ret-w {:w (* 10.0 a)}))
                                   z (trace :z (d/normal v 1))]
                               z))
            (cm/choicemap :a 0.2) (cm/choicemap :a 1.5)]]]
    (testing label
      (let [tr (:trace (gfi/generate model [] before (rng/key 1)))
            {u :trace w :weight} (gfi/update tr cs (rng/key 2))]
        (is (consistent? model u))
        (is (ts/close? (- (assessed model u) (ts/number (:score tr))) w)))))
  (testing "a kernel closing over a letfn cycle: the budget ends the comparison"
    (let [model (gen [] (let [f (cycle-of 1)] (splice :m (comb/map (gen [x] (trace :y (d/normal (+ x (f)) 1)))) [0.0 1.0])))
          tr (gfi/simulate model [] (rng/key 7))
          cs (cm/choicemap [:m 0 :y] 0.5)
          {u :trace w :weight} (gfi/update tr cs (rng/key 8))]
      (is (ts/close? (- (assessed model u) (ts/number (:score tr))) w))))
  (testing "an exact proposal's update keeps its maker (the next regenerate projects by the exact proposal)"
    (let [egf (enum/with-exact-proposal (gen [] (let [b (trace :b (d/bernoulli 0.3))] (trace :x (d/normal (if b 2.0 0.0) 1)))) [:b])
          u (:trace (gfi/update (gfi/simulate egf [] (rng/key 13)) (cm/choicemap :x 0.7) (rng/key 14)))
          s (sel/select :b)
          {r :trace rw :weight} (gfi/regenerate u s (rng/key 15))]
      (is (identical? egf (trace/maker u)))
      (is (ts/close? (- (ts/number (gfi/project r (sel/complement s))) (ts/number (gfi/project u (sel/complement s)))) rw)))))

(deftest with-params-needs-a-dynamic-generative-function
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"with-params needs a dynamic generative function"
                        (dyn/with-params (comb/map kernel) {:w 1.0})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"with-transitions needs a dynamic generative function"
                        (dyn/with-transitions (comb/map kernel) {}))))
