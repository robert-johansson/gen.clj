(ns gen.genjl.custom-determ-test
  "Gen.jl 0.4.8's test/modeling_library/custom_determ.jl, ported: custom
   deterministic generative functions (gen.custom/deterministic), with an
   update that reuses its state and a gradient of its own.

   Gen.jl's MyDeterministicGF multiplies by a trainable parameter of its
   own; gen.clj's custom functions have none (G5), so here it is the
   constant 1.0 in the state, and gen.clj handles a no-change update itself
   (so no update for NoChange is written). Argument gradients are values
   from choice-gradients with :argnums (D7): there is no accumulator to
   scale. Arguments that are differentiated are floats (gen.clj
   differentiates trees of floats only)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as cm]
            [gen.custom :as custom]
            [gen.diff :as diff]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :as ts]))

(defn- vector-sum
  "Gen.jl's incremental sum: the previous sum, plus the changed elements'
   differences (updated's indices), plus the appended elements, less the
   dropped ones."
  [prev-sum prev arr {:keys [updated]}]
  (let [n-prev (count prev) n (count arr)]
    (+ prev-sum
       (reduce + 0 (map #(- (arr %) (prev %)) (keys updated)))
       (reduce + 0 (subvec arr (min n n-prev)))
       (- (reduce + 0 (subvec prev (min n n-prev)))))))

;; custom_determ.jl:13, MyDeterministicGF: sum(arr) * my_param
(def ^:private my-param 1.0)

(def ^:private my-deterministic
  (custom/deterministic
   {:apply-with-state (fn [[arr]] (let [retval (* (reduce + arr) my-param)]
                                    [retval {:sum retval :prev arr :my-param my-param}]))
    :update-with-state (fn [state [arr] [argdiff]]
                         (let [retval (if (diff/vector-diff? argdiff)
                                        (vector-sum (* (:sum state) (:my-param state)) (:prev state) arr argdiff)
                                        (* (reduce + arr) (:my-param state)))]
                           [(assoc state :sum retval :prev arr) retval diff/unknown-change]))
    :gradient-with-state (fn [state _args retgrad]
                           [(vec (repeat (count (:prev state)) (* retgrad (:my-param state))))])}))

(deftest custom-deterministic-generative-function
  ;; custom_determ.jl:1
  (testing "simulate"
    (let [tr (gfi/simulate my-deterministic [[1 2 3]] (rng/key 1))]
      (is (== (+ 1 2 3) (:retval tr)))
      (is (= 0.0 (:score tr)))
      (is (= 0.0 (gfi/project tr sel/none)))
      (is (cm/empty? (:choices tr)))
      (is (= [[1 2 3]] (:args tr)))
      (is (identical? my-deterministic (:gen-fn tr)))))
  (testing "generate"
    (let [{tr :trace w :weight} (gfi/generate my-deterministic [[1 2 3]] cm/EMPTY (rng/key 2))]
      (is (= 0.0 w))
      (is (== (+ 1 2 3) (:retval tr)))))
  (testing "update (UnknownChange)"
    (let [tr (gfi/simulate my-deterministic [[1 2 3]] (rng/key 3))
          {t2 :trace w :weight rd :retdiff} (gfi/update tr [[1 2 4]] [diff/unknown-change] cm/EMPTY (rng/key 4))]
      (is (= 0.0 w))
      (is (== (+ 1 2 4) (:retval t2)))
      (is (= [[1 2 4]] (:args t2)))
      (is (= diff/unknown-change rd))))
  (testing "update (VectorDiff): element 2 changed, one appended"
    (let [tr (gfi/simulate my-deterministic [[1 2 3]] (rng/key 5))
          {t2 :trace w :weight rd :retdiff}
          (gfi/update tr [[1 2 4 5]] [(diff/vector-diff 4 3 {2 diff/unknown-change})] cm/EMPTY (rng/key 6))]
      (is (= 0.0 w))
      (is (== (+ 1 2 4 5) (:retval t2)))
      (is (= [[1 2 4 5]] (:args t2)))
      (is (= diff/unknown-change rd))))
  (testing "regenerate"
    (let [tr (gfi/simulate my-deterministic [[1 2 3]] (rng/key 7))
          {t2 :trace w :weight rd :retdiff} (gfi/regenerate tr [[1 2 4]] [diff/unknown-change] sel/none (rng/key 8))]
      (is (= 0.0 w))
      (is (== (+ 1 2 4) (:retval t2)))
      (is (= [[1 2 4]] (:args t2)))
      (is (= diff/unknown-change rd))))
  (let [arg-grad #(first (:arg-grads (grad/choice-gradients (gfi/simulate my-deterministic [[1.0 2.0 3.0]] %)
                                                            sel/none 2.0 {:argnums [0]})))]
    (testing "choice gradients"
      (is (= [2.0 2.0 2.0] (ts/host (arg-grad (rng/key 9))))))
    (testing "the argument gradient, not scaled by 3 (a value: no accumulator, D7)"
      (is (= [2.0 2.0 2.0] (ts/host (arg-grad (rng/key 10))))))))

;; custom_determ.jl:128, CustomPlus: a + b, its gradient (retgrad, retgrad)
(def ^:private custom-plus
  (custom/deterministic {:apply (fn [[a b]] (+ a b))
                         :gradient (fn [_args _retval retgrad] [retgrad retgrad])}))

(deftest custom-gradient-generative-function
  ;; custom_determ.jl:126
  (let [arg-grads #(:arg-grads (grad/choice-gradients (gfi/simulate custom-plus [1.0 2.0] %) sel/none 2.0 {:argnums [0 1]}))]
    (testing "choice gradients"
      (is (= [2.0 2.0] (ts/host (arg-grads (rng/key 1))))))
    (testing "the argument gradients, not scaled (D7)"
      (is (= [2.0 2.0] (ts/host (arg-grads (rng/key 2))))))))

;; custom_determ.jl:151, MySum: an update for a VectorDiff, Gen.jl's
;; default (recomputing) otherwise
(def ^:private my-sum
  (custom/deterministic
   {:apply-with-state (fn [[arr]] (let [s (reduce + arr)] [s {:prev-arr arr :sum s}]))
    :update-with-state (fn [state [arr] [argdiff]]
                         (let [s (if (diff/vector-diff? argdiff)
                                   (vector-sum (:sum state) (:prev-arr state) arr argdiff)
                                   (reduce + arr))]
                           [{:prev-arr arr :sum s} s diff/unknown-change]))}))

(deftest custom-update-generative-function
  ;; custom_determ.jl:144
  (let [tr (gfi/simulate my-sum [[1 2 3]] (rng/key 1))]
    (testing "simulate"
      (is (== (+ 1 2 3) (:retval tr)))
      (is (= 0.0 (:score tr))))
    (testing "update (UnknownChange)"
      (let [{t2 :trace w :weight rd :retdiff discard :discard}
            (gfi/update tr [[1 2 4]] [diff/unknown-change] cm/EMPTY (rng/key 2))]
        (is (= 0.0 w))
        (is (== (+ 1 2 4) (:retval t2)))
        (is (= [[1 2 4]] (:args t2)))
        (is (= diff/unknown-change rd))
        (is (cm/empty? discard))))
    (testing "update (VectorDiff)"
      (let [{t2 :trace w :weight rd :retdiff discard :discard}
            (gfi/update tr [[1 2 4 5]] [(diff/vector-diff 4 3 {2 diff/unknown-change})] cm/EMPTY (rng/key 3))]
        (is (= 0.0 w))
        (is (== (+ 1 2 4 5) (:retval t2)))
        (is (= [[1 2 4 5]] (:args t2)))
        (is (= diff/unknown-change rd))
        (is (cm/empty? discard))))))

;; Pending:
;; - custom_determ.jl:122, gen_fn.my_grad == 2 * (1 + 2 + 3) * 3: a custom
;;   function's own trainable parameter, scaled by accumulate_param_gradients!
;;   (G5, genclj-2294).
