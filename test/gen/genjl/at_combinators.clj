(ns gen.genjl.at-combinators
  "The battery that Gen.jl 0.4.8's test/modeling_library/call_at.jl and
   choice_at.jl share. Each file tests a combinator whose last argument is
   a key k, its choices placed under k, on arguments (p k): the kernel's
   parameter p, 0.4 and then 0.2, at k 3 and then 4. A spec describes the
   combinator:

     :gf       the generative function under test
     :addr     (fn [k] address): its choice's address under the key
     :density  (fn [p v] log density): of the choice v at the parameter p
     :retval   (fn [p v] return value)
     :values   [v v']: the values the testsets constrain
     :shallow  {:values n :submaps m}: the top level of every choice map
               (one submap for call_at, one value for choice_at)

   A function per Gen.jl testset runs its assertions on a spec: approx?
   for Gen.jl's isapprox, close? for its == (and for the return value,
   isapprox in call_at but == of booleans in choice_at). Where one file
   asserts what its twin does not, both are asserted: the return value in
   propose (choice_at's) and the score in update (call_at's)."
  (:refer-clojure :exclude [- update])
  (:require [clojure.test :refer [is testing]]
            [gen.array :refer [-]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.genjl.support :as support :refer [approx?]]
            [gen.gfi :as gfi]
            [gen.gradients :as g]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.test-support :refer [close?]]))

(def ^:private p 0.4)
(def ^:private p' 0.2)
(def ^:private k 3)
(def ^:private k' 4)

(def ^:private unknown [diff/unknown-change diff/unknown-change])

(defn- choice [{:keys [addr]} c key] (cm/get-choice c (addr key)))

(defn initial-trace
  "Gen.jl's get_trace: the first value constrained, at (p k)."
  [{:keys [gf addr] [v] :values}]
  (:trace (gfi/generate gf [p k] (cm/choicemap (addr k) v) (rng/key 1))))

(defn assess [{:keys [gf addr density retval] [v] :values}]
  (let [{w :weight rv :retval} (gfi/assess gf [p k] (cm/choicemap (addr k) v) (rng/key 2))]
    (is (approx? (density p v) w))
    (is (close? (retval p v) rv))))

(defn propose [{:keys [gf density retval shallow] :as spec}]
  (let [{c :choices w :weight rv :retval} (gfi/propose gf [p k] (rng/key 3))
        v (choice spec c k)]
    (is (approx? (density p v) w))
    (is (close? (retval p v) rv))
    (is (= shallow (support/shallow c)))))

(defn generate [{:keys [gf addr density retval shallow] [v] :values :as spec}]
  (testing "without constraints"
    (let [{t :trace w :weight} (gfi/generate gf [p k] cm/EMPTY (rng/key 4))]
      (is (close? 0.0 w))
      (is (close? (retval p (choice spec (:choices t) k)) (:retval t)))
      (is (= shallow (support/shallow (:choices t))))))
  (testing "with a constraint"
    (let [{t :trace w :weight} (gfi/generate gf [p k] (cm/choicemap (addr k) v) (rng/key 5))]
      (is (close? v (choice spec (:choices t) k)))
      (is (close? (retval p v) (:retval t)))
      (is (approx? (density p v) w))
      (is (= shallow (support/shallow (:choices t)))))))

(defn project [{:keys [addr density] [v] :values :as spec}]
  (let [tr (initial-trace spec)]
    (is (approx? 0.0 (gfi/project tr sel/none)))
    (is (approx? (density p v) (gfi/project tr (sel/select (addr k)))))))

(defn update [{:keys [addr density retval shallow] [v v'] :values :as spec}]
  (let [tr (initial-trace spec)]
    (testing "a new parameter, the same key, no constraint"
      (let [{t :trace w :weight d :discard} (gfi/update tr [p' k] unknown cm/EMPTY (rng/key 6))]
        (is (close? v (choice spec (:choices t) k)))
        (is (= shallow (support/shallow (:choices t))))
        (is (approx? (- (density p' v) (density p v)) w))
        (is (close? (retval p' v) (:retval t)))
        (is (cm/empty? d))
        (is (approx? (density p' v) (:score t)))))
    (doseq [key [k k']]
      (testing (str "a new parameter, a constraint at key " key ": the old value is discarded")
        (let [{t :trace w :weight d :discard} (gfi/update tr [p' key] unknown (cm/choicemap (addr key) v') (rng/key 7))]
          (is (close? v' (choice spec (:choices t) key)))
          (is (= shallow (support/shallow (:choices t))))
          (is (approx? (- (density p' v') (density p v)) w))
          (is (close? (retval p' v') (:retval t)))
          (is (close? v (choice spec d k)))
          (is (= shallow (support/shallow d)))
          (is (approx? (density p' v') (:score t))))))))

(defn regenerate [{:keys [addr density retval shallow] [v] :values :as spec}]
  (let [tr (initial-trace spec)]
    (testing "a new parameter, the same key, not selected"
      (let [{t :trace w :weight} (gfi/regenerate tr [p' k] unknown sel/none (rng/key 8))]
        (is (close? v (choice spec (:choices t) k)))
        (is (= shallow (support/shallow (:choices t))))
        (is (approx? (- (density p' v) (density p v)) w))
        (is (close? (retval p' v) (:retval t)))
        (is (approx? (density p' v) (:score t)))))
    (doseq [[key s what] [[k (sel/select (addr k)) "the same key, selected"]
                          [k' sel/none "another key, not selected"]]]
      (testing (str "a new parameter, " what ": a value from the prior, weight 0")
        (let [{t :trace w :weight} (gfi/regenerate tr [p' key] unknown s (rng/key 9))
              v' (choice spec (:choices t) key)]
          (is (= shallow (support/shallow (:choices t))))
          (is (close? 0.0 w))
          (is (close? (retval p' v') (:retval t)))
          (is (approx? (density p' v') (:score t))))))))

(defn- gradients
  "choice-gradients of tr at selection s, differentiating every argument but
   the key."
  [tr s retgrad]
  (g/choice-gradients tr s retgrad {:argnums (vec (range (dec (count (:args tr)))))}))

(defn- check-arg-grads
  "One gradient per argument, the key's nil (Gen.jl's nothing)."
  [expected gs]
  (is (= (count expected) (count gs)))
  (doseq [[e a] (map vector expected gs)]
    (if (nil? e) (is (nil? a) "the key has no gradient") (is (approx? e a)))))

(defn arg-gradients
  "accumulate_param_gradients!: the arguments' gradients, which gen.clj's
   choice-gradients gives with nothing selected (D7)."
  [tr retgrad expected]
  (check-arg-grads expected (:arg-grads (gradients tr sel/none retgrad))))

(defn unselected-gradients
  "choice_gradients with nothing selected: no values, no gradients, and the
   arguments' gradients."
  [tr retgrad expected-arg-grads]
  (let [{:keys [values gradients arg-grads]} (gradients tr sel/none retgrad)]
    (is (cm/empty? values))
    (is (cm/empty? gradients))
    (check-arg-grads expected-arg-grads arg-grads)))

(defn selected-gradients
  "choice_gradients with tr's choice, v, selected: its value and gradient
   under the key, and the arguments' gradients."
  [{:keys [addr shallow]} tr v retgrad expected-arg-grads expected-grad]
  (let [{:keys [values gradients arg-grads]} (gradients tr (sel/select (addr k)) retgrad)]
    (is (close? v (cm/get-choice values (addr k))))
    (is (approx? expected-grad (cm/get-choice gradients (addr k))))
    (is (= shallow (support/shallow gradients)))
    (is (= shallow (support/shallow values)))
    (check-arg-grads expected-arg-grads arg-grads)))
