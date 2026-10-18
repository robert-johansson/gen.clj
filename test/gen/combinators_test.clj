(ns gen.combinators-test
  "Map, Unfold and Switch: the GFI laws, incremental updates against full
   recomputation (precise change hints must give the same results as
   unknown-change, thesis law 8), and length and branch changes."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as a]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.laws :as laws :refer [close?]]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; ---------------------------------------------------------------------------
;; Models
;; ---------------------------------------------------------------------------

(def element
  (gen [x]
    (let [z (trace :z (d/normal x 1))]
      (when (trace :flip (d/bernoulli 0.5))
        (trace :extra (d/gamma 2 1)))
      z)))

(def step
  "A random walk with drift and a noisy observation."
  (gen [t state drift]
    (let [s (trace :s (d/normal (a/+ state drift) 1))]
      (trace :obs (d/normal s 0.5))
      s)))

(def a-branch (gen [mu] (trace :a (d/normal mu 1))))
(def b-branch (gen [mu] (+ (trace :b (d/gamma 2 1)) (trace :c (d/poisson 3.0)))))

(def mapped (comb/map element))
(def unfolded (comb/unfold step))
(def switched (comb/switch a-branch b-branch))
(def keyed (comb/switch {:a a-branch :b b-branch}))

(def spliced
  "A dynamic model that splices all three."
  (gen [n]
    (let [mu (trace :mu (d/normal 0 1))
          xs (splice :xs mapped (vec (repeat n mu)))
          path (splice :path unfolded n mu 0.1)]
      (splice :pick switched (if (trace :which (d/bernoulli 0.5)) 0 1) (peek path))
      xs)))

(def models
  [[:map mapped [[1.0 2.0 3.0]]]
   [:map-2-args (comb/map (gen [x y] (trace :z (d/normal x y)))) [[0.0 1.0] [1.0 2.0]]]
   [:unfold unfolded [4 0.0 0.1]]
   [:switch switched [1 0.5]]
   [:keyed-switch keyed [:a 0.5]]
   [:spliced spliced [3]]])

(laws/deflaws combinators (laws/cases models) 100)

;; ---------------------------------------------------------------------------
;; Map
;; ---------------------------------------------------------------------------

(defn- subtraces [tr] (:gen.combinators/subtraces tr))

(deftest map-basics
  (let [tr (gfi/simulate mapped [[1.0 2.0 3.0]] (rng/key 1))]
    (is (= 3 (count (:retval tr))))
    (is (= (tr [1 :z]) (get (:retval tr) 1)) "element i's choices are at address i")
    (is (= 2 (count (mapped [1.0 2.0]))) "an untraced call returns the vector")
    (testing "an element-wise update visits only the changed element"
      (let [{t2 :trace w :weight rd :retdiff}
            (gfi/update tr [[1.0 5.0 3.0]] [(diff/vector-diff 3 3 {1 diff/unknown-change})] cm/EMPTY (rng/key 2))]
        (is (identical? ((subtraces tr) 0) ((subtraces t2) 0)))
        (is (identical? ((subtraces tr) 2) ((subtraces t2) 2)))
        (is (= (tr [1 :z]) (t2 [1 :z])) "the choice is kept, and rescored")
        (is (close? w (a/- (:score t2) (:score tr))))
        (is (close? (:score t2) (:weight (gfi/assess mapped [[1.0 5.0 3.0]] (:choices t2)))))
        (is (diff/no-change? rd) "the return values did not change")))
    (testing "shrinking discards the removed elements"
      (let [{t2 :trace w :weight disc :discard rd :retdiff}
            (gfi/update tr [[1.0]] [diff/unknown-change] cm/EMPTY (rng/key 3))]
        (is (= [[0 :z]] (filter #(= :z (peek %)) (cm/addresses (:choices t2)))))
        (is (= (cm/get-submap (:choices tr) 1) (cm/get-submap disc 1)))
        (is (close? w (a/- (:score t2) (:score tr))))
        (is (= (diff/vector-diff 1 3 {}) rd))))
    (testing "growing generates new elements, constrained where asked"
      (let [{t2 :trace w :weight} (gfi/update tr [[1.0 2.0 3.0 4.0]] [diff/unknown-change]
                                              (cm/choicemap [3 :z] 0.25 [3 :flip] false) (rng/key 4))]
        (is (= 0.25 (t2 [3 :z])))
        (is (close? w (gfi/project t2 (sel/select [3 :z] [3 :flip]))))))))

;; ---------------------------------------------------------------------------
;; Unfold
;; ---------------------------------------------------------------------------

(deftest unfold-basics
  (let [tr (gfi/simulate unfolded [5 0.0 0.1] (rng/key 1))]
    (is (= 5 (count (:retval tr))))
    (is (= (tr [2 :s]) (get (:retval tr) 2)))
    (testing "extending by a step (the particle filter's move) visits only the new step"
      (let [{t2 :trace w :weight rd :retdiff}
            (gfi/update tr [6 0.0 0.1] [diff/unknown-change diff/no-change diff/no-change]
                        (cm/choicemap [5 :obs] 1.5) (rng/key 2))]
        (is (every? true? (map identical? (subtraces tr) (take 5 (subtraces t2)))))
        (is (close? w (gfi/project t2 (sel/select [5 :obs]))) "the weight is the new observation's density")
        (is (= (diff/vector-diff 6 5 {}) rd))))
    (testing "changing a state re-scores every later step and keeps their choices"
      (let [{t2 :trace w :weight disc :discard rd :retdiff}
            (gfi/update tr (cm/choicemap [1 :s] 3.0) (rng/key 3))]
        (is (identical? ((subtraces tr) 0) ((subtraces t2) 0)) "earlier steps are untouched")
        (is (= (tr [4 :s]) (t2 [4 :s])))
        (is (= (cm/choicemap [1 :s] (tr [1 :s])) disc))
        (is (close? w (a/- (:score t2) (:score tr))))
        (is (close? (:score t2) (:weight (gfi/assess unfolded [5 0.0 0.1] (:choices t2)))))
        (is (= #{1} (set (keys (:updated rd)))) "only the changed state is reported")))
    (testing "a parameter change revisits every step"
      (let [{t2 :trace w :weight} (gfi/update tr [5 0.0 0.7] [diff/no-change diff/no-change diff/unknown-change]
                                              cm/EMPTY (rng/key 4))]
        (is (not-any? true? (map identical? (subtraces tr) (subtraces t2))))
        (is (close? w (a/- (:score t2) (:score tr))))))
    (testing "regenerating one step resamples it and re-scores the rest"
      (let [{t2 :trace w :weight} (gfi/regenerate tr (sel/select [2 :s]) (rng/key 5))
            later (sel/select [2 :obs] [3 :s] [3 :obs] [4 :s] [4 :obs])]
        (is (not= (tr [2 :s]) (t2 [2 :s])))
        (is (= (tr [3 :s]) (t2 [3 :s])))
        (is (close? w (a/- (gfi/project t2 later) (gfi/project tr later))))))
    (testing "shrinking"
      (let [{t2 :trace w :weight disc :discard} (gfi/update tr [2 0.0 0.1] [diff/unknown-change diff/no-change diff/no-change]
                                                            cm/EMPTY (rng/key 6))]
        (is (= 2 (count (:retval t2))))
        (is (= #{2 3 4} (set (map first (cm/addresses disc)))))
        (is (close? w (a/- (:score t2) (:score tr))))))))

;; ---------------------------------------------------------------------------
;; Precise change hints give the same results as unknown-change (law 8)
;; ---------------------------------------------------------------------------

(def gen-map-change
  "A Map update: a trace's arguments, new arguments of a possibly different
   length, and constraints on some new elements."
  (gen/let [xs (gen/vector (gen/double* {:min -3 :max 3 :NaN? false}) 0 6)
            ys (gen/vector (gen/double* {:min -3 :max 3 :NaN? false}) 0 6)
            reuse (gen/vector gen/boolean 6)
            seed gen/nat]
    (let [ys (mapv (fn [y keep? x] (if (and keep? x) x y)) ys reuse (concat xs (repeat nil)))]
      {:xs xs :ys ys :seed seed})))

(defn- precise-diff [xs ys]
  (diff/vector-diff (count ys) (count xs)
                    (into {} (keep (fn [i] (when (not= (xs i) (ys i)) [i diff/unknown-change])))
                          (range (min (count xs) (count ys))))))

(defn- weight-identity?
  "With no constraints on new elements, update's weight is the new score
   minus the old, less the score of the newly generated elements (the fresh
   choices). Removed elements' scores leave with the old score."
  [tr {t2 :trace w :weight} n n']
  (let [fresh (gfi/project t2 (apply sel/select (range n n')))]
    (close? w (a/- (a/- (:score t2) (:score tr)) fresh))))

(defn- same-result? [r1 r2]
  (and (= (:choices (:trace r1)) (:choices (:trace r2)))
       (= (:retval (:trace r1)) (:retval (:trace r2)))
       (close? (:score (:trace r1)) (:score (:trace r2)))
       (close? (:weight r1) (:weight r2))
       (= (:discard r1) (:discard r2))))

(defspec map-precise-hints-match-unknown-change 200
  (prop/for-all [{:keys [xs ys seed]} gen-map-change]
    (let [tr (gfi/simulate mapped [xs] (rng/key seed))
          k (rng/key (inc seed))
          precise (gfi/update tr [ys] [(precise-diff xs ys)] cm/EMPTY k)
          unknown (gfi/update tr [ys] [diff/unknown-change] cm/EMPTY k)]
      (and (same-result? precise unknown)
           (close? (:score (:trace precise)) (:weight (gfi/assess mapped [ys] (:choices (:trace precise)))))
           (weight-identity? tr precise (count xs) (count ys))))))

(def gen-unfold-change
  (gen/let [n gen/nat
            n' gen/nat
            init (gen/elements [0.0 0.5])
            drift (gen/elements [0.1 -0.2])
            t (gen/choose 0 12)
            v (gen/double* {:min -3 :max 3 :NaN? false})
            seed gen/nat]
    {:n (min n 8) :n' (min n' 8) :init init :drift drift :t t :v v :seed seed}))

(defspec unfold-precise-hints-match-unknown-change 200
  (prop/for-all [{:keys [n n' init drift t v seed]} gen-unfold-change]
    (let [tr (gfi/simulate unfolded [n 0.0 0.1] (rng/key seed))
          k (rng/key (inc seed))
          cs (if (< t (min n n')) (cm/choicemap [t :s] v) cm/EMPTY)
          args [n' init drift]
          precise (gfi/update tr args [diff/unknown-change
                                       (if (= init 0.0) diff/no-change diff/unknown-change)
                                       (if (= drift 0.1) diff/no-change diff/unknown-change)]
                              cs k)
          unknown (gfi/update tr args (vec (repeat 3 diff/unknown-change)) cs k)]
      (and (same-result? precise unknown)
           (close? (:score (:trace precise)) (:weight (gfi/assess unfolded args (:choices (:trace precise)))))
           (weight-identity? tr precise n n')))))

;; ---------------------------------------------------------------------------
;; Switch
;; ---------------------------------------------------------------------------

(deftest switch-basics
  (let [tr (gfi/simulate switched [0 0.5] (rng/key 1))]
    (is (= [[:a]] (cm/addresses (:choices tr))) "the branch's choices, with no extra address level")
    (testing "changing the branch generates it afresh and discards every old choice"
      (let [{t2 :trace w :weight disc :discard rd :retdiff}
            (gfi/update tr [1 0.5] [diff/unknown-change diff/no-change] (cm/choicemap :b 1.5) (rng/key 2))]
        (is (= 1.5 (t2 :b)))
        (is (= (:choices tr) disc))
        (is (close? w (a/- (gfi/project t2 (sel/select :b)) (:score tr))))
        (is (diff/unknown-change? rd))
        (testing "and updating back with the discard restores the trace"
          (let [{t3 :trace w3 :weight} (gfi/update t2 [0 0.5] [diff/unknown-change diff/no-change] disc (rng/key 3))]
            (is (= (:choices tr) (:choices t3)))
            (is (close? w3 (a/- (:score tr) (:score t2))))))))
    (testing "regenerating into a new branch has weight zero"
      (is (zero? (:weight (gfi/regenerate tr [1 0.5] [diff/unknown-change diff/no-change] sel/none (rng/key 4))))))
    (testing "the same branch with a new argument re-scores"
      (let [{t2 :trace w :weight} (gfi/update tr [0 2.0] [diff/no-change diff/unknown-change] cm/EMPTY (rng/key 5))]
        (is (= (tr :a) (t2 :a)))
        (is (close? w (a/- (:score t2) (:score tr))))))
    (is (= [[:b] [:c]] (cm/addresses (:choices (gfi/simulate keyed [:b 0.0] (rng/key 6))))))))

(deftest errors-are-loud
  (let [k (rng/key 0)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one length" (gfi/simulate (comb/map step) [[1] [1 2] [3]] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                          (gfi/generate mapped [[1.0]] (cm/choicemap [3 :z] 1.0) k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                          (gfi/generate unfolded [2 0.0 0.1] (cm/choicemap [:oops :s] 1.0) k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"non-negative integer" (gfi/simulate unfolded [-1 0.0 0.1] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no branch" (gfi/simulate switched [2 0.0] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"single value"
                          (gfi/generate mapped [[1.0]] (cm/->Value 1.0) k)))))

(deftest distributions-rescore-when-their-parameters-change
  ;; a spliced distribution is called through the GFI: an update with a new
  ;; parameter keeps the value and rescores it
  (let [model (gen [mu] (splice :x (d/normal mu 1)))
        tr (gfi/simulate model [0.0] (rng/key 1))
        {t2 :trace w :weight} (gfi/update tr [2.0] [diff/unknown-change] cm/EMPTY (rng/key 2))
        {t3 :trace w3 :weight} (gfi/regenerate tr [2.0] [diff/unknown-change] sel/none (rng/key 3))]
    (is (= (tr :x) (t2 :x) (t3 :x)))
    (is (close? w (a/- (:score t2) (:score tr))))
    (is (close? w3 w))
    (is (not (zero? w)))))
