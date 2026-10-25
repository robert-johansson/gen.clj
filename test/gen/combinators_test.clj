(ns gen.combinators-test
  "Map, Unfold and Switch: the GFI laws, incremental updates against full
   recomputation (precise change hints must give the same results as
   unknown-change, thesis law 8), and length and branch changes."
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.test-support :as ts]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [gen.array :as array :refer [+ - * /]]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.laws :as laws :refer [close?]]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.trace :as trace]))

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
    (let [s (trace :s (d/normal (+ state drift) 1))]
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

(def grow
  "A random binary tree two levels deep at most, a value at each node."
  (gen [depth]
    (let [split? (trace :split (d/bernoulli (if (< depth 2) 0.6 0.0)))]
      {:value (trace :v (d/normal depth 1)) :children (if split? [(inc depth) (inc depth)] [])})))

(def fold
  "A node's sum, observed with noise."
  (gen [v ws] (let [s (reduce + v ws)] (trace :obs (d/normal s 1)) s)))

(def tree (comb/recurse grow fold 2))

(def kern "One choice :x around m." (gen [m] (trace :x (d/normal m 1.0))))

(def as-map "kern mapped." (comb/map kern))

(def as-dynamic
  "A dynamic function making as-map's addresses, its means shifted."
  (gen [ms] (mapv (fn [i m] (splice i kern (+ m 1.0))) (range (count ms)) ms)))

(def as-unfold
  "An Unfold making as-map's addresses at its steps."
  (comb/unfold (gen [_t s] (trace :x (d/normal s 1.0)))))

(def bounded
  "A kernel whose support moves with its argument: a choice an update keeps
   leaves it, and comes back."
  (gen [lo] (trace :v (d/uniform lo (+ lo 1.0)))))

(def bounded-walk
  "A step in [s, s + 1] from the state s, and w's scale v - s, which a v
   outside its support would make negative."
  (gen [_t s] (let [v (trace :v (d/uniform s (+ s 1.0)))] (trace :w (d/normal 0 (- v s))) v)))

;; Map's and Unfold's alternative arguments change their length, so that
;; the exact-discard laws see elements and steps dropped and added (the
;; post-M5 review's P2-10 gap). A planted drop-elements that leaves a
;; dropped element out of the discard fails
;; combinators-updates-are-reversible-with-exact-discards on them.
(def models
  [[:map mapped [[1.0 2.0 3.0]] [[[1.0 2.0]] [[1.0 2.0 3.0 4.0]] [[1.0 5.0 3.0]]]]

   [:map-2-args (comb/map (gen [x y] (trace :z (d/normal x y)))) [[0.0 1.0] [1.0 2.0]]
    [[[0.0] [1.0]] [[0.0 1.0 2.0] [1.0 2.0 3.0]]]]
   [:unfold unfolded [4 0.0 0.1] [[2 0.0 0.1] [6 0.0 0.1] [4 0.5 0.1]]]
   [:switch switched [1 0.5]]
   [:keyed-switch keyed [:a 0.5]]
   [:spliced spliced [3]]
   [:call-at (comb/call-at element) [1.0 :here] [[2.5 :here]]]
   [:call-at-a-path (comb/call-at unfolded) [3 0.0 0.1 [:walk 1]] [[4 0.0 0.1 [:walk 1]]]]
   [:choice-at (comb/choice-at d/normal) [0.0 1.0 :x] [[0.5 2.0 :x]]]
   ;; Switches whose branches are of different kinds (the post-M5 review's
   ;; I2-01): the shared addresses keep their values
   [:switch-of-kinds (comb/switch as-map as-dynamic) [0 [0.0 1.0]] [[1 [0.0 1.0]] [1 [0.5 1.0]]]]
   [:switch-of-a-distribution-and-a-map (comb/switch (d/normal 0.0 1.0) as-map) [0] [[1 [0.0]]]]
   [:switch-of-a-map-and-a-distribution (comb/switch as-map (d/normal 0.0 1.0)) [0 [0.0]] [[1]]]
   ;; elements and steps taken out of their support and brought back
   ;; (genclj-awd0)
   [:map-of-a-bounded-kernel (comb/map bounded) [[0.0 0.0]] [[[5.0 0.0]] [[0.0 -3.0]]]]
   [:unfold-of-a-bounded-step (comb/unfold (gen [_t _s lo] (splice :b bounded lo))) [2 0.0 0.0] [[2 0.0 5.0] [3 0.0 0.0]]]])

(laws/deflaws combinators models 100)
(laws/deflaws combinators models 100 laws/fixed-support)

;; A walk each of whose steps the last bounds (its supports are not
;; fixed): a step that leaves its support stops the Unfold there (the ten
;; decisions' fifth), as the next step's state would be its missing return
;; value. Planted bug (2026-10-25): combinators that run on past a part
;; that stopped (stopped? never true) fail these laws, and Scan's and
;; Recurse's (gen.combinators-extras-test, the recurse suite above).
(laws/deflaws walk [[:unfold-of-a-bounded-walk (comb/unfold bounded-walk) [3 0.0] [[3 0.5] [4 0.0] [2 0.0]]]] 100)

;; Recurse's own set: its trees change shape under update and regenerate
(laws/deflaws recurse [[:recurse tree [0] [[1]]]
                       [:recurse-spliced (gen [d] (let [s (splice :t tree d)] (trace :y (d/normal s 1)))) [0] [[1]]]]
  100)

;; ---------------------------------------------------------------------------
;; Map
;; ---------------------------------------------------------------------------

(deftest call-at-and-choice-at
  (testing "call-at puts the callee's choices under its last argument"
    (let [tr (gfi/simulate (comb/call-at element) [1.0 [:at 3]] (rng/key 1))]
      (is (every? #(= [:at 3] (subvec % 0 2)) (cm/addresses (:choices tr))))
      (is (close? (:score tr) (:weight (gfi/assess element [1.0] (cm/get-submap (cm/get-submap (:choices tr) :at) 3)))))
      (is (= (:retval tr) (tr [:at 3 :z])) "the callee's return value")))
  (testing "choice-at makes one choice at its last argument"
    (let [tr (gfi/simulate (comb/choice-at d/normal) [0.0 1.0 :x] (rng/key 2))]
      (is (= [[:x]] (cm/addresses (:choices tr))))
      (is (= (:retval tr) (tr :x)))))
  (testing "loud errors"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"distribution constructor" (comb/choice-at (d/normal 0 1))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"generative function" (comb/call-at 3)))))

;; Gen.jl's trace[addr] at a call and get_call(tr, addr).subtrace (M5's
;; G1), through every kind that makes calls
(deftest calls-by-address
  (let [tr (gfi/simulate spliced [3] (rng/key 5))
        xs (trace/subtrace tr :xs)]
    (is (= (ts/host (:retval xs)) (ts/host (tr :xs))) "a spliced Map's return value")
    (is (= (ts/host (tr [:xs 1 :z])) (ts/host (tr [:xs 1]))) "an element's, through the Map")
    (is (= (trace/subtrace xs 1) (trace/subtrace tr [:xs 1])) "a path continues in the call's trace")
    (is (= (ts/host (peek (:retval (trace/subtrace tr :path)))) (ts/host (tr [:path 2]))) "an Unfold step's")
    (is (= (ts/host (:retval tr)) (ts/host (tr []))) "the empty path: the trace's own")
    (is (= tr (trace/subtrace tr [])))
    (is (= :value (:reason (ts/thrown-data (trace/subtrace tr :mu)))) "a choice is not a call")
    (is (= :absent (:reason (ts/thrown-data (trace/subtrace tr [:xs 3])))) "nor an element past the end")
    (is (nil? (trace/subtrace tr :mu nil)) "unless given a not-found value")
    (is (= :absent (:reason (ts/thrown-data (tr :absent)))))
    (is (nil? (tr :absent nil)))
    (is (thrown? clojure.lang.ExceptionInfo (trace/subtrace tr [[:xs] 1])) "a vector inside a path")
    (is (thrown? clojure.lang.ExceptionInfo (tr [:xs [1]]))))
  (testing "a path continues past an element"
    (let [tr (gfi/simulate (comb/map (gen [x] (splice :inner element x) :done)) [[1.0 2.0]] (rng/key 9))]
      (is (= :done (tr 1)))
      (is (= (ts/host (tr [1 :inner :z])) (ts/host (tr [1 :inner]))))))
  (testing "a Switch's calls are its branch's"
    (let [tr (gfi/simulate (comb/switch (gen [x] (splice :inner element x)) a-branch) [0 1.0] (rng/key 6))]
      (is (= (ts/host (tr [:inner :z])) (ts/host (tr :inner))))
      (is (= element (:gen-fn (trace/subtrace tr :inner))))))
  (testing "a Recurse's at [node :production] and [node :aggregation]"
    (let [tr (gfi/simulate tree [0] (rng/key 7))]
      (is (= (ts/host (:retval tr)) (ts/host (tr [0 :aggregation]))) "the root's aggregation")
      (is (= grow (:gen-fn (trace/subtrace tr [0 :production]))))
      (is (= :node (:reason (ts/thrown-data (trace/subtrace tr [0])))) "a node is two calls, not one")))
  (testing "call-at's at its address"
    (let [tr (gfi/simulate (comb/call-at element) [1.0 :here] (rng/key 8))]
      (is (= (ts/host (:retval tr)) (ts/host (tr :here)))))))

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
        (is (close? w (- (:score t2) (:score tr))))
        (is (close? (:score t2) (:weight (gfi/assess mapped [[1.0 5.0 3.0]] (:choices t2)))))
        (is (diff/no-change? rd) "the return values did not change")))
    (testing "shrinking discards the removed elements"
      (let [{t2 :trace w :weight disc :discard rd :retdiff}
            (gfi/update tr [[1.0]] [diff/unknown-change] cm/EMPTY (rng/key 3))]
        (is (= [[0 :z]] (filter #(= :z (peek %)) (cm/addresses (:choices t2)))))
        (is (= (cm/get-submap (:choices tr) 1) (cm/get-submap disc 1)))
        (is (close? w (- (:score t2) (:score tr))))
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
        (is (close? w (- (:score t2) (:score tr))))
        (is (close? (:score t2) (:weight (gfi/assess unfolded [5 0.0 0.1] (:choices t2)))))
        (is (= #{1} (set (keys (:updated rd)))) "only the changed state is reported")))
    (testing "a parameter change revisits every step"
      (let [{t2 :trace w :weight} (gfi/update tr [5 0.0 0.7] [diff/no-change diff/no-change diff/unknown-change]
                                              cm/EMPTY (rng/key 4))]
        (is (not-any? true? (map identical? (subtraces tr) (subtraces t2))))
        (is (close? w (- (:score t2) (:score tr))))))
    (testing "regenerating one step resamples it and re-scores the rest"
      (let [{t2 :trace w :weight} (gfi/regenerate tr (sel/select [2 :s]) (rng/key 5))
            later (sel/select [2 :obs] [3 :s] [3 :obs] [4 :s] [4 :obs])]
        (is (not= (tr [2 :s]) (t2 [2 :s])))
        (is (= (tr [3 :s]) (t2 [3 :s])))
        (is (close? w (- (gfi/project t2 later) (gfi/project tr later))))))
    (testing "shrinking"
      (let [{t2 :trace w :weight disc :discard} (gfi/update tr [2 0.0 0.1] [diff/unknown-change diff/no-change diff/no-change]
                                                            cm/EMPTY (rng/key 6))]
        (is (= 2 (count (:retval t2))))
        (is (= #{2 3 4} (set (map first (cm/addresses disc)))))
        (is (close? w (- (:score t2) (:score tr))))))))

;; ---------------------------------------------------------------------------
;; A score that leaves its support and comes back (genclj-awd0)
;; ---------------------------------------------------------------------------

;; Found by the fuzzer: Map, Unfold and Scan keep their scores as a running
;; total, score + new - old, so an update that took an element out of its
;; support (-Inf) and one that brought it back left -Inf + Inf = NaN, where
;; the density is 0.0 (uniforms on [0, 1]). Planted bug (2026-10-24): the
;; running total unchecked (revised returning acc as it is) fails every
;; case, on the host and per particle.
(deftest a-score-that-leaves-its-support-and-comes-back-is-the-density
  (let [kinds [["Map" (comb/map bounded) (fn [lo] [[lo 0.0]])]
               ["Unfold" (comb/unfold (gen [_t _s lo] (trace :v (d/uniform lo (+ lo 1.0))))) (fn [lo] [2 0.0 lo])]
               ["Scan" (comb/scan (gen [c x] (let [v (trace :v (d/uniform x (+ x 1.0)))] [(+ c v) v]))) (fn [lo] [0.0 [lo 0.0]])]]
        unknown #(vec (repeat (count %) diff/unknown-change))
        moved (fn [tr args] (:trace (gfi/update tr args (unknown args) cm/EMPTY (rng/key 2))))]
    (doseq [[label gf args-at] kinds]
      (testing label
        (let [tr (:trace (gfi/generate gf (args-at 0.0) (cm/from-map {0 {:v 0.5} 1 {:v 0.5}}) (rng/key 1)))
              out (moved tr (args-at 5.0))
              back (moved out (args-at 0.0))]
          (is (= ##-Inf (ts/number (:score out))) "out of its support")
          (is (close? 0.0 (:score back)) "and back: the density, not NaN")
          (is (close? (:weight (gfi/assess gf (args-at 0.0) (:choices back))) (:score back))))
        (testing ", per particle"
          ;; particle i's first choice at 0.5 or 0.9: lo 0.6 takes only the
          ;; 0.5s out of their support
          (binding [b/*spot-checks* 0]
            (let [own (mapv #(cm/from-map {0 {:v (if (even? %) 0.5 0.9)} 1 {:v 0.7}}) (range 6))
                  ps (:particles (b/generate gf (args-at 0.0) cm/EMPTY 6 (rng/key 3) {:per-particle own}))
                  step (fn [ps args] (:particles (b/update ps args (unknown args) cm/EMPTY (rng/key 4))))
                  out (step ps (args-at 0.6))
                  back (step out (args-at 0.0))]
              (is (b/batched? back))
              (is (some #(= ##-Inf %) (array/->clj (b/scores out))))
              (is (some ts/finite? (array/->clj (b/scores out))))
              (is (every? true? (map #(close? %1 (:weight (gfi/assess gf (args-at 0.0) (:choices %2))))
                                     (array/->clj (b/scores back)) (b/traces back)))))))))))

;; Element scores at +Inf (beta(a, 1) at 0 for a < 1) and NaN (+Inf beside
;; -Inf), in and out (genclj-ul72): the score is the IEEE sum of the
;; element scores at every stage. A choice out of its support stops its
;; element's run (the ten decisions' fifth, DESIGN.md §9): that element's
;; score is -Inf, even beside a +Inf choice, Unfold and Scan stop at it
;; (the last stage, which drops step 1), and Map runs on. So an element is
;; NaN only as a sum of its own, a Map's (the Map of Maps). Planted bugs
;; (2026-10-24): the counts never decremented fails every kind's stages
;; after a score left an infinity, on the host and per particle (and the
;; test above); a NaN counted only as -Inf fails the stage with a NaN
;; element (2026-10-25: the Map of Maps' at [0.5 0.0 5.0]).
(deftest a-score-at-infinity-or-nan-is-the-sum-of-the-element-scores
  (let [kernel (gen [a lo] (trace :p (d/beta a 1.0)) (trace :u (d/uniform lo (+ lo 1.0))))
        kinds [["Map" (comb/map kernel) (fn [a lo0 lo1] [[a a] [lo0 lo1]]) identity]
               ["Unfold" (comb/unfold (gen [t _s a lo0 lo1]
                                        (let [lo (if (zero? t) lo0 lo1)]
                                          (trace :p (d/beta a 1.0))
                                          (trace :u (d/uniform lo (+ lo 1.0))))))
                (fn [a lo0 lo1] [2 0.0 a lo0 lo1]) identity]
               ["Scan" (comb/scan (gen [c lo a] (trace :p (d/beta a 1.0)) [c (trace :u (d/uniform lo (+ lo 1.0)))]))
                (fn [a lo0 lo1] [0.0 [lo0 lo1] a]) identity]
               ;; one element, a Map of the two: NaN where they are +Inf and -Inf
               ["Map of Maps" (comb/map (comb/map kernel)) (fn [a lo0 lo1] [[[a 1.0]] [[lo0 lo1]]]) #(hash-map 0 %)]]
        ;; [a lo0 lo1] and the score, element 0's :p at 0 and element 1's at 0.5
        stages [[[1.0 0.0 0.0] 0.0]
                [[0.5 0.0 0.0] ##Inf]
                [[0.5 0.0 5.0] ##NaN]
                [[1.0 0.0 5.0] ##-Inf]
                [[1.0 0.0 0.0] 0.0]
                [[0.5 0.0 0.0] ##Inf]
                [[0.5 5.0 0.0] ##-Inf]]
        same? #(ts/close? %1 %2 {:equal-nan? true})
        unknown #(vec (repeat (count %) diff/unknown-change))]
    (doseq [[label gf args-at nest] kinds
            :let [at (fn [[a lo0 lo1]] (args-at a lo0 lo1))
                  first-args (at (ffirst stages))]]
      (testing label
        (let [tr (:trace (gfi/generate gf first-args (cm/from-map (nest {0 {:p 0.0 :u 0.5} 1 {:p 0.5 :u 0.5}})) (rng/key 1)))]
          (reduce (fn [tr [params score]]
                    (let [args (at params)
                          t2 (:trace (gfi/update tr args (unknown args) cm/EMPTY (rng/key 2)))]
                      (is (same? score (:score t2)) (str "at " params))
                      (is (same? (:weight (gfi/assess gf args (:choices t2))) (:score t2)) (str "assessed, at " params))
                      t2))
                  tr stages))
        (testing ", per particle"
          ;; particle i's element 0 at :p 0 (even i) or 0.5 (odd)
          (binding [b/*spot-checks* 0]
            (let [own (mapv #(cm/from-map (nest {0 {:p (if (even? %) 0.0 0.5) :u 0.5} 1 {:p 0.5 :u 0.5}})) (range 4))
                  ps (:particles (b/generate gf first-args cm/EMPTY 4 (rng/key 3) {:per-particle own}))]
              (reduce (fn [ps params]
                        (let [args (at params)
                              out (:particles (b/update ps args (unknown args) cm/EMPTY (rng/key 4)))]
                          (is (b/batched? out))
                          (is (every? true? (map #(same? %1 (:weight (gfi/assess gf args (:choices %2))))
                                                 (array/->clj (b/scores out)) (b/traces out)))
                              (str "at " params))
                          out))
                      ps (map first stages)))))))))

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
    (close? w (- (- (:score t2) (:score tr)) fresh))))

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
    (testing "changing the branch: the new branch's choices are generated, the old ones discarded"
      (let [{t2 :trace w :weight disc :discard}
            (gfi/update tr [1 0.5] [diff/unknown-change diff/no-change] (cm/choicemap :b 1.5) (rng/key 2))]
        (is (= 1.5 (t2 :b)))
        (is (= (:choices tr) disc))
        (is (close? w (- (- (:score t2) (:score tr)) (gfi/project t2 (sel/select :c)))))
        (testing "and updating back with the discard restores the trace"
          (let [{t3 :trace w3 :weight} (gfi/update t2 [0 0.5] [diff/unknown-change diff/no-change] disc (rng/key 3))]
            (is (= (:choices tr) (:choices t3)))
            (is (close? w3 (- (:score tr) (:score t2))))))))
    (testing "regenerating into a new branch has weight zero"
      (is (zero? (ts/number (:weight (gfi/regenerate tr [1 0.5] [diff/unknown-change diff/no-change] sel/none (rng/key 4)))))))
    (testing "the same branch with a new argument re-scores"
      (let [{t2 :trace w :weight} (gfi/update tr [0 2.0] [diff/no-change diff/unknown-change] cm/EMPTY (rng/key 5))]
        (is (= (tr :a) (t2 :a)))
        (is (close? w (- (:score t2) (:score tr))))))
    (is (= [[:b] [:c]] (cm/addresses (:choices (gfi/simulate keyed [:b 0.0] (rng/key 6))))))))

(deftest errors-are-loud
  (let [k (rng/key 0)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one length" (gfi/simulate (comb/map step) [[1] [1 2] [3]] k)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Map's arguments are vectors" (gfi/simulate (comb/map step) [3 1 2] k)))
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
    (is (close? w (- (:score t2) (:score tr))))
    (is (close? w3 w))
    (is (not (zero? (ts/number w))))))

(deftest switch-branches-share-addresses
  ;; Def 2.3.1: a choice both branches make keeps its value across an index
  ;; change, exactly as with the equivalent if
  (let [sw (comb/switch (gen [] (trace :x (d/normal 0 1))) (gen [] (trace :x (d/normal 5 1))))
        as-if (gen [i] (if (zero? i) (trace :x (d/normal 0 1)) (trace :x (d/normal 5 1))))
        tr (:trace (gfi/generate sw [0] (cm/choicemap :x 0.3) (rng/key 1)))
        tr-if (:trace (gfi/generate as-if [0] (cm/choicemap :x 0.3) (rng/key 1)))
        r (gfi/update tr [1] [diff/unknown-change] cm/EMPTY (rng/key 2))
        r-if (gfi/update tr-if [1] [diff/unknown-change] cm/EMPTY (rng/key 2))]
    (is (= 0.3 ((:trace r) :x)))
    (is (close? (:weight r) (:weight r-if)))
    (is (close? (:weight r) (- (dc/log-prob (d/normal 5 1) 0.3) (dc/log-prob (d/normal 0 1) 0.3))))
    (is (= (:discard r) (:discard r-if)))
    (testing "and regenerate keeps it unless selected"
      (let [{t :trace w :weight} (gfi/regenerate tr [1] [diff/unknown-change] sel/none (rng/key 3))]
        (is (= 0.3 (t :x)))
        (is (close? w (- (dc/log-prob (d/normal 5 1) 0.3) (dc/log-prob (d/normal 0 1) 0.3))))))))

(deftest a-callee-that-changes-kind-keeps-the-shared-addresses
  ;; The post-M5 review's I2-01 (and its verifier's V-I2-01, -01b, -01c): a
  ;; class rule decided before gfi/revise, so a callee of another kind lost
  ;; every address the two share, and regenerate with nothing selected
  ;; redrew them. Planted, that rule fails: in a Switch (a new branch of
  ;; another kind generated afresh), the first and third testings and five
  ;; laws on :switch-of-kinds; in a splice (an old call of another kind
  ;; replaced), the second.
  (let [def-2-3-1 (fn [tr {t :trace w :weight d :discard}]
                    (is (= (tr [0 :x]) (t [0 :x])) "[0 :x] keeps its value")
                    (when d (is (cm/empty? d) "nothing is discarded"))
                    (is (close? w (- (:score t) (:score tr))) "the weight is the change of score"))]
    (doseq [[label a b] [["a Map, then a dynamic function" as-map as-dynamic]
                         ["a dynamic function, then a Map" as-dynamic as-map]
                         ["a Map, then an Unfold" as-map as-unfold]
                         ["an Unfold, then a Map" as-unfold as-map]]]
      (testing label
        (let [sw (comb/switch a b)
              args (fn [i g] (if (= g as-unfold) [i 1 0.0] [i [0.0]]))
              tr (gfi/simulate sw (args 0 a) (rng/key 70))
              argdiffs (vec (repeat (count (args 1 b)) diff/unknown-change))]
          (def-2-3-1 tr (gfi/update tr (args 1 b) argdiffs cm/EMPTY (rng/key 71)))
          (def-2-3-1 tr (gfi/regenerate tr (args 1 b) argdiffs sel/none (rng/key 72)))))))
  (testing "a splice whose callee changes kind"
    (let [m (gen [k] (splice :s (if (zero? k) as-dynamic as-map) [0.0]))
          tr (gfi/simulate m [0] (rng/key 73))
          {t :trace d :discard} (gfi/update tr [1] [diff/unknown-change] cm/EMPTY (rng/key 74))]
      (is (= (tr [:s 0 :x]) (t [:s 0 :x])))
      (is (cm/empty? d))))
  (testing "observations stay through a Switch of a Map and a dynamic function (V-I2-01b)"
    (let [ys (gen [mu] (trace :y (d/normal mu 1)))
          sw (comb/switch (comb/map ys) (gen [mus] (mapv (fn [i mu] (splice i ys (+ mu 1.0))) (range (count mus)) mus)))
          m (gen [] (splice :s sw (trace :i (d/categorical [0.5 0.5])) [0.0 0.0 0.0]))
          tr (:trace (gfi/generate m [] (cm/choicemap :i 0 [:s 0 :y] 1.5 [:s 1 :y] 1.7 [:s 2 :y] 2.2) (rng/key 75)))
          {t :trace w :weight} (gfi/update tr (cm/choicemap :i 1) (rng/key 76))]
      (is (= [1.5 1.7 2.2] (mapv #(t [:s % :y]) (range 3))))
      (is (close? w (reduce + (map #(- (dc/log-prob (d/normal 1 1) %) (dc/log-prob (d/normal 0 1) %)) [1.5 1.7 2.2]))))))
  (testing "a distribution's value and indexed choices share no address, and neither is an error (V-I2-01c)"
    (doseq [[label sw args args'] [["a normal, then a Map" (comb/switch (d/normal 0.0 1.0) as-map) [0] [1 [0.0]]]
                                   ["a normal, then an Unfold" (comb/switch (d/normal 0.0 1.0) as-unfold) [0] [1 1 0.0]]
                                   ["a Map, then a normal" (comb/switch as-map (d/normal 0.0 1.0)) [0 [0.0]] [1]]]]
      (testing label
        (let [tr (gfi/simulate sw args (rng/key 77))
              {t :trace w :weight d :discard} (gfi/update tr args' (vec (repeat (count args') diff/unknown-change)) cm/EMPTY
                                                          (rng/key 78))]
          (is (= (:choices tr) d) "the old choices are discarded")
          (is (close? w (- (:score tr))) "the new ones are drawn from the prior"))))))

(def ^:private second-or-first
  "A kernel whose mean is its second argument, when it has one, else its first."
  (gen [a & [b]] (trace :x (d/normal (if b b a) 1))))

(def ^:private stepping
  "An Unfold kernel whose mean is its second parameter, when it has one."
  (gen [_t s a & [b]] (trace :x (d/normal (if b b a) 1)) s))

(deftest a-change-in-the-number-of-arguments-is-a-change
  ;; genclj-5wy4 (DESIGN.md §9, G3): no argdiff can say an argument went or
  ;; came, so Map's and Unfold's revise kept every element with no-change
  ;; argdiffs: weight 0.0 where 30.0 is right, and the old score. Plant:
  ;; without the count check, every weight here is 0.0
  (let [lp (fn [mu] (dc/log-prob (d/normal mu 1) 0.5))
        obs (cm/from-map {0 {:x 0.5} 1 {:x 0.5} 2 {:x 0.5}})
        nc diff/no-change]
    (doseq [[label gf before after] [["Map" (comb/map second-or-first) [[1.0 1.0 1.0] [5.0 5.0 5.0]] [[1.0 1.0 1.0]]]
                                     ["Unfold" (comb/unfold stepping) [3 0.0 1.0 5.0] [3 0.0 1.0]]]
            [change from to expected] [["one fewer" before after (* 3 (- (lp 1.0) (lp 5.0)))]
                                       ["one more" after before (* 3 (- (lp 5.0) (lp 1.0)))]]]
      (testing (str label ", " change)
        (let [tr (:trace (gfi/generate gf from obs (rng/key 1)))
              ncs (vec (repeat (count to) nc))
              {u :trace uw :weight} (gfi/update tr to ncs cm/EMPTY (rng/key 2))
              {rw :weight} (gfi/regenerate tr to ncs sel/none (rng/key 3))]
          (is (close? expected uw))
          (is (close? expected rw) "regenerate")
          (is (close? (:weight (gfi/assess gf to (:choices u))) (:score u)) "the score is the new arguments'"))))))

(deftest switches-of-compatible-branches-keep-shared-addresses
  ;; third review B2: a Switch of two Maps (or two distributions) regenerated
  ;; at its index alone resampled every unselected choice, with weight 0
  (let [ys (gen [mu] (trace :y (d/normal mu 1)))
        sw (comb/switch (comb/map ys) (comb/map (gen [mu] (trace :y (d/normal (+ mu 1.0) 1)))))
        m (gen [] (splice :s sw (trace :i (d/categorical [0.5 0.5])) [0.0 0.0 0.0]))
        obs (cm/choicemap :i 0 [:s 0 :y] 1.5 [:s 1 :y] 1.7 [:s 2 :y] 2.2)
        tr (:trace (gfi/generate m [] obs (rng/key 1)))
        {t :trace w :weight} (gfi/update tr (cm/choicemap :i 1) (rng/key 2))]
    (is (= [1.5 1.7 2.2] (mapv #(t [:s % :y]) (range 3))) "the observations stay")
    (is (close? w (reduce + (map #(- (dc/log-prob (d/normal 1 1) %) (dc/log-prob (d/normal 0 1) %)) [1.5 1.7 2.2])))))
  (let [sw (comb/switch (d/normal 0 1) (d/normal 5 1))
        tr (:trace (gfi/generate sw [0] (cm/->Value 0.3) (rng/key 3)))
        {t :trace w :weight} (gfi/regenerate tr [1] [diff/unknown-change] sel/none (rng/key 4))]
    (is (= 0.3 (:retval t)) "two distributions: the value stays")
    (is (close? w (- (dc/log-prob (d/normal 5 1) 0.3) (dc/log-prob (d/normal 0 1) 0.3))))))

;; ---------------------------------------------------------------------------
;; Recurse
;; ---------------------------------------------------------------------------

(deftest recurse-deep-nodes-draw-afresh
  ;; The post-M5 review's P2-09 (critical): node numbers pass 2^32 at depth
  ;; 32 (max-branch 2), 16 (4) or 11 (8), also across subtrees. With
  ;; fold-in masking its data to 32 bits, such nodes shared one key and
  ;; drew one value (a 40-deep chain had one value at depths 32 to 40).
  ;; Planted, the mask fails every case here.
  (doseq [[m depth from] [[2 40 32] [4 20 16] [8 14 11]]]
    (testing (str "a chain of child 0, max-branch " m)
      (let [chain (comb/recurse (gen [n] {:value (trace :v (d/normal 0 1)) :children (if (< n depth) [(inc n)] [])})
                                (gen [v ws] (into [v] (first ws)))
                                m)
            vs (ts/host (:retval (gfi/simulate chain [0] (rng/key 60))))]
        (is (= (inc depth) (count vs)))
        (is (= (- (inc depth) from) (count (distinct (subvec vs from)))) (str "depths " from " to " depth " differ")))))
  (testing "two subtrees whose depth-17 nodes are equal mod 2^32 (max-branch 4)"
    (let [tree (comb/recurse (gen [[kind n]]
                               {:value (trace :v (d/normal 0 1))
                                :children (case kind
                                            :root [[:chain 1] [:chain 1]]
                                            :chain (if (< n 17) [[:leaf] [:leaf] [:leaf] [:chain (inc n)]] [])
                                            :leaf [])})
                             (gen [v _] v)
                             4)
          tr (gfi/simulate tree [[:root]] (rng/key 61))
          ;; the subtrees' roots are nodes 1 and 2; child 3 of node i is 4i + 4
          deep (fn [start] (nth (iterate #(+ (* 4 %) 4) start) 16))]
      (is (= (mod (deep 1) (bit-shift-left 1 32)) (mod (deep 2) (bit-shift-left 1 32))))
      (is (not= (ts/host (tr [(deep 1) :production :v])) (ts/host (tr [(deep 2) :production :v])))))))

(deftest recurse-builds-and-folds-the-tree
  (let [counted (comb/recurse (gen [n] {:value n :children (if (< n 2) [(inc n) (inc n)] [])})
                              (gen [v ws] (reduce + v ws))
                              2)]
    (is (= 10 (:retval (gfi/assess counted [0] cm/EMPTY)) (:retval (gfi/simulate counted [0] (rng/key 1)))) "0 + 2·1 + 4·2")
    (is (= 10 (counted 0)) "an untraced call"))
  (testing "node indices: the root 0, child j of node i at i·max-branch + j + 1"
    (let [tr (:trace (gfi/generate tree [0] (cm/choicemap [0 :production :split] true [1 :production :split] true
                                                           [2 :production :split] false)
                                   (rng/key 2)))
          nodes (set (keys (cm/to-map (:choices tr))))]
      (is (= #{0 1 2 3 4} nodes))
      (is (ts/close? (:retval tr) (reduce + (map #(tr [% :production :v]) nodes))))
      (is (ts/close? (+ (tr [1 :production :v]) (tr [3 :production :v]) (tr [4 :production :v]))
                     (:retval ((:gen.combinators/aggregations tr) 1)))
          "node 1 folds its value and its children's")))
  (testing "max-branch is enforced"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"more than max-branch"
                          (gfi/simulate (comb/recurse (gen [_] {:value 0 :children [1 2 3]}) fold 2) [0] (rng/key 3))))))

(defspec recurse-numbering-inverts-get-child 200
  (prop/for-all [max-branch (gen/choose 1 6) parent gen/nat j gen/nat]
    (let [j (mod j max-branch)
          child (comb/get-child parent j max-branch)]
      (and (= parent (comb/get-parent child max-branch))
           (= j (comb/get-child-num child max-branch))))))

(deftest recurse-numbering-is-loud
  (doseq [[why f] [["a child past max-branch" #(comb/get-child 0 2 2)]
                   ["a negative node" #(comb/get-child -1 0 2)]
                   ["max-branch 0" #(comb/get-child 0 0 0)]
                   ["the root's parent" #(comb/get-parent 0 2)]
                   ["the root's child number" #(comb/get-child-num 0 2)]
                   ["a node that is not an integer" #(comb/get-parent 1.0 2)]]]
    (is (thrown? clojure.lang.ExceptionInfo (f)) why)))

(deftest recurse-updates-only-the-changed-path
  (let [tr (:trace (gfi/generate tree [0] (cm/choicemap [0 :production :split] true [1 :production :split] true
                                                         [2 :production :split] true)
                                 (rng/key 4)))
        {t' :trace w :weight d :discard} (gfi/update tr [0] [diff/no-change] (cm/choicemap [3 :production :v] 0.25) (rng/key 5))
        prods (fn [t] (:gen.combinators/productions t))
        aggs (fn [t] (:gen.combinators/aggregations t))]
    (is (ts/close? 0.25 (t' [3 :production :v])))
    (is (= {3 {:production {:v (tr [3 :production :v])}}} (cm/to-map d)))
    (is (ts/close? (- (:score t') (:score tr)) w))
    (testing "nodes off the path from 3 to the root keep their subtraces"
      (doseq [n [0 1 2 4 5 6]] (is (identical? ((prods tr) n) ((prods t') n)) (str "production " n)))
      (doseq [n [2 4 5 6]] (is (identical? ((aggs tr) n) ((aggs t') n)) (str "aggregation " n)))
      (doseq [n [0 1 3]] (is (not (identical? ((aggs tr) n) ((aggs t') n))) (str "aggregation " n " revisited"))))
    (testing "pruning a subtree discards it"
      (let [{t'' :trace w :weight d :discard} (gfi/update tr [0] [diff/no-change] (cm/choicemap [1 :production :split] false)
                                                          (rng/key 6))]
        (is (= #{0 1 2 5 6} (set (keys (cm/to-map (:choices t''))))))
        (is (= #{1 3 4} (set (keys (cm/to-map d)))))
        (is (ts/close? (- (:score t'') (:score tr)) w) "the update samples nothing")))
    (testing "constraints off the tree are loud"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"a value at node 0"
                            (gfi/generate tree [0] (cm/choicemap 0 1.0) (rng/key 9))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                            (gfi/update tr [0] [diff/no-change] (cm/choicemap [9 :production :v] 1.0) (rng/key 7))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not visited"
                            (gfi/generate tree [0] (cm/choicemap [0 :elsewhere] 1.0) (rng/key 8)))))))

;; ---------------------------------------------------------------------------
;; Combinators against their bodies written inline (Prop. 2.3.2; §5.3,
;; pp. 176-177; law audit gap G)
;; ---------------------------------------------------------------------------

(def inline-map (gen [xs] (mapv (fn [i x] (splice i element x)) (range (count xs)) xs)))

(def inline-unfold
  (gen [n init drift]
    (loop [t 0 state init acc []]
      (if (< t n)
        (let [s (splice t step t state drift)] (recur (inc t) s (conj acc s)))
        acc))))

(def inline-node
  "Recurse's node as the thesis's recursive function (p. 177), its choices
   at Recurse's addresses."
  (gen [input node]
    (let [{:keys [value children]} (splice [node :production] grow input)
          ws (mapv (fn [j c] (splice inline-node c (+ (* node 2) j 1))) (range (count children)) children)]
      (splice [node :aggregation] fold value ws))))

(deftest combinators-match-their-bodies-inlined
  (doseq [[label gf args inline inline-args moves]
          [[:map mapped [[1.0 2.0 3.0]] inline-map [[1.0 2.0 3.0]]
            [[[[1.0 2.0 3.0]] (cm/choicemap [1 :z] 0.4)] [[[1.0 2.0]] cm/EMPTY] [[[1.0 5.0 3.0]] (cm/choicemap [0 :z] -1.0)]]]
           [:unfold unfolded [4 0.0 0.1] inline-unfold [4 0.0 0.1]
            [[[4 0.0 0.1] (cm/choicemap [1 :s] 0.3)] [[2 0.0 0.1] cm/EMPTY] [[4 0.5 0.2] (cm/choicemap [3 :obs] 1.0)]]]
           [:recurse tree [0] inline-node [0 0]
            [[[0] (cm/choicemap [0 :production :v] 0.25)] [[1] cm/EMPTY] [[0] (cm/choicemap [0 :production :split] false)]]]]]
    (testing (name label)
      (doseq [seed (range 4)]
        (let [tr (gfi/simulate gf args (rng/key seed))
              ti (:trace (gfi/generate inline inline-args (:choices tr) (rng/key 99)))]
          (is (= (:choices tr) (:choices ti)))
          (is (close? (:score tr) (:score ti)) "the same density")
          (doseq [[args' sigma] moves]
            (let [inline-args' (if (= label :recurse) (conj (vec args') 0) args')
                  r (gfi/update tr args' (vec (repeat (count args') diff/unknown-change)) sigma (rng/key 5))
                  ri (gfi/update ti inline-args' (vec (repeat (count inline-args') diff/unknown-change)) sigma (rng/key 5))]
              (when (= (set (cm/addresses (:choices (:trace r)))) (set (cm/addresses (:choices (:trace ri)))))
                ;; the same structure: no fresh choices drawn differently
                (is (close? (:weight r) (:weight ri)) (str label " " args'))
                (is (= (:discard r) (:discard ri)))))))))))

;; ---------------------------------------------------------------------------
;; Elements are independent (law audit gap 1)
;; ---------------------------------------------------------------------------

(deftest ^:slow ^:heavy elements-are-independent
  ;; each element, step and node draws from its own key: the sample
  ;; correlation of two is within 5/sqrt(n) of 0. Planted bug (audit,
  ;; 2026-10-20): an element key that ignores the index passed every other
  ;; test; the correlation is then 1.0
  (let [n 2000
        keys (rng/split (rng/key 90) n)
        band (clojure.core// 5.0 (Math/sqrt n))
        independent? (fn [pairs] (< (Math/abs (ts/correlation pairs)) band))
        at (fn [tr p] (ts/number (tr p)))]
    (testing "Map"
      (let [m (comb/map (gen [x] (trace :z (d/normal x 1))))]
        (is (independent? (map #(let [tr (gfi/simulate m [[0.0 0.0]] %)] [(at tr [0 :z]) (at tr [1 :z])]) keys)))))
    (testing "Map's unconstrained elements in generate"
      (let [m (comb/map (gen [x] (trace :z (d/normal x 1))))]
        (is (independent? (map #(let [tr (:trace (gfi/generate m [[0.0 0.0 0.0]] (cm/choicemap [2 :z] 1.0) %))]
                                  [(at tr [0 :z]) (at tr [1 :z])])
                               keys)))))
    (testing "Unfold's innovations"
      (let [u (comb/unfold (gen [_t s] (trace :s (d/normal (* 0.5 s) 1))))]
        (is (independent? (map #(let [tr (gfi/simulate u [2 0.0] %)]
                                  [(at tr [0 :s]) (clojure.core/- (at tr [1 :s]) (clojure.core/* 0.5 (at tr [0 :s])))])
                               keys)))))
    (testing "Recurse's nodes"
      (let [r (comb/recurse (gen [depth] {:value (trace :v (d/normal 0 1)) :children (if (zero? depth) [1 1] [])})
                            (gen [v ws] (reduce + v ws))
                            2)]
        (is (independent? (map #(let [tr (gfi/simulate r [0] %)] [(at tr [1 :production :v]) (at tr [2 :production :v])]) keys)))))))

(deftest ^:slow ^:heavy elements-an-update-adds-are-independent
  ;; the elements, steps and nodes an update adds draw from keys of their
  ;; own (the final law audit, F9: giving them all one key passed every
  ;; other test; the correlation is then 1.0)
  (let [n 2000
        keys (rng/split (rng/key 92) n)
        band (clojure.core// 5.0 (Math/sqrt n))
        independent? (fn [pairs] (< (Math/abs (ts/correlation pairs)) band))
        at (fn [tr p] (ts/number (tr p)))
        grown (fn [gf args args' argdiffs pick]
                (map #(let [[k1 k2] (rng/split %)
                            tr (gfi/simulate gf args k1)]
                        (pick (:trace (gfi/update tr args' argdiffs cm/EMPTY k2))))
                     keys))]
    (testing "Map: two elements an update adds"
      (let [m (comb/map (gen [x] (trace :z (d/normal x 1))))]
        (is (independent? (grown m [[0.0]] [[0.0 0.0 0.0]] [diff/unknown-change] #(vector (at % [1 :z]) (at % [2 :z])))))))
    (testing "Unfold: the innovations of two steps an update adds"
      (let [u (comb/unfold (gen [_t s] (trace :s (d/normal (* 0.5 s) 1))))]
        (is (independent? (grown u [1 0.0] [3 0.0] [diff/unknown-change diff/no-change]
                                 #(let [s0 (at % [0 :s]) s1 (at % [1 :s]) s2 (at % [2 :s])]
                                    [(clojure.core/- s1 (clojure.core/* 0.5 s0)) (clojure.core/- s2 (clojure.core/* 0.5 s1))]))))))
    (testing "Recurse: two nodes an update adds"
      ;; a root of depth 1 has no children; of depth 0, two
      (let [r (comb/recurse (gen [depth] {:value (trace :v (d/normal 0 1)) :children (if (zero? depth) [1 1] [])})
                            (gen [v ws] (reduce + v ws))
                            2)]
        (is (independent? (grown r [1] [0] [diff/unknown-change] #(vector (at % [1 :production :v]) (at % [2 :production :v])))))))))

;; ---------------------------------------------------------------------------
;; Precise hints for Switch and Recurse (law audit gap 9)
;; ---------------------------------------------------------------------------

(deftest switch-and-recurse-precise-hints-match-unknown-change
  ;; an update with no-change argdiffs and a constraint gives what the same
  ;; update with unknown-change gives
  (doseq [seed (range 8)]
    (let [k (rng/key (+ 91 seed))
          sw (gfi/simulate switched [0 0.5] (rng/key seed))
          tree-tr (gfi/simulate tree [0] (rng/key seed))]
      (is (same-result? (gfi/update sw [0 0.5] [diff/no-change diff/no-change] (cm/choicemap :a 0.3) k)
                        (gfi/update sw [0 0.5] [diff/unknown-change diff/unknown-change] (cm/choicemap :a 0.3) k)))
      (is (same-result? (gfi/update tree-tr [0] [diff/no-change] (cm/choicemap [0 :production :v] 0.7) k)
                        (gfi/update tree-tr [0] [diff/unknown-change] (cm/choicemap [0 :production :v] 0.7) k)))))
  (testing "a Switch of Map branches, whose argdiffs decide what the branch skips (the final law audit: dynamic branches skip nothing, so the test above cannot see a branch given wrong argdiffs)"
    (let [branch (comb/map (gen [x] (trace :z (d/normal x 1))))
          sw (comb/switch branch branch)
          args [0 [0.0 1.0 2.0]]
          args' [0 [0.0 1.5 2.0]]
          precise (mapv diff/compute args args')]
      (doseq [seed (range 8)]
        (let [tr (gfi/simulate sw args (rng/key seed))
              k (rng/key (+ 99 seed))
              r (gfi/update tr args' precise cm/EMPTY k)
              assessed (:weight (gfi/assess sw args' (:choices (:trace r)) (rng/key 0)))]
          (is (same-result? r (gfi/update tr args' [diff/unknown-change diff/unknown-change] cm/EMPTY k)))
          ;; and right: a Switch that told its branch no-change would leave
          ;; element 1 scored at its old mean in both runs
          (is (close? assessed (:score (:trace r))))
          (is (close? (- assessed (:score tr)) (:weight r))))))))
