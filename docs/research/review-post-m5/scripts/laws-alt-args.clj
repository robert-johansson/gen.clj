;; Coverage probe (not a finding): the GFI laws on Map, Unfold and Switch with
;; changed arguments (length changes, a new index), which the suite's
;; combinator models leave out (their entries have no alt args).
(require '[gen.laws :as laws] '[gen.combinators :as comb] '[gen.dist :as d] '[gen.dynamic :refer [gen]]
         '[gen.array :as array] '[clojure.test.check :as tc])

(def element (gen [x] (let [z (trace :z (d/normal x 1))]
                        (when (trace :flip (d/bernoulli 0.5)) (trace :extra (d/gamma 2 1)))
                        z)))
(def step (gen [t state drift] (let [s (trace :s (d/normal (array/+ state drift) 1))] (trace :obs (d/normal s 0.5)) s)))
(def a-branch (gen [mu] (trace :a (d/normal mu 1))))
(def b-branch (gen [mu] (array/+ (trace :b (d/gamma 2 1)) (trace :c (d/poisson 3.0)))))
(def a2 (gen [mu] (let [a (trace :a (d/normal (array/* 2 mu) 1))] (trace :d (d/normal a 1)))))
(def grow (gen [depth] (let [split? (trace :split (d/bernoulli (if (< depth 2) 0.6 0.0)))]
                         {:value (trace :v (d/normal depth 1)) :children (if split? [(inc depth) (inc depth)] [])})))
(def fold (gen [v ws] (let [s (reduce array/+ v ws)] (trace :obs (d/normal s 1)) s)))

(def models
  [[:map (comb/map element) [[1.0 2.0 3.0]] [[[1.0 2.5]] [[1.0 2.0 3.0 4.0 5.0]] [[0.0 2.0 3.0]] [[]]]]
   [:map-of-dist (comb/map (gen [x] (trace :z (d/normal x 1)))) [[0.0 1.0]] [[[0.0]] [[0.0 1.0 2.0]]]]
   [:unfold (comb/unfold step) [4 0.0 0.1] [[2 0.0 0.1] [6 0.0 0.1] [4 0.5 0.1] [4 0.0 -0.3] [0 0.0 0.1]]]
   [:switch (comb/switch a-branch b-branch) [1 0.5] [[0 0.5] [1 1.5] [0 2.0]]]
   [:switch-compatible (comb/switch a-branch a2) [0 0.5] [[1 0.5] [1 1.5] [0 2.0]]]
   [:switch-of-maps (comb/switch (comb/map (gen [x] (trace :z (d/normal x 1)))) (comb/map element)) [0 [0.0 1.0]] [[1 [0.0 1.0]] [1 [0.0]] [0 [0.0 1.0 2.0]]]]
   [:recurse (comb/recurse grow fold 2) [0] [[1] [2]]]])

(doseq [[law f] (sort-by key laws/all)]
  (let [r (tc/quick-check 150 (f (laws/cases models)) :seed 42)]
    (println (if (:pass? r) "pass" "FAIL") (name law) (when-not (:pass? r) (pr-str (-> r :shrunk :smallest))))))
