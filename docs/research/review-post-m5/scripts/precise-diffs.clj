;; Coverage probe (not a finding): precise argdiffs (diff/compute) give the
;; same update and regenerate as unknown-change, with constraints and
;; selections, for every combinator
(require '[gen.combinators :as comb] '[gen.dist :as d] '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi]
         '[gen.diff :as diff] '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.random :as rng]
         '[gen.array :as array])

(def element (gen [x] (let [z (trace :z (d/normal x 1))]
                        (when (trace :flip (d/bernoulli 0.5)) (trace :extra (d/gamma 2 1)))
                        z)))
(def step (gen [t state drift] (let [s (trace :s (d/normal (array/+ state drift) 1))] (trace :obs (d/normal s 0.5)) (array/round s))))
(def grow (gen [depth] (let [split? (trace :split (d/bernoulli (if (< depth 2) 0.6 0.0)))]
                         {:value (trace :v (d/normal depth 1)) :children (if split? [(inc depth) (inc depth)] [])})))
(def fold (gen [v ws] (let [s (reduce array/+ v ws)] (trace :obs (d/normal s 1)) (array/round s))))
(def models
  [[(comb/map element) [[1.0 2.0 3.0]] [[[1.0 2.5 3.0]] [[1.0 2.0]] [[1.0 2.0 3.0 4.0]] [[1.0 2.0 3.0]]]]
   [(comb/unfold step) [4 0.0 0.1] [[4 0.0 0.1] [4 0.5 0.1] [2 0.0 0.1] [6 0.0 0.1] [4 0.0 0.2]]]
   [(comb/switch (comb/map element) (comb/map element)) [0 [0.0 1.0 2.0]] [[0 [0.0 1.5 2.0]] [0 [0.0 1.0]] [1 [0.0 1.0 2.0]]]]
   [(comb/recurse grow fold 2) [0] [[0] [1]]]])

(def failures (atom []))
(defn close [x y] (let [x (double (array/item x)) y (double (array/item y))] (or (= x y) (<= (Math/abs (- x y)) (* 1e-12 (max 1.0 (Math/abs x)))))))
(doseq [[gf args alts] models, args' alts, seed (range 40)]
  (let [tr (gfi/simulate gf args (rng/key seed))
        other (gfi/simulate gf args' (rng/key (+ 1000 seed)))
        addrs (cm/addresses (:choices tr))
        picks (take (count addrs) (cycle (shuffle (concat (repeat 3 false) [true]))))
        chosen (keep (fn [[a p]] (when p a)) (map vector addrs picks))
        oaddrs (cm/addresses (:choices other))
        cs (reduce (fn [c a] (cm/set-choice c a (cm/get-choice (:choices other) a))) cm/EMPTY
                   (keep (fn [[a p]] (when p a)) (map vector oaddrs (cycle [false false true]))))
        precise (mapv diff/compute args args')
        unknown (vec (repeat (count args) diff/unknown-change))
        k (rng/key (+ 2000 seed))
        same? (fn [a b] (and (= (:choices (:trace a)) (:choices (:trace b)))
                             (= (:retval (:trace a)) (:retval (:trace b)))
                             (close (:score (:trace a)) (:score (:trace b)))
                             (close (:weight a) (:weight b))
                             (= (:discard a) (:discard b))))
        u (fn [ds] (try (gfi/update tr args' ds cs k) (catch clojure.lang.ExceptionInfo e [:err (ex-message e)])))
        r (fn [ds] (gfi/regenerate tr args' ds (sel/from-paths chosen) k))]
    (let [a (u precise) b (u unknown)]
      (when-not (or (and (vector? a) (vector? b)) (same? a b)) (swap! failures conj [:update gf args' seed])))
    (when-not (same? (r precise) (r unknown)) (swap! failures conj [:regenerate gf args' seed]))))
(println "cases with differing results:" (count @failures) (take 3 @failures))
