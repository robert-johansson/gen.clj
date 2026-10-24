;; The code of book/017-composition.md. Each region, from a `;; book:` line
;; to the next, is included there by name (gen.book: `clojure -M:test -m
;; gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.composition
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as imp]
            [gen.inference.weights :as weights]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.state-space :as ss]
            [gen.trace :as trace]))

;; book: sensor
(def sensor
  (gen [position]
    (trace :reading (d/normal position 0.5))))

(def fusion
  (gen [position]
    [(splice :a sensor position)
     (splice :b sensor position)]))

;; book: fusion-trace
(def fused (gfi/simulate fusion [5.0] (rng/key 1)))

(cm/addresses (:choices fused))
;=> [[:a :reading] [:b :reading]]

[(fused [:a :reading]) (fused [:b :reading])]
;=> [3.925 4.534] ; float64

(= (:retval fused) [(fused [:a :reading]) (fused [:b :reading])])
;=> true

;; book: fusion-subtrace
(def sensor-a (trace/subtrace fused :a))

[(:args sensor-a) (cm/addresses (:choices sensor-a))]
;=> [[5.0] [[:reading]]]

(= (fused :a) (:retval sensor-a))
;=> true

(array/allclose? (:score fused)
                 (+ (:score sensor-a) (:score (trace/subtrace fused :b))))
;=> true

;; book: fusion-constrain
(def one-reading (cm/choicemap [:b :reading] 5.2))

(array/allclose?
 (:weight (gfi/generate fusion [5.0] one-reading (rng/key 2)))
 (dc/log-prob (d/normal 5.0 0.5) 5.2))
;=> true

;; book: splice-flat
(def one-sensor
  (gen [position]
    (splice sensor position)))

(cm/addresses (:choices (gfi/simulate one-sensor [5.0] (rng/key 3))))
;=> [[:reading]]

;; book: splice-twice
;; error: Address [:reading] is used twice
#_(def two-sensors
  (gen [position]
    [(splice sensor position)
     (splice sensor position)]))

#_(gfi/simulate two-sensors [5.0] (rng/key 3))

;; book: splice-flat-only-gen
;; error: A splice without an address needs a dynamic generative function
#_(def many-sensors
  (gen [positions]
    (splice (comb/map sensor) positions)))

#_(gfi/simulate many-sensors [[1.0 2.0]] (rng/key 3))

;; book: map-model
(def point
  (gen [x]
    (trace :y (d/normal (* 2.0 x) 1.0))))

(def points (comb/map point))

;; book: map-trace
(def xs [1.0 2.0 3.0 4.0 5.0])

(def mapped (gfi/simulate points [xs] (rng/key 4)))

(cm/addresses (:choices mapped))
;=> [[0 :y] [1 :y] [2 :y] [3 :y] [4 :y]]

(:retval mapped)
;=> [1.139 3.786 4.213 7.398 11.199] ; float64

;; book: map-columns
(def scaled
  (comb/map (gen [x sd] (trace :y (d/normal (* 2.0 x) sd)))))

(cm/addresses
 (:choices (gfi/simulate scaled [[1.0 2.0] [0.1 10.0]] (rng/key 5))))
;=> [[0 :y] [1 :y]]

;; book: map-score
(def element-scores
  (mapv #(:score (trace/subtrace mapped %)) (range (count xs))))

(array/allclose? (:score mapped) (reduce + element-scores))
;=> true

;; book: map-constrain
(def element-2 (cm/choicemap [2 :y] 7.0))

(array/allclose? (:weight (gfi/generate points [xs] element-2 (rng/key 6)))
                 (dc/log-prob (d/normal 6.0 1.0) 7.0))
;=> true

;; book: map-wrong-path
;; error: generate: the constraint at [:y] was not visited
#_(gfi/generate points [xs] (cm/choicemap [:y 2] 7.0) (rng/key 6))

;; book: hmm-model
(def hmm-step
  (gen [t state transition means]
    (let [state' (trace :z (d/categorical (array/take transition state)))]
      (trace :y (d/normal (array/take means state') 1.0))
      state')))

(def hmm (comb/unfold hmm-step))

(def transition (array/array [[0.9 0.1] [0.2 0.8]]))

(def means (array/array [-1.0 2.0]))

;; book: hmm-trace
(def chain (gfi/simulate hmm [4 0 transition means] (rng/key 6)))

(cm/addresses (:choices chain))
;=> [[0 :y] [0 :z] [1 :y] [1 :z] [2 :y] [2 :z] [3 :y] [3 :z]]

(:retval chain)
;=> [0 0 1 1] ; float64

(= (:retval chain) (mapv #(chain [% :z]) (range 4)))
;=> true

;; book: hmm-data
(def ys [-0.6 -1.4 1.7 2.5 0.4 2.2])

(def n (count ys))

(def hmm-args [n 0 transition means])

;; book: hmm-enumerate
(defn paths
  "Every sequence of n states, each 0 or 1."
  [n]
  (if (zero? n) [[]] (for [p (paths (dec n)) z [0 1]] (conj p z))))

(defn joint
  "The choice map of the states zs and the observations ys."
  [zs]
  (apply cm/choicemap (mapcat (fn [t z y] [[t :z] z [t :y] y]) (range) zs ys)))

(def log-joints
  (array/stack
   (mapv #(:weight (gfi/assess hmm hmm-args (joint %))) (paths n))))

(def log-evidence (array/logsumexp log-joints))

[(count (paths n)) log-evidence]
;=> [64 -10.247]

;; book: hmm-exact
(def observations (apply cm/choicemap (interleave (range) ys)))

(def forward
  (gfi/assess ss/hmm
              [n (array/take transition 0)
               {:transition transition :emission (d/normal means 1.0)}]
              observations))

(array/allclose? (:weight forward) log-evidence)
;=> true

;; book: hmm-marginals
(def state-1
  "The posterior probability of state 1 at each step, by enumeration."
  (array/matmul (array/softmax log-joints)
                (array/array (mapv #(mapv double %) (paths n)))))

;; book: hmm-figure
;; figure: figures/017-hmm-marginals.svg
(svg/series (array/->clj state-1)
            {:x-label "step t" :y-label "P(state 1 | y)"})

;; book: hmm-step-4
(def likelihood-ratio
  "log p(y_4 | state 0) - log p(y_4 | state 1)."
  (- (dc/log-prob (d/normal -1.0 1.0) (ys 4))
     (dc/log-prob (d/normal 2.0 1.0) (ys 4))))

[likelihood-ratio (array/take state-1 4)]
;=> [0.3 0.957]

;; book: scan-model
(def heat-step
  (gen [temp heater]
    (let [temp' (trace :temp (d/normal (+ (* 0.9 temp) heater) 0.3))]
      [temp' (trace :reading (d/normal temp' 0.5))])))

(def heating (comb/scan heat-step))

;; book: scan-trace
(def heaters [1.0 1.0 0.0 0.0 2.0])

(def heated (gfi/simulate heating [20.0 heaters] (rng/key 8)))

(take 4 (cm/addresses (:choices heated)))
;=> [[0 :reading] [0 :temp] [1 :reading] [1 :temp]]

(let [[temp readings] (:retval heated)]
  [(= temp (heated [4 :temp]))
   (= readings (mapv #(heated [% :reading]) (range 5)))])
;=> [true true]

;; book: map-update
(def xs' (assoc xs 2 3.5))

(def moved
  (gfi/update mapped [xs'] [(diff/vector-diff 5 5 {2 diff/unknown-change})]
              cm/EMPTY (rng/key 9)))

(defn revisited
  "The indices whose subtraces differ between traces a and b."
  [a b]
  (vec (remove #(identical? (trace/subtrace a %) (trace/subtrace b %))
               (range (count (:retval a))))))

(revisited mapped (:trace moved))
;=> [2]

;; book: map-update-check
(def recomputed
  (- (:weight (gfi/assess points [xs'] (:choices mapped))) (:score mapped)))

(array/allclose? (:weight moved) recomputed)
;=> true

;; book: map-update-unknown
(def moved-blind
  (gfi/update mapped [xs'] [diff/unknown-change] cm/EMPTY (rng/key 9)))

[(revisited mapped (:trace moved-blind))
 (array/allclose? (:weight moved-blind) (:weight moved))]
;=> [[0 1 2 3 4] true]

;; book: map-keys
(def six (gfi/simulate points [(conj xs 6.0)] (rng/key 4)))

(= (subvec (:retval six) 0 5) (:retval mapped))
;=> true

;; book: unfold-extend
(def longer
  (gfi/update chain [5 0 transition means]
              [diff/unknown-change diff/no-change diff/no-change diff/no-change]
              (cm/choicemap [4 :y] 1.0) (rng/key 10)))

(revisited chain (:trace longer))
;=> []

(:retdiff longer)
;=> {:new-length 5 :prev-length 4 :updated {}}

;; book: unfold-extend-weight
(def new-state ((:trace longer) [4 :z]))

(array/allclose? (:weight longer)
                 (dc/log-prob (d/normal (array/take means new-state) 1.0)
                              1.0))
;=> true

;; book: switch-models
(def observe
  (gen [f xs]
    (doseq [[i x] (map-indexed vector xs)]
      (trace [:y i] (d/normal (f x) 1.0)))))

(def line
  (gen [xs]
    (let [b (trace :intercept (d/normal 0 2))
          m (trace :slope (d/normal 0 2))]
      (splice observe (fn [x] (+ b (* m x))) xs))))

(def parabola
  (gen [xs]
    (let [b (trace :intercept (d/normal 0 2))
          m (trace :slope (d/normal 0 2))
          c (trace :curvature (d/normal 0 2))]
      (splice observe (fn [x] (+ b (* m x) (* c x x))) xs))))

(def shape (comb/switch line parabola))

;; book: switch-trace
(def xs2 [-2.0 -1.0 0.0 1.0 2.0])

(def as-line (gfi/simulate shape [0 xs2] (rng/key 11)))

(def as-parabola (gfi/simulate shape [1 xs2] (rng/key 11)))

(cm/addresses (:choices as-line))
;=> [[:intercept] [:slope] [:y 0] [:y 1] [:y 2] [:y 3] [:y 4]]

(take 3 (cm/addresses (:choices as-parabola)))
;=> [[:curvature] [:intercept] [:slope]]

;; book: switch-score
(array/allclose?
 (:weight (gfi/assess shape [1 xs2] (:choices as-parabola)))
 (:weight (gfi/assess parabola [xs2] (:choices as-parabola))))
;=> true

;; book: evidence-exact
(def ys2 [3.9 0.6 -0.4 1.2 4.3])

(def observed (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys2)))

(defn exact-evidence
  "log p(ys) of a model whose mean at x is the dot product of weights w
   and (features x), each weight ~ N(0, 2), with noise 1: for X the matrix
   of the features' rows, ys ~ N(0, 4 X X^T + I)."
  [features xs ys]
  (let [X (array/array (mapv features xs))
        k (count xs)
        cov (+ (* 4.0 (array/matmul X (array/transpose X))) (array/eye k))]
    (array/item (dc/log-prob (d/mvnormal (array/zeros [k]) cov)
                             (array/array ys)))))

(def evidence
  [(exact-evidence (fn [x] [1.0 x]) xs2 ys2)
   (exact-evidence (fn [x] [1.0 x (* x x)]) xs2 ys2)])

evidence
;=> [-16.931 -10.231]

;; book: evidence-is
(defn estimate
  "The log evidence of branch k by importance sampling with n samples, and
   its standard error, sqrt(1/ESS - 1/n) for the samples' effective sample
   size ESS."
  [k n key]
  (let [r (imp/importance-sampling shape [k xs2] observed n
                                   {:key key :batched? true})
        ess (weights/ess (:log-normalized-weights r))]
    {:log-evidence (:log-ml-estimate r)
     :se (Math/sqrt (- (/ 1.0 ess) (/ 1.0 n)))}))

(def estimates
  [(estimate 0 20000 (rng/key 12)) (estimate 1 20000 (rng/key 13))])

(mapv (fn [{:keys [log-evidence se]} exact]
        (< (abs (- log-evidence exact)) (* 5 se)))
      estimates evidence)
;=> [true true]

;; book: bayes-factor
(- (evidence 1) (evidence 0))
;=> 6.70

;; book: switch-update
(def to-parabola
  (gfi/update as-line [1 xs2] [diff/unknown-change diff/no-change]
              cm/EMPTY (rng/key 14)))

(every? #(= (as-line %) ((:trace to-parabola) %))
        (cm/addresses (:choices as-line)))
;=> true

(def curvature ((:trace to-parabola) :curvature))

(array/allclose? (:weight to-parabola)
                 (- (:score (:trace to-parabola)) (:score as-line)
                    (dc/log-prob (d/normal 0 2) curvature)))
;=> true

;; book: switch-back
(def to-line
  (gfi/update (:trace to-parabola) [0 xs2] [diff/unknown-change diff/no-change]
              cm/EMPTY (rng/key 15)))

(cm/addresses (:discard to-line))
;=> [[:curvature]]

(array/allclose? (:weight to-line)
                 (- (:score (:trace to-line)) (:score (:trace to-parabola))))
;=> true

;; book: mix-model
(def either (comb/mix line parabola))

(def mixed (gfi/simulate either [[0.3 0.7] xs2] (rng/key 16)))

(mixed :index)
;=> 1 ; float64

(take 3 (cm/addresses (:choices mixed)))
;=> [[:component :curvature] [:component :intercept] [:component :slope]] ; float64

;; book: mix-score
(let [weight (nth [0.3 0.7] (array/item (mixed :index)))]
  (array/allclose? (:score mixed)
                   (+ (Math/log weight) (:score (trace/subtrace mixed :component)))))
;=> true

;; book: mix-frequency
(def indices
  (mapv #((gfi/simulate either [[0.3 0.7] xs2] %) :index)
        (rng/split (rng/key 17) 2000)))

(let [freq (/ (count (filter #(== 1 (array/item %)) indices)) 2000.0)
      se (Math/sqrt (/ (* 0.7 0.3) 2000))]
  (< (abs (- freq 0.7)) (* 5 se)))
;=> true

;; book: tree-plain
(def tree
  (gen [depth]
    (if (or (>= depth 2) (trace :leaf? (d/bernoulli 0.5)))
      (trace :value (d/normal 0 1))
      (+ (splice :left tree (inc depth))
         (splice :right tree (inc depth))))))

(def grown (gfi/simulate tree [0] (rng/key 27)))

(filter #(= :value (peek %)) (cm/addresses (:choices grown)))
;=> [[:left :left :value] [:left :right :value] [:right :value]] ; float64

(mapv #(count (cm/addresses (:choices (gfi/simulate tree [0] %))))
      (rng/split (rng/key 28) 8))
;=> [2 6 2 6 2 2 6 6] ; float64

;; book: tree-figure
;; figure: figures/017-tree.svg
(svg/tree (cm/addresses (:choices grown)))

;; book: recurse-model
(def grow
  (gen [depth]
    (if (or (>= depth 2) (trace :leaf? (d/bernoulli 0.5)))
      {:value (trace :value (d/normal 0 1)) :children []}
      {:value 0.0 :children [(inc depth) (inc depth)]})))

(def total (gen [value children] (reduce + value children)))

(def recursive-tree (comb/recurse grow total 2))

;; book: recurse-numbers
[(comb/get-child 0 0 2) (comb/get-child 0 1 2) (comb/get-child 1 0 2)]
;=> [1 2 3]

[(comb/get-parent 3 2) (comb/get-child-num 3 2)]
;=> [1 0]

;; book: recurse-assess
(def small-tree
  (cm/choicemap [0 :production :leaf?] false
                [1 :production :leaf?] true
                [1 :production :value] 0.3
                [2 :production :leaf?] true
                [2 :production :value] -1.2))

(def assessed (gfi/assess recursive-tree [0] small-tree))

(:retval assessed)
;=> -0.9

(array/allclose? (:weight assessed)
                 (+ (* 3 (Math/log 0.5))
                    (dc/log-prob (d/normal 0 1) 0.3)
                    (dc/log-prob (d/normal 0 1) -1.2)))
;=> true

;; book: recurse-update
(def small (:trace (gfi/generate recursive-tree [0] small-tree (rng/key 31))))

(def regrown
  (gfi/update small (cm/choicemap [2 :production :value] 0.5) (rng/key 32)))

(:retval (:trace regrown))
;=> 0.8

(for [node [0 1 2]
      part [:production :aggregation]
      :let [p [node part]]
      :when (not (identical? (trace/subtrace small p)
                             (trace/subtrace (:trace regrown) p)))]
  p)
;=> [[0 :aggregation] [2 :production] [2 :aggregation]]

;; book: call-at
(def sensor-at (comb/call-at sensor))

(cm/addresses (:choices (gfi/simulate sensor-at [5.0 :c] (rng/key 19))))
;=> [[:c :reading]]

(def normal-at (comb/choice-at d/normal))

(cm/addresses
 (:choices (gfi/simulate (comb/map normal-at)
                         [[0.0 5.0] [1.0 1.0] [:first :second]]
                         (rng/key 20))))
;=> [[0 :first] [1 :second]]

;; book: dimap
(def feet-per-meter 3.28084)

(def sensor-in-feet
  (comb/dimap sensor
              (fn [feet] [(/ feet feet-per-meter)])
              (fn [meters] (* meters feet-per-meter))))

(def in-feet (gfi/simulate sensor-in-feet [16.4] (rng/key 21)))

(cm/addresses (:choices in-feet))
;=> [[:reading]]

(array/allclose? (:retval in-feet) (* (in-feet :reading) feet-per-meter))
;=> true

;; book: contramap
(def from-feet (comb/contramap sensor (fn [feet] [(/ feet feet-per-meter)])))

(def to-feet (comb/map-retval sensor (fn [meters] (* meters feet-per-meter))))

(def five-meters (cm/choicemap :reading 5.0))

[(array/allclose? (:weight (gfi/assess from-feet [16.4] five-meters))
                  (dc/log-prob (d/normal (/ 16.4 feet-per-meter) 0.5) 5.0))
 (array/allclose? (:retval (gfi/assess to-feet [5.0] five-meters))
                  (* 5.0 feet-per-meter))]
;=> [true true]

;; book: nested
(def hmms (comb/map hmm))

(def hmms-args [[3 3] [0 1] [transition transition] [means means]])

(def nested (gfi/simulate hmms hmms-args (rng/key 22)))

(take 4 (cm/addresses (:choices nested)))
;=> [[0 0 :y] [0 0 :z] [0 1 :y] [0 1 :z]]

;; book: nested-figure
;; figure: figures/017-nested.svg
(svg/tree (cm/addresses (:choices nested)))

;; book: nested-gfi
(def all-choices (:choices nested))

[(array/allclose?
  (:weight (gfi/generate hmms hmms-args all-choices (rng/key 23)))
  (:score nested))
 (array/allclose? (:weight (gfi/assess hmms hmms-args all-choices))
                  (:score nested))
 (array/allclose? (gfi/project nested sel/all) (:score nested))
 (:weight (gfi/regenerate nested sel/none (rng/key 24)))]
;=> [true true true 0.0]

;; book: solution-1
(def three-sensors
  (gen [position]
    [(splice :a sensor position)
     (splice :b sensor position)
     (splice :c sensor position)]))

(def three (gfi/simulate three-sensors [5.0] (rng/key 25)))

(cm/addresses (:choices three))
;=> [[:a :reading] [:b :reading] [:c :reading]]

(def resampled
  (:trace (gfi/regenerate three (sel/select [:b :reading]) (rng/key 26))))

(mapv #(= (three [% :reading]) (resampled [% :reading])) [:a :b :c])
;=> [true false true]

;; book: solution-2
(def ys-line [0.2 0.1 1.3 1.4 2.1])

(def evidence-line
  [(exact-evidence (fn [x] [1.0 x]) xs2 ys-line)
   (exact-evidence (fn [x] [1.0 x (* x x)]) xs2 ys-line)])

(- (evidence-line 0) (evidence-line 1))
;=> 2.02

;; book: solution-3
(def hmm-scan-step
  (gen [state t transition means]
    (let [state' (splice hmm-step t state transition means)]
      [state' state'])))

(def hmm-scan (comb/scan hmm-scan-step))

(array/allclose?
 (:weight (gfi/assess hmm-scan [0 (vec (range n)) transition means]
                      (joint [0 0 1 1 1 1])))
 (:weight (gfi/assess hmm hmm-args (joint [0 0 1 1 1 1]))))
;=> true

;; book: solution-4
(defn hmm-exact-args
  "gen.state-space/hmm's arguments for the first k observations."
  [k]
  [k (array/take transition 0)
   {:transition transition :emission (d/normal means 1.0)}])

(def first-five
  (:trace (gfi/generate ss/hmm (hmm-exact-args 5)
                        (apply cm/choicemap (interleave (range 5) ys))
                        (rng/key 29))))

(def sixth
  (gfi/update first-five (hmm-exact-args 6)
              [diff/unknown-change diff/no-change diff/no-change]
              (cm/choicemap 5 (ys 5)) (rng/key 30)))

(array/allclose? (:weight sixth)
                 (- log-evidence (:score first-five)))
;=> true
