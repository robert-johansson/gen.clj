;; The code of book/051-data-driven-proposals.md. Each region, from a
;; `;; book:` line to the next, is included there by name (gen.book:
;; `clojure -M:test -m gen.book sync` writes the Markdown's copies;
;; gen.book-test checks them).

;; book: ns
(ns book.data-driven-proposals
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core :as c]
            [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.inference.importance :as is]
            [gen.inference.weights :as w]
            [gen.learning :as learn]
            [gen.random :as rng]))

;; book: geometry
(defn distance
  "The distance between the points a and b, each [x y]."
  [[ax ay] [bx by]]
  (let [dx (c/- bx ax)
        dy (c/- by ay)]
    (Math/sqrt (c/+ (c/* dx dx) (c/* dy dy)))))

(defn ccw
  "Twice the signed area of the triangle a, b, c: positive when c lies to
   the left of the line from a to b, negative to its right, 0 on it."
  [[ax ay] [bx by] [cx cy]]
  (c/- (c/* (c/- bx ax) (c/- cy ay)) (c/* (c/- cx ax) (c/- by ay))))

(defn segments-meet?
  "Do the segments a1–a2 and b1–b2 meet? They do unless one of them has
   both ends of the other strictly on one side of its line."
  [a1 a2 b1 b2]
  (not (or (pos? (c/* (ccw a1 a2 b1) (ccw a1 a2 b2)))
           (pos? (c/* (ccw b1 b2 a1) (ccw b1 b2 a2))))))

;; book: geometry-checks
(distance [0.0 0.0] [3.0 4.0])
;=> 5.0

[(ccw [0.0 0.0] [1.0 0.0] [0.0 1.0]) (ccw [0.0 0.0] [1.0 0.0] [0.0 -1.0])]
;=> [1.0 -1.0]

[(segments-meet? [0.0 0.0] [1.0 1.0] [0.0 1.0] [1.0 0.0])
 (segments-meet? [0.0 0.0] [1.0 0.0] [0.0 1.0] [1.0 1.0])
 (segments-meet? [0.0 0.0] [1.0 0.0] [0.5 0.0] [0.5 1.0])]
;=> [true false true]

;; book: collinear
(segments-meet? [0.0 0.0] [1.0 0.0] [2.0 0.0] [3.0 0.0])
;=> true

;; book: obstacles
(defn rectangle
  "The obstacle [x, x + w] × [y, y + h], as its lower-left and upper-right
   corners."
  [x y w h]
  [[x y] [(c/+ x w) (c/+ y h)]])

(defn square
  "A square obstacle of side size centered at [x y]: a tree."
  [[x y] size]
  (let [half (c// size 2.0)]
    (rectangle (c/- x half) (c/- y half) size size)))

(defn wall
  "A wall of the given thickness from the corner [x y], length long to
   the right (:horizontal) or up (:vertical)."
  [direction [x y] length thickness]
  (case direction
    :horizontal (rectangle x y length thickness)
    :vertical (rectangle x y thickness length)))

(defn corners
  "The corners of a rectangle, counterclockwise from its lower left."
  [[[x0 y0] [x1 y1]]]
  [[x0 y0] [x1 y0] [x1 y1] [x0 y1]])

(defn crosses?
  "Does the segment a–b meet an edge of the rectangle r? Only if it
   reaches r's span on both axes, and then if it meets one of r's edges."
  [[[x0 y0] [x1 y1] :as r] [ax ay :as a] [bx by :as b]]
  (and (<= (min ax bx) x1) (<= x0 (max ax bx))
       (<= (min ay by) y1) (<= y0 (max ay by))
       (let [cs (corners r)]
         (boolean (some (fn [[v w]] (segments-meet? v w a b))
                        (map vector cs (conj (subvec cs 1) (cs 0))))))))

(defn obstructed?
  "Does an obstacle of scene stand in the way of the segment a–b?"
  [scene a b]
  (boolean (some #(crosses? % a b) (:obstacles scene))))

;; book: scene
(def thickness 0.02)

(def scene
  {:xmin 0.0 :xmax 1.0 :ymin 0.0 :ymax 1.0
   :obstacles [(square [0.30 0.20] 0.1)
               (square [0.83 0.80] 0.1)
               (square [0.80 0.40] 0.1)
               (wall :horizontal [0.20 0.40] 0.40 thickness)
               (wall :vertical [0.60 0.40] 0.40 thickness)
               (wall :horizontal [0.45 0.80] (c/+ 0.15 thickness) thickness)
               (wall :horizontal [0.20 0.80] 0.15 thickness)
               (wall :vertical [0.20 0.40] 0.40 thickness)]})

;; book: obstructed-checks
[(obstructed? scene [0.1 0.6] [0.3 0.6])
 (obstructed? scene [0.1 0.1] [0.1 0.9])
 (obstructed? scene [0.4 0.9] [0.4 0.6])]
;=> [true false false]

;; book: tree
(defn nearest
  "The index of the node of tree nearest the point p."
  [tree p]
  (apply min-key #(distance (:point (tree %)) p) (range (count tree))))

(defn toward
  "The point at most step from a on the segment from a to b."
  [a b step]
  (let [d (distance a b)]
    (if (<= d step)
      b
      (let [s (c// step d)]
        (mapv (fn [u v] (c/+ u (c/* s (c/- v u)))) a b)))))

(defn grow-tree
  "A rapidly exploring random tree from root in scene: :rrt-iters times, a
   uniform point of the scene, and a step of at most :rrt-step from the
   tree's nearest node toward it, kept when nothing obstructs it. A node
   is {:point :parent :cost}, cost its distance from the root along the
   tree, parent its parent's index; the root is node 0."
  [scene root {:keys [rrt-iters rrt-step]} key]
  (let [{:keys [xmin xmax ymin ymax]} scene
        targets (array/->clj
                 (rng/uniform key [rrt-iters 2]
                              {:low (array/array [xmin ymin])
                               :high (array/array [xmax ymax])}))]
    (reduce (fn [tree target]
              (let [i (nearest tree target)
                    {:keys [point cost]} (tree i)
                    new (toward point target rrt-step)]
                (if (obstructed? scene point new)
                  tree
                  (conj tree {:point new :parent i
                              :cost (c/+ cost (distance point new))}))))
            [{:point root :parent nil :cost 0.0}]
            targets)))

;; book: route
(defn route
  "The tree's shortest way from its root to dest, as a vector of points:
   through the node that sees dest unobstructed with the least cost plus
   distance to dest; nil when no node sees it."
  [scene tree dest]
  (let [cost (fn [{:keys [point cost]}] (c/+ cost (distance point dest)))
        sees? (fn [i] (not (obstructed? scene (:point (tree i)) dest)))]
    (when-let [best (first (filter sees? (sort-by #(cost (tree %))
                                                  (range (count tree)))))]
      (let [up (take-while some? (iterate #(:parent (tree %)) best))]
        (conj (mapv #(:point (tree %)) (reverse up)) dest)))))

;; book: refine
(defn path-length
  "The length of the path, a sequence of points."
  [path]
  (reduce c/+ 0.0 (map distance path (rest path))))

(defn refine
  "path, shortened: :refine-iters times, its interior points in turn moved
   by a normal step of :refine-sd in each coordinate, each move kept when
   it shortens the path and leaves it unobstructed. The ends stay."
  [scene path {:keys [refine-iters refine-sd]} key]
  (let [n (c/- (count path) 2)
        steps (array/->clj (rng/normal key [refine-iters 2]))]
    (if (zero? n)
      path
      (reduce (fn [path [i [ex ey]]]
                (let [j (inc (mod i n))
                      [before p after] (subvec path (dec j) (c/+ j 2))
                      q [(c/+ (p 0) (c/* refine-sd ex))
                         (c/+ (p 1) (c/* refine-sd ey))]]
                  (if (and (< (c/+ (distance before q) (distance q after))
                              (c/+ (distance before p) (distance p after)))
                           (not (obstructed? scene before q))
                           (not (obstructed? scene q after)))
                    (assoc path j q)
                    path)))
              path
              (map-indexed vector steps)))))

;; book: plan
(def planner
  "The planner's parameters, Gen.jl's rrt_iters, rrt_dt, refine_iters and
   refine_std: a smaller tree than the tutorial's 600 iterations, and 1000
   finer refining steps for its 3500 of standard deviation 1."
  {:rrt-iters 200 :rrt-step 3.0 :refine-iters 1000 :refine-sd 0.05})

(defn plan-path
  "A path from start to dest in scene that no obstacle obstructs, by
   params (planner's keys), drawing its randomness from key: a tree grown
   from start, its route to dest, refined. nil when the tree finds no
   route."
  [scene start dest params key]
  (let [[k-tree k-refine] (rng/split key)]
    (when-let [path (route scene (grow-tree scene start params k-tree) dest)]
      (refine scene path params k-refine))))

;; book: plan-example
(def start [0.1 0.1])

(def tree (grow-tree scene start planner (rng/key 3)))

(def rough (route scene tree [0.5 0.5]))

(def example-path (refine scene rough planner (rng/key 4)))

[(count tree) (path-length rough) (path-length example-path)]
;=> [153 1.5528 1.2617] ; float64

;; book: plan-checks
(defn clear?
  "Does no obstacle of scene obstruct a segment of path?"
  [scene path]
  (not-any? (fn [[a b]] (obstructed? scene a b)) (partition 2 1 path)))

[(first example-path) (peek example-path)
 (clear? scene rough) (clear? scene example-path)
 (<= (path-length example-path) (path-length rough))]
;=> [[0.1 0.1] [0.5 0.5] true true true]

;; book: plan-figure
;; figure: figures/051-planner.svg
(svg/shapes [{:polygons (map corners (:obstacles scene))}
             {:paths (keep (fn [{:keys [point parent]}]
                             (when parent [(:point (tree parent)) point]))
                           tree)
              :color "silver" :width 0.8}
             {:paths [rough] :color "darkorange" :width 2}
             {:paths [example-path] :color "steelblue" :width 2}
             {:points [start] :r 4}
             {:points [[0.5 0.5]] :color "red" :r 4}])

;; book: plan-cases
[(plan-path scene start [0.16 0.9] planner (rng/key 3))
 (plan-path scene start [0.83 0.80] planner (rng/key 3))]
;=> [[[0.1 0.1] [0.16 0.9]] nil]

;; book: walk
(defn point-along
  "The point at distance s along path, or its end when s is past it."
  [path s]
  (loop [[a b & more] path
         s s]
    (let [d (if b (distance a b) 0.0)]
      (cond (nil? b) a
            (<= s d) (let [f (if (pos? d) (c// s d) 0.0)]
                       (mapv (fn [u v] (c/+ u (c/* f (c/- v u)))) a b))
            :else (recur (cons b more) (c/- s d))))))

(defn walk
  "The agent's locations at n-ticks times dt apart from time 0, moving
   along path at speed and staying at its end once there."
  [path speed dt n-ticks]
  (mapv #(point-along path (c/* speed dt %)) (range n-ticks)))

;; book: walk-checks
(def walked (walk [[0.0 0.0] [0.5 0.0] [0.5 0.5]] 0.25 0.5 10))

(subvec walked 0 2)
;=> [[0.0 0.0] [0.125 0.0]]

(subvec walked 4 6)
;=> [[0.5 0.0] [0.5 0.125]]

(subvec walked 8)
;=> [[0.5 0.5] [0.5 0.5]]

;; book: model
(def noise 0.01)

(def agent-model
  (gen [scene dt n-ticks params]
    (let [{:keys [xmin xmax ymin ymax]} scene
          start [(trace :start-x (d/uniform xmin xmax))
                 (trace :start-y (d/uniform ymin ymax))]
          dest [(trace :dest-x (d/uniform xmin xmax))
                (trace :dest-y (d/uniform ymin ymax))]
          path (plan-path scene (mapv array/item start) (mapv array/item dest)
                          params (untraced-key))
          speed (trace :speed (d/uniform 0.3 1.0))
          locations (if path
                      (walk path (array/item speed) dt n-ticks)
                      (repeat n-ticks start))]
      (doseq [[i [x y]] (map-indexed vector locations)]
        (trace [:meas i :x] (d/normal x noise))
        (trace [:meas i :y] (d/normal y noise)))
      path)))

;; book: model-args
(def dt 0.1)

(def n-ticks 10)

(def model-args [scene dt n-ticks planner])

;; book: simulate
(def t1 (gfi/simulate agent-model model-args (rng/key 5)))

(mapv #(array/item (t1 %)) [:start-x :start-y :dest-x :dest-y :speed])
;=> [0.1965 0.9129 0.9828 0.8668 0.9500] ; float64

(= (:retval t1) (:retval (gfi/simulate agent-model model-args (rng/key 5))))
;=> true

;; book: score
(defn score-by-hand
  "The sum of the log densities of trace t's choices, given the path it
   returned (nil when planning failed)."
  [t]
  (let [{:keys [xmin xmax ymin ymax]} scene
        start [(t :start-x) (t :start-y)]
        locations (if-let [path (:retval t)]
                    (walk path (array/item (t :speed)) dt n-ticks)
                    (repeat n-ticks start))
        lp (fn [dist v] (array/item (dc/log-prob dist v)))]
    (reduce c/+ (lp (d/uniform 0.3 1.0) (t :speed))
            (concat
             (map (fn [a lo hi] (lp (d/uniform lo hi) (t a)))
                  [:start-x :start-y :dest-x :dest-y]
                  [xmin ymin xmin ymin] [xmax ymax xmax ymax])
             (map-indexed (fn [i [x y]]
                            (c/+ (lp (d/normal x noise) (t [:meas i :x]))
                                 (lp (d/normal y noise) (t [:meas i :y]))))
                          locations)))))

(array/allclose? (:score t1) (score-by-hand t1))
;=> true

;; book: measured
(defn measured
  "The measured locations of a trace of agent-model, as host numbers."
  [t]
  (mapv (fn [i] [(array/item (t [:meas i :x])) (array/item (t [:meas i :y]))])
        (range n-ticks)))

;; book: traces-figure
;; figure: figures/051-traces.svg
(let [ts (mapv #(:trace (gfi/generate agent-model model-args
                                      (cm/choicemap :start-x 0.1 :start-y 0.1)
                                      %))
               (rng/split (rng/key 6) 6))]
  (svg/shapes [{:polygons (map corners (:obstacles scene))}
               {:paths (keep :retval ts) :color "darkorange"}
               {:points (mapcat measured ts) :color "black" :r 1.5}
               {:points (mapv #(mapv array/item [(% :dest-x) (% :dest-y)]) ts)
                :color "red" :r 4}
               {:points [start] :r 4}]))

;; book: data
(def measurements
  [[0.0980245 0.104775] [0.113734 0.150773] [0.100412 0.195499]
   [0.114794 0.237386] [0.0957668 0.277711] [0.140181 0.31304]
   [0.124384 0.356242] [0.122272 0.414463] [0.124597 0.462056]
   [0.126227 0.498338]])

(defn observations
  "The choice map of the start and the measured locations ms."
  [start ms]
  (apply cm/choicemap :start-x (start 0) :start-y (start 1)
         (mapcat (fn [i [x y]] [[:meas i :x] x [:meas i :y] y]) (range) ms)))

(def observed (observations start measurements))

;; book: sample-run
(def proposal-args [measurements scene dt])

(defn sample-run
  "One run of importance sampling of n samples with key and the proposal q
   (the model's prior when nil): its effective sample size, and the
   destination of a sample drawn in proportion to the weights."
  [q n key]
  (let [[k-run k-draw] (rng/split key)
        opts (cond-> {:key k-run}
               q (assoc :proposal q :proposal-args proposal-args))
        {:keys [traces log-weights ess]}
        (is/importance-sampling agent-model model-args observed n opts)
        t (traces (first (w/multinomial log-weights 1 k-draw)))]
    {:ess ess :dest [(array/item (t :dest-x)) (array/item (t :dest-y))]}))

(defn mean-se
  "The mean of the numbers vs, and its standard error."
  [vs]
  (let [n (count vs)
        mean (c// (reduce c/+ vs) n)
        sq (reduce c/+ (map #(let [e (c/- % mean)] (c/* e e)) vs))]
    {:mean mean :se (Math/sqrt (c// sq (c/* n (dec n))))}))

;; book: prior-runs
(def n-samples 50)

(def prior-runs
  (mapv #(sample-run nil n-samples %) (rng/split (rng/key 7) 20)))

(mean-se (map :ess prior-runs))
;=> {:mean 1.020 :se 0.012} ; float64

;; book: speed-pinned
(def in-view [0.16 0.9])

(defn log-density-at
  "The log density of the trace with the observations, the destination
   dest and the speed: generate's weight when every choice is given."
  [dest speed]
  (let [choices (cm/choicemap :dest-x (dest 0) :dest-y (dest 1) :speed speed)]
    (array/item (:weight (gfi/generate agent-model model-args
                                       (cm/merge observed choices)
                                       (rng/key 0))))))

(defn dot
  "The dot product of the vectors a and b."
  [a b]
  (reduce c/+ (map c/* a b)))

(defn straight-speed
  "The speed that best explains the measurements ms of an agent walking
   straight from start toward dest, and its standard deviation: there the
   log density is quadratic in the speed."
  [start dest ms]
  (let [u (mapv #(c// (c/- %1 %2) (distance start dest)) dest start)
        ahead (fn [m] (dot (mapv c/- m start) u))
        is (range (count ms))
        sum-i2 (dot is is)]
    {:speed (c// (dot is (map ahead ms)) (c/* dt sum-i2))
     :sd (c// noise (c/* dt (Math/sqrt sum-i2)))}))

(def pinned (straight-speed start in-view measurements))

pinned
;=> {:speed 0.4452 :sd 0.005923}

;; book: speed-pinned-check
(let [{:keys [speed sd]} pinned
      h 0.02
      [below at above] (map #(log-density-at in-view %)
                            [(c/- speed h) speed (c/+ speed h)])
      curvature (c// (c/+ below (c/* -2.0 at) above) (c/* h h))]
  [(> at (max below above))
   (< (abs (c/+ 1.0 (c/* curvature sd sd))) 1e-3)])
;=> [true true]

;; book: speed-odds
(c// (c/* 4.0 (:sd pinned)) 0.7)
;=> 0.0338

;; book: bins
(def n-bins 5)

(defn bin-probs
  "The heuristic's probabilities of n equal bins of [lo, hi], for travel
   from coordinate first to coordinate last: score-high for the bin of
   last and the bins beyond it in the direction of travel, 1 for the
   others, normalized."
  [n lo hi first last score-high]
  (let [bin (min (dec n) (long (Math/floor (c// (c/- last lo)
                                                 (c// (c/- hi lo) n)))))
        ahead? (fn [i] (if (>= last first) (>= i bin) (<= i bin)))
        scores (array/where (array/array (mapv ahead? (range n)))
                            score-high 1.0)]
    (/ scores (array/sum scores))))

;; book: bins-checks
(let [[x-first y-first] (first measurements)
      [x-last y-last] (peek measurements)]
  [(bin-probs n-bins 0.0 1.0 x-first x-last 5.0)
   (array/allclose? (bin-probs n-bins 0.0 1.0 y-first y-last 5.0)
                    (/ (array/array [1.0 1.0 5.0 5.0 5.0]) 17.0))])
;=> [[0.2 0.2 0.2 0.2 0.2] true]

;; book: binned-dest
(def binned-dest
  (gen [ms scene score-high]
    (let [{:keys [xmin xmax ymin ymax]} scene
          [x-first y-first] (first ms)
          [x-last y-last] (peek ms)]
      (trace :dest-x (d/piecewise-uniform
                      (array/linspace xmin xmax (inc n-bins))
                      (bin-probs n-bins xmin xmax x-first x-last score-high)))
      (trace :dest-y (d/piecewise-uniform
                      (array/linspace ymin ymax (inc n-bins))
                      (bin-probs n-bins ymin ymax y-first y-last
                                 score-high))))))

;; book: speed-proposal
(defn speed-guess
  "The agent's speed as the measurements ms suggest: the distance from the
   first to the last over the time between them."
  [ms dt]
  (c// (distance (first ms) (peek ms)) (c/* dt (dec (count ms)))))

(def speed-proposal
  (gen [ms dt]
    (trace :speed (d/truncated-normal (speed-guess ms dt) 0.01 0.3 1.0))))

(speed-guess measurements dt)
;=> 0.4384

;; book: heuristic-proposal
(defn with-speed
  "A proposal of the destination by dest-q, a generative function of the
   measurements and the scene, and of the speed by speed-proposal."
  [dest-q]
  (gen [ms scene dt]
    (splice dest-q ms scene)
    (splice speed-proposal ms dt)))

(def heuristic-dest
  (gen [ms scene] (splice binned-dest ms scene 5.0)))

(def heuristic-proposal (with-speed heuristic-dest))

;; book: heuristic-draws
(def heuristic-draws
  (mapv #(:choices (gfi/propose heuristic-proposal proposal-args %))
        (rng/split (rng/key 8) 1000)))

(let [up (count (filter #(>= (array/item (cm/get-choice % :dest-y)) 0.4)
                        heuristic-draws))
      p (c// 15.0 17.0)
      se (Math/sqrt (c// (c/* p (c/- 1.0 p)) 1000))]
  (< (abs (c/- (c// up 1000.0) p)) (c/* 5 se)))
;=> true

;; book: weight-by-hand
(let [{t :trace weight :weight}
      (is/weighted-sample agent-model model-args observed
                          {:proposal heuristic-proposal
                           :proposal-args proposal-args}
                          (rng/key 9))
      proposed (cm/choicemap :dest-x (t :dest-x) :dest-y (t :dest-y)
                             :speed (t :speed))
      q (:weight (gfi/assess heuristic-proposal proposal-args proposed
                             (rng/key 0)))]
  (array/allclose? weight (- (:score t) q)))
;=> true

;; book: heuristic-runs
(def heuristic-runs
  (mapv #(sample-run heuristic-proposal n-samples %)
        (rng/split (rng/key 10) 80)))

(mean-se (map :ess heuristic-runs))
;=> {:mean 1.96 :se 0.091} ; float64

;; book: beats
(defn gap-in-se
  "How many standard errors of their difference the mean of the numbers as
   lies above the mean of the numbers bs."
  [as bs]
  (let [{ma :mean sa :se} (mean-se as)
        {mb :mean sb :se} (mean-se bs)]
    (c// (c/- ma mb) (Math/sqrt (c/+ (c/* sa sa) (c/* sb sb))))))

(> (gap-in-se (map :ess heuristic-runs) (map :ess prior-runs)) 5)
;=> true

;; book: posterior-figure
;; figure: figures/051-posterior.svg
(svg/shapes [{:polygons (map corners (:obstacles scene))}
             {:points (map :dest prior-runs) :color "gray" :r 3}
             {:points (map :dest heuristic-runs) :color "red" :r 3}
             {:points measurements :color "black" :r 1.5}
             {:points [start] :r 4}])

;; book: examples
(defn example
  "A training example from a run of the model with key whose planning
   succeeded: the proposal's arguments (the measurements and the scene)
   and the destination, as constraints. nil when planning failed."
  [key]
  (let [t (gfi/simulate agent-model model-args key)]
    (when (:retval t)
      [[(measured t) scene]
       (cm/choicemap :dest-x (t :dest-x) :dest-y (t :dest-y))])))

(defn examples
  "n training examples, from runs of the model whose planning succeeded."
  [n key]
  (vec (take n (keep example (rng/split key (* 2 n))))))

(defn picker
  "A function of a key that picks one of the examples uniformly: the
   data of gen.learning/train."
  [examples]
  (fn [key] (examples (array/item (rng/randint key 0 (count examples))))))

(def training (examples 500 (rng/key 11)))

(def held-out (examples 200 (rng/key 12)))

;; book: score-high
(def trainable-dest
  (gen [ms scene]
    (splice binned-dest ms scene (array/exp (param :log-score-high)))))

(defn mean-log-density
  "The mean log density of the destinations of the examples under the
   proposal q: the objective of gen.learning/train."
  [q examples]
  (c// (reduce c/+ (map (fn [[args constraints]]
                          (array/item (:weight (gfi/assess q args constraints
                                                           (rng/key 0)))))
                        examples))
       (count examples)))

;; book: train-score-high
(def score-high-fit
  (learn/train trainable-dest {:log-score-high 0.0} (learn/adam {:lr 0.05})
               (picker training) grad/param-gradients
               {:epochs 20 :epoch-size 250 :batches-per-epoch 5 :batch-size 50
                :key (rng/key 13)}))

(def learned-score-high
  (Math/exp (array/item (:log-score-high (:store score-high-fit)))))

learned-score-high
;=> 11.0 ; float64

;; book: training-figure
;; figure: figures/051-training.svg
(svg/series (:history score-high-fit)
            {:x-label "epoch" :y-label "mean log density of 250 examples"})

;; book: score-high-checks
(def trained-dest (dyn/with-params trainable-dest (:store score-high-fit)))

(def uniform-dest
  (gen [ms scene] (splice binned-dest ms scene 1.0)))

(mapv #(mean-log-density % held-out)
      [uniform-dest heuristic-dest trained-dest])
;=> [0.0 0.8015 0.889] ; float64

;; spreads over ten trainings, each with its own 500 examples (keys 201
;; to 210) and steps (keys 301 to 310): 1.03 of the learned score-high,
;; 0.00075 of the trained proposal's held-out density
(let [guess (mean-log-density heuristic-dest held-out)
      trained (mean-log-density trained-dest held-out)]
  [(> (c/- learned-score-high 1.0) (c/* 5 1.03))
   (> (c/- trained guess) (c/* 5 0.00075))])
;=> [true true]

;; book: network
(defn features
  "The network's inputs: the first and last measurements of ms, scaled to
   [-0.5, 0.5] across scene."
  [ms scene]
  (let [{:keys [xmin xmax ymin ymax]} scene
        scale (fn [[x y]] [(c/- (c// (c/- x xmin) (c/- xmax xmin)) 0.5)
                           (c/- (c// (c/- y ymin) (c/- ymax ymin)) 0.5)])]
    (array/array (into (scale (first ms)) (scale (peek ms))))))

(defn bin-logits
  "The network: a hidden layer of tanh units over the inputs, then the
   logits of the destination's x bins and of its y bins."
  [{:keys [w1 b1 wx bx wy by]} inputs]
  (let [hidden (array/tanh (+ (array/matmul w1 inputs) b1))]
    [(+ (array/matmul hidden wx) bx) (+ (array/matmul hidden wy) by)]))

(def neural-dest
  (gen [ms scene]
    (let [{:keys [xmin xmax ymin ymax]} scene
          weights {:w1 (param :w1) :b1 (param :b1)
                   :wx (param :wx) :bx (param :bx)
                   :wy (param :wy) :by (param :by)}
          [x-logits y-logits] (bin-logits weights (features ms scene))]
      (trace :dest-x (d/piecewise-uniform
                      (array/linspace xmin xmax (inc n-bins))
                      (array/softmax x-logits)))
      (trace :dest-y (d/piecewise-uniform
                      (array/linspace ymin ymax (inc n-bins))
                      (array/softmax y-logits))))))

(defn initial-weights
  "Random weights for h hidden units, each layer's scaled by one over the
   root of its number of inputs, and zero biases."
  [h key]
  (let [[k1 kx ky] (rng/split key 3)]
    {:w1 (/ (rng/normal k1 [h 4]) 2.0)
     :b1 (array/zeros [h])
     :wx (/ (rng/normal kx [h n-bins]) (Math/sqrt h))
     :bx (array/zeros [n-bins])
     :wy (/ (rng/normal ky [h n-bins]) (Math/sqrt h))
     :by (array/zeros [n-bins])}))

;; book: train-network
(def network-fit
  (learn/train neural-dest (initial-weights 16 (rng/key 14))
               (learn/adam {:lr 0.03}) (picker training) grad/param-gradients
               {:epochs 40 :epoch-size 500 :batches-per-epoch 25 :batch-size 20
                :key (rng/key 15)}))

(def network-proposal (dyn/with-params neural-dest (:store network-fit)))

(mean-log-density network-proposal held-out)
;=> 0.810 ; float64

;; book: network-probs
(mapv array/softmax (bin-logits (:store network-fit)
                                (features measurements scene)))
;=> [[0.560 0.269 0.139 0.016 0.017] [0.004 0.041 0.314 0.353 0.288]] ; float64

;; book: proposals-figure
;; figure: figures/051-proposals.svg
(let [draws (fn [q key]
              (mapv #(let [cs (:choices (gfi/propose q [measurements scene] %))]
                       [(cm/get-choice cs :dest-x) (cm/get-choice cs :dest-y)])
                    (rng/split key 300)))]
  (svg/shapes [{:polygons (map corners (:obstacles scene))}
               {:points (draws heuristic-dest (rng/key 16))
                :color "darkorange" :r 2}
               {:points (draws network-proposal (rng/key 17)) :r 2}
               {:points measurements :color "black" :r 1.5}]))

;; book: held-out
(mapv #(mean-log-density % held-out)
      [heuristic-dest trained-dest network-proposal])
;=> [0.8015 0.889 0.8098] ; float64

;; book: network-checks
;; spreads over ten trainings, each with its own 500 examples (keys 201
;; to 210), initial weights (keys 401 to 410) and steps (keys 501 to
;; 510): 0.078 of the network's held-out density, 0.078 of its gap to
;; the trained heuristic's
(let [trained (mean-log-density trained-dest held-out)
      network (mean-log-density network-proposal held-out)]
  [(> network (c/* 5 0.078))
   (< (abs (c/- network trained)) (c/* 5 0.078))])
;=> [true true]

;; book: network-runs
(def network-runs
  (mapv #(sample-run (with-speed network-proposal) n-samples %)
        (rng/split (rng/key 18) 40)))

(mean-se (map :ess network-runs))
;=> {:mean 3.15 :se 0.19} ; float64

[(> (gap-in-se (map :ess network-runs) (map :ess prior-runs)) 5)
 (gap-in-se (map :ess network-runs) (map :ess heuristic-runs))]
;=> [true 5.75] ; float64

;; book: solution-1
(def dest-only
  (gen [ms scene dt] (splice heuristic-dest ms scene)))

(def dest-only-runs
  (mapv #(sample-run dest-only n-samples %) (rng/split (rng/key 20) 20)))

(mean-se (map :ess dest-only-runs))
;=> {:mean 1.067 :se 0.043} ; float64

(> (gap-in-se (map :ess heuristic-runs) (map :ess dest-only-runs)) 5)
;=> true

;; book: solution-2
(def two-doors
  (update scene :obstacles
          #(-> %
               (assoc 3 (wall :horizontal [0.20 0.40] 0.15 thickness))
               (conj (wall :horizontal [0.45 0.40] (c/+ 0.15 thickness)
                           thickness)))))

(defn in-room
  "The log densities of the observations with the destination [0.4 0.6]
   in the room and the pinned speed, in scene, for 40 planner draws."
  [scene key]
  (let [choices (cm/merge observed (cm/choicemap :dest-x 0.4 :dest-y 0.6
                                                 :speed (:speed pinned)))]
    (mapv #(array/item (:weight (gfi/generate agent-model
                                              [scene dt n-ticks planner]
                                              choices %)))
          (rng/split key 40))))

(def one-door-densities (in-room scene (rng/key 21)))

(def two-door-densities (in-room two-doors (rng/key 21)))

;; book: solution-2-check
(defn fraction-se
  "The fraction of the booleans bs that are true, and its standard error."
  [bs]
  (let [n (count bs)
        p (c// (double (count (filter true? bs))) n)]
    {:mean p :se (Math/sqrt (c// (c/* p (c/- 1.0 p)) n))}))

(let [{p1 :mean s1 :se} (fraction-se (map #(> % 0.0) one-door-densities))
      {p2 :mean s2 :se} (fraction-se (map #(> % 0.0) two-door-densities))]
  [p1 p2 (> (c/- p1 p2) (c/* 5 (Math/sqrt (c/+ (c/* s1 s1) (c/* s2 s2)))))])
;=> [0.675 0.025 true] ; float64

;; book: solution-3
(def tutorial-refining (assoc planner :refine-iters 3500 :refine-sd 1.0))

(def length-pairs
  (mapv (fn [key]
          (mapv #(path-length (plan-path scene start [0.5 0.5] % key))
                [tutorial-refining planner]))
        (rng/split (rng/key 22) 40)))

(let [{:keys [mean se]} (mean-se (map (fn [[a b]] (c/- a b)) length-pairs))]
  [mean (> mean (c/* 5 se))])
;=> [0.135 true] ; float64

;; book: solution-4
(let [choices (cm/choicemap :start-x 0.9 :start-y 0.1 :dest-x 0.9 :dest-y 0.8
                            :speed 0.5)
      t (:trace (gfi/generate agent-model model-args choices (rng/key 24)))]
  [(:retval t) (point-along (:retval t) (c/* 0.5 dt 9))])
;=> [[[0.9 0.1] [0.9 0.8]] [0.9 0.55]]
