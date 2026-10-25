;; The code of book/014-handler-loop.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.handler-loop
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.trace :as trace]))

;; book: model
(def two-normals
  (gen []
    (let [x (trace :x (d/normal 0 1))
          y (trace :y (d/normal x 1))]
      (+ x y))))

(def tr (gfi/simulate two-normals [] (rng/key 1)))

(tr :x)
;=> 0.716 ; float64

(let [obs (cm/choicemap :x 3.0)]
  ((:trace (gfi/generate two-normals [] obs (rng/key 1))) :x))
;=> 3.0

(:weight (gfi/assess two-normals [] (cm/choicemap :x 3.0 :y 2.0) (rng/key 1)))
;=> -6.8379

;; book: transition
(def simulate-site (dyn/site-transitions :simulate))

(def empty-state (dyn/initial-state :simulate))

(def step (simulate-site empty-state [:x] (d/normal 0 1) (rng/key 2)))

(let [[x state'] step]
  [x (:scores state') (:score state')])
;=> [-0.192 {[:x] -0.937} -0.937] ; float64

(let [[x state'] step
      [x-again _] (simulate-site empty-state [:x] (d/normal 0 1) (rng/key 2))]
  [(array/equal? x (cm/get-choice (:choices state') :x))
   (array/allclose? (:score state') (dc/log-prob (d/normal 0 1) x))
   (cm/empty? (:choices empty-state))
   (array/equal? x x-again)])
;=> [true true true true]

;; book: generate-transition
(def generate-site (dyn/site-transitions :generate))

(def observing-x
  (dyn/initial-state :generate {:constraints (cm/choicemap :x 3.0)}))

(let [[x state'] (generate-site observing-x [:x] (d/normal 0 1) (rng/key 2))]
  [x (:weight state')])
;=> [3.0 -5.4189]

(let [dist (d/normal 0 1)
      [v state'] (generate-site observing-x [:y] dist (rng/key 2))
      [v' state''] (simulate-site observing-x [:y] dist (rng/key 2))]
  [(array/equal? v v') (array/equal? (:score state') (:score state''))
   (:weight state')])
;=> [true true 0.0]

;; book: assess-error
;; error: assess: no value for the choice at [:y]
#_(gfi/assess two-normals [] (cm/choicemap :x 3.0) (rng/key 3))

;; book: watched
(defn watched
  "model with every operation's transition wrapped to keep each state it
   receives, in order, under ::seen, a key of this namespace, which the
   run returns on its trace. The transitions it wraps are the handler's
   own."
  [model]
  (dyn/with-transitions model
    (into {} (for [[op transition] dyn/site-transitions]
               [op (fn [state p dist k]
                     (let [[v state'] (transition state p dist k)]
                       [v (update state' ::seen (fnil conj []) state)]))]))))

(def watched-run
  (gfi/generate (watched two-normals) [] (cm/choicemap :x 3.0) (rng/key 1)))

(def seen (::seen (:trace watched-run)))

(mapv (juxt :mode :score :weight) seen)
;=> [[:generate 0.0 0.0] [:generate -5.4189 -5.4189]]

;; book: state-keys
(= (set (filter simple-keyword? (keys (first seen))))
   #{:mode :key :choices :scores :score :weight :constraints :prev
     :selection :argdiffs :discard :subtraces :visited
     :omega :omega-key :omega-count})
;=> true

(= (set (keys (first seen))) (set (keys (dyn/initial-state :generate))))
;=> true

;; book: own-keys
(let [plain (gfi/generate two-normals [] (cm/choicemap :x 3.0) (rng/key 1))]
  [(= (:choices (:trace plain)) (:choices (:trace watched-run)))
   (= (:weight plain) (:weight watched-run))])
;=> [true true]

(def updated
  (:trace (gfi/update (:trace watched-run) (cm/choicemap :y 1.0)
                      (rng/key 2))))

(mapv :mode (::seen updated))
;=> [:update :update]

(identical? (:trace watched-run) (:prev (first (::seen updated))))
;=> true

;; book: handler-key-error
;; error: The transition at [:x] changed the handler's :key
#_(gfi/simulate
   (dyn/with-transitions two-normals
     {:simulate (fn [state p dist k]
                  (let [[v state'] (simulate-site state p dist k)]
                    [v (update state' :key #(first (rng/split %)))]))})
   [] (rng/key 1))

;; book: own-keys-error
;; error: Middleware left the key :seen in the handler's state
#_(gfi/simulate
   (dyn/with-transitions two-normals
     {:simulate (fn [state p dist k]
                  (let [[v state'] (simulate-site state p dist k)]
                    [v (assoc state' :seen p)]))})
   [] (rng/key 1))

;; book: fold
(defn site-states
  "Every state of threading state through transition at each [path dist]
   of sites, with one key each from key's split: state itself first."
  [transition state sites key]
  (reductions (fn [s [[p dist] k]] (second (transition s p dist k)))
              state
              (map vector sites (rng/split key (count sites)))))

(def sites [[[:slope] (d/normal 0 10)] [[:noise] (d/normal 0 1)]])

(def states (site-states simulate-site empty-state sites (rng/key 4)))

(mapv :score states)
;=> [0.0 -3.626 -4.728] ; float64

(defn log-density
  "The sum of the log densities of state's choices at sites."
  [state sites]
  (reduce + (map (fn [[p dist]]
                   (dc/log-prob dist (cm/get-choice (:choices state) p)))
                 sites)))

(array/allclose? (:score (last states)) (log-density (last states) sites))
;=> true

;; book: loop
(defn run-body
  "Run body, a function of trace, under transition from state, as gen.clj's
   handler loop does without its checks: at each site, split the state's
   key, give the site one key and carry the other. The final state, with
   the body's value at :retval."
  [transition state body]
  (let [vol (volatile! state)
        trace (fn [addr dist]
                (let [[k-next k-site] (rng/split (:key @vol))
                      [v state'] (transition (assoc @vol :key k-next)
                                             (cm/path addr) dist k-site)]
                  (vreset! vol state')
                  v))
        retval (body trace)]
    (assoc @vol :retval retval)))

;; book: by-hand
(defn two-normals-body
  "two-normals' body, as a function of trace."
  [trace]
  (let [x (trace :x (d/normal 0 1))
        y (trace :y (d/normal x 1))]
    (+ x y)))

(defn same-run?
  "Do a and b (traces or handler states) hold the same choices, score and
   return value, exactly?"
  [a b]
  (let [ca (:choices a) cb (:choices b)]
    (and (= (cm/addresses ca) (cm/addresses cb))
         (every? true? (map array/equal? (cm/values ca) (cm/values cb)))
         (array/equal? (:score a) (:score b))
         (array/equal? (:retval a) (:retval b)))))

(let [k (rng/key 1)
      start (dyn/initial-state :simulate {:key k})
      obs (cm/choicemap :x 3.0)
      observing (dyn/initial-state :generate {:key k :constraints obs})]
  [(same-run? (run-body simulate-site start two-normals-body)
              (gfi/simulate two-normals [] k))
   (same-run? (run-body generate-site observing two-normals-body)
              (:trace (gfi/generate two-normals [] obs k)))])
;=> [true true]

;; book: twice-error
;; error: Address [:x] is used twice
#_(gfi/simulate (gen [] (trace :x (d/normal 0 1)) (trace :x (d/normal 0 1)))
              [] (rng/key 1))

;; book: unvisited-error
;; error: generate: the constraint at [:z] was not visited
#_(gfi/generate two-normals [] (cm/choicemap :z 1.0) (rng/key 1))

;; book: one-key-per-site
(let [k (rng/key 5)
      t (gfi/simulate two-normals [] k)
      again (:trace (gfi/generate two-normals [] (cm/choicemap :x (t :x)) k))]
  (array/equal? (t :y) (again :y)))
;=> true

;; book: expansion
(= (macroexpand-1 '(gen [b] (if b 1 2)))
   '(gen.dynamic/dynamic-gf
     (fn* ([trace splice param untraced b]
           (let* [untraced-key (fn* ([] (untraced)))]
             (if (gen.array/truthy? b) 1 2))))
     (quote ([b] (if b 1 2)))
     nil))
;=> true

;; book: guard
(def one-false (array/array [false]))

((fn [b] (if b :then :else)) one-false)
;=> :then

(:retval (gfi/simulate (gen [b] (if b :then :else)) [one-false] (rng/key 1)))
;=> :else

;; book: splice
(def measurement
  (gen [mu]
    (trace :obs (d/normal mu 1))))

(def two-level
  (gen []
    (let [theta (trace :theta (d/normal 0 2))
          obs (splice :m measurement theta)]
      {:theta theta :obs obs})))

(def nested (gfi/simulate two-level [] (rng/key 6)))

(cm/addresses (:choices nested))
;=> [[:m :obs] [:theta]]

(array/allclose? (:score nested)
                 (+ (dc/log-prob (d/normal 0 2) (nested :theta))
                    (dc/log-prob (d/normal (nested :theta) 1)
                                 (nested [:m :obs]))))
;=> true

;; book: subtrace
(let [[k-next _] (rng/split (rng/key 6))
      [_ k-call] (rng/split k-next)]
  (same-run? (trace/subtrace nested :m)
             (gfi/simulate measurement [(nested :theta)] k-call)))
;=> true

;; book: batched
(def particles (batched/simulate two-normals [] 1000 (rng/key 7)))

[(batched/batched? particles) (array/shape (batched/scores particles))]
;=> [true [1000]]

(let [t (nth (batched/traces particles) 17)]
  (array/allclose? (array/take (batched/scores particles) 17)
                   (:weight (gfi/assess two-normals [] (:choices t)
                                        (rng/key 0)))))
;=> true

;; book: static
(def two-normals-static
  (gen ^:static []
    (let [x (trace :x (d/normal 0 1))
          y (trace :y (d/normal x 1))]
      (+ x y))))

(let [k (rng/key 9)
      choices (cm/choicemap :x 0.5 :y 1.0)]
  [(same-run? (gfi/simulate two-normals-static [] k)
              (gfi/simulate two-normals [] k))
   (array/allclose? (:weight (gfi/assess two-normals-static [] choices k))
                    (:weight (gfi/assess two-normals [] choices k)))])
;=> [true true]

;; book: values
(array/shape (:score tr))
;=> []

(identical? tr (gfi/eval-trace! tr))
;=> true

;; book: implicit-key
(defn two-draws
  "Two simulated values of :x, after seeding the implicit key source."
  [seed]
  (gfi/seed! seed)
  [((gfi/simulate two-normals []) :x) ((gfi/simulate two-normals []) :x)])

(def before (gfi/implicit-seed))

(let [draws (two-draws 14)
      again (two-draws 14)]
  (gfi/seed! before)
  [(array/equal? (first draws) (second draws))
   (every? true? (map array/equal? draws again))])
;=> [false true]

;; book: solution-1
(let [obs (cm/choicemap :y 2.0)
      {t :trace w :weight} (gfi/generate two-normals [] obs (rng/key 10))]
  (array/allclose? w (dc/log-prob (d/normal (t :x) 1) 2.0)))
;=> true

;; book: solution-2
(defn run-body-sparing
  "run-body, except that a site the state's constraints hold takes no key,
   as in GenMLX."
  [transition state body]
  (let [vol (volatile! state)
        trace (fn [addr dist]
                (let [p (cm/path addr)
                      [k-next k-site] (if (cm/has-value? (:constraints @vol) p)
                                        [(:key @vol) nil]
                                        (rng/split (:key @vol)))
                      [v state'] (transition (assoc @vol :key k-next)
                                             p dist k-site)]
                  (vreset! vol state')
                  v))
        retval (body trace)]
    (assoc @vol :retval retval)))

(let [k (rng/key 5)
      t (gfi/simulate two-normals [] k)
      start (dyn/initial-state :generate
                               {:key k :constraints (cm/choicemap :x (t :x))})]
  (mapv (fn [run]
          (let [state (run generate-site start two-normals-body)]
            (array/equal? (t :y) (cm/get-choice (:choices state) :y))))
        [run-body run-body-sparing]))
;=> [true false]

;; book: solution-3
(def three-level
  (gen []
    (splice :outer two-level)))

(def deeper (gfi/simulate three-level [] (rng/key 11)))

(cm/addresses (:choices deeper))
;=> [[:outer :m :obs] [:outer :theta]]

(let [theta (deeper [:outer :theta])]
  (array/allclose? (:score deeper)
                   (+ (dc/log-prob (d/normal 0 2) theta)
                      (dc/log-prob (d/normal theta 1)
                                   (deeper [:outer :m :obs])))))
;=> true

;; book: solution-4
(defn pick
  "A branch on b, unguarded."
  [b]
  (if b :then :else))

(dyn/defn-guarded pick-guarded
  "A branch on b's value."
  [b]
  (if b :then :else))

[(:retval (gfi/simulate (gen [b] (pick b)) [one-false] (rng/key 1)))
 (:retval (gfi/simulate (gen [b] (pick-guarded b)) [one-false] (rng/key 1)))]
;=> [:then :else]
