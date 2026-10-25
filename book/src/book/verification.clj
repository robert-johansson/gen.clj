;; The code of book/035-verification.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.verification
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.string :as str]
            [clojure.test.check.generators :as tcg]
            [clojure.test.check.properties :as prop]
            [gen.array :as array :refer [+ - * /]]
            [gen.book.svg :as svg]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :as dyn :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.proposal :as proposal]
            [gen.laws :as laws]
            [gen.random :as rng]
            [gen.random-static-models :as rsm]
            [gen.selection :as sel]
            [gen.trace :as trace]))

;; book: model
(def measurement
  (gen [mu]
    (let [x (trace :x (d/normal mu 1))]
      (trace :y (d/normal x 0.5)))))

(def models
  "The measurement model as a model entry: a label, the generative
   function, its arguments, and other arguments to update and regenerate
   with."
  [[:measurement measurement [0.0] [[1.5] [-2.0]]]])

;; book: one-law
(def score-law (laws/score-is-the-assessed-density (laws/cases models)))

(select-keys (laws/quick-check 20 score-law {:seed 1})
             [:pass? :num-tests :vacuous])
;=> {:pass? true, :num-tests 20, :vacuous 0}

;; book: a-case
(let [[[label] seed picks alt?] (tcg/generate (laws/cases models) 30 7)]
  [label (int? seed) (count picks) (boolean? alt?)])
;=> [:measurement true 32 true]

;; book: verify
(def few
  "The chapter's runs of a law: 20 cases, from one seed."
  {:num-tests 20 :seed 1})

(laws/verify models few)
;=> {}

;; book: sets
(mapv count [laws/general laws/forward-sampling laws/all])
;=> [15 7 22]

;; book: guided
(defn open?
  "Does the choice map sigma leave the address a open?"
  [sigma a]
  (not (cm/has-value? sigma a)))

(def guided
  (proposal/with-internal-proposal measurement
    (gen [sigma mu]
      (when (open? sigma :x)
        (trace :x (d/normal (+ mu 0.5) 1.5))))))

(def guided-models [[:guided guided [0.0] [[1.5] [-2.0]]]])

(laws/verify guided-models few laws/general)
;=> {}

(count (laws/verify guided-models few laws/forward-sampling))
;=> 5

;; book: guided-weight
(let [{t :trace w :weight} (gfi/generate guided [0.0] cm/EMPTY (rng/key 1))
      x (t :x)]
  [(array/allclose? w (- (dc/log-prob (d/normal 0 1) x)
                         (dc/log-prob (d/normal 0.5 1.5) x)))
   (array/allclose? w (gfi/project t sel/none))])
;=> [true true]

;; book: mixture
(def mixture
  (gen []
    (let [i (untraced (d/categorical [0.2 0.5 0.3]))]
      (trace :a (d/normal (array/take (array/array [-2.0 0.0 3.0]) i) 1)))))

(def mixture-models [[:mixture mixture [] []]])

(contains? (laws/verify mixture-models few laws/general)
           :score-is-the-assessed-density)
;=> true

(laws/verify mixture-models few laws/encapsulated)
;=> {}

;; book: support
(def nested
  (gen []
    (let [x (trace :x (d/uniform 0 1))]
      (trace :y (d/uniform 0 x)))))

(def nested-models [[:nested nested [] []]])

(laws/verify nested-models few)
;=> {}

(keys (laws/verify nested-models {:num-tests 200 :seed 1}
                   laws/fixed-support))
;=> (:generate-weights-are-finite-on-a-fixed-support)

;; book: support-quarter
(defn out-of-support?
  "Is the weight -Inf of a generate of nested constrained on the :y of a
   simulated trace, with x drawn anew? The key splits in two, for each."
  [key]
  (let [[k1 k2] (rng/split key)
        y ((gfi/simulate nested [] k1) :y)
        w (:weight (gfi/generate nested [] (cm/choicemap :y y) k2))]
    (= ##-Inf (array/item w))))

(let [n 2000
      share (/ (count (filter out-of-support? (rng/split (rng/key 2) n))) n)]
  (< (abs (- share 0.25)) (* 5 (Math/sqrt (/ (* 0.25 0.75) n)))))
;=> true

;; book: across
(def shifted
  (gen [mu]
    (let [x (trace :x (d/normal (+ mu (param :shift)) 1))]
      (trace :y (d/normal x 0.5)))))

(laws/verify [[:stores (dyn/with-params shifted {:shift 0.0})
               [0.0] [[1.5] [-2.0]]
               (dyn/with-params shifted {:shift 1.0})]]
             few laws/across)
;=> {}

;; book: record
(defrecord HandWritten [parts]
  gfi/IGenerativeFunction
  (-simulate [gf args key] ((:simulate parts) gf args key))
  (-generate [gf args constraints key]
    ((:generate parts) gf args constraints key))
  (-update [gf tr args argdiffs constraints key]
    ((:update parts) gf tr args argdiffs constraints key))
  (-regenerate [gf tr args argdiffs selection key]
    ((:regenerate parts) gf tr args argdiffs selection key))
  (-assess [gf args choices key] ((:assess parts) gf args choices key))
  (-propose [gf args key] ((:propose parts) gf args key))
  (-project [gf tr selection] ((:project parts) gf tr selection)))

;; book: model-parts
(defn draw
  "The values of the measurement model's choices on mu, as a map from
   address to value: those in given, and the others drawn, each from its
   own key split from key."
  [mu given key]
  (let [[kx ky] (rng/split key)
        x (or (:x given) (dc/sample (d/normal mu 1) kx))]
    {:x x :y (or (:y given) (dc/sample (d/normal x 0.5) ky))}))

(defn log-densities
  "The log density of each of the measurement model's choices m on mu, as
   a map from address to log density."
  [mu {:keys [x y]}]
  {:x (dc/log-prob (d/normal mu 1) x)
   :y (dc/log-prob (d/normal x 0.5) y)})

;; book: helpers
(defn values-of
  "The values in the choice map c, as a map from address to value: loud
   for an address the measurement model does not have."
  [op c]
  (let [paths (cm/addresses c)]
    (when-let [others (seq (remove #{[:x] [:y]} paths))]
      (throw (gfi/unvisited op others)))
    (into {} (map (fn [[a]] [a (cm/get-choice c a)])) paths)))

(defn log-density
  "The sum of the log densities of the choices m on mu at addresses (all
   by default), by gf's density."
  ([gf mu m] (log-density gf mu m [:x :y]))
  ([gf mu m addresses]
   (let [{:keys [log-densities]} (:parts gf)]
     (reduce + 0.0 (vals (select-keys (log-densities mu m) addresses))))))

(defn make-trace
  "The trace of gf on args with the choices m."
  [gf [mu :as args] m]
  (trace/make-trace {:gen-fn gf :args args :choices (cm/from-map m)
                     :retval (:y m) :score (log-density gf mu m)}))

;; book: operations
(defn simulate-op
  "simulate: every choice drawn."
  [gf [mu :as args] key]
  (let [{:keys [draw]} (:parts gf)]
    (make-trace gf args (draw mu {} key))))

(defn generate-op
  "generate: the constrained choices kept, the others drawn."
  [gf [mu :as args] constraints key]
  (let [{:keys [draw]} (:parts gf)
        given (values-of :generate constraints)
        m (draw mu given key)]
    {:trace (make-trace gf args m)
     :weight (log-density gf mu m (keys given))}))

(defn update-op
  "update: the constrained choices replaced, nothing drawn."
  [gf tr args argdiffs constraints _]
  (let [given (values-of :update constraints)
        old (values-of :update (:choices tr))
        t (make-trace gf args (merge old given))]
    {:trace t
     :weight (- (:score t) (:score tr))
     :retdiff (if (and (empty? given) (every? diff/no-change? argdiffs))
                diff/no-change
                diff/unknown-change)
     :discard (cm/from-map (select-keys old (keys given)))}))

(defn regenerate-op
  "regenerate: the selected choices drawn again."
  [gf tr [mu :as args] _ selection key]
  (let [{:keys [draw]} (:parts gf)
        old (values-of :regenerate (:choices tr))
        kept (into {} (remove (fn [[a]] (sel/selected? selection a))) old)
        m (draw mu kept key)]
    {:trace (make-trace gf args m)
     :weight (- (log-density gf mu m (keys kept))
                (log-density gf (first (:args tr)) old (keys kept)))
     :retdiff diff/unknown-change}))

(defn assess-op
  "assess: the log density of every choice, each given."
  [gf [mu] choices _]
  (let [m (values-of :assess choices)]
    (doseq [a [:x :y] :when (not (contains? m a))]
      (throw (ex-info (str "assess: no value for the choice at " [a])
                      {:path [a]})))
    {:weight (log-density gf mu m) :retval (:y m)}))

(defn propose-op
  "propose: simulate's choices, with their log density."
  [gf args key]
  (let [t (simulate-op gf args key)]
    {:choices (:choices t) :weight (:score t) :retval (:retval t)}))

(defn project-op
  "project: the log density of the selected choices."
  [gf tr selection]
  (log-density gf (first (:args tr)) (values-of :project (:choices tr))
               (filter #(sel/selected? selection %) [:x :y])))

;; book: by-hand
(def by-hand
  (->HandWritten {:draw draw :log-densities log-densities
                  :simulate simulate-op :generate generate-op
                  :update update-op :regenerate regenerate-op
                  :assess assess-op :propose propose-op :project project-op}))

(defn entry
  "A model entry of gf, on the measurement model's arguments."
  [label gf]
  [label gf [0.0] [[1.5] [-2.0]]])

;; book: by-hand-check
(let [c (cm/choicemap :x 0.3 :y -0.2)]
  (array/allclose? (:weight (gfi/assess by-hand [0.5] c))
                   (:weight (gfi/assess measurement [0.5] c))))
;=> true

(laws/verify [(entry :by-hand by-hand)] few)
;=> {}

(laws/verify [(entry :by-hand by-hand)] few laws/fixed-support)
;=> {}

;; book: by-hand-loud
;; error: generate: the constraint at [:z] was not visited
#_(gfi/generate by-hand [0.0] (cm/choicemap :z 1.0) (rng/key 2))

;; book: deflaws
(laws/deflaws hand [(entry :by-hand by-hand)] 100)

(count (filter (comp ::laws/law meta) (vals (ns-interns 'book.verification))))
;=> 22

(select-keys (hand-update-weight-is-the-projection-change 10 :seed 1)
             [:pass? :num-tests])
;=> {:pass? true, :num-tests 10}

;; book: battery
(defn deflaws-cases
  "The number of cases the deflaws of prefix gives its laws in the test
   file at path, read from the source."
  [path prefix]
  (let [re (re-pattern (str "\\(laws/deflaws " prefix " \\S+ (\\d+)"))]
    (some-> (re-find re (slurp path)) second parse-long)))

[(deflaws-cases "test/gen/dynamic_test.clj" "dynamic")
 (deflaws-cases "test/gen/combinators_test.clj" "combinators")
 (deflaws-cases "test/gen/fuzz_test.clj" "random")
 (deflaws-cases "test/gen/static_fuzz_test.clj" "static-random")]
;=> [100 100 200 300]

(second (re-find #"factor defaults to (\d+)" (slurp "test/gen/soak.clj")))
;=> "10"

;; book: planted
(defn planted
  "gf with its part k replaced by f: a planted bug."
  [gf k f]
  (assoc-in gf [:parts k] f))

(defn simulate-forgetting-y
  "simulate, its score without y's term."
  [gf [mu :as args] key]
  (let [t (simulate-op gf args key)
        m (values-of :simulate (:choices t))]
    (assoc t :score (log-density gf mu m [:x]))))

(defn update-at-old-arguments
  "update, its weight computed at the trace's old arguments."
  [gf tr args argdiffs constraints key]
  (let [r (update-op gf tr args argdiffs constraints key)
        m (values-of :update (:choices (:trace r)))]
    (assoc r :weight (- (log-density gf (first (:args tr)) m) (:score tr)))))

(defn regenerate-by-score
  "regenerate, its weight the score difference: the proposal forgotten."
  [gf tr args argdiffs selection key]
  (let [r (regenerate-op gf tr args argdiffs selection key)]
    (assoc r :weight (- (:score (:trace r)) (:score tr)))))

(defn draw-wide
  "draw, with x from N(mu, 2) where the density says N(mu, 1)."
  [mu given key]
  (let [[kx ky] (rng/split key)
        x (or (:x given) (dc/sample (d/normal mu 2) kx))]
    {:x x :y (or (:y given) (dc/sample (d/normal x 0.5) ky))}))

(defn update-checking-keywords
  "update, its check of the constraints written with keywords where
   cm/addresses gives paths: every constraint is unvisited."
  [gf tr args argdiffs constraints key]
  (when-let [others (seq (remove #{:x :y} (cm/addresses constraints)))]
    (throw (gfi/unvisited :update others)))
  (update-op gf tr args argdiffs constraints key))

(def plants
  "The planted bugs: a name, the part each replaces and its replacement."
  [[:score-forgets-y :simulate simulate-forgetting-y]
   [:update-at-old-args :update update-at-old-arguments]
   [:regenerate-by-score :regenerate regenerate-by-score]
   [:wide-sampler :draw draw-wide]
   [:keyword-check :update update-checking-keywords]])

;; book: caught
(def caught
  "For each planted bug, the laws that fail on it."
  (into {} (for [[name k f] plants
                 :let [models [(entry name (planted by-hand k f))]]]
             [name (set (keys (laws/verify models few)))])))

(mapv (comp count caught first) plants)
;=> [6 2 4 0 3]

;; book: planted-figure
;; figure: figures/035-planted.svg
(let [names (concat (sort (keys laws/general))
                    (sort (keys laws/forward-sampling)))]
  (svg/grid (map name names)
            (map (comp name first) plants)
            (for [law names]
              (for [[plant] plants] (contains? (caught plant) law)))
            {:rules [(dec (count laws/general))]}))

;; book: score-failure
(def forgets-y (planted by-hand :simulate simulate-forgetting-y))

(def score-failure
  (:score-is-the-assessed-density
   (laws/verify [(entry :forgets-y forgets-y)] few)))

(let [{:keys [model key-seed picks]} (:case score-failure)]
  [model key-seed (not-any? true? picks)])
;=> [:forgets-y 0 true]

(let [{:keys [check score assessed]} score-failure]
  [check (> (array/item score) (array/item assessed))])
;=> [(consistent? (gfi/simulate gf args (rng/key seed))) true]

;; book: old-args
(= (caught :update-at-old-args)
   #{:update-weight-is-the-projection-change
     :update-weight-is-the-score-change-less-fresh-choices})
;=> true

(def at-old-args (planted by-hand :update update-at-old-arguments))

(laws/verify [[:same-args at-old-args [0.0] []]] few)
;=> {}

(keys (laws/verify [(entry :old-args at-old-args)] few laws/general))
;=> (:update-weight-is-the-projection-change)

;; book: by-score
(= (caught :regenerate-by-score)
   #{:regenerate-weight-is-the-projection-ratio
     :regenerate-weight-is-the-retained-density-ratio
     :regenerate-weight-with-new-arguments-is-the-projection-ratio
     :regenerating-everything-has-weight-zero})
;=> true

;; book: coins
(def coins
  (gen [prob-a]
    (let [a (trace :a (d/bernoulli prob-a))
          b (when a (trace :b (d/bernoulli 0.6)))
          val (if a b true)
          c (trace :c (d/bernoulli (if val 0.9 0.2)))]
      (and val c))))

(def coin-models [[:coins coins [0.3] [[0.6]]]])

(def reversible (laws/updates-are-reversible-with-exact-discards
                 (laws/cases coin-models)))

(select-keys (laws/quick-check 100 reversible {:seed 1}) [:pass? :vacuous])
;=> {:pass? true, :vacuous 6} ; float64

;; book: unvisitable
;; error: update: the constraint at [:b] was not visited
#_(let [t (:trace (gfi/generate coins [0.3] (cm/choicemap :a false)
                              (rng/key 3)))]
  (gfi/update t (cm/choicemap :b true) (rng/key 4)))

;; book: keyword-check
(def keyword-check (planted by-hand :update update-checking-keywords))

(def keyword-law
  (laws/update-weight-is-the-projection-change
   (laws/cases [(entry :keyword-check keyword-check)])))

(:result-data (laws/quick-check 20 keyword-law {:seed 1}))
;=> {:vacuous 15, :of 20, :max-vacuous 0.6}

(:pass? (laws/quick-check 20 keyword-law {:seed 1 :max-vacuous 1.0}))
;=> true

;; book: max-vacuous
laws/max-vacuous
;=> 0.6

(/ (- laws/max-vacuous 0.32) (Math/sqrt (/ (* 0.32 0.68) 200)))
;=> 8.5

;; book: evidence
(defn density-at-0
  "The density at 0 of N(0, sd)."
  [sd]
  (array/item (array/exp (dc/log-prob (d/normal 0 sd) 0.0))))

(def exact-evidence
  "p(y = 0) on mu = 0: y's marginal is N(mu, sqrt(1 + 0.5^2))."
  (density-at-0 (Math/sqrt 1.25)))

exact-evidence
;=> 0.3568

(defn evidence-draws
  "exp(weight) of n generates of gf on mu = 0 with y fixed to 0, as host
   numbers: draws whose mean estimates p(y = 0)."
  [gf n key]
  (mapv (fn [k]
          (let [{w :weight} (gfi/generate gf [0.0] (cm/choicemap :y 0.0) k)]
            (array/item (array/exp w))))
        (rng/split key n)))

;; book: within
(defn mean-se
  "The mean of the host numbers xs, and its standard error."
  [xs]
  (let [n (count xs)
        m (/ (reduce + xs) n)
        v (/ (reduce + (map #(* (- % m) (- % m)) xs)) (dec n))]
    [m (Math/sqrt (/ v n))]))

(defn within-5-se?
  "Is the mean of xs within five standard errors of value?"
  [xs value]
  (let [[m se] (mean-se xs)]
    (< (abs (- m value)) (* 5 se))))

(def right-draws (evidence-draws by-hand 2000 (rng/key 5)))

(def wide-draws
  (evidence-draws (planted by-hand :draw draw-wide) 2000 (rng/key 6)))

[(within-5-se? right-draws exact-evidence)
 (within-5-se? wide-draws exact-evidence)]
;=> [true false]

;; book: band-width
(< (* 5 (second (mean-se right-draws))) (* 0.1 exact-evidence))
;=> true

;; book: wide-target
(within-5-se? wide-draws (density-at-0 (Math/sqrt 4.25)))
;=> true

;; book: evidence-figure
;; figure: figures/035-evidence.svg
(defn running-means
  "[n, the mean of the first n of xs] for n from 20 on, every 10th."
  [xs]
  (let [sums (vec (reductions + xs))]
    (for [n (range 20 (inc (count xs)) 10)]
      [n (/ (sums (dec n)) n)])))

(let [sd (* (second (mean-se right-draws)) (Math/sqrt (count right-draws)))
      band (fn [sign]
             (for [n (range 20 2001 10)]
               [n (+ exact-evidence (* sign 5 (/ sd (Math/sqrt n))))]))]
  (svg/paths [{:paths [(band 1) (band -1)] :color "gray" :width 1}
              {:paths [[[20 exact-evidence] [2000 exact-evidence]]]
               :color "black" :width 1}
              {:paths [(running-means right-draws)]}
              {:paths [(running-means wide-draws)] :color "firebrick"}]
             {:x-label "generates" :y-label "estimate of p(y = 0)"}))

;; book: static
(def static-measurement
  (gen ^:static [mu]
    (let [x (trace :x (d/normal mu 1))
          y (trace :y (d/normal x 0.5))]
      y)))

(laws/verify [(entry :static static-measurement)] few laws/static)
;=> {}

;; book: grafted
(def wider-noise
  (gen ^:static [mu]
    (let [x (trace :x (d/normal mu 1))
          y (trace :y (d/normal x 1.0))]
      y)))

(def grafted (assoc static-measurement :static (:static wider-noise)))

(keys (laws/verify [(entry :grafted grafted)] few laws/static))
;=> (:static-matches-dynamic)

;; book: random-model
(def random-model (tcg/generate rsm/gen-model 0 1))

(= (:form (first random-model))
   '(dyn/with-params
      (gen ^:static [i0 xs1 [u2 v2]]
        (let [b0 (trace [:b0 i0] (d/laplace v2 0.7))]
          (trace :ret (d/normal b0 1.0))))
      {:w 1.5}))
;=> true

(subvec random-model 2)
;=> [[2 [0.25] [-1.0 -0.75]] [[2 [0.25] [-1.0 -0.75]] [2 [0.25] [-1.0 1.75]]]]

;; book: fuzz
(def static-law (laws/static-matches-dynamic (laws/cases rsm/gen-model)))

(select-keys (laws/quick-check 20 static-law {:seed 1}) [:pass? :num-tests])
;=> {:pass? true, :num-tests 20}

;; book: lazy-error
;; error: Syntax error compiling
#_(gen [xs]
  (map (fn [x] (trace [:y x] (d/normal x 1))) xs)
  xs)

;; book: compile-error
(defn compile-error
  "The message of the error that compiling form raises (the cause the
   compiler reports), or nil when it compiles."
  [form]
  (try (eval form)
       nil
       (catch clojure.lang.Compiler$CompilerException e
         (ex-message (ex-cause e)))))

(= (compile-error '(gen [xs]
                     (map (fn [x] (trace [:y x] (d/normal x 1))) xs)
                     xs))
   (str "In gen: map here is lazy and its value is discarded, so its trace"
        " calls would never run. Use doseq, run! or mapv."))
;=> true

(str/starts-with?
 (compile-error '(gen ^:static [xs]
                   (let [s (trace :s (d/normal 0 1))]
                     (doseq [x xs] (trace [:y x] (d/normal s 1)))
                     s)))
 "gen ^:static: the let returns one expression")
;=> true

;; book: plain-map
;; error: generate: the constraints must be a choice map
#_(gfi/generate measurement [0.0] {:y 1.0} (rng/key 7))

;; book: argdiffs
;; error: update: 1 arguments but 0 argdiffs
#_(gfi/update (gfi/simulate measurement [0.0] (rng/key 8))
            [1.0] [] cm/EMPTY (rng/key 9))

;; book: rand
(def with-rand
  (gen [mu]
    (let [x (trace :x (d/normal (+ mu (rand)) 1))]
      (trace :y (d/normal x 0.5)))))

(every? (laws/verify [(entry :with-rand with-rand)] few)
        [:runs-are-pure-functions-of-the-key :score-is-the-assessed-density])
;=> true

;; book: solution-1
(defn project-forgetting-y
  "project, without y's term."
  [gf tr selection]
  (log-density gf (first (:args tr)) (values-of :project (:choices tr))
               (filter #(and (= :x %) (sel/selected? selection %)) [:x :y])))

(def project-forgets-y (planted by-hand :project project-forgetting-y))

(= (set (keys (laws/verify [(entry :project-forgets-y project-forgets-y)] few)))
   #{:generate-weight-is-the-constrained-density
     :generate-weight-is-the-projection
     :project-of-everything-is-the-score
     :project-splits-the-score
     :regenerate-weight-is-the-projection-ratio
     :regenerate-weight-is-the-retained-density-ratio
     :regenerate-weight-with-new-arguments-is-the-projection-ratio
     :update-weight-is-the-projection-change})
;=> true

;; book: solution-2
(defn update-to-its-own-choices-changes-nothing
  "A law: updating a trace with its own choices as the constraints has
   weight 0, keeps them, and discards them all."
  [cases]
  (prop/for-all [[[_ gf args] seed] cases]
    (let [tr (gfi/simulate gf args (rng/key seed))
          {t :trace w :weight discard :discard}
          (gfi/update tr (:choices tr) (rng/key (inc seed)))]
      (laws/checks (laws/close? w 0.0)
                   (= (:choices tr) (:choices t))
                   (= (:choices tr) discard)))))

(defn holds?
  "Does the law hold on the chapter's cases of models?"
  [law models]
  (= {} (laws/verify models few {:own law})))

[(holds? update-to-its-own-choices-changes-nothing models)
 (holds? update-to-its-own-choices-changes-nothing [(entry :by-hand by-hand)])]
;=> [true true]

(mapv (fn [[name k f]]
        (holds? update-to-its-own-choices-changes-nothing
                [(entry name (planted by-hand k f))]))
      plants)
;=> [false true true true false]

;; book: solution-3
(defn draw-1
  "draw, with y from N(x, 1)."
  [mu given key]
  (let [[kx ky] (rng/split key)
        x (or (:x given) (dc/sample (d/normal mu 1) kx))]
    {:x x :y (or (:y given) (dc/sample (d/normal x 1) ky))}))

(defn log-densities-1
  "log-densities, with y from N(x, 1)."
  [mu {:keys [x y]}]
  {:x (dc/log-prob (d/normal mu 1) x)
   :y (dc/log-prob (d/normal x 1) y)})

(def noise-1
  (-> by-hand (planted :draw draw-1) (planted :log-densities log-densities-1)))

(laws/verify [(entry :noise-1 noise-1)] few)
;=> {}

(within-5-se? (evidence-draws noise-1 2000 (rng/key 10)) exact-evidence)
;=> false

;; book: solution-4
(defn evidence-by-quadrature
  "p(y = 0) on mu = 0 by gf's assess: the joint density of x and y = 0,
   summed over a grid of x."
  [gf]
  (let [dx 0.01
        log-joint (fn [x]
                    (:weight (gfi/assess gf [0.0] (cm/choicemap :x x :y 0.0))))]
    (* dx (reduce + (map #(array/item (array/exp (log-joint %)))
                         (range -8.0 8.0 dx))))))

(array/allclose? (evidence-by-quadrature by-hand) exact-evidence)
;=> true

[(within-5-se? right-draws (evidence-by-quadrature by-hand))
 (within-5-se? wide-draws
               (evidence-by-quadrature (planted by-hand :draw draw-wide)))]
;=> [true false]

(within-5-se? (evidence-draws noise-1 2000 (rng/key 10))
              (evidence-by-quadrature noise-1))
;=> true
