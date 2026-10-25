;; The code of book/015-choice-maps-traces.md. Each region, from a `;; book:`
;; line to the next, is included there by name (gen.book: `clojure -M:test
;; -m gen.book sync` writes the Markdown's copies; gen.book-test checks them).

;; book: ns
(ns book.choice-maps-traces
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]
            [gen.serialize :as serialize]
            [gen.trace :as trace]))

;; book: two-types
(def leaf (cm/->Value 3.0))

(def node (cm/->Node {:x (cm/->Value 1.0) :y (cm/->Value 2.0)}))

[(cm/has-value? leaf) (cm/has-value? node)
 (cm/has-value? (cm/get-submap node :x))]
;=> [true false true]

(= cm/EMPTY (cm/get-submap node :z))
;=> true

(pr-str node)
;=> "#gen/choicemap {:x 1.0, :y 2.0}"

;; book: build
(def c
  (cm/choicemap :params (cm/choicemap :slope 2.0 :intercept 1.0)
                :y0 5.0))

[(= c (cm/from-map {:params {:slope 2.0 :intercept 1.0} :y0 5.0}))
 (= c (cm/choicemap [:params :slope] 2.0 [:params :intercept] 1.0
                    :y0 5.0))]
;=> [true true]

;; book: read
[(cm/get-choice c [:params :slope]) (c [:params :slope])
 (cm/get-value (cm/get-submap c :y0))]
;=> [2.0 2.0 5.0]

(cm/to-map (cm/get-submap c :params))
;=> {:slope 2.0 :intercept 1.0}

[(cm/addresses c) (cm/values c)]
;=> [[[:params :intercept] [:params :slope] [:y0]] [1.0 2.0 5.0]]

;; book: lookup-error
;; error: get-choice: no value at [:z]: nothing is there
#_(cm/get-choice c :z)

;; book: lookups
(defn reason
  "What the lookup (f) finds: :found, or its error's :reason."
  [f]
  (try (f) :found
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(mapv #(reason (fn [] (cm/get-choice c %)))
      [[:params :slope] :z :params [:y0 :a]])
;=> [:found :absent :node :through-value]

[(c :z nil) (cm/has-value? c :z) (cm/has-submap? c :params)]
;=> [nil false true]

;; book: call-node
;; error: choice map: no value at [:params]: nested choices are there
#_(c :params)

;; book: immutable
(def c2 (cm/set-choice c [:params :slope] 3.0))

[(c2 [:params :slope]) (c [:params :slope])]
;=> [3.0 2.0]

(def a (cm/choicemap :x 1.0 :y 2.0))

(cm/to-map (cm/merge a (cm/choicemap :z 3.0)))
;=> {:x 1.0 :y 2.0 :z 3.0}

;; book: merge-error
;; error: merge: the choice maps overlap
#_(cm/merge a (cm/choicemap :y 99.0 :z 3.0))

;; book: override
[(cm/to-map (cm/override a (cm/choicemap :y 99.0 :z 3.0))) (cm/to-map a)]
;=> [{:x 1.0 :y 99.0 :z 3.0} {:x 1.0 :y 2.0}]

;; book: sharing
(def big
  (apply cm/choicemap (mapcat (fn [i] [[:y i] (double i)]) (range 1000))))

(def revised (cm/set-choice big [:y 500] -1.0))

(count (filter #(identical? (cm/get-submap big %) (cm/get-submap revised %))
               (cm/addresses big)))
;=> 999

;; book: model
(def line-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 2))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      [slope intercept])))

(def tr (gfi/simulate line-model [[1.0 2.0 3.0]] (rng/key 1)))

;; book: fields
(take 5 (keys tr))
;=> [:gen-fn :args :choices :retval :score]

[(identical? line-model (:gen-fn tr)) (:args tr)]
;=> [true [[1.0 2.0 3.0]]]

[(:retval tr) (:score tr)]
;=> [[1.432 -5.475] -9.497] ; float64

(every? qualified-keyword? (drop 5 (keys tr)))
;=> true

;; book: call
[(tr :slope) (tr [:y 0])]
;=> [1.432 -5.395] ; float64

[(= (tr []) (:retval tr)) (tr :noise nil)]
;=> [true nil]

[(:slope tr) (get tr :slope)]
;=> [nil nil]

;; book: call-error
;; error: trace: no choice or call at [:noise]: nothing is there
#_(tr :noise)

;; book: self-contained
(def same (gfi/update tr (:choices tr) (rng/key 2)))

[(array/allclose? (:weight same) 0.0)
 (= (:choices (:trace same)) (:choices tr))]
;=> [true true]

(def moved (:trace (gfi/update tr (cm/choicemap :slope 0.5) (rng/key 3))))

[(moved :slope) (tr :slope)]
;=> [0.5 1.432] ; float64

;; book: sensors
(def sensor
  (gen [mu]
    (trace :value (d/normal mu 0.5))))

(def fusion
  (gen [position]
    (let [a (splice :sensor-a sensor position)
          b (splice :sensor-b sensor position)]
      {:a a :b b})))

(def ft (gfi/simulate fusion [5.0] (rng/key 4)))

(cm/addresses (:choices ft))
;=> [[:sensor-a :value] [:sensor-b :value]]

;; book: subtrace
(def a-trace (trace/subtrace ft :sensor-a))

[(identical? sensor (:gen-fn a-trace)) (:args a-trace)
 (= (:choices a-trace) (cm/get-submap (:choices ft) :sensor-a))
 (= (ft :sensor-a) (:retval a-trace) (a-trace :value))]
;=> [true [5.0] true true]

(array/allclose? (:score ft)
                 (+ (:score a-trace) (:score (trace/subtrace ft :sensor-b))))
;=> true

;; book: write
(serialize/write c)
;=> "#gen/choicemap {:params {:intercept 1.0, :slope 2.0}, :y0 5.0}"

(= c (serialize/read (serialize/write c)))
;=> true

(def tr-again (serialize/read-trace line-model (serialize/write-trace tr)))

(let [parts (juxt :args :choices :retval :score)]
  (= (parts tr) (parts tr-again)))
;=> true

;; book: read-trace-error
;; error: read-trace: the rebuilt score
#_(def wide-sensor
  (gen [mu]
    (trace :value (d/normal mu 1.0))))

#_(serialize/read-trace wide-sensor (serialize/write-trace a-trace))

;; book: select
(def s (sel/select :x :y))

[(sel/selected? s :x) (sel/selected? s :z)
 (sel/selected? sel/all :anything) (sel/selected? sel/none :x)]
;=> [true false true false]

;; book: hierarchical
(def slope-only (sel/hierarchical :params (sel/select :slope)))

[(sel/selected? slope-only [:params :slope])
 (sel/selected? slope-only [:params :intercept])
 (sel/selected? (sel/get-subselection slope-only :params) :slope)]
;=> [true false true]

(= slope-only (sel/select [:params :slope]))
;=> true

[(sel/selected? (sel/select :params) [:params :slope])
 (sel/selected? slope-only :params)]
;=> [true false]

;; book: algebra
[(= sel/none (sel/complement sel/all))
 (= sel/all (sel/complement sel/none))
 (identical? s (sel/complement (sel/complement s)))]
;=> [true true true]

(def some-paths
  [[:x] [:y] [:z] [:y 0] [:params :slope] [:params :intercept]])

(defn negated?
  "Does the complement of selection t select exactly the paths of
   some-paths that t does not?"
  [t]
  (every? #(not= (sel/selected? t %) (sel/selected? (sel/complement t) %))
          some-paths))

(every? negated? [sel/all sel/none s slope-only (sel/select :params)])
;=> true

;; book: project
[(array/allclose? (gfi/project tr sel/none) 0.0)
 (array/allclose? (gfi/project tr sel/all) (:score tr))
 (array/allclose? (gfi/project tr (sel/select :slope))
                  (dc/log-prob (d/normal 0 2) (tr :slope)))]
;=> [true true true]

;; book: partition
(defn splits-score?
  "Do trace t's projections on selection s and on its complement add up to
   its score?"
  [t s]
  (array/allclose? (+ (gfi/project t s) (gfi/project t (sel/complement s)))
                   (:score t)))

(every? #(splits-score? tr %)
        [sel/none (sel/select :slope) (sel/select :y) (sel/select [:y 1])])
;=> true

(every? #(splits-score? ft %)
        [(sel/select :sensor-a) (sel/select [:sensor-b :value])])
;=> true

;; book: latents
(def observations (cm/choicemap [:y 0] -3.9 [:y 1] -2.6 [:y 2] -1.1))

(def latents (sel/complement (sel/from-choicemap observations)))

(cm/addresses (sel/get-selected (:choices tr) latents))
;=> [[:intercept] [:slope]]

;; book: solution-1
(mapv #(identical? (cm/get-submap c %) (cm/get-submap c2 %))
      [[] [:params] [:params :slope] [:params :intercept] [:y0]])
;=> [false false false true true]

;; book: solution-2
(def by-name (sel/select :slope :intercept))

(def but-ys (sel/complement (sel/select :y)))

(array/allclose? (gfi/project tr by-name) (gfi/project tr but-ys))
;=> true

[(sel/selected? by-name :noise) (sel/selected? but-ys :noise)]
;=> [false true]

;; book: solution-3
(defrecord Union [p q]
  sel/ISelection
  (-selected? [_ addr] (or (sel/selected? p addr) (sel/selected? q addr)))
  (-subselection [_ addr]
    (->Union (sel/get-subselection p addr) (sel/get-subselection q addr))))

(defn intersection
  "What both selections p and q select, by De Morgan's law."
  [p q]
  (sel/complement (->Union (sel/complement p) (sel/complement q))))

(let [yz (sel/select :y :z)]
  [(mapv #(sel/selected? (->Union s yz) %) [:w :x :y :z])
   (mapv #(sel/selected? (intersection s yz) %) [:w :x :y :z])])
;=> [[false true true true] [false false true false]]

(array/allclose? (gfi/project tr (->Union (sel/select :slope) (sel/select :y)))
                 (+ (gfi/project tr (sel/select :slope))
                    (gfi/project tr (sel/select :y))))
;=> true
