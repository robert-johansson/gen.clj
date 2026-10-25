;; The code of book/061-from-genjl.md. Each region, from a `;; book:` line
;; to the next, is included there by name (gen.book).

;; book: ns
(ns book.from-genjl
  (:refer-clojure :exclude [+ - * /])
  (:require [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.combinators :as comb]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.random :as rng]
            [gen.selection :as sel]))

;; book: model
(def sensor
  (gen [mu]
    (trace :reading (d/normal mu 0.5))))

(def model
  (gen [n]
    (let [mu (trace :mu (d/normal 0 10))]
      (doseq [i (range n)]
        (splice [:sensor i] sensor mu))
      mu)))

(def tr (gfi/simulate model [3] (rng/key 1)))

(cm/addresses (:choices tr))
;=> [[:mu] [:sensor 0 :reading] [:sensor 1 :reading] [:sensor 2 :reading]]

;; book: splice-namespace
(def shared
  (gen []
    (let [mu (trace :mu (d/normal 0 10))]
      (splice sensor mu))))

(cm/addresses (:choices (gfi/simulate shared [] (rng/key 2))))
;=> [[:mu] [:reading]]

;; book: zero-based
(def flips (comb/map (gen [p] (trace :b (d/bernoulli p)))))

(cm/addresses (:choices (gfi/simulate flips [[0.1 0.9]] (rng/key 3))))
;=> [[0 :b] [1 :b]]

(def die (gen [] (trace :face (d/categorical [0.25 0.25 0.25 0.25]))))

(set (for [k (rng/split (rng/key 4) 200)]
       (array/item ((gfi/simulate die [] k) :face))))
;=> #{0 1 2 3}

;; book: keys
(= (array/->clj (:score (gfi/simulate model [3] (rng/key 1))))
   (array/->clj (:score tr)))
;=> true

;; book: trace-fields
[(= model (:gen-fn tr)) (:args tr) (= (:retval tr) (tr []) (tr :mu))]
;=> [true [3] true]

;; book: choicemap-call
(def readings (:choices tr))

[(= (readings :mu) (tr :mu)) (some? (readings [:sensor 0 :reading]))
 (readings [:sensor 7 :reading] nil)]
;=> [true true nil]

;; book: choicemap-call-node
;; error: choice map: no value at [:sensor]: nested choices are there
#_(readings :sensor)

;; book: record-field-trap
[(:mu tr) (some? (tr :mu))]
;=> [nil true]

;; book: lookup-error
;; error: trace: no choice or call at [:nu]
#_(tr :nu)

;; book: update
(def observed
  (:trace (gfi/generate model [3] (cm/choicemap [:sensor 0 :reading] 1.2)
                        (rng/key 5))))

(let [{t2 :trace w :weight discard :discard}
      (gfi/update observed [3] [diff/no-change] (cm/choicemap :mu 1.0)
                  (rng/key 6))]
  [(t2 :mu)
   (cm/addresses discard)
   (array/allclose? w (- (:score t2) (:score observed)))])
;=> [1.0 [[:mu]] true]

;; book: merge-error
;; error: merge: the choice maps overlap
#_(cm/merge (cm/choicemap :a 1.0) (cm/choicemap :a 2.0))

;; book: booleans
(def coin (gen [] (trace :heads (d/bernoulli 0.0))))

(def ctr (gfi/simulate coin [] (rng/key 7)))

[(ctr :heads)
 (if (ctr :heads) :yes :no)
 (if ((:choices ctr) :heads) :yes :no)
 (if (array/truthy? (cm/get-choice (:choices ctr) :heads)) :yes :no)]
;=> [false :no :no :no]

;; book: plain-call
(def noise (gen [] (trace :z (d/normal 0 1))))

(def with-plain-call (gen [] (+ (noise) 1.0)))

(let [run #(gfi/simulate with-plain-call [] (rng/key 8))]
  [(= (array/->clj (:retval (run))) (array/->clj (:retval (run))))
   (cm/empty? (:choices (run)))])
;=> [true true]
