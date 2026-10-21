(ns gen.genjl.trace-translators-test
  "Gen.jl 0.4.8's test/inference/trace_translators.jl, ported: the
   extending, symmetric and general trace translators, with their
   transforms checked (gen.inference.transform), and the deterministic
   one. Steps are 0-based (D2)."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest is]]
            [gen.array :as array :refer [+ - * /]]
            [gen.choicemap :as cm]
            [gen.diff :as diff]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.transform :as tr]
            [gen.random :as rng]
            [gen.test-support :as ts]))

;; ---------------------------------------------------------------------------
;; trace_translators.jl:3, DeterministicTraceTranslator
;; ---------------------------------------------------------------------------

(def ^:private polar
  (gen [] (trace :r (d/inv-gamma 1 1)) (trace :theta (d/uniform (- (/ Math/PI 2)) (/ Math/PI 2)))))

(def ^:private cartesian
  (gen [] (trace :x (d/normal 0 1)) (trace :y (d/normal 0 1))))

(tr/deftransform ^:private to-cartesian [p1 _q1 p2 _q2]
  (let [r (tr/read p1 :r :continuous)
        theta (tr/read p1 :theta :continuous)]
    (tr/write p2 :x (* r (array/cos theta)) :continuous)
    (tr/write p2 :y (* r (array/sin theta)) :continuous)))

(tr/deftransform ^:private to-polar [p2 _q2 p1 _q1]
  (let [x (tr/read p2 :x :continuous)
        y (tr/read p2 :y :continuous)]
    (tr/write p1 :r (array/sqrt (+ (* x x) (* y y))) :continuous)
    (tr/write p1 :theta (array/atan2 y x) :continuous)))

(deftest deterministic-trace-translator
  ;; trace_translators.jl:3: polar to cartesian and back, each checked (the
  ;; Jacobian's gradient runs through atan2's)
  (let [[f finv] (tr/pair to-cartesian to-polar)
        to-xy (tr/deterministic-translator {:p-new cartesian :p-new-args [] :new-observations cm/EMPTY
                                            :transform f :check? true})
        to-r-theta (tr/deterministic-translator {:p-new polar :p-new-args [] :new-observations cm/EMPTY
                                                 :transform finv :check? true})
        t1 (:trace (gfi/generate polar [] (cm/choicemap :theta 0.0) (rng/key 1)))
        t2 (:trace (gfi/generate cartesian [] (cm/choicemap :y 0.0 :x 1.0) (rng/key 2)))
        {xy :trace xy-weight :weight} (to-xy t1 (rng/key 3))
        {r-theta :trace r-theta-weight :weight} (to-r-theta t2 (rng/key 4))]
    (is (== 0 (array/item (xy :y))))
    (is (== 0 (array/item (r-theta :theta))))
    ;; beyond Gen.jl's assertions, the weights: the scores' difference and
    ;; log |det J|, log r one way and log 1/r the other (r = 1 here)
    (is (ts/close? (+ (- (:score xy) (:score t1)) (Math/log (ts/number (t1 :r)))) xy-weight))
    (is (ts/close? (- (:score r-theta) (:score t2)) r-theta-weight))))

;; ---------------------------------------------------------------------------
;; trace_translators.jl:44, SimpleExtendingTraceTranslator
;; ---------------------------------------------------------------------------

(def ^:private steps
  (gen [n] (doseq [t (range n)]
             (let [z (trace [:z t] (d/normal 0 1))]
               (trace [:x t] (d/normal z 1))))))

(def ^:private next-step
  "z of the step after the trace's last, near x."
  (gen [tr x] (trace [:z (first (:args tr))] (d/normal x 1))))

(deftest simple-extending-trace-translator
  ;; trace_translators.jl:44
  (let [observation (cm/choicemap [:x 1] 5.0)
        translator (tr/extending-translator {:p-new-args [2] :p-argdiffs [diff/unknown-change]
                                             :new-observations observation
                                             :q-forward next-step :q-forward-args [5.0]})
        t1 (gfi/simulate steps [1] (rng/key 1))
        {t2 :trace log-weight :weight} (translator t1 (rng/key 2))
        proposed (cm/choicemap [:z 1] (t2 [:z 1]))
        prop-weight (:weight (gfi/assess next-step [t1 5.0] proposed))
        up-weight (:weight (gfi/update t1 [2] [diff/unknown-change] (cm/merge proposed observation) (rng/key 3)))]
    (is (ts/close? log-weight (- up-weight prop-weight)))))

;; ---------------------------------------------------------------------------
;; trace_translators.jl:73, SymmetricTraceTranslator
;; ---------------------------------------------------------------------------

(def ^:private binned
  (gen [] (if (trace :z (d/bernoulli 0.5))
            (trace :i (d/uniform-discrete 1 10))
            (trace :x (d/uniform 0 1)))))

(def ^:private offset
  (gen [tr] (when (tr :z) (trace :dx (d/uniform 0.0 0.1)))))

(tr/deftransform ^:private bin-or-unbin [p1 q1 p2 q2]
  (if (tr/read p1 :z :discrete)
    (let [i (tr/read p1 :i :discrete)
          dx (tr/read q1 :dx :continuous)]
      (tr/write p2 :z false :discrete)
      (tr/write p2 :x (+ (/ (- i 1) 10.0) dx) :continuous))
    (let [x (tr/read p1 :x :continuous)
          i (array/ceil (* x 10))]
      (tr/write p2 :z true :discrete)
      (tr/write p2 :i i :discrete)
      (tr/write q2 :dx (- x (/ (- i 1) 10.0)) :continuous))))

(deftest symmetric-trace-translator
  ;; trace_translators.jl:73: SymmetricTraceTranslator(q, (), involution),
  ;; the involution checked, is q's choices and the involution applied
  (let [involution (tr/involution (tr/involutive bin-or-unbin) {:check? true})
        t1 (:trace (gfi/generate binned [] (cm/choicemap :z false :x 0.95) (rng/key 1)))
        {u :choices} (gfi/propose offset [t1] (rng/key 2))
        {t2 :trace} (involution t1 u nil [] (rng/key 3))]
    (is (and (true? (array/item (t2 :z))) (== 10 (array/item (t2 :i)))))))

;; ---------------------------------------------------------------------------
;; trace_translators.jl:116, GeneralTraceTranslator
;; ---------------------------------------------------------------------------

(def ^:private unit-square
  (gen [] (trace :x (d/uniform 0 1)) (trace :y (d/uniform 0 1))))

(def ^:private cells
  "i and j: the intervals [(i-1)/10, i/10] and [(j-1)/10, j/10]."
  (gen [] (trace :i (d/uniform-discrete 1 10)) (trace :j (d/uniform-discrete 1 10))))

(def ^:private no-offsets (gen [_p1-trace]))

(def ^:private offsets
  (gen [_p2-trace] (trace :dx (d/uniform 0.0 0.1)) (trace :dy (d/uniform 0.0 0.1))))

(tr/deftransform ^:private to-cells [p1 _q1 p2 q2]
  (let [x (tr/read p1 :x :continuous)
        y (tr/read p1 :y :continuous)
        i (array/ceil (* x 10))
        j (array/ceil (* y 10))]
    (tr/write p2 :i i :discrete)
    (tr/write p2 :j j :discrete)
    (tr/write q2 :dx (- x (/ (- i 1) 10.0)) :continuous)
    (tr/write q2 :dy (- y (/ (- j 1) 10.0)) :continuous)))

(tr/deftransform ^:private from-cells [p2 q2 p1 _q1]
  (let [i (tr/read p2 :i :discrete)
        j (tr/read p2 :j :discrete)
        dx (tr/read q2 :dx :continuous)
        dy (tr/read q2 :dy :continuous)]
    (tr/write p1 :x (+ (/ (- i 1) 10.0) dx) :continuous)
    (tr/write p1 :y (+ (/ (- j 1) 10.0) dy) :continuous)))

(deftest general-trace-translator
  ;; trace_translators.jl:116
  (let [[f _] (tr/pair to-cells from-cells)
        translator (tr/translator {:p-new cells :p-new-args [] :new-observations cm/EMPTY
                                   :q-forward no-offsets :q-forward-args []
                                   :q-backward offsets :q-backward-args []
                                   :transform f :check? true})
        t1 (:trace (gfi/generate unit-square [] (cm/choicemap :x 0.05 :y 0.95) (rng/key 1)))
        {t2 :trace} (translator t1 (rng/key 2))]
    (is (and (== 1 (array/item (t2 :i))) (== 10 (array/item (t2 :j)))))))

