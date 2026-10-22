;; P1-01: a retdiff of no-change from clojure.core/= (0.0 = -0.0) lets the static
;; walk keep a node whose input changed: a stale score and a wrong update weight.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm]
         '[gen.diff :as diff])

;; the callee returns x * 0.0: +0.0 or -0.0 by x's sign
(def inner (gen [] (* (trace :x (d/normal 0.0 1.0)) 0.0)))

;; atan2(+0.0, -1) = pi, atan2(-0.0, -1) = -pi
(def body-static
  (gen ^:static []
    (let [r (splice :in inner)
          y (trace :y (d/normal (Math/atan2 r -1.0) 1.0))]
      y)))
(def body-dynamic
  (gen []
    (let [r (splice :in inner)
          y (trace :y (d/normal (Math/atan2 r -1.0) 1.0))]
      y)))

(defn lp [mu y] (- (* -0.5 (Math/log (* 2 Math/PI))) (* 0.5 (Math/pow (- y mu) 2))))

(doseq [[label gf] [["static " body-static] ["dynamic" body-dynamic]]]
  (let [tr (:trace (gfi/generate gf [] (cm/from-map {:in {:x 1.0} :y 1.0}) (rng/key 1)))
        {t :trace w :weight} (gfi/update tr (cm/from-map {:in {:x -1.0}}) (rng/key 2))
        assessed (:weight (gfi/assess gf [] (:choices t) (rng/key 3)))]
    (println label "| retval of :in before" (:retval (gen.trace/subtrace tr :in))
             "after" (:retval (gen.trace/subtrace t :in))
             "| inner's retdiff" (:retdiff (gfi/update (gen.trace/subtrace tr :in) (cm/from-map {:x -1.0}) (rng/key 2))))
    (println label "| update weight" w "| new score" (:score t) "| assess of its choices" assessed)))

(println "expected weight: lp(x=-1) - lp(x=1) + lp(y=1; mu=-pi) - lp(y=1; mu=pi) ="
         (+ 0.0 (- (lp (- Math/PI) 1.0) (lp Math/PI 1.0))) "(the x terms cancel: N(0,1) is symmetric)")
(println "(= 0.0 -0.0):" (= 0.0 -0.0) "| diff/compute:" (diff/compute 0.0 -0.0)
         "| consistent? no-change:" (diff/consistent? 0.0 -0.0 diff/no-change)
         "| gfi/same-value?:" (gfi/same-value? 0.0 -0.0))

;; the same through a vector and a list, which = also equates
(def inner2 (gen [] (if (trace :b (d/bernoulli 0.5)) [1.0 2.0] (list 1.0 2.0))))
(def outer2 (gen ^:static []
              (let [v (splice :in inner2)
                    y (trace :y (d/normal (first (conj v 5.0)) 1.0))]
                y)))
(let [tr (:trace (gfi/generate outer2 [] (cm/from-map {:in {:b true} :y 0.0}) (rng/key 1)))
      {t :trace w :weight} (gfi/update tr (cm/from-map {:in {:b false}}) (rng/key 2))]
  (println "vector->list | update weight" w "| score" (:score t)
           "| assess" (:weight (gfi/assess outer2 [] (:choices t) (rng/key 3)))
           "| expected weight (b's terms cancel)" (- (lp 5.0 0.0) (lp 1.0 0.0))))
