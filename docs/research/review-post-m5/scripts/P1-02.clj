;; P1-02: a splice's argdiffs ignore a change in the number of arguments, so a
;; static callee's rest parameter is "no-change" when it gained or lost elements.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm]
         '[gen.diff :as diff] '[gen.trace :as trace])

(defn lp [mu y] (- (* -0.5 (Math/log (* 2 Math/PI))) (* 0.5 (Math/pow (- y mu) 2))))

(def a 1.0)
(def b 2.0)

(def callee-static
  (gen ^:static [x & more]
    (let [n (count more)
          y (trace :y (d/normal n 1.0))]
      y)))
(def callee-dynamic
  (gen [x & more]
    (let [n (count more)
          y (trace :y (d/normal n 1.0))]
      y)))

(defn outer [callee]
  (gen [extra?]
    (if extra?
      (splice :c callee a b)
      (splice :c callee a))))

(doseq [[label callee] [["static callee " callee-static] ["dynamic callee" callee-dynamic]]]
  (let [m (outer callee)
        tr (:trace (gfi/generate m [true] (cm/from-map {:c {:y 0.0}}) (rng/key 1)))
        {t :trace w :weight} (gfi/update tr [false] [diff/unknown-change] cm/EMPTY (rng/key 2))]
    (println label "| update weight" w "| new score" (:score t)
             "| assess of its choices" (:weight (gfi/assess m [false] (:choices t) (rng/key 3))))))
(println "expected weight lp(0; 0) - lp(0; 1) =" (- (lp 0.0 0.0) (lp 1.0 0.0)))

;; the other direction: a nil argument added where none was is identical to the
;; previous call's (absent) argument
(def c nil)
(def callee2 (gen ^:static [& xs] (let [n (count xs) y (trace :y (d/normal n 1.0))] y)))
(def outer2 (gen [extra?] (if extra? (splice :c callee2 c) (splice :c callee2))))
(let [tr (:trace (gfi/generate outer2 [false] (cm/from-map {:c {:y 0.0}}) (rng/key 1)))
      {t :trace w :weight} (gfi/update tr [true] [diff/unknown-change] cm/EMPTY (rng/key 2))]
  (println "nil argument added | update weight" w "| new score" (:score t)
           "| assess" (:weight (gfi/assess outer2 [true] (:choices t) (rng/key 3)))
           "| expected" (- (lp 1.0 0.0) (lp 0.0 0.0))))

;; at the top level, Gen.jl's default arguments as DESIGN's G3 writes them, [x & [y]]:
;; update from [0.0 3.0] to [0.0] with the one argdiff the new args take
(def g3 (gen ^:static [x & [y]] (let [mu (if (nil? y) x y) z (trace :z (d/normal mu 1.0))] z)))
(let [tr (:trace (gfi/generate g3 [0.0 3.0] (cm/from-map {:z 0.0}) (rng/key 1)))
      {t :trace w :weight} (gfi/update tr [0.0] [diff/no-change] cm/EMPTY (rng/key 2))]
  (println "top level, G3's [x & [y]] | update weight" w "| new score" (:score t)
           "| assess" (:weight (gfi/assess g3 [0.0] (:choices t) (rng/key 3)))
           "| expected" (- (lp 0.0 0.0) (lp 3.0 0.0))))

;; an estimated density (gen.custom/estimated) keeps its trace, old arguments
;; and all, when the argdiffs given for the new arguments are all no-change
(require '[gen.custom :as custom])
(def est (custom/estimated {:addresses [:y]
                            :estimate (fn [args choices _key] (lp (count args) (cm/get-choice choices :y)))
                            :simulate (fn [args _key] {:choices (cm/from-map {:y 0.0}) :log-estimate (lp (count args) 0.0)})}))
(let [tr (:trace (gfi/generate est [1.0 2.0] (cm/from-map {:y 0.0}) (rng/key 1)))
      {t :trace w :weight} (gfi/update tr [1.0] [diff/no-change] cm/EMPTY (rng/key 2))]
  (println "estimated, [1.0 2.0] -> [1.0] | update weight" w "| new trace's args" (:args t) "| score" (:score t)
           "| assess at [1.0]" (:weight (gfi/assess est [1.0] (:choices t) (rng/key 3)))))
