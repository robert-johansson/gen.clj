;; V-P1-01 (verifier): the retdiff's clojure.core/= hides a change, by two other
;; routes than the reviewer's: an Unfold over a dynamic kernel (no static
;; language involved), and a sorted map against an array map.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.choicemap :as cm] '[gen.diff :as diff]
         '[gen.combinators :as comb])

(defn lp [mu y] (- (* -0.5 (Math/log (* 2 Math/PI))) (* 0.5 (Math/pow (- y mu) 2))))

;; 1. Unfold: step t's mean is atan2(state, -1), its new state x * 0.0
(def kernel (gen [t s] (let [x (trace :x (d/normal (Math/atan2 s -1.0) 1.0))] (* x 0.0))))
(def chain (comb/unfold kernel))
(let [tr (:trace (gfi/generate chain [2 0.0] (cm/from-map {0 {:x 1.0} 1 {:x 1.0}}) (rng/key 1)))
      {t :trace w :weight rd :retdiff} (gfi/update tr (cm/from-map {0 {:x -1.0}}) (rng/key 2))
      assessed (:weight (gfi/assess chain [2 0.0] (:choices t) (rng/key 3)))
      ;; step 0: mean pi, x 1 -> -1; step 1: state +0.0 -> -0.0, mean pi -> -pi, x = 1
      expected (+ (- (lp Math/PI -1.0) (lp Math/PI 1.0)) (- (lp (- Math/PI) 1.0) (lp Math/PI 1.0)))]
  (println "1. Unfold | states before" (:retval tr) "after" (:retval t) "| retdiff" rd)
  (println "   update weight" w "| expected" expected
           "| new score" (:score t) "| assess of its choices" assessed
           "| score - old score" (- assessed (:score tr))))

;; 2. a callee returning a sorted map or an array map with the same entries:
;;    (first m) tells them apart, = does not
(def inner (gen [] (if (trace :b (d/bernoulli 0.5)) (sorted-map :b 1.0 :a 2.0) (array-map :b 1.0 :a 2.0))))
(def outer (gen ^:static []
             (let [m (splice :in inner)
                   y (trace :y (d/normal (val (first m)) 1.0))]
               y)))
(let [tr (:trace (gfi/generate outer [] (cm/from-map {:in {:b true} :y 0.0}) (rng/key 1)))
      {t :trace w :weight} (gfi/update tr (cm/from-map {:in {:b false}}) (rng/key 2))
      assessed (:weight (gfi/assess outer [] (:choices t) (rng/key 3)))]
  (println "2. sorted map -> array map | update weight" w "| expected (b's terms cancel)" (- (lp 1.0 0.0) (lp 2.0 0.0))
           "| new score" (:score t) "| assess" assessed)
  (println "   = :" (= (sorted-map :b 1.0 :a 2.0) (array-map :b 1.0 :a 2.0))
           "| consistent? no-change:" (diff/consistent? (sorted-map :b 1.0 :a 2.0) (array-map :b 1.0 :a 2.0) diff/no-change)
           "| same-value?:" (gfi/same-value? (sorted-map :b 1.0 :a 2.0) (array-map :b 1.0 :a 2.0))))

;; 3. control: a callee whose value really changes is rerun downstream
(def inner3 (gen [] (if (trace :b (d/bernoulli 0.5)) 2.0 1.0)))
(def outer3 (gen ^:static [] (let [v (splice :in inner3) y (trace :y (d/normal v 1.0))] y)))
(let [tr (:trace (gfi/generate outer3 [] (cm/from-map {:in {:b true} :y 0.0}) (rng/key 1)))
      {w :weight} (gfi/update tr (cm/from-map {:in {:b false}}) (rng/key 2))]
  (println "3. control, 2.0 -> 1.0 | update weight" w "| expected" (- (lp 1.0 0.0) (lp 2.0 0.0))))
