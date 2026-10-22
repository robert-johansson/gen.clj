;; gen.gradients' objective pairs retgrad's leaves with the return value's
;; by depth-first position (array/tree-leaves), not by key.
;; Model: x ~ N(0, 1), returning {:b 2x, :a x} (an array map, in that
;; order). retgrad {:a 1.0 :b 0.0} is "the same tree shape": the objective
;; is log N(x; 0, 1) + 1·a + 0·b = log N(x) + x, so ∂/∂x = 1 - x.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.gradients :as g])

(def model (gen [] (let [x (trace :x (d/normal 0 1))] {:b (array/* 2.0 x) :a x})))
(def tr (:trace (gfi/generate model [] (cm/choicemap :x 0.5) (rng/key 1))))

(defn dx [retgrad] (cm/get-choice (:gradients (g/choice-gradients tr (sel/select :x) retgrad)) :x))

(println "retval" (:retval tr))
(println "retgrad {:a 1.0 :b 0.0}: expected 1 - x = 0.5, got" (dx {:a 1.0 :b 0.0}))
(println "retgrad {:b 0.0 :a 1.0} (the retval's order): got" (dx {:b 0.0 :a 1.0}))

;; fewer leaves than the return value: silently truncated
(def model2 (gen [] (let [x (trace :x (d/normal 0 1))] [x (array/* 3.0 x)])))
(def tr2 (:trace (gfi/generate model2 [] (cm/choicemap :x 0.5) (rng/key 1))))
(println "retval [x 3x], retgrad [1.0] (a cotangent missing one leaf): got"
         (cm/get-choice (:gradients (g/choice-gradients tr2 (sel/select :x) [1.0])) :x)
         "(no error; [1.0 0.0] would give 0.5, [1.0 1.0] 3.5)")
;; a scalar retgrad for a vector of two leaves: only the first leaf is paired
(println "retval [x 3x], retgrad 1.0: got"
         (cm/get-choice (:gradients (g/choice-gradients tr2 (sel/select :x) 1.0)) :x))
