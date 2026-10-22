;; Verifier's independent check of P3-03, through param-gradients (the other
;; caller of the objective). Model: x ~ N(0, 1), a parameter w that the
;; density does not read, returning {:y w·x, :z 3w}. The objective is
;; log N(x) + ⟨retgrad, retval⟩, so with retgrad {:z 1, :y 0} its ∂/∂w is 3
;; (whatever x is), and with {:y 1, :z 0} it is x = 0.5.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.gradients :as g])

(def model (dyn/with-params
             (gen [] (let [x (trace :x (d/normal 0 1)) w (param :w)]
                       {:y (array/* w x) :z (array/* 3.0 w)}))
             {:w 1.0}))
(def tr (:trace (gfi/generate model [] (cm/choicemap :x 0.5) (rng/key 1))))
(println "retval" (:retval tr))
(doseq [[retgrad expected] [[{:z 1.0 :y 0.0} 3.0] [{:y 0.0 :z 1.0} 3.0] [{:y 1.0 :z 0.0} 0.5] [{:z 0.0 :y 1.0} 0.5]]]
  (println "retgrad" retgrad "∂/∂w expected" expected "got" (:w (g/param-gradients tr retgrad))))
;; a retgrad with a key the return value lacks, and one missing a key
(println "retgrad {:q 1.0 :y 1.0} (no :q in the retval):" (try (:w (g/param-gradients tr {:q 1.0 :y 1.0}))
                                                              (catch Exception e (str "loud: " (ex-message e)))))
(println "retgrad {:z 1.0} (no :y):" (try (:w (g/param-gradients tr {:z 1.0}))
                                          (catch Exception e (str "loud: " (ex-message e)))))
