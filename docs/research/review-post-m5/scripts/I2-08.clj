;; I2-08: gfi/project reads a trace by its :gen-fn, every other reader by its
;; maker; a trace given another kind of gen-fn projects to 0, silently.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.selection :as sel] '[gen.combinators :as comb] '[gen.trace :as trace])
(def m (gen [] (trace :x (d/normal 0.0 1.0))))
(def other (comb/map (gen [a] (trace :x (d/normal a 1.0)))))
(def t (gfi/simulate m [] (rng/key 3)))
(def t' (assoc t :gen-fn other))
(println "score:" (:score t))
(println "project t all:" (gfi/project t sel/all))
(println "project (assoc t :gen-fn map) all:" (gfi/project t' sel/all) "(maker is m:" (identical? m (trace/maker t')) ")")
(println "project-by-maker of it:" (gfi/project-by-maker t' sel/all))
(println "subtrace/lookup go by maker: (t' :x) =" (t' :x))
