;; A (kern ^:trace …) let that binds a kernel: the check reads the let's
;; value again after its body and compares with =, and a kernel built again
;; is a new closure, so a valid kernel is rejected.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.inference.kernel :as k])

(def model (gen [] (trace :scale (d/gamma 2.0 1.0)) (trace :x (d/normal 0 1))))
(def tr (:trace (gfi/generate model [] (cm/choicemap :scale 1.0) (rng/key 1))))

(def bound-selection ((k/kern ^:trace [tr] (let [s (sel/select :x)] (k/mh s) (k/mh s)))))
(def bound-kernel ((k/kern ^:trace [tr] (let [step (k/mh (sel/select :x))] step step))))

(println "a selection in the let:" (try (some? (:trace (bound-selection tr (rng/key 2)))) (catch Exception e (str "loud: " (ex-message e)))))
(println "a kernel in the let (it does not read the trace at all):"
         (try (some? (:trace (bound-kernel tr (rng/key 2)))) (catch Exception e (str "loud: " (ex-message e)))))
(def looped ((k/kern ^:trace [tr] (doseq [step (mapv k/mh [(sel/select :x) (sel/select :scale)])] step))))
(println "a loop over kernels built in its collection:"
         (try (some? (:trace (looped tr (rng/key 3)))) (catch Exception e (str "loud: " (subs (ex-message e) 0 60) "…"))))
