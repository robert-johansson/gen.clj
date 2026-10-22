;; V-P1-03 (verifier): an infinite return value whose elements are unchanged,
;; by other routes: (repeat x), which = walks without allocating (a spin, not
;; an error), regenerate, and the control of a changed value.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d]
         '[gen.random :as rng] '[gen.choicemap :as cm] '[gen.selection :as sel])

(defn timed [label f secs]
  (let [t0 (System/nanoTime)
        fut (future (try (f) (catch Throwable e (str "threw " (.getName (class e))))))
        r (deref fut (* 1000 secs) ::timeout)]
    (println label "->" (if (= r ::timeout) (str "still running after " secs " s") r)
             (format "(%.1f s)" (/ (- (System/nanoTime) t0) 1e9)))))

(def rep (gen [] (let [x0 (trace :x0 (d/normal 0.0 1.0))] (trace :y (d/normal x0 1.0)) (repeat x0))))
(def it (gen [] (let [x0 (trace :x0 (d/normal 0.0 1.0))] (trace :y (d/normal x0 1.0)) (iterate #(* 0.9 %) x0))))

(let [tr (gfi/simulate rep [] (rng/key 1))
      rt (Runtime/getRuntime)
      used #(quot (- (.totalMemory rt) (.freeMemory rt)) 1048576)]
  (println "simulate (repeat x0) ok:" (take 2 (:retval tr)))
  ;; control: constraining x0 changes the first element, so = stops there
  (timed "1. control, update of :x0" #(select-keys (gfi/update tr (cm/from-map {:x0 0.25}) (rng/key 2)) [:weight :retdiff]) 10)
  (println "   heap before (MB)" (used))
  (timed "2. (repeat x0), update of :y only" #(select-keys (gfi/update tr (cm/from-map {:y 0.5}) (rng/key 2)) [:weight :retdiff]) 10)
  (println "   heap after (MB)" (used)))

(let [tr (gfi/simulate it [] (rng/key 1))]
  (timed "3. (iterate f x0), regenerate of :y" #(select-keys (gfi/regenerate tr (sel/select :y) (rng/key 3)) [:weight :retdiff]) 30))

(shutdown-agents)
(System/exit 0)
