;; I2-09: with-dispatch takes any generative function; on a non-dynamic one the
;; dispatcher is never called, silently (with-params and with-transitions are loud).
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.choicemap :as cm] '[gen.combinators :as comb])
(defmacro probe [label & body]
  `(println ~label "=>" (try (pr-str (do ~@body)) (catch Throwable e# (str "THROWS " (.getName (class e#)) ": " (ex-message e#))))))
(def k (gen [m] (trace :x (d/normal m 1.0))))
(def called (atom 0))
(def disp (fn [& _] (swap! called inc) nil))
(probe "with-dispatch on a Map: accepted?" (some? (dyn/with-dispatch (comb/map k) disp)))
(probe "dispatcher ever called on the Map?" (do (gfi/simulate (dyn/with-dispatch (comb/map k) disp) [[0.0]] (rng/key 1)) @called))
(probe "with-params on a Map" (dyn/with-params (comb/map k) {}))
(probe "dispatcher called on a dynamic gf" (do (gfi/simulate (dyn/with-dispatch k disp) [0.0] (rng/key 1)) @called))
