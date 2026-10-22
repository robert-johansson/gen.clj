;; I2-06: calling a generative function (Gen.jl's (gf args…)) is written four
;; ways; past six arguments a defgf or deterministic record throws
;; AbstractMethodError, and an estimated one is no function at all.
(require '[gen.dynamic :as dyn :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.choicemap :as cm]
         '[gen.combinators :as comb] '[gen.custom :as custom])
(defmacro probe [label & body]
  `(println ~label "=>" (try (pr-str (do ~@body)) (catch Throwable e# (str "THROWS " (.getName (class e#)) ": " (subs (str (ex-message e#)) 0 (min 90 (count (str (ex-message e#))))))))))

(def k (gen [t s a b c d e] (trace :x (d/normal s 1.0))))
(def u (comb/unfold k))
(probe "dynamic gf, 7 args          " (k 0 0.0 1 2 3 4 5))
(probe "unfold (defgf), 7 args      " (u 1 0.0 1 2 3 4 5))
(probe "unfold (defgf), apply 7 args" (apply u [1 0.0 1 2 3 4 5]))
(def det (custom/deterministic {:apply (fn [args] (count args))}))
(probe "deterministic, 6 args       " (det 1 2 3 4 5 6))
(probe "deterministic, 7 args       " (det 1 2 3 4 5 6 7))
(def est (custom/estimated {:addresses [:y] :estimate (fn [_ _ _] 0.0)
                            :simulate (fn [_ _] {:choices (cm/choicemap :y 1.0) :log-estimate 0.0})}))
(probe "estimated, 0 args           " (est))
(probe "estimated, simulate's retval" (:retval (gfi/simulate est [])))
