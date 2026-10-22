(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc :refer [defdist]]
         '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.random :as rng] '[gen.choicemap :as cm]
         '[gen.combinators :as comb] '[gen.diff :as diff]
         '[gen.inference.particle-filter :as pf] '[gen.inference.vi :as vi] '[gen.learning :as learn])

(defmacro show [label & body]
  `(println ~label "=>" (try (pr-str (do ~@body)) (catch Throwable e# (str (.getSimpleName (class e#)) ": " (ex-message e#))))))

;; 1. defdist accepts a misspelled clause silently
(defdist my-exp "x" [rate] (sample [key shape] (array/exp 1.0)) (logprob [v] (array/- v)))
(show "defdist with (logprob ...) clause: construct" (my-exp 1.0))
(show "  ... its log-prob" (dc/log-prob (my-exp 1.0) 0.5))

;; 2. nil parameters pass construction
(show "normal 0 nil" (d/normal 0.0 nil))
(show "  ... sampling it" (dc/sample (d/normal 0.0 nil) (rng/key 0)))

;; 3. unknown option values: case without default
(show "learn/init of an unknown optimizer kind" (learn/init {:kind :rmsprop} {:w 1.0}))

;; 4. pf/step on the scalar path: how many array/eval! calls per step?
(let [kernel (gen [t s] (let [x (trace :x (d/normal s 1.0))] x))
      model (gen [n] (splice :steps (comb/unfold kernel) n 0.0))
      obs (fn [t] (cm/from-map {:steps {t {:x 0.5}}}))
      st (pf/initialize model [1] (obs 0) 50 {:key (rng/key 1)})
      n (atom 0)
      orig array/eval!]
  (with-redefs [array/eval! (fn [x] (swap! n inc) (orig x))]
    (pf/step st [2] [diff/unknown-change] (obs 1) {:key (rng/key 2)}))
  (println "array/eval! calls in one scalar pf/step of 50 particles:" @n))
