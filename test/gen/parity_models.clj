(ns gen.parity-models
  "The GenJAX parity specs' model and data (GenMLX's ~/mlx/genjax-parity:
   linreg_is, linreg_mala, linreg_hmc, linreg_mala_manychain), which the
   parity bench (bench/gen/parity_bench.clj) times and ext/device's tests
   run: Bayesian linear regression on the specs' ten frozen observations,
   slope and intercept N(0, 10), each y N(slope·x + intercept, 1), in three
   forms: :sites, one scalar site per observation (the specs' form, a
   dynamic body with a loop); :static, the same sites in the static
   language; :vector, the ten as one site, gen.clj's idiom. Here, in test/,
   so that the tests need no bench/ (genclj-ri2d)."
  (:refer-clojure :exclude [+ *])
  (:require [gen.array :as array :refer [+ *]]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.selection :as sel]))

(def xs
  "The specs' inputs (frozen 2026-07-28)."
  [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0])

(def ys
  "The specs' observations: 2x - 1 + N(0, 1), Python's random.seed(42),
   frozen 2026-07-28; both sides read these literals."
  [-1.14409 0.827096 2.888684 5.701984 6.872412 7.502647 11.332318 12.732663 14.783041 17.115885])

(def posterior-mean-slope
  "The exact conjugate posterior's mean of the slope (the specs' sanity
   reference)."
  1.991719)

(def sites
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1.0)))
      slope)))

(defmacro ^:private static-sites
  "The sites model in the static language, its n sites spelled out (the
   static language has no loops)."
  [n]
  `(gen ~(with-meta '[xs] {:static true})
     (let [~'slope (~'trace :slope (d/normal 0 10))
           ~'intercept (~'trace :intercept (d/normal 0 10))
           ~@(mapcat (fn [i] [(symbol (str "y" i))
                              `(~'trace [:y ~i] (d/normal (+ (* ~'slope (nth ~'xs ~i)) ~'intercept) 1.0))])
                     (range n))]
       ~'slope)))

(def static (static-sites 10))

(def vector-model
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (trace :ys (d/normal (+ (* slope xs) intercept) 1.0))
      slope)))

(def models
  "Each model with its arguments and its observations."
  {:sites [sites [xs] (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys))]
   :static [static [xs] (apply cm/choicemap (mapcat (fn [i y] [[:y i] y]) (range) ys))]
   :vector [vector-model [(array/array xs)] (cm/choicemap :ys (array/array ys))]})

(def selection (sel/select :slope :intercept))

(def eps
  "The specs' step size: HMC's leapfrog step, and MALA's proposal
   scale (GenJAX's: mean x + eps²/2·∇, standard deviation eps)."
  0.08)

(def tau
  "gen.clj's MALA step for the specs' eps: v' ~ N(v + tau·∇, √(2 tau)),
   so tau = eps²/2."
  (/ (clojure.core/* eps eps) 2.0))
