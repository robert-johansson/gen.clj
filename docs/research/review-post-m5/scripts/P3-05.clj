;; A diverging leapfrog trajectory: HMC on x ~ N(0, 1) with eps = 3 (the
;; leapfrog is unstable past eps = 2: the state grows by about 6.85 a step)
;; and L = 400 steps, which overflows to ±Inf and then NaN. A diverging
;; proposal is an ordinary rejection (its acceptance probability is 0, or
;; undefined, and Stan, PyMC and Gen.jl's NaN comparison reject it).
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.inference.mcmc :as mcmc])

(def model (gen [] (trace :x (d/normal 0 1))))
(def tr (:trace (gfi/generate model [] (cm/choicemap :x 0.3) (rng/key 1))))

(doseq [L [40 120 400]]
  (println "L" L ":"
           (try (select-keys (mcmc/hmc tr (sel/select :x) {:L L :eps 3.0} (rng/key 2)) [:accepted? :log-alpha])
                (catch Exception e (str "threw: " (ex-message e) " " (ex-data e))))))
(println "MALA, tau 1e300:"
         (try (select-keys (mcmc/mala tr (sel/select :x) 1e300 (rng/key 3)) [:accepted? :log-alpha])
              (catch Exception e (str "threw: " (ex-message e)))))

;; Leaving the support: s ~ gamma(2, 1) from s = 0.1, a step to s < 0 has
;; density 0, so the proposal should be rejected.
(def positive (gen [] (trace :s (d/gamma 2.0 1.0))))
(def ptr (:trace (gfi/generate positive [] (cm/choicemap :s 0.1) (rng/key 1))))
(doseq [[L eps] [[1 0.5] [3 0.5] [10 0.5]]]
  (println "gamma, L" L "eps" eps ":"
           (frequencies
            (for [key (rng/split (rng/key 5) 200)]
              (try (let [r (mcmc/hmc ptr (sel/select :s) {:L L :eps eps} key)] (if (:accepted? r) :accepted :rejected))
                   (catch Exception e (str "threw: " (ex-message e))))))))

;; The same at a common setting: a posterior with sd 0.003 (here a prior
;; N(0, 0.003)), eps 0.1 and L 100 (the step is 33 posterior sds).
(def tight (gen [] (trace :x (d/normal 0 0.003))))
(def ttr (:trace (gfi/generate tight [] (cm/choicemap :x 0.001) (rng/key 1))))
(doseq [L [50 100 150 200 300]]
  (println "sd 0.003, eps 0.1, L" L ":"
           (try (select-keys (mcmc/hmc ttr (sel/select :x) {:L L :eps 0.1} (rng/key 2)) [:accepted? :log-alpha])
                (catch Exception e (str "threw: " (ex-message e))))))
