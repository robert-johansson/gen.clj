;; mcmc's compiled-step cache: bounded at 16 when a step compiles, unbounded
;; for the ::plain marks of steps that cannot compile. On MLX a model that
;; reads a value on the host fails its first compiled call with {:op :compile};
;; this stands in for that by making array/compile return such a function.
(require '[gen.array :as array] '[gen.dist :as d] '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi]
         '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.random :as rng] '[gen.inference.mcmc :as mcmc])
(def model (gen [n] (let [mu (trace :mu (d/normal 0.0 1.0))]
                      (mapv (fn [i] (trace [:y i] (d/normal mu 1.0))) (range n)))))
(def cache @#'gen.inference.mcmc/compiled-steps)
(reset! cache {})
(with-redefs [array/compile (fn [_f] (fn [& _] (throw (ex-info "host read while compiling" {:op :compile}))))]
  (doseq [n (range 1 101)]
    (let [tr (:trace (gfi/generate model [n] (cm/from-map {:y {0 0.1}}) (rng/key n)))]
      (mcmc/hmc tr (sel/select :mu) {:L 2 :eps 0.1 :compile? true} (rng/key 0)))))
(println "entries after 100 HMC steps on 100 argument values, none compilable:" (count @cache)
         "(all ::plain:" (every? #{:gen.inference.mcmc/plain} (vals @cache)) ")")
(reset! cache {})
(doseq [n (range 1 101)]
  (let [tr (:trace (gfi/generate model [n] (cm/from-map {:y {0 0.1}}) (rng/key n)))]
    (mcmc/hmc tr (sel/select :mu) {:L 2 :eps 0.1 :compile? true} (rng/key 0))))
(println "entries after 100 HMC steps on 100 argument values, compilable:" (count @cache))
