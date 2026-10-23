(ns gen.batched-gradients-bench
  "Gradients of n particles at once against one particle at a time
   (genclj-7dti, docs/array-api.md §8b): one batched pass of the tape over
   the batch tracer against n scalar passes. Three levels, n = 10, 100 and
   1000:
   - an array function's gradient (a regression's log density in two
     parameters, 50 observations);
   - choice-gradients of a particle collection against each trace's
     (linear regression, K = 10 observations, one trace site each);
   - an hmc-particles step (L = 10) against n hmc steps.
   Each line checks that the two agree (the gradients; the HMC steps' shapes),
   then times them after warm-up. Reported, not asserted.
   Run: clojure [-J-Dgen.backend=mlx] -M:bench -m gen.batched-gradients-bench"
  (:refer-clojure :exclude [+ *])
  (:require [gen.array :as array :refer [+ *]]
            [gen.array.batch :as batch]
            [gen.array.protocols :as p]
            [gen.batched :as b]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dist.core :as dc]
            [gen.dynamic :refer [gen]]
            [gen.gradients :as grad]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def linreg
  (gen [xs]
    (let [slope (trace :slope (d/normal 0 10))
          intercept (trace :intercept (d/normal 0 10))]
      (doseq [[i x] (map-indexed vector xs)]
        (trace [:y i] (d/normal (+ (* slope x) intercept) 1)))
      slope)))

(defn- ms-per-call
  "Mean ms per call of f over reps calls, after 5 warm-ups (each result
   evaluated)."
  [f reps]
  (dotimes [_ 5] (array/eval! (f)))
  (let [t0 (System/nanoTime)]
    (dotimes [_ reps] (array/eval! (f)))
    (/ (- (System/nanoTime) t0) 1e6 reps)))

(defn- report [label n same? batched scalar]
  (printf "%-28s n=%-5d agree: %-5s | batched %9.3f ms  one at a time %9.3f ms  (%.1f×)%n"
          label n same? batched scalar (/ scalar batched))
  (flush))

(defn- array-gradients [n]
  (let [xs (rng/normal (rng/key 1) [50])
        ys (+ (* 2.0 xs) (rng/normal (rng/key 2) [50]))
        lp (fn [theta] (let [slope (array/take theta 0) icpt (array/take theta 1)]
                         (+ (dc/log-prob (d/normal 0 10) slope)
                            (array/sum (dc/log-prob (d/normal (+ (* slope xs) icpt) 1.0) ys)))))
        thetas (rng/normal (rng/key 3) [n 2])
        batched #(batch/unwrap (batch/batched [n] ((array/grad lp) (batch/wrap thetas [n]))) [n])
        scalar #(array/stack (mapv (fn [i] ((array/grad lp) (array/take thetas i 0))) (range n)))]
    (report "an array function" n (array/allclose? (batched) (scalar)) (ms-per-call batched 10) (ms-per-call scalar 10))))

(defn- particle-gradients [n]
  (let [xs (vec (range 10.0))
        obs (apply cm/choicemap (mapcat (fn [i] [[:y i] (* 2.0 i)]) (range 10)))
        ps (:particles (b/generate linreg [xs] obs n (rng/key 4)))
        s (sel/select :slope :intercept)
        batched #(cm/to-map (:gradients (grad/choice-gradients ps s)))
        scalar #(mapv (fn [t] (cm/to-map (:gradients (grad/choice-gradients t s)))) (b/traces ps))
        same? (array/allclose? (:slope (batched)) (array/stack (mapv :slope (scalar))))]
    (report "choice-gradients (K = 10)" n same? (ms-per-call batched 10) (ms-per-call scalar 3))))

(defn- hmc-steps [n]
  (let [xs (vec (range 10.0))
        obs (apply cm/choicemap (mapcat (fn [i] [[:y i] (* 2.0 i)]) (range 10)))
        ps (:particles (b/generate linreg [xs] obs n (rng/key 5)))
        s (sel/select :slope :intercept)
        opts {:L 10 :eps 0.01}
        batched #(b/scores (:particles (mcmc/hmc-particles ps s opts (rng/key 6))))
        trs (b/traces ps)
        scalar #(array/stack (mapv (fn [t k] (:score (:trace (mcmc/hmc t s opts k)))) trs (rng/split (rng/key 6) n)))]
    (report "HMC step (L = 10, K = 10)" n (= (array/shape (batched)) (array/shape (scalar))) (ms-per-call batched 5) (ms-per-call scalar 2))))

(defn -main [& _]
  (println "backend" (p/-name array/*backend*))
  (doseq [n [10 100 1000]] (array-gradients n))
  (doseq [n [10 100 1000]] (particle-gradients n))
  (doseq [n [10 100 1000]] (hmc-steps n))
  (shutdown-agents))
