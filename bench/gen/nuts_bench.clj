(ns gen.nuts-bench
  "NUTS against HMC with a tuned L (genclj-tvc0), both with Stan's warmup
   (gen.inference.nuts: the step size and a diagonal metric adapted), on a
   10-d Gaussian of scales 0.1 to 10 (log-spaced) and correlations
   0.5^|i - j|. Each kernel draws 2000 times after 1000 warmup iterations,
   and is charged a gradient per leapfrog step and one at each draw's
   start. The effective sample size is the smallest over the coordinates
   of their bulk and tail ESSs (gen.inference.diagnostics); HMC's L is the
   best of 1, 2, 4, …, 64 by gradients per effective draw.
   Run: clojure [-J-Dgen.backend=mlx] -M:bench -m gen.nuts-bench"
  (:require [gen.array :as array]
            [gen.array.protocols :as p]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.diagnostics :as dg]
            [gen.inference.nuts :as nuts]
            [gen.random :as rng]
            [gen.selection :as sel]))

(def dim 10)

(def scales (mapv #(Math/pow 10.0 (- (/ (* 2.0 %) (dec dim)) 1.0)) (range dim)))

(def cov
  (array/array (vec (for [i (range dim)]
                      (vec (for [j (range dim)]
                             (* (scales i) (scales j) (Math/pow 0.5 (Math/abs (long (- i j)))))))))))

(def target (gen [] (trace :z (d/mvnormal (array/zeros [dim]) cov))))

(defn- measure [opts]
  (let [start (:trace (gfi/generate target [] (cm/choicemap :z (array/zeros [dim])) (rng/key 0)))
        t0 (System/nanoTime)
        {:keys [traces stats warmup eps divergences]} (nuts/sample start (sel/select :z) (merge {:samples 2000} opts)
                                                                   (rng/key 1))
        secs (/ (- (System/nanoTime) t0) 1e9)
        draws (mapv #(array/->clj (% :z)) traces)
        min-ess (apply min (for [i (range dim) ess [dg/ess-bulk dg/ess-tail]] (ess (mapv #(nth % i) draws))))
        grads (fn [stats] (reduce + (map #(inc (:n-leapfrog %)) stats)))]
    {:per-draw (/ (grads stats) (double (count stats)))
     :per-effective (/ (grads stats) min-ess)
     :warmup (grads (:stats warmup))
     :ess min-ess :eps eps :divergences divergences :secs secs}))

(defn- row [label {:keys [per-draw per-effective warmup ess eps divergences secs]}]
  (printf "%-14s %8.1f %10.1f %8.0f %10d %8.3f %5d %7.1f s%n" label per-draw per-effective ess warmup eps divergences secs)
  (flush))

(defn -main [& _]
  (println "backend" (p/-name array/*backend*) "| a 10-d Gaussian, scales 0.1-10, correlations 0.5^|i-j|")
  (println "kernel         grad/draw  grad/ESS    min ESS  warmup grads  eps   divs   time")
  (row "NUTS" (measure {}))
  (let [hmcs (for [L [1 2 4 8 16 32 64]] [L (measure {:kernel :hmc :L L})])]
    (doseq [[L r] hmcs] (row (str "HMC, L = " L) r))
    (let [[L r] (apply min-key (comp :per-effective second) hmcs)]
      (println (format "tuned HMC: L = %d, %.1f gradients per effective draw" L (:per-effective r)))))
  (shutdown-agents))
