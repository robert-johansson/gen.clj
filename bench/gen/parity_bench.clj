(ns gen.parity-bench
  "GenMLX's GenJAX parity workloads (~/mlx/genjax-parity on fenrir, its
   specs linreg_is, linreg_mala, linreg_hmc and linreg_mala_manychain) on
   gen.clj, to set gen.clj's costs beside GenJAX's on the same card (bean
   genclj-a1hu). The model is Bayesian linear regression on the specs' ten
   frozen observations, slope and intercept N(0, 10), each y N(slope·x +
   intercept, 1), in three forms: :sites, one scalar site per observation
   (the specs' form, a dynamic body with a loop); :static, the same sites
   in the static language; :vector, the ten as one site, gen.clj's idiom.

   Each cell times the specs' unit: for IS, a batched importance-sampling
   run of N particles with its weights read; for MCMC, a constrained
   generate from the prior and an S-step chain (kernel/run, the slope read
   each step), or N chains of mala-particles. The first of three warm-up
   runs is reported as :warmup-ms; then up to 20 timed runs (fewer once a
   cell has spent gen.parity.budget-s seconds, at least 3), each on its own
   key, give :median-ms, :p10-ms and :p90-ms. On MLX a further run counts
   the arrays made and the host reads per step (or per call). Prints one
   EDN map per cell.

   Run: clojure [-J-Dgen.backend=mlx] -M:bench -m gen.parity-bench [cells…]
   cells: is mala hmc manychain (all by default); models with
   -Dgen.parity.models=sites,static,vector (all by default)."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.string :as str]
            [gen.array :as array :refer [+ - * /]]
            [gen.array.protocols :as p]
            [gen.batched :as batched]
            [gen.choicemap :as cm]
            [gen.dist :as d]
            [gen.dynamic :refer [gen]]
            [gen.gfi :as gfi]
            [gen.inference.importance :as is]
            [gen.inference.kernel :as kernel]
            [gen.inference.mcmc :as mcmc]
            [gen.random :as rng]
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
  (/ (* eps eps) 2.0))

;; ---------------------------------------------------------------------------
;; The specs' units: each a function of a key, returning what the cell reports

(defn- chain
  "A constrained generate from the prior, then S steps of kernel k, the
   slope read each step: {:acceptance :slope-tail (the mean slope over the
   chain's second half)}."
  [[model args observations] k steps key]
  (let [[k0 k1] (rng/split key 2)
        tr (:trace (gfi/generate model args observations k0))
        {:keys [draws acceptance-rate]} (kernel/run k tr {:samples steps :key k1 :keep #(% :slope)})
        tail (subvec draws (quot steps 2))]
    {:acceptance acceptance-rate
     :slope-tail (/ (reduce clojure.core/+ tail) (count tail))}))

(defn- manychain
  "N chains: a batched constrained generate from the prior, then S steps of
   mala-particles: {:acceptance (pooled) :slope-tail (pooled over every
   chain's second half)}."
  [[model args observations] n steps key]
  (let [[k0 k1] (rng/split key 2)
        ps (:particles (batched/generate model args observations n k0))
        step-keys (rng/split k1 steps)]
    (loop [i 0 ps ps accepted 0 tail 0.0]
      (if (< i steps)
        (let [{ps :particles acc :accepted} (mcmc/mala-particles ps selection tau (step-keys i))]
          (recur (inc i) ps (clojure.core/+ accepted (count (filter true? acc)))
                 (if (>= i (quot steps 2)) (+ tail (array/sum (batched/choice ps :slope))) tail)))
        {:acceptance (/ (double accepted) (* n steps))
         :slope-tail (/ (double (array/item tail)) (* n (- steps (quot steps 2))))}))))

(defn- importance
  "N batched importance samples from the prior: {:log-ml-estimate}."
  [[model args observations] n key]
  (let [{:keys [log-ml-estimate fallback]} (is/importance-sampling model args observations n {:key key :batched? true})]
    (cond-> {:log-ml-estimate log-ml-estimate} fallback (assoc :fallback fallback))))

;; ---------------------------------------------------------------------------
;; Timing and counting

(defn- ms [^long t0] (/ (- (System/nanoTime) t0) 1e6))

(defn- quantile [sorted q]
  (nth sorted (min (dec (count sorted)) (long (Math/floor (* q (count sorted)))))))

(defn- timed
  "Times (unit key) as the specs do: three warm-ups, the first reported
   (the second and third skipped when it took over 5 s, for the slow cells
   on MLX), then up to 20 timed runs within the budget (at least 3), each
   on its own key. The last timed run's result is merged in."
  [unit]
  (let [budget-ms (* 1000.0 (Double/parseDouble (System/getProperty "gen.parity.budget-s" "30")))
        t0 (System/nanoTime)
        _ (unit (rng/key 1000))
        warmup (ms t0)]
    (when (< warmup 5000.0)
      (unit (rng/key 1001))
      (unit (rng/key 1002)))
    (loop [i 0 times [] spent 0.0 result nil]
      (if (or (< i 3) (and (< i 20) (< spent budget-ms)))
        (let [t (System/nanoTime)
              r (unit (rng/key i))
              dt (ms t)]
          (recur (inc i) (conj times dt) (clojure.core/+ spent dt) r))
        (let [s (vec (sort times))]
          (merge result
                 {:warmup-ms warmup :median-ms (quantile s 0.5) :p10-ms (quantile s 0.1)
                  :p90-ms (quantile s 0.9) :runs (count s)}))))))

(defn- backend-name [] (name (p/-name array/*backend*)))

(defn- mlx? [] (= "mlx" (backend-name)))

(defn- adopted
  "MLX arrays made so far (gen.mlx.MlxRuntime's count), or nil off MLX."
  []
  (when (mlx?)
    (let [f (doto (.getDeclaredField (Class/forName "gen.mlx.MlxRuntime") "ADOPTED") (.setAccessible true))]
      (.get ^java.util.concurrent.atomic.AtomicLong (.get f nil)))))

(def ^:private host-reads (java.util.concurrent.atomic.AtomicLong.))

(defn- counted
  "On MLX, the arrays made and host reads in one run of (unit key), divided
   by per (steps, or 1)."
  [unit per]
  (when (mlx?)
    (let [to-host (requiring-resolve 'gen.backend.mlx/to-host)
          f @to-host]
      (alter-var-root to-host (constantly (fn [x] (.incrementAndGet ^java.util.concurrent.atomic.AtomicLong host-reads) (f x))))
      (try
        (let [a0 (adopted) r0 (.get ^java.util.concurrent.atomic.AtomicLong host-reads)]
          (unit (rng/key 2000))
          {:arrays-per-unit (/ (double (- (long (adopted)) (long a0))) per)
           :host-reads-per-unit (/ (double (- (.get ^java.util.concurrent.atomic.AtomicLong host-reads) r0)) per)})
        (finally (alter-var-root to-host (constantly f)))))))

(defn- now [] (str (.withNano (java.time.LocalTime/now) 0)))

(defn- cell [spec model-name params unit per]
  (let [started (now)
        r (merge {:spec spec :model model-name :backend (backend-name)
                  :device (System/getProperty "gen.mlx.device" "default")}
                 params (timed unit) (counted unit per) {:started started :ended (now)})]
    (prn r)
    (flush)
    r))

(defn- model-names []
  (map keyword (str/split (System/getProperty "gen.parity.models" "sites,static,vector") #",")))

(defn -main [& cells]
  (let [cells (set (or (seq cells) ["is" "mala" "hmc" "manychain"]))]
    (doseq [m (model-names) :let [model (models m)]]
      (when (cells "is")
        (doseq [n [1 100 3000 30000]]
          (cell :linreg_is m {:N n} #(importance model n %) 1)))
      (when (cells "mala")
        (doseq [compile? [false true] s [10 100 1000]]
          (cell :linreg_mala m {:S s :compile? compile?}
                #(chain model (kernel/mala selection tau {:compile? compile?}) s %) s)))
      (when (cells "hmc")
        (doseq [compile? [false true] l [8 32] s [10 100 1000]]
          (cell :linreg_hmc m {:S s :L l :compile? compile?}
                #(chain model (kernel/hmc selection {:L l :eps eps :compile? compile?}) s %) s)))
      (when (cells "manychain")
        (doseq [s [100 1000] n [8 64 512 4096]]
          (cell :linreg_mala_manychain m {:N n :S s} #(manychain model n s %) s)))))
  (shutdown-agents))
