(ns gen.device.cuda-bench
  "gen.parity-bench's MCMC cells as CUDA C kernels, one thread per chain
   (gen.device.cuda, bean genclj-a1hu; parked since 1.2.1, genclj-z1gj): single chains (N = 1) and many
   chains (N up to 65536), MALA and HMC, on the specs' model and data. One
   kernel per model, kernel and S, launched for every N. A timed run is a
   launch on a root key, the wait, and every output copied to pinned host
   memory; :kernel-ms times the launch and wait alone. :compile-ms is
   NVRTC's compile and the module's load; :stage-ms staging and the C
   text. Prints one EDN map per cell.

   Run: clojure -J-Dgen.cuda.nvrtc=<libnvrtc.so> -M:device:bench -m gen.device.cuda-bench [mala hmc]
   models with -Dgen.parity.models=vector,static (both by default)."
  (:require [clojure.string :as str]
            [gen.choicemap :as cm]
            [gen.device.chain :as ch]
            [gen.device.cuda :as cuda]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.parity-models :as pm]
            [gen.random :as rng]))

(defn- ms [^long t0] (/ (- (System/nanoTime) t0) 1e6))

(defn- quantile [sorted q]
  (nth sorted (min (dec (count sorted)) (long (Math/floor (* q (count sorted)))))))

(defn- key-words [i] (rng/key-data (rng/key i)))

(defn- summary
  "Pooled acceptance and the slope's mean over the second half of every
   chain, from the host copies [values accepted start]."
  [[values accepted] n steps d si]
  (let [^floats vs (cuda/read-floats values)
        ^bytes acc (cuda/read-bytes accepted)
        half (quot steps 2)]
    {:acceptance (/ (double (areduce acc i s 0 (+ s (aget acc i)))) (* n steps))
     :slope-tail (/ (loop [c 0 sum 0.0]
                      (if (< c n)
                        (recur (inc c) (loop [t half sum sum]
                                         (if (< t steps) (recur (inc t) (+ sum (aget vs (int (+ (* (+ (* c steps) t) d) si))))) sum)))
                        sum))
                    (* n (- steps half)))}))

(defn- time-cell [k cell n steps d si]
  (let [budget-ms (* 1000.0 (Double/parseDouble (System/getProperty "gen.parity.budget-s" "15")))
        bufs (cuda/buffers k n)]
    (try
      (let [t0 (System/nanoTime)
            _ (cuda/launch! k bufs n (key-words 1000))
            first-ms (ms t0)]
        (dotimes [i 2] (cuda/launch! k bufs n (key-words (+ 1001 i))))
        (let [times (loop [i 0 times [] spent 0.0]
                      (if (or (< i 3) (and (< i 20) (< spent budget-ms)))
                        (let [t (System/nanoTime) _ (cuda/launch! k bufs n (key-words i)) dt (ms t)]
                          (recur (inc i) (conj times dt) (+ spent dt)))
                        (vec (sort times))))
              stats (summary bufs n steps d si)
              kernel (vec (sort (for [i (range (min 10 (count times)))]
                                  (let [t (System/nanoTime)] (cuda/launch! k bufs n (key-words i) false) (ms t)))))
              r (merge cell {:backend "cuda" :first-ms first-ms
                             :median-ms (quantile times 0.5) :p10-ms (quantile times 0.1) :p90-ms (quantile times 0.9)
                             :kernel-ms (quantile kernel 0.5) :runs (count times)}
                       stats)]
          (prn r)
          (flush)
          r))
      (finally (cuda/free-buffers bufs)))))

(defn- model-names []
  (map keyword (str/split (System/getProperty "gen.parity.models" "vector,static") #",")))

(defn -main [& kinds]
  (let [kinds (set (or (seq kinds) ["mala" "hmc"]))
        chains [1 8 64 512 4096 65536]]
    (doseq [m (model-names)
            :let [[model args obs] (pm/models m)
                  proto (grad/continuous-selection (:trace (gfi/generate model args obs (rng/key 0))) pm/selection)
                  d (count (cm/leaves proto))
                  si (first (keep-indexed (fn [i [p _]] (when (= p [:slope]) i)) (cm/leaves proto)))]
            [kind build steps] (concat (when (kinds "mala")
                                         (for [s [10 100 1000]]
                                           [{:spec :linreg_mala} #(ch/mala-chain model args obs pm/selection pm/tau s) s]))
                                       (when (kinds "hmc")
                                         (for [l [8 32] s [10 100 1000]]
                                           [{:spec :linreg_hmc :L l} #(ch/hmc-chain model args obs pm/selection {:L l :eps pm/eps} s) s])))
            :let [t0 (System/nanoTime)
                  src (cuda/source (build))
                  stage-ms (ms t0)
                  k (cuda/compile src)
                  info {:model m :S steps :stage-ms stage-ms :c-chars (count (:source src)) :compile-ms (:compile-ms k)
                        :registers (:registers k) :local-bytes (:local-bytes k)}]]
      (binding [*out* *err*] (println (str/join "\n" (filter #(re-find #"ptxas info|spill" %) (str/split-lines (:log k))))))
      (doseq [n (if (or (= 10 steps) (and (= :linreg_hmc (:spec kind)) (= 32 (:L kind)))) [1] chains)]
        (time-cell k (merge kind info {:N n}) n steps d si))
      (cuda/unload! k)))
  (shutdown-agents))
