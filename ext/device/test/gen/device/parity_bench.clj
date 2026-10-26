(ns gen.device.parity-bench
  "gen.parity-bench's cells (GenMLX's GenJAX parity specs) as device
   programs: each cell's whole unit, a constrained generate and an S-step
   chain, or N importance samples, or N chains, staged from gen.clj's own
   code (gen.device.chain), lowered to StableHLO and run by XLA through
   PJRT (bean genclj-a1hu). A timed run is the spec's unit from a key to
   the samples on the host: the key copied to the device, the program
   executed, every output copied back. :warmup-ms is the first run's whole
   cost, staging, StableHLO, XLA's compile and the first execution
   (reported apart as :stage-ms :emit-ms :compile-ms :first-ms); then up to
   20 timed runs (at least 3, within gen.parity.budget-s), each on its own
   key. Prints one EDN map per cell.

   Run: clojure -J-Dgen.xla.plugin=<xla_cuda_plugin.so> -M:device:bench -m gen.device.parity-bench [cells…]
   cells: is mala hmc manychain (all by default); models with
   -Dgen.parity.models=vector,static (both by default)."
  (:require [clojure.string :as str]
            [gen.array :as array]
            [gen.choicemap :as cm]
            [gen.device.chain :as ch]
            [gen.device.pjrt :as pjrt]
            [gen.device.stablehlo :as hlo]
            [gen.gfi :as gfi]
            [gen.gradients :as grad]
            [gen.parity-bench :as pb]
            [gen.random :as rng]))

(defn- ms [^long t0] (/ (- (System/nanoTime) t0) 1e6))

(defn- quantile [sorted q]
  (nth sorted (min (dec (count sorted)) (long (Math/floor (* q (count sorted)))))))

(defn- key-buffer [client key]
  (let [[k0 k1] (rng/key-data key)]
    (pjrt/->device client (int-array [(unchecked-int (long k0)) (unchecked-int (long k1))]) [2] :u32)))

(defn- run-once
  "One run of the executable on key: every output on the host."
  [client exe key]
  (with-open [k (key-buffer client key)]
    (let [outs (pjrt/execute exe [k])]
      (try (mapv pjrt/->host outs)
           (finally (run! #(.close ^java.io.Closeable %) outs))))))

(defn- floats [{:keys [data]}] (vec (if (instance? (Class/forName "[F") data) (seq ^floats data) (seq data))))

(defn- program-outputs
  "The host outputs as the program's result map (its leaves in order)."
  [prog outs]
  (gen.array.tree/refill (:out-tree prog) outs))

(defn- timed-cell
  "Stages, emits, compiles and times (build), a thunk of the staged
   program; summary reads a result map of host outputs."
  [client cell build summary]
  (let [budget-ms (* 1000.0 (Double/parseDouble (System/getProperty "gen.parity.budget-s" "30")))
        t0 (System/nanoTime)
        prog (build)
        stage-ms (ms t0)
        t1 (System/nanoTime)
        text (hlo/module prog)
        emit-ms (ms t1)
        t2 (System/nanoTime)]
    (with-open [exe (pjrt/compile client text)]
      (let [compile-ms (ms t2)
            t3 (System/nanoTime)
            _ (run-once client exe (rng/key 1000))
            first-ms (ms t3)]
        (dotimes [i 2] (run-once client exe (rng/key (+ 1001 i))))
        (loop [i 0 times [] spent 0.0 last-out nil]
          (if (or (< i 3) (and (< i 20) (< spent budget-ms)))
            (let [t (System/nanoTime)
                  outs (run-once client exe (rng/key i))
                  dt (ms t)]
              (recur (inc i) (conj times dt) (+ spent dt) outs))
            (let [s (vec (sort times))
                  r (merge cell
                           {:backend "xla" :nodes (count (:graph prog)) :mlir-lines (count (str/split-lines text))
                            :stage-ms stage-ms :emit-ms emit-ms :compile-ms compile-ms :first-ms first-ms
                            :warmup-ms (+ stage-ms emit-ms compile-ms first-ms)
                            :median-ms (quantile s 0.5) :p10-ms (quantile s 0.1) :p90-ms (quantile s 0.9) :runs (count s)}
                           (summary (program-outputs prog last-out)))]
              (prn r)
              (flush)
              r)))))))

(defn- slope-index
  "The slope's position among the flat selected values (mcmc's order)."
  [[model args obs]]
  (let [proto (grad/continuous-selection (:trace (gfi/generate model args obs (rng/key 0))) pb/selection)]
    (first (keep-indexed (fn [i [p _]] (when (= p [:slope]) i)) (cm/leaves proto)))))

(defn- chain-summary
  "Acceptance and the slope's mean over the chain's second half."
  [si steps]
  (fn [{:keys [values accepted]}]
    (let [vs (floats values) acc (:data accepted)
          d (quot (count vs) steps)
          tail (map #(nth vs (+ (* % d) si)) (range (quot steps 2) steps))]
      {:acceptance (/ (double (count (filter true? (seq ^booleans acc)))) steps)
       :slope-tail (/ (reduce + tail) (count tail))})))

(defn- chains-summary
  "Pooled acceptance and slope tail of n chains."
  [si n steps]
  (fn [{:keys [values accepted start-ok]}]
    (let [vs (floats values) acc (seq ^booleans (:data accepted))
          d (quot (count vs) (* n steps))
          tail (for [t (range (quot steps 2) steps) c (range n)] (nth vs (+ (* (+ (* t n) c) d) si)))]
      {:acceptance (/ (double (count (filter true? acc))) (* n steps))
       :slope-tail (/ (reduce + tail) (count tail))
       :starts-ok (every? true? (seq ^booleans (:data start-ok)))})))

(defn- model-names []
  (map keyword (str/split (System/getProperty "gen.parity.models" "vector,static") #",")))

(defn -main [& cells]
  (let [cells (set (or (seq cells) ["is" "mala" "hmc" "manychain"]))]
    (with-open [client (pjrt/client)]
      (doseq [m (model-names) :let [[model args obs :as spec] (pb/models m)
                                    si (slope-index spec)
                                    proto (grad/continuous-selection (:trace (gfi/generate model args obs (rng/key 0))) pb/selection)]]
        (when (cells "is")
          (doseq [n [1 100 3000 30000]]
            (timed-cell client {:spec :linreg_is :model m :N n}
                        #(ch/importance-run model args obs n)
                        (fn [{:keys [log-ml-estimate]}] {:log-ml-estimate (first (floats log-ml-estimate))}))))
        (when (cells "mala")
          (doseq [s [10 100 1000]]
            (timed-cell client {:spec :linreg_mala :model m :S s}
                        #(ch/mala-run model args obs pb/selection pb/tau s)
                        (chain-summary si s))))
        (when (cells "hmc")
          (doseq [l [8 32] s [10 100 1000]]
            (timed-cell client {:spec :linreg_hmc :model m :S s :L l}
                        #(ch/hmc-run model args obs pb/selection {:L l :eps pb/eps} s)
                        (chain-summary si s))))
        (when (cells "manychain")
          (doseq [s [100 1000] n [8 64 512 4096]]
            (timed-cell client {:spec :linreg_mala_manychain :model m :N n :S s}
                        #(ch/mala-chains-run model args obs pb/selection proto pb/tau n s)
                        (chains-summary si n s)))))))
  (shutdown-agents))
