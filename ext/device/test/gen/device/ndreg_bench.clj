(ns gen.device.ndreg-bench
  "The ndreg_is cells (gen.ndreg-bench) as device programs through XLA
   (bean genclj-4zmv): batched importance sampling of the model staged from
   gen.clj's handler, X and y program inputs uploaded once and held on the
   device. A timed run is a key copied to the device, the program run, and
   the log-ML and the N weights copied back; :warmup-ms is staging,
   StableHLO, XLA's compile and the first run. Prints one EDN map per cell.

   Run: clojure -J-Dgen.xla.plugin=<so> -J-Dgen.ndreg.dir=<dir> -M:device:bench -m gen.device.ndreg-bench [D,M …]"
  (:require [clojure.string :as str]
            [gen.choicemap :as cm]
            [gen.device.chain :as ch]
            [gen.device.pjrt :as pjrt]
            [gen.device.stablehlo :as hlo]
            [gen.ndreg-bench :as nb]
            [gen.random :as rng]))

(defn- ms [^long t0] (/ (- (System/nanoTime) t0) 1e6))

(defn- quantile [sorted q]
  (nth sorted (min (dec (count sorted)) (long (Math/floor (* q (count sorted)))))))

(defn- key-buffer [client key]
  (let [[k0 k1] (rng/key-data key)]
    (pjrt/->device client (int-array [(unchecked-int (long k0)) (unchecked-int (long k1))]) [2] :u32)))

(defn- run-once
  "The program on key with the data buffers: [log-ml weights] on the host."
  [client exe key data]
  (with-open [k (key-buffer client key)]
    (let [outs (pjrt/execute exe (into [k] data))]
      (try (mapv pjrt/->host outs)
           (finally (run! #(.close ^java.io.Closeable %) outs))))))

(defn -main [& args]
  (let [dir (or (System/getProperty "gen.ndreg.dir") (throw (ex-info "-Dgen.ndreg.dir=<the spec's data>" {})))
        budget-ms (* 1000.0 (Double/parseDouble (System/getProperty "gen.parity.budget-s" "30")))
        chosen (if (seq args) (mapv (fn [a] (mapv #(Long/parseLong %) (str/split a #","))) args) nb/cells)]
    (with-open [client (pjrt/client)]
      (doseq [[D M] chosen]
        (let [stem (str dir "/ndreg_D" D "_M" M)
              {:strs [X y]} (nb/read-safetensors (str stem ".safetensors"))
              [Xa ya log-z] (nb/cell-data dir D M)
              t0 (System/nanoTime)
              {:keys [program]} (ch/importance-program nb/model [Xa] (cm/choicemap :y ya) nb/n-particles)
              stage-ms (ms t0)
              t1 (System/nanoTime)
              text (hlo/module program)
              emit-ms (ms t1)
              t2 (System/nanoTime)]
          (with-open [exe (pjrt/compile client text)
                      xb (pjrt/->device client (:data X) (:shape X) :f32)
                      yb (pjrt/->device client (:data y) (:shape y) :f32)]
            (let [compile-ms (ms t2)
                  data [xb yb]
                  t3 (System/nanoTime)
                  _ (run-once client exe (rng/key 1000) data)
                  first-ms (ms t3)]
              (dotimes [i 2] (run-once client exe (rng/key (+ 1001 i)) data))
              (let [[times out] (loop [i 0 times [] spent 0.0 out nil]
                                  (if (or (< i 3) (and (< i 20) (< spent budget-ms)))
                                    (let [t (System/nanoTime) o (run-once client exe (rng/key i) data) dt (ms t)]
                                      (recur (inc i) (conj times dt) (+ spent dt) o))
                                    [times out]))
                    s (vec (sort times))]
                (prn {:spec :ndreg_is :D D :M M :N nb/n-particles :backend "xla" :stage-ms stage-ms :emit-ms emit-ms
                      :compile-ms compile-ms :first-ms first-ms :warmup-ms (+ stage-ms emit-ms compile-ms first-ms)
                      :median-ms (quantile s 0.5) :p10-ms (quantile s 0.1) :p90-ms (quantile s 0.9) :runs (count s)
                      :log-ml-estimate (aget ^floats (:data (first out)) 0) :exact-log-z log-z})
                (flush))))))))
  (shutdown-agents))
