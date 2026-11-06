(ns gen.device.cuda-ndreg-bench
  "The ndreg_is cells (gen.ndreg-bench) as CUDA C kernels (bean
   genclj-4zmv; gen.device.cuda is parked since 1.2.1, genclj-z1gj): one particle's constrained generate per thread, or per
   group of lanes (a warp at 32), X and y uploaded once and shared; 1000
   particles, particle i on (rng/split root 1000)[i] (the handler's per
   particle runs, not gen.batched's one key). A timed run is a launch, the
   wait and the 1000 weights copied to the host; :kernel-ms the launch and
   wait alone. Each cell checks every particle's weight at lanes > 1
   against lanes 1 (which equals the handler, checked on the host), and
   reports the log mean weight. Prints one EDN map per cell and lane count.

   Run: clojure -J-Dgen.cuda.nvrtc=<so> -J-Dgen.ndreg.dir=<dir> -M:device:bench -m gen.device.cuda-ndreg-bench [D,M …]
   lanes with -Dgen.cuda.lane-counts=1,32 (default)."
  (:require [clojure.string :as str]
            [gen.choicemap :as cm]
            [gen.device.chain :as ch]
            [gen.device.cuda :as cuda]
            [gen.ndreg-bench :as nb]
            [gen.random :as rng]))

(defn- ms [^long t0] (/ (- (System/nanoTime) t0) 1e6))

(defn- quantile [sorted q]
  (nth sorted (min (dec (count sorted)) (long (Math/floor (* q (count sorted)))))))

(defn- log-mean-exp [^floats ws]
  (let [m (areduce ws i m Double/NEGATIVE_INFINITY (max m (double (aget ws i))))]
    (+ m (Math/log (/ (areduce ws i s 0.0 (+ s (Math/exp (- (aget ws i) m)))) (alength ws))))))

(defn -main [& args]
  (let [dir (or (System/getProperty "gen.ndreg.dir") (throw (ex-info "-Dgen.ndreg.dir=<the spec's data>" {})))
        budget-ms (* 1000.0 (Double/parseDouble (System/getProperty "gen.parity.budget-s" "15")))
        lane-counts (mapv #(Long/parseLong %) (str/split (System/getProperty "gen.cuda.lane-counts" "1,32") #","))
        chosen (if (seq args) (mapv (fn [a] (mapv #(Long/parseLong %) (str/split a #","))) args) nb/cells)
        n nb/n-particles
        root (rng/key-data (rng/key 7))]
    (doseq [[D M] chosen]
      (let [{:strs [X y]} (nb/read-safetensors (str dir "/ndreg_D" D "_M" M ".safetensors"))
            [Xa ya log-z] (nb/cell-data dir D M)
            t0 (System/nanoTime)
            {:keys [program]} (ch/weight-program nb/model [Xa] (cm/choicemap :y ya))
            stage-ms (ms t0)
            xb (cuda/upload (:data X))
            yb (cuda/upload (:data y))
            reference (volatile! nil)]
        (try
          (doseq [lanes lane-counts]
            (let [t1 (System/nanoTime)
                  src (cuda/source program {:lanes lanes})
                  emit-ms (ms t1)
                  k (cuda/compile src)
                  bufs (cuda/buffers k n)
                  scratch (cuda/scratch k n)
                  opts {:inputs [xb yb] :scratch scratch}]
              (try
                (let [t2 (System/nanoTime)
                      _ (cuda/launch! k bufs n root opts)
                      first-ms (ms t2)
                      _ (dotimes [i 2] (cuda/launch! k bufs n (rng/key-data (rng/key (+ 1001 i))) opts))
                      times (loop [i 0 times [] spent 0.0]
                              (if (or (< i 3) (and (< i 20) (< spent budget-ms)))
                                (let [t (System/nanoTime) _ (cuda/launch! k bufs n (rng/key-data (rng/key i)) opts) dt (ms t)]
                                  (recur (inc i) (conj times dt) (+ spent dt)))
                                (vec (sort times))))
                      kernel (vec (sort (for [i (range (min 10 (count times)))]
                                          (let [t (System/nanoTime)] (cuda/launch! k bufs n (rng/key-data (rng/key i)) (assoc opts :copy? false)) (ms t)))))
                      _ (cuda/launch! k bufs n root opts)
                      ws (cuda/read-floats (first bufs))
                      _ (when (= 1 lanes) (vreset! reference ws))
                      ^floats ref @reference
                      worst (when (and ref (> lanes 1))
                              (reduce max (map (fn [a b] (/ (Math/abs (- (double a) (double b))) (Math/abs (double b)))) ws ref)))]
                  (prn {:spec :ndreg_is :D D :M M :N n :backend "cuda" :lanes lanes
                        :stage-ms stage-ms :emit-ms emit-ms :compile-ms (:compile-ms k) :c-chars (count (:source src))
                        :registers (:registers k) :local-bytes (:local-bytes k) :scratch-bytes (:scratch-bytes k)
                        :first-ms first-ms :warmup-ms (+ stage-ms emit-ms (:compile-ms k) first-ms)
                        :median-ms (quantile times 0.5) :p10-ms (quantile times 0.1) :p90-ms (quantile times 0.9)
                        :kernel-ms (quantile kernel 0.5) :runs (count times)
                        :log-ml-estimate (log-mean-exp ws) :exact-log-z log-z :max-rel-diff-vs-lanes-1 worst})
                  (flush))
                (finally (cuda/free-buffers bufs) (cuda/free-device scratch) (cuda/unload! k)))))
          (finally (cuda/free-device xb) (cuda/free-device yb))))))
  (shutdown-agents))
