(ns gen.device.pjrt-bench
  "The per-call floor of gen.device.pjrt from the JVM, on the plugin's
   device: an execute of a trivial f32[2] program with its arguments on the
   device; the same with a host-to-device copy and a read back; compiling a
   small program; and two FFM calls with no device work (describe), the
   binding's own cost. Medians and p10/p90 in microseconds, and the
   device memory the process holds (nvidia-smi).

     clojure -J-Dgen.xla.plugin=<xla_cuda_plugin.so> -M:device -m gen.device.pjrt-bench"
  (:require [clojure.java.shell :as shell]
            [clojure.string :as str]
            [gen.device.pjrt :as pjrt]))

(def ^:private add
  "module @m {
     func.func public @main(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> {
       %0 = stablehlo.add %a, %b : tensor<2xf32>
       return %0 : tensor<2xf32>
     }
   }")

(def ^:private loop-sum
  "module @m {
     func.func public @main(%n: tensor<i32>) -> tensor<i32> {
       %c0 = stablehlo.constant dense<0> : tensor<i32>
       %c1 = stablehlo.constant dense<1> : tensor<i32>
       %r:2 = stablehlo.while(%i = %c0, %s = %c0) : tensor<i32>, tensor<i32>
        cond {
         %p = stablehlo.compare LT, %i, %n : (tensor<i32>, tensor<i32>) -> tensor<i1>
         stablehlo.return %p : tensor<i1>
       } do {
         %s2 = stablehlo.add %s, %i : tensor<i32>
         %i2 = stablehlo.add %i, %c1 : tensor<i32>
         stablehlo.return %i2, %s2 : tensor<i32>, tensor<i32>
       }
       return %r#1 : tensor<i32>
     }
   }")

(defn- times
  "The durations of n calls of f (ns), after warmup calls."
  [warmup n f]
  (dotimes [_ warmup] (f))
  (vec (repeatedly n #(let [t0 (System/nanoTime)] (f) (- (System/nanoTime) t0)))))

(defn- report [label ts]
  (let [s (vec (sort ts))
        q (fn [p] (/ (s (min (dec (count s)) (long (* p (count s))))) 1e3))]
    (println (format "%-44s n %5d  median %10.1f us  p10 %10.1f  p90 %10.1f"
                     label (count s) (q 0.5) (q 0.1) (q 0.9)))))

(defn- device-memory
  "The device memory this process holds, as nvidia-smi says (MiB), or why not."
  []
  (try
    (let [pid (str (.pid (java.lang.ProcessHandle/current)))
          {:keys [out exit]} (shell/sh "nvidia-smi" "--query-compute-apps=pid,used_memory"
                                       "--format=csv,noheader,nounits")]
      (if (zero? exit)
        (or (some (fn [line] (let [[p m] (map str/trim (str/split line #","))]
                               (when (= p pid) (str m " MiB"))))
                  (str/split-lines out))
            "none listed")
        "nvidia-smi failed"))
    (catch Exception e (str "no nvidia-smi: " (ex-message e)))))

(defn -main [& _]
  (let [plugin (pjrt/plugin)]
    (println "plugin" (:path plugin) "PJRT API" (:version plugin))
    (with-open [c (pjrt/client)]
      (println "client" (into {} (for [k [:platform :platform-version :device-kind :devices :options]] [k (k c)])))
      (println "device memory after the client:" (device-memory))
      (let [t0 (System/nanoTime)
            exe (pjrt/compile c add)
            first-compile (- (System/nanoTime) t0)
            a (pjrt/->device c (float-array [1 2]) [2])
            b (pjrt/->device c (float-array [3 4]) [2])]
        (try
          (println (format "first compile (f32[2] add)                    %10.1f us" (/ first-compile 1e3)))
          (report "(a) execute, arguments on the device"
                  (times 500 2000 #(run! pjrt/close (pjrt/execute exe [a b]))))
          (report "(b) host->device, execute, device->host"
                  (times 500 2000 #(with-open [x (pjrt/->device c (float-array [1 2]) [2])]
                                     (let [outs (pjrt/execute exe [x b])]
                                       (pjrt/->host (first outs))
                                       (run! pjrt/close outs)))))
          (report "(c) compile f32[2] add"
                  (times 3 50 #(pjrt/close (pjrt/compile c add))))
          (report "(c) compile a while loop (i32 sum)"
                  (times 3 50 #(pjrt/close (pjrt/compile c loop-sum))))
          (report "(d) describe: two FFM calls, no device work"
                  (times 2000 10000 #(pjrt/describe a)))
          (println "device memory at the end:" (device-memory))
          (finally (run! pjrt/close [a b exe])))))
    (shutdown-agents)))
