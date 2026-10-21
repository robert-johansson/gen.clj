(ns gen.choice-profile
  "Where a traced choice's time goes (docs/benchmarks.md, \"Where gen.clj's
   1.7 µs a choice goes\"): gen.model-bench's linreg simulated with K
   observations, on the default backend.

   Run: clojure -M:bench -m gen.choice-profile alloc          (bytes allocated a simulate and a choice, K = 10, 100, 1000)
        clojure -M:bench -m gen.choice-profile jfr out.jfr    (a JFR recording, the profile settings, of 40 s at K = 1000)
   Read the recording with jfr view hot-methods, allocation-by-site, or jfr print."
  (:require [gen.gfi :as gfi]
            [gen.model-bench :as mb]
            [gen.random :as rng])
  (:import [java.lang.management ManagementFactory]
           [jdk.jfr Configuration Recording]))

(defn- simulates
  "How many simulates of linreg on xs run in seconds."
  ^long [xs ^double seconds]
  (let [key (rng/key 1)
        end (+ (System/nanoTime) (long (* seconds 1e9)))]
    (loop [n 0]
      (if (< (System/nanoTime) end)
        (do (gfi/simulate mb/linreg [xs] key) (recur (inc n)))
        n))))

(defn- allocation
  "The calling thread's bytes allocated a simulate, after a warm-up."
  []
  (let [^com.sun.management.ThreadMXBean mx (ManagementFactory/getThreadMXBean)
        tid (.getId (Thread/currentThread))
        key (rng/key 1)]
    (doseq [k [10 100 1000]]
      (let [xs (vec (range (double k)))
            n (quot 2000000 k)]
        (dotimes [_ n] (gfi/simulate mb/linreg [xs] key))
        (let [b0 (.getThreadAllocatedBytes mx tid)]
          (dotimes [_ n] (gfi/simulate mb/linreg [xs] key))
          (let [per (/ (double (- (.getThreadAllocatedBytes mx tid) b0)) n)]
            (println (format "K=%-5d %10.0f bytes a simulate, %6.0f bytes a choice" k per (/ per (+ k 2))))))))))

(defn- recording
  "A 10 s warm-up, then a JFR recording (the profile settings) of 40 s of
   simulates at K = 1000, dumped to out."
  [out]
  (let [k 1000
        xs (vec (range (double k)))]
    (println "warm-up:" (simulates xs 10.0) "simulates in 10 s")
    (let [rec (doto (Recording. (Configuration/getConfiguration "profile")) (.start))
          t0 (System/nanoTime)
          n (simulates xs 40.0)
          s (/ (- (System/nanoTime) t0) 1e9)]
      (.stop rec)
      (.dump rec (java.nio.file.Path/of ^String out (make-array String 0)))
      (println (format "profiled: %d simulates in %.1f s, %.1f µs a simulate, %.3f µs a choice; written to %s"
                       n s (/ (* s 1e6) n) (/ (* s 1e6) n (+ k 2)) out)))))

(defn -main [& [mode out]]
  (case mode
    "alloc" (allocation)
    "jfr" (recording (or out "choice-profile.jfr"))
    (println "usage: -m gen.choice-profile alloc | jfr out.jfr"))
  (shutdown-agents))
