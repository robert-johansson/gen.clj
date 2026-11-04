(ns gen.llm.llama-threads-test
  "llama.cpp's default threads on a Linux hybrid CPU (genclj-xe8c): the
   physical performance cores, read from a sysfs tree (fakes of a Core
   Ultra 9 285K, 8 P-cores without hyper-threads and 16 E-cores; of a
   12th-generation i9, 8 P-cores with two threads each; of a CPU that is
   not hybrid), nil where there is no P-core list."
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [gen.llm.llama :as llama]))

(defn- sysfs
  "A temporary sysfs root: devices/cpu_core/cpus as cpus (unless nil), and
   each CPU's topology from core-of (cpu → [package core])."
  [cpus core-of n]
  (let [root (.toFile (java.nio.file.Files/createTempDirectory "sysfs" (make-array java.nio.file.attribute.FileAttribute 0)))]
    (when cpus
      (io/make-parents (io/file root "devices" "cpu_core" "cpus"))
      (spit (io/file root "devices" "cpu_core" "cpus") (str cpus "\n")))
    (doseq [c (range n) :let [[pkg core] (core-of c) dir (io/file root "devices" "system" "cpu" (str "cpu" c) "topology")]]
      (.mkdirs dir)
      (spit (io/file dir "physical_package_id") (str pkg "\n"))
      (spit (io/file dir "core_id") (str core "\n")))
    (str root)))

(deftest performance-cores
  (testing "a Core Ultra 9 285K: P-cores 0-7, no hyper-threads"
    (is (= 8 (llama/performance-cores (sysfs "0-7" (fn [c] [0 c]) 24)))))
  (testing "P-cores with two threads each count once"
    (is (= 8 (llama/performance-cores (sysfs "0-15" (fn [c] [0 (quot c 2)]) 24)))))
  (testing "a cpulist with gaps"
    (is (= 4 (llama/performance-cores (sysfs "0-1,4,6" (fn [c] [0 c]) 8)))))
  (testing "not a hybrid CPU: llama.cpp's own default"
    (is (nil? (llama/performance-cores (sysfs nil (fn [c] [0 c]) 8))))))
