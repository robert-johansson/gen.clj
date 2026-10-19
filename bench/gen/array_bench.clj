(ns gen.array-bench
  "Per-op cost of gen.array on the JVM backend.
   Run: clojure -M:bench -m gen.array-bench   (add :c2 to compare with C2)"
  (:require [criterium.core :as crit]
            [gen.array :as array]))

(defn- ns-per-call [f]
  (* 1e9 (first (:mean (crit/quick-benchmark* f {})))))

(defn -main [& _]
  (let [x (Double/valueOf 1.5) y (Double/valueOf 2.5)
        v1 (array/array [1.0]) v100 (array/arange 0 100 1 {:dtype :float64})
        v10k (array/arange 0 10000 1 {:dtype :float64})
        m100 (array/reshape (array/arange 0 10000 1 {:dtype :float64}) [100 100])
        cases [["clojure.core/+ on two boxed doubles" #(+ (double x) (double y))]
               ["array/+ on two numbers" #(array/+ x y)]
               ["array/* number · number" #(array/* x y)]
               ["array/exp on a number" #(array/exp x)]
               ["array/+ [1] + [1]" #(array/+ v1 v1)]
               ["array/+ [100] + number" #(array/+ v100 2.0)]
               ["array/+ [100] + [100]" #(array/+ v100 v100)]
               ["array/+ [10000] + [10000]" #(array/+ v10k v10k)]
               ["array/exp [10000]" #(array/exp v10k)]
               ["array/sum [10000]" #(array/sum v10k)]
               ["array/logsumexp [10000]" #(array/logsumexp v10k)]
               ["array/+ [100 100] + [100] (broadcast)" #(array/+ m100 v100)]
               ["array/matmul [100 100] · [100 100]" #(array/matmul m100 m100)]]]
    (println "JVM:" (System/getProperty "java.vm.name") (System/getProperty "java.version"))
    (doseq [[label f] cases]
      (println (format "%-40s %12.1f ns" label (ns-per-call f))))))
