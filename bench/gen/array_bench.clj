(ns gen.array-bench
  "Per-op cost of gen.array on the JVM backend.
   Run: clojure -M:bench -m gen.array-bench   (add :c2 to compare with C2)"
  (:require [criterium.core :as crit]
            [gen.array :as a]))

(defn- ns-per-call [f]
  (* 1e9 (first (:mean (crit/quick-benchmark* f {})))))

(defn -main [& _]
  (let [x (Double/valueOf 1.5) y (Double/valueOf 2.5)
        v1 (a/array [1.0]) v100 (a/arange 0 100 1 {:dtype :float64})
        v10k (a/arange 0 10000 1 {:dtype :float64})
        m100 (a/reshape (a/arange 0 10000 1 {:dtype :float64}) [100 100])
        cases [["clojure.core/+ on two boxed doubles" #(+ (double x) (double y))]
               ["a/+ on two numbers" #(a/+ x y)]
               ["a/* number · number" #(a/* x y)]
               ["a/exp on a number" #(a/exp x)]
               ["a/+ [1] + [1]" #(a/+ v1 v1)]
               ["a/+ [100] + number" #(a/+ v100 2.0)]
               ["a/+ [100] + [100]" #(a/+ v100 v100)]
               ["a/+ [10000] + [10000]" #(a/+ v10k v10k)]
               ["a/exp [10000]" #(a/exp v10k)]
               ["a/sum [10000]" #(a/sum v10k)]
               ["a/logsumexp [10000]" #(a/logsumexp v10k)]
               ["a/+ [100 100] + [100] (broadcast)" #(a/+ m100 v100)]
               ["a/matmul [100 100] · [100 100]" #(a/matmul m100 m100)]]]
    (println "JVM:" (System/getProperty "java.vm.name") (System/getProperty "java.version"))
    (doseq [[label f] cases]
      (println (format "%-40s %12.1f ns" label (ns-per-call f))))))
