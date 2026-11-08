(ns gen.widen-bench
  "The cost of widening 16-bit floats' bit patterns to floats on the host
   (gen.array.host/halves->floats, MLX's ->floats of a float16 or
   bfloat16 array; genclj-9y6y) against the loop of one at a time it
   replaced (here), over 16 rows of Qwen3's vocabulary of 151,936, in ns
   an entry. No device: the bits are random patterns.
   Run: clojure -M:bench -m gen.widen-bench   (add :c2 to compare with C2)"
  (:require [criterium.core :as crit]
            [gen.array.host :as host]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(def ^:private ^"[F" table
  (let [t (float-array 65536)]
    (dotimes [i 65536] (aset t i (Float/float16ToFloat (unchecked-short i))))
    t))

(defn- one-at-a-time
  "The loop halves->floats replaced (genclj-9y6y's read-halves): one entry
   at a time, up to n, a long argument (the array's size)."
  ^floats [^shorts bits ^long n dtype]
  (let [out (float-array n)]
    (if (= :bfloat16 dtype)
      (dotimes [j n] (aset out j (Float/intBitsToFloat (unchecked-int (bit-shift-left (bit-and (long (aget bits j)) 0xFFFF) 16)))))
      (let [t table]
        (dotimes [j n] (aset out j (aget t (bit-and (long (aget bits j)) 0xFFFF))))))
    out))

(defn -main [& _]
  (let [n (+ 3 (* 16 151936))
        r (java.util.Random. 11)
        ;; logits' patterns: 16-bit values of a spread like a row's
        bits {:float16 (let [a (short-array n)] (dotimes [i n] (aset a i (Float/floatToFloat16 (float (* 4.0 (.nextGaussian r)))))) a)
              :bfloat16 (let [a (short-array n)]
                          (dotimes [i n] (aset a i (unchecked-short (bit-shift-right (Float/floatToRawIntBits (float (* 4.0 (.nextGaussian r)))) 16))))
                          a)}
        ns-entry (fn [f] (/ (* 1e9 (double (first (:mean (crit/quick-benchmark* f {}))))) n))
        c2? (some #(= "-XX:-UseJVMCICompiler" %) (.getInputArguments (java.lang.management.ManagementFactory/getRuntimeMXBean)))]
    (println "JVM:" (System/getProperty "java.vm.name") (System/getProperty "java.version") (if c2? "(C2)" "(the default JIT)"))
    (doseq [dtype [:float16 :bfloat16]]
      (let [b (bits dtype)]
        (when-not (java.util.Arrays/equals (one-at-a-time b n dtype) (host/halves->floats b dtype))
          (throw (ex-info "the two loops differ" {:dtype dtype})))
        ;; three rounds, interleaved: the machine's noise shows as the range
        (let [rounds (vec (repeatedly 3 #(vector (ns-entry (fn [] (one-at-a-time b n dtype)))
                                                 (ns-entry (fn [] (host/halves->floats b dtype))))))
              span (fn [k] (let [xs (sort (map #(% k) rounds))] (format "%.3f (%.3f-%.3f)" (second xs) (first xs) (last xs))))]
          (println (format "%-9s ns an entry, median (range) of 3: one at a time %s, halves->floats %s"
                           (name dtype) (span 0) (span 1))))))))
