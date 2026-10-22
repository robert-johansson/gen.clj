;; The cost of gen.array's composite lgamma on a host scalar, the path every
;; scalar gamma, beta, poisson, binomial and negative-binomial log density
;; takes on the JVM backend: time and bytes allocated per call, against exp
;; and against the same Lanczos sum written as a primitive double loop.
(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc])
(import '[java.lang.management ManagementFactory])
(def ^com.sun.management.ThreadMXBean tmx (ManagementFactory/getThreadMXBean))
(defn alloc [] (.getThreadAllocatedBytes tmx (.getId (Thread/currentThread))))

(def ^"[D" cof (double-array [0.99999999999980993 676.5203681218851 -1259.1392167224028 771.32342877765313
                              -176.61502916214059 12.507343278686905 -0.13857109526572012
                              9.9843695780195716e-6 1.5056327351493116e-7]))
(defn lgamma-prim ^double [^double x]
  ;; x >= 0.5 only, for the comparison
  (let [z (- x 1.0)
        s (loop [i 1 s (aget cof 0)] (if (< i 9) (recur (inc i) (+ s (/ (aget cof i) (+ z i)))) s))
        t (+ z 7.5)]
    (+ 0.9189385332046728 (- (* (+ z 0.5) (Math/log t)) t) (Math/log s))))

(defn measure [label f]
  (let [n 2000000
        xs (double-array (map #(+ 0.6 (* 0.001 %)) (range 1000)))]
    (dotimes [_ 3] (dotimes [i n] (f (aget xs (rem i 1000)))))
    (let [a0 (alloc) t0 (System/nanoTime)]
      (dotimes [i n] (f (aget xs (rem i 1000))))
      (let [t (- (System/nanoTime) t0) a (- (alloc) a0)]
        (printf "%-28s %8.1f ns/call %8.0f bytes/call%n" label (/ t (double n)) (/ a (double n)))))))

(measure "array/exp (host)" #(array/exp %))
(measure "array/lgamma (host)" #(array/lgamma %))
(measure "primitive Lanczos" #(lgamma-prim %))
(println "agree:" (array/lgamma 3.7) (lgamma-prim 3.7))
(let [pois (d/poisson 3.0) norm (d/normal 0.0 1.0)]
  (measure "poisson logpdf (host)" #(dc/log-prob pois (long (* 3 %))))
  (measure "normal logpdf (host)" #(dc/log-prob norm %)))
(flush)
