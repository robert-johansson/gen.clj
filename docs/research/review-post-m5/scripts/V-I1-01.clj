;; Verifier's check of I1-01: lgamma's cost on a host scalar, by criterium
;; (quick-bench), at a Lanczos point and at a reflected point; log-beta and
;; the gamma, beta and poisson log densities; and the direction's claim that
;; a primitive host path with the composite's sequence of double operations
;; (reflection and poles included) equals the composite bit for bit.
(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc]
         '[criterium.core :as crit])
(import '[java.lang.management ManagementFactory])
(def ^com.sun.management.ThreadMXBean tmx (ManagementFactory/getThreadMXBean))
(defn alloc ^long [] (.getThreadAllocatedBytes tmx (.getId (Thread/currentThread))))

(def ^"[D" cof (double-array [0.99999999999980993 676.5203681218851 -1259.1392167224028 771.32342877765313
                              -176.61502916214059 12.507343278686905 -0.13857109526572012
                              9.9843695780195716e-6 1.5056327351493116e-7]))

;; The composite's operations in its order, on primitive doubles.
(defn lgamma-host ^double [^double x]
  (let [reflect (< x 0.5)
        zin (if reflect (- 1.0 x) x)
        z (- zin 1.0)
        s (loop [i 1 s (aget cof 0)]
            (if (< i 9) (recur (inc i) (+ s (/ (aget cof i) (+ z (double i))))) s))
        t (+ z 7.5)
        lg (+ (+ 0.9189385332046728 (- (* (+ z 0.5) (Math/log t)) t)) (Math/log s))]
    (cond (and (<= x 0.0) (== x (Math/floor x))) Double/POSITIVE_INFINITY
          reflect (- (- (Math/log Math/PI) (Math/log (Math/abs (Math/sin (* Math/PI x))))) lg)
          :else lg)))

(defn bytes-per-call [f]
  (dotimes [_ 200000] (f))
  (let [n 1000000 a0 (alloc)]
    (dotimes [_ n] (f))
    (/ (- (alloc) a0) (double n))))

(defn bench [label f]
  (let [r (crit/quick-benchmark* f {})
        mean-ns (* 1e9 (first (:mean r)))
        sd-ns (* 1e9 (Math/sqrt (first (:variance r))))]
    (printf "%-30s %9.1f ns (sd %6.1f)  %7.0f bytes/call%n" label mean-ns sd-ns (bytes-per-call f))
    (flush)))

;; 1. Bit-identity of the primitive host path against the composite.
(let [xs (concat [0.5 0.49999999999999994 1.0 2.0 3.7 0.0 -0.0 -1.0 -2.0 -2.5 -0.3 1e-300 -1e-300
                  171.6 1e10 1e300 ##Inf ##-Inf ##NaN]
                 (let [r (java.util.Random. 42)]
                   (repeatedly 200000 #(* (- (.nextDouble r) 0.3) (Math/pow 10.0 (- (* 6.0 (.nextDouble r)) 2.0))))))
      bad (remove (fn [^double x]
                    (let [a (double (array/lgamma x)) b (lgamma-host x)]
                      (or (== (Double/doubleToRawLongBits a) (Double/doubleToRawLongBits b))
                          (and (Double/isNaN a) (Double/isNaN b)))))
                  xs)]
  (println "points checked:" (count xs) " differing bitwise:" (count bad) " first:" (take 5 bad)))
(println "long input 5 ->" (array/lgamma 5) "vs" (lgamma-host 5.0))

;; 2. Costs.
(let [x 3.7 y 0.3 a 2.5 b 1.5
      gam (d/gamma 2.0 1.5) bet (d/beta 2.0 3.0) pois (d/poisson 3.0) norm (d/normal 0.0 1.0)]
  (bench "array/exp 3.7" #(array/exp x))
  (bench "array/lgamma 3.7" #(array/lgamma x))
  (bench "array/lgamma 0.3 (reflected)" #(array/lgamma y))
  (bench "array/digamma 3.7" #(array/digamma x))
  (bench "array/log-beta 2.5 1.5" #(array/log-beta a b))
  (bench "primitive lgamma-host 3.7" #(lgamma-host x))
  (bench "primitive lgamma-host 0.3" #(lgamma-host y))
  (bench "normal log-prob" #(dc/log-prob norm 0.4))
  (bench "gamma log-prob" #(dc/log-prob gam 1.2))
  (bench "beta log-prob" #(dc/log-prob bet 0.4))
  (bench "poisson log-prob" #(dc/log-prob pois 4)))
