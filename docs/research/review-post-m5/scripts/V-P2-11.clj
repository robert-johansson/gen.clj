;; Verifier, P2-11: the binomial inversion on the JVM backend with a float32 p, the real code path.
;; The JVM backend rounds float32 results per op (as MLX), the CDF is accumulated in float32, and u
;; is the backend's default float (float64 on the JVM): a u above the float32 CDF at n counts all 40
;; steps, a draw of 40 > n.
(require '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.random :as rng] '[gen.array :as array])
(defn f ^double [x] (double (float x)))
(defn cdf32 [n p] ; the sampler's recurrence, rounded per op as the JVM backend rounds float32
  (let [p (f p) q (f (- 1.0 p)) p0 (f (Math/exp (f (* n (f (Math/log1p (f (- p))))))))]
    (loop [j 0 pk p0 cdf p0]
      (if (< j 40)
        (let [pk (f (* pk (f (/ (f (* (f (- n j)) p)) (f (* (f (+ j 1)) q))))))] (recur (inc j) pk (f (+ cdf pk))))
        cdf))))
(let [cases (for [n (range 1 40) i (range 1 51) :let [p (* 0.01 i)] :when (< (* n p) 10)] [n p (cdf32 n p)])
      [n p c] (apply min-key #(nth % 2) cases)
      batch 1000000
      p32 (array/astype (array/full [batch] p) :float32)
      dd (d/binom n p32)
      draws (mapcat (fn [k] (array/->clj (dc/sample dd k))) (rng/split (rng/key 41) 8))
      above (frequencies (filter #(> % n) draws))]
  (println "cases" (count cases) "| with a float32 CDF at n below 1:" (count (filter #(< (nth % 2) 1.0) cases)))
  (println "lowest float32 CDF at n: n" n "p" p "CDF" c "; 1 - CDF" (- 1.0 c))
  (println "binom(n, float32 p), 8e6 draws on the JVM: draws above n" above "; expected about" (* 8e6 (- 1.0 c)))
  (println "dtype of p:" (array/dtype p32) "; of a draw's lp at 40:" (dc/log-prob (d/binom n p) 40)))
