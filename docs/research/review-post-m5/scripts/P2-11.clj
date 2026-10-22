;; P2-11 (emulated, float32): binomial-sample's inversion counts CDF values below u for 40
;; steps; past k = n the terms are 0, so when the float32 CDF at n is below the largest
;; uniform (1 - 2^-23) a draw of u above it counts every later step: k = 40 > n.
;; This emulates the sampler's float32 recurrence (gen.dist/binomial-sample) on the host.
(defn f [x] (float x))
(defn cdf-at-n [n p]
  (let [p (f p) q (f (- 1.0 p))
        p0 (f (Math/exp (* n (Math/log1p (- p)))))]
    (loop [j 0 pk p0 cdf p0]
      (if (< j n)
        (let [pk (f (* pk (f (/ (f (* (f (- n j)) p)) (f (* (f (+ j 1)) q))))))]
          (recur (inc j) pk (f (+ cdf pk))))
        cdf))))
(def umax (- 1.0 (Math/pow 2 -23)))
(let [cases (for [n (range 1 41) p (map #(* 0.01 %) (range 1 51)) :when (< (* n p) 10)] [n p])
      short (filter (fn [[n p]] (< (cdf-at-n n p) umax)) cases)]
  (println "float32 (n, p) cases with n·p < 10, p <= 0.5:" (count cases)
           "| CDF at n below the largest uniform:" (count short))
  (println "examples [n p cdf(n) P(k > n)]:" (take 5 (map (fn [[n p]] (let [c (cdf-at-n n p)] [n p c (- umax c)])) short))))
