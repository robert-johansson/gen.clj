;; Verifier: (a) P2-06's direction against batched component parameters; (b) a missed sibling of
;; P2-11: Poisson's 40-step inversion with a float32 lambda (the JVM backend, float64 u).
(require '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.random :as rng] '[gen.array :as array])
(defn lse [& xs] (let [m (apply max xs)] (+ m (Math/log (reduce + (map #(Math/exp (- % m)) xs))))))
(defn lpn [x mu s] (- (+ (* 0.5 (Math/log (* 2 Math/PI))) (Math/log s) (* 0.5 (Math/pow (/ (- x mu) s) 2)))))
;; (a) weights [B K] and means [B K], dims [0 0]: B independent scalar mixtures today
(let [mix ((d/homogeneous-mixture d/normal) [[0.5 0.5] [0.9 0.1]] [[0.0 4.0] [1.0 5.0]] [1.0 1.0])
      comp0 (first (:components (:params mix)))]
  (println "batched params: log-prob of [0 4]" (dc/log-prob mix (array/array [0.0 4.0]))
           "; row by row" [(lse (+ (Math/log 0.5) (lpn 0 0 1)) (+ (Math/log 0.5) (lpn 0 4 1)))
                           (lse (+ (Math/log 0.9) (lpn 4 1 1)) (+ (Math/log 0.1) (lpn 4 5 1)))])
  (println "  component 0's value-shape (what the reviewer's direction would sum over):" (dc/value-shape comp0)))
;; (b) Poisson inversion, float32 lambda
(defn f ^double [x] (double (float x)))
(defn pcdf32 [lam] (let [lam (f lam) p0 (f (Math/exp (f (- lam))))]
                     (loop [j 0 p p0 cdf p0] (if (< j 39) (let [p (f (* p (f (/ lam (f (+ j 1))))))] (recur (inc j) p (f (+ cdf p)))) cdf))))
(let [lams (map #(* 0.1 %) (range 1 100))
      lam (apply min-key pcdf32 lams)
      c (pcdf32 lam)
      l32 (array/astype (array/full [1000000] lam) :float32)
      draws (mapcat #(array/->clj (dc/sample (d/poisson l32) %)) (rng/split (rng/key 51) 8))]
  (println "poisson: lowest float32 CDF at 39 for lam in 0.1..9.9: lam" lam "CDF" c "; 1 - CDF" (- 1.0 c))
  (println "  8e6 draws: value 40 drawn" (count (filter #(= 40 %) draws)) "times; P(K >= 40) exact"
           (Math/exp (array/item (array/logsumexp (dc/log-prob (d/poisson lam) (array/array (vec (range 40 120))))))))
  (println "  log-prob of 40:" (dc/log-prob (d/poisson lam) 40)))
(shutdown-agents)
