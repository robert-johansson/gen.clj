;; P2-07: dirichlet's simplex check (|sum v - 1| <= 1e-6) is float64's tolerance; in float32 a
;; draw of many categories can fail it. MLX is not run here: this emulates its float32 sums
;; (sequential and pairwise accumulation bracket a SIMD reduction) on the draws' own values.
(require '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.random :as rng] '[gen.array :as array])

(defn f32-seq-sum ^double [xs] (double (reduce (fn [^double acc x] (double (float (+ (float acc) (float x))))) 0.0 xs)))
(defn f32-pair-sum ^double [xs]
  (let [v (vec xs) n (count v)]
    (cond (zero? n) 0.0 (= 1 n) (double (float (v 0)))
          :else (let [h (quot n 2)] (double (float (+ (float (f32-pair-sum (subvec v 0 h))) (float (f32-pair-sum (subvec v h))))))))))

(defn trial [k alpha key sum]
  ;; float32 gamma draws, normalized in float32 as the sampler does: g / sum(g)
  (let [g (mapv #(double (float %)) (array/->clj (dc/sample (d/gamma alpha 1.0) key [k])))
        s (float (sum g))
        v (mapv #(double (float (/ (float %) s))) g)]
    (Math/abs (- (sum v) 1.0))))

(doseq [k [10 100 1000 10000] [label sum] [["sequential" f32-seq-sum] ["pairwise" f32-pair-sum]]]
  (let [errs (mapv #(trial k 1.0 % sum) (rng/split (rng/key 3) 200))]
    (println "K" k label ": draws failing the 1e-6 check" (count (filter #(> % 1e-6) errs)) "of 200; max |sum - 1|" (apply max errs))))
;; for reference, float64 (the JVM backend's own draws)
(let [lps (mapv #(dc/log-prob (d/dirichlet (vec (repeat 10000 1.0))) (dc/sample (d/dirichlet (vec (repeat 10000 1.0))) %)) (rng/split (rng/key 4) 20))]
  (println "float64, K 10000: -Inf log-probs" (count (filter #(= ##-Inf (double (array/item %))) lps)) "of 20"))
