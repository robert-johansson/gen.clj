(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc])
(defn kernel [n]
  (let [xs (map #(/ (double %) (dec n)) (range n))]
    (array/array (vec (for [a xs] (vec (for [b xs] (+ (Math/exp (* -0.5 (Math/pow (/ (- a b) 0.1) 2))) (if (= a b) 1e-3 0.0)))))))))
(defn ms [f k] (dotimes [_ 2] (f)) (let [ts (sort (for [_ (range k)] (let [t (System/nanoTime)] (f) (/ (- (System/nanoTime) t) 1e6))))] (nth ts (quot k 2))))
(doseq [n [50 200 500]]
  (let [K (kernel n) mu (array/zeros [n]) y (array/zeros [n]) k (if (= n 500) 3 7)
        build (ms #(d/mvnormal mu K) k)
        dist (d/mvnormal mu K)
        lp (ms #(dc/log-prob dist y) k)
        both (ms #(dc/log-prob (d/mvnormal mu K) y) k)]
    (println (format "n %4d  construct %9.2f ms  log-prob %9.2f ms  construct+log-prob %9.2f ms" n build lp both))))
(shutdown-agents)
