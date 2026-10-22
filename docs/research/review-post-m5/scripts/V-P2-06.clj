;; Verifier, P2-06: homogeneous mixtures of vector-valued normal components.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.dist.core :as dc]
         '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm])

(defn lse [& xs] (let [m (apply max xs)] (+ m (Math/log (reduce + (map #(Math/exp (- % m)) xs))))))
(defn lpn [x mu s] (- (+ (* 0.5 (Math/log (* 2 Math/PI))) (Math/log s) (* 0.5 (Math/pow (/ (- x mu) s) 2)))))

;; 3-d clusters at (0,0,0) and (4,4,4), sd 1, weights 0.3/0.7, at v = (0, 4, 4)
(def v (array/array [0.0 4.0 4.0]))
(let [mix ((d/homogeneous-mixture d/normal [1 1]) [0.3 0.7] [[0.0 0.0 0.0] [4.0 4.0 4.0]] [[1.0 1.0 1.0] [1.0 1.0 1.0]])
      whole (lse (+ (Math/log 0.3) (lpn 0 0 1) (lpn 4 0 1) (lpn 4 0 1)) (+ (Math/log 0.7) (lpn 0 4 1) (lpn 4 4 1) (lpn 4 4 1)))
      per (+ (lse (+ (Math/log 0.3) (lpn 0 0 1)) (+ (Math/log 0.7) (lpn 0 4 1)))
             (* 2 (lse (+ (Math/log 0.3) (lpn 4 0 1)) (+ (Math/log 0.7) (lpn 4 4 1)))))]
  (println "score of (0 4 4): gen.clj" (:weight (gfi/assess mix [] (cm/->Value v) (rng/key 1)))
           "| whole-value mixture (Gen.jl)" whole "| per-coordinate mixture" per))
;; dims [1 1] with means [K d] against dims [0 0] with the transposed means [d K]: the same distribution
(let [a ((d/homogeneous-mixture d/normal [1 1]) [0.3 0.7] [[0.0 0.0 0.0] [4.0 4.0 4.0]] [[1.0 1.0 1.0] [1.0 1.0 1.0]])
      b ((d/homogeneous-mixture d/normal [0 0]) [0.3 0.7] [[0.0 4.0] [0.0 4.0] [0.0 4.0]] [[1.0 1.0] [1.0 1.0] [1.0 1.0]])]
  (println "dims [1 1], means [K d]:" (dc/log-prob a v) "\ndims [0 0], means [d K]:" (dc/log-prob b v)))
;; a heterogeneous mixture of two vector normals does the same
(let [h ((d/heterogeneous-mixture [d/normal d/normal]) [0.3 0.7] [[0.0 0.0 0.0] [1.0 1.0 1.0]] [[4.0 4.0 4.0] [1.0 1.0 1.0]])]
  (println "heterogeneous [normal normal], score of (0 4 4):" (:weight (gfi/assess h [] (cm/->Value v) (rng/key 1)))))
;; batched weights [B K] (the sampler's comment: "the weights' batch axes are independent draws too")
(let [bw ((d/homogeneous-mixture d/normal) [[0.5 0.5] [0.9 0.1]] [0.0 4.0] [1.0 1.0])]
  (println "weights [2 2]: value-shape" (dc/value-shape bw) "log-prob of [0 4]" (dc/log-prob bw (array/array [0.0 4.0]))
           "; rows" (lse (+ (Math/log 0.5) (lpn 0 0 1)) (+ (Math/log 0.5) (lpn 0 4 1))) (lse (+ (Math/log 0.9) (lpn 4 0 1)) (+ (Math/log 0.1) (lpn 4 4 1)))))
