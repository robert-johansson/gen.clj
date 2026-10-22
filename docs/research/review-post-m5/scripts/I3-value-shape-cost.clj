;; The value-shape check on the handler's hot path: an MH step (regenerate
;; of one choice) rescoring 200 kept sites, with dc/value-shape as it is and
;; with a constant-time stand-in for the elementwise families (same result
;; here, where every parameter is a host scalar).
(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.random :as rng]
         '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.array.shape])
(defn time-us [n f]
  (dotimes [_ n] (f))
  (let [t0 (System/nanoTime)] (dotimes [_ n] (f)) (/ (- (System/nanoTime) t0) (* 1000.0 n))))
(def model (gen [] (let [mu (trace :mu (d/normal 0.0 1.0))]
                     (mapv (fn [i] (trace [:y i] (d/normal mu 1.0))) (range 200)))))
(let [obs (cm/from-map {:y (zipmap (range 200) (repeat 0.3))})
      tr (:trace (gfi/generate model [] obs (rng/key 0)))
      s (sel/select :mu) k (rng/key 1)
      orig dc/elementwise-value-shape]
  (dotimes [round 3]
    (println "round" round)
    (println "  regenerate :mu, 200 kept sites, us:" (time-us 3000 #(gfi/regenerate tr s k)))
    (with-redefs [dc/elementwise-value-shape
                  (fn [d] (reduce-kv (fn [s _ v] (if (and (array/array? v) (not (array/host? v)))
                                                   (gen.array.shape/broadcast s (array/shape v)) s))
                                     [] (:params d)))]
      (println "  ... elementwise-value-shape by reduce-kv, us:" (time-us 3000 #(gfi/regenerate tr s k))))
    (with-redefs [dc/value-shape (constantly nil)]
      (println "  ... no value-shape check at all, us:" (time-us 3000 #(gfi/regenerate tr s k))))))
