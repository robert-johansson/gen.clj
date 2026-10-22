;; I3: the cost of a dist pushforward's log-prob against its draw's.
;; (1) every log-prob rebuilds (and revalidates) the draw;
;; (2) a discrete draw without an enumerable support (poisson) throws and
;;     catches an ex-info in enumerable-support on every log-prob;
;; (3) labels over a categorical: one log-prob of the draw per support value,
;;     each recomputing log(probs).
(require '[gen.array :as array] '[gen.dist :as d] '[gen.dist.core :as dc] '[gen.dist.dsl :refer [dist]])

(dist shifted-exp [rate shift] (array/+ (d/exponential rate) shift))
(dist shifted-poisson [lam] (array/+ (d/poisson lam) 1))
(dist labeled-cat [labels probs] (nth labels (d/categorical probs)))

(defn time-ns [n f]
  (dotimes [_ (quot n 4)] (f))
  (let [t0 (System/nanoTime)] (dotimes [_ n] (f)) (/ (- (System/nanoTime) t0) (double n))))

(let [e (d/exponential 2.0) se (shifted-exp 2.0 1.0)
      p (d/poisson 3.0) sp (shifted-poisson 3.0)]
  (println "exponential log-prob ns" (time-ns 200000 #(dc/log-prob e 1.5)))
  (println "shifted-exp log-prob ns" (time-ns 200000 #(dc/log-prob se 2.5)))
  (println "poisson log-prob ns    " (time-ns 200000 #(dc/log-prob p 3)))
  (println "shifted-poisson log-prob ns" (time-ns 200000 #(dc/log-prob sp 4))))

;; count the exceptions thrown inside one shifted-poisson log-prob
(let [sp (shifted-poisson 3.0) n (atom 0)
      orig dc/-support]
  (with-redefs [dc/support (fn [d] (try (orig d) (catch clojure.lang.ExceptionInfo e (swap! n inc) (throw e))))]
    (dc/log-prob sp 4))
  (println "ex-infos thrown by one shifted-poisson log-prob:" @n))

(doseq [k [10 100 1000]]
  (let [labels (vec (map #(keyword (str "l" %)) (range k)))
        probs (vec (repeat k (/ 1.0 k)))
        c (d/categorical probs)
        lc (labeled-cat labels probs)]
    (println "K" k "categorical log-prob us" (/ (time-ns 2000 #(dc/log-prob c 3)) 1000.0)
             " labeled-cat log-prob us" (/ (time-ns (max 20 (quot 20000 k)) #(dc/log-prob lc :l3)) 1000.0))))
