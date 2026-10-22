;; Verifier's independent check of P3-02: the stationarity law stated
;; directly. A stationary kernel applied once to exact draws of its target
;; gives exact draws of the target. Both targets here are priors (no
;; observations), so gfi/simulate draws them exactly.
;;
;; (a) flag, x ~ bernoulli(0.5); (k/mh (sel/select (if (tr :flag) :x :flag))).
;;     One step from the target: flag true stays true (MH on :x), flag false
;;     becomes true w.p. 0.5, so P(flag) = 0.5 + 0.25 = 0.75; the law says 0.5.
;; (b) x ~ N(0, 1); MH with the proposal N(x, s), s = 1 + x² read from tr as
;;     a kernel argument. The law says E[x²] = 1 after one step; a proposal
;;     that reads s itself is the control.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.selection :as sel] '[gen.inference.kernel :as k])

(defn mean-se [xs]
  (let [n (count xs) m (/ (reduce + xs) n)]
    [m (Math/sqrt (/ (reduce + (map #(Math/pow (- % m) 2) xs)) (* n (dec n))))]))

(defn one-step [model kernel n read key]
  (mapv (fn [kk] (let [[k0 k1] (rng/split kk)
                       t (gfi/simulate model [] k0)]
                   (read (:trace (kernel t k1)))))
        (rng/split key n)))

(def flags (gen [] (trace :flag (d/bernoulli 0.5)) (trace :x (d/bernoulli 0.5))))
(def flag-kernel ((k/kern ^:trace [tr] (k/mh (sel/select (if (tr :flag) :x :flag))))))
(println "(a) P(flag) one step from the target, 20000 draws [mean se]:"
         (mean-se (one-step flags flag-kernel 20000 #(if (% :flag) 1.0 0.0) (rng/key 31)))
         " law: 0.5; derived for the unchecked kernel: 0.75")

(def line (gen [] (trace :x (d/normal 0 1))))
(def walk-arg (gen [tr s] (trace :x (d/normal (tr :x) s))))
(def walk-own (gen [tr] (trace :x (d/normal (tr :x) (array/+ 1.0 (array/square (tr :x)))))))
(def by-arg ((k/kern ^:trace [tr] (k/mh walk-arg [(array/+ 1.0 (array/square (tr :x)))]))))
(def by-own ((k/kern ^:trace [tr] (k/mh walk-own []))))
(doseq [[name kernel] [["scale as a kernel argument" by-arg] ["scale read in the proposal" by-own]]]
  (println "(b)" name ": E[x²] one step from the target, 100000 draws [mean se]:"
           (mean-se (one-step line kernel 100000 #(let [x (double (% :x))] (* x x)) (rng/key 32)))
           " law: 1.0"))
