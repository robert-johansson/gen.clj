;; A (kern ^:trace …) kernel statement whose arguments read the trace is a
;; dependent mixture (thesis Prop. 3.4.1) whose invariant nothing checks.
;; Model: flag ~ bernoulli(0.5), x ~ bernoulli(0.5), no observations: the
;; target is the prior, P(flag) = 0.5. Kernel: MH on :x when flag holds, on
;; :flag otherwise. Each MH move is stationary; the state-dependent choice
;; is not (flag = true is absorbing).
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.inference.kernel :as k])

(def model (gen [] (trace :flag (d/bernoulli 0.5)) (trace :x (d/bernoulli 0.5))))

(def by-argument
  ((k/kern ^:trace [tr] (k/mh (sel/select (if (tr :flag) :x :flag))))))

(def by-let
  ((k/kern ^:trace [tr] (let [a (if (tr :flag) :x :flag)] (k/mh (sel/select a))))))

(defn flag-rate [kernel n-chains steps]
  (let [ends (for [key (rng/split (rng/key 1) n-chains)]
               (let [[k0 k1] (rng/split key)
                     tr (:trace (gfi/generate model [] (cm/choicemap :flag false) k0))
                     r (k/run kernel tr {:samples steps :key k1})]
                 (boolean ((last (:traces r)) :flag))))]
    (/ (double (count (filter true? ends))) n-chains)))

(println "target P(flag) = 0.5")
(println "kernel args read tr, P(flag) after 20 steps over 2000 chains:" (flag-rate by-argument 2000 20))
(println "the same through a let:"
         (try (flag-rate by-let 200 20) (catch Exception e (str "loud: " (ex-message e)))))

;; A step size read from the trace and passed as a proposal argument: the
;; backward move is assessed with the same (old) argument, so the MH ratio
;; is wrong. Target: x ~ N(0, 1), E[x²] = 1. The proposal x' ~ N(x, 0.5 + |x|);
;; reading the scale inside the proposal is right (the backward reads trace').
(require '[gen.array :as array])
(def real-line (gen [] (trace :x (d/normal 0 1))))
(def walk-arg (gen [tr s] (trace :x (d/normal (tr :x) s))))
(def walk-own (gen [tr] (trace :x (d/normal (tr :x) (array/+ 0.5 (array/abs (tr :x)))))))
(def by-arg ((k/kern ^:trace [tr] (k/mh walk-arg [(array/+ 0.5 (array/abs (tr :x)))]))))
(def by-own ((k/kern ^:trace [tr] (k/mh walk-own []))))
(defn second-moment
  "[mean sd/sqrt(chains)] of E[x^2] over 8 independent chains of 50,000 steps."
  [kernel]
  (let [ms (for [key (rng/split (rng/key 4) 8)]
             (let [tr (:trace (gfi/generate real-line [] (cm/choicemap :x 0.0) (rng/key 3)))
                   xs (map #(double (% :x)) (:traces (k/run kernel tr {:samples 50000 :burn 1000 :key key})))]
               (/ (reduce + (map #(* % %) xs)) (count xs))))
        m (/ (reduce + ms) 8)]
    [m (Math/sqrt (/ (reduce + (map #(Math/pow (- % m) 2) ms)) (* 8 7)))]))
(println "target E[x^2] = 1; scale as a kernel argument read from tr [mean se]:" (second-moment by-arg)
         "; read inside the proposal:" (second-moment by-own))
