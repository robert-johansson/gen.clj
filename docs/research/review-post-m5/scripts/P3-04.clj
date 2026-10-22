;; trace-kernel/log-abs-det substitutes tracked values for the trace's
;; choices only, (assoc trace :choices mc'): a kernel that reads a call's
;; return value, (tr :sub), sees the old, untracked value, so the Jacobian
;; misses what flows through it.
;; Model: x ~ N(0, 1) inside a call at :sub that returns x; y ~ N(x, 1).
;; Kernel: u ~ N(0, 0.5), x' = x·e^u, u' = -u: |det J| = e^u (a bijection),
;; written once with (tr [:sub :x]) (the choice) and once with (tr :sub)
;; (the call's value, the same number).
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.inference.trace-kernel :as tk])

(def inner (gen [] (trace :x (d/normal 0 1))))
(def model (gen [] (let [x (splice :sub inner)] (trace :y (d/normal x 1)) x)))
(def tr (:trace (gfi/generate model [] (cm/choicemap [:sub :x] 0.7 :y 0.2) (rng/key 1))))

(defn scaler [read-x]
  (gen [tr] (let [u (trace :u (d/normal 0 0.5))]
              [(cm/set-choice (cm/choicemap) [:sub :x] (array/* (read-x tr) (array/exp u)))
               (cm/choicemap :u (array/negate u))])))

(def by-choice (scaler #(% [:sub :x])))
(def by-call (scaler #(% :sub)))

(println "(tr :sub) =" (tr :sub) " (tr [:sub :x]) =" (tr [:sub :x]))
(doseq [[name k] [["reads the choice" by-choice] ["reads the call's value" by-call]]]
  (let [{kc :choices ret :retval} (gfi/propose k [tr] (rng/key 2))
        u (array/item (cm/get-choice kc :u))]
    (println name ": expected log |det J| = u =" u ", got"
             (try (tk/log-abs-det k tr [] kc ret (constantly true)) (catch Exception e (str "loud: " (ex-message e)))))))
(println "MH with the call-reading kernel:"
         (try (select-keys (tk/mh tr by-call [] (rng/key 3)) [:accepted? :log-alpha])
              (catch Exception e (str "loud: " (ex-message e)))))

;; Silently wrong when the choice is also read: x' = x·s·e^u with s = e^x
;; the call's value, u' = -u. ∂x'/∂x = (1 + x)·e^{x+u}, so log |det J| =
;; log(1 + x) + x + u; reading s from the call drops ∂s/∂x, log(1 + x) =
;; log 1.3 = 0.262 at x = 0.3. (Only the determinant at the point is
;; compared; the map is invertible near it.)
(def inner-exp (gen [] (array/exp (trace :x (d/normal 0 1)))))
(def model-exp (gen [] (let [s (splice :sub inner-exp)] (trace :y (d/normal s 1)) s)))
(def tr-exp (:trace (gfi/generate model-exp [] (cm/choicemap [:sub :x] 0.3 :y 1.1) (rng/key 4))))
(defn scaled [read-s]
  (gen [tr] (let [u (trace :u (d/normal 0 1))]
              [(cm/set-choice (cm/choicemap) [:sub :x] (array/* (array/* (tr [:sub :x]) (read-s tr)) (array/exp u)))
               (cm/choicemap :u (array/negate u))])))
(doseq [[name k] [["s = e^x from the choice" (scaled #(array/exp (% [:sub :x])))]
                  ["s from the call's value" (scaled #(% :sub))]]]
  (let [{kc :choices ret :retval} (gfi/propose k [tr-exp] (rng/key 5))
        u (array/item (cm/get-choice kc :u))]
    (println name ": log |det J| ="
             (try (tk/log-abs-det k tr-exp [] kc ret (constantly true)) (catch Exception e (str "loud: " (ex-message e))))
             " expected" (+ (Math/log 1.3) 0.3 u))))
