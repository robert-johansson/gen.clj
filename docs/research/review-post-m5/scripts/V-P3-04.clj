;; Verifier's independent check of P3-04: log-abs-det against the Jacobian
;; of the move's actual map, by central differences, where each perturbed
;; input gets a trace consistent with it (generate on the perturbed choice,
;; so a call's return value moves with its choice).
;; Model: x ~ N(0, 1) inside a call at :sub returning s = x² + 1; y ~ N(x, 1).
;; Kernel: u ~ N(0, 1); x' = x + s·u, u' = -u, with s read from the call.
;; Derived: ∂x'/∂x = 1 + 2xu, ∂x'/∂u = s, ∂u'/∂u = -1, so
;; log |det J| = log |1 + 2xu|; with s held constant it would be 0.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.random :as rng]
         '[gen.array :as array] '[gen.choicemap :as cm] '[gen.inference.trace-kernel :as tk])

(def inner (gen [] (let [x (trace :x (d/normal 0 1))] (array/+ 1.0 (array/square x)))))
(def model (gen [] (let [s (splice :sub inner)] (trace :y (d/normal (array/sqrt (array/- s 1.0)) 1)) s)))
(defn trace-at [x] (:trace (gfi/generate model [] (cm/merge (cm/set-choice (cm/choicemap) [:sub :x] x) (cm/choicemap :y 0.4))
                                         (rng/key 1))))
(def kernel (gen [tr] (let [u (trace :u (d/normal 0 1))]
                        [(cm/set-choice (cm/choicemap) [:sub :x] (array/+ (tr [:sub :x]) (array/* (tr :sub) u)))
                         (cm/choicemap :u (array/negate u))])))

(defn move [x u]
  (let [[upd bwd] (:retval (gfi/assess kernel [(trace-at x)] (cm/choicemap :u u) (rng/key 2)))]
    [(double (array/item (cm/get-choice upd [:sub :x]))) (double (array/item (cm/get-choice bwd :u)))]))

(let [x 0.6 u 0.8 h 1e-6
      col (fn [dx du] (let [[a1 b1] (move (+ x dx) (+ u du)) [a0 b0] (move (- x dx) (- u du))]
                        [(/ (- a1 a0) (* 2 h)) (/ (- b1 b0) (* 2 h))]))
      [[a c] [b e]] [(col h 0.0) (col 0.0 h)]
      fd (Math/log (Math/abs (- (* a e) (* b c))))
      tr (trace-at x)
      kc (cm/choicemap :u u)
      ret (:retval (gfi/assess kernel [tr] kc (rng/key 2)))]
  (println "derived log|1 + 2xu| =" (Math/log (Math/abs (+ 1 (* 2 x u))))
           " central differences of the move =" fd
           " tk/log-abs-det =" (tk/log-abs-det kernel tr [] kc ret (constantly true))))
