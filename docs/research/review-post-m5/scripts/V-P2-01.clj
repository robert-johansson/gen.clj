;; Verifier, P2-01: gamma/beta/dirichlet zero draws, the laws they break, and the derived zero rate.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.dist.core :as dc]
         '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm] '[gen.selection :as sel] '[gen.diff :as diff])

;; 1. The zero rate of gamma(0.005, 1), derived: the draw is G·pow(U, 200), and pow rounds to 0 when
;;    U^200 < 2^-1075 (half the smallest subnormal), so P(zero) = P(U < 2^(-1075/200)) = 2^-5.375
(let [n 20000
      xs (array/->clj (dc/sample (d/gamma 0.005 1.0) (rng/key 31) [n]))
      zeros (count (filter zero? xs))
      p (Math/pow 2.0 (/ -1075.0 200))]
  (println "gamma(0.005): zero draws" zeros "of" n "; derived rate" (format "%.4f" p) "→ expected" (format "%.0f ± %.0f" (* n p) (* 5 (Math/sqrt (* n p (- 1 p))))))
  (println "  log-prob at 0:" (dc/log-prob (d/gamma 0.005 1.0) 0.0)))

;; 2. The laws: a trace whose :g underflowed. An update that revisits :g (a new argument) weighs
;;    Inf - Inf; regenerate of another site likewise. Law: update-weight-is-the-projection-change
;;    (the kept :g contributes lp(g) - lp(g) = 0), so both weights must be finite here.
(def m (gen [s] (let [g (trace :g (d/gamma 0.005 1.0))] (trace :x (d/normal s 1.0)) g)))
(let [tr (first (filter #(zero? (double (% :g))) (map #(gfi/simulate m [0.0] %) (rng/split (rng/key 32) 2000))))
      up (gfi/update tr [1.0] [diff/unknown-change] cm/EMPTY (rng/key 33))
      rg (gfi/regenerate tr [0.0] [diff/no-change] (sel/select :x) (rng/key 34))]
  (println "trace with g = 0: score" (:score tr) "| assess" (:weight (gfi/assess m [0.0] (:choices tr) (rng/key 1))))
  (println "update to s = 1 (x's density changes, g kept): weight" (:weight up)
           "; exact" (- (array/item (dc/log-prob (d/normal 1.0 1.0) (tr :x))) (array/item (dc/log-prob (d/normal 0.0 1.0) (tr :x)))))
  (println "regenerate :x: weight" (:weight rg) "; exact (forward-sampling: retained ratio) 0.0"))

;; 3. beta(0.003, 0.003): a gamma(0.003) draw is 0 with probability p0 = 2^(-1075·0.003); x/(x+y) is
;;    NaN when both are (p0²), 0 or 1 when one is (p0(1 - p0) each)
(let [n 20000
      p0 (Math/pow 2.0 (* -1075 0.003))
      xs (array/->clj (dc/sample (d/beta 0.003 0.003) (rng/key 35) [n]))
      lps (array/->clj (dc/log-prob (d/beta 0.003 0.003) (array/array xs)))]
  (println "beta(0.003, 0.003):" n "draws: NaN" (count (filter #(Double/isNaN %) xs)) (format "(derived %.0f)" (* n p0 p0))
           "| exactly 0" (count (filter #(== 0.0 %) xs)) "| exactly 1" (count (filter #(== 1.0 %) xs)) (format "(derived %.0f each)" (* n p0 (- 1 p0))))
  (println "  their log-probs: +Inf" (count (filter #(= ##Inf %) lps)) "| -Inf" (count (filter #(= ##-Inf %) lps))))
