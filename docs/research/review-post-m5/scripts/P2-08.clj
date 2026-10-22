;; P2-08: laplace's lower bound nextafter(-1, 0) is a float64 constant; in float32 it is -1.0,
;; so u = -1 is drawn and log1p(-|u|) = -Inf: a +Inf draw. Shown with gen.random/uniform at
;; float32 on the JVM backend (the dist itself draws at the backend's float, float32 on MLX).
(require '[gen.random :as rng] '[gen.array :as array])
(let [n (bit-shift-left 1 24)
      u (rng/uniform (rng/key 5) [n] {:low (Math/nextAfter -1.0 0.0) :high 1.0 :dtype :float32})
      xs ^doubles (double-array (array/->clj u))
      at-1 (count (filter #(== -1.0 %) xs))]
  (println "float32 low bound as stored:" (double (float (Math/nextAfter -1.0 0.0))))
  (println "of 2^24 float32 uniforms on [nextafter(-1,0), 1): exactly -1.0:" at-1 "(expected about 2)")
  (println "laplace's transform there: loc + scale * sign(u) * log1p(-|u|) =" (* 1.0 -1.0 (Math/log1p -1.0)))
  (let [un (rng/uniform (rng/key 5) [n] {:low (double (Math/nextAfter (float -1.0) 0.0)) :high 1.0 :dtype :float32})]
    (println "with rng/normal's float32 bound instead: exactly -1.0:" (count (filter #(== -1.0 %) (array/->clj un))))))
