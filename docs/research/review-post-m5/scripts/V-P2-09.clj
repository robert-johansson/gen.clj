;; Verifier, P2-09: independent checks of fold-in's 32-bit data and Recurse's node keys.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.combinators :as comb]
         '[gen.random :as rng] '[gen.random.threefry :as tf] '[gen.choicemap :as cm])

;; 1. fold-in against JAX 0.9.1 (run separately): fold_in(PRNGKey(1), 5) = [3243370355 1313272271],
;;    and JAX gives the same key for np.int64(2^32 + 5) (it wraps typed data; a Python int raises).
(println "fold-in(key 1, 5)        =" (rng/key-data (rng/fold-in (rng/key 1) 5)) " JAX: [3243370355 1313272271]")
(println "fold-in(key 1, 2^32 + 5) =" (rng/key-data (rng/fold-in (rng/key 1) (+ 5 (bit-shift-left 1 32)))))

;; 2. The proposed 64-bit fold: [hi32(d) lo32(d)] equals today's for d < 2^32, and differs beyond.
(defn fold64 [k d] (let [[k0 k1] (rng/key-data k) w (tf/hash2x32 k0 k1 (unsigned-bit-shift-right d 32) (bit-and d 0xFFFFFFFF))]
                     [(unsigned-bit-shift-right w 32) (bit-and w 0xFFFFFFFF)]))
(let [k (rng/key 7) ds (concat [0 1 5 4294967295] (map #(long (* % 1e8)) (range 1 40)))]
  (println "64-bit fold = today's for" (count ds) "values below 2^32:" (every? #(= (fold64 k %) (rng/key-data (rng/fold-in k %))) ds))
  (println "64-bit fold of 5 and 5 + 2^32 differ:" (not= (fold64 k 5) (fold64 k (+ 5 (bit-shift-left 1 32))))))

;; 3. max-branch 8: a chain of child 0 has node (8^d - 1)/7, whose residue mod 2^32 is fixed from depth 11 on
(def chain8 (comb/recurse (gen [depth] {:value (trace :v (d/normal 0.0 1.0)) :children (if (< depth 14) [(inc depth)] [])})
                          (gen [v ws] v) 8))
(let [tr (gfi/simulate chain8 [0] (rng/key 21))
      node (fn [depth] (reduce (fn [n _] (+ (* n 8) 1)) 0 (range depth)))
      v (fn [depth] (tr [(node depth) :production :v]))]
  (println "max-branch 8, v at depths 8..14:" (mapv #(format "%.4f" (double (v %))) (range 8 15)))
  (println "  nodes mod 2^32:" (mapv #(mod (node %) (bit-shift-left 1 32)) (range 8 15))))

;; 4. Two different subtrees (max-branch 4): root children 0 and 1, each a chain of child 3.
;;    After 16 steps (4^16 = 2^32) both chains reach the same residue: two distinct nodes, one key.
(def two-chains (comb/recurse (gen [[side depth]] {:value (trace :v (d/normal 0.0 1.0))
                                                   :children (cond (= side :root) [[:a 1] [:b 1]]
                                                                   (= side :leaf) []
                                                                   (< depth 17) (conj (vec (repeat 3 [:leaf 0])) [side (inc depth)])
                                                                   :else [])})
                              (gen [v ws] v) 4))
(let [tr (gfi/simulate two-chains [[:root 0]] (rng/key 22))
      nodes (sort (map first (cm/submaps (:choices tr))))
      chain (fn [start] (take 18 (iterate #(+ (* % 4) 3 1) start)))   ; child 3 of n is 4n + 4
      a (chain 1) b (chain 2)]
  (println "max-branch 4: node a at depth 17 =" (nth a 16) ", node b at depth 17 =" (nth b 16)
           ", equal mod 2^32:" (= (mod (nth a 16) (bit-shift-left 1 32)) (mod (nth b 16) (bit-shift-left 1 32))))
  (println "  their v:" (tr [(nth a 16) :production :v]) (tr [(nth b 16) :production :v])
           "| at depth 16 (distinct residues):" (tr [(nth a 15) :production :v]) (tr [(nth b 15) :production :v])))

;; 5. The implicit key stream: next-key folds an AtomicLong counter, so it repeats with period 2^32
(gfi/seed! 5)
(let [k0 (gfi/next-key)
      counter @#'gen.gfi/counter]
  (.set ^java.util.concurrent.atomic.AtomicLong counter (bit-shift-left 1 32))
  (println "next-key at counter 0 = next-key at counter 2^32:" (= k0 (gfi/next-key))))
(gfi/seed! 0)
