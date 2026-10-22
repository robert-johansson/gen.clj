;; P2-09: fold-in keeps 32 bits of its data, and Recurse folds in node numbers, which pass 2^32
;; past depth 32 of a binary tree: deep nodes share keys, so their draws repeat.
(require '[gen.dynamic :refer [gen]] '[gen.gfi :as gfi] '[gen.dist :as d] '[gen.combinators :as comb]
         '[gen.random :as rng] '[gen.array :as array] '[gen.choicemap :as cm])

(println "fold-in(k, 5) = fold-in(k, 5 + 2^32):" (= (rng/fold-in (rng/key 1) 5) (rng/fold-in (rng/key 1) (+ 5 (bit-shift-left 1 32)))))

;; a chain 40 deep in a binary Recurse: node d+1 is child 0 of node d, so node d is 2^d - 1
(def chain (comb/recurse (gen [depth] {:value (trace :v (d/normal 0.0 1.0)) :children (if (< depth 40) [(inc depth)] [])})
                         (gen [v ws] v)
                         2))
(let [tr (gfi/simulate chain [0] (rng/key 2))
      node (fn [depth] (dec (bit-shift-left 1 depth)))
      v (fn [depth] (tr [(node depth) :production :v]))]
  (println "v at depths 29..40:" (mapv #(format "%.4f" (double (v %))) (range 29 41)))
  (println "distinct values among depths 32..40:" (count (distinct (map v (range 32 41)))) "(independent draws: 9)"))

;; a chain that stops with probability 0.05 at each node: depth beyond 32 is likely (0.95^32 = 0.19)
(def stops (comb/recurse (gen [depth] {:value depth :children (if (trace :more (d/bernoulli 0.95)) [(inc depth)] [])})
                         (gen [v ws] (if (seq ws) (first ws) v))
                         2))
(let [outcomes (frequencies (map #(try (let [tr (gfi/simulate stops [0] %)] (if (> (:retval tr) 32) :deeper-than-32 :at-most-32))
                                       (catch ArithmeticException e (str "ArithmeticException: " (ex-message e))))
                                 (rng/split (rng/key 3) 400)))]
  (println "400 simulations of the stopping chain:" outcomes)
  (println "expected: deeper than 32 with probability" (Math/pow 0.95 33)))
