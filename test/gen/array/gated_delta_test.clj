(ns gen.array.gated-delta-test
  "gen.array/gated-delta, the gated delta rule (experimental), on the
   backend under test:
   1. each form (:loop, :chunked, :auto) is the rule as its definition
      reads, computed on the host, token by token, in doubles;
   2. the chunked form is the token loop, for reads shorter than a chunk,
      of whole chunks and between them, and chunks of 16 and 64;
   3. the tape's gradients through both forms match central differences
      (the JVM backend, float64);
   4. a batch tracer's run (vmap) is each row's alone;
   5. on MLX, the fused Metal kernel (:auto, Dk a multiple of 32: mlx-lm's
      gated-delta kernel, gen.backend.mlx.nn) is the composite;
   6. misuse is loud.
   Planted bugs fail them (with-redefs-fn), each by more than 0.01
   (measured on the JVM backend, 2026-11-07): the decay ignored (exp the
   identity: 1; of the chunk's cumulative decays: 2); the chunk's
   triangular inverse dropped, (I + A)⁻¹ as I (0.24), or to first order,
   I - A (0.021) (2); the chunk's attention strict, a token blind to its
   own key (1.4) (2); and the gradient of one entry against another's
   difference (3).
   The bands: on the JVM backend 1e-12 (float64's round-off, measured
   ≤ 2.3e-15); on MLX float32's (below)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.nn]
            [gen.random :as random]
            [gen.test-support :as ts]))

(defn- normal [k shape sd] (array/* sd (random/normal (random/key k) shape)))

(defn- l2 [x] (array// x (array/sqrt (array/sum (array/square x) {:axis -1 :keep-dims? true}))))

(defn- inputs
  "Random operands [q k v g beta state] of B rows, T tokens and H heads:
   unit keys and queries, log decays -softplus(N(0, 1)), beta sigmoid(N(0,
   1)), a state N(0, 0.5²)."
  ([b t h dk dv] (inputs b t h dk dv 0))
  ([b t h dk dv seed]
   [(l2 (normal (+ seed 1) [b t h dk] 1.0)) (l2 (normal (+ seed 2) [b t h dk] 1.0)) (normal (+ seed 3) [b t h dv] 1.0)
    (array/- (array/softplus (normal (+ seed 4) [b t h] 1.0))) (array/sigmoid (normal (+ seed 5) [b t h] 1.0))
    (normal (+ seed 6) [b h dv dk] 0.5)]))

(defn- max-diff ^double [a b] (double (array/item (array/max (array/abs (array/- a b))))))

(defn- gap
  "max |Δ| between [y state] pairs a and b."
  [[ya sa] [yb sb]]
  (max (max-diff ya yb) (max-diff sa sb)))

(def ^:private band
  "max |Δ| between two forms of the rule on the backend under test
   (values about 1): float64's round-off on the JVM backend, float32's
   on MLX (the chunked form's triangular inverse and the token loop sum
   in different orders)."
  (if (ts/jvm?) 1e-12 2e-4))

;; ---------------------------------------------------------------------------
;; 1. The definition, on the host
;; ---------------------------------------------------------------------------

(defn- host-rule
  "The rule as its definition reads, on the host in doubles: [y state']
   of host operands (nested vectors)."
  [q k v g beta state]
  (let [b (count q) t (count (first q)) h (count (first (first q)))]
    (reduce (fn [[ys s] [bi hi]]
              (let [[y-rows s'] (reduce (fn [[ys s] ti]
                                          (let [decay (Math/exp (get-in g [bi ti hi]))
                                                kt (get-in k [bi ti hi]) qt (get-in q [bi ti hi]) vt (get-in v [bi ti hi])
                                                bt (get-in beta [bi ti hi])
                                                s (mapv (fn [row] (mapv #(* decay %) row)) s)
                                                kv (mapv (fn [row] (reduce + (map * row kt))) s)
                                                delta (mapv (fn [vi kvi] (* bt (- vi kvi))) vt kv)
                                                s (mapv (fn [row di] (mapv (fn [x kj] (+ x (* di kj))) row kt)) s delta)]
                                            [(conj ys (mapv (fn [row] (reduce + (map * row qt))) s)) s]))
                                        [[] (get-in state [bi hi])]
                                        (range t))]
                [(assoc-in ys [bi hi] y-rows) (assoc-in s [bi hi] s')]))
            [(vec (repeat b (vec (repeat h nil)))) state]
            (for [bi (range b) hi (range h)] [bi hi]))))

(defn- host-gap
  "max |Δ| between gated-delta's form method and the host's rule."
  [operands method]
  (let [[hy hs] (apply host-rule (map array/->clj operands))
        ;; host y is [B H T Dv]
        [y s] (apply array/gated-delta (conj (vec operands) {:method method :chunk-size 4}))]
    (max (max-diff (array/transpose y [0 2 1 3]) (array/array hy))
         (max-diff s (array/array hs)))))

(deftest every-form-is-the-rule
  (let [operands (inputs 2 7 3 4 3)]
    (doseq [method [:loop :chunked :auto]]
      (is (<= (host-gap operands method) band) (name method))))
  (testing "a planted bug fails it: the decay ignored"
    (let [operands (inputs 2 7 3 4 3)]
      (is (< 0.01 (with-redefs-fn {#'array/exp identity} #(host-gap operands :loop)))))))

;; ---------------------------------------------------------------------------
;; 2. The chunked form, the token loop
;; ---------------------------------------------------------------------------

(defn- chunk-gap
  "max |Δ| between the chunked form (chunks of size) and the loop over
   the lengths ts."
  [ts size]
  (reduce max (for [t ts
                    :let [operands (vec (inputs 2 t 2 8 6 t))
                          run #(apply array/gated-delta (conj operands {:method % :chunk-size size}))]]
                (gap (run :loop) (run :chunked)))))

(deftest the-chunks-are-the-token-loop
  (is (<= (chunk-gap [1 5 15 16 17 40] 16) band) "chunks of 16")
  (is (<= (chunk-gap [63 64 65 130] 64) band) "chunks of 64")
  (testing "planted bugs fail it"
    (doseq [[what redefs] {"the triangular inverse dropped: (I + A)⁻¹ as I"
                           {#'gen.array.nn/unit-lower-inverse (fn [_ a] (array/broadcast-to (array/eye (peek (array/shape a))) (array/shape a)))}
                           "the inverse to first order: I - A"
                           {#'gen.array.nn/unit-lower-inverse (fn [_ a] (array/- (array/eye (peek (array/shape a))) a))}
                           "a token blind to its own key: the chunk's attention strict"
                           {#'gen.array.nn/lower-triangle (let [f @#'gen.array.nn/lower-triangle] (fn [b n _] (f b n true)))}
                           "the chunk's decays ignored"
                           {#'array/exp (let [exp array/exp] (fn [x] (if (= 4 (array/ndim x)) x (exp x))))}}]
      (is (< 0.01 (with-redefs-fn redefs #(chunk-gap [40] 16))) what))))

;; ---------------------------------------------------------------------------
;; 3. Gradients (the JVM backend, float64)
;; ---------------------------------------------------------------------------

(defn- loss
  "Σ y·ry + Σ state'·rs of gated-delta's form method (chunks of 4) on the
   operands, a 0-d array: fixed random weights of every output."
  [method operands]
  (let [[y s] (apply array/gated-delta (conj (vec operands) {:method method :chunk-size 4}))]
    (array/+ (array/sum (array/* y (normal 90 (array/shape y) 1.0)))
             (array/sum (array/* s (normal 91 (array/shape s) 1.0))))))

(defn- central-difference [method operands i entry]
  (let [x (array/->clj (nth operands i))
        at (fn [d] (ts/number (loss method (assoc (vec operands) i (array/array (update-in x entry + d))))))
        h 1e-5]
    (/ (- (at h) (at (- h))) (* 2 h))))

(deftest gradients-are-central-differences
  ;; measured 2026-11-07: relative errors ≤ 2.3e-9 (the loop), 4.1e-10
  ;; (chunks); the band 1e-6
  (if-not (ts/jvm?)
    (println "gen.array.gated-delta-test: the gradients are checked on the JVM backend (float64)")
    (let [operands (vec (inputs 1 9 2 4 3))
          entries {0 [0 3 1 2] 1 [0 8 0 1] 2 [0 5 1 2] 3 [0 4 1] 4 [0 0 0] 5 [0 1 2 3]}]
      (doseq [method [:loop :chunked]]
        (let [grads (apply (array/grad (fn [& xs] (loss method xs)) {:argnums (vec (range 6))}) operands)]
          (doseq [[i entry] entries
                  :let [a (get-in (array/->clj (nth grads i)) entry)
                        b (central-difference method operands i entry)]]
            (is (< (/ (Math/abs (- a b)) (max 1e-3 (Math/abs b))) 1e-6) (str (name method) ": operand " i " at " entry)))))
      (testing "a planted bug fails it: one entry's gradient against another's difference"
        (let [g ((array/grad (fn [k] (loss :chunked (assoc operands 1 k)))) (nth operands 1))
              a (get-in (array/->clj g) [0 8 0 1])
              b (central-difference :chunked operands 1 [0 7 0 1])]
          (is (< 0.01 (/ (Math/abs (- a b)) (max 1e-3 (Math/abs b))))))))))

;; ---------------------------------------------------------------------------
;; 4. Batched by vmap
;; ---------------------------------------------------------------------------

(deftest a-batched-run-is-each-row-alone
  (let [operands (vec (inputs 3 6 2 4 3))
        rows (fn [i] (mapv #(array/slice % 0 i (inc i)) operands))]
    (doseq [method [:loop :chunked]]
      ;; each particle a row of one, [1 T H D]: the operands [3 1 T H D]
      (let [[y s] (apply (array/vmap (fn [& xs] (apply array/gated-delta (conj (vec xs) {:method method :chunk-size 4}))))
                         (mapv #(array/expand-dims % 1) operands))
            [y s] [(array/squeeze y [1]) (array/squeeze s [1])]]
        (doseq [i (range 3)
                :let [[yi si] (apply array/gated-delta (conj (rows i) {:method method :chunk-size 4}))]]
          (is (<= (max (max-diff (array/slice y 0 i (inc i)) (array/expand-dims yi 0))
                       (max-diff (array/slice s 0 i (inc i)) (array/expand-dims si 0)))
                  band)
              (str (name method) ", row " i)))))))

;; ---------------------------------------------------------------------------
;; 5. MLX's Metal kernel
;; ---------------------------------------------------------------------------

(deftest the-fused-kernel-is-the-composite
  (if (ts/jvm?)
    (println "gen.array.gated-delta-test: the fused kernel is checked on MLX")
    (doseq [[t dk dv] [[1 32 32] [7 64 32] [100 128 128]]]
      (let [operands (vec (inputs 2 t 4 dk dv t))
            run #(apply array/gated-delta (conj operands {:method %}))]
        (is (<= (gap (run :auto) (run :loop)) band) (str "T " t ", Dk " dk ", Dv " dv))))))

;; ---------------------------------------------------------------------------
;; 6. Loud errors
;; ---------------------------------------------------------------------------

(deftest misuse-is-loud
  (let [[q k v g beta s] (inputs 1 3 2 4 3)
        reason (fn [f] (:reason (ts/thrown-data (f))))]
    (is (= :invalid-args (reason #(array/gated-delta q k v g beta (array/zeros [1 2 4 3])))) "a state [B H Dk Dv]")
    (is (= :invalid-args (reason #(array/gated-delta q (array/slice k 2 0 1) v g beta s))) "fewer key heads")
    (is (= :invalid-args (reason #(array/gated-delta q k v (array/expand-dims g -1) beta s))))
    (is (= :invalid-args (reason #(array/gated-delta (array/slice q 1 0 0) (array/slice k 1 0 0) (array/slice v 1 0 0)
                                                     (array/slice g 1 0 0) (array/slice beta 1 0 0) s)))
        "no tokens")
    (is (= :invalid-args (reason #(array/gated-delta q k v g beta s {:method :scan}))))
    (is (= :invalid-args (reason #(array/gated-delta q k v g beta s {:chunk-size 48}))) "not a power of 2")
    (is (= :unknown-options (reason #(array/gated-delta q k v g beta s {:chunk 64}))))))
