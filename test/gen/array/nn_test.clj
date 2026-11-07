(ns gen.array.nn-test
  "gen.array's neural-network operations (experimental): rms-norm, rope
   and sdpa, and the quantized weights' quantize, dequantize,
   quantized-matmul and gather-qmm, on the backend under test:
   - the composites against direct computations on the host, in float64;
   - their gradients (the tape's, or MLX's native ones) against central
     finite differences in float64 (on the JVM backend);
   - a backend's fused kernels (MLX's fast ops) against the composites,
     values and native gradients;
   - dequantize against the packed words unpacked on the host; quantize's
     error bound; the JVM backend's quantization against MLX's;
     quantized-matmul and gather-qmm against dequantize and matmul, within
     float32 round-off's bound;
   - the batch tracer (vmap: per row), staging (a replay is the direct
     run), and the loud errors.
   Planted mutations, each failing a test here before it was trusted
   (2026-11-07): NEOX's rotation's sign, the traditional pairs' order,
   the per-row offsets ignored, rms-norm's eps outside the root, GQA's
   heads tiled instead of grouped, the causal mask off by one, the no-key
   rows' zeroing dropped, the bit stream's word boundary, quantize's
   rounding (floor for rint), the tape's quantized product not
   transposed, gather-qmm's VJP scattered by rhs, and the tracer's
   per-particle offsets dropped."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.nn :as nn]
            [gen.array.protocols :as p]
            [gen.array.stage :as st]
            [gen.random :as rng]
            [gen.test-support :as ts]))

;; ---------------------------------------------------------------------------
;; Host data and references
;; ---------------------------------------------------------------------------

(defn- host [x] (array/->clj x))

(defn- normal [seed shape] (rng/normal (rng/key seed) shape))

(defn- rows
  "f applied to every row (last axis) of the nested vectors x, with the
   row's index path."
  ([f x] (rows f x []))
  ([f x path]
   (if (number? (first x))
     (f path x)
     (vec (map-indexed (fn [i r] (rows f r (conj path i))) x)))))

(defn- host-rms-norm [x w eps]
  (rows (fn [_ r]
          (let [ms (/ (reduce + (map #(* % %) r)) (count r))
                inv (/ 1.0 (Math/sqrt (+ ms eps)))]
            (mapv (fn [v i] (* v inv (if w (nth w i) 1.0))) r (range))))
        x))

(defn- host-rope
  "RoPE on the host: position (offset + t)·scale, t the row's index along
   axis -2, offset the row's own (by its axis-0 index) when offsets is a
   vector."
  [x dims {:keys [traditional? base scale offset freqs] :or {base 10000.0 scale 1.0 offset 0}}]
  (let [half (quot dims 2)
        inv (if freqs (mapv #(/ 1.0 %) freqs) (mapv #(Math/pow base (/ (* -2.0 %) dims)) (range half)))]
    (rows (fn [path r]
            (let [t (peek path)
                  off (if (vector? offset) (nth offset (first path)) offset)
                  pos (* (+ off t) scale)
                  pair (fn [i] (if traditional? [(* 2 i) (inc (* 2 i))] [i (+ i half)]))]
              (reduce (fn [out i]
                        (let [[a b] (pair i)
                              th (* pos (nth inv i))
                              c (Math/cos th) s (Math/sin th)
                              x1 (nth r a) x2 (nth r b)]
                          (assoc out a (- (* x1 c) (* x2 s)) b (+ (* x1 s) (* x2 c)))))
                      (into [] r) (range half))))
          x)))

(defn- mask-at
  "The mask's entry for batch b, head h, query i, key j (broadcast), or
   nil for none."
  [m b h i j]
  (when m
    (let [shape (fn shape [v] (if (vector? v) (into [(count v)] (shape (first v))) []))
          s (shape m)
          idx (take-last (count s) [b h i j])]
      (get-in m (mapv (fn [d k] (if (= 1 d) 0 k)) s idx)))))

(defn- host-sdpa
  "Attention on the host: GQA by grouping, a bool mask hiding keys (a
   query that sees none gets zeros), an additive one added, :causal the
   lower-right triangle."
  [q k v {:keys [scale mask]}]
  (let [hq (count (first q)) hk (count (first k))
        rep (quot hq hk)
        d (count (first (first (first q))))
        scale (or scale (/ 1.0 (Math/sqrt d)))]
    (vec (for [b (range (count q))]
           (vec (for [h (range hq)]
                  (let [kh (quot h rep)
                        ks (get-in k [b kh]) vs (get-in v [b kh])
                        tq (count (get-in q [b h])) tk (count ks)]
                    (vec (for [i (range tq)]
                           (let [qi (get-in q [b h i])
                                 entry (fn [j] (cond (nil? mask) true
                                                     (= :causal mask) (<= j (+ i (- tk tq)))
                                                     :else (mask-at mask b h i j)))
                                 scores (for [j (range tk)]
                                          (let [s (* scale (reduce + (map * qi (nth ks j))))
                                                e (entry j)]
                                            (cond (true? e) s (false? e) nil :else (+ s e))))
                                 seen (remove nil? scores)]
                             (if (empty? seen)
                               (vec (repeat (count (first vs)) 0.0))
                               (let [mx (reduce max seen)
                                     ws (map #(if % (Math/exp (- % mx)) 0.0) scores)
                                     z (reduce + ws)]
                                 (apply mapv (fn [& col] (/ (reduce + (map * ws col)) z)) vs)))))))))))))

(defn- close-host?
  "Do the nested host values a and b agree within the backend's float
   precision (float32: rtol and atol of rel; float64: 1e-11)?"
  ([a b] (close-host? a b 2e-5))
  ([a b rel]
   (let [fa (flatten [a]) fb (flatten [b])
         tol (if (ts/f32?) rel 1e-11)]
     (and (= (count fa) (count fb))
          (every? true? (map (fn [x y] (<= (Math/abs (- (double x) (double y))) (* tol (+ 1.0 (Math/abs (double y))))))
                             fa fb))))))

;; ---------------------------------------------------------------------------
;; The composites against the host
;; ---------------------------------------------------------------------------

(deftest rms-norm-normalizes-the-last-axis
  (let [x (normal 1 [2 3 8])
        w (array/+ 1.0 (array/* 0.1 (normal 2 [8])))]
    (is (close-host? (host (array/rms-norm x w 1e-5)) (host-rms-norm (host x) (host w) 1e-5)))
    (is (close-host? (host (array/rms-norm x nil 1e-6)) (host-rms-norm (host x) nil 1e-6)) "without a weight")
    (is (close-host? (host (array/rms-norm (array/* 1e-4 x) w 1e-2)) (host-rms-norm (host (array/* 1e-4 x)) (host w) 1e-2))
        "eps inside the root, where it dominates")))

(deftest rope-rotates-pairs-by-position
  (let [x (normal 3 [2 3 4 8])
        hx (host x)]
    (doseq [[label dims opts] [["NEOX" 8 {}]
                               ["traditional" 8 {:traditional? true}]
                               ["partial NEOX" 4 {}]
                               ["partial traditional" 6 {:traditional? true}]
                               ["an offset, a base and a scale" 8 {:offset 5 :base 500.0 :scale 0.5}]
                               ["frequencies" 4 {:freqs [1.0 3.5]}]]]
      (testing label
        (let [opts' (cond-> opts (:freqs opts) (update :freqs array/array))]
          (is (close-host? (host (array/rope x dims opts')) (host-rope hx dims opts) 1e-4)))))
    (testing "an offset per row of axis 0"
      (let [offsets [3 11]]
        (doseq [traditional? [false true]]
          (is (close-host? (host (array/rope x 8 {:offset (array/array offsets {:dtype (ts/int-dtype)}) :traditional? traditional?}))
                           (host-rope hx 8 {:offset offsets :traditional? traditional?}) 1e-4)))))
    (testing "x of 2 and 3 axes"
      (is (close-host? (host (array/rope (normal 4 [5 6]) 6 {:offset 2})) (host-rope (host (normal 4 [5 6])) 6 {:offset 2}) 1e-4))
      (is (close-host? (host (array/rope (normal 5 [2 5 6]) 4 {:offset (array/array [1 7] {:dtype (ts/int-dtype)})}))
                       (host-rope (host (normal 5 [2 5 6])) 4 {:offset [1 7]}) 1e-4)))))

(deftest a-rotation-keeps-norms-and-relative-positions
  ;; RoPE's defining properties: each pair keeps its norm, and q·k depends
  ;; on positions only through their difference
  (let [q (normal 6 [1 1 1 8]) k (normal 7 [1 1 1 8])
        dot (fn [i j] (array/item (array/sum (array/* (array/rope q 8 {:offset i}) (array/rope k 8 {:offset j})))))]
    (is (ts/close? (array/sum (array/square (array/rope q 8 {:offset 9}))) (array/sum (array/square q))))
    (is (< (Math/abs (- (dot 3 1) (dot 12 10))) (if (ts/f32?) 1e-4 1e-11)))
    (is (> (Math/abs (- (dot 3 1) (dot 3 2))) 1e-3) "another difference, another product")))

(deftest sdpa-attends-with-groups-and-masks
  (let [q (normal 8 [2 4 3 8]) k (normal 9 [2 2 5 8]) v (normal 10 [2 2 5 6])
        [hq hk hv] (map host [q k v])
        bool-mask [[true false true true false] [false false false false false] [true true true true true]]
        add-mask (mapv (fn [r] (mapv #(* 0.3 (Math/sin (+ % (* 5 r)))) (range 5))) (range 3))]
    (doseq [[label opts host-opts]
            [["no mask, GQA of 2" {} {}]
             ["a scale" {:scale 0.7} {:scale 0.7}]
             ["causal" {:mask :causal} {:mask :causal}]
             ["a bool mask [Tq Tk], a query seeing no key" {:mask (array/array bool-mask)} {:mask bool-mask}]
             ["an additive mask [Tq Tk]" {:mask (array/array add-mask)} {:mask add-mask}]
             ["a bool mask per batch and head [B Hq Tq Tk]"
              {:mask (array/> (normal 11 [2 4 3 5]) -0.3)} {:mask (host (array/> (normal 11 [2 4 3 5]) -0.3))}]
             ["a bool mask per batch [B 1 Tq Tk]"
              {:mask (array/> (normal 12 [2 1 3 5]) 0.0)} {:mask (host (array/> (normal 12 [2 1 3 5]) 0.0))}]]]
      (testing label
        (is (close-host? (host (array/sdpa q k v opts)) (host-sdpa hq hk hv host-opts) 1e-4))))
    (testing "more queries than keys under :causal: the first see none"
      (let [q (normal 13 [1 2 4 8]) k (normal 14 [1 2 2 8]) v (normal 15 [1 2 2 8])]
        (is (close-host? (host (array/sdpa q k v {:mask :causal})) (host-sdpa (host q) (host k) (host v) {:mask :causal}) 1e-4))))
    (testing "one head of each"
      (let [q (normal 16 [1 1 2 4]) k (normal 17 [1 1 3 4]) v (normal 18 [1 1 3 4])]
        (is (close-host? (host (array/sdpa q k v)) (host-sdpa (host q) (host k) (host v) {}) 1e-4))))))

;; ---------------------------------------------------------------------------
;; Gradients against finite differences (float64, the JVM backend)
;; ---------------------------------------------------------------------------

(def ^:private reference (array/backend-named :jvm))

(defn- on-reference [x]
  (if (array/host? x) x (array/with-backend reference (array/array (array/->clj x) {:dtype (if (= :float (array/dtype-kind (array/dtype x))) :float64 (array/dtype x))}))))

(defn- fd-grad
  "The central-difference gradient of the scalar f at x, step h, in
   float64 on the JVM backend."
  [f x h]
  (array/with-backend reference
    (let [xs (vec (flatten [(array/->clj x)]))
          s (array/shape x)
          at (fn [i d] (double (array/item (f (array/reshape (array/array (update xs i + d)) s)))))]
      (array/reshape (array/array (vec (for [i (range (count xs))] (/ (- (at i h) (at i (- h))) (* 2 h))))) s))))

(defn- projected
  "f's output weighted by fixed weights: a scalar whose gradient covers the
   whole VJP."
  [f]
  (fn [& xs]
    (let [out (apply f xs)
          n (array/size out)
          w (array/reshape (array/with-backend (array/backend out)
                             (array/array (mapv #(+ 0.5 (* 0.37 (Math/sin (* 1.7 (inc %))))) (range n))))
                           (array/shape out))]
      (array/sum (array/* out w)))))

(defn- grads-agree?
  "Do the gradients g agree with the finite differences fds?"
  [g fds]
  (every? true?
          (for [i (range (count fds))]
            (let [fd (nth fds i)
                  gi (on-reference (nth g i))]
              (or (array/allclose? gi fd (if (ts/f32?) {:rtol 2e-3 :atol 2e-4} {:rtol 1e-6 :atol 1e-7}))
                  (do (println "gradient" i "differs:" (array/->clj gi) (array/->clj fd)) false))))))

(defn- grads-match?
  "Does f's gradient (the backend's: the tape, or MLX's native one) with
   respect to each of xs match central differences of f on the JVM
   backend? f takes xs, then consts (arrays it is not differentiated
   with respect to, given to both)."
  ([f xs] (grads-match? f xs []))
  ([f xs consts]
   (let [xs (vec xs)
         n (count xs)
         g (apply (array/grad (projected f) {:argnums (vec (range n))}) (concat xs consts))
         ref (mapv on-reference (concat xs consts))]
     (grads-agree? g (mapv #(fd-grad (fn [x] (apply (projected f) (assoc ref % x))) (nth ref %) 1e-6) (range n))))))

(deftest the-composites-gradients-match-finite-differences
  (let [x (normal 20 [2 3 8]) w (array/+ 1.0 (array/* 0.1 (normal 21 [8])))]
    (is (grads-match? #(array/rms-norm %1 %2 1e-5) [x w]) "rms-norm"))
  (let [x (normal 22 [2 2 3 8])]
    (is (grads-match? #(array/rope % 8 {:offset 3}) [x]) "rope")
    (is (grads-match? #(array/rope % 6 {:traditional? true :offset (array/array [2 9] {:dtype (ts/int-dtype)})}) [x])
        "traditional, partial, per row")
    (is (grads-match? #(array/rope %2 4 {:freqs %1}) [(array/array [1.5 4.0])] [x]) "with respect to the frequencies"))
  (let [q (normal 23 [1 4 3 8]) k (normal 24 [1 2 4 8]) v (normal 25 [1 2 4 8])]
    (is (grads-match? #(array/sdpa %1 %2 %3 {:mask :causal}) [q k v]) "sdpa, causal, GQA")
    (is (grads-match? #(array/sdpa %1 %2 %3 {:mask (array/array [[true false true false] [false false false false] [true true false true]])})
                      [q k v])
        "a bool mask, a query seeing no key")
    (is (grads-match? #(array/sdpa %2 %3 %4 {:mask %1}) [(array/* 0.5 (normal 26 [1 1 3 4]))] [q k v])
        "with respect to an additive mask")))

;; ---------------------------------------------------------------------------
;; A backend's fused kernels against the composites
;; ---------------------------------------------------------------------------

(def ^:private composites
  {:rms-norm #'nn/rms-norm-composite :rope #'nn/rope-composite :sdpa #'nn/sdpa-composite})

(defn- fused-and-composite
  "[the fused kernel's value, the composite's] of op on args (gen.array's
   checked, cast arguments)."
  [op args]
  [(p/-fused array/*backend* op args) (apply (composites op) args)])

(deftest fused-kernels-are-the-composites
  (if-not (instance? gen.array.protocols.IFused array/*backend*)
    (println "gen.array.nn-test: no fused kernels on the" (ts/backend-name) "backend")
    (let [f64? (= :float64 (ts/float-dtype))
          x (normal 30 [2 4 5 64])
          w (array/+ 1.0 (array/* 0.1 (normal 31 [64])))
          k (normal 32 [2 2 7 64]) v (normal 33 [2 2 7 64])
          x12 (normal 37 [2 4 12 64]) k16 (normal 38 [2 2 16 64]) v16 (normal 39 [2 2 16 64])
          rope-opts (fn [o] (merge {:traditional? false :base 10000.0 :scale 1.0 :offset 0 :freqs nil} o))
          cases [[:rms-norm [x w 1e-5]]
                 [:rms-norm [x nil 1e-6]]
                 [:rope [x 64 (rope-opts {})]]
                 [:rope [x 32 (rope-opts {:traditional? true :offset 7 :base 1e6})]]
                 [:rope [x 16 (rope-opts {:offset (array/array [3 40] {:dtype :int32}) :scale 0.25})]]
                 [:rope [x 32 (rope-opts {:base nil :freqs (array/+ 1.0 (array/arange 16))})]]
                 [:sdpa [x k v {:scale 0.125 :mask nil}]]
                 [:sdpa [x k v {:scale 0.125 :mask :causal}]]
                 [:sdpa [(array/slice x 2 0 1) k v {:scale 0.125 :mask :causal}]]
                 [:sdpa [x k v {:scale 0.125 :mask (array/> (normal 34 [2 1 5 7]) -0.5)}]]
                 [:sdpa [x k v {:scale 0.125 :mask (array/array (vec (for [i (range 5)] (vec (for [j (range 7)] (and (pos? i) (<= j (+ i 2))))))))}]]
                 [:sdpa [x k v {:scale 0.125 :mask (array/* 0.5 (normal 35 [5 7]))}]]
                 ;; more than 8 queries: Metal's other kernel
                 [:sdpa [x12 k16 v16 {:scale 0.125 :mask :causal}]]
                 [:sdpa [x12 k16 v16 {:scale 0.125 :mask (array/> (normal 36 [2 4 12 16]) -0.5)}]]]]
      (if f64?
        (testing "float64 is declined: the composite is exact there"
          (is (every? nil? (map (fn [[op args]] (p/-fused array/*backend* op args)) cases))))
        (doseq [[op args] cases]
          (let [[fused composite] (fused-and-composite op args)]
            (is (some? fused) (str op " is fused"))
            (is (close-host? (host fused) (host composite) 1e-4) (str op " " (pr-str (last args))))))))))

(deftest native-gradients-of-fused-kernels-are-the-composites
  ;; MLX's own gradients (-Dgen.mlx.grad=native) differentiate the fused
  ;; kernels by MLX's VJPs: they must be the composites' gradients
  (if-not (ts/native-grad?)
    (println "gen.array.nn-test: native gradients only with -Dgen.mlx.grad=native")
    (let [x (normal 40 [1 4 3 64]) w (array/+ 1.0 (array/* 0.1 (normal 41 [64])))
          k (normal 42 [1 2 6 64]) v (normal 43 [1 2 6 64])
          rope-opts {:traditional? false :base 10000.0 :scale 1.0 :offset 2 :freqs nil}
          check (fn [label fused-f composite-f & xs]
                  (let [argnums (vec (range (count xs)))
                        gf (apply (array/grad (projected fused-f) {:argnums argnums}) xs)
                        gc (apply (array/grad (projected composite-f) {:argnums argnums}) xs)]
                    (is (every? true? (map #(close-host? (host %1) (host %2) 1e-4) gf gc)) label)))]
      (check "rms-norm" #(array/rms-norm %1 %2 1e-5) #(#'nn/rms-norm-composite %1 %2 1e-5) x w)
      (check "rope" #(array/rope % 64 rope-opts) #(#'nn/rope-composite % 64 rope-opts) x)
      (check "rope, per row" #(array/rope % 32 (assoc rope-opts :offset (array/array [5] {:dtype :int32})))
             #(#'nn/rope-composite % 32 (assoc rope-opts :offset (array/array [5] {:dtype :int32}))) x)
      (check "sdpa, causal" #(array/sdpa %1 %2 %3 {:mask :causal}) #(#'nn/sdpa-composite %1 %2 %3 {:scale 0.125 :mask :causal}) x k v)
      (let [m (array/> (normal 44 [1 1 3 6]) -0.5)]
        (check "sdpa, a bool mask" #(array/sdpa %1 %2 %3 {:mask m}) #(#'nn/sdpa-composite %1 %2 %3 {:scale 0.125 :mask m}) x k v)))))

;; ---------------------------------------------------------------------------
;; The batch tracer and staging
;; ---------------------------------------------------------------------------

(deftest batched-runs-are-per-particle
  (let [xs (normal 50 [3 2 2 4 8])
        per-row (fn [f xs] (array/stack (mapv #(f (array/take xs % 0)) (range (first (array/shape xs)))) 0))
        k (normal 51 [2 1 5 8]) v (normal 52 [2 1 5 8])]
    (doseq [[label f] [["rms-norm" #(array/rms-norm % (array/arange 8) 1e-5)]
                       ["rope" #(array/rope % 6 {:offset (array/array [1 4] {:dtype (ts/int-dtype)})})]
                       ["sdpa" #(array/sdpa % k v {:mask :causal})]]]
      (testing label
        (is (ts/close? ((array/vmap f) xs) (per-row f xs)))))
    (testing "a batched mask and per-row offsets"
      (let [ms (array/> (normal 53 [3 4 5]) 0.0)
            offs (array/array [[0 3] [7 1] [2 2]] {:dtype (ts/int-dtype)})]
        (is (ts/close? ((array/vmap (fn [x m o] (array/rope (array/sdpa x k v {:mask m}) 8 {:offset o}))) xs ms offs)
                       (array/stack (mapv (fn [i] (array/rope (array/sdpa (array/take xs i 0) k v {:mask (array/take ms i 0)}) 8
                                                              {:offset (array/take offs i 0)}))
                                          (range 3)))))))))

(deftest staged-composites-replay
  (if-not (ts/jvm?)
    (println "gen.array.nn-test: replays are compared on the JVM backend")
    (binding [st/*float* :float64]
      (let [f (fn [x k v w] (array/sdpa (array/rope (array/rms-norm x w 1e-5) 8 {:offset 2}) k v {:mask :causal}))
            inputs [(normal 60 [1 2 3 8]) (normal 61 [1 1 4 8]) (normal 62 [1 1 4 8]) (normal 63 [8])]
            program (st/stage f [[1 2 3 8] :float64] [[1 1 4 8] :float64] [[1 1 4 8] :float64] [[8] :float64])]
        (is (= (host (st/evaluate program inputs)) (host (apply f inputs))) "bit for bit")))))

;; ---------------------------------------------------------------------------
;; Quantized weights
;; ---------------------------------------------------------------------------

(defn- random-words
  "n pseudo-random uint32 words (every bit pattern), host longs."
  [seed n]
  (let [r (java.util.Random. seed)]
    (vec (repeatedly n #(bit-and (.nextLong r) 0xFFFFFFFF)))))

(defn- unpack
  "The k codes of bits bits of one row's words (host longs): the
   little-endian bit stream."
  [words k bits]
  (let [stream (biginteger (reduce (fn [acc [i w]] (.or ^java.math.BigInteger acc (.shiftLeft (biginteger w) (* 32 i))))
                                   java.math.BigInteger/ZERO (map-indexed vector words)))]
    (mapv #(.longValue (.and (.shiftRight ^java.math.BigInteger stream (* % bits)) (biginteger (dec (bit-shift-left 1 bits)))))
          (range k))))

(defn- host-dequantize
  "The rows of words (each kw words), scales and biases (host values) as
   values: scale·q + bias, with each op rounded to float32 (two-ops?) or
   only the result (fma), or in float64."
  [words scales biases {:keys [k g bits rounding]}]
  (mapv (fn [ws ss bs]
          (vec (map-indexed (fn [j q]
                              (let [s (nth ss (quot j g)) b (nth bs (quot j g))]
                                (case rounding
                                  :two-ops (double (unchecked-float (+ (double (unchecked-float (* q s))) b)))
                                  :fma (double (unchecked-float (+ (* q s) b)))
                                  :float64 (+ (* q s) b))))
                            (unpack ws k bits))))
        words scales biases))

(def ^:private float-dt (ts/float-dtype))

(defn- device []
  (when-not (ts/jvm?) ((requiring-resolve 'gen.backend.mlx/device) array/*backend*)))

(deftest dequantize-unpacks-every-bit-pattern
  ;; random words cover every bit pattern, the words' boundaries crossed by
  ;; 3, 5 and 6 bits included; float32 is MLX's CPU (two ops: the product,
  ;; then the sum, each rounded), Metal's fma (one rounding) allowed on
  ;; MLX's GPU
  (doseq [bits [2 3 4 5 6 8] g [32 64 128]]
    (let [rows 3 k 256 kw (quot (* k bits) 32)
          words (vec (for [r (range rows)] (random-words (+ (* 100 bits) g r) kw)))
          scales (vec (for [r (range rows)] (vec (for [i (range (quot k g))] (double (unchecked-float (* 0.01 (- (* 7 (+ r i)) 9))))))))
          biases (vec (for [r (range rows)] (vec (for [i (range (quot k g))] (double (unchecked-float (Math/sin (+ r i 0.3))))))))
          d (array/dequantize (array/array words {:dtype :uint32}) (array/array scales) (array/array biases) {:group-size g :bits bits})
          got (host d)
          layout {:k k :g g :bits bits}]
      (is (= [rows k] (array/shape d)))
      (is (= float-dt (array/dtype d)))
      (if (= float-dt :float64)
        (is (= got (host-dequantize words scales biases (assoc layout :rounding :float64))) (str bits " bits, groups of " g))
        (let [two (host-dequantize words scales biases (assoc layout :rounding :two-ops))]
          (if (= :gpu (device))
            (let [one (host-dequantize words scales biases (assoc layout :rounding :fma))]
              (is (every? true? (map #(or (= %1 %2) (= %1 %3)) (flatten got) (flatten two) (flatten one)))
                  (str bits " bits, groups of " g ": each value one of the two roundings")))
            (is (= got two) (str bits " bits, groups of " g))))))))

(deftest quantize-is-within-a-step
  ;; MLX's affine algorithm: the extremum of the larger magnitude exact,
  ;; every value rounded to its nearest code (within half a step, the
  ;; scale) but those clipped at the far end (where the edge's exact
  ;; code shortens the step: within one step)
  (doseq [bits [2 3 4 5 6 8] g [32 64 128]]
    (let [w (array/* 0.8 (normal (+ bits g) [4 256]))
          [wq s b] (array/quantize w {:group-size g :bits bits})
          d (array/dequantize wq s b {:group-size g :bits bits})
          hw (host w) hd (host d) hs (host s) hb (host b)
          top (dec (bit-shift-left 1 bits))]
      (is (every? true? (for [r (range 4) j (range 256)
                              :let [sc (get-in hs [r (quot j g)])
                                    q (Math/round (double (/ (- (get-in hd [r j]) (get-in hb [r (quot j g)])) sc)))]
                              :when (< 0 q top)]
                          (<= (Math/abs (- (get-in hd [r j]) (get-in hw [r j]))) (+ (* 0.5001 (Math/abs sc)) 1e-6))))
          (str bits " bits, groups of " g ": the nearest code"))
      (is (= [[4 (quot (* 256 bits) 32)] :uint32] [(array/shape wq) (array/dtype wq)]))
      (is (= [[4 (quot 256 g)] (array/dtype w)] [(array/shape s) (array/dtype s)]))
      (is (every? true? (for [r (range 4) j (range 256)]
                          (<= (Math/abs (- (get-in hd [r j]) (get-in hw [r j])))
                              (+ (* 1.0001 (Math/abs (get-in hs [r (quot j g)]))) 1e-6))))
          (str bits " bits, groups of " g))
      (is (every? true? (for [r (range 4) gi (range (quot 256 g))]
                          (let [grp (subvec (get-in hw [r]) (* gi g) (* (inc gi) g))
                                edge (apply max-key #(Math/abs (double %)) grp)
                                at (.indexOf ^java.util.List grp edge)]
                            (<= (Math/abs (- (get-in hd [r (+ (* gi g) at)]) edge)) (* 1e-6 (Math/abs edge))))))
          "the edge is exact"))))

(deftest quantize-is-mlx-s
  ;; the JVM backend's quantize is MLX's algorithm in float32: on MLX, both
  ;; give the same words, scales and biases, and MLX's words dequantized by
  ;; the JVM backend are MLX's dequantized (on the CPU exactly; Metal's fma
  ;; within an ulp)
  (if (ts/jvm?)
    (println "gen.array.nn-test: MLX's quantization is compared on MLX")
    (doseq [bits [2 3 4 5 6 8] g [32 64 128]]
      (let [w (array/* 0.7 (normal (+ 7 bits g) [8 256]))
            [wq s b] (array/quantize w {:group-size g :bits bits})
            jvm (fn [x] (array/with-backend reference (array/array (array/->clj x) {:dtype (array/dtype x)})))
            [jq js jb] (array/quantize (jvm w) {:group-size g :bits bits})
            md (array/dequantize wq s b {:group-size g :bits bits})
            jd (array/dequantize (jvm wq) (jvm s) (jvm b) {:group-size g :bits bits})]
        (is (= (host wq) (host jq)) (str bits " bits, groups of " g ": the words"))
        (is (= (host s) (host js)) "the scales")
        (is (= (host b) (host jb)) "the biases")
        (if (= :gpu (device))
          ;; an ulp of the terms: scale·q + bias may cancel to a value far
          ;; smaller than they are
          (is (array/allclose? (jvm md) jd {:rtol 2.4e-7 :atol 5e-7}) "within an ulp")
          (is (= (host md) (host jd)) "dequantized exactly"))))))

(defn- quantized
  "[wq scales biases] of a random matrix [rows cols] (or a stack), and its
   options."
  [seed shape opts]
  (conj (array/quantize (array/* 0.5 (normal seed shape)) opts) opts))

(defn- product-bound
  "float32's round-off bound of a product summed over k terms: 2·k·u·Σ|terms|
   (u = 2^-24), from x [.. k] and w [k n]; float64's, the same with 2^-53."
  [x w]
  (let [k (last (array/shape x))
        u (if (ts/f32?) (Math/pow 2 -24) (Math/pow 2 -53))]
    (array/+ (array/* (* 2.0 k u) (array/matmul (array/abs x) (array/abs w))) 1e-30)))

(defn- within? [a b bound]
  (array/all-true? (array/<= (array/abs (array/- a b)) bound)))

(deftest quantized-matmul-is-dequantize-then-matmul
  (doseq [bits [2 3 4 6 8] [g k] [[32 64] [64 128] [128 256]]]
    (let [[wq s b opts] (quantized (+ bits g) [12 k] {:group-size g :bits bits})
          w (array/dequantize wq s b opts)
          x (normal (+ 1 bits g) [2 3 k])
          y (array/quantized-matmul x wq s b opts)
          wt (array/transpose w)]
      (is (= [2 3 12] (array/shape y)))
      (is (within? y (array/matmul x wt) (product-bound x wt)) (str bits " bits, groups of " g))
      (testing "not transposed: x·w"
        (let [x2 (normal (+ 2 bits g) [5 12])]
          (is (within? (array/quantized-matmul x2 wq s b (assoc opts :transpose? false)) (array/matmul x2 w)
                       (product-bound x2 w)))))
      (testing "a vector"
        (let [xv (normal (+ 3 bits g) [k])]
          (is (within? (array/quantized-matmul xv wq s b opts) (array/matmul xv wt) (product-bound xv wt))))))))

(deftest gather-qmm-is-a-gathered-product
  (let [opts {:group-size 32 :bits 4}
        [wq s b] (quantized 70 [4 6 64] opts)
        w (array/dequantize wq s b opts)
        x (normal 71 [5 1 2 64])
        rhs (array/array [[0 2] [3 3] [1 0] [2 1] [3 0]] {:dtype (ts/int-dtype)})
        y (array/gather-qmm x wq s b (assoc opts :rhs-indices rhs))
        expect (array/matmul x (array/transpose (array/take w rhs 0) [0 1 3 2]))]
    (is (= [5 2 2 6] (array/shape y)) "x's [5 1] broadcast with the indices' [5 2]")
    (is (within? y expect (product-bound (array/broadcast-to x [5 2 2 64]) (array/transpose (array/take w rhs 0) [0 1 3 2]))))
    (testing "left indices, repeated and negative"
      (let [xs (normal 72 [3 2 64])
            lhs (array/array [2 0 -1 1] {:dtype (ts/int-dtype)})
            rhs (array/array [1 1 0 3] {:dtype (ts/int-dtype)})
            y (array/gather-qmm xs wq s b (assoc opts :lhs-indices lhs :rhs-indices rhs))
            xl (array/take xs (array/array [2 0 2 1] {:dtype (ts/int-dtype)}) 0)
            wr (array/transpose (array/take w rhs 0) [0 2 1])]
        (is (within? y (array/matmul xl wr) (product-bound xl wr)))))
    (testing "not transposed, the default indices"
      (let [xs (normal 73 [4 3 6])
            y (array/gather-qmm xs wq s b (assoc opts :transpose? false))]
        (is (within? y (array/matmul xs w) (product-bound xs w)))))))

(deftest quantized-gradients-are-x-s
  (let [opts {:group-size 32 :bits 4}
        [wq s b] (quantized 80 [6 64] opts)
        x (normal 81 [2 64])]
    (is (grads-match? #(array/quantized-matmul %1 %2 %3 %4 opts) [x] [wq s b]) "x·wᵀ")
    (is (grads-match? #(array/quantized-matmul %1 %2 %3 %4 (assoc opts :transpose? false)) [(normal 82 [3 6])] [wq s b]) "x·w"))
  (let [opts {:group-size 32 :bits 3}
        [wq s b] (quantized 83 [3 4 64] opts)
        x (normal 84 [2 2 64])]
    (is (grads-match? #(array/gather-qmm %1 %2 %3 %4 (assoc opts :lhs-indices (array/array [1 0 1 1] {:dtype (ts/int-dtype)})
                                                             :rhs-indices (array/array [2 0 0 1] {:dtype (ts/int-dtype)})))
                      [x] [wq s b])
        "gather-qmm, x's matrices repeated")
    (is (grads-match? #(array/gather-qmm %1 %2 %3 %4 (assoc opts :rhs-indices (array/array [[0 2] [1 1]] {:dtype (ts/int-dtype)})))
                      [(array/reshape x [2 1 2 64])] [wq s b])
        "gather-qmm, x broadcast against the indices")))

(deftest quantized-ops-batched-are-per-particle
  (let [opts {:group-size 32 :bits 4}
        [wq s b] (quantized 90 [3 6 64] opts)
        [wq2 s2 b2] (quantized 91 [6 64] opts)
        xs (normal 92 [4 2 3 64])
        rhs (array/array [[0 2] [1 1] [2 0] [0 0]] {:dtype (ts/int-dtype)})
        lhs (array/array [[1 0] [0 0] [1 1] [0 1]] {:dtype (ts/int-dtype)})
        per-row (fn [f & args] (array/stack (mapv (fn [i] (apply f (map #(array/take % i 0) args))) (range 4)) 0))]
    (testing "quantized-matmul of batched x"
      (let [f #(array/quantized-matmul % wq2 s2 b2 opts)]
        (is (ts/close? ((array/vmap f) xs) (per-row f xs)))))
    (testing "gather-qmm of batched x and per-particle experts"
      (let [f (fn [x l r] (array/gather-qmm x wq s b (assoc opts :lhs-indices l :rhs-indices r)))]
        (is (ts/close? ((array/vmap f) xs lhs rhs) (per-row f xs lhs rhs)))))
    (testing "shared x, per-particle experts"
      (let [x (array/take xs 0 0)
            f (fn [r] (array/gather-qmm x wq s b (assoc opts :rhs-indices r)))]
        (is (ts/close? ((array/vmap f) rhs) (per-row f rhs)))))
    (testing "quantize and dequantize of batched matrices"
      (let [ws (array/* 0.5 (normal 93 [4 6 64]))
            f (fn [w] (let [[q s b] (array/quantize w opts)] (array/dequantize q s b opts)))]
        (is (ts/close? ((array/vmap f) ws) (per-row f ws)))))
    (testing "gradients per particle"
      (let [g (array/grad (fn [x] (array/sum (array/gather-qmm x wq s b (assoc opts :rhs-indices (array/array [2 1] {:dtype (ts/int-dtype)}))))))]
        (is (ts/close? ((array/vmap g) xs) (per-row g xs)))))
    (testing "weights that differ across particles: the batched-branch error"
      (let [e (ts/thrown-data ((array/vmap (fn [s] (array/quantized-matmul xs wq2 s b2 opts))) (array/stack [s2 s2] 0)))]
        (is (:gen.array/batched-branch e))))))

(deftest quantized-errors-are-loud
  (let [opts {:group-size 32 :bits 4}
        [wq s b] (quantized 100 [6 64] opts)
        x (normal 101 [2 64])
        reason (fn [f] (:reason (ts/thrown-data (f))))]
    (testing "options"
      (is (= :invalid-args (reason #(array/quantize x {:group-size 16}))))
      (is (= :invalid-args (reason #(array/quantize x {:bits 7}))))
      (is (= :unsupported (reason #(array/quantize x {:mode :mxfp4}))) "uint8 scales")
      (is (= :invalid-args (reason #(array/quantize (array/reshape x [128]) opts))) "a vector")
      (is (= :invalid-args (reason #(array/quantize (normal 102 [2 48]) opts))) "not a multiple of the group")
      (is (= :invalid-args (reason #(array/quantize (array/array [[1 2]] {:dtype (ts/int-dtype)}) opts))) "ints"))
    (testing "shapes"
      (is (= :invalid-args (reason #(array/dequantize wq s b {:group-size 64 :bits 4}))) "another group size")
      (is (= :invalid-args (reason #(array/dequantize wq s (array/slice b 0 0 3) opts))))
      (is (= :invalid-args (reason #(array/dequantize (array/astype wq (ts/int-dtype)) s b opts))) "words not uint32")
      (is (= :invalid-args (reason #(array/quantized-matmul (normal 103 [2 32]) wq s b opts))) "x's last axis")
      (is (= :invalid-args (reason #(array/quantized-matmul x (array/reshape wq [1 6 8]) (array/reshape s [1 6 2]) (array/reshape b [1 6 2]) opts)))
          "a stack (gather-qmm's)")
      (is (= :invalid-args (reason #(array/gather-qmm x wq s b (assoc opts :rhs-indices (array/array [0.5]))))) "float indices")
      (is (= :invalid-args (reason #(array/dequantize wq s nil opts))) "affine needs biases"))
    (when-not (ts/native-grad?)
      ;; MLX's native gradients differentiate them as MLX does
      (testing "the weights are frozen"
        (is (= :no-gradient (reason #((array/grad (fn [s] (array/sum (array/quantized-matmul x wq s b opts)))) s))))
        (is (= :no-gradient (reason #((array/grad (fn [b] (array/sum (array/dequantize wq s b opts)))) b))))
        (is (= :no-gradient (reason #((array/grad (fn [w] (array/sum (first (array/quantize w opts))))) x))))))
    (when (ts/jvm?)
      (testing "the JVM backend refuses non-finite weights, and out-of-range indices"
        (is (= :non-finite (reason #(array/quantize (array/array [(assoc (vec (repeat 64 0.5)) 3 ##NaN)] {:dtype :float64}) opts))))
        (is (some? (ts/thrown-data (array/gather-qmm x (array/reshape wq [1 6 8]) (array/reshape s [1 6 2]) (array/reshape b [1 6 2])
                                                     (assoc opts :rhs-indices (array/array [1] {:dtype :int64}))))))))
    (testing "staging is loud"
      (is (= :unsupported (reason #(st/stage (fn [x wq s b] (array/quantized-matmul x wq s b opts))
                                             [[2 64] :float32] [[6 8] :uint32] [[6 2] :float32] [[6 2] :float32])))))))

(deftest a-query-that-sees-no-key-gets-zeros-on-every-path
  ;; on MLX's GPU with a head size of its kernels (64): its fused kernels
  ;; give zeros or another row there, its fallback the mean of the values
  (let [mask (array/array (vec (for [i (range 12)] (vec (for [j (range 16)] (and (pos? i) (<= j i)))))))]
    (doseq [tq [1 4 12]]
      (let [q (normal 110 [1 2 tq 64]) k (normal 111 [1 2 16 64]) v (normal 112 [1 2 16 64])
            out (array/sdpa q k v {:mask (array/slice mask 0 0 tq)})]
        (is (array/equal? (array/slice out 2 0 1) (array/zeros [1 2 1 64])) (str tq " queries"))))))
