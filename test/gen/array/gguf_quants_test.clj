(ns gen.array.gguf-quants-test
  "GGML's types encoded on the host (gen.array.gguf.quants, llama.cpp's
   reference quantizers ported), against llama.cpp's quantizer on the
   fixtures of test/fixtures/gguf (generate.py): qwen3-small-f16.gguf's
   tensors, quantized by llama.cpp v0.9.0 (llama_model_quantize, on macOS
   arm64) as qwen3-small-q4_k_m.gguf (Q4_K, Q5_K, Q6_K, Q8_0) and
   qwen3-small-mixed.gguf (Q4_0, Q4_1, Q5_0, Q5_1, BF16, F32, Q8_0); and
   quants-random.gguf, rows of eight kinds of random values (float32's
   subnormals and bfloat16's precision among them) by llama.cpp's
   quantize_row_*_ref (by ctypes) as each of the 8 quantized types.
   - Each tensor's values, encoded here as the reference's type, are its
     bytes, every one: in one range and in ranges on the fork-join pool.
   - Quantizing and dequantizing stays within the type's step, on random
     blocks and on edge blocks (zeros, a constant, a lone outlier, values
     at float16's least normal magnitude): Q8_0's |x - x̂| at most d/2 +
     127·|d - D| (d = max |x| / 127, D its float16 rounding); Q4_0's and
     Q5_0's a step (the most negative code clamps) and 2^(b-1)·|d - D|;
     Q4_1's and Q5_1's half a step and the float16 roundings of d and the
     minimum; the K-quants' at most 2
     of their sub-block's steps (d·sc; Q6_K's, a 16-group's max |x| / 31),
     a quarter on average (0.3), as gen.llm.models.qwen3-kquants-test
     measured llama.cpp's own; F16's and BF16's half an ulp (2^-11,
     2^-8 relative).
   Planted (2026-11-08), each failing here: Q4_1's minimum written with
   its sign flipped; Q4_0's values j and j + 16 swapped in their byte;
   Q5_K's fifth bits written to the next pair's mask; Q6_K's codes not
   offset by 32; make_qx_quants' first sums fused (clang vectorizes that
   loop, rounding each product); nearest_int as rint (an infinite or NaN
   scaled value, a scale's reciprocal past float's range: the subnormal
   rows); Q4_0's and Q4_1's bytes masked to their nibbles (C ors them
   whole, and a degenerate block's code is a byte). Not caught here:
   make_qkx2_quants' first error sum fused, which changes no byte of
   these fixtures and one tensor of Qwen3-0.6B's (gen.llm.convert-real-
   test catches it); Q5_1's codes converted as signed (fcvtzs), not
   unsigned (fcvtzu), which no input tells apart (the values are never
   negative)."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.array.gguf :as gguf]
            [gen.array.gguf.quants :as quants])
  (:import [java.nio ByteBuffer]
           [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]
           [java.util Random]))

(def ^:private dir "test/fixtures/gguf/")
(def ^:private source (str dir "qwen3-small-f16.gguf"))

(defn- by-name [path] (into {} (map (juxt :name identity)) (:tensors (gguf/header path))))

(defn- file-bytes
  "The bytes of the tensor info of the file at path."
  ^bytes [path {:keys [offset bytes]}]
  (let [out (byte-array bytes)]
    (with-open [f (java.io.RandomAccessFile. (str path) "r")]
      (.seek f (long offset))
      (.readFully f out))
    out))

(defn- encoded ^bytes [t ^floats x]
  (let [^ByteBuffer b (quants/encode t x)
        out (byte-array (.remaining b))]
    (.get b out)
    out))

(deftest encoded-tensors-are-llama-cpp-s-byte-for-byte
  (let [src (by-name source)]
    (doseq [reference ["qwen3-small-q4_k_m.gguf" "qwen3-small-mixed.gguf"]
            {:keys [name type] :as info} (:tensors (gguf/header (str dir reference)))
            :when (not= type (:type (src name)))]
      (testing (str reference ": " name " (" (clojure.core/name type) ")")
        (let [x (gguf/host-values source (src name))
              expected (file-bytes (str dir reference) info)]
          (is (java.util.Arrays/equals expected (encoded type x)) "in one range")
          (with-redefs-fn {#'quants/chunk-blocks 1}
            #(is (java.util.Arrays/equals expected (encoded type x)) "in ranges on the pool")))))))

(deftest random-blocks-are-llama-cpp-s-byte-for-byte
  ;; quants-random.gguf: rows of eight kinds of values (normal, uniform,
  ;; Laplace, mostly zeros, sub-blocks of scales 10^6 apart, small
  ;; integers, bfloat16's precision, float32's subnormals), and each by
  ;; llama.cpp's quantize_row_*_ref
  (let [f (str dir "quants-random.gguf")
        ts (by-name f)]
    (doseq [[name info] (sort-by key ts)
            :let [[_ t kind] (re-find #"^([a-z0-9_]+)\.([a-z0-9]+)\.weight$" name)]
            :when (not= t "x")]
      (testing name
        (is (java.util.Arrays/equals (file-bytes f info)
                                     (encoded (keyword t) (gguf/host-values f (ts (str "x." kind ".weight")))))
            "the same bytes")))))

(deftest the-references-cover-every-type-encoded
  (is (= (set (keys (dissoc quants/types :i32 :f16)))
         (set (mapcat (fn [f] (map :type (:tensors (gguf/header (str dir f)))))
                      ["qwen3-small-q4_k_m.gguf" "qwen3-small-mixed.gguf"])))
      "F16 aside, the sources' own type (each value rounded to nearest even: the next test)"))

;; ---------------------------------------------------------------------------
;; Within the step
;; ---------------------------------------------------------------------------

(defn- random-values
  "n values: blocks of 32 of random scales, a few edge blocks first."
  ^floats [^long n ^long seed]
  (let [r (Random. seed)
        out (float-array n)]
    (dotimes [i n]
      (let [block (quot i 32)
            scale (Math/pow 10.0 (- (* 3.0 (.nextDouble r)) 2.0))]
        (aset out i (float (case (int block)
                             0 0.0
                             1 0.75
                             2 (if (== i 70) 3.5 1e-3)
                             3 (if (even? i) 6.1e-5 -6.1e-5)
                             4 (if (== i 130) -60000.0 1.0)
                             (* scale (.nextGaussian r)))))))
    out))

(defn- decoded
  "The values of x encoded as t, decoded by gen.array.gguf's reader."
  ^floats [t shape ^floats x]
  (let [d (str (Files/createTempDirectory "quants-test" (make-array FileAttribute 0)))
        f (str d "/t.gguf")]
    (gguf/write-gguf f {:metadata [] :tensors [{:name "t.weight" :type t :shape shape :data x}]})
    (gguf/host-values f (first (:tensors (gguf/header f))))))

(defn- half-round ^double [^double x] (double (Float/float16ToFloat (Float/floatToFloat16 (float x)))))

(defn- block-bound
  "The largest |x - x̂| the 32-value type t allows in the block xs."
  ^double [t xs]
  (let [amax (reduce max (map #(Math/abs (double %)) xs))
        mn (reduce min (map double xs))
        mx (reduce max (map double xs))
        e (Math/ulp (float amax))]
    (case t
      :q8_0 (let [d (/ amax 127.0)] (+ (* 0.5 d) (* 127.0 (Math/abs (- d (half-round d)))) e))
      :q4_0 (let [d (/ amax 8.0)] (+ d (* 8.0 (Math/abs (- d (half-round d)))) e))
      :q5_0 (let [d (/ amax 16.0)] (+ d (* 16.0 (Math/abs (- d (half-round d)))) e))
      :q4_1 (let [d (/ (- mx mn) 15.0)] (+ (* 0.5 d) (* 15.0 (Math/abs (- d (half-round d)))) (Math/abs (- mn (half-round mn))) e))
      :q5_1 (let [d (/ (- mx mn) 31.0)] (+ (* 0.5 d) (* 31.0 (Math/abs (- d (half-round d)))) (Math/abs (- mn (half-round mn))) e)))))

(deftest blocks-of-32-are-their-values-within-a-step
  (let [x (random-values 2048 1)]
    (doseq [t [:q8_0 :q4_0 :q4_1 :q5_0 :q5_1]]
      (let [y (decoded t [8 256] x)
            worst (apply max (for [b (range 64)
                                   :let [xs (map #(aget x %) (range (* 32 b) (* 32 (inc b))))
                                         ys (map #(aget y %) (range (* 32 b) (* 32 (inc b))))]]
                               (/ (reduce max (map #(Math/abs (- (double %1) (double %2))) xs ys))
                                  (max 1e-30 (block-bound t xs)))))]
        (is (<= worst 1.0) (str (name t) ": the worst block at " worst " of its bound"))
        (is (= (vec (take 32 y)) (vec (repeat 32 0.0))) (str (name t) ": a block of zeros"))))))

(defn- super-block-values
  "n values in super-blocks of 256 of one random scale each, as a
   matrix's are, the first super-block's 8 groups the edge blocks of
   random-values (zeros, a constant, a lone outlier, …: their scales 10^5
   apart)."
  ^floats [^long n ^long seed]
  (let [r (Random. seed)
        edges (random-values 256 seed)
        out (float-array n)]
    (dotimes [sb (quot n 256)]
      (let [scale (Math/pow 10.0 (- (* 3.0 (.nextDouble r)) 2.0))]
        (dotimes [i 256]
          (aset out (+ (* 256 sb) i) (float (if (zero? sb) (aget edges i) (* scale (.nextGaussian r))))))))
    out))

(deftest k-quants-are-their-values-within-their-steps
  ;; a group's step: Q4_K's and Q5_K's its span over 15 or 31 codes (its
  ;; minimum at most 0), Q6_K's its max |x| / 31; the first super-block's
  ;; groups, of scales 10^5 apart, in steps of their super-block's largest
  ;; (a group 10^5 below it has a 6-bit scale of 0)
  (let [x (super-block-values 4096 2)]
    (doseq [t [:q4_k :q5_k :q6_k]]
      (let [y (decoded t [16 256] x)
            group (if (= t :q6_k) 16 32)
            levels (case t :q4_k 15.0 :q5_k 31.0 :q6_k 31.0)
            step (fn [g]
                   (let [xs (map #(double (aget x %)) (range (* group g) (* group (inc g))))]
                     (/ (if (= t :q6_k)
                          (reduce max (map #(Math/abs (double %)) xs))
                          (- (reduce max 0.0 xs) (reduce min 0.0 xs)))
                        levels)))
            per-sb (quot 256 group)
            errors (fn [gs step-of]
                     (for [g gs
                           :let [s (step-of g)]
                           :when (pos? s)
                           i (range (* group g) (* group (inc g)))]
                       (/ (Math/abs (- (double (aget x i)) (double (aget y i)))) s)))
            own (errors (range per-sb (quot 4096 group)) step)
            edge (errors (range per-sb) (constantly (reduce max (map step (range per-sb)))))
            mean (/ (reduce + own) (count own))]
        (is (<= (reduce max own) 2.0) (str (name t) ": the largest " (reduce max own) " steps"))
        (is (<= mean 0.3) (str (name t) ": the mean " mean " steps"))
        (is (<= (reduce max edge) 2.0) (str (name t) ": the edge super-block's largest " (reduce max edge) " of its steps"))
        (is (= (vec (take 32 y)) (vec (repeat 32 0.0))) (str (name t) ": zeros"))))))

(deftest floats-round-to-nearest-even
  (let [x (random-values 1024 3)]
    (doseq [[t rel] [[:f16 (Math/pow 2 -11)] [:bf16 (Math/pow 2 -8)] [:f32 0.0]]]
      (let [y (decoded t [4 256] x)]
        (is (every? true? (map (fn [a b] (<= (Math/abs (- (double a) (double b)))
                                             (max (* rel (Math/abs (double a))) (if (= t :f32) 0.0 6e-8))))
                               x y))
            (name t))))
    (testing "F16 is Java's rounding, BF16 ggml's (ties to even, NaN quiet)"
      (let [x (float-array [1.0 (Float/intBitsToFloat 0x3F808000) (Float/intBitsToFloat 0x3F818000) Float/NaN
                            Float/POSITIVE_INFINITY -0.0])
            bits (let [^ByteBuffer b (quants/encode :bf16 x)] (mapv #(bit-and (long (.getShort b (int (* 2 %)))) 0xffff) (range 6)))]
        (is (= [0x3F80 0x3F80 0x3F82 0x7FC0 0x7F80 0x8000] bits))))))

(deftest encode-is-loud
  (let [reason (fn [t x] (try (quants/encode t x) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (= :unsupported (reason :q3_k (float-array 256))))
    (is (= :unsupported (reason :i32 (float-array 32))) "I32 is copied, not encoded")
    (is (= :invalid-args (reason :q4_k (float-array 32))) "not a whole number of super-blocks")))
