(ns gen.array.gguf-test
  "gen.array/load-gguf (experimental) on the backend under test: GGUF
   files read as MLX's loader reads them, on the fixtures of
   test/fixtures/gguf (generate.py):
   - every metadata value type, read as host values;
   - every tensor type loaded (F32, F16, I32; Q8_0, Q4_0 and Q4_1 as MLX's
     affine quantization), against gguf-py's reader and dequantizer
     (types.edn), exactly in float32;
   - the types MLX's loader does not read, Q5_0, Q5_1 and the K-quants
     (Q4_K and Q5_K as MLX's affine quantization at 4 and 5 bits, a
     sub-block a group; Q6_K decoded to float32), against llama.cpp's own
     dequantization (kquants.edn: gguf-py's values, which generate.py
     checked bit for bit against llama.cpp's dequantize_row_*), exactly in
     float32: random blocks, every scale, min and code bit, a negative, a
     subnormal and a large float16 scale among them; and decoded in
     ranges on the fork-join pool, as a large tensor is;
   - the loaded triples multiplied by quantized-matmul and gather-qmm at
     their own bits (4 and 5), as their dequantized matrices;
   - on MLX, mlx_load_gguf's arrays against the JVM backend's reader, bit
     for bit; and the host reader's on MLX (a file of K-quants loads so
     there) against both;
   - a model split over files against the same model in one;
   - the loud errors: unsupported tensor types, files that are no GGUF or
     cut short, a split's file missing or loaded from another than its
     first, and on MLX float64 metadata (which MLX's loader misreads).
   Planted mutations of gen.array.gguf, each failing a test here before
   it was trusted (2026-11-07): Q8_0's int8 read without the shift to
   uint8 (x ^ 0x80); a 4-bit block's values 16 to 31 taken from the low
   nibbles; Q8_0's bias -128·d rounded to float16 (it overflows in the
   block whose scale is past 511.75); the dimensions not reversed; the
   data's start not aligned; a split's metadata taken from its last file.
   And of the K-quants' decoding (2026-11-08): a Q4_K sub-block's scale
   and min swapped (get_scale_min_k4's two reads, sub-blocks 0 to 3); the
   nibbles of a scale byte swapped (sub-blocks 4 to 7); their high scale
   bits read from the min's byte; the bias -(dmin·m) with its sign
   flipped; the low- and high-nibble sub-blocks of a chunk swapped; Q5_K's
   fifth bit read from bit s + 1; a 5-bit group's second 40 bits shifted
   7, not 8; Q6_K's scale of the second 16 values of a 32 taken from the
   first's; Q6_K's codes not offset by 32; Q5_0's bias -16·d taken as
   -8·d; a range of the decoding on the fork-join pool ending a block
   short."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.gguf :as gguf]
            [gen.random :as random]
            [gen.test-support :as ts])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private dir "test/fixtures/gguf/")

(defn- fixture [name] (str dir name))

(defn- host [x] (array/->clj x))

(def ^:private reference
  (delay (edn/read-string (slurp (fixture "types.edn")))))

(def ^:private kquants-reference
  (delay (edn/read-string (slurp (fixture "kquants.edn")))))

(defn- dequantized
  "A quantized tensor's values (load-gguf's map), dequantized in float32."
  [{:keys [wq scales biases] :as q}]
  (array/dequantize wq scales biases (assoc (select-keys q [:bits :group-size]) :dtype :float32)))

(defn- values [t] (host (if (map? t) (dequantized t) t)))

(defn- thrown [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (ex-data e))))

(defn- temp-dir []
  (str (Files/createTempDirectory "gguf-test" (make-array FileAttribute 0))))

(deftest metadata-reads-every-value-type
  (let [{:keys [metadata types files]} (array/load-gguf (fixture "types.gguf"))]
    (is (= {"test.u8" 200 "test.i8" -100 "test.u16" 60000 "test.i16" -30000 "test.u32" 4000000000
            "test.i32" -2000000000 "test.f32" (double (float 0.1)) "test.bool" true
            "test.string" "Stockholm är Sveriges huvudstad" "test.u64" (+ 9223372036854775807N 6)
            "test.i64" -4611686018427387904 "test.ints" [1 -2 3] "test.strings" ["a" "" "ö"]}
           (into {} (filter #(.startsWith ^String (key %) "test.")) metadata)))
    (is (= "test" (metadata "general.architecture")))
    (is (= [(fixture "types.gguf")] files))
    (is (= {"t.f32.weight" :f32 "t.vector" :f32 "t.f16.weight" :f16 "t.i32" :i32 "t.q8_0.weight" :q8_0
            "t.q8_0_3d.weight" :q8_0 "t.q4_0.weight" :q4_0 "t.q4_1.weight" :q4_1}
           types))))

(deftest tensors-load-as-gguf-py-reads-them
  ;; gguf-py's dequantization is d·q (Q8_0), d·(q - 8) (Q4_0), d·q + m
  ;; (Q4_1), in float32: the values exactly, read in float32
  (let [{:keys [tensors]} (array/load-gguf (fixture "types.gguf"))
        f16 (if (ts/jvm?) :float32 :float16)]
    (doseq [[name expected] @reference]
      (testing name
        (is (= expected (values (tensors name))))))
    (testing "dtypes and shapes"
      (is (= [:float32 [2 64]] ((juxt array/dtype array/shape) (tensors "t.f32.weight"))))
      (is (= [:float32 [5]] ((juxt array/dtype array/shape) (tensors "t.vector"))))
      (is (= [f16 [2 64]] ((juxt array/dtype array/shape) (tensors "t.f16.weight"))) "float16, or float32 without it")
      (is (= [:int32 [5]] ((juxt array/dtype array/shape) (tensors "t.i32"))))
      (doseq [[name bits shape] [["t.q8_0.weight" 8 [3 64]] ["t.q8_0_3d.weight" 8 [2 3 64]]
                                 ["t.q4_0.weight" 4 [3 64]] ["t.q4_1.weight" 4 [3 64]]]]
        (let [{:keys [wq scales biases] :as q} (tensors name)
              lead (pop shape)]
          (is (= {:bits bits :group-size 32} (select-keys q [:bits :group-size])) name)
          (is (= [:uint32 (conj lead (quot (* 64 bits) 32))] ((juxt array/dtype array/shape) wq)) name)
          (is (= [:float32 (conj lead 2)] ((juxt array/dtype array/shape) scales)) name)
          (is (= [:float32 (conj lead 2)] ((juxt array/dtype array/shape) biases)) name))))
    (testing "a Q8_0 block whose scale is past 511.75: its bias -128·d is exact in float32, where float16's overflows"
      (let [{:keys [scales biases]} (tensors "t.q8_0.weight")
            d (get-in (host scales) [2 1])
            b (get-in (host biases) [2 1])]
        (is (< 511.75 d))
        (is (= (* -128.0 d) b))
        (is (every? #(Double/isFinite %) (flatten (values (tensors "t.q8_0.weight")))))))))

(deftest k-quants-load-as-llama-cpp-dequantizes-them
  ;; llama.cpp's dequantization in float32 (kquants.edn): Q4_K's and
  ;; Q5_K's d·sc·q - dmin·m, each product exact (an 11-bit significand
  ;; times a 6-bit integer, times a code of 4 or 5 bits), so MLX's
  ;; scale·q + bias with the scale d·sc and the bias -(dmin·m) in float32
  ;; is the same rounding of the same value; Q6_K's d·sc·(q - 32) exact in
  ;; float32 (11, 8 and 6 significant bits), loaded as it
  (let [{:keys [tensors types]} (array/load-gguf (fixture "kquants.gguf"))]
    (is (= {"t.q5_0.weight" :q5_0 "t.q5_1.weight" :q5_1 "t.q4_k.weight" :q4_k "t.q4_k_3d.weight" :q4_k
            "t.q5_k.weight" :q5_k "t.q6_k.weight" :q6_k}
           types))
    (doseq [[name expected] @kquants-reference]
      (testing name
        (is (= expected (values (tensors name))))))
    (testing "decoded in ranges on the fork-join pool (as a large tensor is): the same values"
      (with-redefs-fn {#'gguf/chunk-bytes 1}
        #(let [ranged (:tensors (array/load-gguf (fixture "kquants.gguf")))]
           (doseq [[name expected] @kquants-reference]
             (is (= expected (values (ranged name))) name)))))
    (testing "the fixture's blocks: negative, subnormal and large float16 scales"
      (let [ss (flatten (host (:scales (tensors "t.q4_k.weight"))))]
        (is (some neg? ss))
        (is (some #(< 0.0 (Math/abs (double %)) 1e-6) ss) "a subnormal d times a scale")
        (is (some #(< 1000.0 (Math/abs (double %))) ss))))
    (testing "dtypes and shapes: groups of 32 at the type's bits; Q6_K dense"
      (doseq [[name bits shape] [["t.q5_0.weight" 5 [3 64]] ["t.q5_1.weight" 5 [3 64]] ["t.q4_k.weight" 4 [2 512]]
                                 ["t.q4_k_3d.weight" 4 [2 1 256]] ["t.q5_k.weight" 5 [2 512]]]]
        (let [{:keys [wq scales biases] :as q} (tensors name)
              lead (pop shape)
              k (peek shape)]
          (is (= {:bits bits :group-size 32} (select-keys q [:bits :group-size])) name)
          (is (= [:uint32 (conj lead (quot (* k bits) 32))] ((juxt array/dtype array/shape) wq)) name)
          (is (= [:float32 (conj lead (quot k 32))] ((juxt array/dtype array/shape) scales)) name)
          (is (= [:float32 (conj lead (quot k 32))] ((juxt array/dtype array/shape) biases)) name)))
      (is (= [:float32 [2 512]] ((juxt array/dtype array/shape) (tensors "t.q6_k.weight")))))))

(deftest loaded-triples-multiply-at-their-own-bits
  ;; quantized-matmul and gather-qmm take each tensor's bits and group
  ;; size (4 and 5 here), as a mixed file's model does: the products of
  ;; the dequantized matrices, within the round-off of a sum of 512
  ;; products in float32 (or float64), relative to the largest
  (let [{:keys [tensors]} (array/load-gguf (fixture "kquants.gguf"))
        opts (fn [q] (select-keys q [:bits :group-size]))
        dense (fn [{:keys [wq scales biases] :as q} x]
                (array/dequantize wq scales biases (assoc (opts q) :dtype (array/dtype x))))
        near? (fn [expected actual]
                (let [scale (array/item (array/max (array/abs expected)))]
                  (ts/close? actual expected {:rtol 0.0 :atol (* (ts/tol 1e-12) 50 (double scale))})))]
    (doseq [name ["t.q4_k.weight" "t.q5_k.weight"]]
      (let [{:keys [wq scales biases] :as q} (tensors name)
            x (random/normal (random/key 3) [3 512])
            w (dense q x)]
        (is (near? (array/matmul x (array/transpose w)) (array/quantized-matmul x wq scales biases (opts q))) name)))
    (let [{:keys [wq scales biases] :as q} (tensors "t.q4_k_3d.weight")
          x (random/normal (random/key 4) [2 1 1 256])
          w (dense q x)
          rhs (array/array [[1] [0]] {:dtype :int32})
          expected (array/stack [(array/matmul (array/reshape (array/slice x 0 0 1) [1 256])
                                               (array/transpose (array/reshape (array/slice w 0 1 2) [1 256])))
                                 (array/matmul (array/reshape (array/slice x 0 1 2) [1 256])
                                               (array/transpose (array/reshape (array/slice w 0 0 1) [1 256])))])]
      (is (near? (array/reshape expected [2 1 1 1])
                 (array/gather-qmm x wq scales biases (assoc (opts q) :lhs-indices (array/array [[0] [1]] {:dtype :int32})
                                                             :rhs-indices rhs)))
          "gather-qmm"))))

(deftest mlx-loads-what-the-jvm-reads
  ;; MLX's loader (mlx_load_gguf, its float16 scales cast to float32)
  ;; against the JVM backend's reader, which is gen.array.gguf's: the same
  ;; words, scales, biases and values, bit for bit. A file of K-quants
  ;; (kquants.gguf, and a model mixing them with Q8_0) loads on MLX by the
  ;; same host reader, made MLX's arrays; and the host reader's arrays on
  ;; MLX are its loader's for every type it reads (types.gguf)
  (if (ts/jvm?)
    (println "gen.array.gguf-test: MLX's loader is compared on MLX")
    (let [jvm (array/backend-named :jvm)
          raw (fn [t] (if (map? t) (mapv host [(:wq t) (:scales t) (:biases t)]) (host t)))
          load-host (requiring-resolve 'gen.backend.mlx.io/load-host)]
      (doseq [file ["types.gguf" "qwen3-tiny-q8_0.gguf" "qwen3-tiny-f32.gguf" "kquants.gguf" "qwen3-small-q4_k_m.gguf"]]
        (let [mine (array/load-gguf (fixture file))
              theirs (array/with-backend jvm (array/load-gguf (fixture file)))]
          (is (= (:metadata mine) (:metadata theirs)) file)
          (is (= (set (keys (:tensors mine))) (set (keys (:tensors theirs)))) file)
          (doseq [[name t] (:tensors mine)]
            (is (= (raw t) (raw (get-in theirs [:tensors name]))) (str file ": " name)))))
      (let [mlx (:tensors (array/load-gguf (fixture "types.gguf")))
            hosted (load-host array/*backend* (first (gguf/headers (fixture "types.gguf"))))]
        (is (= (set (keys mlx)) (set (keys hosted))))
        (doseq [[name t] mlx]
          (let [h (hosted name)]
            (is (= (raw t) (raw (if (vector? h) (zipmap [:wq :scales :biases] h) h))) (str "host reader: " name))
            (is (= (if (vector? h) [:uint32 :float32 :float32] (array/dtype t))
                   (if (vector? h) (mapv array/dtype h) (array/dtype h)))
                name)))))))

(deftest split-models-load-from-their-first-file
  (let [one (array/load-gguf (fixture "qwen3-tiny-q8_0.gguf"))
        split (array/load-gguf (fixture "qwen3-tiny-q8_0-00001-of-00003.gguf"))]
    (is (= (mapv #(fixture (format "qwen3-tiny-q8_0-%05d-of-00003.gguf" %)) [1 2 3]) (:files split)))
    (is (= (:metadata one) (apply dissoc (:metadata split) ["split.no" "split.count" "split.tensors.count"])))
    (is (= {"split.no" 0 "split.count" 3 "split.tensors.count" 35}
           (select-keys (:metadata split) ["split.no" "split.count" "split.tensors.count"])))
    (is (= (:types one) (:types split)))
    (is (= 35 (count (:tensors split))))
    (doseq [[name t] (:tensors one)]
      (is (= (values t) (values (get-in split [:tensors name]))) name))
    (testing "loud: another file than the first, a file missing, a first file not named as a split's"
      (is (= :invalid-gguf (:reason (thrown #(array/load-gguf (fixture "qwen3-tiny-q8_0-00002-of-00003.gguf"))))))
      (let [d (temp-dir)
            copy (fn [from to] (io/copy (io/file (fixture from)) (io/file d to)))]
        (copy "qwen3-tiny-q8_0-00001-of-00003.gguf" "m-00001-of-00003.gguf")
        (copy "qwen3-tiny-q8_0-00002-of-00003.gguf" "m-00002-of-00003.gguf")
        (copy "qwen3-tiny-q8_0-00001-of-00003.gguf" "m.gguf")
        (let [e (thrown #(array/load-gguf (str d "/m-00001-of-00003.gguf")))]
          (is (= :invalid-gguf (:reason e)))
          (is (= (str d "/m-00003-of-00003.gguf") (:missing e))))
        (is (= :invalid-gguf (:reason (thrown #(array/load-gguf (str d "/m.gguf"))))))))))

(deftest refused-files-are-loud
  (doseq [[file t] [["unsupported-q3_k.gguf" :q3_k] ["unsupported-bf16.gguf" :bf16]]]
    (is (= {:reason :unsupported :tensor "refused.weight" :type t}
           (select-keys (thrown #(array/load-gguf (fixture file))) [:reason :tensor :type]))
        file))
  (is (= :invalid-gguf (:reason (thrown #(array/load-gguf (fixture "generate.py"))))) "not a GGUF file")
  (is (= :no-file (:reason (thrown #(array/load-gguf (fixture "no-such-file.gguf"))))))
  (let [d (temp-dir)
        ^bytes bs (Files/readAllBytes (.toPath (io/file (fixture "qwen3-tiny-f32.gguf"))))
        reason (fn [name ^bytes content]
                 (let [f (io/file d name)]
                   (io/copy content f)
                   (:reason (thrown #(array/load-gguf (str f))))))]
    (is (= :invalid-gguf (reason "header.gguf" (java.util.Arrays/copyOf bs 600))) "the header cut short")
    (is (= :invalid-gguf (reason "data.gguf" (java.util.Arrays/copyOf bs (- (alength bs) 100)))) "a tensor's data cut short")
    (is (= :invalid-gguf (reason "v1.gguf" (doto (aclone bs) (aset-byte 4 1)))) "version 1"))
  (testing "float64 metadata: read on the JVM backend; refused on MLX, whose loader misreads it"
    (if (ts/jvm?)
      (let [{:keys [metadata tensors]} (array/load-gguf (fixture "float64-metadata.gguf"))]
        (is (= [0.1 [0.5 -2.25] "read after the float64 values"] (mapv metadata ["test.f64" "test.f64s" "test.after"])))
        (is (= (mapv #(vec (range (* 32 %) (* 32 (inc %)))) [0 1]) (mapv #(mapv long %) (host (tensors "t.weight"))))))
      (is (= {:reason :unsupported :keys ["test.f64" "test.f64s"]}
             (select-keys (thrown #(array/load-gguf (fixture "float64-metadata.gguf"))) [:reason :keys]))))))

(deftest a-model-s-quantized-file-is-its-float-file-within-a-step
  ;; the same weights as F32 and as Q8_0 (gguf-py's quantizer, which
  ;; rounds x/d to q and then d to float16, d = max |x| / 127): every value
  ;; within d/2 + 127·|d - float16(d)| <= (1/2 + 127·2^-11)·d of its
  ;; block's, so the files' tensors are read from where they lie
  (let [f32 (:tensors (array/load-gguf (fixture "qwen3-tiny-f32.gguf")))
        q8 (array/load-gguf (fixture "qwen3-tiny-q8_0.gguf"))]
    (is (= (set (keys f32)) (set (keys (:tensors q8)))))
    (doseq [[name t] (:tensors q8)]
      (if (map? t)
        (let [w (host (f32 name))
              d (host (dequantized t))
              bound (mapv (fn [row] (mapcat #(repeat 32 (* (+ 0.5 (/ 127.0 2048)) (Math/abs (double %)))) row)) (host (:scales t)))]
          (is (every? true? (map (fn [wr dr sr] (every? true? (map #(<= (Math/abs (- %1 %2)) %3) wr dr sr))) w d bound))
              name))
        (is (= (host (f32 name)) (host t)) name)))))

;; ---------------------------------------------------------------------------
;; Writing (gen.array.gguf/write-gguf, for gen.llm.lora's adapters)
;; ---------------------------------------------------------------------------

(deftest the-writer-writes-what-the-reader-reads
  ;; every metadata value type, read back as written (types.gguf's values;
  ;; float64 through the host reader only, as MLX's loader misreads it),
  ;; and F32 tensors of 1 to 4 axes loaded as written. That the bytes are
  ;; gguf-py's own is gen.llm.lora-test's: an adapter gguf-py wrote is
  ;; written again byte for byte. Planted (2026-11-07): the dimensions not
  ;; reversed, a tensor's data not padded to 32 bytes: each fails here.
  (let [d (temp-dir)
        metadata [["general.architecture" :string "test"]
                  ["test.u8" :uint8 200] ["test.i8" :int8 -100] ["test.u16" :uint16 60000] ["test.i16" :int16 -30000]
                  ["test.u32" :uint32 4000000000] ["test.i32" :int32 -2000000000] ["test.f32" :float32 0.1]
                  ["test.bool" :bool true] ["test.string" :string "Stockholm är Sveriges huvudstad"]
                  ["test.u64" :uint64 (+ 9223372036854775807N 6)] ["test.i64" :int64 -4611686018427387904]
                  ["test.ints" [:array :int32] [1 -2 3]] ["test.strings" [:array :string] ["a" "" "ö"]]]
        tensors [{:name "t.vector" :type :f32 :shape [5] :data (float-array [1 2 3 4 5])}
                 {:name "t.matrix.weight" :type :f32 :shape [3 2] :data (float-array [0.5 -1 2 -0.25 3 1e-3])}
                 {:name "t.cube" :type :f32 :shape [2 1 3] :data (float-array (range 6))}
                 {:name "t.four" :type :f32 :shape [1 2 1 2] :data (float-array [7 8 9 10])}]
        f (str d "/w.gguf")
        g (str d "/f64.gguf")]
    (is (= f (gguf/write-gguf f {:metadata metadata :tensors tensors})))
    (let [{md :metadata ts :tensors types :types} (array/load-gguf f)
          header (gguf/header f)]
      (is (= (into {} (map (fn [[k t v]] [k (if (= :float32 t) (double (float v)) v)])) metadata) md))
      (is (= (into {} (map (fn [[k t]] [k t])) metadata) (:value-types header)))
      (is (= (mapv :name tensors) (mapv :name (:tensors header))) "in their order")
      (is (every? #(zero? (rem (long (:offset %)) 32)) (:tensors header)) "aligned")
      (is (= (zipmap (map :name tensors) (repeat :f32)) types))
      (doseq [{:keys [name shape data]} tensors]
        (is (= shape (array/shape (ts name))) name)
        (is (= (mapv double data) (mapv double (array/->floats (ts name)))) name)))
    (gguf/write-gguf g {:metadata (conj metadata ["test.f64" :float64 0.1]) :tensors []})
    (is (= 0.1 (get-in (gguf/header g) [:metadata "test.f64"])))
    (testing "loud"
      (let [reason (fn [spec] (:reason (thrown #(gguf/write-gguf (str d "/x.gguf") spec))))]
        (is (= :invalid-args (reason {:metadata [["k" :uint8 256]]})) "a value past its type")
        (is (= :invalid-args (reason {:metadata [["k" :int32 1.5]]})))
        (is (= :invalid-args (reason {:metadata [["k" :string 1]]})))
        (is (= :invalid-args (reason {:metadata [["k" :float16 1.0]]})) "no such value type")
        (is (= :invalid-args (reason {:metadata [["k" :string "a"] ["k" :string "b"]]})) "a key twice")
        (is (= :invalid-args (reason {:tensors [(first tensors) (first tensors)]})) "a tensor twice")
        (is (= :invalid-args (reason {:tensors [(assoc (first tensors) :shape [4])]})) "data of another size")
        (is (= :unsupported (reason {:tensors [(assoc (first tensors) :type :f16)]})))
        (is (not (.exists (io/file d "x.gguf"))) "nothing written")))))
