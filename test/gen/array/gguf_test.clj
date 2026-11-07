(ns gen.array.gguf-test
  "gen.array/load-gguf (experimental) on the backend under test: GGUF
   files read as MLX's loader reads them, on the fixtures of
   test/fixtures/gguf (generate.py):
   - every metadata value type, read as host values;
   - every tensor type loaded (F32, F16, I32; Q8_0, Q4_0 and Q4_1 as MLX's
     affine quantization), against gguf-py's reader and dequantizer
     (types.edn), exactly in float32;
   - on MLX, mlx_load_gguf's arrays against the JVM backend's reader, bit
     for bit;
   - a model split over files against the same model in one;
   - the loud errors: unsupported tensor types, files that are no GGUF or
     cut short, a split's file missing or loaded from another than its
     first, and on MLX float64 metadata (which MLX's loader misreads).
   Planted mutations of gen.array.gguf, each failing a test here before
   it was trusted (2026-11-07): Q8_0's int8 read without the shift to
   uint8 (x ^ 0x80); a 4-bit block's values 16 to 31 taken from the low
   nibbles; Q8_0's bias -128·d rounded to float16 (it overflows in the
   block whose scale is past 511.75); the dimensions not reversed; the
   data's start not aligned; a split's metadata taken from its last file."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.test-support :as ts])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private dir "test/fixtures/gguf/")

(defn- fixture [name] (str dir name))

(defn- host [x] (array/->clj x))

(def ^:private reference
  (delay (edn/read-string (slurp (fixture "types.edn")))))

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

(deftest mlx-loads-what-the-jvm-reads
  ;; MLX's loader (mlx_load_gguf, its float16 scales cast to float32)
  ;; against the JVM backend's reader, which is gen.array.gguf's: the same
  ;; words, scales, biases and values, bit for bit
  (if (ts/jvm?)
    (println "gen.array.gguf-test: MLX's loader is compared on MLX")
    (let [jvm (array/backend-named :jvm)
          raw (fn [t] (if (map? t) (mapv host [(:wq t) (:scales t) (:biases t)]) (host t)))]
      (doseq [file ["types.gguf" "qwen3-tiny-q8_0.gguf" "qwen3-tiny-f32.gguf"]]
        (let [mine (array/load-gguf (fixture file))
              theirs (array/with-backend jvm (array/load-gguf (fixture file)))]
          (is (= (:metadata mine) (:metadata theirs)) file)
          (is (= (set (keys (:tensors mine))) (set (keys (:tensors theirs)))) file)
          (doseq [[name t] (:tensors mine)]
            (is (= (raw t) (raw (get-in theirs [:tensors name]))) (str file ": " name))))))))

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
  (doseq [[file t] [["unsupported-q4_k.gguf" :q4_k] ["unsupported-bf16.gguf" :bf16]]]
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
