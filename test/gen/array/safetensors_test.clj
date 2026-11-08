(ns gen.array.safetensors-test
  "gen.array/load-safetensors (experimental) on the backend under test,
   on the fixtures of test/fixtures/safetensors (generate.py):
   - every dtype loaded (F32, F16, BF16 with a subnormal and -0, I32, I64
     at ±2^53, U32 past 2^31, BOOL), a 0-d and an empty tensor, the
     metadata, against types.edn, exactly; F16 and BF16 as float16 and
     bfloat16 on MLX, float32 on the JVM backend; F64 on the JVM backend
     (a file holding it is loud on MLX, whose loader refuses it);
   - :select's tensors only (a tensor of a dtype not loaded among the
     others is never read);
   - a checkpoint sharded over three files, by its directory and by its
     index, against its shards read one by one;
   - on MLX, mlx_load_safetensors's arrays against the JVM backend's
     reader, bit for bit;
   - the loud errors: a dtype not loaded or not the format's, an int64
     past 2^53 on the JVM backend, files that are no safetensors or cut
     short, headers whose tensors leave a gap, overlap, or do not match
     their shapes, metadata not of strings, a shard missing or holding
     other tensors than its index maps to it, an index naming a file
     outside its directory, options unknown.
   Planted mutations of gen.array.safetensors, each failing a test here
   before it was trusted (2026-11-08): BF16's bits shifted 15, not 16;
   F16 decoded as BF16; a tensor's offset not past the header (8 + n);
   the coverage check skipped (a gap accepted); every shard's tensors
   checked against the first shard's names. (U32 read as signed is no
   mutation on the JVM backend: its uint32 arrays wrap as int32's.)"
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.array :as array]
            [gen.array.safetensors :as st]
            [gen.test-support :as ts])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))

(def ^:private dir "test/fixtures/safetensors/")

(defn- fixture [name] (str dir name))

(defn- host [x] (array/->clj x))

(def ^:private reference (delay (edn/read-string (slurp (fixture "types.edn")))))

(defn- thrown [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (ex-data e))))

(defn- temp-dir []
  (str (Files/createTempDirectory "safetensors-test" (make-array FileAttribute 0))))

(defn- write-file
  "A safetensors file at path of the header (a JSON string, as written)
   and the data (bytes)."
  [path ^String header ^bytes data]
  (let [h (.getBytes header "UTF-8")
        b (doto (ByteBuffer/allocate (+ 8 (alength h) (alength data))) (.order ByteOrder/LITTLE_ENDIAN))]
    (.putLong b (alength h))
    (.put b h)
    (.put b data)
    (io/copy (.array b) (io/file path))
    path))

(defn- floats-bytes ^bytes [xs]
  (let [b (doto (ByteBuffer/allocate (* 4 (count xs))) (.order ByteOrder/LITTLE_ENDIAN))]
    (doseq [x xs] (.putFloat b (float x)))
    (.array b)))

(defn- bits ^long [x] (Double/doubleToRawLongBits (double x)))

(deftest every-dtype-loads
  (let [skipped #{"t.i8"}
        {:keys [metadata tensors types files]} (array/load-safetensors (fixture "types.safetensors")
                                                                       {:select (complement skipped)})
        half (if (ts/jvm?) :float32 nil)]
    (is (= {"format" "pt" "note" "Stockholm är Sveriges huvudstad"} metadata))
    (is (= [(fixture "types.safetensors")] files))
    (is (= {"t.f32" :f32 "t.f16" :f16 "t.bf16" :bf16 "t.i32" :i32 "t.i64" :i64 "t.u32" :u32 "t.bool" :bool
            "t.scalar" :f32 "t.empty" :f32}
           types))
    (doseq [[name expected] (apply dissoc @reference skipped)]
      (is (= expected (host (tensors name))) name))

    (testing "BF16's -0 and subnormal, F16's subnormal, bit for bit"
      (let [bf16 (host (tensors "t.bf16"))]
        (is (= (bits -0.0) (bits (nth bf16 1))))
        (is (= (bits (Float/intBitsToFloat 0x10000)) (bits (nth bf16 2))))
        (is (= (bits -0.0) (bits (get-in (host (tensors "t.f16")) [0 1]))))))
    (is (= (cond-> {"t.f32" :float32 "t.f16" (or half :float16) "t.bf16" (or half :bfloat16) "t.i32" :int32
                    "t.i64" :int64 "t.u32" :uint32 "t.bool" :bool "t.scalar" :float32 "t.empty" :float32}
             (ts/jvm?) (dissoc "t.scalar"))
           (into {} (keep (fn [[k v]] (when-not (array/host? v) [k (array/dtype v)]))) tensors))
        "the dtypes (a 0-d array is a host number on the JVM backend)")
    (is (= [0 3] (array/shape (tensors "t.empty"))))
    (is (= [] (array/shape (tensors "t.scalar"))))))

(deftest f64-loads-on-the-jvm-backend
  ;; MLX 0.32.4's loader knows no F64 and refuses the whole file: loud
  ;; there, the F32 tensor beside it too
  (if (ts/jvm?)
    (let [{:keys [tensors types]} (array/load-safetensors (fixture "f64.safetensors"))]
      (is (= {"t.f64" :f64 "t.f32" :f32} types))
      (is (= [0.1 -2.5E300 4.9E-324] (host (tensors "t.f64"))))
      (is (= :float64 (array/dtype (tensors "t.f64")))))
    (is (= {:reason :unsupported :tensor "t.f64"}
           (select-keys (thrown #(array/load-safetensors (fixture "f64.safetensors") {:select #{"t.f32"}}))
                        [:reason :tensor])))))

(deftest select-reads-only-what-it-accepts
  (let [{:keys [tensors types]} (array/load-safetensors (fixture "types.safetensors") {:select #{"t.u32" "t.bool"}})]
    (is (= #{"t.u32" "t.bool"} (set (keys tensors)) (set (keys types))))
    (is (= [0 2147483653 4294967295] (host (tensors "t.u32")))))
  (is (= {} (:tensors (array/load-safetensors (fixture "types.safetensors") {:select (constantly false)})))))

(deftest shards-load-by-their-index
  (let [d (fixture "qwen3-tiny-8bit")
        shards (mapv #(str d (format "/model-%05d-of-00003.safetensors" %)) [1 2 3])
        whole (array/load-safetensors d)
        by-index (array/load-safetensors (str d "/model.safetensors.index.json"))
        one-by-one (map array/load-safetensors shards)]
    (is (= shards (:files whole) (:files by-index)))
    (is (= {"format" "mlx"} (:metadata whole)))
    (is (= (apply merge (map :types one-by-one)) (:types whole) (:types by-index)))
    (is (= 3 (count (filter seq (map :tensors one-by-one)))))
    (doseq [[name t] (apply merge (map :tensors one-by-one))]
      (is (= (host t) (host (get-in whole [:tensors name])) (host (get-in by-index [:tensors name]))) name))
    (testing "a matrix's words in one shard and its scales in another"
      (let [where (fn [k] (first (keep-indexed (fn [i s] (when (contains? (:tensors s) k) i)) one-by-one)))]
        (is (not= (where "model.layers.0.self_attn.o_proj.weight") (where "model.layers.0.self_attn.o_proj.scales")))))))

(deftest mlx-loads-what-the-jvm-reads
  ;; MLX's loader (mlx_load_safetensors) against the JVM backend's reader
  ;; (gen.array.safetensors's): the same values, bit for bit, F16 and BF16
  ;; in their own dtypes
  (if (ts/jvm?)
    (println "gen.array.safetensors-test: MLX's loader is compared on MLX")
    (let [jvm (array/backend-named :jvm)]
      (doseq [path [(fixture "types.safetensors") (fixture "qwen3-tiny-8bit") (fixture "qwen35-tiny-mlx-8bit")
                    (fixture "qwen3-tiny-4bit")]
              :let [opts {:select (complement #{"t.i8" "t.f64"})}
                    mine (array/load-safetensors path opts)
                    theirs (array/with-backend jvm (array/load-safetensors path opts))]]
        (is (= (:types mine) (:types theirs)) path)
        (doseq [[name t] (:tensors mine)
                :let [u (get-in theirs [:tensors name])
                      raw #(mapv bits (flatten [(host %)]))]]
          (is (= (host u) (host t)) (str path ": " name))
          (when (#{:f16 :bf16 :f32} (get-in mine [:types name]))
            (is (= (raw u) (raw t)) (str path ": " name ", bit for bit"))))))))

(deftest refused-and-malformed-files-are-loud
  (is (= {:reason :unsupported :tensor "t.i8" :type :i8}
         (select-keys (thrown #(array/load-safetensors (fixture "types.safetensors"))) [:reason :tensor :type]))
      "a dtype not loaded, selected")
  (testing "an int64 past 2^53: loud on the JVM backend, whose doubles do not hold it; MLX holds it"
    (if (ts/jvm?)
      (is (= :unsupported (:reason (thrown #(array/load-safetensors (fixture "i64-past-2-53.safetensors"))))))
      (is (= [9007199254740993] (host (get-in (array/load-safetensors (fixture "i64-past-2-53.safetensors"))
                                              [:tensors "t.i64"]))))))
  (is (= :no-file (:reason (thrown #(array/load-safetensors (fixture "no-such.safetensors"))))))
  (is (= :no-file (:reason (thrown #(array/load-safetensors (temp-dir))))) "a directory with neither file")
  (is (= :invalid-safetensors (:reason (thrown #(array/load-safetensors (fixture "generate.py"))))) "not a safetensors file")
  (let [d (temp-dir)
        f (fn [name] (str d "/" name))
        two (floats-bytes [1 2])
        reason (fn [name header data] (:reason (thrown #(array/load-safetensors (write-file (f name) header data)))))
        ^bytes whole (Files/readAllBytes (.toPath (io/file (fixture "types.safetensors"))))]
    (is (= :invalid-safetensors (do (io/copy (byte-array 5) (io/file (f "short.safetensors")))
                                    (:reason (thrown #(array/load-safetensors (f "short.safetensors"))))))
        "shorter than its header's length")
    (is (= :invalid-safetensors (do (io/copy (java.util.Arrays/copyOf whole 300) (io/file (f "cut.safetensors")))
                                    (:reason (thrown #(array/load-safetensors (f "cut.safetensors"))))))
        "the header cut short")
    (is (= :invalid-safetensors (do (io/copy (java.util.Arrays/copyOf whole (- (alength whole) 4)) (io/file (f "data.safetensors")))
                                    (:reason (thrown #(array/load-safetensors (f "data.safetensors"))))))
        "the data cut short")
    (is (= :invalid-safetensors (reason "json.safetensors" "{\"a\": " two)) "no JSON")
    (is (= :invalid-safetensors (reason "array.safetensors" "[1, 2]" two)) "no JSON object")
    (is (= :invalid-safetensors (reason "twice.safetensors" "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}" two))
        "a tensor named twice")
    (is (= :invalid-safetensors (reason "keys.safetensors" "{\"a\":{\"dtype\":\"F32\",\"shape\":[2]}}" two)) "an entry without its offsets")
    (is (= :invalid-safetensors (reason "extra.safetensors" "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8],\"x\":1}}" two))
        "an entry of other keys")
    (is (= :unsupported (reason "dtype.safetensors" "{\"a\":{\"dtype\":\"F7\",\"shape\":[2],\"data_offsets\":[0,8]}}" two))
        "a dtype not the format's")
    (is (= :invalid-safetensors (reason "size.safetensors" "{\"a\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,8]}}" two))
        "offsets that do not match the shape")
    (is (= :invalid-safetensors (reason "gap.safetensors" "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}" two))
        "a gap")
    (is (= :invalid-safetensors (reason "overlap.safetensors" "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}" two))
        "an overlap")
    (is (= :invalid-safetensors (reason "trailing.safetensors" "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}" two))
        "bytes past the last tensor's")
    (is (= :invalid-safetensors (reason "past.safetensors" "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}" two))
        "data past the file's end")
    (is (= :invalid-safetensors (reason "meta.safetensors" "{\"__metadata__\":{\"n\":1},\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}" two))
        "metadata not of strings")
    (is (= [1.0 2.0] (host (get-in (array/load-safetensors (write-file (f "ok.safetensors") "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}  " two))
                                   [:tensors "a"])))
        "the same file well formed (a header padded with spaces)"))
  (testing "an index: a shard missing, a shard holding other tensors, a file outside its directory"
    (let [d (temp-dir)
          src (fixture "qwen3-tiny-8bit")
          index (slurp (str src "/model.safetensors.index.json"))
          copy (fn [name] (io/copy (io/file src name) (io/file d name)))
          index! (fn [s] (spit (str d "/model.safetensors.index.json") s))]
      (index! index)
      (copy "model-00001-of-00003.safetensors")
      (copy "model-00002-of-00003.safetensors")
      (let [e (thrown #(array/load-safetensors d))]
        (is (= :invalid-safetensors (:reason e)))
        (is (= "model-00003-of-00003.safetensors" (:missing e))))
      (copy "model-00003-of-00003.safetensors")
      (is (map? (array/load-safetensors d)) "the copy, whole")
      (index! (str/replace-first index "\"model.norm.weight\": \"model-00003-of-00003.safetensors\""
                                 "\"model.norm.weight\": \"model-00001-of-00003.safetensors\""))
      (let [e (thrown #(array/load-safetensors d))]
        (is (= :invalid-safetensors (:reason e)))
        (is (some #{"model.norm.weight"} (concat (:missing e) (:unmapped e)))))
      (index! (str/replace-first index "\"model.norm.weight\": \"model-00003-of-00003.safetensors\""
                                 "\"model.norm.weight\": \"../model-00003-of-00003.safetensors\""))
      (is (= :invalid-safetensors (:reason (thrown #(array/load-safetensors d)))))
      (index! "{\"metadata\": {}}")
      (is (= :invalid-safetensors (:reason (thrown #(array/load-safetensors d)))) "no weight map")))
  (testing "options"
    (is (= :unknown-options (:reason (thrown #(array/load-safetensors (fixture "types.safetensors") {:only #{}})))))
    (is (= :invalid-options (:reason (thrown #(array/load-safetensors (fixture "types.safetensors") {:select 3})))))))

(deftest headers-describe-the-tensors
  (let [{:keys [tensors metadata]} (st/header (fixture "types.safetensors"))]
    (is (= "pt" (metadata "format")))
    (is (= (sort-by :offset tensors) tensors) "in the data's order")
    (is (= {:type :bf16 :dtype :bfloat16 :shape [5] :bytes 10}
           (select-keys (first (filter #(= "t.bf16" (:name %)) tensors)) [:type :dtype :shape :bytes])))
    (is (nil? (:dtype (first (filter #(= "t.i8" (:name %)) tensors)))) "a dtype not loaded")))
