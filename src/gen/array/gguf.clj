(ns gen.array.gguf
  "GGUF files (llama.cpp's format for models) read on the host, below the
   backends: the body of gen.array/load-gguf, and the reader the JVM
   backend loads tensors with.

   - A file's header: its metadata (keys to host values) and its tensors'
     descriptions, read here on every backend, so that a model's metadata
     is the same wherever it loads.
   - A model split over files (llama.cpp's gguf-split, path
     <prefix>-00001-of-0000N.gguf): every file, from the first.
   - The tensor types loaded, as MLX's loader (mlx/io/gguf.cpp,
     gguf_quants.cpp) loads them: F32, F16 and I32 as arrays (and BF16,
     which MLX's loader is not trusted with: below), and Q8_0,
     Q4_0 and Q4_1 as MLX's affine quantization in groups of 32 (the
     quantized words, scales and biases gen.array/dequantize takes), the
     scales and biases in float32 (DESIGN §9, 1.3.0's engine design).
   - And the types MLX's loader does not read (it would turn them into
     float16, silently), decoded here on every backend, each value the
     file's exactly (llama.cpp's dequantize_row_*, bit for bit):
     - Q5_0 and Q5_1 (Q4_K_M's fallbacks for rows not of 256 values) as
       the affine quantization at 5 bits, groups of 32;
     - Q4_K and Q5_K (super-blocks of 256: 8 sub-blocks of 32, each a
       6-bit scale sc and min m over the super-block's float16 d and
       dmin) as the affine quantization at 4 and 5 bits, a sub-block a
       group: scale d·sc and bias -(dmin·m) in float32, both products
       exact, as llama.cpp computes them;
     - Q6_K (16 sub-blocks of 16, int8 scales) as float32, d·sc·(q - 32)
       (exact): MLX's affine kernels take groups of 32, 64 or 128, and
       two sub-blocks of different scales share no affine group;
     - BF16 as bfloat16 (float32 on a backend without it), its bits.
     Every other type is a loud error.
   - Writing (write-gguf): a file of F32, F16, BF16, Q8_0, Q4_0, Q4_1,
     Q5_0, Q5_1, Q4_K, Q5_K and Q6_K tensors, from their values (encoded
     by llama.cpp's reference quantizers, gen.array.gguf.quants) or their
     bytes (copied), for gen.llm.lora's adapters and gen.llm.convert's
     models.

   Dimensions: GGML lists a tensor's innermost first; a shape here is
   gen.array's (outermost first, as MLX reverses them), so a matrix
   weight is [out in]. Experimental, with gen.array/load-gguf (DESIGN
   §9). Written in plain Clojure: it requires nothing of gen.array."
  {:api :internal}
  (:require [clojure.string :as str]
            [gen.array.gguf.quants :as quants]
            [gen.array.host :as host]
            [gen.array.protocols :as p])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.util Collection]
           [java.util.concurrent ExecutionException ForkJoinPool Future]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.charset StandardCharsets]
           [java.nio.file Files LinkOption OpenOption Path StandardOpenOption]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; Types
;; ---------------------------------------------------------------------------

(def ggml-types
  "GGML's tensor type ids (ggml.h's enum ggml_type), as keywords."
  {0 :f32 1 :f16 2 :q4_0 3 :q4_1 6 :q5_0 7 :q5_1 8 :q8_0 9 :q8_1 10 :q2_k 11 :q3_k 12 :q4_k
   13 :q5_k 14 :q6_k 15 :q8_k 16 :iq2_xxs 17 :iq2_xs 18 :iq3_xxs 19 :iq1_s 20 :iq4_nl 21 :iq3_s
   22 :iq2_s 23 :iq4_xs 24 :i8 25 :i16 26 :i32 27 :i64 28 :f64 29 :iq1_m 30 :bf16 34 :tq1_0
   35 :tq2_0 39 :mxfp4})

(def loadable
  "The tensor types load-gguf reads: a plain type's dtype and bytes per
   value; a block type's block, :values values in :block bytes, and
   either the bits of the affine quantization it loads as (:bits, in
   groups of 32) or the dtype it is decoded to (:dtype, Q6_K's)."
  {:f32 {:dtype :float32 :bytes 4}
   :f16 {:dtype :float16 :bytes 2}
   :bf16 {:dtype :bfloat16 :bytes 2}
   :i32 {:dtype :int32 :bytes 4}
   :q8_0 {:values 32 :block 34 :bits 8}
   :q4_0 {:values 32 :block 18 :bits 4}
   :q4_1 {:values 32 :block 20 :bits 4}
   :q5_0 {:values 32 :block 22 :bits 5}
   :q5_1 {:values 32 :block 24 :bits 5}
   :q4_k {:values 256 :block 144 :bits 4}
   :q5_k {:values 256 :block 176 :bits 5}
   :q6_k {:values 256 :block 210 :dtype :float32}})

(def ^:const group-size
  "The group size of the affine quantization quantized tensors load as:
   a block of the 32-value types, a sub-block of Q4_K and Q5_K."
  32)

(defn quantized?
  "Is the tensor type t one that loads as a quantized triple?"
  [t]
  (contains? #{:q8_0 :q4_0 :q4_1 :q5_0 :q5_1 :q4_k :q5_k} t))

(defn mlx-reads?
  "Does MLX's GGUF loader read the tensor type t as load-gguf loads it?
   F32, F16, I32, Q8_0, Q4_0 and Q4_1: it turns the others into float16
   (a file holding one loads through the host reader on MLX)."
  [t]
  (contains? #{:f32 :f16 :i32 :q8_0 :q4_0 :q4_1} t))

(defn- invalid
  [path message data]
  (ex-info (str "load-gguf: " path ": " message) (merge {:reason :invalid-gguf :path (str path)} data)))

;; ---------------------------------------------------------------------------
;; The header
;; ---------------------------------------------------------------------------

(defn- path-of ^Path [path] (Path/of (str path) (make-array String 0)))

(defn- map-file
  "The bytes [offset, offset + n) of the file at path, little-endian."
  ^ByteBuffer [path ^long offset ^long n]
  (with-open [ch (FileChannel/open (path-of path) (into-array OpenOption [StandardOpenOption/READ]))]
    (doto (.map ch FileChannel$MapMode/READ_ONLY offset n)
      (.order ByteOrder/LITTLE_ENDIAN))))

(defn- u64
  "An unsigned 64-bit integer of the buffer as a host integer: a long, or a
   BigInt past Long/MAX_VALUE."
  [^ByteBuffer b]
  (let [v (.getLong b)]
    (if (neg? v) (bigint (Long/toUnsignedString v)) v)))

(defn- count-of
  "A count (u64) that must fit in an int's range: a header that claims
   more is corrupt."
  ^long [path ^ByteBuffer b what]
  (let [n (.getLong b)]
    (when-not (<= 0 n Integer/MAX_VALUE)
      (throw (invalid path (str "a count of " what " of " (Long/toUnsignedString n) " is not plausible") {:count n})))
    n))

(defn- read-string*
  [path ^ByteBuffer b]
  (let [n (count-of path b "bytes in a string")]
    (when (> n (.remaining b))
      (throw (invalid path "a string runs past the end of the file" {})))
    (let [bs (byte-array n)]
      (.get b bs)
      (String. bs StandardCharsets/UTF_8))))

(def value-types
  "GGUF's metadata value type ids, as keywords."
  {0 :uint8 1 :int8 2 :uint16 3 :int16 4 :uint32 5 :int32 6 :float32 7 :bool 8 :string 9 :array
   10 :uint64 11 :int64 12 :float64})

(defn- read-value
  "A metadata value of GGUF value type t: an integer as a long (a u64 past
   Long/MAX_VALUE as a BigInt), a float32 or float64 as a double, a bool,
   a string, or an array as a vector of these."
  [path ^ByteBuffer b ^long t]
  (case t
    0 (long (bit-and (.get b) 0xff))
    1 (long (.get b))
    2 (long (bit-and (.getShort b) 0xffff))
    3 (long (.getShort b))
    4 (Integer/toUnsignedLong (.getInt b))
    5 (long (.getInt b))
    6 (double (.getFloat b))
    7 (not (zero? (.get b)))
    8 (read-string* path b)
    9 (let [et (Integer/toUnsignedLong (.getInt b))
            n (count-of path b "array elements")]
        (when (= 9 et)
          (throw (invalid path "an array of arrays in the metadata is not supported" {})))
        (loop [i 0 out (transient [])]
          (if (< i n)
            (recur (inc i) (conj! out (read-value path b et)))
            (persistent! out))))
    10 (u64 b)
    11 (.getLong b)
    12 (.getDouble b)
    (throw (invalid path (str "unknown metadata value type " t) {:value-type t}))))

(defn- tensor-bytes
  "The bytes a tensor of type t and shape holds, or nil for a type
   load-gguf does not read."
  [t shape]
  (let [n (reduce * 1 shape)
        {:keys [bytes block values]} (loadable t)]
    (cond bytes (* (long bytes) (long n))
          block (* (long block) (quot (long n) (long values))))))

(defn header
  "The header of the GGUF file at path: {:path :version :metadata
   :value-types :keys :tensors}, :metadata a map of keys (strings) to host
   values, :value-types each key's GGUF value type (a keyword of
   value-types, or [:array element-type]), :keys the keys in the file's
   order, :tensors a
   vector of {:name :type :shape :offset :bytes} in the file's order:
   :type a keyword of ggml-types, :shape gen.array's (outermost first),
   :offset the data's absolute position in the file, and :bytes its size
   (nil for a type load-gguf does not read). Loud (:invalid-gguf) on a
   file that is no GGUF of version 2 or 3, or whose header is cut short
   or claims data past the file's end."
  [path]
  (let [p (path-of path)]
    (when-not (Files/isRegularFile p (make-array LinkOption 0))
      (throw (ex-info (str "load-gguf: no file at " path) {:reason :no-file :path (str path)})))
    (let [size (Files/size p)
          ^ByteBuffer b (map-file path 0 (min size Integer/MAX_VALUE))]
      (try
        (when (or (< size 24) (not= 0x46554747 (.getInt b)))
          (throw (invalid path "not a GGUF file (no GGUF magic)" {})))
        (let [version (Integer/toUnsignedLong (.getInt b))
              _ (when-not (#{2 3} version)
                  (throw (invalid path (str "GGUF version " version " is not supported (2 and 3 are)") {:version version})))
              n-tensors (count-of path b "tensors")
              n-kv (count-of path b "metadata keys")
              [metadata types order] (loop [i 0 m (transient {}) ts (transient {}) ks (transient [])]
                                       (if (< i n-kv)
                                         (let [k (read-string* path b)
                                               t (Integer/toUnsignedLong (.getInt b))
                                               ;; an array's element type follows
                                               et (when (= 9 t) (Integer/toUnsignedLong (.getInt b (.position b))))
                                               v (read-value path b t)]
                                           (recur (inc i) (assoc! m k v)
                                                  (assoc! ts k (if et [:array (value-types et et)] (value-types t t)))
                                                  (conj! ks k)))
                                         [(persistent! m) (persistent! ts) (persistent! ks)]))
              infos (loop [i 0 out (transient [])]
                      (if (< i n-tensors)
                        (let [nm (read-string* path b)
                              nd (Integer/toUnsignedLong (.getInt b))
                              _ (when (> nd 4) (throw (invalid path (str "tensor " nm " has " nd " dimensions") {:tensor nm})))
                              dims (vec (repeatedly nd #(.getLong b)))
                              t (Integer/toUnsignedLong (.getInt b))
                              off (.getLong b)]
                          (recur (inc i) (conj! out {:name nm :dims dims :type (ggml-types t (keyword (str "type-" t)))
                                                     :offset off})))
                        (persistent! out)))
              align (long (get metadata "general.alignment" 32))
              data-start (* align (quot (+ (.position b) (dec align)) align))
              tensors (mapv (fn [{:keys [name dims type offset]}]
                              (let [shape (vec (reverse dims))
                                    n (tensor-bytes type shape)
                                    at (+ data-start (long offset))]
                                (when (or (neg? (long offset)) (some neg? dims)
                                          (and n (> (+ at (long n)) size)))
                                  (throw (invalid path (str "tensor " name "'s data lies past the end of the file") {:tensor name})))
                                {:name name :type type :shape shape :offset at :bytes n}))
                            infos)]
          {:path (str path) :version version :metadata metadata :value-types types :keys order :tensors tensors})
        (catch java.nio.BufferUnderflowException _
          (throw (invalid path "the header is cut short" {})))))))

;; ---------------------------------------------------------------------------
;; Split files
;; ---------------------------------------------------------------------------

(def ^:private split-pattern #"^(.*)-(\d{5})-of-(\d{5})\.gguf$")

(defn- split-count ^long [{:keys [metadata]}]
  (long (get metadata "split.count" 1)))

(defn headers
  "The headers of every file of the model whose first (or only) file is
   at path, in order: one, or the split's files <prefix>-0000i-of-0000N.gguf
   named as llama.cpp's gguf-split names them, each holding split.no i - 1
   and split.count N. Loud (:invalid-gguf) on a split's file missing or
   out of place, or a path that is not the split's first file (llama.cpp
   loads a split model from its first file too)."
  [path]
  (let [first-header (header path)
        n (split-count first-header)]
    (if (= 1 n)
      [first-header]
      (let [[_ prefix no of] (re-find split-pattern (str path))]
        (when-not prefix
          (throw (invalid path (str "a split model's file (split.count " n ") must be named <prefix>-00001-of-"
                                    (format "%05d" n) ".gguf") {:split-count n})))
        (when-not (and (= 1 (Long/parseLong no)) (= n (Long/parseLong of))
                       (= 0 (long (get-in first-header [:metadata "split.no"] -1))))
          (throw (invalid path (str "load a split model from its first file, " prefix (format "-00001-of-%05d.gguf" n))
                          {:split-count n})))
        (into [first-header]
              (for [i (range 1 n)
                    :let [i (long i)
                          p (format "%s-%05d-of-%05d.gguf" prefix (inc i) n)
                          h (try (header p)
                                 (catch clojure.lang.ExceptionInfo e
                                   (if (= :no-file (:reason (ex-data e)))
                                     (throw (invalid path (str "the split's file " p " is missing") {:split-count n :missing p}))
                                     (throw e))))]]
                (do (when-not (and (= i (long (get-in h [:metadata "split.no"] -1))) (= n (split-count h)))
                      (throw (invalid p (str "it is not file " (inc i) " of " n " of the split") {:split-count n})))
                    h)))))))

(defn check-tensors
  "The tensors of a model's headers, each checked loadable: one of the
   types load-gguf reads (else :unsupported, naming it), a quantized one
   named <base>.weight with nothing else named <base>.scales or
   <base>.biases (the names its triple loads under on MLX), a block
   type's innermost axis a multiple of its block (32 values, or 256 for
   the K-quants), no name twice (:invalid-gguf), and as many as a split's
   first file says (split.tensors.count)."
  [headers]
  (let [tensors (vec (mapcat :tensors headers))
        path (:path (first headers))
        names (frequencies (map :name tensors))
        expected (get-in (first headers) [:metadata "split.tensors.count"])]
    (when-let [twice (seq (keep (fn [[k v]] (when (> (long v) 1) k)) names))]
      (throw (invalid path (str "tensors named more than once: " (str/join ", " (sort twice))) {:tensors (vec twice)})))
    (when (and expected (not= (long expected) (count tensors)))
      (throw (invalid path (str "the split holds " (count tensors) " tensors where its first file says " expected)
                      {:split-tensors expected :tensors (count tensors)})))
    (doseq [{:keys [name type shape]} tensors]
      (when-not (loadable type)
        (throw (ex-info (str "load-gguf: " path ": tensor " name " is " (str/upper-case (clojure.core/name type))
                             ", a type not supported (F32, F16, BF16, I32, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K,"
                             " Q5_K and Q6_K are)")
                        {:reason :unsupported :path path :tensor name :type type})))
      (when (quantized? type)
        (when-not (str/ends-with? name ".weight")
          (throw (invalid path (str "the quantized tensor " name " is not named <base>.weight") {:tensor name})))
        (let [base (subs name 0 (- (count name) (count ".weight")))]
          (when (some names [(str base ".scales") (str base ".biases")])
            (throw (invalid path (str "the quantized tensor " name " shares its base with a tensor named "
                                      base ".scales or .biases") {:tensor name})))))
      (when-let [values (:values (loadable type))]
        (when-not (and (seq shape) (zero? (rem (long (peek shape)) (long values))))
          (throw (invalid path (str "the " (str/upper-case (clojure.core/name type)) " tensor " name
                                    "'s innermost axis is not a multiple of " values ": " shape)
                          {:tensor name :shape shape})))))
    tensors))

;; ---------------------------------------------------------------------------
;; Tensors on the host: the JVM backend's loader, and MLX's for the types its
;; own loader does not read. Every value is the file's exactly: a block's
;; codes repacked as MLX's affine layout packs them (value j of a row at bits
;; [j·b, (j+1)·b) of its words, little-endian), its scale and bias computed
;; in float32 as llama.cpp's dequantize_row_* computes them, where each
;; product is exact
;; ---------------------------------------------------------------------------

(defn- f16 ^double [^ByteBuffer b ^long at]
  (double (Float/float16ToFloat (.getShort b (int at)))))

(defn- byte-at ^long [^ByteBuffer b ^long at] (bit-and (long (.get b (int at))) 0xff))

(defn- nibbles
  "The low nibbles of the 8 bytes of x (byte i at bits 8i), as one word:
   byte i's at bits 4i."
  ^long [^long x]
  (let [y (bit-and x 0x0F0F0F0F0F0F0F0F)
        y (bit-and (bit-or y (unsigned-bit-shift-right y 4)) 0x00FF00FF00FF00FF)
        y (bit-and (bit-or y (unsigned-bit-shift-right y 8)) 0x0000FFFF0000FFFF)]
    (bit-and (bit-or y (unsigned-bit-shift-right y 16)) 0xFFFFFFFF)))

(defn- fives
  "The 5-bit values in the 8 bytes of x (byte i at bits 8i), as 40 bits:
   byte i's at bits 5i."
  ^long [^long x]
  (let [y (bit-or (bit-and x 0x001F001F001F001F) (unsigned-bit-shift-right (bit-and x 0x1F001F001F001F00) 3))
        y (bit-or (bit-and y 0x000003FF000003FF) (unsigned-bit-shift-right (bit-and y 0x03FF000003FF0000) 6))]
    (bit-or (bit-and y 0xFFFFF) (unsigned-bit-shift-right (bit-and y 0x000FFFFF00000000) 12))))

(defmacro ^:private put-fives!
  "A group's 32 values of 5 bits, four longs of 8 (byte i of each the
   value, as fives takes them), as the 5 words at w of words."
  [words w x0 x1 x2 x3]
  `(let [^"[I" ws# ~words
         w# (long ~w)
         t0# (fives ~x0) t1# (fives ~x1) t2# (fives ~x2) t3# (fives ~x3)]
     (aset ws# w# (unchecked-int t0#))
     (aset ws# (+ w# 1) (unchecked-int (bit-or (unsigned-bit-shift-right t0# 32) (bit-shift-left t1# 8))))
     (aset ws# (+ w# 2) (unchecked-int (bit-or (unsigned-bit-shift-right t1# 24) (bit-shift-left t2# 16))))
     (aset ws# (+ w# 3) (unchecked-int (bit-or (unsigned-bit-shift-right t2# 16) (bit-shift-left t3# 24))))
     (aset ws# (+ w# 4) (unchecked-int (unsigned-bit-shift-right t3# 8)))))

(def ^:private ^"[J" spread
  "Bit i of a byte at bit 8i of a long, for every byte."
  (let [t (long-array 256)]
    (dotimes [v 256]
      (aset t v (long (reduce (fn [^long acc ^long i] (if (bit-test v i) (bit-or acc (bit-shift-left 1 (* 8 i))) acc))
                              0 (range 8)))))
    t))

(defn- high-bits
  "The bits 8k to 8k + 7 of the word qh, each at bit 4 of its byte: the
   fifth bits of a Q5_0 or Q5_1 block's values 8k to 8k + 7."
  ^long [^long qh ^long k]
  (bit-shift-left (aget spread (bit-and (unsigned-bit-shift-right qh (* 8 k)) 0xff)) 4))

(defmacro ^:private each-block
  "body for each block i of [from, to)."
  [[i from to] & body]
  `(let [to# (long ~to)]
     (loop [~i (long ~from)]
       (when (< ~i to#)
         ~@body
         (recur (inc ~i))))))

(def ^:private chunk-bytes
  "The least bytes of a tensor a task of the host's decoding reads (a var
   the tests redefine, to decode their small tensors in ranges)."
  (* 4 1024 1024))

(defn- decode!
  "(f from to) for every block of the n blocks (of size bytes each) of a
   tensor, in contiguous ranges: on the common fork-join pool, a range of
   at least chunk-bytes a task, when the tensor has several; each block's
   values are written by one call, to their own places."
  [^long n ^long size f]
  (let [pool (ForkJoinPool/commonPool)
        parts (min (long (.getParallelism pool)) (quot (* n size) (long chunk-bytes)))]
    (if (<= parts 1)
      (f 0 n)
      (let [step (quot (+ n (dec parts)) parts)
            tasks (vec (for [p (range parts)
                             :let [from (* (long p) step)]
                             :when (< from n)]
                         ^Callable (fn [] (f from (min n (+ from step))))))]
        (doseq [^Future fut (.invokeAll pool ^Collection tasks)]
          (try (.get fut)
               (catch ExecutionException e (throw (.getCause e)))))))))

(defn- blocks-32!
  "The blocks of 32 values of type t (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1), each
   one group of MLX's affine quantization: its words; its scale d; its
   bias, Q4_1's and Q5_1's m, or -128·d, -8·d, -16·d in float32 (exact),
   where MLX's loader computes Q8_0's and Q4_0's in float16, which
   overflows for d past 511.75 (they are equal where it does not). The
   codes: Q8_0's int8 shifted to uint8 (x ^ 0x80), as MLX's loader
   shifts them; a 4-bit type's nibbles, values 0 to 15 the low nibbles
   of its 16 bytes, 16 to 31 the high ones; Q5_0's and Q5_1's the same
   with their fifth bits from the block's word qh (value j's bit j)."
  [^ByteBuffer b t ^ints words ^floats scales ^floats biases from to]
  (let [from (long from) to (long to)]
    (case t
      :q8_0 (each-block [i from to]
              (let [at (* i 34)
                    d (f16 b at)]
                (aset scales i (unchecked-float d))
                (aset biases i (unchecked-float (* -128.0 d)))
                (dotimes [k 8]
                  (aset words (+ (* 8 i) k) (unchecked-int (bit-xor (.getInt b (int (+ at 2 (* 4 k)))) (unchecked-int 0x80808080)))))))
      (:q4_0 :q4_1) (let [q4-1? (= t :q4_1)
                          size (if q4-1? 20 18)]
                      (each-block [i from to]
                        (let [at (* i size)
                              d (f16 b at)
                              qs (+ at (if q4-1? 4 2))
                              x0 (.getLong b (int qs))
                              x1 (.getLong b (int (+ qs 8)))
                              w (* 4 i)]
                          (aset scales i (unchecked-float d))
                          (aset biases i (unchecked-float (if q4-1? (f16 b (+ at 2)) (* -8.0 d))))
                          (aset words w (unchecked-int (nibbles x0)))
                          (aset words (+ w 1) (unchecked-int (nibbles x1)))
                          (aset words (+ w 2) (unchecked-int (nibbles (unsigned-bit-shift-right x0 4))))
                          (aset words (+ w 3) (unchecked-int (nibbles (unsigned-bit-shift-right x1 4)))))))
      (:q5_0 :q5_1) (let [q5-1? (= t :q5_1)
                          size (if q5-1? 24 22)]
                      (each-block [i from to]
                        (let [at (* i size)
                              d (f16 b at)
                              qh-at (+ at (if q5-1? 4 2))
                              qh (Integer/toUnsignedLong (.getInt b (int qh-at)))
                              x0 (.getLong b (int (+ qh-at 4)))
                              x1 (.getLong b (int (+ qh-at 12)))
                              lo 0x0F0F0F0F0F0F0F0F]
                          (aset scales i (unchecked-float d))
                          (aset biases i (unchecked-float (if q5-1? (f16 b (+ at 2)) (* -16.0 d))))
                          (put-fives! words (* 5 i)
                                      (bit-or (bit-and x0 lo) (high-bits qh 0))
                                      (bit-or (bit-and x1 lo) (high-bits qh 1))
                                      (bit-or (bit-and (unsigned-bit-shift-right x0 4) lo) (high-bits qh 2))
                                      (bit-or (bit-and (unsigned-bit-shift-right x1 4) lo) (high-bits qh 3)))))))))

(defn- scale-min
  "Sub-block j's scale and min (6 bits each) from a Q4_K or Q5_K
   super-block's 12 scale bytes at at, as llama.cpp's get_scale_min_k4
   reads them: the scale in bits 0-7 of the result, the min in 8-15."
  ^long [^ByteBuffer b ^long at ^long j]
  (if (< j 4)
    (bit-or (bit-and (byte-at b (+ at j)) 63)
            (bit-shift-left (bit-and (byte-at b (+ at j 4)) 63) 8))
    (let [s (byte-at b (+ at j 4))]
      (bit-or (bit-or (bit-and s 0xF) (bit-shift-left (bit-shift-right (byte-at b (+ at (- j 4))) 6) 4))
              (bit-shift-left (bit-or (bit-shift-right s 4) (bit-shift-left (bit-shift-right (byte-at b (+ at j)) 6) 4)) 8)))))

(defmacro ^:private sub-blocks!
  "Super-block i's (at at in b) 8 groups' scales d·sc and biases -(dmin·m),
   the products in float32 as llama.cpp's dequantize_row_q4_K and _q5_K
   compute them: exact, an 11-bit significand times a 6-bit integer."
  [b at i scales biases]
  `(let [^ByteBuffer b# ~b
         at# (long ~at)
         g# (* 8 (long ~i))
         ^"[F" scales# ~scales
         ^"[F" biases# ~biases
         d# (f16 b# at#)
         dmin# (f16 b# (+ at# 2))]
     (dotimes [j# 8]
       (let [sm# (scale-min b# (+ at# 4) j#)]
         (aset scales# (+ g# j#) (unchecked-float (* d# (double (bit-and sm# 0xff)))))
         (aset biases# (+ g# j#) (unchecked-float (- (* dmin# (double (bit-shift-right sm# 8))))))))))

(defn- q4-k!
  "Q4_K's super-blocks (256 values: 8 sub-blocks of 32, d, dmin, the
   sub-blocks' 6-bit scales and mins, the codes) as MLX's affine
   quantization at 4 bits, a sub-block a group: in chunk c (64 values,
   32 bytes), sub-block 2c's codes the low nibbles, 2c + 1's the high
   ones."
  [^ByteBuffer b ^ints words ^floats scales ^floats biases from to]
  (each-block [i from to]
    (let [at (* i 144)]
      (sub-blocks! b at i scales biases)
      (dotimes [c 4]
        (dotimes [k 4]
          (let [x (.getLong b (int (+ at 16 (* 32 c) (* 8 k))))
                w (+ (* 32 i) (* 8 c) k)]
            (aset words w (unchecked-int (nibbles x)))
            (aset words (+ w 4) (unchecked-int (nibbles (unsigned-bit-shift-right x 4))))))))))

(defn- five
  "The 5-bit values of a Q5_K sub-block s's 8 values in x (the bytes of
   their nibbles, the low ones at shift 0, the high ones at 4) and h
   (their bytes of qh, bit s the fifth bit), byte i of the result value
   i's."
  ^long [^long x ^long h ^long s ^long shift]
  (bit-or (bit-and (unsigned-bit-shift-right x shift) 0x0F0F0F0F0F0F0F0F)
          (bit-shift-left (bit-and (unsigned-bit-shift-right h s) 0x0101010101010101) 4)))

(defn- q5-k!
  "Q5_K's super-blocks as MLX's affine quantization at 5 bits, a
   sub-block a group: Q4_K's layout, and the fifth bit of sub-block s's
   value l is bit s of byte l of the super-block's 32 bytes qh."
  [^ByteBuffer b ^ints words ^floats scales ^floats biases from to]
  (each-block [i from to]
    (let [at (* i 176)
          h0 (.getLong b (int (+ at 16))) h1 (.getLong b (int (+ at 24)))
          h2 (.getLong b (int (+ at 32))) h3 (.getLong b (int (+ at 40)))]
      (sub-blocks! b at i scales biases)
      (dotimes [c 4]
        (let [qs (+ at 48 (* 32 c))
              x0 (.getLong b (int qs)) x1 (.getLong b (int (+ qs 8)))
              x2 (.getLong b (int (+ qs 16))) x3 (.getLong b (int (+ qs 24)))
              s0 (* 2 c)
              s1 (inc s0)]
          (put-fives! words (+ (* 40 i) (* 5 s0)) (five x0 h0 s0 0) (five x1 h1 s0 0) (five x2 h2 s0 0) (five x3 h3 s0 0))
          (put-fives! words (+ (* 40 i) (* 5 s1)) (five x0 h0 s1 4) (five x1 h1 s1 4) (five x2 h2 s1 4) (five x3 h3 s1 4)))))))

(defn- q6-k!
  "Q6_K's super-blocks (256 values: 16 sub-blocks of 16, each an int8
   scale over the super-block's d; codes of 6 bits, 4 low in ql and 2
   high in qh) decoded, as llama.cpp's dequantize_row_q6_K decodes them:
   d·sc·(q - 32) in float32, exact (11, 8 and 6 significant bits)."
  [^ByteBuffer b ^floats out from to]
  (let [ds (double-array 16)]
    (each-block [i from to]
      (let [at (* i 210)
            d (f16 b (+ at 208))]
        (dotimes [k 16]
          (aset ds k (* d (double (.get b (int (+ at 192 k)))))))
        (dotimes [h 2]
          (let [ql (+ at (* 64 h))
                qh (+ at 128 (* 32 h))
                y (+ (* 256 i) (* 128 h))]
            (dotimes [l 32]
              (let [s (+ (* 8 h) (quot l 16))
                    hb (byte-at b (+ qh l))
                    a (byte-at b (+ ql l))
                    c (byte-at b (+ ql l 32))]
                (aset out (+ y l) (unchecked-float (* (aget ds s) (double (- (bit-or (bit-and a 0xF) (bit-shift-left (bit-and hb 3) 4)) 32)))))
                (aset out (+ y l 32) (unchecked-float (* (aget ds (+ s 2)) (double (- (bit-or (bit-and c 0xF) (bit-shift-left (bit-and (bit-shift-right hb 2) 3) 4)) 32)))))
                (aset out (+ y l 64) (unchecked-float (* (aget ds (+ s 4)) (double (- (bit-or (bit-shift-right a 4) (bit-shift-left (bit-and (bit-shift-right hb 4) 3) 4)) 32)))))
                (aset out (+ y l 96) (unchecked-float (* (aget ds (+ s 6)) (double (- (bit-or (bit-shift-right c 4) (bit-shift-left (bit-and (bit-shift-right hb 6) 3) 4)) 32)))))))))))))

(defn host-tensor
  "The tensor of the file at path described by info (header's), decoded
   on the host into primitive arrays:
   - a plain type, and Q6_K: {:dtype :shape :data}, data a float[]
     (:float32: F32's values, and Q6_K's, d·sc·(q - 32) in float32 as
     llama.cpp computes them, exactly), a short[] (:float16 and
     :bfloat16: F16's and BF16's bits) or an int[] (:int32);
   - a quantized type: {:quantized [words scales biases] :shapes [words'
     scales' biases'] :bits}, MLX's affine quantization in groups of 32:
     words an int[] (uint32 bits), scales and biases float[]s.
   Loud (:unsupported) for a tensor of more than 2^31 - 1 values, or of
   bytes."
  [path {:keys [type shape offset bytes]}]
  (let [n (reduce * 1 shape)]
    (when (or (> (long n) Integer/MAX_VALUE) (> (long bytes) Integer/MAX_VALUE))
      (throw (ex-info (str "load-gguf: a tensor of " n " values (" bytes " bytes) is too large to read on the host")
                      {:reason :unsupported :path (str path) :shape shape})))
    (let [^ByteBuffer b (map-file path offset bytes)
          n (long n)]
      (if (quantized? type)
        (let [bits (long (:bits (loadable type)))
              k (long (peek shape))
              words (int-array (quot (* n bits) 32))
              scales (float-array (quot n group-size))
              biases (float-array (quot n group-size))]
          (let [{:keys [block values]} (loadable type)
                blocks (quot n (long values))]
            (decode! blocks block (case type
                                    :q4_k #(q4-k! b words scales biases %1 %2)
                                    :q5_k #(q5-k! b words scales biases %1 %2)
                                    #(blocks-32! b type words scales biases %1 %2))))
          {:quantized [words scales biases]
           :shapes [(conj (pop shape) (quot (* k bits) 32))
                    (conj (pop shape) (quot k group-size))
                    (conj (pop shape) (quot k group-size))]
           :bits bits})
        {:dtype (:dtype (loadable type))
         :shape shape
         :data (case type
                 :f32 (let [out (float-array n)] (.get (.asFloatBuffer b) out) out)
                 (:f16 :bf16) (let [out (short-array n)] (.get (.asShortBuffer b) out) out)
                 :i32 (let [out (int-array n)] (.get (.asIntBuffer b) out) out)
                 :q6_k (let [out (float-array n)] (decode! (quot n 256) 210 #(q6-k! b out %1 %2)) out))}))))

(defn- doubles-of
  "A host array's values as doubles: an int's as an integer, a 16-bit
   float's bits (dtype's: :float16 or :bfloat16) as their value
   (gen.array.host/halves->floats)."
  ^doubles [data dtype]
  (condp instance? data
    float/1 (let [^floats a data out (double-array (alength a))]
              (dotimes [i (alength a)] (aset out i (double (aget a i))))
              out)
    int/1 (let [^ints a data out (double-array (alength a))]
            (dotimes [i (alength a)] (aset out i (double (aget a i))))
            out)
    short/1 (doubles-of (host/halves->floats data dtype) :float32)))

(defn- words-of
  "The uint32 words of an int[] (their bits), as doubles."
  ^doubles [^ints a]
  (let [out (double-array (alength a))]
    (dotimes [i (alength a)] (aset out i (double (Integer/toUnsignedLong (aget a i)))))
    out))

(defn read-tensor
  "The tensor of the file at path described by info (header's), on the
   host as doubles, the JVM backend's storage: {:dtype :shape :data} for
   a plain type and Q6_K (data a double[] of the values, row-major), or
   {:quantized [words scales biases] :shapes [words' scales' biases']
   :bits} for a quantized one (MLX's affine layout in groups of 32: words
   uint32 values, scales and biases float32 values, as doubles)."
  [path info]
  (let [{:keys [quantized data dtype] :as t} (host-tensor path info)]
    (if quantized
      (let [[words scales biases] quantized]
        (assoc t :quantized [(words-of words) (doubles-of scales :float32) (doubles-of biases :float32)]))
      (assoc t :data (doubles-of data dtype)))))

(defn host-values
  "The values of the tensor of the file at path described by info
   (header's), row-major, as a new float[]: a float type's, or a quantized
   one's dequantized as llama.cpp's dequantize_row_* computes them (each
   value scale·q + bias in float32, the product exact), so exactly its
   values. Loud (:unsupported) for an I32 tensor, which holds no floats."
  ^floats [path {:keys [name type] :as info}]
  (let [{:keys [dtype data quantized bits]} (host-tensor path info)]
    (cond
      quantized
      (let [[^ints words ^floats scales ^floats biases] quantized
            bits (long bits)
            mask (dec (bit-shift-left 1 bits))
            n (* (alength scales) (long group-size))
            out (float-array n)]
        (dotimes [g (alength scales)]
          (let [w0 (* g bits)
                s (double (aget scales g))
                c (double (aget biases g))]
            (dotimes [t group-size]
              (let [bit (* t bits)
                    w (+ w0 (quot bit 32))
                    lo (Integer/toUnsignedLong (aget words w))
                    hi (if (< (+ w 1) (+ w0 bits)) (Integer/toUnsignedLong (aget words (+ w 1))) 0)
                    q (bit-and (unsigned-bit-shift-right (bit-or lo (bit-shift-left hi 32)) (rem bit 32)) mask)]
                (aset out (+ (* g (long group-size)) t) (unchecked-float (+ (* s (double q)) c)))))))
        out)
      (= :float32 dtype) data
      (#{:float16 :bfloat16} dtype) (host/halves->floats data dtype)
      :else
      (throw (ex-info (str "host-values: tensor " name " is " (clojure.core/name type) ", which holds no floats")
                      {:reason :unsupported :tensor name :type type})))))

;; ---------------------------------------------------------------------------
;; Loading
;; ---------------------------------------------------------------------------

(defn load-gguf
  "gen.array/load-gguf's body, on backend b: every file's tensors by the
   backend (p/-load-gguf), evaluated, with the first file's metadata."
  [b path]
  (let [hs (headers path)
        infos (check-tensors hs)
        loaded (into {} (mapcat (fn [h] (p/load-gguf b h))) hs)
        tensors (into {}
                      (map (fn [{:keys [name type]}]
                             (let [v (or (get loaded name)
                                         (throw (ex-info (str "load-gguf: the " (clojure.core/name (p/-name b))
                                                              " backend did not load tensor " name)
                                                         {:reason :invalid-gguf :path (str path) :tensor name})))]
                               [name (if (quantized? type)
                                       (let [[wq scales biases] v]
                                         {:wq wq :scales scales :biases biases
                                          :bits (:bits (loadable type)) :group-size group-size})
                                       v)])))
                      infos)]
    (p/-eval! b (mapcat (fn [v] (if (map? v) [(:wq v) (:scales v) (:biases v)] [v])) (vals tensors)))
    {:metadata (:metadata (first hs))
     :tensors tensors
     :types (into {} (map (juxt :name :type)) infos)
     :files (mapv :path hs)}))

;; ---------------------------------------------------------------------------
;; Writing (gen.llm.lora's adapters for llama.cpp, gen.llm.convert's models)
;; ---------------------------------------------------------------------------

(def ^:private value-type-ids (into {} (map (fn [[id t]] [t id])) value-types))

(def ^:private scalar-bytes
  {:uint8 1 :int8 1 :uint16 2 :int16 2 :uint32 4 :int32 4 :float32 4 :bool 1 :uint64 8 :int64 8 :float64 8})

(defn- bad-value [k t v]
  (ex-info (str "write-gguf: the metadata value " (pr-str v) " of " k " is not a " (pr-str t))
           {:reason :invalid-args :key k :value-type t :value v}))

(defn- integer-in? [v lo hi]
  (and (integer? v) (<= (bigint lo) (bigint v) (bigint hi))))

(defn- put-scalar!
  "A metadata scalar v of value type t into b, checked to fit."
  [^ByteBuffer b k t v]
  (let [ok? (case t
              :uint8 (integer-in? v 0 255) :int8 (integer-in? v -128 127)
              :uint16 (integer-in? v 0 65535) :int16 (integer-in? v -32768 32767)
              :uint32 (integer-in? v 0 4294967295) :int32 (integer-in? v Integer/MIN_VALUE Integer/MAX_VALUE)
              :uint64 (integer-in? v 0 18446744073709551615N) :int64 (integer-in? v Long/MIN_VALUE Long/MAX_VALUE)
              (:float32 :float64) (number? v)
              :bool (boolean? v))]
    (when-not ok? (throw (bad-value k t v)))
    (case t
      (:uint8 :int8) (.put b (unchecked-byte (long v)))
      (:uint16 :int16) (.putShort b (unchecked-short (long v)))
      (:uint32 :int32) (.putInt b (unchecked-int (long v)))
      :uint64 (.putLong b (.longValue ^Number (bigint v)))
      :int64 (.putLong b (long v))
      :float32 (.putFloat b (float v))
      :float64 (.putDouble b (double v))
      :bool (.put b (byte (if v 1 0))))))

(defn- string-bytes ^bytes [s] (.getBytes (str s) StandardCharsets/UTF_8))

(defn- value-size
  "The bytes a metadata value v of value type t takes."
  ^long [k t v]
  (cond (= :string t) (if (string? v) (+ 8 (alength (string-bytes v))) (throw (bad-value k t v)))
        (vector? t) (let [et (second t)]
                      (when-not (and (= :array (first t)) (contains? value-type-ids et) (not= :array et) (sequential? v))
                        (throw (bad-value k t v)))
                      (+ 12 (long (reduce + 0 (map #(value-size k et %) v)))))
        (scalar-bytes t) (long (scalar-bytes t))
        :else (throw (ex-info (str "write-gguf: " (pr-str t) " is not a GGUF value type (" k ")")
                              {:reason :invalid-args :key k :value-type t}))))

(defn- put-string! [^ByteBuffer b s]
  (let [bs (string-bytes s)]
    (.putLong b (alength bs))
    (.put b bs)))

(defn- put-value! [^ByteBuffer b k t v]
  (cond (= :string t) (put-string! b v)
        (vector? t) (let [et (second t)]
                      (.putInt b (unchecked-int (long (value-type-ids et))))
                      (.putLong b (count v))
                      (doseq [x v] (put-value! b k et x)))
        :else (put-scalar! b k t v)))

(defn- type-id ^long [t] (long (value-type-ids (if (vector? t) :array t))))

(defn- padded ^long [^long n ^long align] (* align (quot (+ n (dec align)) align)))

(defn- writable-bytes
  "The bytes a tensor {:type :shape} takes in the file."
  ^long [{:keys [type shape]}]
  (quants/tensor-bytes type (reduce * 1 shape)))

(defn- check-data
  "data, tensor t's values: a float[] of its size."
  ^floats [{:keys [name shape]} data]
  (let [n (reduce * 1 shape)]
    (when-not (and (instance? (Class/forName "[F") data) (= (long n) (alength ^floats data)))
      (throw (ex-info (str "write-gguf: tensor " name "'s data is a float[] of its " n " values")
                      {:reason :invalid-args :tensor name :shape shape})))
    data))

(defn- check-tensor [{:keys [name type shape data bytes] :as t}]
  (when-not (and (string? name) (seq name))
    (throw (ex-info (str "write-gguf: a tensor's name is a non-empty string, not " (pr-str name)) {:reason :invalid-args :tensor name})))
  (when-not (quants/types type)
    (throw (ex-info (str "write-gguf: tensor " name " is " (pr-str type) ", a type not written ("
                         (str/join ", " (map pr-str (sort (keys quants/types)))) " are)")
                    {:reason :unsupported :tensor name :type type})))
  (when-not (and (vector? shape) (<= 1 (count shape) 4) (every? pos-int? shape))
    (throw (ex-info (str "write-gguf: tensor " name "'s shape is 1 to 4 positive sizes, not " (pr-str shape))
                    {:reason :invalid-args :tensor name :shape shape})))
  (let [values (long (:values (quants/types type)))]
    (when-not (zero? (rem (long (peek shape)) values))
      (throw (ex-info (str "write-gguf: tensor " name "'s innermost axis (" (peek shape) ") is not a multiple of "
                           (clojure.core/name type) "'s block of " values " values")
                      {:reason :invalid-args :tensor name :shape shape :type type}))))
  (when-not (= 1 (count (filter some? [data bytes])))
    (throw (ex-info (str "write-gguf: tensor " name " has :data (its values) or :bytes (its encoding), one of them")
                    {:reason :invalid-args :tensor name})))
  (when (and data (= :i32 type))
    (throw (ex-info (str "write-gguf: tensor " name " is I32, which is copied (:bytes), not encoded from :data")
                    {:reason :unsupported :tensor name :type type})))
  (when (and data (not (fn? data)))
    (check-data t data))
  t)

(defn- tensor-buffer
  "Tensor t's bytes in the file, from :bytes (a byte[] or a ByteBuffer, or
   a function of no arguments giving one) or from :data (a float[], or a
   function giving one), encoded as its type (gen.array.gguf.quants)."
  ^ByteBuffer [{:keys [name type data bytes] :as t}]
  (if bytes
    (let [v (if (fn? bytes) (bytes) bytes)
          ^ByteBuffer b (cond (bytes? v) (ByteBuffer/wrap ^bytes v)
                              (instance? ByteBuffer v) (.duplicate ^ByteBuffer v)
                              :else (throw (ex-info (str "write-gguf: tensor " name "'s bytes are a byte[] or a ByteBuffer")
                                                    {:reason :invalid-args :tensor name})))]
      (when-not (= (writable-bytes t) (long (.remaining b)))
        (throw (ex-info (str "write-gguf: tensor " name "'s bytes are " (.remaining b) ", not its " (writable-bytes t))
                        {:reason :invalid-args :tensor name :bytes (.remaining b) :expected (writable-bytes t)})))
      b)
    (let [x (check-data t (if (fn? data) (data) data))]
      (quants/encode type x))))

(defn- alignment
  "The data's alignment metadata asks for (general.alignment, a uint32), or
   32, GGUF's default."
  ^long [metadata]
  (if-let [[_ t v] (first (filter #(= "general.alignment" (first %)) metadata))]
    (do (when-not (and (= :uint32 t) (pos-int? v) (zero? (bit-and (long v) (dec (long v)))) (zero? (rem (long v) 8)))
          (throw (ex-info (str "write-gguf: general.alignment is a uint32 power of 2 of at least 8, not " (pr-str [t v]))
                          {:reason :invalid-args :key "general.alignment" :value v})))
        (long v))
    32))

(defn write-gguf
  "Writes a GGUF file (version 3) at path, as llama.cpp's gguf-py writes
   one: the header, the metadata entries in their order, the tensors'
   descriptions, then each tensor's data, every part aligned to 32 bytes
   (or general.alignment's). metadata is a sequence of [key value-type
   value] entries, value-type a keyword of value-types (:string,
   :float32, :uint32, …) or [:array element-type]; tensors a sequence of
   {:name :type :shape} maps, :shape gen.array's (outermost first; the
   file lists GGML's reversed order), each with one of
   - :data, its values, row-major: a float[], or a function of no
     arguments that gives one when the tensor is written (so a converter
     holds one tensor at a time); encoded as :type (F32, F16, BF16, Q8_0,
     Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K or Q6_K: a keyword of
     gen.array.gguf.quants/types) by llama.cpp's reference quantizer
     (gen.array.gguf.quants);
   - :bytes, its encoding as :type already: a byte[] or a ByteBuffer, or
     a function that gives one (a tensor copied from another file).
   Loud (:invalid-args) for a value that does not fit its type, a key or a
   tensor named twice, data of another size, or an innermost axis that is
   not a whole number of the type's blocks; another tensor type is
   :unsupported. The file is written beside path and moved over it once
   complete and forced to the disk, so a reader never sees it half
   written. Returns path."
  [path {:keys [metadata tensors]}]
  (let [metadata (vec metadata)
        tensors (vec tensors)]
    (doseq [[k v] (frequencies (concat (map first metadata) (map :name tensors)))]
      (when (> (long v) 1)
        (throw (ex-info (str "write-gguf: " (pr-str k) " is named more than once") {:reason :invalid-args :name k}))))
    (doseq [e metadata]
      (when-not (and (vector? e) (= 3 (count e)) (string? (first e)))
        (throw (ex-info (str "write-gguf: a metadata entry is [key value-type value], not " (pr-str e)) {:reason :invalid-args :entry e}))))
    (run! check-tensor tensors)
    (let [align (alignment metadata)
          sizes (mapv writable-bytes tensors)
          offsets (vec (reductions (fn [^long o ^long n] (+ o (padded n align))) 0 sizes))
          header-size (+ 24
                         (long (reduce + 0 (map (fn [[k t v]] (+ 12 (alength (string-bytes k)) (value-size k t v))) metadata)))
                         (long (reduce + 0 (map (fn [{:keys [name shape]}] (+ 8 (alength (string-bytes name)) 4 (* 8 (count shape)) 4 8))
                                                tensors))))
          head (doto (ByteBuffer/allocate (padded header-size align)) (.order ByteOrder/LITTLE_ENDIAN))
          target (path-of path)
          tmp (path-of (str path ".tmp"))]
      (.putInt head 0x46554747)
      (.putInt head 3)
      (.putLong head (count tensors))
      (.putLong head (count metadata))
      (doseq [[k t v] metadata]
        (put-string! head k)
        (.putInt head (unchecked-int (type-id t)))
        (put-value! head k t v))
      (doseq [[{:keys [name type shape]} offset] (map vector tensors offsets)]
        (put-string! head name)
        (.putInt head (count shape))
        (doseq [d (reverse shape)] (.putLong head (long d)))
        (.putInt head (unchecked-int (long (:id (quants/types type)))))
        (.putLong head (long offset)))
      (.rewind head)
      (try
        (with-open [ch (FileChannel/open tmp (into-array OpenOption [StandardOpenOption/CREATE StandardOpenOption/WRITE
                                                                     StandardOpenOption/TRUNCATE_EXISTING]))]
          (while (.hasRemaining head) (.write ch head))
          (doseq [[t ^long size] (map vector tensors sizes)]
            (let [b (tensor-buffer t)
                  pad (ByteBuffer/allocate (- (padded size align) size))]
              (while (.hasRemaining b) (.write ch b))
              (while (.hasRemaining pad) (.write ch pad))))
          (.force ch true))
        (catch Throwable e
          (Files/deleteIfExists tmp)
          (throw e)))
      (Files/move tmp target (into-array java.nio.file.CopyOption [java.nio.file.StandardCopyOption/REPLACE_EXISTING
                                                                   java.nio.file.StandardCopyOption/ATOMIC_MOVE]))
      (str path))))
