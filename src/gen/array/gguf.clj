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
     gguf_quants.cpp) loads them: F32, F16 and I32 as arrays, and Q8_0,
     Q4_0 and Q4_1 as MLX's affine quantization in groups of 32 (the
     quantized words, scales and biases gen.array/dequantize takes), the
     scales and biases in float32 (DESIGN §9, 1.3.0's engine design).
     Every other type is a loud error: MLX's loader would turn it into
     float16, silently, and the JVM backend reads only these.

   Dimensions: GGML lists a tensor's innermost first; a shape here is
   gen.array's (outermost first, as MLX reverses them), so a matrix
   weight is [out in]. Experimental, with gen.array/load-gguf (DESIGN
   §9). Written in plain Clojure: it requires nothing of gen.array."
  {:api :internal}
  (:require [clojure.string :as str]
            [gen.array.protocols :as p])
  (:import [java.nio ByteBuffer ByteOrder]
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
   value, or a quantized type's block (32 values) in bytes and its bits."
  {:f32 {:dtype :float32 :bytes 4}
   :f16 {:dtype :float16 :bytes 2}
   :i32 {:dtype :int32 :bytes 4}
   :q8_0 {:block 34 :bits 8}
   :q4_0 {:block 18 :bits 4}
   :q4_1 {:block 20 :bits 4}})

(def ^:const group-size
  "The values a GGUF quantization block holds, and so the group size of
   the affine quantization its tensors load as."
  32)

(defn quantized?
  "Is the tensor type t one that loads as a quantized triple?"
  [t]
  (contains? #{:q8_0 :q4_0 :q4_1} t))

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
        {:keys [bytes block]} (loadable t)]
    (cond bytes (* (long bytes) (long n))
          block (* (long block) (quot (long n) group-size)))))

(defn header
  "The header of the GGUF file at path: {:path :version :metadata
   :value-types :tensors}, :metadata a map of keys (strings) to host
   values, :value-types each key's GGUF value type (a keyword of
   value-types, or [:array element-type]), :tensors a
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
              [metadata types] (loop [i 0 m (transient {}) ts (transient {})]
                                 (if (< i n-kv)
                                   (let [k (read-string* path b)
                                         t (Integer/toUnsignedLong (.getInt b))
                                         ;; an array's element type follows
                                         et (when (= 9 t) (Integer/toUnsignedLong (.getInt b (.position b))))
                                         v (read-value path b t)]
                                     (recur (inc i) (assoc! m k v)
                                            (assoc! ts k (if et [:array (value-types et et)] (value-types t t)))))
                                   [(persistent! m) (persistent! ts)]))
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
          {:path (str path) :version version :metadata metadata :value-types types :tensors tensors})
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
   <base>.biases (the names its triple loads under on MLX), its innermost
   axis a multiple of 32, no name twice (:invalid-gguf), and as many as a
   split's first file says (split.tensors.count)."
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
                             ", a type not supported (F32, F16, I32, Q8_0, Q4_0 and Q4_1 are)")
                        {:reason :unsupported :path path :tensor name :type type})))
      (when (quantized? type)
        (when-not (str/ends-with? name ".weight")
          (throw (invalid path (str "the quantized tensor " name " is not named <base>.weight") {:tensor name})))
        (let [base (subs name 0 (- (count name) (count ".weight")))]
          (when (some names [(str base ".scales") (str base ".biases")])
            (throw (invalid path (str "the quantized tensor " name " shares its base with a tensor named "
                                      base ".scales or .biases") {:tensor name}))))
        (when-not (and (seq shape) (zero? (rem (long (peek shape)) group-size)))
          (throw (invalid path (str "the quantized tensor " name "'s innermost axis is not a multiple of 32: " shape)
                          {:tensor name :shape shape})))))
    tensors))

;; ---------------------------------------------------------------------------
;; Tensors on the host (the JVM backend's loader)
;; ---------------------------------------------------------------------------

(defn- f16 ^double [^ByteBuffer b ^long at]
  (double (Float/float16ToFloat (.getShort b (int at)))))

(defn- plain
  "The n values of a plain tensor, as doubles."
  ^doubles [^ByteBuffer b t ^long n]
  (let [out (double-array n)]
    (case t
      :f32 (dotimes [i n] (aset out i (double (.getFloat b (int (* 4 i))))))
      :f16 (dotimes [i n] (aset out i (f16 b (* 2 i))))
      :i32 (dotimes [i n] (aset out i (double (.getInt b (int (* 4 i)))))))
    out))

(defn- code
  "Value j (of 32) of the block at at, as MLX's loader stores it: Q8_0's
   int8 shifted to uint8 (x ^ 0x80), a 4-bit type's nibble (values 0-15
   the low nibbles of its 16 bytes, 16-31 the high ones)."
  ^long [^ByteBuffer b t ^long at ^long j]
  (case t
    :q8_0 (bit-xor (bit-and (long (.get b (int (+ at 2 j)))) 0xff) 0x80)
    (let [qs (+ at (if (= t :q4_1) 4 2))
          byte (bit-and (long (.get b (int (+ qs (bit-and j 15))))) 0xff)]
      (if (< j 16) (bit-and byte 15) (bit-shift-right byte 4)))))

(defn- quantized-block!
  "Block i's words (32 values of bits bits, packed little-endian: 8 or 4
   words), scale and bias, into words, scales and biases."
  [^ByteBuffer b t i ^doubles words ^doubles scales ^doubles biases]
  (let [i (long i)
        {:keys [block bits]} (loadable t)
        bits (long bits)
        at (* i (long block))
        per-word (quot 32 bits)
        n-words (quot 32 per-word)
        d (f16 b at)]
    (aset scales i d)
    ;; MLX's loader computes Q8_0's and Q4_0's biases in float16 (-128·d
    ;; and -8·d), which overflows for d past 511.75; in float32 they are
    ;; exact, and equal where float16's do not overflow
    (aset biases i (case t
                     :q8_0 (double (unchecked-float (* -128.0 d)))
                     :q4_0 (double (unchecked-float (* -8.0 d)))
                     :q4_1 (f16 b (+ at 2))))
    ;; value j at bits [j·bits, (j+1)·bits) of the block's words: the
    ;; values' bits are disjoint, so a word is their sum (exact in a double)
    (dotimes [j 32]
      (let [w (+ (* i n-words) (quot j per-word))]
        (aset words w (+ (aget words w) (double (bit-shift-left (code b t at j) (* (rem j per-word) bits)))))))))

(defn- quantized
  "[words scales biases] of a quantized tensor of n values, doubles: MLX's
   affine layout, groups of 32."
  [^ByteBuffer b t ^long n]
  (let [blocks (quot n group-size)
        bits (long (:bits (loadable t)))
        words (double-array (quot (* n bits) 32))
        scales (double-array blocks)
        biases (double-array blocks)]
    (dotimes [i blocks]
      (quantized-block! b t i words scales biases))
    [words scales biases]))

(defn read-tensor
  "The tensor of the file at path described by info (header's), on the
   host: {:dtype :shape :data} for a plain type (data a double[] of the
   values, row-major), or {:quantized [words scales biases] :shapes
   [words' scales' biases'] :bits} for a quantized one (MLX's affine
   layout in groups of 32: words uint32 values, scales and biases float32
   values, as doubles)."
  [path {:keys [type shape offset bytes]}]
  (let [n (reduce * 1 shape)]
    (when (> (long n) Integer/MAX_VALUE)
      (throw (ex-info (str "load-gguf: a tensor of " n " values is too large to read on the host")
                      {:reason :unsupported :path (str path) :shape shape})))
    (let [^ByteBuffer b (map-file path offset bytes)]
      (if (quantized? type)
        (let [bits (long (:bits (loadable type)))
              k (long (peek shape))]
          {:quantized (quantized b type n)
           :shapes [(conj (pop shape) (quot (* k bits) 32))
                    (conj (pop shape) (quot k group-size))
                    (conj (pop shape) (quot k group-size))]
           :bits bits})
        {:dtype (:dtype (loadable type)) :shape shape :data (plain b type n)}))))

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
