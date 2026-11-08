(ns gen.backend.mlx.io
  "The MLX backend's files (gen.backend.mlx): GGUF tensors by MLX's loader
   (p/ILoading: mlx_load_gguf), for gen.array/load-gguf, and safetensors
   files' by mlx_load_safetensors, for gen.array/load-safetensors.

   MLX's loader (mlx/io/gguf.cpp, gguf_quants.cpp) reads F32, F16 and I32
   tensors as they are, and Q8_0, Q4_0 and Q4_1 as its affine
   quantization: name.weight's packed words, and name.scales and
   name.biases in float16. The scales are cast to float32 here (exact),
   so that dequantize and quantized_matmul compute in float32 and give
   the file's values (docs/research/llm-mlx-engine.md §7). Q4_1's biases
   are cast too; Q8_0's and Q4_0's are computed again in float32, -128·d
   and -8·d, as MLX's loader computes them in float16, which overflows
   for a scale d past 511.75: where it does not, the two are equal.
   gen.array has checked the tensors' types (gen.array.gguf), so MLX's
   own conversion of the others to float16 never runs, and a file with
   float64 metadata is refused before MLX misreads it (check-metadata).
   MLX's loader reads the file through the file cache (it maps it); the
   pages its load brought in are evicted after it (msync MS_INVALIDATE
   on macOS, posix_fadvise DONTNEED on Linux), the pages that were cached
   before kept, so that a load leaves the file cache as it found it, as
   the host reader does (unless -Dgen.gguf.cache=true: evict!).

   A file holding a type MLX's loader does not read as gen.array loads it
   (Q5_0, Q5_1 and the K-quants, which it would turn into float16, and
   BF16, whose reading there is unverified) is read
   by gen.array.gguf's host reader instead, every tensor of it: the
   quantized ones decoded and repacked on the host as MLX's affine
   quantization (the same words, scales and biases as MLX's loader gives
   for Q8_0, Q4_0 and Q4_1), Q6_K decoded to float32, and the arrays made
   from the host's data (load-host).

   A safetensors file's tensors are mlx_load_safetensors's, each of the
   file's dtype (BF16 as bfloat16, F16 as float16), as gen.array has read
   and checked its header (gen.array.safetensors): only the tensors
   selected are taken from the loader's map, the others never read (MLX
   loads lazily)."
  {:api :internal}
  (:require [clojure.string :as str]
            [gen.array.gguf :as gguf]
            [gen.backend.mlx.array :refer [code cpu-stream fence stream wrap]]
            [gen.backend.mlx.operands :refer [made operand scalar-handle with-temps]])
  (:import [gen.mlx Mlx MlxRaw]
           [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option MemoryLayout MemorySegment ValueLayout]
           [java.lang.invoke MethodHandle]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.file OpenOption Path StandardOpenOption]))

(set! *warn-on-reflection* true)

(defn- quantized-triple
  "[wq scales biases] of the quantized tensor name (of type t) in the
   loaded file g: the words as MLX loaded them, the scales and biases
   float32."
  [b ^long g name t]
  (let [s (stream b)
        base (subs name 0 (- (count name) (count ".weight")))
        f32 (int (code :float32))]
    (with-temps [tmp]
      (let [wq (wrap b (Mlx/mlx_io_gguf_get_array g name))
            scales (wrap b (Mlx/mlx_astype (tmp (Mlx/mlx_io_gguf_get_array g (str base ".scales"))) f32 s))
            hs (operand b scales nil)
            biases (wrap b (case t
                             :q4_1 (Mlx/mlx_astype (tmp (Mlx/mlx_io_gguf_get_array g (str base ".biases"))) f32 s)
                             :q8_0 (Mlx/mlx_multiply hs (tmp (scalar-handle -128.0 :float32)) s)
                             :q4_0 (Mlx/mlx_multiply hs (tmp (scalar-handle -8.0 :float32)) s)))]
        (fence scales)
        [wq scales biases]))))

(defn- check-metadata
  "Refuse a file with float64 metadata, which MLX's loader misreads: it
   keeps a float64 value as float32 and steps over 4 of its 8 bytes, so
   it reads every key after it from the wrong place, and past the file's
   end the process dies (SIGBUS in gguf_get_key; MLX 0.32.4,
   mlx/io/gguf.cpp's set_mx_value_from_gguf, found 2026-11-07)."
  [{:keys [path value-types]}]
  (when-let [ks (seq (sort (keep (fn [[k t]] (when (or (= t :float64) (= t [:array :float64])) k)) value-types)))]
    (throw (ex-info (str "load-gguf: " path ": the metadata holds float64 values (" (str/join ", " ks) "), which MLX's"
                         " GGUF loader misreads, crashing the process; the JVM backend reads the file")
                    {:reason :unsupported :path path :keys (vec ks)}))))

(def ^:private layouts
  "Each dtype the host reader gives (gen.array.gguf/parts): its element's
   layout, as the reader's arrays hold it."
  {:float32 ValueLayout/JAVA_FLOAT :float16 ValueLayout/JAVA_SHORT :bfloat16 ValueLayout/JAVA_SHORT
   :int32 ValueLayout/JAVA_INT :uint32 ValueLayout/JAVA_INT})

(defn- host-arrays
  "Tensor info of the file r reads (gen.array.gguf/reader's), read by the
   host reader a chunk at a time into native memory, a part's values
   whole, then made MLX's arrays (mlx_array_new_data copies them) and the
   memory freed: an array, or a quantized tensor's [words scales biases]."
  [b r info]
  (with-open [arena (Arena/ofConfined)]
    (let [ps (gguf/parts info)
          segs (mapv (fn [{:keys [dtype shape]}]
                       (let [^ValueLayout l (layouts dtype)]
                         (.allocate arena (* (.byteSize l) (long (reduce * 1 shape))) 16)))
                     ps)]
      (gguf/read-tensor! r info (fn [part ^long index data count]
                                  (let [{:keys [dtype]} (ps part)
                                        ^ValueLayout l (layouts dtype)]
                                    (MemorySegment/copy data 0 ^MemorySegment (segs part) l (* index (.byteSize l))
                                                        (int count)))))
      (let [arrays (mapv (fn [^MemorySegment seg {:keys [dtype shape]}]
                           (let [shp (.allocateFrom arena ValueLayout/JAVA_INT (int-array (map int shape)))]
                             (wrap b (made "mlx_array_new_data"
                                           #(MlxRaw/mlx_array_new_data seg shp (int (count shape)) (int (code dtype)))))))
                         segs ps)]
        (if (gguf/quantized? (:type info)) arrays (first arrays))))))

(defn load-host
  "The tensors of the GGUF file of header, by name, read by gen.array.gguf's
   host reader (past the file cache, a chunk at a time: read-tensor!) and
   made MLX's arrays a tensor at a time, each of its floats in its :as
   dtype when given: what load-gguf gives for a file MLX's loader reads,
   and for the types it does not."
  [b {:keys [path tensors]}]
  (with-open [^java.io.Closeable r (gguf/reader path)]
    (into {} (map (fn [info] [(:name info) (host-arrays b r info)])) tensors)))

(def ^:private libc
  "libc's functions for a file's pages in the file cache, as method
   handles (nil for one the system lacks: posix_fadvise on macOS)."
  (delay
    (let [linker (Linker/nativeLinker)
          lookup (.defaultLookup linker)
          f (fn [name ret & args]
              (when-let [sym (.orElse (.find lookup name) nil)]
                (.downcallHandle linker ^MemorySegment sym
                                 (FunctionDescriptor/of ret (into-array MemoryLayout args))
                                 ^"[Ljava.lang.foreign.Linker$Option;" (make-array Linker$Option 0))))
          i ValueLayout/JAVA_INT
          l ValueLayout/JAVA_LONG
          a ValueLayout/ADDRESS]
      {:getpagesize (f "getpagesize" i)
       :mincore (f "mincore" i a l a)
       :msync (f "msync" i a l i)
       :open (f "open" i a i)
       :posix_fadvise (f "posix_fadvise" i i l l i)
       :close (f "close" i i)})))

(defn- os
  "The system whose file cache evict! knows: :macos, :linux or nil."
  []
  (let [n (str/lower-case (System/getProperty "os.name"))]
    (cond (str/starts-with? n "mac") :macos
          (str/starts-with? n "linux") :linux)))

(defn- libc-call
  "libc's function k called with args (each of its parameter's type: an
   int, a long or a MemorySegment), its int result; loud (:io) when it
   reports a failure (posix_fadvise: a nonzero result; the others: -1)."
  [path k & args]
  (let [r (long (.invokeWithArguments ^MethodHandle (k @libc) ^java.util.List (vec args)))]
    (when (if (= k :posix_fadvise) (not (zero? r)) (neg? r))
      (throw (ex-info (str "load-gguf: " path ": " (name k) " failed (" r "), evicting the file's pages from the file cache")
                      {:reason :io :path (str path) :call k :result r})))
    r))

(defn- residency
  "The file at path's pages in the file cache now: the file mapped in
   arena (read-only, shared; mapping it reads nothing), its page size,
   and mincore's vector, a byte a page, bit 0 set for a page in the
   cache."
  [^Arena arena path]
  (with-open [ch (FileChannel/open (Path/of (str path) (make-array String 0))
                                   ^"[Ljava.nio.file.OpenOption;" (into-array OpenOption [StandardOpenOption/READ]))]
    (let [size (.size ch)
          segment (.map ch FileChannel$MapMode/READ_ONLY 0 size arena)
          page (long (libc-call path :getpagesize))
          v (.allocate arena (quot (+ size (dec page)) page))]
      (libc-call path :mincore segment (long size) v)
      {:path path :segment segment :page page :vector v})))

(defn- cached-bytes
  "The bytes of the file at path in the file cache now (whole pages)."
  ^long [path]
  (with-open [arena (Arena/ofConfined)]
    (let [{:keys [^MemorySegment vector ^long page]} (residency arena path)]
      (* page (long (reduce (fn [^long n ^long i] (+ n (bit-and (long (.get vector ValueLayout/JAVA_BYTE i)) 1)))
                            0 (range (.byteSize vector))))))))

(defn- uncached-runs
  "The byte ranges [offset length] of the file's pages that residency r
   saw out of the file cache, adjacent pages one range."
  [{:keys [^MemorySegment segment ^MemorySegment vector ^long page]}]
  (let [n (.byteSize vector)
        size (.byteSize segment)
        cached? (fn [^long i] (= 1 (bit-and (long (.get vector ValueLayout/JAVA_BYTE i)) 1)))
        run (fn [^long from ^long to] [(* from page) (- (min size (* to page)) (* from page))])]
    (loop [i 0 from -1 runs []]
      (cond
        (= i n) (cond-> runs (>= from 0) (conj (run from n)))
        (cached? i) (recur (inc i) -1 (cond-> runs (>= from 0) (conj (run from i))))
        :else (recur (inc i) (if (neg? from) i from) runs)))))

(defn- evict!
  "Evict from the file cache the pages of the file that residency r saw
   out of it (those a load since brought in); the pages cached then stay
   (another reader's). macOS: msync MS_INVALIDATE on the mapping; Linux:
   posix_fadvise POSIX_FADV_DONTNEED."
  [^Arena arena {:keys [path ^MemorySegment segment] :as r}]
  (let [runs (uncached-runs r)]
    (case (os)
      :macos (doseq [[off len] runs]
               (libc-call path :msync (.asSlice segment (long off) (long len)) (long len) (int 2)))
      :linux (let [fd (long (libc-call path :open (.allocateFrom arena (str path)) (int 0)))]
               (try
                 (doseq [[off len] runs]
                   (libc-call path :posix_fadvise (int fd) (long off) (long len) (int 4)))
                 (finally (libc-call path :close (int fd))))))))

(defn load-gguf
  "p/-load-gguf on the MLX backend b: the file's tensors by mlx_load_gguf,
   by name, the pages its load brought into the file cache evicted after
   it (unless -Dgen.gguf.cache=true); or, for a file holding a type MLX's
   loader does not read, by the host reader (load-host)."
  [b {:keys [path tensors] :as header}]
  (if-not (every? (comp gguf/mlx-reads? :type) tensors)
    (load-host b header)
    (do
      (check-metadata header)
      (with-open [arena (Arena/ofConfined)]
        (let [before (when (and (gguf/uncached?) (os)) (residency arena path))
              g (Mlx/mlx_load_gguf (str path) (stream b))]
          (try
            (into {}
                  (map (fn [{:keys [name type]}]
                         [name (if (gguf/quantized? type)
                                 (quantized-triple b g name type)
                                 (wrap b (Mlx/mlx_io_gguf_get_array g name)))]))
                  tensors)
            (finally
              (Mlx/mlx_io_gguf_free g)
              (when before (evict! arena before)))))))))

(defn load-safetensors
  "p/-load-safetensors on the MLX backend b: the tensors of the header
   (those selected) by mlx_load_safetensors, by name, read on b's CPU
   stream (MLX's Load runs there; its GPU kernels have none). A file
   holding a tensor of a dtype MLX 0.32.4's loader does not know (F64,
   F8_E5M2), selected or not, is loud (:unsupported): the loader refuses
   the whole file (and MLX's GPU has no float64)."
  [b {:keys [path tensors all-types]}]
  (when-let [[name t] (first (filter (comp #{:f64 :f8_e5m2} val) all-types))]
    (throw (ex-info (str "load-safetensors: " path ": tensor " name " is " (str/upper-case (clojure.core/name t))
                         ", which MLX's loader does not read (it refuses the file); the JVM backend reads F64")
                    {:reason :unsupported :path (str path) :tensor name :type t})))
  (let [[arrays metadata] (Mlx/mlx_load_safetensors (str path) (cpu-stream b))]
    (try
      (into {} (map (fn [{:keys [name]}] [name (wrap b (Mlx/mlx_map_string_to_array_get arrays name))])) tensors)
      (finally
        (Mlx/mlx_map_string_to_array_free arrays)
        (Mlx/mlx_map_string_to_string_free metadata)))))
