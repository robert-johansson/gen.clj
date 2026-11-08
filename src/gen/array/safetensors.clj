(ns gen.array.safetensors
  "Safetensors files (Hugging Face's format for checkpoints) read on the
   host, below the backends: the body of gen.array/load-safetensors, and
   the reader the JVM backend loads tensors with.

   - A file: 8 bytes, the header's length n (a little-endian u64); the
     header, n bytes of JSON (gen.array.json), an object from each
     tensor's name to {\"dtype\" \"shape\" \"data_offsets\"} and an optional
     \"__metadata__\" of strings; then the data, each tensor's bytes
     row-major and little-endian at its offsets [begin, end) from the
     header's end. The tensors' bytes cover the data exactly, in order,
     with no gap and no overlap (the format's rule, as its Rust reader
     checks it).
   - A checkpoint sharded over files: model.safetensors.index.json, its
     \"weight_map\" from each tensor's name to the file (beside the index)
     that holds it, every file's tensors exactly those the map gives it.
   - The dtypes loaded: F32, F64, F16, BF16, I32, I64, U32 (the packed
     words of MLX's quantized checkpoints) and BOOL. Every other one (I8,
     I16, U8, U16, U64, the float8 types, C64) is a loud error when a
     tensor of it is loaded: gen.array has no dtype for it.

   Shapes are gen.array's, outermost first, as the file lists them.
   Experimental, with gen.array/load-safetensors (DESIGN §9). Written in
   plain Clojure: it requires nothing of gen.array."
  {:api :internal}
  (:require [clojure.string :as str]
            [gen.array.json :as json]
            [gen.array.protocols :as p])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.file Files LinkOption OpenOption Path StandardOpenOption]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

;; ---------------------------------------------------------------------------
;; Dtypes
;; ---------------------------------------------------------------------------

(def sizes
  "The bytes a value of each dtype of the format takes (its every dtype,
   so that every tensor's bytes are checked, loaded or not)."
  {"F64" 8 "F32" 4 "F16" 2 "BF16" 2 "I64" 8 "I32" 4 "I16" 2 "I8" 1 "U64" 8 "U32" 4 "U16" 2 "U8" 1
   "BOOL" 1 "F8_E4M3" 1 "F8_E5M2" 1 "F8_E8M0" 1 "C64" 8})

(def loadable
  "The format's dtypes load-safetensors loads, to gen.array's dtypes:
   F16 and BF16 as float16 and bfloat16, or float32 (the same values) on
   a backend without them."
  {"F64" :float64 "F32" :float32 "F16" :float16 "BF16" :bfloat16 "I64" :int64 "I32" :int32 "U32" :uint32
   "BOOL" :bool})

(defn- invalid
  [path message data]
  (ex-info (str "load-safetensors: " path ": " message) (merge {:reason :invalid-safetensors :path (str path)} data)))

;; ---------------------------------------------------------------------------
;; The header
;; ---------------------------------------------------------------------------

(defn- path-of ^Path [path] (Path/of (str path) (make-array String 0)))

(defn- file? [path] (Files/isRegularFile (path-of path) (make-array LinkOption 0)))

(defn- directory? [path] (Files/isDirectory (path-of path) (make-array LinkOption 0)))

(defn- map-file
  "The bytes [offset, offset + n) of the file at path, little-endian."
  ^ByteBuffer [path ^long offset ^long n]
  (with-open [ch (FileChannel/open (path-of path) (into-array OpenOption [StandardOpenOption/READ]))]
    (doto (.map ch FileChannel$MapMode/READ_ONLY offset n)
      (.order ByteOrder/LITTLE_ENDIAN))))

(def ^:private max-header
  "The longest header read, the format's own limit (its Rust reader's,
   and MLX's)."
  100000000)

(defn- tensor-info
  "A tensor's entry of the header, checked: {:name :type :dtype :shape
   :offset :bytes}, :offset absolute in the file."
  [path data-start data-size name entry]
  (when-not (and (map? entry) (= #{"dtype" "shape" "data_offsets"} (set (keys entry))))
    (throw (invalid path (str "tensor " name "'s entry is not {\"dtype\" \"shape\" \"data_offsets\"}: " (pr-str entry))
                    {:tensor name})))
  (let [{dt "dtype" shape "shape" [begin end :as offsets] "data_offsets"} entry
        size (sizes dt)]
    (when-not size
      (throw (ex-info (str "load-safetensors: " path ": tensor " name "'s dtype " (pr-str dt) " is not one of the format's")
                      {:reason :unsupported :path (str path) :tensor name :dtype dt})))
    (when-not (and (vector? shape) (every? #(and (integer? %) (<= 0 (long %))) shape))
      (throw (invalid path (str "tensor " name "'s shape is not a vector of sizes: " (pr-str shape)) {:tensor name})))
    (when-not (and (vector? offsets) (= 2 (count offsets)) (every? integer? offsets) (<= 0 (long begin) (long end)))
      (throw (invalid path (str "tensor " name "'s data_offsets are not [begin end]: " (pr-str offsets)) {:tensor name})))
    (let [n (* (long size) (long (reduce * 1 shape)))]
      (when-not (== n (- (long end) (long begin)))
        (throw (invalid path (str "tensor " name " of " dt " " shape " takes " n " bytes, not its offsets' "
                                  (- (long end) (long begin)))
                        {:tensor name :offsets offsets})))
      (when (> (long end) (long data-size))
        (throw (invalid path (str "tensor " name "'s data lies past the end of the file") {:tensor name :offsets offsets})))
      {:name name :type (keyword (str/lower-case dt)) :dtype (loadable dt) :shape shape
       :offset (+ (long data-start) (long begin)) :bytes n :begin (long begin) :end (long end)})))

(defn header
  "The header of the safetensors file at path: {:path :metadata
   :tensors}, :metadata its \"__metadata__\" (strings to strings, {} when
   it has none), :tensors a vector of {:name :type :dtype :shape :offset
   :bytes} in the data's order: :type the file's dtype as a keyword (:f32,
   :bf16, …), :dtype gen.array's it loads as (nil for one it does not
   load), :offset the data's absolute position in the file and :bytes its
   size. Loud (:invalid-safetensors) on a file cut short, a header that is
   no JSON object of entries, or tensors whose bytes do not cover the data
   exactly (a gap, an overlap, bytes past the last); (:unsupported) on a
   dtype the format does not have."
  [path]
  (when-not (file? path)
    (throw (ex-info (str "load-safetensors: no file at " path) {:reason :no-file :path (str path)})))
  (let [size (Files/size (path-of path))
        _ (when (< size 8) (throw (invalid path "not a safetensors file (shorter than its header's length)" {})))
        n (.getLong ^ByteBuffer (map-file path 0 8))]
    (when-not (and (< 1 n max-header) (<= (+ 8 n) size))
      (throw (invalid path (str "not a safetensors file (a header of " (Long/toUnsignedString n) " bytes in a file of "
                                size ")")
                      {:header-bytes n})))
    (let [bs (byte-array n)
          _ (.get ^ByteBuffer (map-file path 8 n) bs)
          h (try (json/read-bytes bs)
                 (catch clojure.lang.ExceptionInfo e
                   (throw (invalid path (str "its header is no JSON: " (ex-message e)) {}))))
          _ (when-not (map? h) (throw (invalid path "its header is no JSON object" {})))
          metadata (get h "__metadata__" {})
          _ (when-not (and (map? metadata) (every? string? (vals metadata)))
              (throw (invalid path (str "its __metadata__ is not a map of strings: " (pr-str metadata)) {})))
          data-start (+ 8 n)
          data-size (- size data-start)
          infos (->> (dissoc h "__metadata__")
                     (map (fn [[name entry]] (tensor-info path data-start data-size name entry)))
                     (sort-by (juxt :begin :end))
                     vec)]
      ;; the bytes covered exactly, in order: each tensor's begin the last's end
      (reduce (fn [^long at {:keys [name ^long begin ^long end]}]
                (when-not (== at begin)
                  (throw (invalid path (str "tensor " name "'s data begins at " begin ", not at " at
                                            (if (< begin at) " (it overlaps another's)" " (a gap before it)"))
                                  {:tensor name})))
                end)
              0 infos)
      (let [covered (if (seq infos) (long (:end (peek infos))) 0)]
        (when-not (== covered data-size)
          (throw (invalid path (str "the tensors cover " covered " bytes of the data's " data-size) {}))))
      {:path (str path) :metadata metadata :tensors (mapv #(dissoc % :begin :end) infos)})))

;; ---------------------------------------------------------------------------
;; Files: one, or a checkpoint's shards by its index
;; ---------------------------------------------------------------------------

(def index-name "model.safetensors.index.json")

(defn- shard-path
  "The shard file of an index at index-path: beside the index, and no
   path that leaves its directory."
  [index-path file]
  (let [dir (.toAbsolutePath (.getParent (.toAbsolutePath (path-of index-path))))
        p (.normalize (.resolve dir (str file)))]
    (when-not (and (string? file) (.startsWith p dir))
      (throw (invalid index-path (str "the weight map names " (pr-str file) ", which is not a file beside the index")
                      {:file file})))
    (str (.resolve (.getParent (path-of index-path)) (str file)))))

(defn- index-headers
  "The headers of an index's shards (in the order of their names), each
   checked to hold exactly the tensors the weight map gives it."
  [path]
  (let [index (try (json/read-file path)
                   (catch clojure.lang.ExceptionInfo e
                     (throw (invalid path (str "its index is no JSON: " (ex-message e)) {}))))
        weight-map (when (map? index) (get index "weight_map"))]
    (when-not (and (map? weight-map) (seq weight-map) (every? string? (vals weight-map)))
      (throw (invalid path "the index has no \"weight_map\" from tensor names to files" {})))
    (let [by-file (group-by val weight-map)]
      (vec (for [file (sort (keys by-file))
                 :let [h (try (header (shard-path path file))
                              (catch clojure.lang.ExceptionInfo e
                                (if (= :no-file (:reason (ex-data e)))
                                  (throw (invalid path (str "the shard " file " is missing") {:missing file}))
                                  (throw e))))
                       mapped (set (map key (by-file file)))
                       held (set (map :name (:tensors h)))]]
             (do (when-not (= mapped held)
                   (throw (invalid (:path h) (str "the shard holds other tensors than the index maps to it"
                                                  (when-let [m (seq (sort (remove held mapped)))]
                                                    (str "; missing: " (str/join ", " (take 4 m))))
                                                  (when-let [x (seq (sort (remove mapped held)))]
                                                    (str "; not in the index: " (str/join ", " (take 4 x)))))
                                   {:missing (vec (sort (remove held mapped))) :unmapped (vec (sort (remove mapped held)))})))
                 h))))))

(defn headers
  "The headers of the files at path: a safetensors file; an index (a
   .json file, model.safetensors.index.json's form), every shard it maps
   (each holding exactly the tensors the map gives it, so none is in two);
   or a checkpoint's directory, by its index when it has one, else its
   model.safetensors. Loud (:no-file) when there is none."
  [path]
  (let [p (str path)]
    (cond (directory? p)
          (let [index (str (.resolve (path-of p) ^String index-name))
                single (str (.resolve (path-of p) "model.safetensors"))]
            (cond (file? index) (index-headers index)
                  (file? single) [(header single)]
                  :else (throw (ex-info (str "load-safetensors: the directory " p " holds neither " index-name
                                             " nor model.safetensors")
                                        {:reason :no-file :path p}))))
          (str/ends-with? p ".json") (do (when-not (file? p)
                                           (throw (ex-info (str "load-safetensors: no file at " p) {:reason :no-file :path p})))
                                         (index-headers p))
          :else [(header p)])))

;; ---------------------------------------------------------------------------
;; Tensors on the host: the JVM backend's loader
;; ---------------------------------------------------------------------------

(def ^:private exact-limit
  "2^53: the integers a double holds exactly, up to it."
  9007199254740992)

(defn read-tensor
  "The tensor of the file at path described by info (header's), on the
   host as doubles, the JVM backend's storage: {:dtype :shape :data},
   data a double[] of its values row-major (a float16's or bfloat16's
   value, as float32 holds it exactly; a uint32 as unsigned). Loud for a
   tensor of more than 2^31 - 1 values or bytes (:unsupported), an int64
   value past 2^53, which a double does not hold exactly (:unsupported),
   and a BOOL byte neither 0 nor 1 (:invalid-safetensors)."
  [path {:keys [name type dtype shape offset bytes]}]
  (let [n (long (reduce * 1 shape))
        bytes (long bytes)]
    (when (or (> n Integer/MAX_VALUE) (> bytes Integer/MAX_VALUE))
      (throw (ex-info (str "load-safetensors: tensor " name " (" n " values, " bytes " bytes) is too large to read on the host")
                      {:reason :unsupported :path (str path) :tensor name :shape shape})))
    (let [^ByteBuffer b (if (zero? bytes) (ByteBuffer/allocate 0) (map-file path offset bytes))
          out (double-array n)]
      (case type
        :f32 (let [a (float-array n)] (.get (.asFloatBuffer b) a) (dotimes [i n] (aset out i (double (aget a i)))))
        :f64 (.get (.asDoubleBuffer b) out)
        :f16 (let [a (short-array n)] (.get (.asShortBuffer b) a) (dotimes [i n] (aset out i (double (Float/float16ToFloat (aget a i))))))
        :bf16 (let [a (short-array n)]
                (.get (.asShortBuffer b) a)
                (dotimes [i n] (aset out i (double (Float/intBitsToFloat (bit-shift-left (bit-and (long (aget a i)) 0xffff) 16))))))
        :i32 (let [a (int-array n)] (.get (.asIntBuffer b) a) (dotimes [i n] (aset out i (double (aget a i)))))
        :u32 (let [a (int-array n)] (.get (.asIntBuffer b) a) (dotimes [i n] (aset out i (double (Integer/toUnsignedLong (aget a i))))))
        :i64 (let [a (long-array n)]
               (.get (.asLongBuffer b) a)
               (dotimes [i n]
                 (let [v (aget a i)]
                   (when (> (Math/abs v) (long exact-limit))
                     (throw (ex-info (str "load-safetensors: tensor " name " holds the int64 " v
                                          ", past 2^53: the JVM backend's doubles do not hold it exactly")
                                     {:reason :unsupported :path (str path) :tensor name :value v})))
                   (aset out i (double v)))))
        :bool (dotimes [i n]
                (let [v (long (.get b (int i)))]
                  (when-not (or (== v 0) (== v 1))
                    (throw (invalid path (str "tensor " name " holds the BOOL byte " v ", neither 0 nor 1") {:tensor name})))
                  (aset out i (double v)))))
      {:dtype dtype :shape shape :data out})))

;; ---------------------------------------------------------------------------
;; Loading
;; ---------------------------------------------------------------------------

(defn- check-options [opts]
  (when-not (or (nil? opts) (map? opts))
    (throw (ex-info (str "load-safetensors: options are a map, not " (pr-str opts)) {:reason :invalid-options})))
  (when-let [unknown (seq (remove #{:select} (keys opts)))]
    (throw (ex-info (str "load-safetensors: unknown options " (vec unknown) "; it takes [:select]")
                    {:reason :unknown-options :unknown (vec unknown) :accepted #{:select}})))
  (when-not (or (nil? (:select opts)) (ifn? (:select opts)))
    (throw (ex-info (str "load-safetensors: :select is a predicate of tensor names, not " (pr-str (:select opts)))
                    {:reason :invalid-options}))))

(defn load-safetensors
  "gen.array/load-safetensors's body, on backend b: every file's selected
   tensors by the backend (p/-load-safetensors), evaluated, with the first
   file's metadata."
  [b path opts]
  (check-options opts)
  (let [hs (headers path)
        select (or (:select opts) (constantly true))
        hs (mapv (fn [h] (-> h
                             (assoc :all-types (into {} (map (juxt :name :type)) (:tensors h)))
                             (update :tensors #(filterv (comp boolean select :name) %))))
                 hs)]
    (doseq [{:keys [name type dtype]} (mapcat :tensors hs)]
      (when-not dtype
        (throw (ex-info (str "load-safetensors: " path ": tensor " name " is " (str/upper-case (clojure.core/name type))
                             ", a dtype not loaded (F32, F64, F16, BF16, I32, I64, U32 and BOOL are)")
                        {:reason :unsupported :path (str path) :tensor name :type type}))))
    (let [loaded (into {} (mapcat (fn [h] (when (seq (:tensors h)) (p/load-safetensors b h)))) hs)
          tensors (into {}
                        (map (fn [{:keys [name]}]
                               [name (or (get loaded name)
                                         (throw (ex-info (str "load-safetensors: the " (clojure.core/name (p/-name b))
                                                              " backend did not load tensor " name)
                                                         {:reason :invalid-safetensors :path (str path) :tensor name})))]))
                        (mapcat :tensors hs))]
      (p/-eval! b (vals tensors))
      {:metadata (:metadata (first hs))
       :tensors tensors
       :types (into {} (map (juxt :name :type)) (mapcat :tensors hs))
       :files (mapv :path hs)})))
