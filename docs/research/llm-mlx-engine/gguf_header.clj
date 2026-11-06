;; A GGUF file's header on the pure JVM, for docs/research/llm-mlx-engine.md:
;; its metadata (arrays summarized) and every tensor's name, shape (GGML
;; order: innermost first), type and offset from :data-start.
;; Run: clojure -M docs/research/llm-mlx-engine/gguf_header.clj <model.gguf> …
(import '[java.nio ByteOrder MappedByteBuffer] '[java.nio.channels FileChannel FileChannel$MapMode]
        '[java.nio.file Path StandardOpenOption])

(def ggml-types {0 :F32 1 :F16 2 :Q4_0 3 :Q4_1 6 :Q5_0 7 :Q5_1 8 :Q8_0 9 :Q8_1 10 :Q2_K 11 :Q3_K 12 :Q4_K
                 13 :Q5_K 14 :Q6_K 15 :Q8_K 24 :I8 25 :I16 26 :I32 27 :I64 28 :F64 30 :BF16})

(defn header [path]
  (with-open [ch (FileChannel/open (Path/of path (make-array String 0)) (into-array [StandardOpenOption/READ]))]
    (let [^MappedByteBuffer b (doto (.map ch FileChannel$MapMode/READ_ONLY 0 (min (.size ch) (* 256 1024 1024)))
                                (.order ByteOrder/LITTLE_ENDIAN))
          u32 #(Integer/toUnsignedLong (.getInt b)) u64 #(.getLong b)
          string #(let [n (u64) bs (byte-array n)] (.get b bs) (String. bs "UTF-8"))
          value (fn value [t]
                  (case (int t)
                    0 (bit-and (.get b) 0xff) 1 (.get b) 2 (bit-and (.getShort b) 0xffff) 3 (.getShort b)
                    4 (u32) 5 (.getInt b) 6 (.getFloat b) 7 (not (zero? (.get b))) 8 (string)
                    9 (let [et (u32) n (u64) xs (vec (repeatedly n #(value et)))]
                        (if (> n 8) {:array (count xs) :of et :first (vec (take 3 xs))} xs))
                    10 (u64) 11 (u64) 12 (.getDouble b)))]
      (assert (= 0x46554747 (.getInt b)) "not a GGUF file")
      (let [version (u32) n-tensors (u64) n-kv (u64)
            kv (into (sorted-map) (repeatedly n-kv #(let [k (string) t (u32)] [k (value t)])))
            tensors (vec (repeatedly n-tensors #(let [nm (string) nd (u32) dims (vec (repeatedly nd u64)) t (u32) off (u64)]
                                                  {:name nm :dims dims :type (ggml-types t t) :offset off})))
            align (long (get kv "general.alignment" 32))
            data-start (* align (quot (+ (.position b) (dec align)) align))]
        {:version version :kv kv :tensors tensors :data-start data-start}))))

(doseq [path *command-line-args*]
  (let [{:keys [version kv tensors]} (header path)]
    (println "==" path "GGUF v" version)
    (doseq [[k v] kv :when (not (re-find #"^tokenizer\.ggml\.(tokens|token_type|merges|scores)$" k))]
      (println " " k (pr-str v)))
    (println "  tensor types:" (frequencies (map :type tensors)))
    (doseq [t (filter #(re-find #"^(token_embd|output|blk\.0\.|blk\.3\.)" (:name %)) tensors)]
      (println "   " (:name t) (:dims t) (:type t)))))
