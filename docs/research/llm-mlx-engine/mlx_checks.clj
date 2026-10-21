;; The checks of docs/research/llm-mlx-engine.md §7, on MLX (one JVM, with
;; llama.cpp for the fourth):
;;   1. MLX's GGUF loader reads both models: every F32 tensor as float32,
;;      every Q8_0 one as a packed uint32 weight with float16 scales and
;;      biases in groups of 32;
;;   2. MLX's Q8_0 dequantization equals the file's blocks (d · q) exactly,
;;      and quantized_matmul agrees with dequantize-then-matmul;
;;   3. a slice_update writes in place when nothing else holds its input
;;      (MLX donates the buffer), and copies when something does;
;;   4. Qwen3-0.6B's next-token log-probabilities from tiny_forward.clj's
;;      composites over the GGUF's weights, dequantized, against the llama
;;      engine's on the same file.
;; Run (from the repository, after clojure -T:build java):
;;   clojure -Sdeps '{:paths ["src" "test" "target/classes"]}' -J--enable-native-access=ALL-UNNAMED \
;;     -J-Dgen.mlx.lib=<mlx lib> -J-Dgen.llama.lib=<llama lib> \
;;     -M docs/research/llm-mlx-engine/mlx_checks.clj <dir of the GGUF files>
(ns mlx-checks
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [gen.array :as array :refer [+ - * / < > <= >= ==]]
            [gen.random :as random]
            [gen.llm.branching :as br])
  (:import [gen.mlx Mlx]
           [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.nio ByteOrder]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.file Path StandardOpenOption]))

(def dir (first *command-line-args*))
(def qwen3 (str dir "/Qwen3-0.6B-Q8_0.gguf"))
(def qwen35 (str dir "/Qwen3.5-0.8B-Q8_0.gguf"))

(def b (array/backend-named :mlx))
(def stream (#'gen.backend.mlx/stream b))
(defn wrap [h] (#'gen.backend.mlx/wrap b h))
(defn handle ^long [x] (clojure.lang.Reflector/getInstanceField x "h"))
(def f32 ((deref #'gen.backend.mlx/dtype-codes) :float32))

(let [s (slurp "docs/research/llm-mlx-engine/gguf_header.clj")] (load-string (subs s 0 (.indexOf s "(doseq"))))
(def header (resolve 'user/header))

(defn gguf-arrays
  "Every array MLX's GGUF loader makes of the file at path, by name."
  [path]
  (with-open [arena (Arena/ofConfined)]
    (let [g (Mlx/mlx_load_gguf path stream)
          ks (Mlx/mlx_io_gguf_get_keys g)
          out (.allocate arena ValueLayout/ADDRESS)
          names (vec (for [i (range (Mlx/mlx_vector_string_size ks))]
                       (do (Mlx/mlx_vector_string_get out ks i)
                           (.getString (.reinterpret ^MemorySegment (.get out ValueLayout/ADDRESS 0) Long/MAX_VALUE) 0))))
          arrays (into {} (map (fn [k] [k (wrap (Mlx/mlx_io_gguf_get_array g k))])) names)]
      (Mlx/mlx_vector_string_free ks)
      (Mlx/mlx_io_gguf_free g)
      arrays)))

(defn check-load [path]
  (let [{:keys [tensors]} (header path)
        arrays (gguf-arrays path)
        expect (fn [{:keys [name dims type]}]
                 (let [shape (vec (reverse dims))
                       base (subs name 0 (- (count name) (count ".weight")))
                       grouped (conj (pop shape) (quot (peek shape) 32))]
                   (case type
                     :F32 {name [:float32 shape]}
                     :Q8_0 {name [:uint32 (conj (pop shape) (quot (peek shape) 4))]
                            (str base ".scales") [:float16 grouped]
                            (str base ".biases") [:float16 grouped]})))
        expected (into {} (map expect) tensors)
        got (into {} (map (fn [[k x]] [k [(array/dtype x) (array/shape x)]])) arrays)
        wrong (remove (fn [[k v]] (= v (got k))) expected)]
    (println (format "1. %s: %d tensors, %d arrays; %s" (.getName (java.io.File. ^String path)) (count tensors) (count arrays)
                     (if (and (empty? wrong) (= (set (keys expected)) (set (keys got))))
                       "every one as expected"
                       (str "MISMATCHES " (vec (take 5 wrong)) " extra " (vec (take 5 (remove expected (keys got))))))))
    arrays))

(defn file-rows
  "Rows r of a Q8_0 tensor [rows, cols], dequantized on the host from the
   file's blocks: f16 scale d, then 32 int8 q, value d·q in float32."
  [path tensor rows]
  (let [{:keys [data-start tensors]} (header path)
        {:keys [dims offset]} (first (filter #(= tensor (:name %)) tensors))
        cols (first dims)
        row-bytes (* (quot cols 32) 34)]
    (with-open [ch (FileChannel/open (Path/of path (make-array String 0)) (into-array [StandardOpenOption/READ]))]
      (let [m (doto (.map ch FileChannel$MapMode/READ_ONLY (+ data-start offset) (* row-bytes (inc (reduce max rows))))
                (.order ByteOrder/LITTLE_ENDIAN))]
        (vec (for [r rows]
               (vec (for [j (range (quot cols 32)) i (range 32)]
                      (let [at (+ (* r row-bytes) (* j 34))]
                        (double (* (Float/float16ToFloat (.getShort m (int at))) (float (.get m (int (+ at 2 i)))))))))))))))

(defn check-quantized [arrays]
  (array/with-backend b
    (let [w (arrays "blk.0.attn_q.weight") s (arrays "blk.0.attn_q.scales") z (arrays "blk.0.attn_q.biases")
          deq (wrap (Mlx/mlx_dequantize (handle w) (handle s) (handle z) (int 32) (int 8) "affine" 0 (int f32) stream))
          mine (array/->clj (array/slice deq 0 0 3))
          theirs (file-rows qwen3 "blk.0.attn_q.weight" [0 1 2])
          x (random/normal (random/key 1) [1 4 1024])
          qmm (wrap (Mlx/mlx_quantized_matmul (handle x) (handle w) (handle s) (handle z) true (int 32) (int 8) "affine" stream))
          ref (array/matmul x (array/transpose deq))
          d (array/item (array/max (array/abs (- qmm ref))))
          scale (array/item (array/max (array/abs ref)))]
      (println (format "2. Q8_0: MLX's dequantized rows 0-2 equal the file's d·q %s; quantized_matmul vs dequantize+matmul: max |Δ| %.2e (of max |y| %.2f)"
                       (if (= mine theirs) "exactly" (str "NOT exactly (first difference at "
                                                         (first (keep-indexed (fn [i [a c]] (when (not= a c) [i a c]))
                                                                              (map vector (flatten mine) (flatten theirs)))) ")"))
                       d scale)))))

(defn check-donation []
  (let [data (fn ^long [^long h] (.address ^MemorySegment (Mlx/mlx_array_data_float32 h)))
        upd (Mlx/mlx_ones (int-array [1 8 1 128]) (int f32) stream)
        step (fn [^long a] (Mlx/mlx_slice_update a upd (int-array [0 0 5 0]) (int-array [1 8 6 128]) (int-array [1 1 1 1]) stream))
        fresh #(doto (Mlx/mlx_zeros (int-array [1 8 1024 128]) (int f32) stream) (Mlx/mlx_array_eval))
        ;; unshared: the input's only holder is the update
        a (fresh) pa (data a) u (step a)
        _ (Mlx/mlx_array_free a)
        _ (Mlx/mlx_array_eval u)
        pu (data u)
        ;; shared: the input is still held
        a2 (fresh) pa2 (data a2) u2 (step a2)
        _ (Mlx/mlx_array_eval u2)
        pu2 (data u2)]
    (println (format "3. slice_update of an unshared 4 MiB array: %s; of a held one: %s"
                     (if (== pa pu) "in place (the buffer donated)" "COPIED")
                     (if (== pa2 pu2) "IN PLACE (the held array overwritten!)" "copied")))
    (run! #(Mlx/mlx_array_free %) [u a2 u2 upd])))

(defn check-logits [arrays]
  (let [s (slurp "docs/research/llm-mlx-engine/tiny_forward.clj")]
    (load-string (subs s 0 (.indexOf s ";; ---------------------------------------------------------------------------\n;; The checks"))))
  (let [forward @(resolve 'tiny-forward/forward)
        cfg {:heads 16 :kv-heads 8 :head-dim 128 :eps 1e-6 :theta 1000000.0}
        deq (fn [base] (if-let [sc (arrays (str base ".scales"))]
                         (wrap (Mlx/mlx_dequantize (handle (arrays (str base ".weight"))) (handle sc)
                                                   (handle (arrays (str base ".biases"))) (int 32) (int 8) "affine" 0 (int f32) stream))
                         (arrays (str base ".weight"))))
        model {:embed (deq "token_embd") :norm (deq "output_norm")
               :layers (vec (for [i (range 28) :let [p (str "blk." i ".")]]
                              {:kind :attn :attn-norm (deq (str p "attn_norm")) :ffn-norm (deq (str p "ffn_norm"))
                               :q (deq (str p "attn_q")) :k (deq (str p "attn_k")) :v (deq (str p "attn_v"))
                               :o (deq (str p "attn_output")) :q-norm (deq (str p "attn_q_norm")) :k-norm (deq (str p "attn_k_norm"))
                               :gate (deq (str p "ffn_gate")) :up (deq (str p "ffn_up")) :down (deq (str p "ffn_down"))}))}]
    (with-open [^java.io.Closeable e ((requiring-resolve 'gen.llm.llama/engine) {:model qwen3 :max-branches 1 :n-ctx 512})]
      (doseq [text ["The capital of Sweden is a city called" "def fibonacci(n):\n    if n <"]]
        (let [prompt ((requiring-resolve 'gen.llm.llama/tokenize) e text)
              n (count prompt)
              ref (let [x (br/open e prompt)] (br/logprobs (doto x br/dispose!)))
              lp (array/with-backend b
                   (let [[logits _] (forward cfg model nil (array/array [prompt] {:dtype :int32}) (array/array [(vec (range n))]))]
                     (float-array (array/->clj (array/log-softmax (array/reshape (array/slice logits 1 (dec n) n) [(last (array/shape logits))]))))))
              top (fn [^floats a k] (vec (take k (sort-by #(- (aget a (int %))) (range (alength a))))))
              ref-top (top ref 64)
              d (reduce max (map #(Math/abs (- (aget ^floats ref (int %)) (aget ^floats lp (int %)))) ref-top))]
          (println (format "4. %s (%d tokens): argmax %s, top 5 %s, max |Δ log p| over llama.cpp's top 64: %.3g"
                           (pr-str text) n (if (= (first ref-top) (first (top lp 1))) "equal" "DIFFERS")
                           (if (= (set (take 5 ref-top)) (set (top lp 5))) "equal as sets" "DIFFER") d)))))))

(let [a3 (check-load qwen3)]
  (check-load qwen35)
  (check-quantized a3)
  (check-donation)
  (check-logits a3))
(shutdown-agents)
