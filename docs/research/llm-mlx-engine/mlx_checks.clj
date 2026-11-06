;; The checks of docs/research/llm-mlx-engine.md §7, on MLX (one JVM, with
;; llama.cpp for the fourth and fifth), and their results there:
;;   1. MLX's GGUF loader reads both models: every F32 tensor as float32,
;;      every Q8_0 one as a packed uint32 weight with float16 scales and
;;      biases in groups of 32;
;;   2. MLX's Q8_0 dequantization against the file's blocks (d · q), with
;;      the float16 scales as loaded and cast to float32, and
;;      quantized_matmul against dequantize-then-matmul;
;;   3. a slice_update writes in place when nothing else holds its input
;;      (MLX donates the buffer), and copies when something does;
;;   4. Qwen3-0.6B's next-token log-probabilities from tiny_forward.clj's
;;      composites over the GGUF's weights, dequantized, against the llama
;;      engine's on the same file; with planted convention errors (4-heads,
;;      4-rope) it must fail;
;;   5. the same for Qwen3.5-0.8B, whole and as a cached prefill plus a
;;      step, with planted errors of its own (5-conv-shift, 5-conv-kernel,
;;      5-q-gate).
;; Run (from the repository, after clojure -T:build java), with check
;; numbers after the directory to run only those:
;;   clojure -Sdeps '{:paths ["src" "test" "target/classes"]}' -J--enable-native-access=ALL-UNNAMED \
;;     -J-Dgen.mlx.lib=<mlx lib> -J-Dgen.llama.lib=<llama lib> \
;;     -M docs/research/llm-mlx-engine/mlx_checks.clj <dir of the GGUF files> [1 2 3 4 5 4-heads …]
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

;; header, from gguf_header.clj, defined here
(let [s (slurp "docs/research/llm-mlx-engine/gguf_header.clj")] (load-string (subs s 0 (.indexOf s "(doseq"))))
(declare header)

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
          deq (fn [s z] (wrap (Mlx/mlx_dequantize (handle w) (handle s) (handle z) (int 32) (int 8) "affine" 0 (int f32) stream)))
          qmm (fn [x s z] (wrap (Mlx/mlx_quantized_matmul (handle x) (handle w) (handle s) (handle z) true (int 32) (int 8) "affine" stream)))
          rows #(array/->clj (array/slice % 0 0 3))
          theirs (file-rows qwen3 "blk.0.attn_q.weight" [0 1 2])
          f16 (fn [rs] (mapv (fn [r] (mapv #(double (Float/float16ToFloat (Float/floatToFloat16 (float %)))) r)) rs))
          s32 (array/astype s :float32) z32 (array/astype z :float32)
          same (fn [mine ref] (if (= mine ref)
                                "exactly"
                                (str "NOT exactly (first difference "
                                     (first (keep-indexed (fn [i [a c]] (when (not= a c) [i a c])) (map vector (flatten mine) (flatten ref))))
                                     ")")))
          x (random/normal (random/key 1) [1 4 1024])
          gap (fn [y ref] (format "max |Δ| %.2e of max |y| %.2f" (array/item (array/max (array/abs (- y ref))))
                                  (array/item (array/max (array/abs ref)))))]
      (println (format "2. Q8_0, blk.0.attn_q, rows 0-2: dequantized with the loaded float16 scales, equal to the file's d·q %s, and to d·q rounded to float16 %s"
                       (same (rows (deq s z)) theirs) (same (rows (deq s z)) (f16 theirs))))
      (println (format "   with the scales and biases cast to float32 first: equal to d·q %s" (same (rows (deq s32 z32)) theirs)))
      (println (format "   quantized_matmul against dequantize then matmul: float16 scales %s; float32 scales %s"
                       (gap (qmm x s z) (array/matmul x (array/transpose (deq s z))))
                       (gap (qmm x s32 z32) (array/matmul x (array/transpose (deq s32 z32)))))))))

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

;; ---------------------------------------------------------------------------
;; 4 and 5: the next-token log-probabilities against the llama engine's
;; ---------------------------------------------------------------------------

(def mutations
  "Planted convention errors, for the checks' power, as edits of
   tiny_forward.clj's source:
   - heads (Qwen3): KV heads tiled (query head h reading KV head h mod KV)
     where llama.cpp groups them (h / (H/KV));
   - rope (Qwen3): RoPE rotating the other way. (Swapping RoPE's two halves
     in both q and k is no error: their dot products do not change, and
     neither do the logits.)
   - conv-shift (Qwen3.5): the conv state keeps the oldest inputs, not the
     newest (it matters only to a cached step);
   - conv-kernel (Qwen3.5): the conv kernel applied newest input first;
   - q-gate (Qwen3.5): the query and the output gate swapped in q's
     projection."
  {"heads" #(.replace ^String % "(array/broadcast-to (array/expand-dims x [2]) [b kv rep s d])"
                      "(array/broadcast-to (array/expand-dims x [1]) [b rep kv s d])")
   "rope" #(.replace ^String % "c (array/cos ang) s (array/sin ang)" "c (array/cos ang) s (- (array/sin ang))")
   "conv-shift" #(.replace ^String % "{:conv (array/slice padded 1 t (+ t (dec conv))) :state state}"
                           "{:conv (array/slice padded 1 0 (dec conv)) :state state}")
   "conv-kernel" #(.replace ^String % "(array/slice (:conv-w w) 1 j (inc j))" "(array/slice (:conv-w w) 1 (- conv j 1) (- conv j))")
   "q-gate" #(-> ^String %
                 (.replace "(array/slice qg 3 0 head-dim)" "QUERY")
                 (.replace "(array/slice qg 3 head-dim (* 2 head-dim))" "(array/slice qg 3 0 head-dim)")
                 (.replace "QUERY" "(array/slice qg 3 head-dim (* 2 head-dim))"))})

(defn- forward-fn
  "tiny_forward.clj's forward, with mutation (a key of mutations, or nil)
   planted in its source."
  [mutation]
  (let [s (slurp "docs/research/llm-mlx-engine/tiny_forward.clj")
        s' (if mutation ((mutations mutation) s) s)]
    (when mutation (assert (not= s s') (str "the mutation " mutation " changed nothing")))
    (load-string (subs s' 0 (.indexOf s' ";; ---------------------------------------------------------------------------\n;; The checks")))
    @(resolve 'tiny-forward/forward)))

(defn- dequantizer
  "base -> its tensor: base.weight, dequantized to float32 if quantized
   (scales :float16 as loaded, or :float32 cast first)."
  [arrays scales]
  (let [cast (fn [x] (if (= scales :float32) (array/astype x :float32) x))]
    (fn [base]
      (if-let [sc (arrays (str base ".scales"))]
        (wrap (Mlx/mlx_dequantize (handle (arrays (str base ".weight"))) (handle (cast sc))
                                  (handle (cast (arrays (str base ".biases")))) (int 32) (int 8) "affine" 0 (int f32) stream))
        (or (arrays (str base ".weight")) (throw (ex-info (str "no tensor " base) {:base base})))))))

(defn- qwen3-model [arrays scales]
  (let [deq (dequantizer arrays scales)]
    {:path qwen3
     :cfg {:heads 16 :kv-heads 8 :head-dim 128 :eps 1e-6 :theta 1000000.0}
     :model {:embed (deq "token_embd") :norm (deq "output_norm")
             :layers (vec (for [i (range 28) :let [p (str "blk." i ".")]]
                            {:kind :attn :attn-norm (deq (str p "attn_norm")) :ffn-norm (deq (str p "ffn_norm"))
                             :q (deq (str p "attn_q")) :k (deq (str p "attn_k")) :v (deq (str p "attn_v"))
                             :o (deq (str p "attn_output")) :q-norm (deq (str p "attn_q_norm")) :k-norm (deq (str p "attn_k_norm"))
                             :gate (deq (str p "ffn_gate")) :up (deq (str p "ffn_up")) :down (deq (str p "ffn_down"))}))}}))

(defn- qwen35-model
  "Qwen3.5-0.8B from its GGUF's tensors, by llama.cpp's names: 24 layers
   (blk.24, the MTP block, left out), layer i full attention when i mod 4
   is 3, else GatedDeltaNet."
  [arrays scales]
  (let [deq (dequantizer arrays scales)]
    {:path qwen35
     :cfg {:heads 8 :kv-heads 2 :head-dim 256 :rope-dims 64 :gated? true :eps 1e-6 :theta 1.0e7
           :lin-heads 16 :dk 128 :dv 128 :conv 4}
     :model {:embed (deq "token_embd") :norm (deq "output_norm")
             :layers (vec (for [i (range 24) :let [p (str "blk." i ".")]]
                            (merge {:attn-norm (deq (str p "attn_norm")) :ffn-norm (deq (str p "post_attention_norm"))
                                    :gate (deq (str p "ffn_gate")) :up (deq (str p "ffn_up")) :down (deq (str p "ffn_down"))}
                                   (if (= 3 (mod i 4))
                                     {:kind :attn :q (deq (str p "attn_q")) :k (deq (str p "attn_k")) :v (deq (str p "attn_v"))
                                      :o (deq (str p "attn_output")) :q-norm (deq (str p "attn_q_norm")) :k-norm (deq (str p "attn_k_norm"))}
                                     {:kind :gdn :qkv (deq (str p "attn_qkv")) :z (deq (str p "attn_gate"))
                                      :b (deq (str p "ssm_beta")) :a (deq (str p "ssm_alpha")) :conv-w (deq (str p "ssm_conv1d"))
                                      :A (arrays (str p "ssm_a")) :dt-bias (arrays (str p "ssm_dt.bias"))
                                      :ssm-norm (deq (str p "ssm_norm")) :out (deq (str p "ssm_out"))}))))}}))

(defn check-logits
  "The next-token log-probabilities of two prompts from the prototype's
   forward on MLX, against the llama engine's on the same GGUF: computed
   whole, and as a cached prefill of all but the last token and one step
   (the caches' conventions: KV, conv and recurrent state). label names
   the check; {:keys [path cfg model]} from qwen3-model or qwen35-model."
  [label {:keys [path cfg model]} mutation]
  (let [forward (forward-fn mutation)]
    (with-open [^java.io.Closeable e ((requiring-resolve 'gen.llm.llama/engine) {:model path :max-branches 1 :n-ctx 512})]
      (doseq [text ["The capital of Sweden is a city called" "def fibonacci(n):\n    if n <"]]
        (let [prompt ((requiring-resolve 'gen.llm.llama/tokenize) e text)
              n (count prompt)
              ref (let [x (br/open e prompt)] (br/logprobs (doto x br/dispose!)))
              last-row (fn [logits] (let [t (second (array/shape logits))]
                                      (float-array (array/->clj (array/log-softmax (array/reshape (array/slice logits 1 (dec t) t)
                                                                                                  [(last (array/shape logits))]))))))
              [whole cached] (array/with-backend b
                               (let [ids (fn [from to] (array/array [(subvec prompt from to)] {:dtype :int32}))
                                     pos (fn [from to] (array/array [(vec (range from to))]))
                                     [lw _] (forward cfg model nil (ids 0 n) (pos 0 n))
                                     [_ c] (forward cfg model nil (ids 0 (dec n)) (pos 0 (dec n)))
                                     [ls _] (forward cfg model c (ids (dec n) n) (pos (dec n) n))]
                                 [(last-row lw) (last-row ls)]))
              top (fn [^floats a k] (vec (take k (sort-by #(- (aget a (int %))) (range (alength a))))))
              ref-top (top ref 64)
              verdict (fn [^floats lp]
                        (format "argmax %s, top 5 %s, max |Δ log p| %.3g"
                                (if (= (first ref-top) (first (top lp 1))) "equal" "DIFFERS")
                                (if (= (set (take 5 ref-top)) (set (top lp 5))) "equal" "DIFFER")
                                (reduce max (map #(Math/abs (- (aget ^floats ref (int %)) (aget lp (int %)))) ref-top))))]
          (println (format "%s%s, %s (%d tokens):\n     whole: %s\n    cached: %s"
                           label (if mutation (str ", mutation " mutation) "") (pr-str text) n
                           (verdict whole) (verdict cached))))))))

(let [only (set (rest *command-line-args*))
      run? #(or (empty? only) (only %))
      a3 (when (some run? ["1" "2" "4"]) (check-load qwen3))
      a35 (when (some run? ["1" "5"]) (check-load qwen35))]
  (when (run? "2") (check-quantized a3))
  (when (run? "3") (check-donation))
  (when (run? "4")
    (check-logits "4. Qwen3-0.6B, float16 scales" (qwen3-model a3 :float16) nil)
    (check-logits "4. Qwen3-0.6B, float32 scales" (qwen3-model a3 :float32) nil))
  (doseq [m ["heads" "rope"] :when (only (str "4-" m))]
    (check-logits "4. Qwen3-0.6B, float32 scales" (qwen3-model (or a3 (check-load qwen3)) :float32) m))
  (when (run? "5")
    (check-logits "5. Qwen3.5-0.8B, float32 scales" (qwen35-model a35 :float32) nil))
  (doseq [m ["conv-shift" "conv-kernel" "q-gate"] :when (only (str "5-" m))]
    (check-logits "5. Qwen3.5-0.8B, float32 scales" (qwen35-model (or a35 (check-load qwen35)) :float32) m)))
(shutdown-agents)
