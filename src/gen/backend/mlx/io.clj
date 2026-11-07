(ns gen.backend.mlx.io
  "The MLX backend's files (gen.backend.mlx): GGUF tensors by MLX's loader
   (p/ILoading: mlx_load_gguf), for gen.array/load-gguf.

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
   float64 metadata is refused before MLX misreads it (check-metadata)."
  {:api :internal}
  (:require [clojure.string :as str]
            [gen.array.gguf :as gguf]
            [gen.backend.mlx.array :refer [code stream wrap]]
            [gen.backend.mlx.operands :refer [scalar-handle with-temps]])
  (:import [gen.mlx Mlx]))

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
            scales16 (tmp (Mlx/mlx_io_gguf_get_array g (str base ".scales")))
            hs (Mlx/mlx_astype scales16 f32 s)
            scales (wrap b hs)
            biases (wrap b (case t
                             :q4_1 (Mlx/mlx_astype (tmp (Mlx/mlx_io_gguf_get_array g (str base ".biases"))) f32 s)
                             :q8_0 (Mlx/mlx_multiply hs (tmp (scalar-handle -128.0 :float32)) s)
                             :q4_0 (Mlx/mlx_multiply hs (tmp (scalar-handle -8.0 :float32)) s)))]
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

(defn load-gguf
  "p/-load-gguf on the MLX backend b: the file's tensors by mlx_load_gguf,
   by name."
  [b {:keys [path tensors] :as header}]
  (check-metadata header)
  (let [g (Mlx/mlx_load_gguf (str path) (stream b))]
    (try
      (into {}
            (map (fn [{:keys [name type]}]
                   [name (if (gguf/quantized? type)
                           (quantized-triple b g name type)
                           (wrap b (Mlx/mlx_io_gguf_get_array g name)))]))
            tensors)
      (finally (Mlx/mlx_io_gguf_free g)))))
