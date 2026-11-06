(ns gen.backend.mlx.random
  "The MLX backend's randomness (gen.backend.mlx): JAX's partitionable
   Threefry-2x32 (gen.random.threefry's schedule) in MLX's uint32 ops, so
   that a key draws the same bits as on the JVM backend."
  {:api :internal}
  (:require [gen.array.shape :as sh]
            [gen.backend.mlx.array :refer [code stream uint64-code wrap]]
            [gen.backend.mlx.operands :refer [constant made scalar-handle with-temps]]
            [gen.random.threefry :as threefry])
  (:import [gen.mlx Mlx MlxRaw]
           [gen.random.threefry Key]
           [java.lang.foreign Arena ValueLayout]))

(set! *warn-on-reflection* true)

(defn random-uniform
  "-random-uniform: uniforms of shape and dtype (float32 or float64) from
   key, as the JVM backend's for the same key."
  [b ^Key key shape dtype]
  (when-not (#{:float32 :float64} dtype)
    (throw (ex-info (str "random-uniform needs a float dtype, got " dtype) {:dtype dtype})))
  (let [s (stream b)
        n (sh/size shape)
        _ (when (>= n 0x100000000)
            (throw (ex-info "random-uniform: more than 2^32 elements" {:shape shape})))]
    (with-temps [t]
      (let [;; a shared constant, or a new handle (registered) when the cache is full
            konst (fn ^long [v dtype] (if-let [h (constant v dtype)] (long h) (t (scalar-handle v dtype))))
            ;; the key's two words, one buffer
            kw (t (with-open [arena (Arena/ofConfined)]
                    (let [words (.allocateFrom arena ValueLayout/JAVA_INT (int-array [(unchecked-int (.-k0 key)) (unchecked-int (.-k1 key))]))
                          shape (.allocateFrom arena ValueLayout/JAVA_INT (int-array [2]))]
                      (made "mlx_array_new_data" #(MlxRaw/mlx_array_new_data words shape (int 1) (int (code :uint32)))))))
            k0 (t (Mlx/mlx_slice kw (int-array [0]) (int-array [1]) (int-array [1]) s))
            k1 (t (Mlx/mlx_slice kw (int-array [1]) (int-array [2]) (int-array [1]) s))
            ;; counters [hi32(i) lo32(i)] = [0 i]
            c (t (Mlx/mlx_arange 0.0 (double n) 1.0 (int (code :uint32)) s))
            ;; Threefry in MLX's uint32 ops: everything but the key a shared
            ;; constant, so a pending draw pins only its key's buffer
            word (fn ^long [v] (long (konst v :uint32)))
            [w0 w1] (threefry/hash-words
                     {:add (fn ^long [^long x ^long y] (t (Mlx/mlx_add x y s)))
                      :xor (fn ^long [^long x ^long y] (t (Mlx/mlx_bitwise_xor x y s)))
                      :rotl (fn ^long [^long x r]
                              (t (Mlx/mlx_bitwise_or (t (Mlx/mlx_left_shift x (word r) s))
                                                     (t (Mlx/mlx_right_shift x (word (- 32 (long r))) s))
                                                     s)))
                      :word word
                      :zeros (fn ^long [_] (t (Mlx/mlx_zeros (int-array [n]) (int (code :uint32)) s)))}
                     k0 k1 c)
            u (case dtype
                ;; 23 mantissa bits of w0 ^ w1, as gen.random.threefry/uniform-floats
                :float32 (let [bits (t (Mlx/mlx_bitwise_xor w0 w1 s))
                               m (t (Mlx/mlx_bitwise_or (t (Mlx/mlx_right_shift bits (konst 9 :uint32) s))
                                                        (konst 0x3F800000 :uint32) s))
                               f (t (Mlx/mlx_view m (int (code :float32)) s))]
                           (t (Mlx/mlx_subtract f (konst 1.0 :float32) s)))
                ;; 52 mantissa bits of w0 << 32 | w1, as uniform-doubles
                :float64 (let [u64 (int uint64-code)
                               hi (t (Mlx/mlx_astype w0 u64 s))
                               lo (t (Mlx/mlx_astype w1 u64 s))
                               c (fn ^long [v] (t (Mlx/mlx_astype (konst v :int64) u64 s)))
                               m (t (Mlx/mlx_bitwise_or
                                     (t (Mlx/mlx_bitwise_or (t (Mlx/mlx_left_shift hi (c 20) s))
                                                            (t (Mlx/mlx_right_shift lo (c 12) s)) s))
                                     (c 0x3FF0000000000000) s))
                               f (t (Mlx/mlx_view m (int (code :float64)) s))]
                           (t (Mlx/mlx_subtract f (konst 1.0 :float64) s))))]
        (wrap b (Mlx/mlx_reshape u (int-array (map int shape)) s))))))
