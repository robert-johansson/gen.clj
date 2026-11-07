(ns gen.backend.mlx.paged
  "The MLX backend's write into a paged KV cache's pools
   (gen.array/paged-write, p/IFused's :paged-write,
   gen.backend.mlx.nn/fused): one MLX scatter a pool, indexed on its block
   and offset axes (0 and 3 of both of vLLM's layouts), each position's
   key and value a slice. The pools themselves are the scatters' inputs,
   so MLX writes them in place when nothing else holds them (array/compile's
   :donate) and copies them otherwise, as put-along-axis does; the
   composite's put-along-axis on the pools' elements went through a
   reshape, and a compiled step over it measured slower the larger the
   pool (genclj-1daf, 2026-11-07). The composite is the reference
   (gen.array.paged-test). Every device; repeated or out-of-range slots
   are unspecified, as MLX's scatter leaves them."
  {:api :internal}
  (:require [gen.backend.mlx.array :refer [code stream wrap-all]]
            [gen.backend.mlx.operands :refer [dtype-on scalar-handle shape-of with-operands with-temps]])
  (:import [gen.mlx Mlx MlxRuntime]))

(set! *warn-on-reflection* true)

(defn paged-write
  "[k-pool' v-pool'] of gen.array/paged-write's operands (MlxArrays of
   backend b, k and v of the pools' dtype): two scatters on axes 0 and 3."
  [b k-pool v-pool k v slots]
  (let [[_ hk dx bs x] (shape-of k-pool)
        [n] (shape-of slots)
        d (* (long dx) (long x))
        dt (int (code (dtype-on b k-pool)))
        s (stream b)]
    (with-operands b :paged-write [hkp k-pool nil hvp v-pool nil hk' k nil hv v nil hs slots nil]
      (with-temps [tmp]
        (let [s32 (tmp (Mlx/mlx_astype hs (int (code :int32)) s))
              bsh (tmp (scalar-handle bs :int32))
              idx (tmp :vector (MlxRuntime/vectorOf (long-array [(tmp (Mlx/mlx_floor_divide s32 bsh s))
                                                                 (tmp (Mlx/mlx_remainder s32 bsh s))])))
              ;; a position's key [Hk D] as the slice [1 Hk D/x 1 x] at its
              ;; block and offset, its value [Hk D] as [1 Hk D 1]
              ku (tmp (Mlx/mlx_reshape (tmp (Mlx/mlx_astype hk' dt s)) (int-array [n 1 hk dx 1 x]) s))
              vu (tmp (Mlx/mlx_reshape (tmp (Mlx/mlx_astype hv dt s)) (int-array [n 1 hk d 1]) s))
              axes (int-array [0 3])
              ko (Mlx/mlx_scatter hkp idx ku axes s)]
          (try
            (wrap-all b (long-array [ko (Mlx/mlx_scatter hvp idx vu axes s)]))
            (catch Throwable t
              (Mlx/mlx_array_free ko)
              (throw t))))))))
