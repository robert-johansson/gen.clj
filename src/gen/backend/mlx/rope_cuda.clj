(ns gen.backend.mlx.rope-cuda
  "Where MLX's CUDA rope kernel launches (genclj-6fik). MLX 0.32.4's
   RoPE::eval_gpu (mlx/backend/cuda/rope.cu) runs x [B N… T D], rotating
   dims, over the threads (dims/2, T, B·⌈N/4⌉), in blocks that
   get_block_dims_common (mlx/backend/common/utils.cpp) shares out: the
   three axes' powers of 2 raised in turn, each while it fits its axis,
   to 1,024 threads. That takes no account of CUDA's limit of 64 threads
   on a block's z axis, so where the first two axes are small and the
   third large the launch fails (cudaLaunchKernelExC: invalid argument).
   It needs a rotation of 16 dimensions or fewer: 16 at single positions
   (T = 1) over 128 rows or more (B·⌈N/4⌉), as the tiny Qwen3 fixture's
   chunks of 128 tokens are (a row each, at its own position; the block
   (8 1 128) measured on fenrir's RTX PRO 6000, 2026-11-08). Such shapes
   take gen.array's composite (gen.backend.mlx.nn/fused declines them).
   A rotation of 32 dimensions or more always launches: every real
   model's (Qwen3's 128, Qwen3.5's 64)."
  {:api :internal}
  (:require [gen.backend.mlx.array :refer [device]]
            [gen.backend.mlx.operands :refer [shape-of]]))

(set! *warn-on-reflection* true)

(def ^:private mac?
  (delay (.contains (.toLowerCase (System/getProperty "os.name")) "mac")))

(defn block-dims
  "The block [bx by bz] MLX's get_block_dims_common gives threads
   (d0 d1 d2): each axis's power of 2 raised in turn, while 2·its block
   fits the axis, until 2^10 threads or a round raises none."
  [d0 d1 d2]
  (let [ds [d0 d1 d2]
        block (fn [pows] (mapv #(bit-shift-left 1 (long %)) pows))]
    (loop [pows [0 0 0] i 0 raised? false]
      (cond
        (== 10 (long (reduce + pows))) (block pows)
        (== i 3) (if raised? (recur pows 0 false) (block pows))
        (>= (long (ds i)) (bit-shift-left 1 (inc (long (pows i))))) (recur (update pows i inc) (inc i) true)
        :else (recur pows (inc i) raised?)))))

(defn- ceil-div ^long [^long a ^long b] (quot (+ a (dec b)) b))

(defn launches?
  "Does MLX's CUDA rope kernel launch for x (an MlxArray of backend b)
   rotating dims: a block within CUDA's (1024, 1024, 64) threads and a
   grid within its 65,535 on y and z? True off CUDA. (For B = T = 1 MLX
   may launch its single-position kernel, a block of 1 on z, instead:
   this asks of the other, so it may decline a shape that would launch,
   which the composite then computes.)"
  [b x dims]
  (or (not= :gpu (device b)) @mac?
      (let [s (vec (shape-of x))
            r (count s)
            rows (long (first s))
            t (long (s (- r 2)))
            n (long (reduce * 1 (subvec s 1 (- r 2))))
            d0 (quot (long dims) 2)
            d2 (* rows (ceil-div n 4))
            [_ by bz] (block-dims d0 t d2)]
        (and (<= (long bz) 64)
             (<= (ceil-div t by) 65535)
             (<= (ceil-div d2 bz) 65535)))))
