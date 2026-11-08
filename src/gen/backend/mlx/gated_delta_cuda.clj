(ns gen.backend.mlx.gated-delta-cuda
  "MLX's CUDA kernel for gen.array/gated-delta (p/IFused's :gated-delta,
   gen.backend.mlx.nn/fused): mlx-lm's gated-delta Metal kernel
   (gen.backend.mlx.gated-delta) in CUDA, through mlx_fast_cuda_kernel.
   A warp holds one row of one head's state (Dk/32 entries a lane, in
   float32 registers), four rows a block, and walks the tokens in order:
   decay, the state read at the key (a warp sum), the corrected value
   written back, the state read at the query; the output and the state
   after the last token in the operands' dtype. The token loop of the
   composite in one launch, computed in float32.

   On MLX's CUDA backend the composite's token loop took about 10 kernels
   a GatedDeltaNet layer of a decode step, and MLX's host encodes a
   kernel in about 5 µs (genclj-yesw): this one replaces them. It takes
   reads shorter than gated-delta's chunk (its :chunk-size, 64 by
   default), where the composite runs the token loop; longer reads keep
   the chunked composite. MLX 0.32.4's own fast::gated_delta_update has no
   CUDA kernel (its fallback is a composite).

   Its operands are gen.array/gated-delta's (every head with its own keys,
   Hk = Hv), the decay exp(g) computed by MLX first, as the Metal
   kernel's. MLX's CUDA backend only (the GPU off macOS): float32 and
   float16; Dk a multiple of 32, Dv of 4; no VJP (declined under MLX's
   gradients). NVRTC compiles it once a process for each dtype and head
   shape, at its first launch."
  {:api :internal}
  (:require [gen.backend.mlx.array :refer [code device stream wrap-all]]
            [gen.backend.mlx.operands :refer [dtype-on shape-of with-operands with-temps]]
            [gen.backend.mlx.transforms :refer [*differentiating*]])
  (:import [gen.mlx Mlx MlxRuntime]
           [java.lang.foreign Arena ValueLayout]))

(set! *warn-on-reflection* true)

;; Ported from mlx-lm: https://github.com/ml-explore/mlx-lm/blob/760c5abcc80565132a43126561601152356de16a/mlx_lm/models/gated_delta.py
;; Upstream commit: 760c5abcc80565132a43126561601152356de16a
;; License: MIT (Copyright 2023 Apple Inc.); see gen.clj's THIRD-PARTY-NOTICES
;; Changes: modified by gen.clj. The Metal source _make_gated_delta_kernel
;;          formats without a mask and with a gate per head (has_mask=False,
;;          vectorized=False), translated to CUDA for mlx_fast_cuda_kernel:
;;          the thread positions from blockIdx and threadIdx, simd_sum as a
;;          warp's butterfly of __shfl_xor_sync, the token count read from
;;          q's shape (not an input), the operands cast to float explicitly;
;;          this header added. The kernel's inputs, outputs, template
;;          arguments, grid and blocks are gated_delta_kernel's, made in
;;          Clojure below.
(def ^:private source
  "// Ported from mlx-lm: https://github.com/ml-explore/mlx-lm/blob/760c5abcc80565132a43126561601152356de16a/mlx_lm/models/gated_delta.py
// Upstream commit: 760c5abcc80565132a43126561601152356de16a
// License: MIT (Copyright 2023 Apple Inc.); see gen.clj's THIRD-PARTY-NOTICES
// Changes: modified by gen.clj. The Metal source _make_gated_delta_kernel
//          formats without a mask and with a gate per head (has_mask=False,
//          vectorized=False), translated to CUDA for mlx_fast_cuda_kernel:
//          the thread positions from blockIdx and threadIdx, simd_sum as a
//          warp's butterfly of __shfl_xor_sync, the token count read from
//          q's shape (not an input), the operands cast to float explicitly;
//          this header added.

        const int n = blockIdx.z;
        const int b_idx = n / Hv;
        const int hv_idx = n % Hv;
        const int hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        const int T = q_shape[1];

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;

        const int dk_idx = threadIdx.x;
        const int dv_idx = blockIdx.y * blockDim.y + threadIdx.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * static_cast<float>(g_[hv_idx]);
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
          }
          for (int o = 16; o > 0; o /= 2) {
            kv_mem += __shfl_xor_sync(0xffffffffu, kv_mem, o);
          }

          auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
            out += state[i] * static_cast<float>(q_[s_idx]);
          }
          for (int o = 16; o > 0; o /= 2) {
            out += __shfl_xor_sync(0xffffffffu, out, o);
          }
          if (dk_idx == 0) {
            y_[dv_idx] = static_cast<InT>(out);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y_ += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
")

(defn- strings ^long [xs]
  (let [v (Mlx/mlx_vector_string_new)]
    (doseq [x xs] (Mlx/mlx_vector_string_append_value v ^String x))
    v))

(defn- make-kernel
  "A new kernel handle of the source src."
  ^long [^String src]
  (let [ins (strings ["q" "k" "v" "g" "beta" "state_in"])
        outs (strings ["y" "state_out"])]
    (try
      (with-open [arena (Arena/ofConfined)]
        (Mlx/mlx_fast_cuda_kernel_new (.allocateFrom arena "gen_gated_delta_step") ins outs
                                      (.allocateFrom arena src) (.allocateFrom arena "") true (int 0)))
      (finally (Mlx/mlx_vector_string_free ins) (Mlx/mlx_vector_string_free outs)))))

(def ^:private kernel
  "The kernel for a source, made once and kept for the process (MLX
   compiles it at its first launch for each set of template arguments,
   and caches it): a test may plant a bug in the source."
  (memoize make-kernel))

(def ^:private mac?
  (delay (.contains (.toLowerCase (System/getProperty "os.name")) "mac")))

(defn applies?
  "Does the kernel compute gated-delta of these operands (q [B T H Dk], v
   [B T H Dv]) on backend b's device (CUDA)? A read shorter than chunk
   tokens, Dk a multiple of 32 and Dv of 4, float32 or float16."
  [b q v chunk]
  (let [[_ t _ dk] (shape-of q)
        dv (peek (shape-of v))]
    (boolean
     (and (= :gpu (device b)) (not @mac?) (not *differentiating*)
          (#{:float32 :float16} (dtype-on b q))
          (< (long t) (long (or chunk 64)))
          (pos? (long dk)) (zero? (rem (long dk) 32))
          (pos? (long dv)) (zero? (rem (long dv) 4))))))

(defn gated-delta
  "[y state'] of gen.array/gated-delta's operands (MlxArrays of backend b,
   one float dtype) by the kernel: one launch, grid (32, Dv, B·H)
   threads, blocks (32, 4, 1), as mlx-lm's gated_delta_kernel."
  [b q k v g beta state]
  (let [[bsz t h dk] (shape-of q)
        dv (peek (shape-of v))
        dt (dtype-on b q)
        s (stream b)
        cfg (Mlx/mlx_fast_cuda_kernel_config_new)]
    (try
      (with-open [arena (Arena/ofConfined)]
        (doseq [shape [[bsz t h dv] [bsz h dv dk]]]
          (Mlx/mlx_fast_cuda_kernel_config_add_output_arg cfg (.allocateFrom arena ValueLayout/JAVA_INT (int-array (map int shape)))
                                                           (long (count shape)) (int (code dt))))
        (Mlx/mlx_fast_cuda_kernel_config_set_grid cfg (int 32) (int dv) (int (* (long bsz) (long h))))
        (Mlx/mlx_fast_cuda_kernel_config_set_thread_group cfg (int 32) (int 4) (int 1))
        (Mlx/mlx_fast_cuda_kernel_config_add_template_arg_dtype cfg "InT" (int (code dt)))
        (doseq [[nm x] [["Dk" dk] ["Dv" dv] ["Hk" h] ["Hv" h]]]
          (Mlx/mlx_fast_cuda_kernel_config_add_template_arg_int cfg ^String nm (int x)))
        (with-operands b :gated-delta [hq q nil hk k nil hv v nil hg g nil hb beta nil hs state nil]
          (with-temps [tmp]
            (let [decay (tmp (Mlx/mlx_exp hg s))
                  in (tmp :vector (MlxRuntime/vectorOf (long-array [hq hk hv decay hb hs])))
                  out (tmp :vector (Mlx/mlx_fast_cuda_kernel_apply (long (kernel source)) in cfg s))]
              (wrap-all b (MlxRuntime/vectorArrays out))))))
      (finally (Mlx/mlx_fast_cuda_kernel_config_free cfg)))))
