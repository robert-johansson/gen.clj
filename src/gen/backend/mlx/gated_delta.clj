(ns gen.backend.mlx.gated-delta
  "MLX's Metal kernel for gen.array/gated-delta (p/IFused's
   :gated-delta, gen.backend.mlx.nn/fused): mlx-lm's gated-delta kernel,
   one dispatch for a read of any length. Each threadgroup holds 4 rows
   of one head's state (a row of Dk across a SIMD group of 32 threads,
   Dk/32 entries each, in float32 registers) and walks the tokens in
   order: decay, the state read at the key (simd_sum), the corrected
   value written back, the state read at the query; the output and the
   state after the last token in the operands' dtype. The token loop of
   the composite, computed in float32.

   Its operands are gen.array/gated-delta's, every head with its own keys
   (Hk = Hv), the decay exp(g) computed by MLX first: mlx-lm's kernel
   takes the decay, gated-delta the log decay. Metal only (the GPU on
   macOS): MLX's CPU device and CUDA have no Metal kernels, and float64
   none at all; Dk a multiple of 32."
  {:api :internal}
  (:require [gen.backend.mlx.array :refer [code stream wrap-all]]
            [gen.backend.mlx.operands :refer [dtype-on scalar-handle shape-of with-operands with-temps]])
  (:import [gen.mlx Mlx MlxRuntime]
           [java.lang.foreign Arena ValueLayout]))

(set! *warn-on-reflection* true)

;; Ported from mlx-lm: https://github.com/ml-explore/mlx-lm/blob/760c5abcc80565132a43126561601152356de16a/mlx_lm/models/gated_delta.py
;; Upstream commit: 760c5abcc80565132a43126561601152356de16a
;; License: MIT (Copyright 2023 Apple Inc.); see gen.clj's THIRD-PARTY-NOTICES
;; Changes: the Metal source _make_gated_delta_kernel formats without a
;;          mask and with a gate per head (has_mask=False,
;;          vectorized=False), as one string; this header added. The
;;          kernel's inputs, outputs, template arguments, grid and
;;          threadgroup are gated_delta_kernel's, made in Clojure below.
(def ^:private source
  "// Ported from mlx-lm: https://github.com/ml-explore/mlx-lm/blob/760c5abcc80565132a43126561601152356de16a/mlx_lm/models/gated_delta.py
// Upstream commit: 760c5abcc80565132a43126561601152356de16a
// License: MIT (Copyright 2023 Apple Inc.); see gen.clj's THIRD-PARTY-NOTICES
// Changes: the Metal source _make_gated_delta_kernel formats without a
//          mask and with a gate per head (has_mask=False,
//          vectorized=False), as one string; this header added.

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

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
          if (true) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
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

(def ^:private kernel
  "The kernel's handle, made once (MLX compiles it at its first run for
   each set of template arguments, and caches it), kept for the process."
  (delay
    (let [ins (strings ["q" "k" "v" "g" "beta" "state_in" "T"])
          outs (strings ["y" "state_out"])]
      (try
        (with-open [arena (Arena/ofConfined)]
          (Mlx/mlx_fast_metal_kernel_new (.allocateFrom arena "gen_gated_delta_step") ins outs
                                         (.allocateFrom arena ^String source) (.allocateFrom arena "")
                                         true false))
        (finally (Mlx/mlx_vector_string_free ins) (Mlx/mlx_vector_string_free outs))))))

(defn applies?
  "Does the kernel compute gated-delta of these operands on backend b's
   device (Metal)? Dk a multiple of 32, a float dtype Metal has."
  [b q]
  (let [dk (long (peek (shape-of q)))]
    (and (pos? dk) (zero? (rem dk 32)) (#{:float32 :float16 :bfloat16} (dtype-on b q)))))

(defn gated-delta
  "[y state'] of gen.array/gated-delta's operands (MlxArrays of backend b,
   one float dtype) by the kernel: one dispatch, grid (32, Dv, B·H),
   threadgroups (32, 4, 1), as mlx-lm's gated_delta_kernel."
  [b q k v g beta state]
  (let [[bsz t h dk] (shape-of q)
        dv (peek (shape-of v))
        dt (dtype-on b q)
        s (stream b)
        cfg (Mlx/mlx_fast_metal_kernel_config_new)]
    (try
      (with-open [arena (Arena/ofConfined)]
        (doseq [shape [[bsz t h dv] [bsz h dv dk]]]
          (Mlx/mlx_fast_metal_kernel_config_add_output_arg cfg (.allocateFrom arena ValueLayout/JAVA_INT (int-array (map int shape)))
                                                            (long (count shape)) (int (code dt))))
        (Mlx/mlx_fast_metal_kernel_config_set_grid cfg (int 32) (int dv) (int (* (long bsz) (long h))))
        (Mlx/mlx_fast_metal_kernel_config_set_thread_group cfg (int 32) (int 4) (int 1))
        (Mlx/mlx_fast_metal_kernel_config_add_template_arg_dtype cfg "InT" (int (code dt)))
        (doseq [[nm x] [["Dk" dk] ["Dv" dv] ["Hk" h] ["Hv" h]]]
          (Mlx/mlx_fast_metal_kernel_config_add_template_arg_int cfg ^String nm (int x)))
        (with-operands b :gated-delta [hq q nil hk k nil hv v nil hg g nil hb beta nil hs state nil]
          (with-temps [tmp]
            (let [decay (tmp (Mlx/mlx_exp hg s))
                  ht (tmp (scalar-handle t :int32))
                  in (tmp :vector (MlxRuntime/vectorOf (long-array [hq hk hv decay hb hs ht])))
                  out (tmp :vector (Mlx/mlx_fast_metal_kernel_apply (long @kernel) in cfg s))]
              (wrap-all b (MlxRuntime/vectorArrays out))))))
      (finally (Mlx/mlx_fast_metal_kernel_config_free cfg)))))
