(ns gen.backend.mlx.paged-attention
  "MLX's Metal kernels for gen.array/paged-attention (p/IFused's
   :paged-attention, gen.backend.mlx.nn/fused): kernels-community's
   paged_attention kernels (vLLM's paged attention, as ported to Metal),
   v1 and v2, through mlx_fast_metal_kernel. One threadgroup of 256 threads a
   query head of a sequence reads the sequence's blocks where they lie in
   the pools (its block table), keeps every position's score in
   threadgroup memory, softmaxes them in float32 and sums the values: no
   gather, one dispatch for all sequences.

   Its operands are gen.array/paged-attention's, the pools in vLLM's
   layouts (gen.array.paged). The tables and lengths are read as uint32,
   each block id clamped to the pool and each length to its table (a lazy
   backend's indices clamp, as take's), and a table of fewer than 8
   entries is padded (metal_kernel passes such an input in constant
   memory, which the kernel does not read).

   v1 holds a sequence's scores in threadgroup memory, sized when the
   kernel is made for its table's positions (MB·BS): MLX makes one kernel
   per table width (a template argument), up to 32 KiB of scores (about
   8,000 positions). Past them v2 runs (genclj-6qb6): a threadgroup a
   partition of 512 positions of a query head, writing its partial
   output, its maximum score and its sum of exponentials, then
   paged_attention_v2_reduce's dispatch rescales and sums the partitions'
   outputs (its threadgroup memory 8 bytes a partition: millions of
   positions). The partials are in the operands' dtype, as upstream
   keeps them. Metal only (the
   GPU on macOS): float32 and the half floats, blocks of 8, 16 or 32
   positions, head sizes up to 256; no VJP (declined under MLX's
   gradients)."
  {:api :internal}
  (:require [clojure.string :as str]
            [gen.backend.mlx.array :refer [code device stream wrap-all]]
            [gen.backend.mlx.operands :refer [dtype-on scalar-handle shape-of with-operands with-temps]]
            [gen.backend.mlx.transforms :refer [*differentiating*]])
  (:import [gen.mlx Mlx MlxRuntime]
           [java.lang.foreign Arena ValueLayout]))

(set! *warn-on-reflection* true)

;; Ported from kernels-community/paged-attention: https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/attention/paged_attention.metal
;;         (with https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/utils.metal and https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/float8.metal)
;; Upstream commit: 28a49f721313edf7c787f48a60c4eba9c4f6f9be
;; License: Apache-2.0 (Copyright 2024-2025 Eric Buehler and the kernels-community
;;          contributors; after vLLM's paged attention kernels, Copyright contributors
;;          to the vLLM project); see gen.clj's THIRD-PARTY-NOTICES
;; Changes: modified by gen.clj. The three files joined (their includes inlined);
;;          the paged_attention and paged_attention_v2_reduce kernels' bodies
;;          kept and their signatures replaced by mlx_fast_metal_kernel's: their
;;          buffers the kernels' named inputs and outputs, their scalars template
;;          arguments, their function constants constexpr and their threadgroup
;;          memory a static array, declared in a prologue (v1: one partition,
;;          without exp_sums and max_logits; v2: partitions of PARTITION_SIZE,
;;          exp_sums and max_logits outputs, then the reduction); the
;;          instantiations and the cache kernels not ported; this header added.
(def ^:private header
  "// Ported from kernels-community/paged-attention: https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/attention/paged_attention.metal
//         (with https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/utils.metal and https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/float8.metal)
// Upstream commit: 28a49f721313edf7c787f48a60c4eba9c4f6f9be
// License: Apache-2.0 (Copyright 2024-2025 Eric Buehler and the kernels-community
//          contributors; after vLLM's paged attention kernels, Copyright contributors
//          to the vLLM project); see gen.clj's THIRD-PARTY-NOTICES
// Changes: modified by gen.clj. The three files joined (their includes inlined);
//          the paged_attention and paged_attention_v2_reduce kernels' bodies
//          kept and their signatures replaced by mlx_fast_metal_kernel's: their
//          buffers the kernels' named inputs and outputs, their scalars template
//          arguments, their function constants constexpr and their threadgroup
//          memory a static array, declared in a prologue (v1: one partition,
//          without exp_sums and max_logits; v2: partitions of PARTITION_SIZE,
//          exp_sums and max_logits outputs, then the reduction); the
//          instantiations and the cache kernels not ported; this header added.

#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

#else

/////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  // Check for nan
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  // Take bits
  uint32_t float_bits = as_type<uint32_t>(x);

  // Round to nearest even
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);

  // Take upper 16 bits
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  // Upper 16 bits are the data and lower 16 bits are 0s
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

/////////////////////////////////////////////////////////////////////////////
// Bfloat struct
/////////////////////////////////////////////////////////////////////////////

struct _MLX_BFloat16 {
  /////////////////////////////////////////////////////////////////////////////
  // Constructors
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions to bfloat

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions from bfloat

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

/////////////////////////////////////////////////////////////////////////////
// Bfloat operators
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

/////////////////////////////////////////////////////////////////////////////
// Binary operators
#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype)    \\
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {              \\
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \\
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)         \\
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) {      \\
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \\
  }                                                                            \\
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) {      \\
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \\
  }

/////////////////////////////////////////////////////////////////////////////
// Arithmetic Operators
#define bfloat_binop(_op_, _operator_)                                         \\
  bfloat_binop_base(_op_, _operator_, _MLX_BFloat16, _MLX_BFloat16,            \\
                    _MLX_BFloat16, float);                                     \\
  bfloat_binop_helper(_op_, _operator_, float, float, float);                  \\
  bfloat_binop_helper(_op_, _operator_, float, half, float);                   \\
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);        \\
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);       \\
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);        \\
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

/////////////////////////////////////////////////////////////////////////////
// Comparison ops
#define bfloat_compop(__op__, __operator__)                                    \\
  bfloat_binop_base(__op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16,  \\
                    float);                                                    \\
  bfloat_binop_helper(__op__, __operator__, bool, float, float);               \\
  bfloat_binop_helper(__op__, __operator__, bool, half, float);                \\
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);             \\
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);            \\
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);             \\
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

/////////////////////////////////////////////////////////////////////////////
// Inplace Operators
#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space)      \\
  constexpr METAL_FUNC addr_space _MLX_BFloat16 &__operator__(                 \\
      addr_space _MLX_BFloat16 &lhs, itype rhs) {                              \\
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \\
    return lhs;                                                                \\
  }                                                                            \\
  constexpr METAL_FUNC addr_space itype &__operator__(addr_space itype &lhs,   \\
                                                      _MLX_BFloat16 rhs) {     \\
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \\
    return lhs;                                                                \\
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype)       \\
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);               \\
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);               \\
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);

#define bfloat_inplace_op(itype)                                               \\
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype);                   \\
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype);                   \\
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype);                   \\
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);

bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space)             \\
  constexpr METAL_FUNC addr_space _MLX_BFloat16 &__operator__(                 \\
      addr_space _MLX_BFloat16 &lhs, _MLX_BFloat16 rhs) {                      \\
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \\
    return lhs;                                                                \\
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__)              \\
  bfloat_inplace_op_helper(__op__, __operator__, device);                      \\
  bfloat_inplace_op_helper(__op__, __operator__, thread);                      \\
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);

bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

/////////////////////////////////////////////////////////////////////////////
// Bfloat typedef
/////////////////////////////////////////////////////////////////////////////

typedef struct _MLX_BFloat16 bfloat16_t;

#endif

#include <metal_stdlib>
using namespace metal;

// Helpers ------------------------------------------------------------
static inline uint as_bits(float x) { return as_type<uint>(x); }
static inline float from_bits(uint b) { return as_type<float>(b); }

// -------------------------------------------------------------------
// FP8 E4M3 (bias = 7)
// -------------------------------------------------------------------
inline float fp8_e4m3_to_float(uchar v) {
  const uint s = v >> 7;
  const uint exp = (v >> 3) & 0xF;
  const uint man = v & 0x7;

  if (exp == 0) { // zero / sub-normal
    if (man == 0)
      return s ? -0.f : 0.f;
    const float m = float(man) / 8.f; // already scaled by 2^-3
    float val = ldexp(m, 1 - 7);      // 2^(1-bias) = 2^-6
    return s ? -val : val;
  }

  if (exp == 0xF) { // Inf / NaN  (E4M3FN keeps only NaN)
    if (man != 0)
      return NAN;
    return s ? -INFINITY : INFINITY;
  }

  const float m = 1.f + float(man) / 8.f;
  float val = ldexp(m, int(exp) - 7);
  return s ? -val : val;
}

// -------------------------------------------------------------------
// FP8 E5M2 (bias = 15)
// -------------------------------------------------------------------
inline float fp8_e5m2_to_float(uchar v) {
  const uint s = v >> 7;
  const uint exp = (v >> 2) & 0x1F;
  const uint man = v & 0x3;

  if (exp == 0) {
    if (man == 0)
      return s ? -0.f : 0.f;
    const float m = float(man) / 4.f;
    float val = ldexp(m, 1 - 15); // 2^(1-bias) = 2^-14
    return s ? -val : val;
  }

  if (exp == 0x1F) {
    if (man != 0)
      return NAN;
    return s ? -INFINITY : INFINITY;
  }

  const float m = 1.f + float(man) / 4.f;
  float val = ldexp(m, int(exp) - 15);
  return s ? -val : val;
}

// -------------------------------------------------------------------
// Encoding helpers (round-to-nearest-even, gradual under-flow, sat-to-∞)
// -------------------------------------------------------------------
namespace detail {
template <int EXP_BITS, int MAN_BITS, int BIAS>
inline uchar fp32_to_fp8(float f) {
  const uint bits = as_bits(f);
  const uint s = bits >> 31;
  const uint abs = bits & 0x7FFFFFFF;

  // NaN propagates, Inf saturates
  if (abs >= 0x7F800000u) {
    return uchar((s << 7) | (((1u << EXP_BITS) - 1u) << MAN_BITS) |
                 (abs != 0x7F800000u));
  }

  int e = int((abs >> 23) & 0xFF) - 127;   // unbiased exponent
  uint m = abs & 0x7FFFFFu;                // 23-bit mantissa
  const int EXP_MAX = (1 << EXP_BITS) - 2; // last finite exponent

  // ---------- Normal path -------------------------------------------------
  int e_fp8 = e + BIAS;
  if (e_fp8 >= 1 && e_fp8 <= EXP_MAX) {
    // round-to-nearest-even
    const int shift = 23 - MAN_BITS;
    uint mant = m >> shift;
    const uint lsb = mant & 1u;
    const uint round = (m >> (shift - 1)) & 1u;
    const uint sticky = (m & ((1u << (shift - 1)) - 1u)) != 0u;
    mant += (round & (sticky | lsb));
    if (mant >> MAN_BITS) { // mantissa overflow
      mant = 0;
      ++e_fp8;
      if (e_fp8 > EXP_MAX)
        return uchar((s << 7) | (((1u << EXP_BITS) - 1u) << MAN_BITS)); // ∞
    }
    return uchar((s << 7) | (uint(e_fp8) << MAN_BITS) |
                 (mant & ((1u << MAN_BITS) - 1u)));
  }

  // ---------- Sub-normal / under-flow ------------------------------------
  if (e_fp8 < 1 - MAN_BITS) // too small -> ±0
    return uchar(s << 7);

  // shift so that exponent becomes 1
  int rshift = (1 - e_fp8) + (23 - MAN_BITS);
  uint mant = (0x800000u | m); // implicit 1
  uint rounded = (mant + (1u << (rshift - 1))) >> rshift;
  if (rounded == 0)
    return uchar(s << 7); // rounds to zero

  return uchar((s << 7) | (rounded & ((1u << MAN_BITS) - 1u)));
}
} // namespace detail

inline uchar float_to_fp8_e4m3(float f) {
  return detail::fp32_to_fp8<4, 3, 7>(f);
}
inline uchar float_to_fp8_e5m2(float f) {
  return detail::fp32_to_fp8<5, 2, 15>(f);
}
// Updated from MLX commit has f70764a

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// ========================================== Generic vector types

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE> struct Vec {};

// A vector type to store FP32 accumulators.
template <typename T> struct FloatVec {};

// Template vector operations.
template <typename Acc, typename A, typename B> inline Acc mul(A a, B b);

template <typename T> inline float sum(T v);

template <typename T> inline float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T> inline float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

// FP32 vector data types.
struct Float8_ {
  float4 x;
  float4 y;
};

template <> struct Vec<float, 1> {
  using Type = float;
};
template <> struct Vec<float, 2> {
  using Type = float2;
};
template <> struct Vec<float, 4> {
  using Type = float4;
};
template <> struct Vec<float, 8> {
  using Type = Float8_;
};

template <> struct FloatVec<float> {
  using Type = float;
};
template <> struct FloatVec<float2> {
  using Type = float2;
};
template <> struct FloatVec<float4> {
  using Type = float4;
};
template <> struct FloatVec<Float8_> {
  using Type = Float8_;
};

template <> inline float mul(float a, float b) { return a * b; }

template <> inline float2 mul(float2 a, float2 b) { return a * b; }

template <> inline float4 mul(float4 a, float4 b) { return a * b; }

template <> inline Float8_ mul(Float8_ a, Float8_ b) {
  Float8_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <> inline float sum(float a) { return a; }

template <> inline float sum(float2 a) { return a.x + a.y; }

template <> inline float sum(float4 a) { return a.x + a.y + a.z + a.w; }

template <> inline float sum(Float8_ a) { return sum(a.x) + sum(a.y); }

inline Float8_ fma(Float8_ a, Float8_ b, Float8_ c) {
  Float8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread float &dst, float src) { dst = src; }
inline void from_float(thread float2 &dst, float2 src) { dst = src; }
inline void from_float(thread float4 &dst, float4 src) { dst = src; }
inline void from_float(thread Float8_ &dst, Float8_ src) { dst = src; }

// BF16 vector data types.
// #if defined(__HAVE_BFLOAT__)

// struct Bfloat8_ {
//   bfloat4 x;
//   bfloat4 y;
// };

// template<>
// struct Vec<bfloat, 1> {
//   using Type = bfloat;
// };
// template<>
// struct Vec<bfloat, 2> {
//   using Type = bfloat2;
// };
// template<>
// struct Vec<bfloat, 4> {
//   using Type = bfloat4;
// };
// template<>
// struct Vec<bfloat, 8> {
//   using Type = Bfloat8_;
// };

// template<>
// struct FloatVec<bfloat> {
//   using Type = float;
// };
// template<>
// struct FloatVec<bfloat2> {
//   using Type = float2;
// };
// template<>
// struct FloatVec<bfloat4> {
//   using Type = float4;
// };
// template<>
// struct FloatVec<Bfloat8_> {
//   using Type = Float8_;
// };

// template<>
// inline float mul(bfloat a, bfloat b) {
//   return (float)a * (float)b;
// }
// template<>
// inline bfloat mul(bfloat a, bfloat b) {
//   return a*b;
// }

// template<>
// inline float2 mul(bfloat2 a, bfloat2 b) {
//   return (float2)a * (float2)b;
// }
// template<>
// inline bfloat2 mul(bfloat2 a, bfloat2 b) {
//   return a * b;
// }

// template<>
// inline float4 mul(bfloat4 a, bfloat4 b) {
//   return (float4)a * (float4)b;
// }
// template<>
// inline bfloat4 mul(bfloat4 a, bfloat4 b) {
//   return a * b;
// }

// template<>
// inline Float8_ mul(Bfloat8_ a, Bfloat8_ b) {
//   Float8_ c;
//   c.x = mul<float4, bfloat4, bfloat4>(a.x, b.x);
//   c.y = mul<float4, bfloat4, bfloat4>(a.y, b.y);
//   return c;
// }
// template<>
// inline Bfloat8_ mul(Bfloat8_ a, Bfloat8_ b) {
//   Bfloat8_ c;
//   c.x = mul<bfloat4, bfloat4, bfloat4>(a.x, b.x);
//   c.y = mul<bfloat4, bfloat4, bfloat4>(a.y, b.y);
//   return c;
// }

// template<>
// inline float sum(bfloat a) {
//   return (float)a;
// }

// template<>
// inline float sum(bfloat2 a) {
//   return (float)a.x + (float)a.y;
// }

// template<>
// inline float sum(bfloat4 a) {
//   return sum(a.x) + sum(a.y);
// }

// template<>
// inline float sum(Bfloat8_ a) {
//   return sum(a.x) + sum(a.y);
// }

// inline float fma(bfloat a, bfloat b, float c) {
//   return (float)a * (float)b + c;
// }

// inline float2 fma(bfloat2 a, bfloat2 b, float2 c) {
//   return (float2)a * (float2)b + c;
// }

// inline float4 fma(bfloat4 a, bfloat4 b, float4 c) {
//   return (float4)a * (float4)b + c;
// }

// inline Float8_ fma(Bfloat8_ a, Bfloat8_ b, Float8_ c) {
//   Float8_ res;
//   res.x = fma((float4)a.x, (float4)b.x, (float4)c.x);
//   res.y = fma((float4)a.y, (float4)b.y, (float4)c.y);
//   return res;
// }
// inline Bfloat8_ fma(Bfloat8_ a, Bfloat8_ b, Bfloat8_ c) {
//   Bfloat8_ res;
//   res.x = (bfloat4)fma((float4)a.x, (float4)b.x, (float4)c.x);
//   res.y = (bfloat4)fma((float4)a.y, (float4)b.x, (float4)c.y);
//   return c;
// }

// inline void from_float(thread bfloat& dst, float src) {
//   dst = static_cast<bfloat>(src);
// }
// inline void from_float(thread bfloat2& dst, float2 src) {
//   dst.x = static_cast<bfloat>(src.x);
//   dst.y = static_cast<bfloat>(src.y);
// }
// inline void from_float(thread bfloat4& dst, float4 src) {
//   dst.x = static_cast<bfloat>(src.x);
//   dst.y = static_cast<bfloat>(src.y);
//   dst.z = static_cast<bfloat>(src.z);
//   dst.w = static_cast<bfloat>(src.w);
// }
// inline void from_float(thread Bfloat8_& dst, Float8_ src) {
//   bfloat4 x;
//   bfloat4 y;
//   from_float(x, src.x);
//   from_float(y, src.y);
//   dst.x = x;
//   dst.y = y;
// }

// #else

struct Bfloat2_ {
  bfloat16_t x;
  bfloat16_t y;
};

struct Bfloat4_ {
  Bfloat2_ x;
  Bfloat2_ y;
};

struct Bfloat8_ {
  Bfloat4_ x;
  Bfloat4_ y;
};

template <> struct Vec<bfloat16_t, 1> {
  using Type = bfloat16_t;
};
template <> struct Vec<bfloat16_t, 2> {
  using Type = Bfloat2_;
};
template <> struct Vec<bfloat16_t, 4> {
  using Type = Bfloat4_;
};
template <> struct Vec<bfloat16_t, 8> {
  using Type = Bfloat8_;
};

template <> struct FloatVec<bfloat16_t> {
  using Type = float;
};
template <> struct FloatVec<Bfloat2_> {
  using Type = float2;
};
template <> struct FloatVec<Bfloat4_> {
  using Type = float4;
};
template <> struct FloatVec<Bfloat8_> {
  using Type = Float8_;
};

template <> inline float mul(bfloat16_t a, bfloat16_t b) {
  return (float)a * (float)b;
}
template <> inline bfloat16_t mul(bfloat16_t a, bfloat16_t b) { return a * b; }

template <> inline float2 mul(Bfloat2_ a, Bfloat2_ b) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f;
}
template <> inline Bfloat2_ mul(Bfloat2_ a, Bfloat2_ b) {
  Bfloat2_ c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <> inline float4 mul(Bfloat4_ a, Bfloat4_ b) {
  float2 x = mul<float2, Bfloat2_, Bfloat2_>(a.x, b.x);
  float2 y = mul<float2, Bfloat2_, Bfloat2_>(a.y, b.y);
  float4 c;
  c.x = x.x;
  c.y = x.y;
  c.z = y.x;
  c.w = y.y;
  return c;
}
template <> inline Bfloat4_ mul(Bfloat4_ a, Bfloat4_ b) {
  Bfloat4_ c;
  c.x = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.x, b.x);
  c.y = mul<Bfloat2_, Bfloat2_, Bfloat2_>(a.y, b.y);
  return c;
}

template <> inline Float8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Float8_ c;
  c.x = mul<float4, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<float4, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}
template <> inline Bfloat8_ mul(Bfloat8_ a, Bfloat8_ b) {
  Bfloat8_ c;
  c.x = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.x, b.x);
  c.y = mul<Bfloat4_, Bfloat4_, Bfloat4_>(a.y, b.y);
  return c;
}

template <> inline float sum(bfloat16_t a) { return (float)a; }

template <> inline float sum(Bfloat2_ a) { return (float)a.x + (float)a.y; }

template <> inline float sum(Bfloat4_ a) { return sum(a.x) + sum(a.y); }

template <> inline float sum(Bfloat8_ a) { return sum(a.x) + sum(a.y); }

inline float fma(bfloat16_t a, bfloat16_t b, float c) {
  return (float)a * (float)b + c;
}
inline bfloat16_t fma(bfloat16_t a, bfloat16_t b, bfloat16_t c) {
  return a * b + c;
}

inline float2 fma(Bfloat2_ a, Bfloat2_ b, float2 c) {
  float2 a_f((float)a.x, (float)a.y);
  float2 b_f((float)b.x, (float)b.y);
  return a_f * b_f + c;
}
inline Bfloat2_ fma(Bfloat2_ a, Bfloat2_ b, Bfloat2_ c) {
  Bfloat2_ res;
  res.x = a.x * b.x + c.x;
  res.y = a.y * b.y + c.y;
  return res;
}

inline float4 fma(Bfloat4_ a, Bfloat4_ b, float4 c) {
  float4 res;
  res.x = fma(a.x.x, b.x.x, c.x);
  res.y = fma(a.x.y, b.x.y, c.y);
  res.z = fma(a.y.x, b.y.x, c.z);
  res.w = fma(a.y.y, b.y.y, c.w);
  return res;
}
inline Bfloat4_ fma(Bfloat4_ a, Bfloat4_ b, Bfloat4_ c) {
  Bfloat4_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline Float8_ fma(Bfloat8_ a, Bfloat8_ b, Float8_ c) {
  float4 x = fma(a.x, b.x, c.x);
  float4 y = fma(a.y, b.y, c.y);
  Float8_ res;
  res.x = x;
  res.y = y;
  return res;
}
inline Bfloat8_ fma(Bfloat8_ a, Bfloat8_ b, Bfloat8_ c) {
  Bfloat8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread bfloat16_t &dst, float src) {
  dst = static_cast<bfloat16_t>(src);
}
inline void from_float(thread Bfloat2_ &dst, float2 src) {
  dst.x = static_cast<bfloat16_t>(src.x);
  dst.y = static_cast<bfloat16_t>(src.y);
}
inline void from_float(thread Bfloat4_ &dst, float4 src) {
  dst.x.x = static_cast<bfloat16_t>(src.x);
  dst.x.y = static_cast<bfloat16_t>(src.y);
  dst.y.x = static_cast<bfloat16_t>(src.z);
  dst.y.y = static_cast<bfloat16_t>(src.w);
}
inline void from_float(thread Bfloat8_ &dst, Float8_ src) {
  Bfloat4_ x;
  Bfloat4_ y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// #endif

// FP16 vector data types.
struct Half8_ {
  half4 x;
  half4 y;
};

template <> struct Vec<half, 1> {
  using Type = half;
};
template <> struct Vec<half, 2> {
  using Type = half2;
};
template <> struct Vec<half, 4> {
  using Type = half4;
};
template <> struct Vec<half, 8> {
  using Type = Half8_;
};

template <> struct FloatVec<half> {
  using Type = float;
};
template <> struct FloatVec<half2> {
  using Type = float2;
};
template <> struct FloatVec<half4> {
  using Type = float4;
};
template <> struct FloatVec<Half8_> {
  using Type = Float8_;
};

template <> inline float mul(half a, half b) { return (float)a * (float)b; }
template <> inline half mul(half a, half b) { return a * b; }

template <> inline float2 mul(half2 a, half2 b) {
  return (float2)a * (float2)b;
}
template <> inline half2 mul(half2 a, half2 b) { return a * b; }

template <> inline float4 mul(half4 a, half4 b) {
  return (float4)a * (float4)b;
}
template <> inline half4 mul(half4 a, half4 b) { return a * b; }

template <> inline Float8_ mul(Half8_ a, Half8_ b) {
  float4 x = mul<float4, half4, half4>(a.x, b.x);
  float4 y = mul<float4, half4, half4>(a.y, b.y);
  Float8_ c;
  c.x = x;
  c.y = y;
  return c;
}
template <> inline Half8_ mul(Half8_ a, Half8_ b) {
  Half8_ c;
  c.x = mul<half4, half4, half4>(a.x, b.x);
  c.y = mul<half4, half4, half4>(a.y, b.y);
  return c;
}

template <> inline float sum(half a) { return (float)a; }

template <> inline float sum(half2 a) { return (float)a.x + (float)a.y; }

template <> inline float sum(half4 a) { return a.x + a.y + a.z + a.w; }

template <> inline float sum(Half8_ a) { return sum(a.x) + sum(a.y); }

inline float fma(half a, half b, float c) { return (float)a * (float)b + c; }

inline float2 fma(half2 a, half2 b, float2 c) {
  return (float2)a * (float2)b + c;
}

inline float4 fma(half4 a, half4 b, float4 c) {
  return (float4)a * (float4)b + c;
}

inline Float8_ fma(Half8_ a, Half8_ b, Float8_ c) {
  float4 x = fma(a.x, b.x, c.x);
  float4 y = fma(a.y, b.y, c.y);
  Float8_ res;
  res.x = x;
  res.y = y;
  return res;
}
inline Half8_ fma(Half8_ a, Half8_ b, Half8_ c) {
  Half8_ res;
  res.x = fma(a.x, b.x, c.x);
  res.y = fma(a.y, b.y, c.y);
  return res;
}

inline void from_float(thread half &dst, float src) {
  dst = static_cast<half>(src);
}
inline void from_float(thread half2 &dst, float2 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
}
inline void from_float(thread half4 &dst, float4 src) {
  dst.x = static_cast<half>(src.x);
  dst.y = static_cast<half>(src.y);
  dst.z = static_cast<half>(src.z);
  dst.w = static_cast<half>(src.w);
}
inline void from_float(thread Half8_ &dst, Float8_ src) {
  half4 x;
  half4 y;
  from_float(x, src.x);
  from_float(y, src.y);
  dst.x = x;
  dst.y = y;
}

// ========================================== FP8 (uchar) vector data types.

// 8‑lane uchar vector – Metal only provides up to uchar4, so build our own.
struct Uchar8_ {
  uchar4 x;
  uchar4 y;
};

// Vec specialisations so Vec<uchar, N>::Type resolves correctly.
template <> struct Vec<uchar, 1> {
  using Type = uchar;
};
template <> struct Vec<uchar, 2> {
  using Type = uchar2;
};
template <> struct Vec<uchar, 4> {
  using Type = uchar4;
};
template <> struct Vec<uchar, 8> {
  using Type = Uchar8_;
};

// General case: not uchar
template <typename T> inline constexpr bool is_uchar() { return false; }

// Specialization: T is uchar
template <> inline constexpr bool is_uchar<uchar>() { return true; }

// Generic fallback – will fail to compile if a required specialisation is
// missing.
template <typename Vec, typename Quant_vec>
inline Vec fp8_convert(const thread Quant_vec &, float scale) {
  static_assert(sizeof(Vec) == 0, \"Missing fp8_convert specialisation\");
}

// ========================================== FP8 → float/half/bfloat
inline float __dequant_single(uchar v, float scale) {
  return fp8_e4m3_to_float(v) * scale;
}

// ---- 1‑lane ----
template <>
inline float fp8_convert<float, uchar>(const thread uchar &in, float scale) {
  return __dequant_single(in, scale);
}
template <>
inline half fp8_convert<half, uchar>(const thread uchar &in, float scale) {
  return half(__dequant_single(in, scale));
}
template <>
inline bfloat16_t fp8_convert<bfloat16_t, uchar>(const thread uchar &in,
                                                 float scale) {
  return bfloat16_t(__dequant_single(in, scale));
}

// ---- 2‑lane ----
template <>
inline float2 fp8_convert<float2, uchar2>(const thread uchar2 &in,
                                          float scale) {
  return float2(__dequant_single(in.x, scale), __dequant_single(in.y, scale));
}
template <>
inline half2 fp8_convert<half2, uchar2>(const thread uchar2 &in, float scale) {
  half2 out;
  out.x = half(__dequant_single(in.x, scale));
  out.y = half(__dequant_single(in.y, scale));
  return out;
}
template <>
inline Bfloat2_ fp8_convert<Bfloat2_, uchar2>(const thread uchar2 &in,
                                              float scale) {
  Bfloat2_ out;
  out.x = bfloat16_t(__dequant_single(in.x, scale));
  out.y = bfloat16_t(__dequant_single(in.y, scale));
  return out;
}

// ---- 4‑lane ----
template <>
inline float4 fp8_convert<float4, uchar4>(const thread uchar4 &in,
                                          float scale) {
  return float4(__dequant_single(in.x, scale), __dequant_single(in.y, scale),
                __dequant_single(in.z, scale), __dequant_single(in.w, scale));
}
template <>
inline half4 fp8_convert<half4, uchar4>(const thread uchar4 &in, float scale) {
  half4 out;
  out.x = half(__dequant_single(in.x, scale));
  out.y = half(__dequant_single(in.y, scale));
  out.z = half(__dequant_single(in.z, scale));
  out.w = half(__dequant_single(in.w, scale));
  return out;
}
template <>
inline Bfloat4_ fp8_convert<Bfloat4_, uchar4>(const thread uchar4 &in,
                                              float scale) {
  Bfloat4_ out;
  out.x.x = bfloat16_t(__dequant_single(in.x, scale));
  out.x.y = bfloat16_t(__dequant_single(in.y, scale));
  out.y.x = bfloat16_t(__dequant_single(in.z, scale));
  out.y.y = bfloat16_t(__dequant_single(in.w, scale));
  return out;
}

// ---- 8‑lane ----
template <>
inline Float8_ fp8_convert<Float8_, Uchar8_>(const thread Uchar8_ &in,
                                             float scale) {
  Float8_ out;
  out.x =
      float4(__dequant_single(in.x.x, scale), __dequant_single(in.x.y, scale),
             __dequant_single(in.x.z, scale), __dequant_single(in.x.w, scale));
  out.y =
      float4(__dequant_single(in.y.x, scale), __dequant_single(in.y.y, scale),
             __dequant_single(in.y.z, scale), __dequant_single(in.y.w, scale));
  return out;
}
template <>
inline Half8_ fp8_convert<Half8_, Uchar8_>(const thread Uchar8_ &in,
                                           float scale) {
  Half8_ out;
  out.x = half4(half(__dequant_single(in.x.x, scale)),
                half(__dequant_single(in.x.y, scale)),
                half(__dequant_single(in.x.z, scale)),
                half(__dequant_single(in.x.w, scale)));
  out.y = half4(half(__dequant_single(in.y.x, scale)),
                half(__dequant_single(in.y.y, scale)),
                half(__dequant_single(in.y.z, scale)),
                half(__dequant_single(in.y.w, scale)));
  return out;
}
template <>
inline Bfloat8_ fp8_convert<Bfloat8_, Uchar8_>(const thread Uchar8_ &in,
                                               float scale) {
  Bfloat8_ out;
  // first 4
  out.x.x.x = bfloat16_t(__dequant_single(in.x.x, scale));
  out.x.x.y = bfloat16_t(__dequant_single(in.x.y, scale));
  out.x.y.x = bfloat16_t(__dequant_single(in.x.z, scale));
  out.x.y.y = bfloat16_t(__dequant_single(in.x.w, scale));
  // second 4
  out.y.x.x = bfloat16_t(__dequant_single(in.y.x, scale));
  out.y.x.y = bfloat16_t(__dequant_single(in.y.y, scale));
  out.y.y.x = bfloat16_t(__dequant_single(in.y.z, scale));
  out.y.y.y = bfloat16_t(__dequant_single(in.y.w, scale));
  return out;
}

// ========================================== Dot product utilities

// TODO(EricLBuehler): optimize with vectorization
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline float qk_dot_(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  using A_vec = typename FloatVec<Vec>::Type;
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += simd_shuffle_xor(qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE> struct Qk_dot {
  template <typename Vec, int N>
  static inline float dot(const threadgroup Vec (&q)[N],
                          const thread Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

// ========================================== Block sum utility

// Utility function for attention softmax.
template <int NUM_WARPS, int NUM_SIMD_LANES>
inline float block_sum(threadgroup float *red_smem, float sum, uint simd_tid,
                       uint simd_lid) {
  // Compute the sum per simdgroup.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Simd leaders store the data to shared memory.
  if (simd_lid == 0) {
    red_smem[simd_tid] = sum;
  }

  // Make sure the data is in shared memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The warps compute the final sums.
  if (simd_lid < NUM_WARPS) {
    sum = red_smem[simd_lid];
  }

  // Parallel reduction inside the simd group.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }

  // Broadcast to other threads.
  return simd_shuffle(sum, 0);
}

// ========================================== Paged Attention kernel

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
")

;; Ported from kernels-community/paged-attention: https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/attention/paged_attention.metal
;;         (with https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/utils.metal and https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/float8.metal)
;; Upstream commit: 28a49f721313edf7c787f48a60c4eba9c4f6f9be
;; License: Apache-2.0 (Copyright 2024-2025 Eric Buehler and the kernels-community
;;          contributors; after vLLM's paged attention kernels, Copyright contributors
;;          to the vLLM project); see gen.clj's THIRD-PARTY-NOTICES
;; Changes: modified by gen.clj. The three files joined (their includes inlined);
;;          the paged_attention and paged_attention_v2_reduce kernels' bodies
;;          kept and their signatures replaced by mlx_fast_metal_kernel's: their
;;          buffers the kernels' named inputs and outputs, their scalars template
;;          arguments, their function constants constexpr and their threadgroup
;;          memory a static array, declared in a prologue (v1: one partition,
;;          without exp_sums and max_logits; v2: partitions of PARTITION_SIZE,
;;          exp_sums and max_logits outputs, then the reduction); the
;;          instantiations and the cache kernels not ported; this header added.
(def ^:private body
  "// Ported from kernels-community/paged-attention: https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/attention/paged_attention.metal
//         (with https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/utils.metal and https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/float8.metal)
// Upstream commit: 28a49f721313edf7c787f48a60c4eba9c4f6f9be
// License: Apache-2.0 (Copyright 2024-2025 Eric Buehler and the kernels-community
//          contributors; after vLLM's paged attention kernels, Copyright contributors
//          to the vLLM project); see gen.clj's THIRD-PARTY-NOTICES
// Changes: modified by gen.clj. The three files joined (their includes inlined);
//          the paged_attention and paged_attention_v2_reduce kernels' bodies
//          kept and their signatures replaced by mlx_fast_metal_kernel's: their
//          buffers the kernels' named inputs and outputs, their scalars template
//          arguments, their function constants constexpr and their threadgroup
//          memory a static array, declared in a prologue (v1: one partition,
//          without exp_sums and max_logits; v2: partitions of PARTITION_SIZE,
//          exp_sums and max_logits outputs, then the reduction); the
//          instantiations and the cache kernels not ported; this header added.

  // gen.clj's prologue: what the kernel's signature and function constants
  // declared, for mlx_fast_metal_kernel's (its inputs q, k_cache, v_cache,
  // block_tables, context_lens and scale, its output out; template
  // arguments T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
  // NUM_SIMD_LANES, PARTITION_SIZE, SHMEM, NUM_KV_HEADS, MAX_BLOCKS,
  // Q_STRIDE, KV_BLOCK_STRIDE, KV_HEAD_STRIDE)
  threadgroup char shared_mem[SHMEM];
  const uint simd_tid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  constexpr bool use_partitioning = PARTITION_SIZE > 0;
  constexpr bool use_alibi = false;
  const int num_kv_heads = NUM_KV_HEADS;
  const float softcapping = 1.0f;
  const int max_num_blocks_per_seq = MAX_BLOCKS;
  const int q_stride = Q_STRIDE;
  const int kv_block_stride = KV_BLOCK_STRIDE;
  const int kv_head_stride = KV_HEAD_STRIDE;
  const device float* k_scale = nullptr;
  const device float* v_scale = nullptr;
  const device float* alibi_slopes = nullptr;
  device float* exp_sums = nullptr;
  device float* max_logits = nullptr;
  const int seq_idx = threadgroup_position_in_grid.y;
  const int partition_idx = threadgroup_position_in_grid.z;
  const int max_num_partitions = threadgroups_per_grid.z;
  const int thread_idx = thread_position_in_threadgroup.x;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const uint32_t context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(NUM_SIMD_LANES / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE
                                       // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, NUM_SIMD_LANES);
  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  const int head_idx = threadgroup_position_in_grid.x;
  const int num_heads = threadgroups_per_grid.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = !use_alibi ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(T)), 1);
  using K_vec = typename Vec<T, VEC_SIZE>::Type;
  using Q_vec = typename Vec<T, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<CACHE_T, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the thread group size is 4, then the first thread in the
  // group has 0, 4, 8, ... th vectors of the query, and the second thread has
  // 1, 5, 9, ... th vectors of the query, and so on.
  const device T *q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  threadgroup Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const device Q_vec *>(q_ptr + vec_idx * VEC_SIZE);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Use fp32 on softmax logits for better accuracy
  threadgroup float *logits = reinterpret_cast<threadgroup float *>(shared_mem);
  // Workspace for reduction
  threadgroup float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(CACHE_T);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const device uint32_t *block_table =
      block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE: The block number is stored in int32. However, we cast it to int64
    // because int32 can lead to overflow when this variable is multiplied by
    // large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the thread group size is 4, then the first thread in the
    // group has 0, 4, 8, ... th vectors of the key, and the second thread has
    // 1, 5, 9, ... th vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * NUM_SIMD_LANES) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const device CACHE_T *k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;

        if constexpr (is_uchar<CACHE_T>()) {
          // FP8 support
          Quant_vec k_vec_quant = *reinterpret_cast<const device Quant_vec *>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = fp8_convert<K_vec, Quant_vec>(k_vec_quant, *k_scale);
        } else {
          // Non-FP8 default
          k_vecs[j] = *reinterpret_cast<const device K_vec *>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<T, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);

      // Apply softcapping
      if (softcapping != 1.0) {
        qk = precise::tanh(qk / softcapping) * softcapping;
      }

      // Add the ALiBi bias if slopes are given.
      if (use_alibi && alibi_slope != 0) {
        // Compute bias with explicit float precision to minimize precision loss
        int position_offset = token_idx - int(context_len) + 1;
        float alibi_bias = alibi_slope * float(position_offset);
        qk += alibi_bias;
      }

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE: It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : max(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each \"warp\" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = max(qk_max, simd_shuffle_xor(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = simd_shuffle(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = exp(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(&red_smem[NUM_WARPS], exp_sum,
                                                 simd_tid, simd_lid);

  // Compute softmax.
  const float inv_sum = divide(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0 && use_partitioning) {
    device float *max_logits_ptr =
        max_logits + seq_idx * num_heads * max_num_partitions +
        head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    device float *exp_sums_ptr = exp_sums +
                                 seq_idx * num_heads * max_num_partitions +
                                 head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(T), BLOCK_SIZE);
  using V_vec = typename Vec<T, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<T, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;
  using V_quant_vec = typename Vec<CACHE_T, V_VEC_SIZE>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = NUM_SIMD_LANES / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE: We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  T zero_value = 0;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE: The block number is stored in int32. However, we cast it to int64
    // because int32 can lead to overflow when this variable is multiplied by
    // large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    Float_L_vec logits_float_vec = *reinterpret_cast<threadgroup Float_L_vec *>(
        logits + token_idx - start_token_idx);
    from_float(logits_vec, logits_float_vec);

    const device CACHE_T *v_ptr = v_cache + physical_block_number * kv_block_stride +
                                  kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        // NOTE: When v_vec contains the tokens that are out of the context,
        // we should explicitly zero out the values since they may contain NaNs.
        // See
        // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
        V_vec v_vec;

        if constexpr (is_uchar<CACHE_T>()) {
          // FP8 support
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const device V_quant_vec *>(v_ptr + offset);
          v_vec = fp8_convert<V_vec, V_quant_vec>(v_quant_vec, *v_scale);
        } else {
          // Non-FP8 default
          v_vec = *reinterpret_cast<const device V_vec *>(v_ptr + offset);
        }

        if (block_idx == num_context_blocks - 1) {
          thread T *v_vec_ptr = reinterpret_cast<thread T *>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] =
                token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += simd_shuffle_xor(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE: A barrier is required because the shared memory space for logits
  // is reused for the output.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Perform reduction across warps.
  threadgroup float *out_smem =
      reinterpret_cast<threadgroup float *>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      threadgroup float *dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lower warps update the output.
    if (warp_idx < mid) {
      const threadgroup float *src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write the final output.
  if (warp_idx == 0) {
    device T *out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        *(out_ptr + row_idx) = T(accs[i]);
      }
    }
  }")

;; Ported from kernels-community/paged-attention: https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/attention/paged_attention.metal
;; Upstream commit: 28a49f721313edf7c787f48a60c4eba9c4f6f9be
;; License: Apache-2.0 (Copyright 2024-2025 Eric Buehler and the kernels-community
;;          contributors; after vLLM's paged attention kernels, Copyright contributors
;;          to the vLLM project); see gen.clj's THIRD-PARTY-NOTICES
;; Changes: modified by gen.clj. The paged_attention_v2_reduce kernel's body
;;          kept and its signature replaced by mlx_fast_metal_kernel's: its
;;          buffers the kernel's named inputs and output, max_num_partitions a
;;          template argument and its threadgroup memory a static array,
;;          declared in a prologue; this header added.
(def ^:private reduce-body
  "// Ported from kernels-community/paged-attention: https://huggingface.co/kernels-community/paged-attention/blob/28a49f721313edf7c787f48a60c4eba9c4f6f9be/paged-attention-metal/attention/paged_attention.metal
// Upstream commit: 28a49f721313edf7c787f48a60c4eba9c4f6f9be
// License: Apache-2.0 (Copyright 2024-2025 Eric Buehler and the kernels-community
//          contributors; after vLLM's paged attention kernels, Copyright contributors
//          to the vLLM project); see gen.clj's THIRD-PARTY-NOTICES
// Changes: modified by gen.clj. The paged_attention_v2_reduce kernel's body
//          kept and its signature replaced by mlx_fast_metal_kernel's: its
//          buffers the kernel's named inputs and output, max_num_partitions a
//          template argument and its threadgroup memory a static array,
//          declared in a prologue; this header added.

  // gen.clj's prologue: what the kernel's signature declared, for
  // mlx_fast_metal_kernel's (its inputs exp_sums, max_logits, tmp_out and
  // context_lens, its output out; template arguments T, HEAD_SIZE,
  // NUM_THREADS, NUM_SIMD_LANES, PARTITION_SIZE, SHMEM, MAX_PARTITIONS)
  threadgroup char shared_mem[SHMEM];
  const uint simd_tid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int max_num_partitions = MAX_PARTITIONS;
  const int num_heads = threadgroups_per_grid.x;
  const int head_idx = threadgroup_position_in_grid.x;
  const int seq_idx = threadgroup_position_in_grid.y;
  const uint32_t context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    device T *out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const device T *tmp_out_ptr =
        tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE;
         i += threads_per_threadgroup.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / NUM_SIMD_LANES;
  const int warp_idx = simd_tid;
  const int lane = simd_lid;

  // Workspace for reduction.
  threadgroup float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  threadgroup float *shared_max_logits =
      reinterpret_cast<threadgroup float *>(shared_mem);
  const device float *max_logits_ptr =
      max_logits + seq_idx * num_heads * max_num_partitions +
      head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = thread_position_in_threadgroup.x; i < num_partitions;
       i += threads_per_threadgroup.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = max(max_logit, l);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = NUM_SIMD_LANES / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = max(max_logit, simd_shuffle_xor(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = simd_shuffle(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  threadgroup float *shared_exp_sums = reinterpret_cast<threadgroup float *>(
      shared_mem + sizeof(float) * num_partitions);
  const device float *exp_sums_ptr = exp_sums +
                                     seq_idx * num_heads * max_num_partitions +
                                     head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = thread_position_in_threadgroup.x; i < num_partitions;
       i += threads_per_threadgroup.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * exp(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  global_exp_sum = block_sum<NUM_WARPS, NUM_SIMD_LANES>(
      &red_smem[NUM_WARPS], global_exp_sum, simd_tid, simd_lid);
  const float inv_global_exp_sum = divide(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const device T *tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  device T *out_ptr =
      out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = thread_position_in_threadgroup.x; i < HEAD_SIZE;
       i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
             inv_global_exp_sum;
    }
    out_ptr[i] = T(acc);
  }")

;; ---------------------------------------------------------------------------
;; The kernel and its dispatch (gen.clj's)
;; ---------------------------------------------------------------------------

(def ^:private threads 256)
(def ^:private lanes 32)
(def ^:private threadgroup-memory
  "Apple GPUs' threadgroup memory a threadgroup (MTLDevice's
   maxThreadgroupMemoryLength)."
  32768)

(def ^:private item-bytes {:float32 4 :float16 2 :bfloat16 2})

(defn- strings ^long [xs]
  (let [v (Mlx/mlx_vector_string_new)]
    (doseq [x xs] (Mlx/mlx_vector_string_append_value v ^String x))
    v))

(def ^:private kernels
  "The kernels' handles by [name source], each made once (MLX compiles a
   kernel at its first run for each set of template arguments, and caches
   it), kept for the process."
  (atom {}))

(defn- kernel
  "The handle of kernel name, its source (after header) with inputs ins
   and outputs outs (their names)."
  ^long [^String nm ^String source ins outs]
  (locking kernels
    (or (get @kernels [nm source])
        (let [is (strings ins)
              os (strings outs)
              h (try
                  (with-open [arena (Arena/ofConfined)]
                    (Mlx/mlx_fast_metal_kernel_new (.allocateFrom arena nm) is os
                                                   (.allocateFrom arena source) (.allocateFrom arena ^String header)
                                                   true false))
                  (finally (Mlx/mlx_vector_string_free is) (Mlx/mlx_vector_string_free os)))]
          (swap! kernels assoc [nm source] h)
          h))))

(def ^:private partition-size
  "v2's positions a partition (kernels-community's paged_attention_v2)."
  512)

(defn- v2-body
  "v2's source: body, its partitions' maxima and sums (exp_sums,
   max_logits) the kernel's outputs."
  ^String []
  (str/replace body "  device float* exp_sums = nullptr;\n  device float* max_logits = nullptr;\n" ""))

(def ^:private mac?
  (delay (.contains (.toLowerCase (System/getProperty "os.name")) "mac")))

(defn- scores-bytes
  "The kernel's threadgroup memory for the scores of n positions (or the
   warps' outputs, if more): its SHMEM."
  ^long [^long n ^long d]
  (max (* 4 n) (* (quot (quot threads lanes) 2) d 4)))

(defn- shared-bytes
  "v1's threadgroup memory: the scores of mb·bs positions (or the warps'
   outputs, if more), then the query and the reductions'."
  ^long [^long mb ^long bs ^long d dtype]
  (+ (scores-bytes (* mb bs) d) (* d (long (item-bytes dtype))) (* 2 (quot threads lanes) 4)))

(defn- partitions
  "v2's partitions of a table of mb blocks of bs."
  ^long [^long mb ^long bs]
  (quot (+ (* mb bs) (dec (long partition-size))) (long partition-size)))

(defn- v2-fits?
  "Does v2 fit in threadgroup memory: its partitions' scores, and the
   reduction's maxima and sums of p partitions?"
  [^long p ^long d dtype]
  (and (<= (shared-bytes 1 partition-size d dtype) threadgroup-memory)
       (<= (+ (* 8 p) (* 2 (quot threads lanes) 4)) threadgroup-memory)))

(defn- table-width
  "The width the kernel reads tables of shape [B MB] at: MB, or more so
   that they hold 8 entries (metal_kernel passes a smaller input in
   constant memory, which the kernel does not read)."
  ^long [[nq mb]]
  (max (long mb) (quot (+ 7 (long nq)) (long nq))))

(defn applies?
  "Does a kernel compute gen.array/paged-attention of these operands on
   backend b's device (Metal): v1 where its scores fit in threadgroup
   memory, else v2?"
  [b q k-pool tables]
  (let [dt (dtype-on b q)
        [_ _ _ bs x] (shape-of k-pool)
        d (peek (shape-of q))
        mb (table-width (shape-of tables))]
    (and (= :gpu (device b)) @mac? (not *differentiating*)
         (item-bytes dt)
         (= dt (dtype-on b k-pool))
         (== (long x) (quot 16 (long (item-bytes dt))))
         (#{8 16 32} bs)
         (<= (long d) 256)
         (or (<= (shared-bytes mb bs d dt) threadgroup-memory)
             (v2-fits? (partitions mb bs) d dt)))))

(defn- config
  "A kernel's configuration: outputs [[shape dtype] …], grid, threadgroup
   and template arguments [[name dtype-or-int] …]. Free it."
  [arena outputs [gx gy gz] [tx ty tz] templates]
  (let [cfg (Mlx/mlx_fast_metal_kernel_config_new)]
    (doseq [[shape dt] outputs]
      (Mlx/mlx_fast_metal_kernel_config_add_output_arg cfg (.allocateFrom ^Arena arena ValueLayout/JAVA_INT (int-array (map int shape)))
                                                        (long (count shape)) (int (code dt))))
    (Mlx/mlx_fast_metal_kernel_config_set_grid cfg (int gx) (int gy) (int gz))
    (Mlx/mlx_fast_metal_kernel_config_set_thread_group cfg (int tx) (int ty) (int tz))
    (doseq [[nm x] templates]
      (if (keyword? x)
        (Mlx/mlx_fast_metal_kernel_config_add_template_arg_dtype cfg ^String nm (int (code x)))
        (Mlx/mlx_fast_metal_kernel_config_add_template_arg_int cfg ^String nm (int x))))
    cfg))

(defn- apply-kernel
  "The outputs (handles) of kernel k on the input handles ins under cfg,
   on stream s; tmp holds them (with-temps)."
  [tmp k cfg ins s]
  (MlxRuntime/vectorArrays (long (tmp :vector (Mlx/mlx_fast_metal_kernel_apply (long k) (long (tmp :vector (MlxRuntime/vectorOf (long-array ins)))) (long cfg) (long s))))))

(defn paged-attention
  "[B Hq D] of gen.array/paged-attention's operands (MlxArrays of backend
   b; opts {:scale}) by the kernels: v1 (one dispatch, grid (Hq·256, B,
   1), threadgroups of 256) where its scores fit in threadgroup memory,
   else v2 (grid (Hq·256, B, P) over partitions of 512 positions, each's
   output, maximum and sum written, then the reduction's dispatch, grid
   (Hq·256, B, 1))."
  [b q k-pool v-pool tables lens {:keys [scale]}]
  (let [[nq hq d] (shape-of q)
        [nb hk _ bs _] (shape-of k-pool)
        [_ mb] (shape-of tables)
        width (table-width (shape-of tables))
        dt (dtype-on b q)
        v1? (<= (shared-bytes width bs d dt) threadgroup-memory)
        p (partitions width bs)
        s (stream b)
        u32 (int (code :uint32))
        attention [["T" dt] ["CACHE_T" dt] ["HEAD_SIZE" d] ["BLOCK_SIZE" bs] ["NUM_THREADS" threads] ["NUM_SIMD_LANES" lanes]
                   ["PARTITION_SIZE" (if v1? 0 partition-size)]
                   ["SHMEM" (scores-bytes (if v1? (* width (long bs)) partition-size) d)]
                   ["NUM_KV_HEADS" hk] ["MAX_BLOCKS" width] ["Q_STRIDE" (* (long hq) (long d))]
                   ["KV_BLOCK_STRIDE" (* (long hk) (long d) (long bs))] ["KV_HEAD_STRIDE" (* (long d) (long bs))]]
        ins ["q" "k_cache" "v_cache" "block_tables" "context_lens" "scale"]]
    (with-open [arena (Arena/ofConfined)]
      (let [cfgs (if v1?
                   [(config arena [[[nq hq d] dt]] [(* (long hq) threads) nq 1] [threads 1 1] attention)]
                   [(config arena [[[nq hq p] :float32] [[nq hq p] :float32] [[nq hq p d] dt]] [(* (long hq) threads) nq p]
                            [threads 1 1] attention)
                    (config arena [[[nq hq d] dt]] [(* (long hq) threads) nq 1] [threads 1 1]
                            [["T" dt] ["HEAD_SIZE" d] ["NUM_THREADS" threads] ["NUM_SIMD_LANES" lanes]
                             ["PARTITION_SIZE" partition-size] ["SHMEM" (* 8 p)] ["MAX_PARTITIONS" p]])])]
        (try
          (with-operands b :paged-attention [hq' q nil hk' k-pool nil hv v-pool nil ht tables nil hl lens nil]
            (with-temps [tmp]
              (let [clamp (fn [h hi] (tmp (Mlx/mlx_clip h (tmp (scalar-handle 0 :int32)) (tmp (scalar-handle hi :int32)) s)))
                    t32 (tmp (Mlx/mlx_astype ht (int (code :int32)) s))
                    t32 (if (> width (long mb))
                          (tmp (Mlx/mlx_pad t32 (int-array [1]) (int-array [0]) (int-array [(- width (long mb))])
                                            (tmp (scalar-handle 0 :int32)) "constant" s))
                          t32)
                    tbl (tmp (Mlx/mlx_astype (clamp t32 (dec (long nb))) u32 s))
                    len (tmp (Mlx/mlx_astype (clamp (tmp (Mlx/mlx_astype hl (int (code :int32)) s)) (* (long mb) (long bs))) u32 s))
                    hs (tmp (scalar-handle (double scale) :float32))
                    args [hq' hk' hv tbl len hs]]
                (if v1?
                  (first (wrap-all b (apply-kernel tmp (kernel "gen_paged_attention_v1" body ins ["out"]) (first cfgs) args s)))
                  (let [[exp-sums max-logits tmp-out] (apply-kernel tmp (kernel "gen_paged_attention_v2" (v2-body) ins
                                                                                 ["exp_sums" "max_logits" "out"])
                                                                    (first cfgs) args s)
                        _ (run! tmp [exp-sums max-logits tmp-out])]
                    (first (wrap-all b (apply-kernel tmp (kernel "gen_paged_attention_v2_reduce" reduce-body
                                                                 ["exp_sums" "max_logits" "tmp_out" "context_lens"] ["out"])
                                                     (second cfgs) [exp-sums max-logits tmp-out len] s))))))))
          (finally (run! #(Mlx/mlx_fast_metal_kernel_config_free (long %)) cfgs)))))))
