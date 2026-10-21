#pragma once

// Signed int4 (i4), per-token G64 KV-cache codec shared by append and causal-attention kernels.
// This is the RK4V4E8 (rk4v4-e8) 4090 live-4090 storage: 4-bit keys (E8-lattice refined on the
// encode side) + 4-bit values, group-64 FP16 scales. i4 = 2 codes/byte: a 256-dim vector is 128
// bytes of packed codes + 4 FP16 group scales. The decode is i4 unpack -> int8 -> s8/f16 MMA
// (sm_89-safe: no NVFP4 block-scale GEMM, no e2m1 cvt). This header owns index math, the scalar
// encode (quant/pack), and vectorized decode (unpack/dequant); there is deliberately no standalone
// transcode kernel in the production path.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheI4HeadDim = 256;
inline constexpr int kKVCacheI4Group   = 64;
inline constexpr int kKVCacheI4Groups  = kKVCacheI4HeadDim / kKVCacheI4Group;
// 256 dims at 4 bits each = 128 packed bytes (2 codes/byte).
inline constexpr int kKVCacheI4CodeBytes = kKVCacheI4HeadDim / 2;

// ---------------------------------------------------------------------------
// Index math (byte-level into the i4 code plane, element-level into the fp16
// scale plane). Mirrors int8_g64_codec.cuh; the code index is in packed-byte
// units (d/2) because two i4 codes share one byte.
// ---------------------------------------------------------------------------

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_i4_code_index(int physical_page, int kv_head, int d, int page_offset) {
    // d is a dimension (0..255); the packed byte holding dims {d, d+1} is at d/2.
    return paged_kv_element_offset<kKVCacheI4CodeBytes, Geometry::KVHeads>(physical_page, kv_head,
                                                                           page_offset, d / 2);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_i4_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheI4Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                        page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_i4_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheI4HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheI4QuantParams {
    __half scale;
    float inverse_scale;
};

// Exact persistent group-scale boundary shared by standalone and fused append. The stored scale is
// FP16-RNE(absmax/7) (i4 codes top out at magnitude 7); codes always use the reciprocal of that
// represented FP16 value.
__device__ __forceinline__ KVCacheI4QuantParams kv_cache_i4_quant_params(float absmax) {
    const __half scale            = __float2half_rn(absmax > 0.0f ? absmax / 7.0f : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

// Plain i4 quantize (the V path and the non-E8 K path): round x*inv, clamp to [-7, 7] (15 of the
// 16 codes; -8 is reserved for the E8-lattice K path which can round to it).
__device__ __forceinline__ std::int8_t kv_cache_i4_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-7, min(7, q));
    return static_cast<std::int8_t>(q);
}

// E8-lattice K path: the caller has already scaled by the inverse group scale AND refined onto the
// E8 lattice; this just rounds and clamps to the full 4-bit range [-8, 7].
__device__ __forceinline__ std::int8_t kv_cache_i4_quant_code_e8(float x) {
    int q = __float2int_rn(x);
    q     = max(-8, min(7, q));
    return static_cast<std::int8_t>(q);
}

// Pack two i4 codes (even dim -> low nibble, odd dim -> high nibble) into one byte. Codes are
// two's-complement int8; the low 4 bits are stored, so -8..7 -> 8..15..7.
__device__ __forceinline__ std::uint8_t kv_cache_i4_pack(std::int8_t lo, std::int8_t hi) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 0x0fu) |
                                     ((static_cast<unsigned>(hi) & 0x0fu) << 4));
}

// Unpack one i4 code from a packed byte. `high` selects the high nibble (odd dim). Recovered code
// is in [-8, 7]: (nibble ^ 8) - 8 maps stored 0..15 -> -8..7.
__device__ __forceinline__ std::int8_t kv_cache_i4_unpack(std::uint8_t packed, int high) {
    const unsigned nibble = high ? (packed >> 4) : (packed & 0x0fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

// Vectorized decode: unpack 8 i4 codes (4 packed bytes) and dequantize to 8 BF16 (x8 * scale).
// `codes4` points at 4 consecutive packed bytes (dims d..d+7). Returns 8 bf16 as an int4.
__device__ __forceinline__ int4 kv_cache_i4_dequant_bf16x8_from(const std::uint8_t* codes4,
                                                                float s) {
    const int raw         = load_vec<int>(codes4);
    const std::uint8_t* b = reinterpret_cast<const std::uint8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(kv_cache_i4_unpack(b[i], 0)) * s;
        const float x1 = static_cast<float>(kv_cache_i4_unpack(b[i], 1)) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Unpack 16 i4 codes (8 packed bytes) to 16 int8 codes (no scale). `codes8` points at 8
// consecutive packed bytes (dims d..d+15). Out codes[16] = int8 codes for dims d..d+15.
__device__ __forceinline__ void kv_cache_i4_unpack_i8x16(const std::uint8_t* codes8,
                                                          std::int8_t* codes16) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::uint8_t* b = reinterpret_cast<const std::uint8_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        codes16[2 * i]     = kv_cache_i4_unpack(b[i], 0);
        codes16[2 * i + 1] = kv_cache_i4_unpack(b[i], 1);
    }
}

} // namespace ninfer::ops
