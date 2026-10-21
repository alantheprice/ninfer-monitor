#pragma once

// RK4V4E8 (rk4v4-e8) append kernels: signed i4, per-token G64 KV-cache fill with E8-lattice
// refined keys. One warp owns one (token, kv_head) row of 256 dims, mirroring the INT8-G64
// append structure. Per 256-dim vector: 128 B of packed i4 codes (2 codes/byte, even dim in the
// low nibble) + 4 FP16 group-64 scales, for both K and V.
//
// Numerics match the 4090 fork's rk4v4-e8 encode (gqa_attention_prefill_i8.cuh, PackedK/E8
// branch), with two deliberate substitutions so the mainline i8 structure is reused unchanged:
// (1) the mainline uses the fixed normalized D256 rotation (normalized_hadamard_d256_inplace) for
// K and Q instead of the fork's per-G64 hadamard64; (2) V is NOT rotated (matching the mainline
// i8 kernel, which stores V unrotated), so the attention PV output = P*V_orig is correct with no
// output de-rotation. K and Q share the D256 rotation (QK dot product preserved by orthogonality);
// V is i4-quantized in the original coordinates. The E8 projection is exact in the D256-rotated
// K space. K codes are clamped to [-8, 7] (E8 can round to -8); V codes to [-7, 7]. The stored
// scale is FP16_RNE(absmax/7) and codes use the reciprocal of that represented FP16 value.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/e8_lattice.cuh"
#include "ops/kv_cache/i4_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// One 256-dim row of K and V, D256-rotated and i4-quantized. `lane` holds dims lane+32*r; within
// each G64 group the 32 lanes hold dims group*64+lane and +32. All shuffles are warp-local.
template <typename Geometry>
__device__ __forceinline__ void kv_cache_append_full_i4_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, int token, int kv_head,
    int physical_page, int page_off, int lane) {
    constexpr unsigned FullMask = 0xffffffffU;

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] =
            __bfloat162float(k[kv_cache_i4_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

    float v_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        v_values[r] =
            __bfloat162float(v[kv_cache_i4_src_index<Geometry>(kv_head, d, token)]);
    }
    // V is NOT rotated (mainline i8 convention): it is i4-quantized in the original coordinates,
    // so the attention PV output = P*V_orig is correct with no output de-rotation. K and Q share
    // the D256 rotation instead.

#pragma unroll
    for (int group = 0; group < kKVCacheI4Groups; ++group) {
        const float k0 = k_values[2 * group];
        const float k1 = k_values[2 * group + 1];
        const float v0 = v_values[2 * group];
        const float v1 = v_values[2 * group + 1];
        const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant = kv_cache_i4_quant_params(k_abs);
        const auto v_quant = kv_cache_i4_quant_params(v_abs);

        // E8-lattice K: scale, refine onto E8 (one 8-dim lattice vector per (lane, r) pair), then
        // round/clamp the projected value. V is plain i4 on the D256-rotated value.
        float k0_scaled = k0 * k_quant.inverse_scale;
        float k1_scaled = k1 * k_quant.inverse_scale;
        e8_project_8d_warp(k0_scaled, k1_scaled, lane);
        const std::int8_t kc0 = kv_cache_i4_quant_code_e8(k0_scaled);
        const std::int8_t kc1 = kv_cache_i4_quant_code_e8(k1_scaled);
        const std::int8_t vc0 = kv_cache_i4_quant_code(v0, v_quant.inverse_scale);
        const std::int8_t vc1 = kv_cache_i4_quant_code(v1, v_quant.inverse_scale);

        // Pack (even dim -> lo, odd dim -> hi); adjacent lanes pair to one byte, stored by the
        // even lane of the pair.
        const std::int8_t kc0_hi = static_cast<std::int8_t>(__shfl_down_sync(FullMask, kc0, 1));
        const std::int8_t kc1_hi = static_cast<std::int8_t>(__shfl_down_sync(FullMask, kc1, 1));
        const std::int8_t vc0_hi = static_cast<std::int8_t>(__shfl_down_sync(FullMask, vc0, 1));
        const std::int8_t vc1_hi = static_cast<std::int8_t>(__shfl_down_sync(FullMask, vc1, 1));
        if ((lane & 1) == 0) {
            const int d0 = group * kKVCacheI4Group + lane;
            const int d1 = d0 + 32;
            cache_k[kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d0, page_off)] =
                kv_cache_i4_pack(kc0, kc0_hi);
            cache_k[kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d1, page_off)] =
                kv_cache_i4_pack(kc1, kc1_hi);
            cache_v[kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d0, page_off)] =
                kv_cache_i4_pack(vc0, vc0_hi);
            cache_v[kv_cache_i4_code_index<Geometry>(physical_page, kv_head, d1, page_off)] =
                kv_cache_i4_pack(vc1, vc1_hi);
        }
        if (lane == 0) {
            const std::int64_t scale_off =
                kv_cache_i4_scale_index<Geometry>(physical_page, kv_head, group, page_off);
            scale_k[scale_off] = k_quant.scale;
            scale_v[scale_off] = v_quant.scale;
        }
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_i4_kernel(const __nv_bfloat16* __restrict__ k,
                                        const __nv_bfloat16* __restrict__ v,
                                        const std::int32_t* __restrict__ positions,
                                        Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                        std::uint8_t* __restrict__ cache_v,
                                        __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                        std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    kv_cache_append_full_i4_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                          physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_i4_page_kernel(const __nv_bfloat16* __restrict__ k,
                                             const __nv_bfloat16* __restrict__ v,
                                             const std::int32_t* __restrict__ positions,
                                             Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                             std::uint8_t* __restrict__ cache_v,
                                             __half* __restrict__ scale_k,
                                             __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    const int position              = base_position + token;
    kv_cache_append_full_i4_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                          physical_page, position & kPagedKVPageMask, lane);
}

} // namespace ninfer::ops
