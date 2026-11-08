#pragma once

#include "ops/common/cuda/memory.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace ninfer::ops::common {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Block-scaled FP8 MMA. On RTX 5090 (sm_120a family), the block-scale form avoids the
// plain form's half-rate FP32 accumulation. UE8M0 0x7f is exactly 1: preserve the raw
// E4M3 dot product and keep the callers' existing row scales in FP32 after the MMA. No
// operand repacking is needed. On datacenter Blackwell (sm_100a) the block-scale PTX is
// not part of the target ISA; fall back to the plain FP8 MMA (functional, reduced
// accumulate rate) until a tcgen05 path lands for that target.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1200)
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_fp8_e4m3_scaled(float& c0, float& c1, float& c2, float& c3,
                                                    unsigned a0, unsigned a1, unsigned a2,
                                                    unsigned a3, unsigned b0, unsigned b1,
                                                    unsigned scale_a, unsigned scale_b) {
    // No hardware scale operands on sm_100a plain MMA: fold the UE8M0 scales into the
    // accumulators the way the block-scale form would have distributed them.
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    const float a_scale = exp2f(static_cast<float>(static_cast<std::int8_t>(scale_a)) - 127.0f);
    const float b_scale = exp2f(static_cast<float>(static_cast<std::int8_t>(scale_b)) - 127.0f);
    c0 *= a_scale * b_scale;
    c1 *= a_scale * b_scale;
    c2 *= a_scale * b_scale;
    c3 *= a_scale * b_scale;
}
#else
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

// scale_vec::1X uses the low byte. For B, lane%4==0 supplies column lane/4.
// A uses lanes 0 and 1 in each quad for rows lane/4 and lane/4+8.
__device__ __forceinline__ void mma_fp8_e4m3_scaled(float& c0, float& c1, float& c2, float& c3,
                                                    unsigned a0, unsigned a1, unsigned a2,
                                                    unsigned a3, unsigned b0, unsigned b1,
                                                    unsigned scale_a, unsigned scale_b) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(scale_b), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}
#endif

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

// FP4 dot-product MMA. The kind::mxf4nvf4 block-scale form is sm_120a-family PTX:
// datacenter Blackwell (sm_100a) has no warp-level FP4 MMA and no e2m1 hardware cvt
// (tcgen05-only). On sm_100a every consumer must route through
// nvfp4_mma_via_bf16() (dequant-to-bf16 + plain bf16 MMA) instead; the mma_nvfp4_e4m3
// wrapper below keeps that choice in one place.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1200)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
// sm_100a: no warp-level FP4 MMA
// fp4 type conversions decode e2m1 in software. Dequantize each 16-element fragment to
// FP16, convert to bf16, and accumulate with the plain bf16 MMA — the same numeric
// contract as the block-scale form (UE4M3 scales folded at the fragment boundary into
// FP32 accumulators). Replaced by a tcgen05 path for this target later.
__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    const unsigned a_words[4] = {a0, a1, a2, a3};
    const unsigned b_words[2] = {b0, b1};
    float a_f[8], b_f[8];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const auto h2 = *reinterpret_cast<const __half2*>(&a_words[w]);
        a_f[2 * w]     = __half2float(h2.x);
        a_f[2 * w + 1] = __half2float(h2.y);
    }
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const auto h2 = *reinterpret_cast<const __half2*>(&b_words[w]);
        b_f[2 * w]     = __half2float(h2.x);
        b_f[2 * w + 1] = __half2float(h2.y);
    }
    unsigned a_bf[4], b_bf[2];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const __nv_bfloat162 t = __floats2bfloat162_rn(a_f[2 * w], a_f[2 * w + 1]);
        a_bf[w] = *reinterpret_cast<const unsigned*>(&t);
    }
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const __nv_bfloat162 t = __floats2bfloat162_rn(b_f[2 * w], b_f[2 * w + 1]);
        b_bf[w] = *reinterpret_cast<const unsigned*>(&t);
    }
    mma_bf16(c0, c1, c2, c3, a_bf[0], a_bf[1], a_bf[2], a_bf[3], b_bf[0], b_bf[1]);
    const float fa = __half2float(__half{__nv_fp8_e4m3{static_cast<unsigned char>(sfa)}});
    const float fb = __half2float(__half{__nv_fp8_e4m3{static_cast<unsigned char>(sfb)}});
    const float s  = fa * fb;
    c0 *= s;
    c1 *= s;
    c2 *= s;
    c3 *= s;
}
#else
__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}
#endif

} // namespace ninfer::ops::common
