#pragma once

// The FP4 (NVFP4) activation of the routed experts' W4A4 matrix product (moe_fp4_product in
// ggml_bridge.h): the rounding of one activation column, shared by the device quantizer and the
// host reference of the tests, so both produce the same bytes.
//
// A column of k values x is stored as E2M1 codes with one UE4M3 scale per 16 values and one FP32
// factor for the whole column (the "global" scale of NVFP4):
//
//   amax = max |x|                         (0: every code and scale 0, factor 0)
//   g    = amax / 2688                     (2688 = 6 * 448, E2M1's and UE4M3's largest values)
//   s    = a16 / (amax / 448)              a16 = max |x| over the 16 values, so s <= 448
//   S    = UE4M3(s)                        round to nearest, ties to even, saturating at 448
//   q_i  = E2M1(x_i / (S * g))             round to nearest, ties to the even code
//
// and the product reads x_i ~ q_i * S * g: the tensor cores multiply E2M1 codes by both operands'
// UE4M3 scales, and the kernel's write-back multiplies each column's sums by g. The per-column
// factor is what keeps UE4M3's narrow range (2^-9 .. 448) usable: without it, values past
// 6 * 448 = 2688 or blocks below 2^-9 * 6 would saturate or flush. A column with a non-finite value
// gets a NaN factor, so its products are NaN as they would be in BF16.
//
// Every step is one IEEE operation rounded to nearest (divisions and products only, nothing an FMA
// could contract), so the host and the device agree bit for bit.

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define NINFER_NVFP4_HD __host__ __device__ __forceinline__
#else
#define NINFER_NVFP4_HD inline
#endif

namespace ninfer::ops::gguf::nvfp4 {

inline constexpr int kSubValues   = 16;   // values per UE4M3 scale
inline constexpr int kGroupValues = 64;   // values per scale word (four scales)
inline constexpr int kBlockValues = 256;  // values per block_fp4_mmq
inline constexpr float kScaleMax  = 448.0f;
inline constexpr float kRangeMax  = 2688.0f; // 6 * 448

NINFER_NVFP4_HD std::uint32_t float_bits(float v) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(v);
#else
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
#endif
}

NINFER_NVFP4_HD float bits_float(std::uint32_t b) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(b);
#else
    float v;
    std::memcpy(&v, &b, 4);
    return v;
#endif
}

NINFER_NVFP4_HD float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

NINFER_NVFP4_HD float mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

NINFER_NVFP4_HD float abs_value(float v) { return bits_float(float_bits(v) & 0x7FFFFFFFu); }

NINFER_NVFP4_HD bool finite(float v) { return (float_bits(v) & 0x7F800000u) != 0x7F800000u; }

// The UE4M3 code nearest to v >= 0 (ties to the even code), saturating at 448 (0x7E); 0 for v <= 0
// or NaN. Codes are e << 3 | m: e = 0 holds m * 2^-9, e > 0 holds (8 + m) * 2^(e - 10).
NINFER_NVFP4_HD std::uint8_t scale_code(float v) {
    if (!(v > 0.0f)) { return 0; }
    if (v >= kScaleMax) { return 0x7E; }
    const std::uint32_t b = float_bits(v);
    const int e           = int((b >> 23) & 0xFF) - 127;
    if (e < -6) {
        // Subnormal codes step by 2^-9: q = v * 512 (exact) rounded to an integer, half to even.
        const float q    = v * 512.0f;
        int qi           = int(q);
        const float frac = q - float(qi); // exact: q < 8
        if (frac > 0.5f || (frac == 0.5f && (qi & 1) != 0)) { ++qi; }
        return std::uint8_t(qi); // 8 is 2^-6, the first normal code
    }
    // Three mantissa bits of the 23, half to even; a carry moves into the exponent field.
    std::uint32_t keep       = (b >> 20) & 0x7;
    const std::uint32_t rest = b & 0xFFFFF;
    if (rest > 0x80000u || (rest == 0x80000u && (keep & 1u) != 0)) { ++keep; }
    const int code = ((e + 7) << 3) + int(keep);
    return std::uint8_t(code > 0x7E ? 0x7E : code);
}

// The value of a UE4M3 code (not ggml's halved ggml_ue4m3_to_fp32): exact in FP32.
NINFER_NVFP4_HD float scale_value(std::uint8_t code) {
    const int e = (code >> 3) & 0xF, m = code & 0x7;
    if (e == 0) { return float(m) * (1.0f / 512.0f); }
    return bits_float(std::uint32_t(e - 7 + 127) << 23 | std::uint32_t(m) << 20);
}

// The E2M1 code nearest to v (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives; ties to the even
// code, magnitudes past 6 to 6); sign in bit 3. NaN maps to 0 (its column's factor is NaN).
NINFER_NVFP4_HD std::uint8_t e2m1_code(float v) {
    const float a = abs_value(v);
    const int c   = int(a > 0.25f) + int(a >= 0.75f) + int(a > 1.25f) + int(a >= 1.75f) +
                  int(a > 2.5f) + int(a >= 3.5f) + int(a > 5.0f);
    return std::uint8_t(c | ((float_bits(v) >> 31) << 3));
}

NINFER_NVFP4_HD float e2m1_value(std::uint8_t code) {
    constexpr float kValues[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float a              = kValues[code & 7];
    return (code & 8) != 0 ? -a : a;
}

// The column's factor g from its largest magnitude (and whether every value is finite).
NINFER_NVFP4_HD float column_factor(float amax, bool all_finite) {
    if (!all_finite) { return bits_float(0x7FC00000u); }
    return amax > 0.0f ? div_rn(amax, kRangeMax) : 0.0f;
}

// One 16-value sub-block: its UE4M3 code from its largest magnitude a16, given the column's amax.
NINFER_NVFP4_HD std::uint8_t sub_scale(float a16, float amax) {
    if (!(amax > 0.0f) || !finite(amax)) { return 0; }
    return scale_code(div_rn(a16, div_rn(amax, kScaleMax)));
}

// Value v's code in a sub-block of scale code `s` in a column of factor g.
NINFER_NVFP4_HD std::uint8_t value_code(float v, std::uint8_t s, float g) {
    if (s == 0 || !(g > 0.0f) || !finite(g)) { return 0; }
    return e2m1_code(div_rn(v, mul_rn(scale_value(s), g)));
}

// The 8 bytes of a 16-value sub-block as the tensor cores read them, in block_nvfp4's order (the
// order ggml's NVFP4 weights already have, ggml_cuda_mmq_load_tiles_nvfp4): byte j holds value j in
// its low nibble and value j + 8 in its high nibble.
NINFER_NVFP4_HD void pack_sub(const std::uint8_t codes[16], std::uint32_t out[2]) {
    std::uint32_t w[2] = {0, 0};
    for (int j = 0; j < 8; ++j) {
        const std::uint32_t byte = std::uint32_t(codes[j]) | std::uint32_t(codes[j + 8]) << 4;
        w[j / 4] |= byte << (8 * (j % 4));
    }
    out[0] = w[0];
    out[1] = w[1];
}

} // namespace ninfer::ops::gguf::nvfp4
