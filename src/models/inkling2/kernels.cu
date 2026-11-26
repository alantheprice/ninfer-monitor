#include "models/inkling2/kernels.h"

#include "core/device_launch.h"
#include "models/inkling2/config.h"
#include "models/inkling2/kv_codec.cuh"
#include "models/inkling2/situ.h"
#include "ops/common/mma.cuh"
#include "ops/kernel/sampling_device.cuh"

#include "core/pdl.cuh"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::models::inkling2::kernels {
namespace {

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling2 ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("inkling2 cuBLAS ") + what + " failed (" +
                                 std::to_string(int(status)) + ")");
    }
}

cublasHandle_t blas_handle() {
    static constexpr int kMaxDevices = 16;
    static cublasHandle_t handles[kMaxDevices]{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::runtime_error("inkling2: device index out of range");
    }
    if (handles[device] == nullptr) {
        check_blas(cublasCreate(&handles[device]), "create");
        // FP32 products: no TF32.
        check_blas(cublasSetMathMode(handles[device], CUBLAS_PEDANTIC_MATH), "math mode");
    }
    return handles[device];
}

unsigned blocks_for(std::int64_t n, int threads) {
    return static_cast<unsigned>((n + threads - 1) / threads);
}

constexpr int kD   = 128; // head width
constexpr int kRel = 16;  // relative feature width
constexpr int kMaxGroup = 4;
constexpr int kTaps     = 4;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFu, v, o); }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o)); }
    return v;
}

// Sums of N values over the block (blockDim.x a multiple of 32, at most 1024), in place: every
// thread gets the totals.
template <int N>
__device__ void block_sums(float (&v)[N]) {
    __shared__ float partial[N][32];
    __shared__ float total[N];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warps = int(blockDim.x >> 5);
#pragma unroll
    for (int i = 0; i < N; ++i) { v[i] = warp_sum(v[i]); }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) { partial[i][warp] = v[i]; }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float s = 0.0f;
        for (int w = 0; w < warps; ++w) { s += partial[i][w]; }
        total[i] = s;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) { v[i] = total[i]; }
    __syncthreads();
}

__device__ __forceinline__ float block_sum(float v) {
    float a[1] = {v};
    block_sums(a);
    return a[0];
}

__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 w;
    w.x = *reinterpret_cast<const unsigned*>(&lo);
    w.y = *reinterpret_cast<const unsigned*>(&hi);
    *reinterpret_cast<uint2*>(p) = w;
}
__device__ __forceinline__ float dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__device__ __forceinline__ float4 mul4(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__device__ __forceinline__ float4 scale4(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 fma4(float4 a, float s, float4 c) {
    return make_float4(fmaf(a.x, s, c.x), fmaf(a.y, s, c.y), fmaf(a.z, s, c.z), fmaf(a.w, s, c.w));
}
__device__ __forceinline__ float4 ld4(const float* p, int i) {
    return reinterpret_cast<const float4*>(p)[i];
}
__device__ __forceinline__ float get(float4 v, int j) {
    return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}
__device__ __forceinline__ float4 fixed4(const unsigned long long* p, int i) {
    const auto* v  = p + 4 * std::int64_t(i);
    const double s = 1.0 / 4294967296.0;
    return make_float4(float(double(static_cast<long long>(v[0])) * s),
                       float(double(static_cast<long long>(v[1])) * s),
                       float(double(static_cast<long long>(v[2])) * s),
                       float(double(static_cast<long long>(v[3])) * s));
}
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// The q8_1 form of this thread's four values, elements 4i..4i+3 of row `row` of width k: eight
// consecutive threads hold one 32-value block (k % 32 == 0 keeps every group of eight either in or
// out of the row). Every lane of the warp calls it; `valid` lanes store.
__device__ __forceinline__ void store_q8x4(const Q8Rows& q, int k, std::int64_t row, int i,
                                           float4 v, bool valid) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    float sum  = v.x + v.y + v.z + v.w;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
    }
    if (!valid) { return; }
    const float d = amax / 127.0f;
    char4 c       = make_char4(0, 0, 0, 0);
    if (amax != 0.0f) {
        c = make_char4(static_cast<signed char>(roundf(v.x / d)),
                       static_cast<signed char>(roundf(v.y / d)),
                       static_cast<signed char>(roundf(v.z / d)),
                       static_cast<signed char>(roundf(v.w / d)));
    }
    *reinterpret_cast<char4*>(q.qs + row * k + 4 * i) = c;
    if ((i & 7) == 0) { q.ds[row * (k / 32) + i / 8] = make_half2(__float2half(d), __float2half(sum)); }
}

// A row kernel's block: one thread per 4 values of a row of Inkling-2-Mini's width (2304). The bound
// lets the depth mix (eight states, four queries in registers) launch at that width.
constexpr int kRowBlock = 576;

int row_threads(int hidden) {
    if (hidden % 32 != 0 || hidden / 4 > kRowBlock) {
        throw std::invalid_argument("inkling2 row kernels: rows of whole q8_1 blocks, at most 2304");
    }
    return (hidden / 4 + 31) / 32 * 32;
}

// ---- Rows and their history ---------------------------------------------------------------------

// Where the input `lag` positions before row t lies: a row of the call (>= 0), the history ring
// (-1, `position` its position) or before position 0 (-2).
__device__ __forceinline__ int history_source(const Rows& r, int t, int lag, int& position) {
    const int p = r.positions[t] - lag;
    position    = p;
    if (lag == 0) { return t; }
    if (p < 0) { return -2; }
    if (!r.chunk) {
        const int s = t - lag;
        if (s >= 0 && r.positions[s] == p && row_sequence(r, s) == row_sequence(r, t)) { return s; }
        return -1;
    }
    if (r.starts != nullptr && t >= r.shared) {
        const int b0 = r.starts[t];
        const int p0 = r.positions[b0];
        if (p >= p0) { return b0 + (p - p0); }
    }
    if (p >= r.first) { return p - r.first; }
    return -1;
}

__device__ __forceinline__ float* history_slot(const History& h, int sequence, int position) {
    return h.ring + std::int64_t(sequence) * h.seq_stride +
           std::int64_t(position % h.slots) * h.slot_stride;
}

// ---- Norms and depth mixes ----------------------------------------------------------------------

__global__ void __launch_bounds__(kRowBlock)
    rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, int hidden, float eps,
                    __nv_bfloat16* __restrict__ out, Q8Rows q8, float* f32) {
    ninfer::pdl::enter();
    const int q = hidden >> 2, i = threadIdx.x;
    const std::int64_t row = std::int64_t(blockIdx.x) * hidden;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 v    = i < q ? ld4(x + row, i) : zero;
    const float4 wv   = i < q && w != nullptr ? ld4(w, i) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    const float inv   = rsqrtf(block_sum(dot4(v, v)) / float(hidden) + eps);
    float4 y          = zero;
    if (i < q) {
        y = mul4(scale4(v, inv), wv);
        if (out != nullptr) { store_bf16x4(out + row + 4 * i, y); }
        if (f32 != nullptr) { reinterpret_cast<float4*>(f32 + row)[i] = y; }
    }
    if (q8.qs != nullptr) { store_q8x4(q8, hidden, blockIdx.x, i, y, i < q); }
}

// The outputs of one query's normalized row.
__device__ __forceinline__ void store_outputs(__nv_bfloat16* out, Q8Rows out_q, float* f32,
                                              std::int64_t offset, std::int64_t q8_offset,
                                              int hidden, int t, int i, float4 y, bool valid) {
    const std::int64_t row = std::int64_t(t) * hidden;
    if (valid && out != nullptr) { store_bf16x4(out + offset + row + 4 * i, y); }
    if (valid && f32 != nullptr) { reinterpret_cast<float4*>(f32 + offset + row)[i] = y; }
    if (out_q.qs != nullptr) {
        const Q8Rows q{out_q.qs + q8_offset,
                       reinterpret_cast<half2*>(reinterpret_cast<char*>(out_q.ds) + q8_offset)};
        store_q8x4(q, hidden, t, i, y, valid);
    }
}

// The mix's sums per row: |s_j|^2 for each state j, then query q . s_j at kMaxStates +
// q * kMaxStates + j.
constexpr int kMixValues = kMaxStates * (1 + kMaxQueries);

// The states of row t's mix at quad i, the current state (the last) from registers when
// `has_current`.
__device__ __forceinline__ void mix_states(const MixArgs& a, int t, int i, bool valid,
                                           bool has_current, float4 current,
                                           float4 (&s)[kMaxStates]) {
    const std::int64_t row = std::int64_t(t) * a.hidden;
    const float4 zero      = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int j = 0; j < kMaxStates; ++j) {
        const bool last = j == a.count - 1;
        s[j] = j < a.count && valid ? (last && has_current ? current : ld4(a.states[j] + row, i))
                                    : zero;
    }
}

// The queries at quad i (zeros past query_count and outside the row).
__device__ __forceinline__ void mix_queries(const MixArgs& a, int i, bool valid,
                                            float4 (&qv)[kMaxQueries]) {
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
        qv[q] = q < a.query_count && valid ? ld4(a.queries + std::int64_t(q) * a.hidden, i)
                                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

// Quad i's terms of the mix's sums.
__device__ __forceinline__ void mix_terms(const float4 (&s)[kMaxStates],
                                          const float4 (&qv)[kMaxQueries], float (&v)[kMixValues]) {
#pragma unroll
    for (int j = 0; j < kMaxStates; ++j) { v[j] = dot4(s[j], s[j]); }
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
#pragma unroll
        for (int j = 0; j < kMaxStates; ++j) { v[kMaxStates + q * kMaxStates + j] = dot4(qv[q], s[j]); }
    }
}

// Each query's mixed quad from the row's sums v, and its square's term.
__device__ __forceinline__ void mix_combine(const MixArgs& a, const float (&v)[kMixValues],
                                            const float4 (&s)[kMaxStates],
                                            float4 (&mixed)[kMaxQueries],
                                            float (&squares)[kMaxQueries]) {
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float inv[kMaxStates];
#pragma unroll
    for (int j = 0; j < kMaxStates; ++j) { inv[j] = rsqrtf(v[j] / float(a.hidden) + kMixEps); }
    const float root = rsqrtf(float(a.hidden));
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
        mixed[q]   = zero;
        squares[q] = 0.0f;
        if (q >= a.query_count) { continue; }
        float score[kMaxStates], top = -INFINITY;
#pragma unroll
        for (int j = 0; j < kMaxStates; ++j) {
            score[j] = j < a.count ? v[kMaxStates + q * kMaxStates + j] * inv[j] * root : -INFINITY;
            top      = fmaxf(top, score[j]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < kMaxStates; ++j) {
            score[j] = j < a.count ? expf(score[j] - top) : 0.0f;
            sum += score[j];
        }
        float4 y = zero;
#pragma unroll
        for (int j = 0; j < kMaxStates; ++j) {
            if (j < a.count) { y = fma4(s[j], score[j] / sum * inv[j], y); }
        }
        mixed[q]   = y;
        squares[q] = dot4(y, y);
    }
}

__device__ __forceinline__ void mix_raw(const MixArgs& a, int t, int i, bool valid,
                                        const float4 (&mixed)[kMaxQueries]) {
    if (a.raw == nullptr || !valid) { return; }
    const std::int64_t row = std::int64_t(t) * a.hidden;
    for (int q = 0; q < a.query_count; ++q) {
        reinterpret_cast<float4*>(a.raw + q * a.query_stride + row)[i] = mixed[q];
    }
}

__device__ __forceinline__ bool mix_normed(const MixArgs& a) {
    return a.out != nullptr || a.out_q.qs != nullptr || a.f32 != nullptr;
}

// The normed outputs of each query's mixed quad, `squares` the row sums. Every lane of the warp
// calls it.
__device__ __forceinline__ void mix_outputs(const MixArgs& a, int t, int i, bool valid,
                                            const float4 (&mixed)[kMaxQueries],
                                            const float (&squares)[kMaxQueries]) {
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
        if (q >= a.query_count) { continue; }
        const float n   = rsqrtf(squares[q] / float(a.hidden) + a.eps);
        const float4 wv = valid && a.norm[q] != nullptr ? ld4(a.norm[q], i)
                                                        : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        store_outputs(a.out, a.out_q, a.f32, q * a.query_stride, q * a.q8_stride, a.hidden, t, i,
                      mul4(scale4(mixed[q], n), wv), valid);
    }
}

// The depth mix of row t, its current state `current` (the last state) given in registers when
// `has_current`. Every thread of the block calls it.
__device__ void mix_row(const MixArgs& a, int t, bool has_current, float4 current) {
    const int quads = a.hidden >> 2, i = threadIdx.x;
    const bool valid = i < quads;
    float4 s[kMaxStates], qv[kMaxQueries];
    float v[kMixValues];
    mix_states(a, t, i, valid, has_current, current, s);
    mix_queries(a, i, valid, qv);
    mix_terms(s, qv, v);
    block_sums(v);
    float4 mixed[kMaxQueries];
    float squares[kMaxQueries];
    mix_combine(a, v, s, mixed, squares);
    mix_raw(a, t, i, valid, mixed);
    if (!mix_normed(a)) { return; }
    block_sums(squares);
    mix_outputs(a, t, i, valid, mixed, squares);
}

__global__ void __launch_bounds__(kRowBlock) mix_kernel(MixArgs a) {
    ninfer::pdl::enter();
    mix_row(a, blockIdx.x, false, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
}

// ---- Split rows ---------------------------------------------------------------------------------
// A call of at most kSplitRows rows spreads each row over one-warp blocks: block (w, t) holds the
// quads [32w, 32w + 32) of row t, those the row kernel's warp w holds. Its warp sums go to the
// scratch (MixArgs::split) and every reader adds them in warp order as block_sums does, so the
// values are the row kernel's.

__host__ __device__ __forceinline__ int row_warps(int hidden) { return (hidden / 4 + 31) / 32; }
constexpr int kMaxRowWarps = kRowBlock / 32;

// The warp's sums of v, kept at `out` by lane 0.
template <int N>
__device__ __forceinline__ void keep_warp_sums(float* out, float (&v)[N]) {
#pragma unroll
    for (int j = 0; j < N; ++j) { v[j] = warp_sum(v[j]); }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < N; ++j) { out[j] = v[j]; }
    }
}

// The row sums of the `warps` warps' sums at `in` (N per warp), to every thread of the block: each
// total's terms loaded at once, then added in warp order.
template <int N>
__device__ __forceinline__ void add_warp_sums(const float* in, int warps, float (&v)[N]) {
    __shared__ float total[N];
    for (int j = threadIdx.x; j < N; j += 32) {
        float part[kMaxRowWarps];
#pragma unroll
        for (int w = 0; w < kMaxRowWarps; ++w) { part[w] = w < warps ? in[w * N + j] : 0.0f; }
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < kMaxRowWarps; ++w) {
            if (w < warps) { s += part[w]; }
        }
        total[j] = s;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < N; ++j) { v[j] = total[j]; }
}

// The scratch: the warps' sums of the mix's terms [T][warps][kMixValues], of the mixed rows'
// squares [T][warps][kMaxQueries], then the mixed rows [kMaxQueries][T][hidden].
__device__ __forceinline__ float* split_terms(const MixArgs& a, int t, int w) {
    return a.split + (std::int64_t(t) * row_warps(a.hidden) + w) * kMixValues;
}
__device__ __forceinline__ float* split_squares(const MixArgs& a, int t, int w) {
    return a.split + std::int64_t(a.tokens) * row_warps(a.hidden) * kMixValues +
           (std::int64_t(t) * row_warps(a.hidden) + w) * kMaxQueries;
}
__device__ __forceinline__ float* split_mixed(const MixArgs& a, int q, int t) {
    return a.split + std::int64_t(a.tokens) * row_warps(a.hidden) * (kMixValues + kMaxQueries) +
           (std::int64_t(q) * a.tokens + t) * a.hidden;
}

__global__ void __launch_bounds__(32) mix_split_terms_kernel(MixArgs a) {
    ninfer::pdl::enter();
    const int w = int(blockIdx.x), t = int(blockIdx.y), i = 32 * w + int(threadIdx.x);
    const bool valid = i < (a.hidden >> 2);
    float4 s[kMaxStates], qv[kMaxQueries];
    float v[kMixValues];
    mix_states(a, t, i, valid, false, make_float4(0.0f, 0.0f, 0.0f, 0.0f), s);
    mix_queries(a, i, valid, qv);
    mix_terms(s, qv, v);
    keep_warp_sums(split_terms(a, t, w), v);
}

// Block (w, t): the mixed quads of row t from its sums. The first pass (`normed` false) writes the
// raw mixes and, for normed outputs, the mixed rows and the warps' sums of their squares; the
// second the normed outputs.
__global__ void __launch_bounds__(32) mix_split_rows_kernel(MixArgs a, bool normed) {
    ninfer::pdl::enter();
    const int w = int(blockIdx.x), t = int(blockIdx.y), i = 32 * w + int(threadIdx.x);
    const bool valid = i < (a.hidden >> 2);
    const int warps  = row_warps(a.hidden);
    float4 mixed[kMaxQueries];
    float squares[kMaxQueries];
    if (normed) {
#pragma unroll
        for (int q = 0; q < kMaxQueries; ++q) {
            mixed[q] = q < a.query_count && valid ? ld4(split_mixed(a, q, t), i)
                                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        add_warp_sums(split_squares(a, t, 0), warps, squares);
        mix_outputs(a, t, i, valid, mixed, squares);
        return;
    }
    float4 s[kMaxStates];
    mix_states(a, t, i, valid, false, make_float4(0.0f, 0.0f, 0.0f, 0.0f), s);
    float v[kMixValues];
    add_warp_sums(split_terms(a, t, 0), warps, v);
    mix_combine(a, v, s, mixed, squares);
    mix_raw(a, t, i, valid, mixed);
    if (!mix_normed(a)) { return; }
    if (valid) {
        for (int q = 0; q < a.query_count; ++q) {
            reinterpret_cast<float4*>(split_mixed(a, q, t))[i] = mixed[q];
        }
    }
    keep_warp_sums(split_squares(a, t, w), squares);
}

// ---- FP32 products ------------------------------------------------------------------------------

constexpr int kGemvThreads = 256;
constexpr int kGemvRows    = 8;
constexpr int kGemvAhead   = 4; // float4 steps whose loads a thread issues before their products

// Output row `row` of N tokens (at most kGemvRows), its k values split over the block.
template <int N>
__device__ __forceinline__ void gemv_row(const float* __restrict__ w, const float* __restrict__ x,
                                         int row, int k, std::int64_t x_stride,
                                         float* __restrict__ out, std::int64_t out_stride,
                                         int cols = N) {
    __shared__ float part[kGemvThreads / 32][N];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* wr = w + std::int64_t(row) * k;
    float dot[N] = {};
    if (k % 4 == 0 && x_stride % 4 == 0) {
        const int quads = k / 4;
        const auto* w4  = reinterpret_cast<const float4*>(wr);
        // The thread's quads i, i + 256, ... in order, the loads of up to kGemvAhead of them
        // issued before their products.
        // Fewer loads in flight past 8 columns keep the registers; the products stay in order
        // (quads i, i + 256, ... whatever the depth), so the sums do not change.
        constexpr int kAhead = N > 8 ? 2 : kGemvAhead;
        for (int i = threadIdx.x; i < quads; i += kAhead * kGemvThreads) {
            float4 wv[kAhead], xv[kAhead][N];
#pragma unroll
            for (int a = 0; a < kAhead; ++a) {
                const int at = i + a * kGemvThreads;
                if (at >= quads) { continue; }
                wv[a] = w4[at];
#pragma unroll
                for (int t = 0; t < N; ++t) {
                    xv[a][t] = t < cols ? reinterpret_cast<const float4*>(x + std::int64_t(t) * x_stride)[at]
                                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
#pragma unroll
            for (int a = 0; a < kAhead; ++a) {
                if (i + a * kGemvThreads >= quads) { continue; }
#pragma unroll
                for (int t = 0; t < N; ++t) { dot[t] += dot4(wv[a], xv[a][t]); }
            }
        }
    } else {
        for (int i = threadIdx.x; i < k; i += kGemvThreads) {
#pragma unroll
            for (int t = 0; t < N; ++t) {
                if (t < cols) { dot[t] += wr[i] * x[std::int64_t(t) * x_stride + i]; }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < N; ++t) { dot[t] = warp_sum(dot[t]); }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < N; ++t) { part[warp][t] = dot[t]; }
    }
    __syncthreads();
    if (int(threadIdx.x) < cols) {
        float s = 0.0f;
        for (int w2 = 0; w2 < kGemvThreads / 32; ++w2) { s += part[w2][threadIdx.x]; }
        out[std::int64_t(threadIdx.x) * out_stride + row] = s;
    }
}

// Columns c0 .. c0 + cols of part `part` when n columns split into `parts` (the host's split).
__device__ __forceinline__ void column_part(int n, int parts, int part, int& c0, int& cols) {
    c0 = 0;
    for (int i = 0;; ++i) {
        cols = (n - c0 + (parts - i) - 1) / (parts - i);
        if (i == part) { return; }
        c0 += cols;
    }
}

// Block (m, part[, b]): output row m of the part's columns (of product b). One launch for every
// part of a wide vector call; a column's sums are those of the part's own narrower call.
template <int N>
__global__ void gemv_parts_kernel(const float* __restrict__ w, const float* __restrict__ x, int k,
                                  std::int64_t x_stride, float* __restrict__ out,
                                  std::int64_t out_stride, int n, int parts) {
    ninfer::pdl::enter();
    int c0, cols;
    column_part(n, parts, int(blockIdx.y), c0, cols);
    gemv_row<N>(w, x + std::int64_t(c0) * x_stride, int(blockIdx.x), k, x_stride,
                out + std::int64_t(c0) * out_stride, out_stride, cols);
}

template <int N>
__global__ void gemv_batch_parts_kernel(GemvBatch b, int k, std::int64_t x_stride,
                                        std::int64_t out_stride, int n, int parts) {
    ninfer::pdl::enter();
    const int e = int(blockIdx.y);
    int c0, cols;
    column_part(n, parts, int(blockIdx.z), c0, cols);
    gemv_row<N>(b.w[e], b.x[e] + std::int64_t(c0) * x_stride, int(blockIdx.x), k, x_stride,
                b.out[e] + std::int64_t(c0) * out_stride, out_stride, cols);
}

// Block m: output row m.
template <int N>
__global__ void gemv_kernel(const float* __restrict__ w, const float* __restrict__ x, int m, int k,
                            std::int64_t x_stride, float* __restrict__ out,
                            std::int64_t out_stride) {
    ninfer::pdl::enter();
    gemv_row<N>(w, x, int(blockIdx.x), k, x_stride, out, out_stride);
}

// Block (m, b): output row m of product b.
template <int N>
__global__ void gemv_batch_kernel(GemvBatch b, int k, std::int64_t x_stride,
                                  std::int64_t out_stride) {
    ninfer::pdl::enter();
    const int e = int(blockIdx.y);
    gemv_row<N>(b.w[e], b.x[e], int(blockIdx.x), k, x_stride, b.out[e], out_stride);
}

constexpr int kShortRows = kGemvThreads / 32; // output rows per block of the short product

// A short product (k <= 128, k % 4 == 0, x_stride % 4 == 0), block (rows, b): warp r output row
// 8 blockIdx.x + r of product b, lane l its values 4l..4l+3. The sums are gemv_row's: its threads
// past k / 4 and its other warps add zeros.
// Wide vector calls: blockIdx.z the part of `parts` (n columns), as gemv_parts_kernel.
template <int N>
__global__ void __launch_bounds__(kGemvThreads)
    gemv_short_kernel(GemvBatch b, int m, int k, std::int64_t x_stride, std::int64_t out_stride,
                      int n = N, int parts = 1) {
    ninfer::pdl::enter();
    const int e = int(blockIdx.y), lane = threadIdx.x & 31;
    const int row = int(blockIdx.x) * kShortRows + int(threadIdx.x >> 5);
    if (row >= m) { return; }
    int c0 = 0, cols = N;
    if (parts > 1) { column_part(n, parts, int(blockIdx.z), c0, cols); }
    const float* x = b.x[e] + std::int64_t(c0) * x_stride;
    float* out     = b.out[e] + std::int64_t(c0) * out_stride;
    float dot[N]   = {};
    if (lane < k / 4) {
        const float4 wv = reinterpret_cast<const float4*>(b.w[e] + std::int64_t(row) * k)[lane];
#pragma unroll
        for (int t = 0; t < N; ++t) {
            if (t < cols) {
                dot[t] += dot4(wv, reinterpret_cast<const float4*>(x + std::int64_t(t) * x_stride)[lane]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < N; ++t) { dot[t] = warp_sum(dot[t]); }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < N; ++t) {
            if (t < cols) { out[std::int64_t(t) * out_stride + row] = 0.0f + dot[t]; }
        }
    }
}

// ---- Loop controllers ---------------------------------------------------------------------------

// Block (t, input): [rms(x); rms(message)] into concat.
__global__ void __launch_bounds__(kRowBlock) control_inputs_kernel(ControlInputs a) {
    ninfer::pdl::enter();
    const int t = blockIdx.x, c = blockIdx.y, i = threadIdx.x, quads = a.hidden >> 2;
    const std::int64_t row = std::int64_t(t) * a.hidden;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 x    = i < quads ? ld4(a.x[c] + row, i) : zero;
    const float4 m    = i < quads ? ld4(a.message[c] + row, i) : zero;
    float v[2]        = {dot4(x, x), dot4(m, m)};
    block_sums(v);
    if (i >= quads) { return; }
    float* out = a.concat + (std::int64_t(c) * a.tokens + t) * 2 * a.hidden;
    reinterpret_cast<float4*>(out)[i] = scale4(x, rsqrtf(v[0] / float(a.hidden) + kMixEps));
    reinterpret_cast<float4*>(out + a.hidden)[i] = scale4(m, rsqrtf(v[1] / float(a.hidden) + kMixEps));
}

// Block (t, input).
__global__ void __launch_bounds__(kRowBlock) control_apply_kernel(ControlApply a) {
    ninfer::pdl::enter();
    const int t = blockIdx.x, c = blockIdx.y, i = threadIdx.x, quads = a.hidden >> 2;
    const bool valid = i < quads;
    const std::int64_t row = std::int64_t(t) * a.hidden;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 y          = zero;
    if (valid) {
        const float4 x = ld4(a.x[c] + row, i), m = ld4(a.message[c] + row, i);
        const float4 f = ld4(a.forget[c] + row, i), g = ld4(a.gain[c] + row, i);
        y = make_float4(x.x - tanhf(f.x) * x.x + tanhf(g.x) * m.x,
                        x.y - tanhf(f.y) * x.y + tanhf(g.y) * m.y,
                        x.z - tanhf(f.z) * x.z + tanhf(g.z) * m.z,
                        x.w - tanhf(f.w) * x.w + tanhf(g.w) * m.w);
    }
    const float n   = rsqrtf(block_sum(dot4(y, y)) / float(a.hidden) + a.eps);
    const float4 wv = valid && a.norm[c] != nullptr ? ld4(a.norm[c], i)
                                                    : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    store_outputs(a.out, a.out_q, a.f32, c * a.query_stride, c * a.q8_stride, a.hidden, t, i,
                  mul4(scale4(y, n), wv), valid);
}

// ---- Attention inputs ---------------------------------------------------------------------------

// Block (t, head) of 128 threads: Q's value d, and the relative feature's in the first 16.
__device__ __forceinline__ void query_post_body(const QueryArgs& a, int t, int h, int d) {
    const float* row = a.qgr + std::int64_t(t) * a.stride;
    float tau = 1.0f;
    if (a.log_scale) {
        const float p = float(a.positions[t]);
        tau           = 1.0f + a.alpha * logf(fmaxf(1.0f, (p + 1.0f) / a.floor));
    }
    const float x    = row[h * kD + d] * a.query_clip[h * kD + d];
    const float norm = sqrtf(block_sum(x * x)) + 1e-6f;
    const float f    = fminf(1.0f, fmaxf(0.0f, a.qk_clip / norm));
    a.q_out[(std::int64_t(t) * a.heads + h) * kD + d] = x * f * tau;
    if (d >= 32) { return; }
    const float r  = d < kRel ? row[a.relative_column + h * kRel + d] * a.relative_clip[h * kRel + d]
                              : 0.0f;
    const float rn = sqrtf(warp_sum(r * r)) + 1e-6f;
    const float rf = fminf(1.0f, fmaxf(0.0f, a.rel_clip / rn));
    if (d < kRel) { a.r_out[(std::int64_t(t) * a.heads + h) * kRel + d] = r * rf * tau; }
}

__global__ void query_post_kernel(QueryArgs a) {
    ninfer::pdl::enter();
    query_post_body(a, int(blockIdx.x), int(blockIdx.y), int(threadIdx.x));
}

// The short convolution of channel c of row t over its site's raw inputs x (rows `stride` apart,
// the channel at `column`; in a history slot at c): x itself plus the static taps (tap 3 the row's
// own input) and, with `taps` ([kTaps] of the row's head), the dynamic ones (tap `lag` the input lag
// positions back).
__device__ __forceinline__ float convolve(const Rows& rows, int t, const float* x,
                                          std::int64_t stride, int column, int c, const float* w,
                                          const float* taps, const History& history) {
    const int sequence = row_sequence(rows, t);
    float in[kTaps];
#pragma unroll
    for (int lag = 0; lag < kTaps; ++lag) {
        int position     = 0;
        const int source = history_source(rows, t, lag, position);
        in[lag] = source >= 0   ? x[std::int64_t(source) * stride + column]
                  : source == -1 ? history_slot(history, sequence, position)[c]
                                 : 0.0f;
    }
    float out = 0.0f;
#pragma unroll
    for (int i = 0; i < kTaps; ++i) { out += w[i] * in[kTaps - 1 - i]; }
    if (taps != nullptr) {
#pragma unroll
        for (int lag = 0; lag < kTaps; ++lag) { out += in[lag] * taps[lag]; }
    }
    return in[0] + out;
}

// Block (t, kv head) of 128 threads, channel d of the head.
__device__ __forceinline__ void kv_post_body(const KvArgs& a, int t, int kh, int d) {
    const int c = kh * kD + d, v_column = a.kv_heads * kD + c;
    const bool dynamic = a.k_taps != nullptr;
    const float* k_taps = dynamic ? a.k_taps + std::int64_t(t) * a.taps_stride + kh * kTaps : nullptr;
    const float* v_taps = dynamic ? a.v_taps + std::int64_t(t) * a.taps_stride + kh * kTaps : nullptr;
    const float divisor = rsqrtf(float((dynamic ? 2 : 1) * kTaps + 1));
    const float k = convolve(a.rows, t, a.kv, a.kv_stride, c, c, a.k_conv + c * kTaps, k_taps,
                             a.k_history) * divisor;
    const float v = convolve(a.rows, t, a.kv, a.kv_stride, v_column, c, a.v_conv + c * kTaps,
                             v_taps, a.v_history) * divisor;
    float sums[2] = {k * k, v * v};
    block_sums(sums);
    const float kf = fminf(1.0f, fmaxf(0.0f, a.qk_clip / (sqrtf(sums[0]) + 1e-6f)));
    const float vn = rsqrtf(sums[1] / float(kD) + a.eps);
    const int sequence = row_sequence(a.rows, t);
    const int position = physical(a.rows, t);
    const int slot     = a.row_slots ? t : (a.ring != 0 ? position % a.ring : position);
    if (a.k_plane.coding == KvCoding::Mxfp8) {
        // A warp holds one group of 32 dimensions of the row: its exponent byte, then its codes.
        const float kx = k * kf, vx = v * vn;
        const unsigned ke = kv::mxfp8_exponent(ops::warp_max(fabsf(kx), 0xffffffffu));
        const unsigned ve = kv::mxfp8_exponent(ops::warp_max(fabsf(vx), 0xffffffffu));
        const KvPlaneView& kp = a.k_plane;
        const KvPlaneView& vp = a.v_plane;
        const std::int64_t row = std::int64_t(slot);
        kp.codes[sequence * kp.code_seq + kh * kp.code_head + row * kp.code_row + d] = kv::mxfp8_code(kx, ke);
        vp.codes[sequence * vp.code_seq + kh * vp.code_head + row * vp.code_row + d] = kv::mxfp8_code(vx, ve);
        if ((d & 31) == 0) {
            kp.scales[sequence * kp.scale_seq + kh * kp.scale_head + row * kp.scale_row + d / 32] =
                std::uint8_t(ke);
            vp.scales[sequence * vp.scale_seq + kh * vp.scale_head + row * vp.scale_row + d / 32] =
                std::uint8_t(ve);
        }
    } else {
        const std::int64_t at = std::int64_t(sequence) * a.seq_stride +
                                std::int64_t(kh) * a.head_stride + std::int64_t(slot) * kD + d;
        a.k_cache[at] = __float2half(k * kf);
        a.v_cache[at] = __float2half(v * vn);
    }
    if (a.keep) {
        const int p = a.rows.positions[t];
        history_slot(a.k_history, sequence, p)[c] = a.kv[std::int64_t(t) * a.kv_stride + c];
        history_slot(a.v_history, sequence, p)[c] = a.kv[std::int64_t(t) * a.kv_stride + v_column];
    }
}

__global__ void kv_post_kernel(KvArgs a) {
    ninfer::pdl::enter();
    kv_post_body(a, int(blockIdx.x), int(blockIdx.y), int(threadIdx.x));
}

// query_post's blocks (y < heads) and kv_post's (the rest) in one launch: independent outputs.
__global__ void qkv_post_kernel(QueryArgs q, KvArgs kv) {
    ninfer::pdl::enter();
    if (int(blockIdx.y) < q.heads) {
        query_post_body(q, int(blockIdx.x), int(blockIdx.y), int(threadIdx.x));
    } else {
        kv_post_body(kv, int(blockIdx.x), int(blockIdx.y) - q.heads, int(threadIdx.x));
    }
}

// ---- Attention ----------------------------------------------------------------------------------
// Heads of 128 values with up to four query heads per key and value head. A tile's 16 keys and
// values pass through shared memory once for the block; lane l holds dimensions [4 l, 4 l + 4) of
// the query, and a warp keeps the online softmax of its (row, head) pairs.
//   Decode and verification (kRows 1): block (row group, kv head, split) of four warps a row,
//   warp 4 i + h row i's head h. A group is one row or, with a.row_group, up to that many
//   consecutive rows of one sequence (row_group_bounds), for which the tiles pass through shared
//   memory once; each row reads the tiles of its own split as a group of one row does.
//   Prompt (kRows 8): block (eight rows, kv head) of eight warps, warp r row r, every head.
constexpr int kTile = 16;
constexpr int kSplitTiles = 4; // the fewest tiles a split reads

__device__ __forceinline__ int active_splits(int keys, int splits) {
    return min(splits, max(1, (keys + kSplitTiles * kTile - 1) / (kSplitTiles * kTile)));
}

// Group g's rows [t0, t1): consecutive rows of one sequence, at most `group_rows` of them, the
// groups in row order (group_rows 1: row g alone). False: the call has fewer groups.
__device__ __forceinline__ bool row_group_bounds(const Rows& r, int group_rows, int g, int& t0,
                                                 int& t1) {
    if (group_rows <= 1) {
        t0 = g;
        t1 = g + 1;
        return g < r.tokens;
    }
    for (int t = 1, begin = 0, count = 0; t <= r.tokens; ++t) {
        if (t == r.tokens || t - begin == group_rows ||
            row_sequence(r, t) != row_sequence(r, begin)) {
            if (count == g) {
                t0 = begin;
                t1 = t;
                return true;
            }
            ++count;
            begin = t;
        }
    }
    return false;
}

// The sums of 16 per-lane values v[k] over the warp, by halving exchanges: afterwards lanes 2j and
// 2j + 1 both hold the total of key (lane & 16 ? 8 : 0) + (lane & 8 ? 4 : 0) + (lane & 4 ? 2 : 0) +
// (lane & 2 ? 1 : 0).
__device__ __forceinline__ float key_reduce16(float (&v)[16], int lane) {
#pragma unroll
    for (int width = 8, offset = 16; width >= 1; width >>= 1, offset >>= 1) {
        const bool upper = (lane & offset) != 0;
#pragma unroll
        for (int i = 0; i < width; ++i) {
            const float send = upper ? v[i] : v[i + width];
            const float keep = upper ? v[i + width] : v[i];
            v[i]             = keep + __shfl_xor_sync(0xFFFFFFFFu, send, offset);
        }
    }
    return v[0] + __shfl_xor_sync(0xFFFFFFFFu, v[0], 1);
}

// Whether row t sees the key at physical position `key`, and its distance.
__device__ __forceinline__ bool sees(const AttentionArgs& a, int t, int key, int& distance) {
    const Rows& r = a.rows;
    const int position = r.positions[t];
    int logical        = key;
    if (r.chunk && key >= r.first) {
        const int row = key - r.first;
        if (row > t) { return false; }
        if (r.starts != nullptr && row >= r.shared && r.starts[row] != r.starts[t]) { return false; }
        logical = r.positions[row];
    } else if (key > position) {
        return false;
    }
    distance = position - logical;
    return distance >= 0 && (a.window == 0 || distance < a.window);
}

// The physical key positions [lo, hi] rows [t0, t1) read.
__device__ __forceinline__ void key_range(const AttentionArgs& a, int t0, int t1, int& lo, int& hi) {
    if (a.uniform) {
        // Blocks of eight rows, each one sequence's DSpark block: its last row's keys.
        hi = a.rows.positions[min(a.rows.tokens, (t0 / 8) * 8 + 8) - 1];
        lo = a.window != 0 ? max(0, hi - a.window + 1) : 0;
        return;
    }
    lo = INT_MAX;
    hi = 0;
    for (int t = t0; t < t1; ++t) {
        const int position = a.rows.positions[t];
        lo = min(lo, a.window != 0 ? max(0, position - a.window + 1) : 0);
        hi = max(hi, physical(a.rows, t));
    }
}

// The far field of a full visit's prompt chunk: for the rows [g, g + kFarRows), the keys [0, end)
// before the chunk that lie at least `extent` positions before every row's logical position, so
// every row of the group sees each of them, and none with a relative bias; `end` is a whole number
// of kFarKeys tiles. Each warp computes it (all lanes taking part) from the group's positions.
constexpr int kFarRows = 32;
constexpr int kFarKeys = 32;

__device__ __forceinline__ int far_end(const AttentionArgs& a, int g) {
    const int t = g + (threadIdx.x & 31);
    int p       = t < a.rows.tokens ? a.rows.positions[t] : INT_MAX;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { p = min(p, __shfl_xor_sync(0xFFFFFFFFu, p, o)); }
    const int end = min(a.rows.first, p - a.extent + 1);
    return end <= 0 ? 0 : end / kFarKeys * kFarKeys;
}

// The keys [begin, end) that split `split` of the rows [t0, t1) reads, none before `from`; false
// when the rows use fewer splits (split_rows: as many as their keys need, else one).
__device__ __forceinline__ bool split_keys(const AttentionArgs& a, int t0, int t1, int from,
                                           bool split_rows, int split, int& begin, int& end) {
    int lo = 0, hi = 0;
    key_range(a, t0, t1, lo, hi);
    lo               = max(lo, from);
    const int keys   = hi + 1 - lo;
    const int active = split_rows ? active_splits(keys, a.splits) : 1;
    const int chunk  = ((keys + active - 1) / active + kTile - 1) / kTile * kTile;
    begin            = lo + split * chunk;
    end              = min(hi + 1, begin + chunk);
    return split < active;
}

// kFar: a prompt chunk whose far field far_attention_kernel computed (a.far): its rows start from
// those states and read the keys after the far field. A separate instantiation keeps the other
// calls' kernels free of that code.
template <int kRows, bool kFar = false>
__global__ void __launch_bounds__(kRows == 1 ? 128 * kMaxRowGroup : 256)
    attention_kernel(AttentionArgs a) {
    ninfer::pdl::enter();
    constexpr int kWarps = kRows == 1 ? 4 * kMaxRowGroup : 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group   = a.query_heads / a.kv_heads;
    const int kv_head = blockIdx.y;
    int t0 = 0, t1 = 0;
    if constexpr (kRows == 1) {
        if (!row_group_bounds(a.rows, a.row_group, int(blockIdx.x), t0, t1)) { return; }
    } else {
        t0 = blockIdx.x * kRows;
        t1 = min(a.rows.tokens, t0 + kRows);
    }
    const int t       = kRows == 1 ? t0 + warp / 4 : t0 + warp;
    const int head_w  = kRows == 1 ? warp % 4 : 0;
    const bool in_row = t < t1;
    const bool row_ok = in_row && (kRows != 1 || head_w < group);
    const int heads   = kRows == 1 ? 1 : group;
    const int head0   = kv_head * group + head_w;
    // A prompt chunk of a full visit with its far field done: the keys after it.
    int far = 0;
    if constexpr (kFar) { far = far_end(a, t0 / kFarRows * kFarRows); }
    const int split = int(blockIdx.z);
    // The row's keys [begin, end) of its split (every row's, kRows 8), and the tiles [first, last)
    // the block loads for its rows.
    int begin = 0, end = 0, first = INT_MAX, last = 0;
    bool active = false;
    if constexpr (kRows == 1) {
        active = in_row && split_keys(a, t, t + 1, far, true, split, begin, end);
        bool any = false;
        for (int r = t0; r < t1; ++r) {
            int b = 0, e = 0;
            if (split_keys(a, r, r + 1, far, true, split, b, e)) {
                any = true;
                if (b < e) {
                    first = min(first, b);
                    last  = max(last, e);
                }
            }
        }
        if (!any) { return; }
    } else {
        active = split_keys(a, t0, t1, far, a.uniform, split, begin, end);
        if (!active) { return; }
        first = begin;
        last  = end;
    }

    __shared__ __align__(16) __half k_s[kTile][kD];
    __shared__ __align__(16) __half v_s[kTile][kD];
    // A decode warp holds one head's relative features; a prompt warp every head's.
    constexpr int kRelHeads = kRows == 1 ? 1 : kMaxGroup;
    __shared__ float r_s[kWarps][kRelHeads][kRel];

    float q_r[kMaxGroup][4], acc[kMaxGroup][4], m[kMaxGroup], l[kMaxGroup];
#pragma unroll
    for (int h = 0; h < kMaxGroup; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[h][j] = 0.0f;
            q_r[h][j] = 0.0f;
        }
        if (row_ok && h < heads) {
            const std::int64_t at = std::int64_t(t) * a.query_heads + head0 + h;
            const float4 q = reinterpret_cast<const float4*>(a.q + at * kD)[lane];
            q_r[h][0] = q.x * a.scale;
            q_r[h][1] = q.y * a.scale;
            q_r[h][2] = q.z * a.scale;
            q_r[h][3] = q.w * a.scale;
            if (lane < kRel && a.r != nullptr) { r_s[warp][h % kRelHeads][lane] = a.r[at * kRel + lane]; }
            if (kFar && far > 0) {
                const std::int64_t n = std::int64_t(a.rows.tokens) * a.query_heads;
                const float4 o = reinterpret_cast<const float4*>(a.far + at * kD)[lane];
                acc[h][0] = o.x;
                acc[h][1] = o.y;
                acc[h][2] = o.z;
                acc[h][3] = o.w;
                m[h] = a.far[n * kD + 2 * at];
                l[h] = a.far[n * kD + 2 * at + 1];
            }
        }
    }
    const int sequence = row_sequence(a.rows, t0);
    const std::int64_t base = std::int64_t(sequence) * a.seq_stride +
                              std::int64_t(kv_head) * a.head_stride;
    // After the score butterfly lanes 2j and 2j + 1 hold key `lane_key`; each computes half of its
    // relative bias.
    const int lane_key = ((lane & 16) ? 8 : 0) + ((lane & 8) ? 4 : 0) + ((lane & 4) ? 2 : 0) +
                         ((lane & 2) ? 1 : 0);
    const int half = lane & 1;
    for (int tile = first; tile < last; tile += kTile) {
        const int count = min(kTile, last - tile);
        __syncthreads();
        for (int w = threadIdx.x; w < 2 * kTile * (kD / 8); w += blockDim.x) {
            const int matrix = w / (kTile * (kD / 8));
            const int rest   = w % (kTile * (kD / 8));
            const int r = rest / (kD / 8), c = rest % (kD / 8);
            uint4 word = make_uint4(0, 0, 0, 0);
            const KvPlaneView& plane = matrix == 0 ? a.k_plane : a.v_plane;
            if (r < count) {
                const int p    = tile + r;
                const int slot = a.ring != 0 ? p % a.ring : p;
                if (plane.coding != KvCoding::F16) {
                    const std::int64_t codes_row =
                        std::int64_t(sequence) * plane.code_seq +
                        std::int64_t(kv_head) * plane.code_head +
                        std::int64_t(slot) * plane.code_row;
                    const std::int64_t scales_row =
                        std::int64_t(sequence) * plane.scale_seq +
                        std::int64_t(kv_head) * plane.scale_head +
                        std::int64_t(slot) * plane.scale_row;
                    word = chunk_word(plane, plane.codes + codes_row, plane.scales + scales_row, c);
                } else {
                    const __half* cache = matrix == 0 ? a.k_cache : a.v_cache;
                    word = *reinterpret_cast<const uint4*>(cache + base + std::int64_t(slot) * kD + c * 8);
                }
            }
            *reinterpret_cast<uint4*>(matrix == 0 ? &k_s[r][c * 8] : &v_s[r][c * 8]) = word;
        }
        __syncthreads();
        if (!row_ok || !active || tile < begin || tile >= end) { continue; }
        int distance = 0;
        const bool visible =
            lane_key < min(kTile, end - tile) && (a.uniform || sees(a, t, tile + lane_key, distance));
        const bool biased  = visible && !a.uniform && distance < a.extent;
#pragma unroll
        for (int h = 0; h < kMaxGroup; ++h) {
            if (h >= heads) { continue; }
            float part[kTile];
#pragma unroll
            for (int kk = 0; kk < kTile; ++kk) {
                const uint2 w = *reinterpret_cast<const uint2*>(&k_s[kk][4 * lane]);
                const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&w.x));
                const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&w.y));
                part[kk] = q_r[h][0] * f0.x + q_r[h][1] * f0.y + q_r[h][2] * f1.x + q_r[h][3] * f1.y;
            }
            const float dot = key_reduce16(part, lane);
            float bias = 0.0f;
            if (biased) {
                const float* row = a.table + std::int64_t(distance) * kRel + 8 * half;
#pragma unroll
                for (int c = 0; c < 8; ++c) { bias += r_s[warp][h % kRelHeads][8 * half + c] * row[c]; }
            }
            bias += __shfl_xor_sync(0xFFFFFFFFu, bias, 1);
            const float sc     = visible ? dot + bias : -INFINITY;
            const float m_tile = warp_max(sc);
            const float m_new  = fmaxf(m[h], m_tile);
            if (m_new == -INFINITY) { continue; }
            const float scale = expf(m[h] - m_new);
            const float p     = visible ? expf(sc - m_new) : 0.0f;
            // Every key's probability sits on two lanes.
            l[h] = l[h] * scale + 0.5f * warp_sum(p);
            m[h] = m_new;
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[h][j] *= scale; }
#pragma unroll
            for (int kk = 0; kk < kTile; ++kk) {
                const int src = ((kk & 8) ? 16 : 0) | ((kk & 4) ? 8 : 0) | ((kk & 2) ? 4 : 0) |
                                ((kk & 1) ? 2 : 0);
                const float pk = __shfl_sync(0xFFFFFFFFu, p, src);
                const uint2 w = *reinterpret_cast<const uint2*>(&v_s[kk][4 * lane]);
                const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&w.x));
                const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&w.y));
                acc[h][0] += pk * f0.x;
                acc[h][1] += pk * f0.y;
                acc[h][2] += pk * f1.x;
                acc[h][3] += pk * f1.y;
            }
        }
    }
    if (!row_ok || !active) { return; }
    const int width = a.query_heads * kD;
#pragma unroll
    for (int h = 0; h < kMaxGroup; ++h) {
        if (h >= heads) { continue; }
        const int head = head0 + h;
        if (a.splits > 1) {
            // Partials: [B][Hq][splits] rows of D, then [B][Hq][splits] (m, l) pairs.
            const std::int64_t rows = std::int64_t(a.rows.tokens) * a.query_heads * a.splits;
            const std::int64_t row  = (std::int64_t(t) * a.query_heads + head) * a.splits + split;
            *reinterpret_cast<float4*>(a.partial + row * kD + 4 * lane) =
                make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
            if (lane == 0) {
                a.partial[rows * kD + 2 * row]     = m[h];
                a.partial[rows * kD + 2 * row + 1] = l[h];
            }
            continue;
        }
        const float sink = a.sinks != nullptr ? a.sinks[std::int64_t(t) * a.sink_stride + head]
                                              : -INFINITY;
        const float top  = fmaxf(m[h], sink);
        const float w    = expf(m[h] - top);
        const float inv  = 1.0f / (l[h] * w + expf(sink - top));
        float4 g         = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        if (a.gate != nullptr) {
            const float4 x = reinterpret_cast<const float4*>(
                a.gate + std::int64_t(t) * a.gate_stride + head * kD)[lane];
            g = make_float4(sigmoidf(x.x), sigmoidf(x.y), sigmoidf(x.z), sigmoidf(x.w));
        }
        const float4 y = make_float4(acc[h][0] * w * inv * g.x, acc[h][1] * w * inv * g.y,
                                     acc[h][2] * w * inv * g.z, acc[h][3] * w * inv * g.w);
        if (a.out_q.qs != nullptr) {
            store_q8x4(a.out_q, width, t, head * (kD / 4) + lane, y, true);
        } else {
            store_bf16x4(a.out + std::int64_t(t) * width + head * kD + 4 * lane, y);
        }
    }
}

// ---- Prompt attention on the tensor cores --------------------------------------------------------
// A prompt chunk without branches (rows at positions first + t, logical and physical alike) by
// mma.sync m16n8k16 (F16 in, FP32 sums), FlashAttention-2 style. Block (32 rows, kv head) of
// kGroup * 2 warps, warp (g, h) rows [16 g, 16 g + 16) of the group's head h; the block's tiles of
// 32 keys and values pass through shared memory as F16 (an E4M3 cache's codes converted times
// its scale), double-buffered. Scores run in the log2 domain. The relative bias of a tile within
// `extent` of its rows is one more product, R (16 x 16) by the table rows of the 48 distances the
// tile spans, gathered from the warp's shared scratch. Q and P round to F16 for the products: the
// sums differ from attention_kernel's FP32 ones in the last bits.
constexpr int kTcRows   = 16;              // rows a warp
constexpr int kTcGroups = 2;               // row groups a block
constexpr int kTcKeys   = 32;              // keys a tile
constexpr int kTcSpan   = kTcKeys + kTcRows; // distances a tile spans for a warp's rows

__device__ __forceinline__ std::uint32_t pack_half2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const std::uint32_t*>(&h);
}
__device__ __forceinline__ void mma16816(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0,
                                         std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                 "{%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], const void* p) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at));
}
__device__ __forceinline__ void ldsm_x4_t(std::uint32_t (&r)[4], const void* p) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at));
}
// Element offset of (key, 16-byte chunk c) in a [kTcKeys][kD] F16 tile, chunks XOR-swizzled by
// key so ldmatrix's eight rows hit eight different bank groups.
__device__ __forceinline__ int tc_at(int key, int c) { return key * kD + ((c ^ (key & 7)) << 3); }

std::size_t attention_tc_smem(int group) {
    return std::size_t(2 * 2 * kTcKeys * kD) * sizeof(__half) +
           std::size_t(group * kTcGroups * kTcRows * kTcSpan) * sizeof(float);
}

// A launch's KV form: both planes F16, both MXFP8 (decoded when stored), or any (chunk_word).
constexpr int kKvF16 = 0, kKvMxfp8 = 1, kKvAny = 2;

__host__ int kv_mode(const AttentionArgs& a) {
    if (a.k_plane.coding == KvCoding::F16 && a.v_plane.coding == KvCoding::F16) { return kKvF16; }
    if (a.k_plane.coding == KvCoding::Mxfp8 && a.v_plane.coding == KvCoding::Mxfp8) { return kKvMxfp8; }
    return kKvAny;
}

template <int kGroup, int kMode>
__global__ void __launch_bounds__(kGroup * kTcGroups * 32) attention_tc_kernel(AttentionArgs a) {
    ninfer::pdl::enter();
    constexpr int kWarps   = kGroup * kTcGroups;
    constexpr int kThreads = kWarps * 32;
    extern __shared__ __align__(16) unsigned char tc_smem[];
    __half* tiles = reinterpret_cast<__half*>(tc_smem); // [stage][K, V][kTcKeys][kD]
    float* scratch = reinterpret_cast<float*>(tc_smem + std::size_t(2 * 2 * kTcKeys * kD) * sizeof(__half));
    const int warp = int(threadIdx.x) >> 5, lane = int(threadIdx.x) & 31;
    const int g = warp / kGroup, hq = warp % kGroup;
    const int kv_head = int(blockIdx.y);
    const int head    = kv_head * kGroup + hq;
    const int T       = a.rows.tokens;
    const int first   = a.rows.first;
    // The longest blocks (latest rows) first.
    const int t0  = (int(gridDim.x) - 1 - int(blockIdx.x)) * kTcRows * kTcGroups;
    const int wt0 = t0 + g * kTcRows;
    const int r = lane >> 2, q = lane & 3;
    constexpr float kLog2e = 1.4426950408889634f;

    const int block_last = min(T, t0 + kTcRows * kTcGroups) - 1;
    const int lo = a.window != 0 ? max(0, first + t0 - a.window + 1) : 0;
    const int hi = first + block_last;
    const bool live = wt0 < T;
    const int w_last = min(T - 1, wt0 + kTcRows - 1);
    const int w_lo   = a.window != 0 ? max(0, first + wt0 - a.window + 1) : 0;
    const int w_hi   = first + w_last;

    // Q (16 rows x 128) as eight A fragments, times scale * log2(e); R likewise, times log2(e).
    std::uint32_t qa[kD / 16][4];
    std::uint32_t ra[4] = {0, 0, 0, 0};
    {
        const int ta = wt0 + r, tb = wt0 + r + 8;
        const bool va = ta < T, vb = tb < T;
        const float* qra = a.q + (std::int64_t(va ? ta : 0) * a.query_heads + head) * kD;
        const float* qrb = a.q + (std::int64_t(vb ? tb : 0) * a.query_heads + head) * kD;
        const float s = a.scale * kLog2e;
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks) {
            const int c = 16 * ks + 2 * q;
            const float2 x0 = va ? *reinterpret_cast<const float2*>(qra + c) : make_float2(0, 0);
            const float2 x1 = vb ? *reinterpret_cast<const float2*>(qrb + c) : make_float2(0, 0);
            const float2 x2 = va ? *reinterpret_cast<const float2*>(qra + c + 8) : make_float2(0, 0);
            const float2 x3 = vb ? *reinterpret_cast<const float2*>(qrb + c + 8) : make_float2(0, 0);
            qa[ks][0] = pack_half2(x0.x * s, x0.y * s);
            qa[ks][1] = pack_half2(x1.x * s, x1.y * s);
            qa[ks][2] = pack_half2(x2.x * s, x2.y * s);
            qa[ks][3] = pack_half2(x3.x * s, x3.y * s);
        }
        if (a.extent > 0) {
            const float* rra = a.r + (std::int64_t(va ? ta : 0) * a.query_heads + head) * kRel;
            const float* rrb = a.r + (std::int64_t(vb ? tb : 0) * a.query_heads + head) * kRel;
            const float2 x0 = va ? *reinterpret_cast<const float2*>(rra + 2 * q) : make_float2(0, 0);
            const float2 x1 = vb ? *reinterpret_cast<const float2*>(rrb + 2 * q) : make_float2(0, 0);
            const float2 x2 = va ? *reinterpret_cast<const float2*>(rra + 2 * q + 8) : make_float2(0, 0);
            const float2 x3 = vb ? *reinterpret_cast<const float2*>(rrb + 2 * q + 8) : make_float2(0, 0);
            ra[0] = pack_half2(x0.x * kLog2e, x0.y * kLog2e);
            ra[1] = pack_half2(x1.x * kLog2e, x1.y * kLog2e);
            ra[2] = pack_half2(x2.x * kLog2e, x2.y * kLog2e);
            ra[3] = pack_half2(x3.x * kLog2e, x3.y * kLog2e);
        }
    }

    float o[kD / 8][4];
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f; }
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    const int sequence = row_sequence(a.rows, 0);
    const std::int64_t base = std::int64_t(sequence) * a.seq_stride + std::int64_t(kv_head) * a.head_stride;
    constexpr int kChunks = 2 * kTcKeys * (kD / 8);
    constexpr int kPer    = (kChunks + kThreads - 1) / kThreads;
    // Prefetch registers: F16 and decoded words (uint4), or, MXFP8, the raw codes (uint2) with
    // their group exponents four to a word: a thread's chunks share their column (keeps the
    // prefetch out of local memory).
    constexpr bool kMx = kMode == kKvMxfp8;
    uint4 raw[kMx ? 1 : kPer];
    uint2 mxc[kMx ? kPer : 1];
    // MXFP8: lane l's scale words (four group exponents) of the tile's key l, K then V.
    std::uint32_t mx_kw = 0, mx_vw = 0;
    static_assert(kTcKeys == 32, "MXFP8 scale lanes: 32 keys a tile");
    const auto fetch = [&](int tile) {
        if constexpr (kMx) {
            const int p = tile + lane;
            mx_kw = mx_vw = 0;
            if (p <= hi) {
                const int slot = a.ring != 0 ? p % a.ring : p;
                const auto word = [&](const KvPlaneView& plane) {
                    return *reinterpret_cast<const std::uint32_t*>(
                        plane.scales + std::int64_t(sequence) * plane.scale_seq +
                        std::int64_t(kv_head) * plane.scale_head + std::int64_t(slot) * plane.scale_row);
                };
                mx_kw = word(a.k_plane);
                mx_vw = word(a.v_plane);
            }
        }
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int w = int(threadIdx.x) + i * kThreads;
            if constexpr (kMx) {
                mxc[i] = make_uint2(0, 0);
            } else {
                raw[i] = make_uint4(0, 0, 0, 0);
            }
            if (w >= kChunks) { continue; }
            const int matrix = w / (kTcKeys * (kD / 8));
            const int rest   = w % (kTcKeys * (kD / 8));
            const int key = rest / (kD / 8), c = rest % (kD / 8);
            const int p   = tile + key;
            if (p > hi) { continue; }
            const int slot      = a.ring != 0 ? p % a.ring : p;
            const KvPlaneView& plane = matrix == 0 ? a.k_plane : a.v_plane;
            if constexpr (kMode == kKvF16) {
                const __half* cache = matrix == 0 ? a.k_cache : a.v_cache;
                raw[i] = *reinterpret_cast<const uint4*>(cache + base + std::int64_t(slot) * kD + c * 8);
            } else if constexpr (kMx) {
                mxc[i] = *reinterpret_cast<const uint2*>(
                    plane.codes + std::int64_t(sequence) * plane.code_seq +
                    std::int64_t(kv_head) * plane.code_head + std::int64_t(slot) * plane.code_row + c * 8);
            } else if (plane.coding != KvCoding::F16) {
                // A quantized plane's chunk decoded to eight F16 values.
                raw[i] = chunk_word(plane,
                                    plane.codes + std::int64_t(sequence) * plane.code_seq +
                                        std::int64_t(kv_head) * plane.code_head +
                                        std::int64_t(slot) * plane.code_row,
                                    plane.scales + std::int64_t(sequence) * plane.scale_seq +
                                        std::int64_t(kv_head) * plane.scale_head +
                                        std::int64_t(slot) * plane.scale_row,
                                    c);
            } else {
                const __half* cache = matrix == 0 ? a.k_cache : a.v_cache;
                raw[i] = *reinterpret_cast<const uint4*>(cache + base + std::int64_t(slot) * kD + c * 8);
            }
        }
    };
    const auto store = [&](int stage) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int w = int(threadIdx.x) + i * kThreads;
            if (w >= kChunks) { continue; }
            const int matrix = w / (kTcKeys * (kD / 8));
            const int rest   = w % (kTcKeys * (kD / 8));
            const int key = rest / (kD / 8), c = rest % (kD / 8);
            uint4 word;
            if constexpr (kMx) {
                // The row's exponents from the lane holding its key (both planes' words move,
                // then the row's matrix picks one).
                const std::uint32_t kw = __shfl_sync(0xFFFFFFFFu, mx_kw, key);
                const std::uint32_t vw = __shfl_sync(0xFFFFFFFFu, mx_vw, key);
                const unsigned e = ((matrix == 0 ? kw : vw) >> (8 * (c / 4))) & 0xFFu;
                if (e >= 113u && e <= 142u) {
                    // The scale 2^(e - 127) a normal FP16 power of two: code (exact in FP16)
                    // times it is exact, the float decode's value.
                    const __half2 sc = __half2half2(__ushort_as_half(
                        static_cast<unsigned short>((e - 127u + 15u) << 10)));
                    const std::uint32_t parts[2] = {mxc[i].x, mxc[i].y};
                    __half2 h[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(
                            __nv_fp8x2_storage_t(parts[q / 2] >> (16 * (q % 2))), __NV_E4M3);
                        h[q] = __hmul2(__half2(hr), sc);
                    }
                    word = *reinterpret_cast<const uint4*>(h);
                } else {
                    float f[8];
                    ops::kv_row::Chunk<KvCoding::Mxfp8>::decode(mxc[i], std::uint8_t(e), f);
                    word = ops::kv_row::pack8(f);
                }
            } else {
                word = raw[i];
            }
            *reinterpret_cast<uint4*>(tiles + (stage * 2 + matrix) * kTcKeys * kD + tc_at(key, c)) = word;
        }
    };

    float* bias = scratch + warp * kTcRows * kTcSpan;
    if (lo <= hi) { fetch(lo); }
    int stage = 0;
    for (int tile = lo; tile <= hi; tile += kTcKeys, stage ^= 1) {
        store(stage);
        __syncthreads();
        if (tile + kTcKeys <= hi) { fetch(tile + kTcKeys); }
        if (!live || tile > w_hi || tile + kTcKeys - 1 < w_lo) { continue; }
        const __half* ks_ = tiles + (stage * 2 + 0) * kTcKeys * kD;
        const __half* vs_ = tiles + (stage * 2 + 1) * kTcKeys * kD;
        // S = Q K^T: four n-tiles of eight keys.
        float s[kTcKeys / 8][4];
#pragma unroll
        for (int i = 0; i < kTcKeys / 8; ++i) { s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f; }
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks) {
#pragma unroll
            for (int np = 0; np < kTcKeys / 16; ++np) {
                std::uint32_t b[4];
                const int key = np * 16 + ((lane >> 4) << 3) + (lane & 7);
                ldsm_x4(b, ks_ + tc_at(key, 2 * ks + ((lane >> 3) & 1)));
                mma16816(s[2 * np], qa[ks], b[0], b[1]);
                mma16816(s[2 * np + 1], qa[ks], b[2], b[3]);
            }
        }
        // The relative bias of the distances [dmin, dmin + kTcSpan) this tile spans.
        const int p0   = first + wt0;
        const int dmin = p0 - (tile + kTcKeys - 1);
        const bool biased = a.extent > 0 && p0 + kTcRows - 1 - tile >= 0 && dmin < a.extent;
        if (biased) {
#pragma unroll
            for (int nt = 0; nt < kTcSpan / 8; ++nt) {
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const int d = dmin + nt * 8 + r;
                std::uint32_t b0 = 0, b1 = 0;
                if (d >= 0 && d < a.extent) {
                    const float* row = a.table + std::int64_t(d) * kRel;
                    const float2 x = *reinterpret_cast<const float2*>(row + 2 * q);
                    const float2 y = *reinterpret_cast<const float2*>(row + 2 * q + 8);
                    b0 = pack_half2(x.x, x.y);
                    b1 = pack_half2(y.x, y.y);
                }
                mma16816(c, ra, b0, b1);
                bias[r * kTcSpan + nt * 8 + 2 * q]           = c[0];
                bias[r * kTcSpan + nt * 8 + 2 * q + 1]       = c[1];
                bias[(r + 8) * kTcSpan + nt * 8 + 2 * q]     = c[2];
                bias[(r + 8) * kTcSpan + nt * 8 + 2 * q + 1] = c[3];
            }
            __syncwarp();
        }
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nt = 0; nt < kTcKeys / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = r + 8 * (e >> 1);
                const int col = nt * 8 + 2 * q + (e & 1);
                const int key = tile + col;
                const int d   = p0 + row - key;
                const bool visible = key <= hi && wt0 + row < T && d >= 0 &&
                                     (a.window == 0 || d < a.window);
                float v = s[nt][e];
                if (biased && visible && d < a.extent) { v += bias[row * kTcSpan + row + kTcKeys - 1 - col]; }
                s[nt][e] = visible ? v : -INFINITY;
                mx[e >> 1] = fmaxf(mx[e >> 1], s[nt][e]);
            }
        }
        if (biased) { __syncwarp(); }
        float alpha[2];
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            mx[h2] = fmaxf(mx[h2], __shfl_xor_sync(0xFFFFFFFFu, mx[h2], 1));
            mx[h2] = fmaxf(mx[h2], __shfl_xor_sync(0xFFFFFFFFu, mx[h2], 2));
            const float m_new = fmaxf(m[h2], mx[h2]);
            alpha[h2] = m_new == -INFINITY ? 1.0f : exp2f(m[h2] - m_new);
            m[h2]     = m_new;
        }
        float sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int nt = 0; nt < kTcKeys / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float mm = m[e >> 1];
                const float p  = mm == -INFINITY ? 0.0f : exp2f(s[nt][e] - mm);
                s[nt][e] = p;
                sum[e >> 1] += p;
            }
        }
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            sum[h2] += __shfl_xor_sync(0xFFFFFFFFu, sum[h2], 1);
            sum[h2] += __shfl_xor_sync(0xFFFFFFFFu, sum[h2], 2);
            l[h2] = l[h2] * alpha[h2] + sum[h2];
        }
#pragma unroll
        for (int i = 0; i < kD / 8; ++i) {
            o[i][0] *= alpha[0];
            o[i][1] *= alpha[0];
            o[i][2] *= alpha[1];
            o[i][3] *= alpha[1];
        }
        // O += P V.
#pragma unroll
        for (int kk = 0; kk < kTcKeys / 16; ++kk) {
            const std::uint32_t pa[4] = {pack_half2(s[2 * kk][0], s[2 * kk][1]),
                                         pack_half2(s[2 * kk][2], s[2 * kk][3]),
                                         pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                         pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int np = 0; np < kD / 16; ++np) {
                std::uint32_t b[4];
                const int key = 16 * kk + (((lane >> 3) & 1) << 3) + (lane & 7);
                ldsm_x4_t(b, vs_ + tc_at(key, 2 * np + (lane >> 4)));
                mma16816(o[2 * np], pa, b[0], b[1]);
                mma16816(o[2 * np + 1], pa, b[2], b[3]);
            }
        }
    }
    if (!live) { return; }
    const int width = a.query_heads * kD;
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
        const int t       = wt0 + r + 8 * h2;
        const bool row_ok = t < T;
        const int tt      = row_ok ? t : 0;
        const float sink  = a.sinks != nullptr ? a.sinks[std::int64_t(tt) * a.sink_stride + head] * kLog2e
                                               : -INFINITY;
        const float top = fmaxf(m[h2], sink);
        const float w   = m[h2] == -INFINITY ? 0.0f : exp2f(m[h2] - top);
        const float inv = 1.0f / (l[h2] * w + (sink == -INFINITY ? 0.0f : exp2f(sink - top)));
        const float* gate = a.gate != nullptr ? a.gate + std::int64_t(tt) * a.gate_stride + head * kD : nullptr;
        float y[kD / 8][2];
#pragma unroll
        for (int i = 0; i < kD / 8; ++i) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int dcol = i * 8 + 2 * q + e;
                const float gv = gate != nullptr ? sigmoidf(gate[dcol]) : 1.0f;
                y[i][e] = o[i][2 * h2 + e] * w * inv * gv;
            }
        }
        if (a.out_q.qs != nullptr) {
            // q8_1 blocks of 32 values: n-tiles 4b..4b+3, eight values a thread, the quad's four
            // threads one block.
#pragma unroll
            for (int b = 0; b < kD / 32; ++b) {
                float amax = 0.0f, total = 0.0f;
#pragma unroll
                for (int i = 4 * b; i < 4 * b + 4; ++i) {
                    amax = fmaxf(amax, fmaxf(fabsf(y[i][0]), fabsf(y[i][1])));
                    total += y[i][0] + y[i][1];
                }
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
                total += __shfl_xor_sync(0xFFFFFFFFu, total, 1);
                total += __shfl_xor_sync(0xFFFFFFFFu, total, 2);
                if (!row_ok) { continue; }
                const float dq = amax / 127.0f;
#pragma unroll
                for (int i = 4 * b; i < 4 * b + 4; ++i) {
                    char2 c = make_char2(0, 0);
                    if (amax != 0.0f) {
                        c = make_char2(static_cast<signed char>(roundf(y[i][0] / dq)),
                                       static_cast<signed char>(roundf(y[i][1] / dq)));
                    }
                    *reinterpret_cast<char2*>(a.out_q.qs + std::int64_t(t) * width + head * kD + i * 8 + 2 * q) = c;
                }
                if (q == 0) {
                    a.out_q.ds[std::int64_t(t) * (width / 32) + (head * kD) / 32 + b] =
                        make_half2(__float2half(dq), __float2half(total));
                }
            }
        } else if (row_ok) {
#pragma unroll
            for (int i = 0; i < kD / 8; ++i) {
                *reinterpret_cast<__nv_bfloat162*>(a.out + std::int64_t(t) * width + head * kD + i * 8 + 2 * q) =
                    __floats2bfloat162_rn(y[i][0], y[i][1]);
            }
        }
    }
}

// ---- Decode and verification attention on the tensor cores -----------------------------------
// Span-split calls (AttentionArgs::span; 32-key tiles, spans multiples of 32): block (run of up to
// eight rows of one sequence, kv head, split), its (row, head) pairs m = row * kGroup + head the
// rows of two m16 tiles, one a warp. Every pair's arithmetic depends on its own row alone (each
// mma output element sums its own row's products; tiles at fixed positions), so a row's partials
// are the same in a decode call and in any verification run. Partials as attention_kernel's.
constexpr int kDtKeys  = 32;
constexpr int kDtWarps = 2;
constexpr int kDtSpan  = kDtKeys + 8; // distances a tile spans for a run's rows

// One K/V tile stage (a barrier after each tile's compute frees it; the next tile waits in
// registers): 21 KB a block, four blocks a multiprocessor where two stages allowed two.
constexpr int kDtStages = 1;

std::size_t attention_dtc_smem() {
    return std::size_t(kDtStages * 2 * kDtKeys * kD) * sizeof(__half) +
           std::size_t(kDtWarps * 16 * kDtSpan) * sizeof(float);
}

template <int kGroup, int kMode>
__global__ void __launch_bounds__(kDtWarps * 32) attention_dtc_kernel(AttentionArgs a) {
    ninfer::pdl::enter();
    constexpr int kThreads = kDtWarps * 32;
    extern __shared__ __align__(16) unsigned char dt_smem[];
    __half* tiles  = reinterpret_cast<__half*>(dt_smem);
    float* scratch = reinterpret_cast<float*>(dt_smem + std::size_t(kDtStages * 2 * kDtKeys * kD) * sizeof(__half));
    const int warp = int(threadIdx.x) >> 5, lane = int(threadIdx.x) & 31;
    const int r = lane >> 2, q = lane & 3;
    const int kv_head = int(blockIdx.y);
    // Uniform calls (a DSpark block of eight rows a block): every row sees its block's keys.
    const int t0 = a.uniform ? int(blockIdx.x) * 8 : a.runs != nullptr ? a.runs[blockIdx.x] : int(blockIdx.x);
    const int t1 = a.uniform ? min(a.rows.tokens, t0 + 8)
                             : a.runs != nullptr ? a.runs[blockIdx.x + 1] : t0 + 1;
    if (t0 >= t1) { return; }
    const int pairs = (t1 - t0) * kGroup;
    constexpr float kLog2e = 1.4426950408889634f;
    int lo = 0, hi = 0;
    key_range(a, t0, t1, lo, hi);
    const int split = lo / a.span + int(blockIdx.z);
    const int begin = max(split * a.span, lo & ~(kDtKeys - 1));
    const int end   = min(hi + 1, (split + 1) * a.span);
    if (begin >= end) { return; }
    const int p_first = a.rows.positions[t0];

    // This warp's pairs m0 + r and m0 + r + 8.
    const int m0 = warp * 16;
    const int ma = m0 + r, mb = m0 + r + 8;
    const bool va = ma < pairs, vb = mb < pairs;
    const int ta = t0 + (va ? ma / kGroup : 0), tb = t0 + (vb ? mb / kGroup : 0);
    const int ha = kv_head * kGroup + (va ? ma % kGroup : 0), hb = kv_head * kGroup + (vb ? mb % kGroup : 0);
    const int pa = a.uniform ? hi : a.rows.positions[ta], pb = a.uniform ? hi : a.rows.positions[tb];
    const bool live = m0 < pairs;

    std::uint32_t qa[kD / 16][4];
    std::uint32_t ra[4] = {0, 0, 0, 0};
    if (live) {
        const float* qra = a.q + (std::int64_t(ta) * a.query_heads + ha) * kD;
        const float* qrb = a.q + (std::int64_t(tb) * a.query_heads + hb) * kD;
        const float s = a.scale * kLog2e;
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks) {
            const int c = 16 * ks + 2 * q;
            const float2 x0 = va ? *reinterpret_cast<const float2*>(qra + c) : make_float2(0, 0);
            const float2 x1 = vb ? *reinterpret_cast<const float2*>(qrb + c) : make_float2(0, 0);
            const float2 x2 = va ? *reinterpret_cast<const float2*>(qra + c + 8) : make_float2(0, 0);
            const float2 x3 = vb ? *reinterpret_cast<const float2*>(qrb + c + 8) : make_float2(0, 0);
            qa[ks][0] = pack_half2(x0.x * s, x0.y * s);
            qa[ks][1] = pack_half2(x1.x * s, x1.y * s);
            qa[ks][2] = pack_half2(x2.x * s, x2.y * s);
            qa[ks][3] = pack_half2(x3.x * s, x3.y * s);
        }
        if (a.extent > 0) {
            const float* rra = a.r + (std::int64_t(ta) * a.query_heads + ha) * kRel;
            const float* rrb = a.r + (std::int64_t(tb) * a.query_heads + hb) * kRel;
            const float2 x0 = va ? *reinterpret_cast<const float2*>(rra + 2 * q) : make_float2(0, 0);
            const float2 x1 = vb ? *reinterpret_cast<const float2*>(rrb + 2 * q) : make_float2(0, 0);
            const float2 x2 = va ? *reinterpret_cast<const float2*>(rra + 2 * q + 8) : make_float2(0, 0);
            const float2 x3 = vb ? *reinterpret_cast<const float2*>(rrb + 2 * q + 8) : make_float2(0, 0);
            ra[0] = pack_half2(x0.x * kLog2e, x0.y * kLog2e);
            ra[1] = pack_half2(x1.x * kLog2e, x1.y * kLog2e);
            ra[2] = pack_half2(x2.x * kLog2e, x2.y * kLog2e);
            ra[3] = pack_half2(x3.x * kLog2e, x3.y * kLog2e);
        }
    }
    float o[kD / 8][4];
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f; }
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    const int sequence = row_sequence(a.rows, t0);
    const std::int64_t base = std::int64_t(sequence) * a.seq_stride + std::int64_t(kv_head) * a.head_stride;
    constexpr int kChunks = 2 * kDtKeys * (kD / 8);
    constexpr int kPer    = kChunks / kThreads;
    static_assert(kPer * kThreads == kChunks, "whole chunks a thread");
    // Prefetch registers: F16 and decoded words (uint4), or, MXFP8, the raw codes (uint2) with
    // their group exponents four to a word: a thread's chunks share their column (keeps the
    // prefetch out of local memory).
    constexpr bool kMx = kMode == kKvMxfp8;
    uint4 raw[kMx ? 1 : kPer];
    uint2 mxc[kMx ? kPer : 1];
    // MXFP8: lane l's scale words (four group exponents) of the tile's key l, K then V.
    std::uint32_t mx_kw = 0, mx_vw = 0;
    static_assert(kDtKeys == 32 && kThreads == 64 && kPer == 16, "MXFP8 scale lanes: 32 keys, 2 warps");

    const auto fetch = [&](int tile) {
        if constexpr (kMx) {
            const int p = tile + lane;
            mx_kw = mx_vw = 0;
            if (p < end) {
                const int slot = a.ring != 0 ? p % a.ring : p;
                const auto word = [&](const KvPlaneView& plane) {
                    return *reinterpret_cast<const std::uint32_t*>(
                        plane.scales + std::int64_t(sequence) * plane.scale_seq +
                        std::int64_t(kv_head) * plane.scale_head + std::int64_t(slot) * plane.scale_row);
                };
                mx_kw = word(a.k_plane);
                mx_vw = word(a.v_plane);
            }
        }
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int w = int(threadIdx.x) + i * kThreads;
            if constexpr (kMx) {
                mxc[i] = make_uint2(0, 0);
            } else {
                raw[i] = make_uint4(0, 0, 0, 0);
            }
            const int matrix = w / (kDtKeys * (kD / 8));
            const int rest   = w % (kDtKeys * (kD / 8));
            const int key = rest / (kD / 8), c = rest % (kD / 8);
            const int p   = tile + key;
            if (p >= end) { continue; }
            const int slot      = a.ring != 0 ? p % a.ring : p;
            const KvPlaneView& plane = matrix == 0 ? a.k_plane : a.v_plane;
            if constexpr (kMode == kKvF16) {
                const __half* cache = matrix == 0 ? a.k_cache : a.v_cache;
                raw[i] = *reinterpret_cast<const uint4*>(cache + base + std::int64_t(slot) * kD + c * 8);
            } else if constexpr (kMx) {
                mxc[i] = *reinterpret_cast<const uint2*>(
                    plane.codes + std::int64_t(sequence) * plane.code_seq +
                    std::int64_t(kv_head) * plane.code_head + std::int64_t(slot) * plane.code_row + c * 8);
            } else if (plane.coding != KvCoding::F16) {
                // A quantized plane's chunk decoded to eight F16 values.
                raw[i] = chunk_word(plane,
                                    plane.codes + std::int64_t(sequence) * plane.code_seq +
                                        std::int64_t(kv_head) * plane.code_head +
                                        std::int64_t(slot) * plane.code_row,
                                    plane.scales + std::int64_t(sequence) * plane.scale_seq +
                                        std::int64_t(kv_head) * plane.scale_head +
                                        std::int64_t(slot) * plane.scale_row,
                                    c);
            } else {
                const __half* cache = matrix == 0 ? a.k_cache : a.v_cache;
                raw[i] = *reinterpret_cast<const uint4*>(cache + base + std::int64_t(slot) * kD + c * 8);
            }
        }
    };
    const auto store = [&](int stage) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int w = int(threadIdx.x) + i * kThreads;
            const int matrix = w / (kDtKeys * (kD / 8));
            const int rest   = w % (kDtKeys * (kD / 8));
            const int key = rest / (kD / 8), c = rest % (kD / 8);
            uint4 word;
            if constexpr (kMx) {
                // Chunk i's row: matrix i / 8 (K then V), key (lane / 16 + 2 warp + 4 i) % 32.
                const std::uint32_t words = __shfl_sync(0xFFFFFFFFu, i < kPer / 2 ? mx_kw : mx_vw, key);
                const unsigned e = (words >> (8 * (c / 4))) & 0xFFu;
                if (e >= 113u && e <= 142u) {
                    // The scale 2^(e - 127) a normal FP16 power of two: code (exact in FP16)
                    // times it is exact, the float decode's value.
                    const __half2 sc = __half2half2(__ushort_as_half(
                        static_cast<unsigned short>((e - 127u + 15u) << 10)));
                    const std::uint32_t parts[2] = {mxc[i].x, mxc[i].y};
                    __half2 h[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(
                            __nv_fp8x2_storage_t(parts[q / 2] >> (16 * (q % 2))), __NV_E4M3);
                        h[q] = __hmul2(__half2(hr), sc);
                    }
                    word = *reinterpret_cast<const uint4*>(h);
                } else {
                    float f[8];
                    ops::kv_row::Chunk<KvCoding::Mxfp8>::decode(mxc[i], std::uint8_t(e), f);
                    word = ops::kv_row::pack8(f);
                }
            } else {
                word = raw[i];
            }
            *reinterpret_cast<uint4*>(tiles + (stage * 2 + matrix) * kDtKeys * kD + tc_at(key, c)) = word;
        }
    };
    float* bias = scratch + warp * 16 * kDtSpan;
    fetch(begin);
    int stage = 0;
    for (int tile = begin; tile < end; tile += kDtKeys, stage = (stage + 1) % kDtStages) {
        if (kDtStages == 1 && tile != begin) { __syncthreads(); } // the stage's last readers
        store(stage);
        __syncthreads();
        if (tile + kDtKeys < end) { fetch(tile + kDtKeys); }
        if (!live) { continue; }
        const __half* ks_ = tiles + (stage * 2 + 0) * kDtKeys * kD;
        const __half* vs_ = tiles + (stage * 2 + 1) * kDtKeys * kD;
        float s[kDtKeys / 8][4];
#pragma unroll
        for (int i = 0; i < kDtKeys / 8; ++i) { s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f; }
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks) {
#pragma unroll
            for (int np = 0; np < kDtKeys / 16; ++np) {
                std::uint32_t b[4];
                const int key = np * 16 + ((lane >> 4) << 3) + (lane & 7);
                ldsm_x4(b, ks_ + tc_at(key, 2 * ks + ((lane >> 3) & 1)));
                mma16816(s[2 * np], qa[ks], b[0], b[1]);
                mma16816(s[2 * np + 1], qa[ks], b[2], b[3]);
            }
        }
        // Relative bias: distances [dmin, dmin + kDtSpan) cover every pair's row on this tile.
        const int dmin   = p_first - (tile + kDtKeys - 1);
        const bool biased = a.extent > 0 && p_first + 7 - tile >= 0 && dmin < a.extent;
        if (biased) {
#pragma unroll
            for (int nt = 0; nt < kDtSpan / 8; ++nt) {
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const int d = dmin + nt * 8 + r;
                std::uint32_t b0 = 0, b1 = 0;
                if (d >= 0 && d < a.extent) {
                    const float* row = a.table + std::int64_t(d) * kRel;
                    const float2 x = *reinterpret_cast<const float2*>(row + 2 * q);
                    const float2 y = *reinterpret_cast<const float2*>(row + 2 * q + 8);
                    b0 = pack_half2(x.x, x.y);
                    b1 = pack_half2(y.x, y.y);
                }
                mma16816(c, ra, b0, b1);
                bias[r * kDtSpan + nt * 8 + 2 * q]           = c[0];
                bias[r * kDtSpan + nt * 8 + 2 * q + 1]       = c[1];
                bias[(r + 8) * kDtSpan + nt * 8 + 2 * q]     = c[2];
                bias[(r + 8) * kDtSpan + nt * 8 + 2 * q + 1] = c[3];
            }
            __syncwarp();
        }
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nt = 0; nt < kDtKeys / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int hrow = e >> 1;
                const int p    = hrow ? pb : pa;
                const bool ok  = hrow ? vb : va;
                const int col  = nt * 8 + 2 * q + (e & 1);
                const int key  = tile + col;
                const int d    = p - key;
                const bool visible = ok && key < end && d >= 0 && (a.window == 0 || d < a.window);
                float v = s[nt][e];
                if (biased && visible && d < a.extent) {
                    v += bias[(r + 8 * hrow) * kDtSpan + (d - dmin)];
                }
                s[nt][e] = visible ? v : -INFINITY;
                mx[hrow] = fmaxf(mx[hrow], s[nt][e]);
            }
        }
        if (biased) { __syncwarp(); }
        float alpha[2];
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            mx[h2] = fmaxf(mx[h2], __shfl_xor_sync(0xFFFFFFFFu, mx[h2], 1));
            mx[h2] = fmaxf(mx[h2], __shfl_xor_sync(0xFFFFFFFFu, mx[h2], 2));
            const float m_new = fmaxf(m[h2], mx[h2]);
            alpha[h2] = m_new == -INFINITY ? 1.0f : exp2f(m[h2] - m_new);
            m[h2]     = m_new;
        }
        float sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int nt = 0; nt < kDtKeys / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float mm = m[e >> 1];
                const float p  = mm == -INFINITY ? 0.0f : exp2f(s[nt][e] - mm);
                s[nt][e] = p;
                sum[e >> 1] += p;
            }
        }
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            sum[h2] += __shfl_xor_sync(0xFFFFFFFFu, sum[h2], 1);
            sum[h2] += __shfl_xor_sync(0xFFFFFFFFu, sum[h2], 2);
            l[h2] = l[h2] * alpha[h2] + sum[h2];
        }
#pragma unroll
        for (int i = 0; i < kD / 8; ++i) {
            o[i][0] *= alpha[0];
            o[i][1] *= alpha[0];
            o[i][2] *= alpha[1];
            o[i][3] *= alpha[1];
        }
#pragma unroll
        for (int kk = 0; kk < kDtKeys / 16; ++kk) {
            const std::uint32_t pa4[4] = {pack_half2(s[2 * kk][0], s[2 * kk][1]),
                                          pack_half2(s[2 * kk][2], s[2 * kk][3]),
                                          pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                          pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int np = 0; np < kD / 16; ++np) {
                std::uint32_t b[4];
                const int key = 16 * kk + (((lane >> 3) & 1) << 3) + (lane & 7);
                ldsm_x4_t(b, vs_ + tc_at(key, 2 * np + (lane >> 4)));
                mma16816(o[2 * np], pa4, b[0], b[1]);
                mma16816(o[2 * np + 1], pa4, b[2], b[3]);
            }
        }
    }
    if (!live) { return; }
    // Partials of each valid pair: [B][Hq][splits] rows of D, then (m, l) pairs; a row's splits
    // from its own first key's.
    const std::int64_t prow = std::int64_t(a.rows.tokens) * a.query_heads * a.splits;
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
        const bool ok = h2 ? vb : va;
        if (!ok) { continue; }
        const int t = h2 ? tb : ta, head = h2 ? hb : ha, p = h2 ? pb : pa;
        const int row_lo = a.window != 0 ? max(0, p - a.window + 1) : 0;
        const int local  = split - row_lo / a.span;
        if (local < 0 || split > p / a.span) { continue; }
        const std::int64_t row = (std::int64_t(t) * a.query_heads + head) * a.splits + local;
#pragma unroll
        for (int i = 0; i < kD / 8; ++i) {
            *reinterpret_cast<float2*>(a.partial + row * kD + i * 8 + 2 * q) =
                make_float2(o[i][2 * h2], o[i][2 * h2 + 1]);
        }
        if (q == 0) {
            // The combine's maxima are natural-log units; the sums are the same in either.
            a.partial[prow * kD + 2 * row]     = m[h2] == -INFINITY ? -INFINITY : m[h2] * 0.6931471805599453f;
            a.partial[prow * kD + 2 * row + 1] = l[h2];
        }
    }
}

template <int kGroup, int kMode>
void launch_attention_dtc_mode(const AttentionArgs& args, unsigned blocks, cudaStream_t stream) {
    const std::size_t smem = attention_dtc_smem();
    static const bool configured = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(attention_dtc_kernel<kGroup, kMode>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
        return true;
    }();
    (void)configured;
    (void)ninfer::pdl::launch_consumer(
        ninfer::pdl::LaunchConfig{dim3(blocks, args.kv_heads, args.splits), dim3(kDtWarps * 32), smem, stream},
        attention_dtc_kernel<kGroup, kMode>, args);
}

template <int kGroup>
void launch_attention_dtc(const AttentionArgs& args, unsigned blocks, cudaStream_t stream) {
    switch (kv_mode(args)) {
    case kKvF16: return launch_attention_dtc_mode<kGroup, kKvF16>(args, blocks, stream);
    case kKvMxfp8: return launch_attention_dtc_mode<kGroup, kKvMxfp8>(args, blocks, stream);
    default: return launch_attention_dtc_mode<kGroup, kKvAny>(args, blocks, stream);
    }
}

// NINFER_INKLING_DTC_ATTENTION=0 keeps span-split calls on the FP32 attention kernels.
bool dtc_attention_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_INKLING_DTC_ATTENTION");
        return v == nullptr || std::string(v) != "0";
    }();
    return on;
}

template <int kGroup, int kMode>
void launch_attention_tc_mode(const AttentionArgs& args, cudaStream_t stream) {
    const std::size_t smem = attention_tc_smem(kGroup);
    static const bool configured = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(attention_tc_kernel<kGroup, kMode>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
        return true;
    }();
    (void)configured;
    const int rows_a_block = kTcRows * kTcGroups;
    (void)ninfer::pdl::launch_consumer(
        ninfer::pdl::LaunchConfig{dim3((args.rows.tokens + rows_a_block - 1) / rows_a_block, args.kv_heads, 1),
                                  dim3(kGroup * kTcGroups * 32), smem, stream},
        attention_tc_kernel<kGroup, kMode>, args);
}

template <int kGroup>
void launch_attention_tc(const AttentionArgs& args, cudaStream_t stream) {
    switch (kv_mode(args)) {
    case kKvF16: return launch_attention_tc_mode<kGroup, kKvF16>(args, stream);
    case kKvMxfp8: return launch_attention_tc_mode<kGroup, kKvMxfp8>(args, stream);
    default: return launch_attention_tc_mode<kGroup, kKvAny>(args, stream);
    }
}

// NINFER_INKLING_TC_ATTENTION=0 keeps prompt chunks on attention_kernel<8> (FP32 sums).
bool tc_attention_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_INKLING_TC_ATTENTION");
        return v == nullptr || std::string(v) != "0";
    }();
    return on;
}

// Block (head, row): merges the splits of one query head and its sink, thread d the output's
// dimension d. Every split counts with weight exp(m_s - M), the sink as a key without a value:
// M = max(max_s m_s, sink), l = sum_s l_s exp(m_s - M) + exp(sink - M). Thread d also holds the
// splits d + 128 j, whose weights it sums before the block's sum.
constexpr int kCombineThreads = kD;
constexpr int kMaxSplits      = 4 * kCombineThreads;

__global__ void attention_combine_kernel(AttentionArgs a) {
    ninfer::pdl::enter();
    const int head = blockIdx.x, t = blockIdx.y;
    const int d    = threadIdx.x;
    const std::int64_t rows  = std::int64_t(a.rows.tokens) * a.query_heads * a.splits;
    const std::int64_t first = (std::int64_t(t) * a.query_heads + head) * a.splits;
    int lo = 0, hi = 0;
    key_range(a, t, t + 1, lo, hi);
    const int active = a.span > 0 ? hi / a.span - lo / a.span + 1
                                                : active_splits(hi + 1 - lo, a.splits);
    __shared__ float weight[kMaxSplits];
    __shared__ float max_s, inv_l;
    float ms[kMaxSplits / kCombineThreads];
    float m_local = -INFINITY;
#pragma unroll
    for (int j = 0; j < kMaxSplits / kCombineThreads; ++j) {
        const int sp = d + j * kCombineThreads;
        ms[j]   = sp < active ? a.partial[rows * kD + 2 * (first + sp)] : -INFINITY;
        m_local = fmaxf(m_local, ms[j]);
    }
    const float m = warp_max(m_local);
    if ((threadIdx.x & 31) == 0) { weight[threadIdx.x >> 5] = m; }
    __syncthreads();
    const float sink =
        a.sinks != nullptr ? a.sinks[std::int64_t(t) * a.sink_stride + head] : -INFINITY;
    if (threadIdx.x == 0) {
        float all = sink;
        for (int w = 0; w < int(blockDim.x >> 5); ++w) { all = fmaxf(all, weight[w]); }
        max_s = all;
    }
    __syncthreads();
    float l_s = 0.0f;
    float w_s[kMaxSplits / kCombineThreads];
#pragma unroll
    for (int j = 0; j < kMaxSplits / kCombineThreads; ++j) {
        const int sp = d + j * kCombineThreads;
        w_s[j]       = ms[j] != -INFINITY ? expf(ms[j] - max_s) : 0.0f;
        l_s += w_s[j] != 0.0f ? a.partial[rows * kD + 2 * (first + sp) + 1] * w_s[j] : 0.0f;
    }
    const float l = block_sum(l_s);
#pragma unroll
    for (int j = 0; j < kMaxSplits / kCombineThreads; ++j) { weight[d + j * kCombineThreads] = w_s[j]; }
    if (threadIdx.x == 0) { inv_l = 1.0f / (l + expf(sink - max_s)); }
    __syncthreads();
    float o = 0.0f;
    for (int s = 0; s < active; ++s) {
        const float w = weight[s];
        if (w != 0.0f) { o += a.partial[(first + s) * kD + d] * w; }
    }
    const float g = a.gate != nullptr
                        ? sigmoidf(a.gate[std::int64_t(t) * a.gate_stride + head * kD + d])
                        : 1.0f;
    const float y = o * inv_l * g;
    const int width = a.query_heads * kD, at = head * kD + d;
    if (a.out_q.qs == nullptr) {
        a.out[std::int64_t(t) * width + at] = __float2bfloat16(y);
        return;
    }
    // One warp holds 32 consecutive values: one q8_1 block of the output projection's input.
    float amax = fabsf(y), sum = y;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
    }
    const float dq = amax / 127.0f;
    a.out_q.qs[std::int64_t(t) * width + at] =
        amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(y / dq));
    if ((threadIdx.x & 31) == 0) {
        a.out_q.ds[std::int64_t(t) * (width / 32) + at / 32] =
            make_half2(__float2half(dq), __float2half(sum));
    }
}

// The far field's softmax states (far_end): block (group of kFarRows rows, KV head) of two warps
// per query head of the KV head, warp w head w / 2 over rows 16 (w % 2) .. + 16 of the group. Q
// (scaled) and the probabilities enter the MMAs as F16, K and V as the attention reads them (F16,
// or a plane's chunks decoded to F16); scores, softmax and sums stay FP32. Writes (acc, m, l) of
// each row and head to a.far in attention_workspace_floats(T, query_heads, 1)'s layout.
constexpr int kFarPad = kD + 8; // halves per shared row: 272 bytes keep the ldmatrix rows apart

__device__ __forceinline__ unsigned half2_bits(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const unsigned*>(&h);
}

template <int Group>
__global__ void __launch_bounds__(64 * Group) far_attention_kernel(AttentionArgs a) {
    constexpr int kThreads = 64 * Group;
    constexpr int kWords   = 2 * kFarKeys * (kD / 8); // uint4 words of a K and V tile
    constexpr int kPer     = (kWords + kThreads - 1) / kThreads;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tid4 = lane & 3;
    const int kv_head = blockIdx.y;
    const int g       = blockIdx.x * kFarRows;
    const int end     = far_end(a, g);
    if (end == 0) { return; }
    const int head = kv_head * Group + warp / 2;
    const int r0   = g + 16 * (warp & 1);
    const int sequence = row_sequence(a.rows, g);
    const std::int64_t base = std::int64_t(sequence) * a.seq_stride +
                              std::int64_t(kv_head) * a.head_stride;

    __shared__ __align__(16) __half k_s[2][kFarKeys][kFarPad];
    __shared__ __align__(16) __half v_s[2][kFarKeys][kFarPad];

    uint4 words[kPer];
    const auto fetch = [&](int tile) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int w = int(threadIdx.x) + i * kThreads;
            words[i]    = make_uint4(0, 0, 0, 0);
            if (kWords % kThreads != 0 && w >= kWords) { continue; }
            const int matrix = w / (kFarKeys * (kD / 8));
            const int rest   = w % (kFarKeys * (kD / 8));
            const int r = rest / (kD / 8), c = rest % (kD / 8);
            const int p    = tile * kFarKeys + r;
            const int slot = a.ring != 0 ? p % a.ring : p;
            const KvPlaneView& plane = matrix == 0 ? a.k_plane : a.v_plane;
            if (plane.coding != KvCoding::F16) {
                const std::int64_t codes_row = std::int64_t(sequence) * plane.code_seq +
                                               std::int64_t(kv_head) * plane.code_head +
                                               std::int64_t(slot) * plane.code_row;
                const std::int64_t scales_row = std::int64_t(sequence) * plane.scale_seq +
                                                std::int64_t(kv_head) * plane.scale_head +
                                                std::int64_t(slot) * plane.scale_row;
                words[i] = chunk_word(plane, plane.codes + codes_row, plane.scales + scales_row, c);
            } else {
                const __half* cache = matrix == 0 ? a.k_cache : a.v_cache;
                words[i] = *reinterpret_cast<const uint4*>(cache + base +
                                                            std::int64_t(slot) * kD + c * 8);
            }
        }
    };
    const auto put = [&](int buffer) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int w = int(threadIdx.x) + i * kThreads;
            if (kWords % kThreads != 0 && w >= kWords) { continue; }
            const int matrix = w / (kFarKeys * (kD / 8));
            const int rest   = w % (kFarKeys * (kD / 8));
            const int r = rest / (kD / 8), c = rest % (kD / 8);
            __half* to = matrix == 0 ? &k_s[buffer][r][c * 8] : &v_s[buffer][r][c * 8];
            *reinterpret_cast<uint4*>(to) = words[i];
        }
    };

    // Q's A fragments: rows r0 + gid (+ 8), dimensions 16 ks + 2 tid4 (+ 8).
    unsigned qf[kD / 16][4];
#pragma unroll
    for (int ks = 0; ks < kD / 16; ++ks) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r0 + gid + (i & 1) * 8;
            const int dim = 16 * ks + 2 * tid4 + (i >> 1) * 8;
            float2 q      = make_float2(0.0f, 0.0f);
            if (row < a.rows.tokens) {
                q = *reinterpret_cast<const float2*>(
                    a.q + (std::int64_t(row) * a.query_heads + head) * kD + dim);
            }
            qf[ks][i] = half2_bits(q.x * a.scale, q.y * a.scale);
        }
    }
    float o[kD / 8][4];
#pragma unroll
    for (int n = 0; n < kD / 8; ++n) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f; }
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    const int tiles = end / kFarKeys;
    fetch(0);
    put(0);
    __syncthreads();
    for (int tile = 0; tile < tiles; ++tile) {
        const int buffer = tile & 1;
        if (tile + 1 < tiles) { fetch(tile + 1); }
        // S = Q K^T over the tile's 32 keys: n-tile j holds keys 8 j .. 8 j + 7.
        float sc[kFarKeys / 8][4];
#pragma unroll
        for (int j = 0; j < kFarKeys / 8; ++j) { sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f; }
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks) {
#pragma unroll
            for (int jp = 0; jp < kFarKeys / 16; ++jp) {
                const int key = 8 * (2 * jp + (lane >> 4)) + (lane & 7);
                const int dim = 16 * ks + ((lane >> 3) & 1) * 8;
                unsigned b0, b1, b2, b3;
                ops::ldmatrix_x4(b0, b1, b2, b3, ops::smem_addr(&k_s[buffer][key][dim]));
                ops::mma_f16(sc[2 * jp][0], sc[2 * jp][1], sc[2 * jp][2], sc[2 * jp][3], qf[ks][0],
                             qf[ks][1], qf[ks][2], qf[ks][3], b0, b1);
                ops::mma_f16(sc[2 * jp + 1][0], sc[2 * jp + 1][1], sc[2 * jp + 1][2],
                             sc[2 * jp + 1][3], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b2, b3);
            }
        }
        // Online softmax of rows gid (elements 0, 1) and gid + 8 (elements 2, 3), over the quad.
        float top[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < kFarKeys / 8; ++j) {
            top[0] = fmaxf(top[0], fmaxf(sc[j][0], sc[j][1]));
            top[1] = fmaxf(top[1], fmaxf(sc[j][2], sc[j][3]));
        }
        float corr[2], sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            top[i] = fmaxf(top[i], __shfl_xor_sync(0xFFFFFFFFu, top[i], 1));
            top[i] = fmaxf(top[i], __shfl_xor_sync(0xFFFFFFFFu, top[i], 2));
            const float m_new = fmaxf(m[i], top[i]);
            corr[i]           = expf(m[i] - m_new);
            m[i]              = m_new;
        }
#pragma unroll
        for (int j = 0; j < kFarKeys / 8; ++j) {
            sc[j][0] = expf(sc[j][0] - m[0]);
            sc[j][1] = expf(sc[j][1] - m[0]);
            sc[j][2] = expf(sc[j][2] - m[1]);
            sc[j][3] = expf(sc[j][3] - m[1]);
            sum[0] += sc[j][0] + sc[j][1];
            sum[1] += sc[j][2] + sc[j][3];
        }
        l[0] = l[0] * corr[0] + sum[0];
        l[1] = l[1] * corr[1] + sum[1];
#pragma unroll
        for (int n = 0; n < kD / 8; ++n) {
            o[n][0] *= corr[0];
            o[n][1] *= corr[0];
            o[n][2] *= corr[1];
            o[n][3] *= corr[1];
        }
        // O += P V: k16 step s over keys 16 s .. 16 s + 15 (n-tiles 2 s and 2 s + 1 of S).
#pragma unroll
        for (int s16 = 0; s16 < kFarKeys / 16; ++s16) {
            const unsigned p0 = half2_bits(sc[2 * s16][0], sc[2 * s16][1]);
            const unsigned p1 = half2_bits(sc[2 * s16][2], sc[2 * s16][3]);
            const unsigned p2 = half2_bits(sc[2 * s16 + 1][0], sc[2 * s16 + 1][1]);
            const unsigned p3 = half2_bits(sc[2 * s16 + 1][2], sc[2 * s16 + 1][3]);
#pragma unroll
            for (int n2 = 0; n2 < kD / 16; ++n2) {
                const int key = 16 * s16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                const int dim = 8 * (2 * n2 + (lane >> 4));
                unsigned v0, v1, v2, v3;
                ops::ldmatrix_x4_t(v0, v1, v2, v3, ops::smem_addr(&v_s[buffer][key][dim]));
                ops::mma_f16(o[2 * n2][0], o[2 * n2][1], o[2 * n2][2], o[2 * n2][3], p0, p1, p2, p3,
                             v0, v1);
                ops::mma_f16(o[2 * n2 + 1][0], o[2 * n2 + 1][1], o[2 * n2 + 1][2],
                             o[2 * n2 + 1][3], p0, p1, p2, p3, v2, v3);
            }
        }
        if (tile + 1 < tiles) { put(buffer ^ 1); }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xFFFFFFFFu, l[i], 1);
        l[i] += __shfl_xor_sync(0xFFFFFFFFu, l[i], 2);
    }
    const std::int64_t n = std::int64_t(a.rows.tokens) * a.query_heads;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int row = r0 + gid + 8 * i;
        if (row >= a.rows.tokens) { continue; }
        const std::int64_t at = std::int64_t(row) * a.query_heads + head;
#pragma unroll
        for (int nt = 0; nt < kD / 8; ++nt) {
            *reinterpret_cast<float2*>(a.far + at * kD + 8 * nt + 2 * tid4) =
                make_float2(o[nt][2 * i], o[nt][2 * i + 1]);
        }
        if (tid4 == 0) {
            a.far[n * kD + 2 * at]     = m[i];
            a.far[n * kD + 2 * at + 1] = l[i];
        }
    }
}

// ---- Residual updates ---------------------------------------------------------------------------

// The four values of the product at row s, channels 4i..4i+3.
__device__ __forceinline__ float4 product4(const ResidualArgs& a, int s, int i) {
    const std::int64_t row = std::int64_t(s) * a.hidden;
    if (a.fixed != nullptr) { return fixed4(a.fixed + row, i); }
    if (a.moe_rows != nullptr) {
        float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int k = 0; k < a.top_k; ++k) {
            const std::int64_t pair = std::int64_t(s) * a.top_k + k;
            y = fma4(ld4(a.moe_rows + pair * a.hidden, i), a.moe_weights[pair], y);
        }
        return y;
    }
    return ld4(a.y + row, i);
}

// The residual update of row t's quad i (zeros outside the row): the new current state.
__device__ __forceinline__ float4 residual_quad(const ResidualArgs& a, int t, int i) {
    float4 cur = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i < (a.hidden >> 2)) {
        const int sequence = row_sequence(a.rows, t);
        float4 in[kTaps];
#pragma unroll
        for (int lag = 0; lag < kTaps; ++lag) {
            int position     = 0;
            const int source = history_source(a.rows, t, lag, position);
            in[lag] = source >= 0   ? product4(a, source, i)
                      : source == -1 ? ld4(history_slot(a.history, sequence, position), i)
                                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        float4 c = in[0];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 w = ld4(a.conv + (4 * i + j) * kTaps, 0);
            float v = get(c, j);
            v += get(w, 0) * get(in[3], j) + get(w, 1) * get(in[2], j) + get(w, 2) * get(in[1], j) +
                 get(w, 3) * get(in[0], j);
            if (j == 0) { c.x = v; } else if (j == 1) { c.y = v; } else if (j == 2) { c.z = v; } else { c.w = v; }
        }
        c = scale4(c, rsqrtf(float(kTaps + 1)));
        const std::int64_t row = std::int64_t(t) * a.hidden;
        cur = a.start ? c : add4(ld4(a.cur + row, i), c);
        reinterpret_cast<float4*>(a.cur + row)[i] = cur;
        if (a.keep) {
            reinterpret_cast<float4*>(history_slot(a.history, sequence, a.rows.positions[t]))[i] =
                in[0];
        }
    }
    return cur;
}

__global__ void __launch_bounds__(kRowBlock) residual_kernel(ResidualArgs a) {
    ninfer::pdl::enter();
    const int t = blockIdx.x;
    const float4 cur = residual_quad(a, t, int(threadIdx.x));
    if (a.mix.query_count > 0) { mix_row(a.mix, t, true, cur); }
}

// Block (w, t) of a split call: the update of warp w's quads of row t and their sums of its mix,
// whose other states and queries load before the update.
__global__ void __launch_bounds__(32) residual_split_kernel(ResidualArgs a) {
    ninfer::pdl::enter();
    const int w = int(blockIdx.x), t = int(blockIdx.y), i = 32 * w + int(threadIdx.x);
    const MixArgs& m = a.mix;
    const bool valid = i < (a.hidden >> 2);
    float4 s[kMaxStates], qv[kMaxQueries];
    if (m.query_count > 0) {
        mix_states(m, t, i, valid, true, make_float4(0.0f, 0.0f, 0.0f, 0.0f), s);
        mix_queries(m, i, valid, qv);
    }
    const float4 cur = residual_quad(a, t, i);
    if (m.query_count == 0) { return; }
#pragma unroll
    for (int j = 0; j < kMaxStates; ++j) {
        if (j == m.count - 1) { s[j] = cur; }
    }
    float v[kMixValues];
    mix_terms(s, qv, v);
    keep_warp_sums(split_terms(m, t, w), v);
}

// Block per row: its raw inputs into its history slot.
__global__ void keep_history_kernel(const float* __restrict__ x, std::int64_t stride, int channels,
                                    const std::int32_t* __restrict__ positions, int begin,
                                    History h, int sequence) {
    ninfer::pdl::enter();
    const int t = begin + blockIdx.x;
    float* slot = history_slot(h, sequence, positions[t]);
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        slot[c] = x[std::int64_t(t) * stride + c];
    }
}

// Block per row of [begin, end): its product into its history slot.
__global__ void __launch_bounds__(kRowBlock) keep_products_kernel(ResidualArgs a, int begin) {
    ninfer::pdl::enter();
    const int t = begin + blockIdx.x, i = threadIdx.x;
    if (i >= a.hidden / 4) { return; }
    reinterpret_cast<float4*>(
        history_slot(a.history, row_sequence(a.rows, t), a.rows.positions[t]))[i] = product4(a, t, i);
}

// ---- Routing ------------------------------------------------------------------------------------

constexpr int kRouteThreads = 256; // at least the experts

// Block per row: thread e the expert. The top_k by p + bias, lower id first: expert e is chosen
// k-th when k experts precede it in that order.
__global__ void route_kernel(RouteArgs a) {
    ninfer::pdl::enter();
    const int t = blockIdx.x, e = threadIdx.x;
    __shared__ float selections[kRouteThreads];
    __shared__ int chosen[8];
    __shared__ float chosen_p[8];
    const float logit = e < a.experts ? a.logits[std::int64_t(t) * a.experts + e] : 0.0f;
    const float p     = sigmoidf(logit);
    const float selection = e < a.experts ? p + a.bias[e] : -INFINITY;
    selections[e]         = selection;
    __syncthreads();
    if (e < a.experts) {
        int rank = 0;
        for (int j = 0; j < a.experts; ++j) {
            const float other = selections[j];
            rank += other > selection || (other == selection && j < e) ? 1 : 0;
        }
        if (rank < a.top_k) {
            chosen[rank]   = e;
            chosen_p[rank] = p;
        }
    }
    __syncthreads();
    // Thread k the k-th choice, each adding the chosen p in order.
    const int k = e;
    if (k >= a.top_k) { return; }
    float sum = 0.0f;
    for (int j = 0; j < a.top_k; ++j) { sum += chosen_p[j]; }
    const int x           = chosen[k];
    const std::int64_t at = std::int64_t(t) * a.top_k + k;
    a.ids[at]             = x;
    float w               = chosen_p[k] / (sum + 1e-20f) * a.scale;
    if (a.down_scale != nullptr) { w *= a.down_scale[x]; }
    a.weights[at]    = w;
    a.pair_scale[at] = a.gate_up_scale != nullptr ? a.gate_up_scale[x] : 1.0f;
}

// One block, thread (t, e) expert e of row t (at most 32 rows of 128 experts): each row's top_k as
// route_kernel picks them, then the pairs grouped by expert, each expert's in row order. An expert
// holds at most one pair a row (32), so it fills one column tile of every width but the narrowest
// (16), where past 16 pairs it takes two, as moe_sort_routes lists them.
constexpr int kRouteTileWidths = 4;
// The block holds up to kSortedRouteBlockRows rows of threads; each thread takes rows t,
// t + kSortedRouteBlockRows, ... of up to kSortedRouteRows.
__global__ void __launch_bounds__(kSortedRouteBlockRows * kSortedRouteExperts)
    route_sorted_kernel(RouteArgs a) {
    ninfer::pdl::enter();
    constexpr int E = kSortedRouteExperts;
    const int tt = int(threadIdx.x) / E, e = int(threadIdx.x) % E;
    const int row_step = int(blockDim.x) / E;
    __shared__ float selections[kSortedRouteRows][E];
    __shared__ float probs[kSortedRouteRows][E];
    __shared__ std::int8_t pick[kSortedRouteRows][E]; // the choice's k, -1 when not chosen
    __shared__ int chosen[kSortedRouteRows][8];
    __shared__ float chosen_p[kSortedRouteRows][8];
    __shared__ int warp_total[E / 32];
    const bool expert = e < a.experts;
    for (int t = tt; t < a.tokens; t += row_step) {
        const float logit = expert ? a.logits[std::int64_t(t) * a.experts + e] : 0.0f;
        const float p     = sigmoidf(logit);
        probs[t][e]       = p;
        selections[t][e]  = expert ? p + a.bias[e] : -INFINITY;
    }
    __syncthreads();
    for (int t = tt; t < a.tokens; t += row_step) {
        const float selection = selections[t][e];
        int rank = -1;
        if (expert) {
            int r = 0;
            for (int j = 0; j < a.experts; ++j) {
                const float other = selections[t][j];
                r += other > selection || (other == selection && j < e) ? 1 : 0;
            }
            if (r < a.top_k) {
                rank           = r;
                chosen[t][r]   = e;
                chosen_p[t][r] = probs[t][e];
            }
        }
        pick[t][e] = std::int8_t(rank);
    }
    __syncthreads();
    for (int t = tt; t < a.tokens; t += row_step) {
    if (e < a.top_k) {
        float sum = 0.0f;
        for (int j = 0; j < a.top_k; ++j) { sum += chosen_p[t][j]; }
        const int x           = chosen[t][e];
        const std::int64_t at = std::int64_t(t) * a.top_k + e;
        a.ids[at]             = x;
        float w               = chosen_p[t][e] / (sum + 1e-20f) * a.scale;
        if (a.down_scale != nullptr) { w *= a.down_scale[x]; }
        a.weights[at]    = w;
        a.pair_scale[at] = a.gate_up_scale != nullptr ? a.gate_up_scale[x] : 1.0f;
    }
    }
    // Row 0's threads, expert e each: its pairs, then exclusive scans of the pairs (low half) and
    // of the experts with any (high half) over the experts.
    int count = 0;
    if (tt == 0) {
        for (int r = 0; r < a.tokens; ++r) { count += pick[r][e] >= 0 ? 1 : 0; }
    }
    // Packed for one scan: pairs (bits 0-9, at most 256), experts with any (bits 10-19) and their
    // 16-wide tiles (bits 20-29).
    const int narrow = (count + 15) / 16;
    const int packed = count | (count > 0 ? 1 << 10 : 0) | narrow << 20;
    int inclusive    = packed;
    const int lane = e & 31, warp = e >> 5;
    if (tt == 0) {
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xFFFFFFFFu, inclusive, o);
            if (lane >= o) { inclusive += v; }
        }
        if (lane == 31) { warp_total[warp] = inclusive; }
    }
    __syncthreads();
    if (tt != 0) { return; }
    int offset = 0, total = 0;
#pragma unroll
    for (int w = 0; w < E / 32; ++w) {
        offset += w < warp ? warp_total[w] : 0;
        total += warp_total[w];
    }
    const int before = inclusive - packed + offset;
    const int bound = before & 0x3FF, listed = (before >> 10) & 0x3FF, narrow_at = before >> 20;
    const RouteSort& s = a.sorted;
    if (expert) { s.bounds[e] = bound; }
    if (e == 0) {
        s.bounds[a.experts] = total & 0x3FF;
        *s.active_count     = (total >> 10) & 0x3FF;
        s.tile_count[0]     = total >> 20;
#pragma unroll
        for (int w = 1; w < kRouteTileWidths; ++w) { s.tile_count[w] = (total >> 10) & 0x3FF; }
    }
    if (count == 0) { return; }
    s.active[listed] = e;
    for (int tile = 0; tile < narrow; ++tile) { s.tiles[narrow_at + tile] = e | tile << 16; }
#pragma unroll
    for (int w = 1; w < kRouteTileWidths; ++w) { s.tiles[std::int64_t(w) * s.tile_capacity + listed] = e; }
    int n = 0;
    for (int r = 0; r < a.tokens; ++r) {
        if (pick[r][e] >= 0) { s.sorted[bound + n++] = r * a.top_k + pick[r][e]; }
    }
}

// ---- SiTU ---------------------------------------------------------------------------------------

constexpr int kSituThreads = 128;

// Block (pair, part): the pair's middles [128 part, 128 part + 128).
__global__ void situ_rows_kernel(const float* __restrict__ gu, std::int64_t stride, int width,
                                 const float* __restrict__ pair_scale, float gate_cap, float up_cap,
                                 __nv_bfloat16* __restrict__ out) {
    ninfer::pdl::enter();
    const std::int64_t p = blockIdx.x;
    const int r          = int(blockIdx.y) * kSituThreads + int(threadIdx.x);
    if (r >= width) { return; }
    const float s    = pair_scale[p];
    const float* row = gu + p * stride;
    const float g    = ops::silu_detail::mul(row[r], s);
    const float u    = ops::silu_detail::mul(row[width + r], s);
    out[p * width + r] = __float2bfloat16(situ(g, u, gate_cap, up_cap));
}

// ---- BF16 experts --------------------------------------------------------------------------------

constexpr int kBfWarps = 4, kBfRowsPerWarp = 2, kBfPairs = 32, kBfChunk = 256;
constexpr int kBfRows = kBfWarps * kBfRowsPerWarp;

__device__ __forceinline__ void bf16x8(const uint4& v, float (&f)[8]) {
    const auto* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 x = __bfloat1622float2(h[i]);
        f[2 * i]       = x.x;
        f[2 * i + 1]   = x.y;
    }
}

// Block (row tile, active slot a): expert e = active[a]'s rows [8 tile, 8 tile + 8), each warp
// two of them, against the expert's pairs 32 at a time. The pairs' inputs pass through shared
// memory 256 columns at a time; lane l multiplies columns [8 l, 8 l + 8) of each, in order, into
// one FP32 sum per (row, pair), and the warp adds its lanes' sums.
__global__ void __launch_bounds__(kBfWarps * 32)
    bf16_expert_rows_kernel(const void* const* __restrict__ table, int rows, int k,
                            const std::int32_t* __restrict__ bounds,
                            const std::int32_t* __restrict__ sorted,
                            const std::int32_t* __restrict__ active,
                            const std::int32_t* __restrict__ active_count,
                            const __nv_bfloat16* __restrict__ x, int x_div,
                            float* __restrict__ out) {
    if (int(blockIdx.y) >= *active_count) { return; }
    const int e     = active[blockIdx.y];
    const int begin = bounds[e], end = bounds[e + 1];
    const auto* w   = static_cast<const __nv_bfloat16*>(table[e]);
    const int warp = int(threadIdx.x) >> 5, lane = int(threadIdx.x) & 31;
    const int row0 = int(blockIdx.x) * kBfRows + warp * kBfRowsPerWarp;
    __shared__ __align__(16) __nv_bfloat16 xs[kBfPairs][kBfChunk];
    for (int g = begin; g < end; g += kBfPairs) {
        const int n = min(kBfPairs, end - g);
        float acc[kBfRowsPerWarp][kBfPairs];
#pragma unroll
        for (int r = 0; r < kBfRowsPerWarp; ++r) {
#pragma unroll
            for (int p = 0; p < kBfPairs; ++p) { acc[r][p] = 0.0f; }
        }
        for (int c = 0; c < k; c += kBfChunk) {
            const int vectors = min(kBfChunk, k - c) / 8;
            __syncthreads();
            for (int i = int(threadIdx.x); i < n * vectors; i += int(blockDim.x)) {
                const int p = i / vectors, v = i % vectors;
                const std::int64_t src = std::int64_t(sorted[g + p] / x_div) * k + c + 8 * v;
                reinterpret_cast<uint4*>(xs[p])[v] = *reinterpret_cast<const uint4*>(x + src);
            }
            __syncthreads();
            if (lane >= vectors) { continue; }
            float wf[kBfRowsPerWarp][8];
#pragma unroll
            for (int r = 0; r < kBfRowsPerWarp; ++r) {
                const int row = min(row0 + r, rows - 1); // a row past the end is never written
                bf16x8(*reinterpret_cast<const uint4*>(w + std::int64_t(row) * k + c + 8 * lane),
                       wf[r]);
            }
#pragma unroll
            for (int p = 0; p < kBfPairs; ++p) {
                if (p >= n) { break; }
                float xf[8];
                bf16x8(reinterpret_cast<const uint4*>(xs[p])[lane], xf);
#pragma unroll
                for (int r = 0; r < kBfRowsPerWarp; ++r) {
                    float sum = acc[r][p];
#pragma unroll
                    for (int j = 0; j < 8; ++j) { sum = fmaf(wf[r][j], xf[j], sum); }
                    acc[r][p] = sum;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kBfRowsPerWarp; ++r) {
            const int row = row0 + r;
#pragma unroll
            for (int p = 0; p < kBfPairs; ++p) {
                if (p >= n) { break; }
                const float v = warp_sum(acc[r][p]);
                if (lane == 0 && row < rows) { out[std::int64_t(sorted[g + p]) * rows + row] = v; }
            }
        }
    }
}

// Block (row, part): FP32 values [256 part, 256 part + 256) of table row ids[row].
__global__ void bf16_rows_f32_kernel(const __nv_bfloat16* __restrict__ table, int k,
                                     const std::int32_t* __restrict__ ids, float* __restrict__ out,
                                     std::int64_t out_stride) {
    const int i = int(blockIdx.y) * blockDim.x + int(threadIdx.x);
    if (i >= k) { return; }
    const std::int64_t r = blockIdx.x;
    out[r * out_stride + i] = __bfloat162float(table[std::int64_t(ids[r]) * k + i]);
}

__global__ void softcap_kernel(const float* in, __nv_bfloat16* out, std::int64_t n, float cap) {
    ninfer::pdl::enter();
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __float2bfloat16(cap * tanhf(in[i] / cap)); }
}

// Block (part, row): every 256th logit of the row from the part's first.
__global__ void softcap_rows_kernel(const float* in, std::int64_t in_stride, __nv_bfloat16* out,
                                    std::int64_t out_stride, int width, float cap) {
    ninfer::pdl::enter();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) { return; }
    const std::int64_t r = blockIdx.y;
    out[r * out_stride + i] = __float2bfloat16(cap * tanhf(in[r * in_stride + i] / cap));
}

__global__ void f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, std::int64_t n) {
    ninfer::pdl::enter();
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __float2bfloat16(in[i]); }
}

__global__ void place_rows_kernel(const __nv_bfloat16* __restrict__ rows, std::int64_t n,
                                  float* __restrict__ h) {
    ninfer::pdl::enter();
    const std::int64_t k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n) { h[k] = __bfloat162float(rows[k]); }
}

// Block (position, head): the position's K and V rows, 16-byte words.
__global__ void ring_copy_kernel(RingCopyArgs a) {
    ninfer::pdl::enter();
    const int p = a.begin + blockIdx.x, h = blockIdx.y;
    const std::int64_t src = h * a.src_head_stride + std::int64_t(p % a.src_ring) * kD;
    const std::int64_t dst = h * a.dst_head_stride + std::int64_t(p % a.dst_ring) * kD;
    for (int w = threadIdx.x; w < kD / 8; w += blockDim.x) {
        reinterpret_cast<uint4*>(a.dst_k + dst)[w] = reinterpret_cast<const uint4*>(a.src_k + src)[w];
        reinterpret_cast<uint4*>(a.dst_v + dst)[w] = reinterpret_cast<const uint4*>(a.src_v + src)[w];
    }
}

__global__ void add_fixed_kernel(unsigned long long* sums, const unsigned long long* products,
                                 const std::int32_t* flag, std::int64_t n) {
    ninfer::pdl::enter();
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || *flag == 0) { return; }
    sums[i] += products[i];
}

// ---- The DSpark drafter -------------------------------------------------------------------------

// Block (row): every mix of the row, thread i its values 4i..4i+3.
__global__ void __launch_bounds__(kRowBlock) spark_mix_kernel(SparkMixArgs a) {
    ninfer::pdl::enter();
    const int t = a.begin + int(blockIdx.x);
    const int q = a.hidden >> 2, i = threadIdx.x;
    const bool valid = i < q;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 tap[kMaxSparkTaps];
#pragma unroll
    for (int j = 0; j < kMaxSparkTaps; ++j) {
        tap[j] = valid && j < a.tap_count ? ld4(a.taps[j] + std::int64_t(t) * a.hidden, i) : zero;
    }
    for (int m = 0; m <= a.context_count; ++m) {
        const float* f = a.fuse + std::int64_t(m) * a.tap_count;
        float top = -INFINITY;
        for (int j = 0; j < a.tap_count; ++j) { top = fmaxf(top, f[j]); }
        float total = 0.0f;
        for (int j = 0; j < a.tap_count; ++j) { total += expf(f[j] - top); }
        float4 x = zero;
#pragma unroll
        for (int j = 0; j < kMaxSparkTaps; ++j) {
            if (j < a.tap_count) { x = add4(x, scale4(tap[j], expf(f[j] - top) / total)); }
        }
        const float inv = rsqrtf(block_sum(dot4(x, x)) / float(a.hidden) + a.eps);
        const float4 y  = valid ? mul4(scale4(x, inv), ld4(a.norm[m], i)) : zero;
        if (m < a.context_count) {
            store_outputs(a.out, a.out_q, nullptr, std::int64_t(m) * a.rows.tokens * a.hidden,
                          std::int64_t(m) * a.q8_stride, a.hidden, t, i, y, valid);
        } else if (valid && t >= a.ring_begin) {
            const int sequence = row_sequence(a.rows, t);
            const int slot     = a.rows.positions[t] % a.slots;
            reinterpret_cast<float4*>(a.ring + std::int64_t(sequence) * a.seq_stride +
                                      std::int64_t(slot) * a.hidden)[i] = y;
        }
    }
}

// Block (row, head) of 128 threads, thread d the head's value d.
__global__ void spark_rotate_kernel(SparkRotateArgs a) {
    ninfer::pdl::enter();
    const int t = a.begin + int(blockIdx.x), h = blockIdx.y, d = threadIdx.x;
    float* row = a.x + std::int64_t(t) * a.stride + a.column;
    const float v   = row[h * kD + d];
    const float inv = rsqrtf(block_sum(v * v) / float(kD) + a.eps);
    __shared__ float y[kD];
    y[d] = v * inv * a.norm[d];
    __syncthreads();
    // The angle in FP64, reduced to [-pi, pi] before the FP32 sine and cosine: an FP32 product
    // would be off by up to position * 2^-23 radians (0.1 at a million positions).
    const int i = d & (kD / 2 - 1);
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    double angle = double(a.rows.positions[t]) * pow(double(a.theta), -double(2 * i) / double(kD));
    angle -= kTwoPi * rint(angle / kTwoPi);
    float sn = 0.0f, cs = 0.0f;
    sincosf(float(angle), &sn, &cs);
    const float out = d < kD / 2 ? y[d] * cs - y[d + kD / 2] * sn : y[d] * cs + y[d - kD / 2] * sn;
    if (a.k_cache == nullptr) {
        row[h * kD + d] = out;
        return;
    }
    const int sequence = row_sequence(a.rows, t);
    const int position = physical(a.rows, t);
    const int slot     = a.ring != 0 ? position % a.ring : position;
    const std::int64_t at = std::int64_t(sequence) * a.seq_stride + std::int64_t(h) * a.head_stride +
                            std::int64_t(slot) * kD + d;
    a.k_cache[at] = __float2half(out);
    a.v_cache[at] = __float2half(row[(a.heads + h) * kD + d]);
}

__global__ void add_rows_kernel(float* h, const float* y, std::int64_t y_stride, int hidden,
                                std::int64_t n) {
    ninfer::pdl::enter();
    const std::int64_t at = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= n) { return; }
    const std::int64_t t = at / hidden, c = at % hidden;
    h[at] += y[t * y_stride + c];
}

__global__ void silu_mul_kernel(const float* gu, int width, std::int64_t n, __nv_bfloat16* out) {
    ninfer::pdl::enter();
    const std::int64_t at = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= n) { return; }
    const std::int64_t t = at / width, c = at % width;
    const float g = gu[t * 2 * width + c], u = gu[t * 2 * width + width + c];
    out[at]       = __float2bfloat16(g / (1.0f + expf(-g)) * u);
}

constexpr int kSparkRank     = 256;
constexpr int kSparkPartials = 256;
// A lane's step scratch (spark_step_floats): joint, logits, partials.
__host__ __device__ constexpr std::int64_t spark_scratch(int vocab) {
    return std::int64_t(3 * kSparkRank) + vocab + 5 * kSparkPartials;
}

__global__ void spark_table_kernel(const float* p, std::int64_t n, __nv_bfloat16* m, float* c) {
    ninfer::pdl::enter();
    const std::int64_t at = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= n * (kSparkRank + 1)) { return; }
    const std::int64_t v = at / (kSparkRank + 1), r = at % (kSparkRank + 1);
    if (r < kSparkRank) {
        m[v * kSparkRank + r] = __float2bfloat16(p[at]);
    } else {
        c[v] = p[at];
    }
}

__global__ void spark_head_inputs_kernel(const float* hidden_out, const float* bias, int hidden,
                                         int rank, float* out) {
    ninfer::pdl::enter();
    const std::int64_t at = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= std::int64_t(rank + 1) * hidden) { return; }
    const std::int64_t r = at / hidden, k = at % hidden;
    out[at] = r < rank ? hidden_out[k * rank + r] : bias[k];
}

// Block of eight warps, warp w output 8 blockIdx.x + w of the joint projection; the block's input
// vector in shared memory.
// The joint product takes its lanes' inputs in shared memory (20 KiB a lane at Inkling 2's
// width), so spark_step runs it over groups of at most this many lanes.
constexpr int kSparkJointLanes = 4;

__global__ void __launch_bounds__(256) spark_joint_kernel(SparkStepArgs a) {
    ninfer::pdl::enter();
    extern __shared__ float x[];
    const int k = 2 * kSparkRank + 2 * a.hidden_size;
    const std::int64_t scratch = spark_scratch(a.vocab);
    for (int j = 0; j < a.lanes; ++j) {
        const int r0   = j * a.block;
        const int prev = a.prev[j];
        const int sequence = a.sequences[r0];
        const int slot = (a.positions[r0] - 1 + a.slots) % a.slots;
        const float* context = a.ring + std::int64_t(sequence) * a.seq_stride +
                               std::int64_t(slot) * a.hidden_size;
        const float* hidden = a.hidden + std::int64_t(r0 + a.row) * a.hidden_size;
        const float* state  = a.state + j * kSparkRank;
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            float v = 0.0f;
            if (i < kSparkRank) {
                v = state[i];
            } else if (i < 2 * kSparkRank) {
                v = __bfloat162float(a.embed[std::int64_t(prev) * kSparkRank + i - kSparkRank]);
            } else if (i < 2 * kSparkRank + a.hidden_size) {
                v = hidden[i - 2 * kSparkRank];
            } else {
                v = context[i - 2 * kSparkRank - a.hidden_size];
            }
            x[j * k + i] = v;
        }
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o    = blockIdx.x * 8 + warp;
    if (o >= 3 * kSparkRank) { return; }
    const float4* w  = reinterpret_cast<const float4*>(a.weight + std::int64_t(o) * k);
    float sum[kSparkJointLanes] = {};
    for (int i = lane; i < k / 4; i += 32) {
        const float4 wi = w[i];
#pragma unroll
        for (int j = 0; j < kSparkJointLanes; ++j) {
            if (j < a.lanes) { sum[j] += dot4(wi, reinterpret_cast<const float4*>(x + j * k)[i]); }
        }
    }
#pragma unroll
    for (int j = 0; j < kSparkJointLanes; ++j) {
        if (j >= a.lanes) { break; }
        const float total = warp_sum(sum[j]);
        if (lane == 0) { a.joint[j * scratch + o] = total + a.bias[o]; }
    }
}

// Block of eight warps, warp w vocabulary row 8 blockIdx.x + w: its soft-capped logit.
__global__ void __launch_bounds__(256) spark_logits_kernel(SparkStepArgs a) {
    ninfer::pdl::enter();
    __shared__ float latent[kSparkMaxLanes][kSparkRank];
    const std::int64_t scratch = spark_scratch(a.vocab);
    for (int j = 0; j < a.lanes; ++j) {
        for (int r = threadIdx.x; r < kSparkRank; r += blockDim.x) {
            latent[j][r] = tanhf(a.joint[j * scratch + 2 * kSparkRank + r]);
        }
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const std::int64_t v = std::int64_t(blockIdx.x) * 8 + warp;
    if (v >= a.vocab) { return; }
    const uint4 word = reinterpret_cast<const uint4*>(a.m + v * kSparkRank)[lane];
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&word);
    const float cv = a.c[v];
    for (int j = 0; j < a.lanes; ++j) {
        float dot = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float2 f = __bfloat1622float2(pairs[q]);
            dot += f.x * latent[j][8 * lane + 2 * q] + f.y * latent[j][8 * lane + 2 * q + 1];
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            const float logit =
                a.base[std::int64_t(j * a.block + a.row) * a.vocab_stride + v] + cv + dot;
            (a.joint + j * scratch + 3 * kSparkRank)[v] = a.cap * tanhf(logit * (1.0f / a.cap));
        }
    }
}

// The running (maximum, its first index, sum of e^(x - maximum)) of a set of logits.
struct Pick {
    float top = -INFINITY;
    int index = INT_MAX;
    float sum = 0.0f;
};

__device__ __forceinline__ Pick merge_pick(Pick a, Pick b) {
    if (b.top > a.top || (b.top == a.top && b.index < a.index)) {
        const Pick t = a;
        a = b;
        b = t;
    }
    if (b.top != -INFINITY) { a.sum += b.sum * expf(b.top - a.top); }
    return a;
}

__device__ Pick block_pick(Pick p) {
    __shared__ float tops[32], sums[32];
    __shared__ int indices[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        Pick other;
        other.top   = __shfl_xor_sync(0xFFFFFFFFu, p.top, o);
        other.index = __shfl_xor_sync(0xFFFFFFFFu, p.index, o);
        other.sum   = __shfl_xor_sync(0xFFFFFFFFu, p.sum, o);
        p           = merge_pick(p, other);
    }
    if (lane == 0) {
        tops[warp]    = p.top;
        indices[warp] = p.index;
        sums[warp]    = p.sum;
    }
    __syncthreads();
    Pick all;
    for (int w = 0; w < int(blockDim.x >> 5); ++w) {
        Pick other;
        other.top   = tops[w];
        other.index = indices[w];
        other.sum   = sums[w];
        all         = merge_pick(all, other);
    }
    return all;
}

// The Gumbel draw's running best: its score, token and scaled logit.
struct GumbelPick {
    float score = -INFINITY;
    int index   = INT_MAX;
    float x     = -INFINITY;
};

__device__ __forceinline__ GumbelPick merge_gumbel(GumbelPick a, GumbelPick b) {
    return b.score > a.score || (b.score == a.score && b.index < a.index) ? b : a;
}

__device__ GumbelPick block_gumbel(GumbelPick p) {
    __shared__ float scores[32], xs[32];
    __shared__ int indices[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        GumbelPick other;
        other.score = __shfl_xor_sync(0xFFFFFFFFu, p.score, o);
        other.index = __shfl_xor_sync(0xFFFFFFFFu, p.index, o);
        other.x     = __shfl_xor_sync(0xFFFFFFFFu, p.x, o);
        p           = merge_gumbel(p, other);
    }
    if (lane == 0) {
        scores[warp]  = p.score;
        indices[warp] = p.index;
        xs[warp]      = p.x;
    }
    __syncthreads();
    GumbelPick all;
    for (int w = 0; w < int(blockDim.x >> 5); ++w) {
        GumbelPick other;
        other.score = scores[w];
        other.index = indices[w];
        other.x     = xs[w];
        all         = merge_gumbel(all, other);
    }
    return all;
}

__device__ __forceinline__ float spark_inv_temperature(const SparkStepArgs& a, int j) {
    return a.inv_temperature != nullptr ? a.inv_temperature[j] : 0.0f;
}

// Lane j's sampled pass: the softmax at 1 / temperature (top and sum, for the confidence) and the
// Gumbel draw at the target's key, partials [5 * blocks] after the argmax form's [3 * blocks].
__device__ void spark_pick_partials_gumbel(const SparkStepArgs& a, int j, const float* logits,
                                           float* partials, int lo, int hi, float inv_t) {
    const unsigned long long seed = a.seeds[j];
    const int position            = a.positions[j * a.block + a.row];
    Pick p;
    GumbelPick g;
    for (int v = lo + threadIdx.x; v < hi; v += blockDim.x) {
        const float x = logits[v] * inv_t;
        if (x > p.top) {
            p.sum   = p.sum * expf(p.top - x) + 1.0f;
            p.top   = x;
            p.index = v;
        } else {
            p.sum += expf(x - p.top);
        }
        const float u = ninfer::ops::sampling_uniform_open(
            seed, position, ninfer::ops::kSamplePurposeDecode, static_cast<unsigned int>(v) + 1u);
        const float score = x - logf(-logf(u));
        if (score > g.score) {
            g.score = score;
            g.index = v;
            g.x     = x;
        }
    }
    p = block_pick(p);
    __syncthreads();
    g = block_gumbel(g);
    if (threadIdx.x == 0) {
        partials[5 * blockIdx.x]     = p.top;
        partials[5 * blockIdx.x + 1] = p.sum;
        partials[5 * blockIdx.x + 2] = g.score;
        partials[5 * blockIdx.x + 3] = __int_as_float(g.index);
        partials[5 * blockIdx.x + 4] = g.x;
    }
}

__global__ void __launch_bounds__(256) spark_pick_partials_kernel(SparkStepArgs a) {
    ninfer::pdl::enter();
    const int j = blockIdx.y;
    const std::int64_t scratch = spark_scratch(a.vocab);
    const float* logits = a.joint + j * scratch + 3 * kSparkRank;
    float* partials     = a.joint + j * scratch + 3 * kSparkRank + a.vocab;
    const int per = (a.vocab + kSparkPartials - 1) / kSparkPartials;
    const int lo = blockIdx.x * per, hi = min(a.vocab, lo + per);
    if (const float inv_t = spark_inv_temperature(a, j); inv_t > 0.0f) {
        spark_pick_partials_gumbel(a, j, logits, partials, lo, hi, inv_t);
        return;
    }
    Pick p;
    for (int v = lo + threadIdx.x; v < hi; v += blockDim.x) {
        const float x = logits[v];
        if (x > p.top) {
            p.sum   = p.sum * expf(p.top - x) + 1.0f;
            p.top   = x;
            p.index = v;
        } else {
            p.sum += expf(x - p.top);
        }
    }
    p = block_pick(p);
    if (threadIdx.x == 0) {
        partials[3 * blockIdx.x]     = p.top;
        partials[3 * blockIdx.x + 1] = __int_as_float(p.index);
        partials[3 * blockIdx.x + 2] = p.sum;
    }
}

__global__ void __launch_bounds__(kSparkPartials) spark_pick_final_kernel(SparkStepArgs a) {
    ninfer::pdl::enter();
    const int j = blockIdx.x;
    const std::int64_t scratch = spark_scratch(a.vocab);
    const float* joint    = a.joint + j * scratch;
    const float* partials = joint + 3 * kSparkRank + a.vocab;
    float* state          = a.state + j * kSparkRank;
    Pick p;
    const int b = threadIdx.x;
    if (spark_inv_temperature(a, j) > 0.0f) {
        // The sampled pass: the Gumbel winner, its probability at the temperature its confidence.
        p.top   = partials[5 * b];
        p.index = b;
        p.sum   = partials[5 * b + 1];
        GumbelPick g;
        g.score = partials[5 * b + 2];
        g.index = __float_as_int(partials[5 * b + 3]);
        g.x     = partials[5 * b + 4];
        p       = block_pick(p);
        __syncthreads();
        g = block_gumbel(g);
        if (threadIdx.x == 0) {
            a.prev[j]                             = g.index;
            a.drafts[j * a.block + a.row - 1]     = g.index;
            a.confidence[j * a.block + a.row - 1] = expf(g.x - p.top) / p.sum;
        }
    } else {
        p.top   = partials[3 * b];
        p.index = __float_as_int(partials[3 * b + 1]);
        p.sum   = partials[3 * b + 2];
        p       = block_pick(p);
        if (threadIdx.x == 0) {
            a.prev[j]                               = p.index;
            a.drafts[j * a.block + a.row - 1]       = p.index;
            a.confidence[j * a.block + a.row - 1]   = 1.0f / p.sum;
        }
    }
    for (int r = threadIdx.x; r < kSparkRank; r += blockDim.x) {
        const float gate      = sigmoidf(joint[r]);
        const float candidate = tanhf(joint[kSparkRank + r]);
        state[r]              = candidate + gate * (state[r] - candidate);
    }
}

void check_mix(const MixArgs& a) {
    if (a.count < 1 || a.count > kMaxStates || a.query_count < 0 || a.query_count > kMaxQueries) {
        throw std::invalid_argument("inkling2 mix: 1..8 states and at most 4 queries");
    }
}

} // namespace

void rms_norm(const float* x, const float* w, int hidden, int tokens, float eps,
              __nv_bfloat16* out, cudaStream_t stream, Q8Rows q, float* f32) {
    if (tokens <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(tokens), dim3(row_threads(hidden)), std::size_t(0), stream}, rms_norm_kernel, x, w, hidden, eps, out, q, f32);
    check_launch("rms_norm");
}

std::size_t mix_split_floats(int tokens, int hidden) {
    return std::size_t(tokens) * (std::size_t(row_warps(hidden)) * (kMixValues + kMaxQueries) +
                                  std::size_t(kMaxQueries) * std::size_t(hidden));
}

namespace {

bool split_mix(const MixArgs& a) { return a.split != nullptr && a.tokens <= kSplitRows; }

// The split mix's passes after its sums.
void mix_split_rows(const MixArgs& a, cudaStream_t stream) {
    const dim3 grid(unsigned(row_warps(a.hidden)), unsigned(a.tokens));
    const bool normed = a.out != nullptr || a.out_q.qs != nullptr || a.f32 != nullptr;
    if (a.raw != nullptr || normed) {
        (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(grid), dim3(32), std::size_t(0), stream}, mix_split_rows_kernel, a, false);
        check_launch("mix_split_rows");
    }
    if (normed) {
        (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(grid), dim3(32), std::size_t(0), stream}, mix_split_rows_kernel, a, true);
        check_launch("mix_split_rows");
    }
}

} // namespace

void mix(const MixArgs& args, cudaStream_t stream) {
    check_mix(args);
    row_threads(args.hidden);
    if (args.tokens <= 0 || args.query_count == 0) { return; }
    if (split_mix(args)) {
        (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(unsigned(row_warps(args.hidden)), unsigned(args.tokens))), dim3(32), std::size_t(0), stream}, mix_split_terms_kernel, args);
        check_launch("mix_split_terms");
        mix_split_rows(args, stream);
        return;
    }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(args.tokens), dim3(row_threads(args.hidden)), std::size_t(0), stream}, mix_kernel, args);
    check_launch("mix");
}

// f(std::integral_constant<int, n>) for n in 1..16.
template <class F> void with_columns(int n, F&& f) {
    switch (n) {
    case 1: return f(std::integral_constant<int, 1>{});
    case 2: return f(std::integral_constant<int, 2>{});
    case 3: return f(std::integral_constant<int, 3>{});
    case 4: return f(std::integral_constant<int, 4>{});
    case 5: return f(std::integral_constant<int, 5>{});
    case 6: return f(std::integral_constant<int, 6>{});
    case 7: return f(std::integral_constant<int, 7>{});
    case 8: return f(std::integral_constant<int, 8>{});
    case 9: return f(std::integral_constant<int, 9>{});
    case 10: return f(std::integral_constant<int, 10>{});
    case 11: return f(std::integral_constant<int, 11>{});
    case 12: return f(std::integral_constant<int, 12>{});
    case 13: return f(std::integral_constant<int, 13>{});
    case 14: return f(std::integral_constant<int, 14>{});
    case 15: return f(std::integral_constant<int, 15>{});
    default: return f(std::integral_constant<int, 16>{});
    }
}

// vector_call: a decode call (the vector path's rows), whose 9..32 columns take the GEMV forms so a
// row's sums do not depend on its batch; prompt chunks keep cuBLAS past 8 columns.
void gemm_f32(int m, int n, int k, const float* w, const float* x, float* out, cudaStream_t stream,
              std::int64_t x_stride, std::int64_t out_stride, bool vector_call) {
    if (m <= 0 || n <= 0) { return; }
    if (x_stride == 0) { x_stride = k; }
    if (out_stride == 0) { out_stride = m; }
    if (vector_call && n > 2 * kGemvRows) {
        // Past 16 columns: launches of at most 16 (the matrix is small; its later reads come from
        // L2). Each column's sums are those of a narrower call.
        // One launch: the parts on blockIdx.y run side by side.
        const int parts = (n + 2 * kGemvRows - 1) / (2 * kGemvRows);
        with_columns((n + parts - 1) / parts, [&](auto tokens) {
            (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(unsigned(m), unsigned(parts)), dim3(kGemvThreads), std::size_t(0), stream}, gemv_parts_kernel<decltype(tokens)::value>, w, x, k, x_stride, out, out_stride, n, parts);
        });
        check_launch("gemv parts");
        return;
    }
    if (vector_call && n > 2 * kGemvRows) {
        // A decode or verify call of more than 16 columns: the batch form's parts, whose columns'
        // sums are those of a narrower call (not SGEMM's).
        GemvBatch one{};
        one.count  = 1;
        one.w[0]   = w;
        one.x[0]   = x;
        one.out[0] = out;
        gemm_f32_batch(m, n, k, one, stream, x_stride, out_stride, true);
        return;
    }
    if (n <= kGemvRows || (vector_call && n <= 2 * kGemvRows)) {
        // Up to 16 columns (a decode call of 16 lanes) in one launch; each column's sums are those
        // of a narrower call.
        with_columns(n, [&](auto tokens) {
            (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(m), dim3(kGemvThreads), std::size_t(0), stream}, gemv_kernel<decltype(tokens)::value>, w, x, m, k, x_stride, out, out_stride);
        });
        check_launch("gemv");
        return;
    }
    const cublasHandle_t handle = blas_handle();
    check_blas(cublasSetStream(handle, stream), "set stream");
    const float one = 1.0f, zero = 0.0f;
    check_blas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, w, k, x, int(x_stride),
                           &zero, out, int(out_stride)),
               "SGEMM");
}

void gemm_f32_batch(int m, int n, int k, const GemvBatch& batch, cudaStream_t stream,
                    std::int64_t x_stride, std::int64_t out_stride, bool vector_call) {
    if (batch.count < 0 || batch.count > kMaxGemvBatch) {
        throw std::invalid_argument("inkling2 gemm_f32_batch: at most 8 products");
    }
    if (m <= 0 || n <= 0 || batch.count == 0) { return; }
    if (x_stride == 0) { x_stride = k; }
    if (out_stride == 0) { out_stride = m; }
    if (vector_call && n > 2 * kGemvRows) {
        // One launch: the parts on blockIdx.z run side by side.
        const int parts   = (n + 2 * kGemvRows - 1) / (2 * kGemvRows);
        const bool short_k = k <= 128 && k % 4 == 0 && x_stride % 4 == 0;
        with_columns((n + parts - 1) / parts, [&](auto tokens) {
            constexpr int N = decltype(tokens)::value;
            if (short_k) {
                (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(blocks_for(m, kShortRows), unsigned(batch.count), unsigned(parts)), dim3(kGemvThreads), std::size_t(0), stream}, gemv_short_kernel<N>, batch, m, k, x_stride, out_stride, n, parts);
            } else {
                (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(unsigned(m), unsigned(batch.count), unsigned(parts)), dim3(kGemvThreads), std::size_t(0), stream}, gemv_batch_parts_kernel<N>, batch, k, x_stride, out_stride, n, parts);
            }
        });
        check_launch("gemv_batch parts");
        return;
    }
    if (n > (vector_call ? 2 * kGemvRows : kGemvRows)) {
        for (int e = 0; e < batch.count; ++e) {
            gemm_f32(m, n, k, batch.w[e], batch.x[e], batch.out[e], stream, x_stride, out_stride);
        }
        return;
    }
    if (x_stride == 0) { x_stride = k; }
    if (out_stride == 0) { out_stride = m; }
    const bool short_k = k <= 128 && k % 4 == 0 && x_stride % 4 == 0;
    const auto launch = [&](auto tokens) {
        constexpr int N = decltype(tokens)::value;
        if (short_k) {
            (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(blocks_for(m, kShortRows), unsigned(batch.count))), dim3(kGemvThreads), std::size_t(0), stream}, gemv_short_kernel<N>, batch, m, k, x_stride, out_stride, N, 1);
        } else {
            (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(unsigned(m), unsigned(batch.count))), dim3(kGemvThreads), std::size_t(0), stream}, gemv_batch_kernel<N>, batch, k, x_stride, out_stride);
        }
    };
    with_columns(n, launch);
    check_launch("gemv_batch");
}

void control_inputs(const ControlInputs& args, cudaStream_t stream) {
    if (args.count <= 0 || args.tokens <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(args.tokens, args.count)), dim3(row_threads(args.hidden)), std::size_t(0), stream}, control_inputs_kernel, args);
    check_launch("control_inputs");
}

void control_apply(const ControlApply& args, cudaStream_t stream) {
    if (args.count <= 0 || args.tokens <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(args.tokens, args.count)), dim3(row_threads(args.hidden)), std::size_t(0), stream}, control_apply_kernel, args);
    check_launch("control_apply");
}

void query_post(const QueryArgs& args, cudaStream_t stream) {
    if (args.tokens <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(args.tokens, args.heads)), dim3(kD), std::size_t(0), stream}, query_post_kernel, args);
    check_launch("query_post");
}

void qkv_post(const QueryArgs& q, const KvArgs& kv, cudaStream_t stream) {
    if (q.tokens != kv.rows.tokens) { throw std::invalid_argument("inkling2 qkv_post: tokens"); }
    (void)ninfer::pdl::launch_consumer(
        ninfer::pdl::LaunchConfig{dim3(q.tokens, q.heads + kv.kv_heads), dim3(kD), std::size_t(0),
                                  stream},
        qkv_post_kernel, q, kv);
    check_launch("qkv_post");
}

void kv_post(const KvArgs& args, cudaStream_t stream) {
    if (args.rows.tokens <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(args.rows.tokens, args.kv_heads)), dim3(kD), std::size_t(0), stream}, kv_post_kernel, args);
    check_launch("kv_post");
}

std::size_t attention_workspace_floats(int rows, int query_heads, int splits) {
    const std::size_t n = std::size_t(rows) * query_heads * splits;
    return n * kD + 2 * n;
}

bool tc_prompt_attention() { return tc_attention_enabled(); }

void attention(const AttentionArgs& args, cudaStream_t stream) {
    const int group = args.kv_heads > 0 ? args.query_heads / args.kv_heads : 0;
    if (group < 1 || group > kMaxGroup || group * args.kv_heads != args.query_heads ||
        args.splits < 1 || args.splits > kMaxSplits ||
        (args.out == nullptr && args.out_q.qs == nullptr) ||
        (args.r == nullptr && args.extent > 0)) {
        throw std::invalid_argument("inkling2 attention: 1..4 query heads per key head, 1..512 "
                                    "splits and the relative features of a biased call");
    }
    const int t = args.rows.tokens;
    if (t <= 0) { return; }
    if (args.uniform) {
        if (args.rows.chunk || (t > 8 && t % 8 != 0) || (args.splits > 1 && args.partial == nullptr)) {
            throw std::invalid_argument("inkling2 attention: uniform blocks of 1..8 decode rows, or "
                                        "of eight rows each");
        }
        if (args.span > 0 && args.window == 0 && args.ring == 0 && group == 3 &&
            args.span % kDtKeys == 0 && args.extent == 0 && dtc_attention_enabled()) {
            // A full layer's DSpark blocks on the tensor cores, split at fixed span positions.
            launch_attention_dtc<3>(args, unsigned((t + 7) / 8), stream);
        } else {
            AttentionArgs fp32 = args;
            fp32.span = 0;
            (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3((t + 7) / 8, args.kv_heads, args.splits)), dim3(256), std::size_t(0), stream}, attention_kernel<8>, fp32);
            check_launch("attention");
            if (args.splits > 1) {
                (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(args.query_heads, t, 1)), dim3(kCombineThreads), std::size_t(0), stream}, attention_combine_kernel, fp32);
                check_launch("attention_combine");
            }
            return;
        }
        check_launch("attention");
        (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(args.query_heads, t, 1)), dim3(kCombineThreads), std::size_t(0), stream}, attention_combine_kernel, args);
        check_launch("attention_combine");
        return;
    }
    if (args.rows.chunk) {
        if (args.splits != 1) { throw std::invalid_argument("inkling2 attention: chunk splits"); }
        if (args.rows.starts == nullptr && args.far == nullptr && tc_attention_enabled()) {
            // Prompt chunks on the tensor cores (any plane coding through kv128::chunk_word).
            if (group == 1) { launch_attention_tc<1>(args, stream); }
            else if (group == 2) { launch_attention_tc<2>(args, stream); }
            else if (group == 3) { launch_attention_tc<3>(args, stream); }
            else { launch_attention_tc<4>(args, stream); }
            check_launch("attention_tc");
            return;
        }
        if (args.far != nullptr) {
            if (args.window != 0 || args.ring != 0) {
                throw std::invalid_argument("inkling2 attention: a far field is a full visit's");
            }
            const dim3 grid(blocks_for(t, kFarRows), unsigned(args.kv_heads));
            switch (group) {
            case 1: far_attention_kernel<1><<<grid, 64, 0, stream>>>(args); break;
            case 2: far_attention_kernel<2><<<grid, 128, 0, stream>>>(args); break;
            case 3: far_attention_kernel<3><<<grid, 192, 0, stream>>>(args); break;
            default: far_attention_kernel<4><<<grid, 256, 0, stream>>>(args); break;
            }
            check_launch("far_attention");
        }
        if (args.far != nullptr) {
            attention_kernel<8, true><<<dim3(blocks_for(t, 8), args.kv_heads, 1), 256, 0, stream>>>(
                args);
        } else {
            attention_kernel<8><<<dim3(blocks_for(t, 8), args.kv_heads, 1), 256, 0, stream>>>(args);
        }
        check_launch("attention");
        return;
    }
    if (args.span > 0) {
        // Span splits (fixed key positions a row's splits begin at) on the tensor cores, a
        // verification's runs of one sequence a block each.
        if (args.partial == nullptr || args.span % kDtKeys != 0 || group != 3) {
            throw std::invalid_argument("inkling2 attention: span splits need partials, 32-key "
                                        "spans and three query heads a key head");
        }
        const unsigned blocks = args.runs != nullptr
                                    ? unsigned(args.run_count > 0 ? std::min(args.run_count, t) : t)
                                    : unsigned(t);
        launch_attention_dtc<3>(args, blocks, stream);
    } else {
        if (args.row_group < 1 || args.row_group > kMaxRowGroup ||
            (args.row_group > 1 && (args.window != 0 || args.ring != 0)) || args.row_groups < 0) {
            throw std::invalid_argument("inkling2 attention: row groups of 1..4 rows, a full visit's");
        }
        const int groups = args.row_group == 1 || args.row_groups == 0 ? t : std::min(t, args.row_groups);
        attention_kernel<1><<<dim3(groups, args.kv_heads, args.splits), 128 * args.row_group, 0,
                              stream>>>(args);
    }
    check_launch("attention");
    if (args.splits > 1 || args.span > 0) {
        (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(args.query_heads, t, 1)), dim3(kCombineThreads), std::size_t(0), stream}, attention_combine_kernel, args);
        check_launch("attention_combine");
    }
}

void residual(const ResidualArgs& args, cudaStream_t stream) {
    if (args.rows.tokens <= 0) { return; }
    if (args.mix.query_count > 0) {
        check_mix(args.mix);
        if (args.mix.states[args.mix.count - 1] != args.cur) {
            throw std::invalid_argument("inkling2 residual: the mix's last state is the current one");
        }
    }
    const int threads = row_threads(args.hidden);
    // A short call given the split scratch spreads its rows over one-warp blocks.
    if (args.mix.split != nullptr && args.rows.tokens <= kSplitRows &&
        (args.mix.query_count == 0 || args.mix.tokens == args.rows.tokens)) {
        (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(unsigned(row_warps(args.hidden)), unsigned(args.rows.tokens))), dim3(32), std::size_t(0), stream}, residual_split_kernel, args);
        check_launch("residual_split");
        if (args.mix.query_count > 0) { mix_split_rows(args.mix, stream); }
        return;
    }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(args.rows.tokens), dim3(threads), std::size_t(0), stream}, residual_kernel, args);
    check_launch("residual");
}

void keep_products(const ResidualArgs& args, int begin, int end, cudaStream_t stream) {
    if (end <= begin) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(end - begin), dim3(row_threads(args.hidden)), std::size_t(0), stream}, keep_products_kernel, args, begin);
    check_launch("keep_products");
}

void keep_history(const float* x, std::int64_t stride, int channels, const std::int32_t* positions,
                  int begin, int end, const History& history, int sequence, cudaStream_t stream) {
    if (end <= begin) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(end - begin), dim3(256), std::size_t(0), stream}, keep_history_kernel, x, stride, channels, positions, begin,
                                                         history, sequence);
    check_launch("keep_history");
}

void route(const RouteArgs& args, cudaStream_t stream) {
    if (args.top_k > 8 || args.top_k > args.experts || args.experts > kRouteThreads) {
        throw std::invalid_argument("inkling2 route: at most 8 of 256 experts");
    }
    if (args.tokens <= 0) { return; }
    if (args.sorted.bounds != nullptr) {
        if (args.tokens > kSortedRouteRows || args.experts > kSortedRouteExperts) {
            throw std::invalid_argument("inkling2 route: sorted pairs of at most 32 rows of 128 experts");
        }
        (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(1), dim3(std::min(args.tokens, kSortedRouteBlockRows) * kSortedRouteExperts), std::size_t(0), stream}, route_sorted_kernel, args);
        check_launch("route_sorted");
        return;
    }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(args.tokens), dim3(kRouteThreads), std::size_t(0), stream}, route_kernel, args);
    check_launch("route");
}

namespace {

// situ_rows then gguf::quantize_vector_activation in one kernel: each value is rounded to BF16
// as situ_rows stores it, then quantized as that quantizer quantizes a BF16 column (q8_1 blocks
// of 32 with the same butterfly max and sum), so the down product reads the same bytes. The
// grid also zeroes `zero` (the fixed-point sums the down product adds into).
__global__ void situ_quantize_kernel(const float* __restrict__ gu, std::int64_t stride, int width,
                                     const float* __restrict__ pair_scale, float gate_cap,
                                     float up_cap, std::int8_t* __restrict__ qs,
                                     half2* __restrict__ ds, unsigned long long* __restrict__ zero,
                                     std::int64_t zero_count) {
    ninfer::pdl::enter();
    {
        const std::int64_t threads = std::int64_t(gridDim.x) * gridDim.y * blockDim.x;
        for (std::int64_t i = (std::int64_t(blockIdx.x) * gridDim.y + blockIdx.y) * blockDim.x +
                              threadIdx.x;
             i < zero_count; i += threads) {
            zero[i] = 0ULL;
        }
    }
    const std::int64_t p = blockIdx.x;
    const int r          = int(blockIdx.y) * kSituThreads + int(threadIdx.x);
    if (r >= width) { return; } // whole warps: width and the block are whole groups of 32
    const float s    = pair_scale[p];
    const float* row = gu + p * stride;
    const float g    = ops::silu_detail::mul(row[r], s);
    const float u    = ops::silu_detail::mul(row[width + r], s);
    const float xi   = __bfloat162float(__float2bfloat16(situ(g, u, gate_cap, up_cap)));
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset, 32);
    }
    const float d          = amax / 127.0f;
    qs[p * width + r]      = amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
    if (r % 32 == 0) { ds[p * (width / 32) + r / 32] = make_half2(d, sum); }
}

} // namespace

void situ_quantize(const float* gu, std::int64_t stride, int pairs, int width,
                   const float* pair_scale, float gate_cap, float up_cap, void* activation,
                   std::size_t scales_offset, unsigned long long* zero, std::int64_t zero_count,
                   cudaStream_t stream) {
    if (width % 32 != 0 || kSituThreads % 32 != 0) {
        throw std::invalid_argument("inkling2 situ_quantize: whole 32-value groups");
    }
    if (pairs <= 0) { return; }
    auto* base = static_cast<std::uint8_t*>(activation);
    (void)ninfer::pdl::launch_consumer(
        ninfer::pdl::LaunchConfig{dim3(pairs, blocks_for(width, kSituThreads)), dim3(kSituThreads),
                                  std::size_t(0), stream},
        situ_quantize_kernel, gu, stride, width, pair_scale, gate_cap, up_cap,
        reinterpret_cast<std::int8_t*>(base), reinterpret_cast<half2*>(base + scales_offset), zero,
        zero_count);
    check_launch("situ_quantize");
}

void situ_rows(const float* gu, std::int64_t stride, int pairs, int width, const float* pair_scale,
               float gate_cap, float up_cap, __nv_bfloat16* out, cudaStream_t stream) {
    if (width % 32 != 0) { throw std::invalid_argument("inkling2 situ_rows: whole 32-value groups"); }
    if (pairs <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(pairs, blocks_for(width, kSituThreads))), dim3(kSituThreads), std::size_t(0), stream}, situ_rows_kernel, 
        gu, stride, width, pair_scale, gate_cap, up_cap, out);
    check_launch("situ_rows");
}

void bf16_expert_rows(const void* const* table, int rows, int k, const std::int32_t* bounds,
                      const std::int32_t* sorted, const std::int32_t* active,
                      const std::int32_t* active_count, int max_active, const __nv_bfloat16* x,
                      int x_div, float* out, cudaStream_t stream) {
    if (rows <= 0 || k <= 0 || k % 8 != 0 || x_div <= 0) {
        throw std::invalid_argument("inkling2 bf16_expert_rows: rows, k a multiple of 8, x_div");
    }
    if (max_active <= 0) { return; }
    bf16_expert_rows_kernel<<<dim3(blocks_for(rows, kBfRows), unsigned(max_active)), kBfWarps * 32,
                              0, stream>>>(table, rows, k, bounds, sorted, active, active_count, x,
                                           x_div, out);
    check_launch("bf16_expert_rows");
}

void bf16_rows_f32(const __nv_bfloat16* table, int k, const std::int32_t* ids, int rows,
                   float* out, std::int64_t out_stride, cudaStream_t stream) {
    if (rows <= 0) { return; }
    bf16_rows_f32_kernel<<<dim3(unsigned(rows), blocks_for(k, 256)), 256, 0, stream>>>(
        table, k, ids, out, out_stride);
    check_launch("bf16_rows_f32");
}

void softcap_bf16(const float* in, __nv_bfloat16* out, std::int64_t n, float cap,
                  cudaStream_t stream) {
    if (n <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(blocks_for(n, 256)), dim3(256), std::size_t(0), stream}, softcap_kernel, in, out, n, cap);
    check_launch("softcap");
}

void softcap_bf16(const float* in, std::int64_t in_stride, __nv_bfloat16* out,
                  std::int64_t out_stride, int width, int rows, float cap, cudaStream_t stream) {
    if (width <= 0 || rows <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(blocks_for(width, 256), rows)), dim3(256), std::size_t(0), stream}, softcap_rows_kernel, 
        in, in_stride, out, out_stride, width, cap);
    check_launch("softcap rows");
}

void f32_to_bf16(const float* in, __nv_bfloat16* out, std::int64_t n, cudaStream_t stream) {
    if (n <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(blocks_for(n, 256)), dim3(256), std::size_t(0), stream}, f32_to_bf16_kernel, in, out, n);
    check_launch("f32_to_bf16");
}

void place_rows(const __nv_bfloat16* rows, int count, int hidden, float* h, cudaStream_t stream) {
    const std::int64_t n = std::int64_t(count) * hidden;
    if (n <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(blocks_for(n, 256)), dim3(256), std::size_t(0), stream}, place_rows_kernel, rows, n, h);
    check_launch("place_rows");
}

void ring_copy(const RingCopyArgs& a, cudaStream_t stream) {
    if (a.end <= a.begin) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(a.end - a.begin, a.kv_heads, 1)), dim3(64), std::size_t(0), stream}, ring_copy_kernel, a);
    check_launch("ring_copy");
}

void spark_mix(const SparkMixArgs& args, cudaStream_t stream) {
    if (args.end <= args.begin) { return; }
    if (args.tap_count < 1 || args.tap_count > kMaxSparkTaps || args.context_count < 0 ||
        args.context_count + 1 > kMaxSparkMixes || args.slots <= 0) {
        throw std::invalid_argument("inkling2 spark_mix: 1..8 taps and mixes");
    }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(args.end - args.begin), dim3(row_threads(args.hidden)), std::size_t(0), stream}, spark_mix_kernel, args);
    check_launch("spark_mix");
}

void spark_rotate(const SparkRotateArgs& args, cudaStream_t stream) {
    if (args.end <= args.begin || args.heads <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(dim3(args.end - args.begin, args.heads)), dim3(kD), std::size_t(0), stream}, spark_rotate_kernel, args);
    check_launch("spark_rotate");
}

void add_rows(float* h, const float* y, std::int64_t y_stride, int hidden, int rows,
              cudaStream_t stream) {
    const std::int64_t n = std::int64_t(rows) * hidden;
    if (n <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(blocks_for(n, 256)), dim3(256), std::size_t(0), stream}, add_rows_kernel, h, y, y_stride, hidden, n);
    check_launch("add_rows");
}

void silu_mul(const float* gu, int width, int rows, __nv_bfloat16* out, cudaStream_t stream) {
    const std::int64_t n = std::int64_t(rows) * width;
    if (n <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(blocks_for(n, 256)), dim3(256), std::size_t(0), stream}, silu_mul_kernel, gu, width, n, out);
    check_launch("silu_mul");
}

void spark_table(const float* p, int n, __nv_bfloat16* m, float* c, cudaStream_t stream) {
    const std::int64_t total = std::int64_t(n) * (kSparkRank + 1);
    if (total <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(blocks_for(total, 256)), dim3(256), std::size_t(0), stream}, spark_table_kernel, p, n, m, c);
    check_launch("spark_table");
}

void spark_head_inputs(const float* hidden_out, const float* bias, int hidden, int rank, float* out,
                       cudaStream_t stream) {
    if (rank != kSparkRank) { throw std::invalid_argument("inkling2 spark: a Markov rank of 256"); }
    const std::int64_t total = std::int64_t(rank + 1) * hidden;
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(blocks_for(total, 256)), dim3(256), std::size_t(0), stream}, spark_head_inputs_kernel, hidden_out, bias, hidden,
                                                                         rank, out);
    check_launch("spark_head_inputs");
}

std::size_t spark_step_floats(int vocab) { return std::size_t(spark_scratch(vocab)); }

void spark_step(const SparkStepArgs& args, cudaStream_t stream) {
    SparkStepArgs a = args;
    if (a.block == 0) { a.block = a.lanes == 1 ? 0 : -1; }
    if (a.row < 1 || a.vocab <= 0 || a.hidden_size % 4 != 0 || a.cap <= 0.0f || a.lanes < 1 ||
        a.lanes > kSparkMaxLanes || (a.lanes > 1 && a.block < 1)) {
        throw std::invalid_argument("inkling2 spark_step: invalid step");
    }
    // The scratch layout is fixed per lane: joint (3 * 256), logits (vocab), partials.
    if (a.logits != a.joint + 3 * kSparkRank || a.partials != a.logits + a.vocab) {
        throw std::invalid_argument("inkling2 spark_step: scratch out of layout");
    }
    const int k = 2 * kSparkRank + 2 * a.hidden_size;
    static const bool configured = [k] {
        CUDA_CHECK(cudaFuncSetAttribute(spark_joint_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(std::size_t(k) * 4 * kSparkJointLanes)));
        return true;
    }();
    (void)configured;
    // The joint product a group of lanes at a time (each lane's rows, state, previous draft and
    // scratch shifted to the group's first); the head and the picks take every lane at once.
    const std::int64_t scratch = spark_scratch(a.vocab);
    const int rows_per_lane    = a.lanes == 1 ? 1 : a.block;
    for (int g0 = 0; g0 < a.lanes; g0 += kSparkJointLanes) {
        SparkStepArgs g = a;
        g.lanes         = std::min(kSparkJointLanes, a.lanes - g0);
        g.prev          = a.prev + g0;
        g.state         = a.state + std::int64_t(g0) * kSparkRank;
        g.sequences     = a.sequences + std::int64_t(g0) * rows_per_lane;
        g.positions     = a.positions + std::int64_t(g0) * rows_per_lane;
        g.hidden        = a.hidden + std::int64_t(g0) * rows_per_lane * a.hidden_size;
        g.joint         = a.joint + std::int64_t(g0) * scratch;
        const std::size_t smem = std::size_t(k) * 4 * g.lanes;
        (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(3 * kSparkRank / 8), dim3(256), smem, stream}, spark_joint_kernel, g);
        check_launch("spark_joint");
    }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(blocks_for(a.vocab, 8)), dim3(256), std::size_t(0), stream}, spark_logits_kernel, a);
    check_launch("spark_logits");
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(kSparkPartials, a.lanes), dim3(256), std::size_t(0), stream}, spark_pick_partials_kernel, a);
    check_launch("spark_pick");
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(a.lanes), dim3(kSparkPartials), std::size_t(0), stream}, spark_pick_final_kernel, a);
    check_launch("spark_pick_final");
}

void add_fixed(unsigned long long* sums, const void* products, const std::int32_t* flag,
               std::int64_t n, cudaStream_t stream) {
    if (n <= 0) { return; }
    (void)ninfer::pdl::launch_consumer(ninfer::pdl::LaunchConfig{dim3(blocks_for(n, 256)), dim3(256), std::size_t(0), stream}, add_fixed_kernel, 
        sums, static_cast<const unsigned long long*>(products), flag, n);
    check_launch("add_fixed");
}

} // namespace ninfer::models::inkling2::kernels
