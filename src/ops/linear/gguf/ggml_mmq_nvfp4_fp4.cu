// The routed experts' W4A4 matrix product on Blackwell's block-scaled FP4 tensor cores (sm_120a):
// NVFP4 expert rows times an NVFP4 activation (nvfp4_activation.h), one
// mma.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 per 64 values, both operands' UE4M3 scales
// applied by the tensor cores and the activation column's FP32 factor at the write-back. The
// weights go to shared memory as they are stored (no conversion to int8, unlike the q8_1 route of
// ggml_bridge_mmq.cuh), the activation is half the q8_1 one's bytes, and each MMA covers twice the
// values of the int8 m16n8k32.
//
// The fragments, their shared-memory layout, the MMA and the write-back are the vendored ggml's
// (ggml_cuda_mmq_vec_dot_fp4_fp4_mma, ggml_cuda_mmq_write_back_mma, block_fp4_mmq); the K loop is
// this file's own so a row of any whole number of 64-value blocks runs without reading past it:
// ggml's loop steps 512 values (MMQ_ITER_K_FP4) and Inkling 2's widths (hidden 2304, expert 1152)
// are 4.5 and 2.25 of them. Here the last step's second 256-value half is skipped when the row
// ends in the first, and a row's blocks past its end are zeros in the tile, never loaded.
//
// Two schedules (Fp4Mode): kSync loads each step's tiles when it starts (any column tile, the
// shape of ggml's own loop); kStaged keeps two weight tiles, copying the next step's into one with
// cp.async (4-byte copies straight into the MMA layout, zero-filled past the row) and its
// activation into registers while the tensor cores work on the other. Two 38 KB weight tiles leave
// room for column tiles of up to 64 pairs (96.5 KB of the 99 KB a CTA may take).

#include "ggml_bridge.h"
#include "ggml_bridge_internal.cuh"
#include "nvfp4_activation.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>

namespace ninfer::ops::gguf {
namespace detail {
namespace {

#if defined(BLACKWELL_MMA_AVAILABLE) &&                                                            \
    (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||                 \
     defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
#define NINFER_FP4_MMA_AVAILABLE
#endif

constexpr int kFp4Threads   = 256; // 8 warps, ggml's Blackwell FP4 configuration
constexpr int kFp4Warps     = kFp4Threads / 32;
constexpr int kFp4RowTile   = 128; // I
constexpr int kFp4StepBlocks = 8;  // NVFP4 blocks (64 values) a step: 512 values
constexpr int kFp4SramStride = 2 * MMQ_TILE_NE_K + 8 + 4; // GGML_CUDA_MMQ_SRAM_LAYOUT_FP4: 76 words
constexpr int kFp4YInts      = MMQ_TILE_Y_K;               // 36 ints a column of block_fp4_mmq
static_assert(sizeof(block_fp4_mmq) == kFp4YInts * 4, "block_fp4_mmq is 36 words");
static_assert(sizeof(block_nvfp4) == 36, "block_nvfp4 is 4 scale bytes and 32 code bytes");
// The activation over-reads up to one widest tile of columns past the last pair.
constexpr std::size_t kFp4Slack = 128 * sizeof(block_fp4_mmq);

constexpr std::size_t align256(std::size_t n) { return (n + 255) / 256 * 256; }

enum Fp4Mode : int { kSync = 0, kStaged = 1 };

constexpr std::size_t fp4_x_tile_bytes() { return std::size_t(kFp4RowTile) * kFp4SramStride * 4; }

constexpr std::size_t fp4_shared_bytes(int J, int mode) {
    return std::size_t(J) * 4 + 2 * std::size_t(J) * kFp4YInts * 4 +
           (mode == kStaged ? 2 : 1) * fp4_x_tile_bytes();
}

// The weights of one step a thread moves: 4 rows (one every 32) of one block each, the block's
// scale word and its eight code words.
struct Fp4XRegs {
    std::uint32_t scale[kFp4RowTile / 32];
    uint4 qs[kFp4RowTile / 32][2];
};

template <int J>
struct Fp4YRegs {
    static constexpr int chunks    = 2 * J * kFp4YInts / 4; // int4s of both halves' tiles
    static constexpr int per_thread = (chunks + kFp4Threads - 1) / kFp4Threads;
    int4 v[per_thread];
};

#if defined(NINFER_FP4_MMA_AVAILABLE)

// Thread (lane, warp) moves block kbx = lane % 8 of rows r * 32 + warp * 4 + lane / 8.
__device__ __forceinline__ void fp4_load_x(Fp4XRegs& r, const char* __restrict__ x, int offset_x,
                                           int stride_row_x, int kb0, int k_blocks) {
    const int kbx = threadIdx.x % 8;
    const bool in = kb0 + kbx < k_blocks;
#pragma unroll
    for (int p = 0; p < kFp4RowTile / 32; ++p) {
        const int i = p * 32 + threadIdx.y * 4 + threadIdx.x / 8;
        if (in) {
            const auto* w = reinterpret_cast<const std::uint32_t*>(
                x + (std::int64_t(offset_x) + std::int64_t(i) * stride_row_x + kb0 + kbx) *
                        std::int64_t(sizeof(block_nvfp4)));
            r.scale[p] = __ldg(w);
            r.qs[p][0] = make_uint4(__ldg(w + 1), __ldg(w + 2), __ldg(w + 3), __ldg(w + 4));
            r.qs[p][1] = make_uint4(__ldg(w + 5), __ldg(w + 6), __ldg(w + 7), __ldg(w + 8));
        } else {
            r.scale[p] = 0;
            r.qs[p][0] = make_uint4(0, 0, 0, 0);
            r.qs[p][1] = make_uint4(0, 0, 0, 0);
        }
    }
}

// The tile as ggml_cuda_mmq_load_tiles_nvfp4_nvfp4 lays it out: a row's 64 code words, then its
// 8 scale words (one a block), sram stride 76 words.
__device__ __forceinline__ void fp4_store_x(const Fp4XRegs& r, int* __restrict__ tile_x) {
    const int kbx = threadIdx.x % 8;
    auto* x_u32   = reinterpret_cast<std::uint32_t*>(tile_x);
#pragma unroll
    for (int p = 0; p < kFp4RowTile / 32; ++p) {
        const int i = p * 32 + threadIdx.y * 4 + threadIdx.x / 8;
        auto* row   = x_u32 + i * kFp4SramStride;
        reinterpret_cast<uint4*>(row + 8 * kbx)[0] = r.qs[p][0];
        reinterpret_cast<uint4*>(row + 8 * kbx)[1] = r.qs[p][1];
        row[2 * MMQ_TILE_NE_K + kbx]               = r.scale[p];
    }
}

// The same tile through cp.async: each of the thread's 4 blocks is 9 copies of 4 bytes (a block's
// scale word, then its code words; block_nvfp4 rows only start on 4 bytes), zeros past the row.
__device__ __forceinline__ void fp4_stage_x(int* __restrict__ tile_x, const char* __restrict__ x,
                                            int offset_x, int stride_row_x, int kb0,
                                            int k_blocks) {
    const int kbx          = threadIdx.x % 8;
    const bool in          = kb0 + kbx < k_blocks;
    const int bytes        = in ? 4 : 0;
    const unsigned int dst = ggml_cuda_cvta_generic_to_shared(tile_x);
#pragma unroll
    for (int p = 0; p < kFp4RowTile / 32; ++p) {
        const int i = p * 32 + threadIdx.y * 4 + threadIdx.x / 8;
        const char* src =
            in ? x + (std::int64_t(offset_x) + std::int64_t(i) * stride_row_x + kb0 + kbx) *
                         std::int64_t(sizeof(block_nvfp4))
               : x;
        const unsigned int row = dst + unsigned(i * kFp4SramStride) * 4u;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
                     :: "r"(row + unsigned(2 * MMQ_TILE_NE_K + kbx) * 4u), "l"(src), "r"(bytes));
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
                         :: "r"(row + unsigned(8 * kbx + w) * 4u), "l"(src + 4 + 4 * w),
                            "r"(bytes));
        }
    }
    asm volatile("cp.async.commit_group;");
}

// Both 256-value halves of a step's activation tile: half h of column j at
// y[(kb256 + h) * ncols_y * 36 + j * 36]; the second half only when the row reaches it.
template <int J>
__device__ __forceinline__ void fp4_load_y(Fp4YRegs<J>& r, const int* __restrict__ y, int ncols_y,
                                           int kb256, bool second) {
    constexpr int half = J * kFp4YInts / 4;
    const int tid      = threadIdx.y * 32 + threadIdx.x;
#pragma unroll
    for (int s = 0; s < Fp4YRegs<J>::per_thread; ++s) {
        const int l = s * kFp4Threads + tid;
        const int h = l >= half ? 1 : 0;
        if (l < Fp4YRegs<J>::chunks && (h == 0 || second)) {
            const int4* src = reinterpret_cast<const int4*>(
                y + std::int64_t(kb256 + h) * ncols_y * kFp4YInts);
            r.v[s] = __ldg(src + (l - h * half));
        } else {
            r.v[s] = make_int4(0, 0, 0, 0);
        }
    }
}

template <int J>
__device__ __forceinline__ void fp4_store_y(const Fp4YRegs<J>& r, int* __restrict__ tile_y) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
#pragma unroll
    for (int s = 0; s < Fp4YRegs<J>::per_thread; ++s) {
        const int l = s * kFp4Threads + tid;
        if (l < Fp4YRegs<J>::chunks) { reinterpret_cast<int4*>(tile_y)[l] = r.v[s]; }
    }
}

#endif // NINFER_FP4_MMA_AVAILABLE

// One CTA per row tile and listed (expert, column tile), as moe_mul_mat_q (ggml_bridge_mmq.cuh):
// column c of the activation is the routing's c-th pair, and pair p's FP32 row lands at
// dst[p * rows], times the column's factor y_scale[c].
template <int J, int Mode>
__launch_bounds__(kFp4Threads, 1) __global__
    void moe_mul_mat_fp4(const void* const* __restrict__ experts, const int* __restrict__ y,
                         const float* __restrict__ y_scale, const std::int32_t* __restrict__ sorted,
                         const std::int32_t* __restrict__ bounds,
                         const std::int32_t* __restrict__ tiles,
                         const std::int32_t* __restrict__ tile_count, float* __restrict__ dst,
                         int k_blocks, int rows, int stride_row_x, int ncols_y) {
#if defined(NINFER_FP4_MMA_AVAILABLE)
    constexpr ggml_type type = GGML_TYPE_NVFP4;
    static_assert(ggml_cuda_mmq_get_I(type, J, false, GGML_PREC_Q4) == kFp4RowTile &&
                      ggml_cuda_mmq_get_nthreads(type, J, false, GGML_PREC_Q4) == kFp4Threads &&
                      ggml_cuda_mmq_get_sram_stride(type, J, false, GGML_PREC_Q4) == kFp4SramStride,
                  "ggml's Blackwell FP4 configuration changed");
    static_assert(Mode == kSync || J <= 64, "two weight tiles leave room for 64 columns");
    if (int(blockIdx.y) >= *tile_count) { return; }
    const int tile     = tiles[blockIdx.y];
    const int expert   = tile & 0xFFFF;
    const int jt       = tile >> 16;
    const int col_low  = bounds[expert];
    const int col_diff = bounds[expert + 1] - col_low;
    const int it       = blockIdx.x;

    extern __shared__ int fp4_shared[];
    int* ids     = fp4_shared;
    int* tile_a  = ids + J;                // the step's first 256 values of the J columns
    int* tile_b  = tile_a + J * kFp4YInts; // its second 256
    int* tile_x0 = tile_b + J * kFp4YInts; // 16-byte aligned: J is a multiple of 16
    int* tile_x1 = tile_x0 + kFp4RowTile * kFp4SramStride;

    const int tid = threadIdx.y * 32 + threadIdx.x;
    for (int j = tid; j < J; j += kFp4Threads) {
        ids[j] = jt * J + j < col_diff ? sorted[col_low + jt * J + j] : 0;
    }

    const char* x      = static_cast<const char*>(experts[expert]);
    const int offset_x = it * kFp4RowTile * stride_row_x;
    const int* y_tile  = y + std::int64_t(col_low + jt * J) * kFp4YInts;

    float sum[J * kFp4RowTile / kFp4Threads] = {0.0f};
    Fp4YRegs<J> yr;
    if constexpr (Mode == kStaged) {
        fp4_stage_x(tile_x0, x, offset_x, stride_row_x, 0, k_blocks);
        fp4_load_y<J>(yr, y_tile, ncols_y, 0, 4 < k_blocks);
        int stage = 0;
        for (int kb0 = 0; kb0 < k_blocks; kb0 += kFp4StepBlocks, stage ^= 1) {
            const bool second = kb0 + 4 < k_blocks; // the row reaches the step's second half
            int* tile_x       = stage == 0 ? tile_x0 : tile_x1;
            fp4_store_y<J>(yr, tile_a);
            cp_async_wait_all();
            __syncthreads();
            const int next = kb0 + kFp4StepBlocks;
            if (next < k_blocks) {
                // The other weight tile's last readers finished before the previous step's end.
                fp4_stage_x(stage == 0 ? tile_x1 : tile_x0, x, offset_x, stride_row_x, next,
                            k_blocks);
                fp4_load_y<J>(yr, y_tile, ncols_y, next / 4, next + 4 < k_blocks);
            }
            ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, false>(tile_x, tile_a, sum, 0);
            if (second) {
                ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, false>(tile_x, tile_b, sum,
                                                                  MMQ_TILE_NE_K);
            }
            __syncthreads();
        }
    } else {
        Fp4XRegs xr;
        for (int kb0 = 0; kb0 < k_blocks; kb0 += kFp4StepBlocks) {
            const bool second = kb0 + 4 < k_blocks;
            fp4_load_x(xr, x, offset_x, stride_row_x, kb0, k_blocks);
            fp4_load_y<J>(yr, y_tile, ncols_y, kb0 / 4, second);
            fp4_store_x(xr, tile_x0);
            fp4_store_y<J>(yr, tile_a);
            __syncthreads();
            ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, false>(tile_x0, tile_a, sum, 0);
            if (second) {
                ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, false>(tile_x0, tile_b, sum,
                                                                  MMQ_TILE_NE_K);
            }
            __syncthreads();
        }
    }
    ggml_cuda_mmq_write_back_mma<type, J, false, float, false>(
        sum, ids, dst + it * kFp4RowTile, y_scale + col_low + jt * J, rows,
        rows - it * kFp4RowTile - 1, col_diff - jt * J - 1);
#else
    NO_DEVICE_CODE;
#endif
}

// Whether this binary holds the FP4 kernels for the running device (sm_120a / sm_121a code).
__global__ void fp4_probe_kernel(int* out) {
#if defined(NINFER_FP4_MMA_AVAILABLE)
    *out = 1;
#else
    *out = 0;
#endif
}

// One warp a column (pair) c, four columns a CTA: lane l takes the column's 64-value groups l,
// l + 32, ... First the column's largest magnitude, then each group's four sub-blocks.
constexpr int kFp4QuantizeColumns = 4;

__device__ __forceinline__ void fp4_load_group(const __nv_bfloat16* __restrict__ col, int g,
                                               float v[64]) {
    const uint4* src = reinterpret_cast<const uint4*>(col + std::int64_t(g) * 64);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const uint4 w        = src[q];
        const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[h]));
            v[8 * q + 2 * h]     = f.x;
            v[8 * q + 2 * h + 1] = f.y;
        }
    }
}

__global__ void __launch_bounds__(32 * kFp4QuantizeColumns)
    quantize_fp4_kernel(const __nv_bfloat16* __restrict__ x, int k, int k_blocks256,
                        const std::int32_t* __restrict__ sorted, int column_group, int pairs,
                        block_fp4_mmq* __restrict__ y, float* __restrict__ y_scale) {
    const int c = int(blockIdx.x) * kFp4QuantizeColumns + int(threadIdx.x) / 32;
    if (c >= pairs) { return; }
    const int lane             = threadIdx.x % 32;
    const __nv_bfloat16* col   = x + std::int64_t(sorted[c] / column_group) * k;
    const int groups           = k / nvfp4::kGroupValues;
    const int groups_padded    = k_blocks256 * (nvfp4::kBlockValues / nvfp4::kGroupValues);
    float amax                 = 0.0f;
    bool all_finite            = true;
    for (int g = lane; g < groups; g += 32) {
        float v[64];
        fp4_load_group(col, g, v);
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            amax       = fmaxf(amax, nvfp4::abs_value(v[i]));
            all_finite = all_finite && nvfp4::finite(v[i]);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
    }
    all_finite    = __all_sync(0xFFFFFFFFu, all_finite);
    const float g = nvfp4::column_factor(amax, all_finite);
    if (lane == 0) { y_scale[c] = g; }
    for (int gr = lane; gr < groups_padded; gr += 32) {
        std::uint32_t qs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        std::uint32_t scales = 0;
        if (gr < groups && all_finite) {
            float v[64];
            fp4_load_group(col, gr, v);
#pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                float a16 = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) { a16 = fmaxf(a16, nvfp4::abs_value(v[16 * sub + i])); }
                const std::uint8_t s = nvfp4::sub_scale(a16, amax);
                std::uint8_t codes[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) { codes[i] = nvfp4::value_code(v[16 * sub + i], s, g); }
                nvfp4::pack_sub(codes, qs + 2 * sub);
                scales |= std::uint32_t(s) << (8 * sub);
            }
        }
        block_fp4_mmq* block = y + std::int64_t(gr / 4) * pairs + c;
        const int f          = gr % 4;
        block->d4[f]         = scales;
        auto* words          = reinterpret_cast<uint4*>(block->qs) + 2 * f;
        words[0]             = make_uint4(qs[0], qs[1], qs[2], qs[3]);
        words[1]             = make_uint4(qs[4], qs[5], qs[6], qs[7]);
    }
}

bool fp4_compiled_for_device() {
    static std::array<int, 16> state{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    check(cudaGetDevice(&device), "device");
    std::call_once(once[device & 15], [&] {
        const int cc = device_facts().cc;
        if (cc < GGML_CUDA_CC_BLACKWELL || cc >= GGML_CUDA_CC_RUBIN) { return; }
        int* flag = nullptr;
        if (cudaMalloc(&flag, sizeof(int)) != cudaSuccess) {
            (void)cudaGetLastError();
            return;
        }
        int host = 0;
        fp4_probe_kernel<<<1, 1>>>(flag);
        if (cudaGetLastError() == cudaSuccess &&
            cudaMemcpy(&host, flag, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess) {
            state[device & 15] = host;
        }
        (void)cudaGetLastError();
        cudaFree(flag);
    });
    return state[device & 15] == 1;
}

int fp4_env(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? std::atoi(v) : fallback;
}

template <int J, int Mode>
void launch_fp4(const MoeTable& table, const MoeRouting& routing, int max_active, int pairs,
                const block_fp4_mmq* y, const float* y_scale, float* out, cudaStream_t stream) {
    constexpr std::size_t shared = fp4_shared_bytes(J, Mode);
    static std::once_flag raised[16];
    int device = 0;
    check(cudaGetDevice(&device), "device");
    std::call_once(raised[device & 15], [&] {
        check(cudaFuncSetAttribute(moe_mul_mat_fp4<J, Mode>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)),
              "fp4 moe matrix shared memory");
    });
    constexpr int width = J == 16 ? 0 : J == 32 ? 1 : J == 64 ? 2 : 3;
    static_assert(kMatrixColumnTiles[width] == J, "MoeRouting lists tiles of width 16 << w");
    const int listed = (pairs + J - 1) / J + max_active;
    if (listed > routing.tile_capacity || listed > 65535) {
        throw std::invalid_argument("gguf fp4 moe product: the routing lists fewer tiles");
    }
    const dim3 grid(table.rows / kFp4RowTile, listed, 1);
    const dim3 block(32, kFp4Warps, 1);
    moe_mul_mat_fp4<J, Mode><<<grid, block, shared, stream>>>(
        table.experts, reinterpret_cast<const int*>(y), y_scale, routing.sorted, routing.bounds,
        routing.tiles + std::int64_t(width) * routing.tile_capacity, routing.tile_count + width,
        out, table.k / QK_NVFP4, table.rows, int(table.row_bytes / sizeof(block_nvfp4)), pairs);
    check(cudaGetLastError(), "fp4 moe matrix product launch");
}

void launch_fp4_tile(int J, int mode, const MoeTable& table, const MoeRouting& routing,
                     int max_active, int pairs, const block_fp4_mmq* y, const float* y_scale,
                     float* out, cudaStream_t stream) {
    if (mode == kStaged) {
        switch (J) {
        case 16: return launch_fp4<16, kStaged>(table, routing, max_active, pairs, y, y_scale, out, stream);
        case 32: return launch_fp4<32, kStaged>(table, routing, max_active, pairs, y, y_scale, out, stream);
        case 64: return launch_fp4<64, kStaged>(table, routing, max_active, pairs, y, y_scale, out, stream);
        default: break;
        }
    } else {
        switch (J) {
        case 16: return launch_fp4<16, kSync>(table, routing, max_active, pairs, y, y_scale, out, stream);
        case 32: return launch_fp4<32, kSync>(table, routing, max_active, pairs, y, y_scale, out, stream);
        case 64: return launch_fp4<64, kSync>(table, routing, max_active, pairs, y, y_scale, out, stream);
        case 128: return launch_fp4<128, kSync>(table, routing, max_active, pairs, y, y_scale, out, stream);
        default: break;
        }
    }
    throw std::invalid_argument("gguf fp4 moe product: no kernel for this column tile and schedule");
}

int fp4_k_blocks256(int k) { return (k + nvfp4::kBlockValues - 1) / nvfp4::kBlockValues; }

std::size_t fp4_scales_bytes(int pairs) { return align256(std::size_t(pairs) * sizeof(float)); }

} // namespace
} // namespace detail

bool moe_fp4_supported(GgmlType type, int rows, int k) {
    return type == GgmlType::NVFP4 && rows > 0 && rows % detail::kFp4RowTile == 0 && k > 0 &&
           k % QK_NVFP4 == 0 &&
           detail::device_facts().shared_per_block_optin >=
               std::max(detail::fp4_shared_bytes(128, detail::kSync),
                        detail::fp4_shared_bytes(64, detail::kStaged)) &&
           detail::fp4_compiled_for_device();
}

std::size_t moe_fp4_activation_bytes(int k, int pairs) {
    return detail::fp4_scales_bytes(pairs) +
           std::size_t(pairs) * detail::fp4_k_blocks256(k) * sizeof(block_fp4_mmq) +
           detail::kFp4Slack;
}

void quantize_moe_fp4_activation(const __nv_bfloat16* x, int k, const MoeRouting& routing,
                                 int pairs, int column_group, void* out, cudaStream_t stream) {
    if (x == nullptr || out == nullptr || routing.sorted == nullptr || column_group <= 0 ||
        k <= 0 || k % QK_NVFP4 != 0 || pairs <= 0) {
        throw std::invalid_argument("gguf fp4 moe activation: invalid routing or K");
    }
    auto* base     = static_cast<std::uint8_t*>(out);
    auto* scales   = reinterpret_cast<float*>(base);
    auto* blocks   = reinterpret_cast<block_fp4_mmq*>(base + detail::fp4_scales_bytes(pairs));
    const int ctas = (pairs + detail::kFp4QuantizeColumns - 1) / detail::kFp4QuantizeColumns;
    detail::quantize_fp4_kernel<<<ctas, 32 * detail::kFp4QuantizeColumns, 0, stream>>>(
        x, k, detail::fp4_k_blocks256(k), routing.sorted, column_group, pairs, blocks, scales);
    detail::check(cudaGetLastError(), "fp4 moe activation launch");
}

namespace {

// NINFER_FP4_MOE_STAGED=0 takes the synchronous schedule (any tile up to 128), else the staged one
// (tiles up to 64).
bool fp4_staged() {
    static const bool staged = detail::fp4_env("NINFER_FP4_MOE_STAGED", 1) != 0;
    return staged;
}

} // namespace

int moe_fp4_column_tile(int pairs, int max_active, int staged, int tile) {
    const int widest = (staged < 0 ? fp4_staged() : staged != 0) ? 64 : 128;
    const int forced = tile != 0 ? tile : detail::fp4_env("NINFER_FP4_MOE_TILE", 0);
    if ((forced == 16 || forced == 32 || forced == 64 || forced == 128) && forced <= widest) {
        return forced;
    }
    // The narrowest tile that holds an active expert's pairs on average, as select_moe_launch.
    const int columns = (pairs + max_active - 1) / std::max(1, max_active);
    for (const int J : detail::kMatrixColumnTiles) {
        if (J >= columns || J == widest) { return J; }
    }
    return widest;
}

void moe_fp4_product(const MoeTable& table, const MoeRouting& routing, int max_active, int pairs,
                     const void* activation, float* out, cudaStream_t stream, int staged,
                     int tile) {
    if (table.experts == nullptr || routing.bounds == nullptr || routing.tiles == nullptr ||
        activation == nullptr || out == nullptr || max_active <= 0 || pairs <= 0 ||
        table.rows % detail::kFp4RowTile != 0 || table.k <= 0 || table.k % QK_NVFP4 != 0 ||
        table.row_bytes % std::int64_t(sizeof(block_nvfp4)) != 0 ||
        table.row_bytes < std::int64_t(table.k / QK_NVFP4) * std::int64_t(sizeof(block_nvfp4))) {
        throw std::invalid_argument("gguf fp4 moe product: invalid geometry");
    }
    const auto* base    = static_cast<const std::uint8_t*>(activation);
    const auto* scales  = reinterpret_cast<const float*>(base);
    const auto* blocks  =
        reinterpret_cast<const block_fp4_mmq*>(base + detail::fp4_scales_bytes(pairs));
    const bool use_staged = staged < 0 ? fp4_staged() : staged != 0;
    detail::launch_fp4_tile(moe_fp4_column_tile(pairs, max_active, use_staged ? 1 : 0, tile),
                            use_staged ? detail::kStaged : detail::kSync, table, routing,
                            max_active, pairs, blocks, scales, out, stream);
}

} // namespace ninfer::ops::gguf
