#include "ops/linear_add/q8/q8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_ksplit_grouped_mma.cuh"

namespace ninfer::ops::detail {
namespace {

template <int K>
void launch(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    constexpr int kColumns     = 128;
    constexpr int kSplits      = 2;
    constexpr int kTokenGroups = 4;
    const dim3 grid(5120 / 16, div_up(x.ne[1], kColumns));
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual.data), 5120};
    constexpr std::size_t kGroupedSmemBytes =
        16 * (kSplits * 64) + (kSplits * kTokenGroups) * (kColumns / kTokenGroups) * 64 * 2;
    static const cudaError_t smem_attr = cudaFuncSetAttribute(
        q8_ksplit_grouped_mma_kernel<K, kColumns, kSplits, kTokenGroups, 1, Q8ContiguousOutput,
                                    true, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kGroupedSmemBytes));
    CUDA_CHECK(smem_attr);
    q8_ksplit_grouped_mma_kernel<K, kColumns, kSplits, kTokenGroups, 1, Q8ContiguousOutput, true,
                                 true><<<grid, kSplits * kTokenGroups * 32, kGroupedSmemBytes,
                                          stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q8_linear_add_grouped_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                  cudaStream_t stream) {
    if (w.k == 6144) {
        launch<6144>(x, w, residual, stream);
    } else {
        launch<17408>(x, w, residual, stream);
    }
}

} // namespace ninfer::ops::detail
