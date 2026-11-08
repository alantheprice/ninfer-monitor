#include "ninfer/ops/sampling.h"

#include "ops/sampling/sample/plan.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ninfer::ops {
namespace {
// Diagnostic only: classifies the first decode-step logits. Gated by NINFER_LOGIT_DEBUG.
void probe_logits(const Tensor& logits, std::int32_t token_domain, cudaStream_t stream) {
    static const bool on = [] {
        const char* v = getenv("NINFER_LOGIT_DEBUG");
        return v && *v;
    }();
    static std::uint32_t budget = 0;
    if (!on || budget >= 8) return;
    ++budget;
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "LOGITDBG sync failed\n");
        return;
    }
    constexpr int kWords = 8;
    std::uint16_t head[kWords]{}, tail[kWords]{};
    const auto* base = static_cast<const std::uint16_t*>(logits.data);
    const auto rows  = logits.ne[0];
    cudaMemcpy(head, base, kWords * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(tail, base + (rows - kWords) * (logits.nb[0] / 2), kWords * sizeof(std::uint16_t),
               cudaMemcpyDeviceToHost);
    std::uint32_t nan = 0;
    for (int i = 0; i < kWords; ++i) {
        nan += (head[i] & 0x7f80u) == 0x7f80u && (head[i] & 0x007fu);
        nan += (tail[i] & 0x7f80u) == 0x7f80u && (tail[i] & 0x007fu);
    }
    std::fprintf(stderr, "LOGITDBG rows=%d B=%d domain=%d head=", rows, logits.ne[1], token_domain);
    for (int i = 0; i < kWords; ++i) std::fprintf(stderr, "%04x ", head[i]);
    std::fprintf(stderr, "tail=");
    for (int i = 0; i < kWords; ++i) std::fprintf(stderr, "%04x ", tail[i]);
    std::fprintf(stderr, "nan_in_sample=%u\n", nan);
}
} // namespace

std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain, std::int32_t min_lanes,
                                              std::int32_t max_lanes) {
    if (token_domain <= 0 || min_lanes <= 0 || max_lanes < min_lanes) {
        throw std::invalid_argument("sampling workspace: invalid profile or lane interval");
    }
    return sample_detail::capacity(token_domain, min_lanes, max_lanes);
}

void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            MemoryArena& workspace, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("sample: workspace belongs to another device");
    }
    if (logits.device != execution.device() || out.device != execution.device() ||
        logical_positions.device != execution.device()) {
        throw std::invalid_argument("sample: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("sample: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("sample: out must be I32"); }
    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("sample: logits must be rank-2 [physical_rows,B]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("sample: out must be rank-1 [B]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("sample: physical rows must be positive");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("sample: token_domain must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) { throw std::invalid_argument("sample: out shape must be [B]"); }
    if (logical_positions.dtype != DType::I32 || logical_positions.ne[0] != logits.ne[1] ||
        logical_positions.ne[1] != 1 || logical_positions.ne[2] != 1 ||
        logical_positions.ne[3] != 1) {
        throw std::invalid_argument("sample: logical_positions must be I32 [logits.ne[1]]");
    }
    if (logits.ne[1] <= 0) { throw std::invalid_argument("sample: B must be positive"); }
    if (!logits.is_contiguous() || !out.is_contiguous() || !logical_positions.is_contiguous()) {
        throw std::invalid_argument("sample: logits/out/logical_positions must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr || logical_positions.data == nullptr) {
        throw std::invalid_argument("sample: tensor data must be non-null");
    }
    if (configs == nullptr) { throw std::invalid_argument("sample: configs must be non-null"); }

    auto scratch_scope       = workspace.scope();
    const auto plan          = sample_detail::select_plan(token_domain, logits.ne[1]);
    probe_logits(logits, token_domain, stream);
    const std::size_t bytes  = plan.workspace.bytes;
    const MemorySpan scratch = bytes == 0 ? MemorySpan{} : workspace.alloc_bytes(bytes);
    sample_detail::launch(logits, out, token_domain, configs, logical_positions, purpose, plan,
                          scratch, stream);
}

} // namespace ninfer::ops
