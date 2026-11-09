#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"
#include "core/platform.h"

namespace ninfer::ops::detail::moe_complete {
namespace {
Plan select_plan(int tokens, int, ExpertSubsetKind) {
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::BlockScaled;
    const int columns   = tokens <= 16 ? 8 : tokens <= 128 ? 32 : 64;
    // sm_100a has no warp-level FP4 MMA, so mma_nvfp4_e4m3 falls back to a software path
    // that expects fragments pre-dequantized to f16x2. stage_inputs still stages raw e2m1
    // codes, which the software path reinterprets as f16 and turns into NaN. Until the
    // staging dequantizes for arch < 1200, keep the (correct) vector kernels for every
    // token count on this target.
    const bool vector = NINFER_TARGET_B200 ? true : tokens == 1;
    plan.block_scaled   = {.format                = QType::NVFP4,
                           .gate_columns          = tokens >= 3584 ? 128 : columns,
                           .down_columns          = columns,
                           .vector                = vector,
                           .atomic_output         = tokens >= 128,
                           .single_block_grouping = tokens <= 64,
                           .shared_algorithm      = tokens < 8
                                                        ? kernel::block_scaled::SharedAlgorithm::Vector
                                                        : kernel::block_scaled::SharedAlgorithm::Mma,
                           .queued_ctas_per_sm    = 6};
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h2560_i640_e512_k10() {
    static constexpr int boundaries[]{2, 17, 65, 128, 129, 3584};
    static const ShapePolicy policy{.select = select_plan, .allocation_boundaries = boundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_complete
