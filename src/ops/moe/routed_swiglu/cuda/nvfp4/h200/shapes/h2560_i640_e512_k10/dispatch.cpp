#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "core/platform.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
Plan select_plan(const SwiGLUProfile&, int tokens, int) {
    Plan plan;
    plan.implementation = Implementation::BlockScaled;
    const int columns   = tokens <= 16 ? 8 : tokens <= 128 ? 32 : 64;
    // Same hazard as the b200 target (fix 4e3b7ac9): the software NVFP4 MMA path
    // (mma.cuh, arch < 1200) expects fragments pre-dequantized to f16x2, but
    // staging copies raw e2m1 codes, which decode reinterprets as f16 -> NaN.
    // sm_90a is < 1200 too, so route every token count through the (correct)
    // vector kernels here as well until staging dequantizes for arch < 1200.
    plan.block_scaled   = {.format                = QType::NVFP4,
                           .gate_columns          = tokens >= 3584 ? 128 : columns,
                           .down_columns          = columns,
                           .vector                = true,
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
} // namespace ninfer::ops::detail::moe_routed
