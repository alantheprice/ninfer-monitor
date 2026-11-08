#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace ninfer::models::qwen4_exp::execution::debug {

// Env-gated activation probes for the sm_100a numerics hunt. Each probe is off unless its
// variable is set, and each is sampled at most kBudget times per process so a decode loop
// cannot flood stderr. Probes synchronize their stream: they are diagnostic only.

// Cached per call site: the probe flags are read once, so the decode hot path never calls
// getenv. A call site passes a distinct literal name and gets its own static.
template <const char* Name>
bool enabled_cached() {
    static const bool on = [] {
        const char* v = getenv(Name);
        return v && *v;
    }();
    return on;
}

// Convenience for call sites that cannot template on a literal; still caches per name string
// address, which is stable for string literals.
inline bool enabled(const char* name) {
    struct Entry {
        const char* name;
        bool on;
    };
    static Entry cache[16]{};
    static std::size_t used = 0;
    for (std::size_t i = 0; i < used; ++i)
        if (cache[i].name == name) return cache[i].on;
    const char* v = getenv(name);
    const bool on = v && *v;
    if (used < 16) cache[used++] = {name, on};
    return on;
}

struct Sample {
    static constexpr int kWords = 16;
    std::uint16_t head[kWords]{};
    std::uint16_t tail[kWords]{};
    std::uint32_t nan = 0, inf = 0, finite = 0;
};

// Classifies a BF16 exponent field: 0xff with a nonzero mantissa is NaN, 0xff with a zero
// mantissa is Inf. Everything else is finite, including zero and subnormals.
inline void classify(const std::uint16_t* words, int n, Sample& s) {
    for (int i = 0; i < n; ++i) {
        const auto exponent = words[i] & 0x7f80u;
        const auto mantissa = words[i] & 0x007fu;
        if (exponent != 0x7f80u)
            ++s.finite;
        else if (mantissa)
            ++s.nan;
        else
            ++s.inf;
    }
}

// Classifies every BF16 element of a tensor's first `elements` words and captures the leading
// and trailing kWords for display. `row` selects a batch row of a [., ., tokens, batch]
// activation; pass row 0 for a rank-2 [rows, B] tensor such as logits.
inline void sample_tensor(const Tensor& t, std::int32_t row, Sample& s) {
    if (!t.data || t.dtype != DType::BF16) return;
    const auto* base = static_cast<const std::uint16_t*>(t.data);
    // Rank-2 [rows, B]: one row is ne[0] words with stride nb[0]. Rank-3/4 activations:
    // [features, streams, tokens, batch], so a batch row is the whole feature*stream extent
    // with nb[1] as its element stride.
    const bool rank2 = t.ne[2] == 1 && t.ne[3] == 1;
    const auto n      = rank2 ? t.ne[0] : t.ne[0] * t.ne[1];
    const auto stride = rank2 ? t.nb[0] / 2 : t.nb[1] / 2;
    const auto offset = rank2 ? static_cast<std::int64_t>(row) * (t.nb[1] / 2)
                              : static_cast<std::int64_t>(row) * (t.nb[2] / 2);
    if (n <= 0 || stride <= 0) return;
    const auto* row_base = base + offset;

    // Full-row classification. The row is a device pointer, so copy it out in chunks rather
    // than classifying only the displayed words.
    constexpr int kChunk = 4096;
    std::uint16_t chunk[kChunk];
    for (int start = 0; start < n; start += kChunk) {
        const int count = n - start < kChunk ? n - start : kChunk;
        if (cudaMemcpy(chunk, row_base + static_cast<std::int64_t>(start) * stride,
                       count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            return;
        classify(chunk, count, s);
    }

    const int head = n < Sample::kWords ? n : Sample::kWords;
    cudaMemcpy(s.head, row_base, head * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    if (n > head) {
        const int tail = n - head < Sample::kWords ? n - head : Sample::kWords;
        cudaMemcpy(s.tail, row_base + static_cast<std::int64_t>(n - tail) * stride,
                   tail * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    }
}

inline void print(const char* tag, std::uint32_t layer, std::uint32_t token, const Sample& s) {
    std::fprintf(stderr, "LAYERDBG %s layer=%u tok=%u nan=%u inf=%u head=", tag, layer, token,
                 s.nan, s.inf);
    for (int i = 0; i < 8; ++i) std::fprintf(stderr, "%04x ", s.head[i]);
    std::fprintf(stderr, "tail=");
    for (int i = 0; i < 8; ++i) std::fprintf(stderr, "%04x ", s.tail[i]);
    std::fprintf(stderr, "\n");
}

} // namespace ninfer::models::qwen4_exp::execution::debug
