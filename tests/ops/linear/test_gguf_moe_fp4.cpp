// The W4A4 routed-expert product (quantize_moe_fp4_activation, moe_fp4_product) at Inkling 2
// Mini's expert shapes: a [gate; up] parent of 2 x 1152 rows over 2304 inputs and a down
// projection of 2304 rows over 1152 (k = 4.5 and 2.25 of ggml's 512-value FP4 steps, which the
// kernel's own K loop covers without reading past a row), eight experts a token.
//
// 1. The activation the GPU writes is the host reference's byte for byte: the per-column factor,
//    every UE4M3 scale and E2M1 code (nvfp4_activation.h), in block_fp4_mmq's layout and the
//    routing's column order, zeros past k. Special columns: all zeros, one huge value among small
//    ones, values near 1e-6.
// 2. Every schedule and column tile's product is held against an FP64 product of the weights
//    ggml's dequantizer decodes exactly and of the activation the codes, scales and factor define:
//    only the kernel's FP32 sums separate the two.
// 3. The product is held, loosely, against the product of the BF16 activation itself, with the
//    q8_1 matrix route's error printed beside it: the 4-bit activation's own error.
// Runs only where moe_fp4_supported (sm_120a / sm_121a code); exit 77 elsewhere.

#include "core/device_buffer.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/nvfp4_activation.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf  = ops::gguf;
namespace nvfp4 = ops::gguf::nvfp4;

constexpr int kHidden = 2304, kWidth = 1152, kExperts = 16, kTopK = 8;
constexpr int kBanks       = 3;  // expert e's matrices are bank e % kBanks's
constexpr int kSample      = 32; // one row in kSample is held against the oracle
constexpr int kBlockBytes  = 36, kBlockValues = 64; // block_nvfp4
constexpr int kFp4Block    = 144;                   // block_fp4_mmq
constexpr std::size_t kTail = 4096;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

// Random NVFP4 rows: random codes, every UE4M3 scale finite and positive (0x20 .. 0x3F: 2^-3 ..
// 0.94); the tensor cores read a NaN scale (0x7F) as NaN, which no quantizer writes.
std::vector<std::uint8_t> nvfp4_rows(int rows, int k, std::mt19937& rng) {
    std::vector<std::uint8_t> bytes(std::size_t(rows) * (k / kBlockValues) * kBlockBytes);
    std::uniform_int_distribution<int> byte(0, 255), scale(0x20, 0x3F);
    for (std::size_t at = 0; at < bytes.size(); at += kBlockBytes) {
        for (int j = 0; j < 4; ++j) { bytes[at + j] = std::uint8_t(scale(rng)); }
        for (int j = 4; j < kBlockBytes; ++j) { bytes[at + j] = std::uint8_t(byte(rng)); }
    }
    return bytes;
}

std::size_t scales_bytes(int pairs) { return (std::size_t(pairs) * 4 + 255) / 256 * 256; }

// The host reference of one column: its factor and its block_fp4_mmq blocks [k_blocks256][144].
struct Column {
    float factor = 0.0f;
    std::vector<std::uint8_t> blocks;
    std::vector<double> values; // codes x scales x factor, the product's activation
};

Column quantize_column(const float* x, int k) {
    const int nkb = (k + 255) / 256;
    Column out;
    out.blocks.assign(std::size_t(nkb) * kFp4Block, 0);
    out.values.assign(std::size_t(k), 0.0);
    float amax      = 0.0f;
    bool all_finite = true;
    for (int i = 0; i < k; ++i) {
        amax       = std::max(amax, nvfp4::abs_value(x[i]));
        all_finite = all_finite && nvfp4::finite(x[i]);
    }
    out.factor = nvfp4::column_factor(amax, all_finite);
    if (!all_finite) { return out; }
    for (int g = 0; g < k / 64; ++g) {
        std::uint8_t* block = out.blocks.data() + std::size_t(g / 4) * kFp4Block;
        const int f         = g % 4;
        for (int sub = 0; sub < 4; ++sub) {
            const float* v = x + g * 64 + sub * 16;
            float a16      = 0.0f;
            for (int i = 0; i < 16; ++i) { a16 = std::max(a16, nvfp4::abs_value(v[i])); }
            const std::uint8_t s = nvfp4::sub_scale(a16, amax);
            std::uint8_t codes[16];
            for (int i = 0; i < 16; ++i) {
                codes[i] = nvfp4::value_code(v[i], s, out.factor);
                out.values[std::size_t(g * 64 + sub * 16 + i)] =
                    double(nvfp4::e2m1_value(codes[i])) * double(nvfp4::scale_value(s)) *
                    double(out.factor);
            }
            std::uint32_t words[2];
            nvfp4::pack_sub(codes, words);
            block[4 * f + sub] = s;
            std::memcpy(block + 16 + 32 * f + 8 * sub, words, 8);
        }
    }
    return out;
}

struct Error {
    double rms = 0.0, max = 0.0; // relative to the oracle's root mean square
};

Error compare(const std::vector<double>& got, const std::vector<double>& want) {
    double num = 0.0, den = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double e = got[i] - want[i];
        num += e * e;
        den += want[i] * want[i];
        worst = std::max(worst, std::fabs(e));
    }
    const double rms = std::sqrt(den / double(want.size()));
    return {std::sqrt(num / den), worst / rms};
}

int report(const std::string& what, const Error& e, double rms_bound, double max_bound) {
    const bool ok = std::isfinite(e.rms) && e.rms < rms_bound && e.max < max_bound;
    std::cout << (ok ? "OK   " : "FAIL ") << what << ": rms " << e.rms << ", max " << e.max
              << " (bounds " << rms_bound << ", " << max_bound << ")\n";
    return ok ? 0 : 1;
}

// One projection of every expert (rows x k) over `tokens` tokens; the activation's columns are the
// tokens (column_group kTopK) or the pairs (1).
int run(const char* name, int rows, int k, int tokens, int column_group, std::mt19937& rng) {
    const int pairs              = tokens * kTopK;
    const int columns            = pairs / column_group;
    const std::int64_t row_bytes = std::int64_t(k / kBlockValues) * kBlockBytes;
    std::vector<std::vector<std::uint8_t>> host(kBanks);
    std::vector<DeviceBuffer> banks;
    for (int b = 0; b < kBanks; ++b) {
        host[std::size_t(b)] = nvfp4_rows(rows, k, rng);
        // No `tail` is promised: the bank ends where its rows do, plus a guard the kernel must not
        // need (it is filled with NaN scales so a read past a row would show).
        banks.emplace_back(host[std::size_t(b)].size() + kTail);
        banks.back().fill(0x7F);
        banks.back().copy_from_host(host[std::size_t(b)].data(), host[std::size_t(b)].size());
    }
    std::vector<const void*> table(kExperts);
    for (int e = 0; e < kExperts; ++e) { table[std::size_t(e)] = banks[std::size_t(e % kBanks)].p; }
    DeviceBuffer d_table(table.size() * sizeof(void*));
    d_table.copy_from_host(table.data(), table.size() * sizeof(void*));
    const gguf::MoeTable moe{static_cast<const void* const*>(d_table.p), row_bytes, rows, k,
                             row_bytes % 16 == 0};

    std::vector<std::int32_t> sampled;
    for (int i = 0; i * kSample < rows; ++i) { sampled.push_back(i * kSample + i % kSample); }
    const int n = int(sampled.size());
    DeviceBuffer d_sampled(sampled.size() * 4), d_exact(std::size_t(n) * k * 4);
    d_sampled.copy_from_host(sampled.data(), sampled.size() * 4);
    std::vector<std::vector<float>> exact(kBanks, std::vector<float>(std::size_t(n) * k));
    for (int b = 0; b < kBanks; ++b) {
        gguf::dequantize_rows(gguf::GgmlType::NVFP4, banks[std::size_t(b)].p, row_bytes, k,
                              static_cast<const std::int32_t*>(d_sampled.p), n,
                              static_cast<float*>(d_exact.p), k, nullptr);
        check(cudaDeviceSynchronize(), "dequantize");
        d_exact.copy_to_host(exact[std::size_t(b)].data(), exact[std::size_t(b)].size() * 4);
    }

    std::vector<std::int32_t> ids(static_cast<std::size_t>(pairs));
    std::vector<int> order(kExperts);
    for (int t = 0; t < tokens; ++t) {
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        std::copy_n(order.begin(), kTopK, ids.begin() + std::ptrdiff_t(t) * kTopK);
    }
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x(std::size_t(columns) * k);
    std::vector<float> xf(x.size());
    // With enough columns the first four are special; the errors below are measured over the
    // regular columns and over each special one apart, each against its own scale.
    const int specials = columns >= 16 ? 4 : 0;
    for (int c = 0; c < columns; ++c) {
        for (int i = 0; i < k; ++i) {
            float v = normal(rng);
            if (c < specials) {
                if (c == 0) { v = 0.0f; }                          // all zeros
                if (c == 1) { v = i == 7 ? 30000.0f : 0.01f * v; } // one huge value
                if (c == 2) { v *= 1e-6f; }                        // tiny values
                if (c == 3 && i >= 128 && i < 192) { v = 0.0f; }   // a zero group
            }
            const std::size_t at = std::size_t(c) * k + std::size_t(i);
            x[at]                = __float2bfloat16(v);
            xf[at]               = __bfloat162float(x[at]);
        }
    }
    DeviceBuffer d_ids(ids.size() * 4), d_x(x.size() * 2),
        routing_space(gguf::moe_routing_bytes(kExperts, pairs)),
        fp4_act(gguf::moe_fp4_activation_bytes(k, pairs)),
        q8_act(gguf::moe_matrix_activation_bytes(k, pairs)), d_q8(std::size_t(pairs) * rows * 4),
        d_out(std::size_t(pairs) * rows * 4);
    d_ids.copy_from_host(ids.data(), ids.size() * 4);
    d_x.copy_from_host(x.data(), x.size() * 2);
    fp4_act.fill(0xAB); // the quantizer writes every byte the product reads
    const auto* xb       = static_cast<const __nv_bfloat16*>(d_x.p);
    const auto routing   = gguf::moe_sort_routes(static_cast<const std::int32_t*>(d_ids.p), pairs,
                                                 kExperts, routing_space.p, nullptr);
    const int max_active = std::min(pairs, kExperts);
    gguf::quantize_moe_fp4_activation(xb, k, routing, pairs, column_group, fp4_act.p, nullptr);
    check(cudaDeviceSynchronize(), "fp4 activation");
    std::vector<std::int32_t> sorted(static_cast<std::size_t>(pairs));
    check(cudaMemcpy(sorted.data(), routing.sorted, sorted.size() * 4, cudaMemcpyDeviceToHost),
          "sorted");

    const std::string what = std::string(name) + " (" + std::to_string(rows) + " x " +
                             std::to_string(k) + "), " + std::to_string(tokens) + " tokens";
    int failures = 0;

    // 1. The activation, byte for byte.
    const int nkb = (k + 255) / 256;
    std::vector<std::uint8_t> got_act(gguf::moe_fp4_activation_bytes(k, pairs));
    fp4_act.copy_to_host(got_act.data(), got_act.size());
    std::vector<Column> ref(static_cast<std::size_t>(columns));
    for (int c = 0; c < columns; ++c) { ref[std::size_t(c)] = quantize_column(&xf[std::size_t(c) * k], k); }
    std::size_t act_mismatches = 0;
    for (int c = 0; c < pairs; ++c) {
        const Column& col = ref[std::size_t(sorted[std::size_t(c)] / column_group)];
        float factor      = 0.0f;
        std::memcpy(&factor, got_act.data() + std::size_t(c) * 4, 4);
        std::uint32_t a, b;
        std::memcpy(&a, &factor, 4);
        std::memcpy(&b, &col.factor, 4);
        act_mismatches += a != b;
        for (int kb = 0; kb < nkb; ++kb) {
            const std::uint8_t* gb =
                got_act.data() + scales_bytes(pairs) + (std::size_t(kb) * pairs + c) * kFp4Block;
            act_mismatches += std::memcmp(gb, col.blocks.data() + std::size_t(kb) * kFp4Block,
                                          kFp4Block) != 0;
        }
    }
    std::cout << (act_mismatches == 0 ? "OK   " : "FAIL ") << what
              << ", activation vs the host's: " << act_mismatches << " mismatched blocks/factors\n";
    failures += act_mismatches != 0;

    // The FP64 oracles at the sampled rows: the FP4 activation's and the BF16 activation's.
    std::vector<double> want_fp4, want_exact;
    std::vector<int> sample_column; // the activation column of each sample
    for (int p = 0; p < pairs; ++p) {
        const int column = p / column_group;
        const auto& w    = exact[std::size_t(ids[std::size_t(p)] % kBanks)];
        const auto& a    = ref[std::size_t(column)].values;
        const float* xc  = &xf[std::size_t(column) * k];
        for (int s = 0; s < n; ++s) {
            const float* row = w.data() + std::size_t(s) * k;
            double fp4 = 0.0, bf16 = 0.0;
            for (int i = 0; i < k; ++i) {
                fp4 += double(row[i]) * a[std::size_t(i)];
                bf16 += double(row[i]) * double(xc[i]);
            }
            want_fp4.push_back(fp4);
            want_exact.push_back(bf16);
            sample_column.push_back(column);
        }
    }
    const auto gather = [&](const DeviceBuffer& out) {
        std::vector<float> all(std::size_t(pairs) * rows);
        out.copy_to_host(all.data(), all.size() * 4);
        std::vector<double> got;
        for (int p = 0; p < pairs; ++p) {
            for (int s = 0; s < n; ++s) {
                got.push_back(all[std::size_t(p) * rows + std::size_t(sampled[std::size_t(s)])]);
            }
        }
        return got;
    };

    // The samples of the regular columns (group -1) or of special column `group` alone.
    const auto part = [&](const std::vector<double>& v, int group) {
        std::vector<double> out;
        for (std::size_t i = 0; i < v.size(); ++i) {
            const int c = sample_column[i];
            if (group < 0 ? c >= specials : c == group) { out.push_back(v[i]); }
        }
        return out;
    };
    // Each group against its oracle; the all-zero column must be exactly zero.
    const auto held = [&](const std::string& label, const std::vector<double>& got,
                          const std::vector<double>& want, double rms_bound, double max_bound) {
        int f = 0;
        if (!part(got, -1).empty()) {
            f += report(label, compare(part(got, -1), part(want, -1)), rms_bound, max_bound);
        }
        for (int c = 1; c < specials; ++c) {
            f += report(label + " (special column " + std::to_string(c) + ")",
                        compare(part(got, c), part(want, c)), rms_bound, max_bound);
        }
        if (specials > 0) {
            const auto zero = part(got, 0);
            const bool ok   = std::all_of(zero.begin(), zero.end(), [](double v) { return v == 0.0; });
            std::cout << (ok ? "OK   " : "FAIL ") << label << " (zero column): exactly zero\n";
            f += ok ? 0 : 1;
        }
        return f;
    };

    // 2. Every schedule and column tile against the FP4 activation's product.
    struct Variant {
        int staged, tile;
    };
    std::vector<double> got_default;
    for (const Variant v : {Variant{1, 16}, Variant{1, 32}, Variant{1, 64}, Variant{0, 16},
                            Variant{0, 32}, Variant{0, 64}, Variant{0, 128}}) {
        d_out.fill(0xFF); // NaN: every sampled row must be written
        gguf::moe_fp4_product(moe, routing, max_active, pairs, fp4_act.p,
                              static_cast<float*>(d_out.p), nullptr, v.staged, v.tile);
        check(cudaDeviceSynchronize(), "fp4 product");
        const auto got = gather(d_out);
        failures += held(what + ", " + (v.staged ? "staged" : "sync") + " J=" +
                             std::to_string(v.tile) + " vs its activation",
                         got, want_fp4, 1e-5, 2e-4);
        if (got_default.empty()) { got_default = got; }
    }

    // 3. Against the BF16 activation, beside the q8_1 route's error.
    gguf::quantize_moe_matrix_activation(gguf::GgmlType::NVFP4, xb, k, routing, pairs,
                                         column_group, q8_act.p, nullptr);
    // The q8_1 route reads past each row's end into zeros: its banks for it get a zero tail.
    std::vector<DeviceBuffer> zero_banks;
    std::vector<const void*> zero_table(kExperts);
    for (int b = 0; b < kBanks; ++b) {
        zero_banks.emplace_back(host[std::size_t(b)].size() + kTail);
        zero_banks.back().fill(0);
        zero_banks.back().copy_from_host(host[std::size_t(b)].data(), host[std::size_t(b)].size());
    }
    for (int e = 0; e < kExperts; ++e) {
        zero_table[std::size_t(e)] = zero_banks[std::size_t(e % kBanks)].p;
    }
    DeviceBuffer d_zero_table(zero_table.size() * sizeof(void*));
    d_zero_table.copy_from_host(zero_table.data(), zero_table.size() * sizeof(void*));
    const gguf::MoeTable q8_moe{static_cast<const void* const*>(d_zero_table.p), row_bytes, rows,
                                k, row_bytes % 16 == 0};
    gguf::moe_matrix_product(gguf::GgmlType::NVFP4, q8_moe, routing, max_active, pairs, true,
                             q8_act.p, static_cast<float*>(d_q8.p), nullptr);
    check(cudaDeviceSynchronize(), "q8 product");
    const auto got_q8 = gather(d_q8);
    const Error q8    = compare(part(got_q8, -1), part(want_exact, -1));
    std::cout << "     " << what << ", q8_1 route vs the BF16 activation: rms " << q8.rms
              << ", max " << q8.max << "\n";
    // Regular columns only: the special ones lose their small values by design (a block whose
    // largest value is below 2^-9 / 448 of its column's flushes to zero).
    failures += report(what + ", fp4 route vs the BF16 activation",
                       compare(part(got_default, -1), part(want_exact, -1)), 0.15, 1.0);
    return failures;
}

} // namespace

int main() {
    try {
        if (!gguf::moe_fp4_supported(gguf::GgmlType::NVFP4, 2 * kWidth, kHidden) ||
            !gguf::moe_fp4_supported(gguf::GgmlType::NVFP4, kHidden, kWidth)) {
            std::cout << "SKIP: no FP4 tensor-core route on this device or build (sm_120a code)\n";
            return 77;
        }
        int failures = 0;
        if (gguf::moe_fp4_supported(gguf::GgmlType::Q4_0, 2 * kWidth, kHidden) ||
            gguf::moe_fp4_supported(gguf::GgmlType::NVFP4, 2 * kWidth + 64, kHidden) ||
            gguf::moe_fp4_supported(gguf::GgmlType::NVFP4, kHidden, kWidth + 32)) {
            std::cout << "FAIL: moe_fp4_supported accepts a type or geometry it has no kernel for\n";
            ++failures;
        }
        std::mt19937 rng(20261125);
        for (const int tokens : {3, 200}) {
            failures += run("gate_up", 2 * kWidth, kHidden, tokens, kTopK, rng);
            failures += run("down", kHidden, kWidth, tokens, 1, rng);
        }
        std::cout << (failures == 0 ? "PASS" : "FAIL") << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
