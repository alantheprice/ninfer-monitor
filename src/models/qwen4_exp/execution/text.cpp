#include "models/qwen4_exp/execution/text.h"
#include "models/qwen4_exp/execution/debug_probe.h"
#include "models/qwen4_exp/execution/gdn.h"
#include "models/qwen4_exp/execution/qsa.h"
#include "models/qwen4_exp/execution/ple.h"
#include "core/transfer.h"
#include "core/nvtx.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

TextRun::TextRun(const Parameters& p, TextExecution execution, BatchView batch, StateUpdate update,
                 CudaExecutionView cuda)
    : parameters_(p), execution_(execution), batch_(batch), update_(update), cuda_(cuda),
      scope_(execution.workspace.scope()),
      roots_(workspace::roots(execution.workspace, p.profile->config.text,
                              batch.tokens * batch.batch)),
      expert_activity_(execution.experts ? execution.experts->begin(batch.tokens * batch.batch)
                                         : std::nullopt) {
    if (batch.tokens <= 0 || batch.batch < 1 ||
        batch.batch > static_cast<std::int32_t>(kMaximumConcurrency) ||
        (std::holds_alternative<SequenceState>(update) && batch.batch != 1) ||
        (std::holds_alternative<StepUpdate>(update) && batch.tokens != 1) ||
        (std::holds_alternative<VerifyRecords>(update) && batch.tokens < 2))
        throw std::invalid_argument("invalid Qwen4Exp sequence/step window");
}

void TextRun::stem(std::optional<models::vision::EmbeddingView> visual) {
    const auto& p = parameters_.text;
    ops::embedding(batch_.ids.view({batch_.tokens * batch_.batch}), p.embedding, roots_.input,
                   cuda_);
    if (debug::enabled("NINFER_LAYER_DEBUG") && batch_.tokens == 1)
        probe_activation("embedding", 0);
    if (visual)
        for (const auto& slice : visual->slices)
            ops::scatter(slice.features, slice.columns, roots_.input, cuda_);
    ops::hc_expand_norm(roots_.input, p.layers.front().attention_hc.norm,
                        parameters_.profile->config.text.norm_epsilon, roots_.residual,
                        roots_.normalized, cuda_);
    if (debug::enabled("NINFER_LAYER_DEBUG") && batch_.tokens == 1)
        probe_activation("expand_norm", 0);
}

bool TextRun::advance_layer(std::uint32_t i) {
    const nvtx::ScopedRange range(nvtx::Name::TextLayer, nvtx::Category::Runtime, i);
    const auto& p = parameters_.text.layers[i];
    if (waiting_moe_) {
        if (!execution_.experts->poll()) return false;
        waiting_moe_ = false;
    } else {
        if (execution_.experts && !execution_.experts->try_before_layer(p.moe.bank)) return false;
        if (debug::enabled("NINFER_MOE_DEBUG") && i < 2) {
            CUDA_CHECK(cudaStreamSynchronize(cuda_.stream()));
            debug::Sample n, r, inp;
            debug::sample_tensor(roots_.normalized, 0, n);
            debug::sample_tensor(roots_.residual, 0, r);
            debug::sample_tensor(roots_.input, 0, inp);
            debug::print("pre_mixer_norm", i, batch_.tokens, n);
            debug::print("pre_mixer_resid", i, batch_.tokens, r);
            debug::print("pre_mixer_input", i, batch_.tokens, inp);
        }
        mixer(i);
        if (debug::enabled("NINFER_LAYER_DEBUG") && i < 3 && batch_.tokens == 1)
            probe_activation("mixer", i);        if (execution_.experts) {
            if (!execution_.experts->start(p.moe.bank, roots_.input, roots_.output,
                                           execution_.workspace, {})) {
                waiting_moe_ = true;
                return false;
            }
        } else
            resident_moe(i);
        if (debug::enabled("NINFER_LAYER_DEBUG") && i < 3 && batch_.tokens == 1)
            probe_activation("moe", i);
    }
    combine(i);
    if (debug::enabled("NINFER_LAYER_DEBUG") && batch_.tokens == 1) probe_activation("combine", i);
    return true;
}

// Diagnostic only: synchronizes the compute stream and dumps the residual/normalized head and
// tail for batch row 0. Gated by NINFER_LAYER_DEBUG and capped so a decode loop cannot flood.
void TextRun::probe_activation(const char* stage, std::uint32_t layer) {
    static std::uint32_t budget = 0;
    if (budget >= 512) return;
    ++budget;
    CUDA_CHECK(cudaStreamSynchronize(cuda_.stream()));
    debug::Sample residual, normalized;
    debug::sample_tensor(roots_.residual, 0, residual);
    debug::sample_tensor(roots_.normalized, 0, normalized);
    debug::print(stage, layer, batch_.tokens, residual);
    if (normalized.nan || normalized.inf) debug::print("norm", layer, batch_.tokens, normalized);
}

void TextRun::mixer(std::uint32_t i) {
    local_mixer(i);
    finish_mixer(i);
}

void TextRun::local_mixer(std::uint32_t i) {
    const auto& config = parameters_.profile->config.text;
    const auto& p      = parameters_.text.layers[i];
    const auto& state  = config.layers[i];
    auto& arena        = execution_.workspace;
    {
        auto scratch = arena.scope();
        ops::hc_mix(roots_.normalized, p.attention_hc.mix, arena, roots_.input, roots_.injection,
                    cuda_);
    }
    if (const auto* g = std::get_if<GdnParameters>(&p.mixer)) {
        gdn(roots_.input, parameters_.profile->local.gdn, *g, batch_, execution_.states.gdn(),
            state.state_index, update_, config.norm_epsilon, roots_.output, arena, cuda_);
    } else {
        ops::QsaRawRecordWriteView records;
        if (!std::holds_alternative<SequenceState>(update_)) {
            records = {
                execution_.qsa_raw_records.slice(3, state.state_index, 1).slice(2, 0, batch_.batch),
                execution_.qsa_coordinate_records.slice(3, state.state_index, 1)
                    .slice(2, 0, batch_.batch),
                batch_.record_offsets};
        }
        const auto after = std::holds_alternative<VerifyRecords>(update_)
                               ? std::nullopt
                               : std::optional{execution_.states.main_tail(
                                     state.state_index, batch_.destination_offsets)};
        qsa(roots_.input, parameters_.profile->local.qsa, std::get<QsaParameters>(p.mixer), batch_,
            execution_.states.main_tail(state.state_index, batch_.source_offsets), after,
            execution_.cache.layer(state.state_index), records, roots_.output, arena, cuda_);
    }
}

void TextRun::finish_mixer(std::uint32_t i) {
    const auto& config = parameters_.profile->config.text;
    const auto& p      = parameters_.text.layers[i];
    auto& arena        = execution_.workspace;
    if (i == 0 && execution_.first_collective_ready)
        execution_.first_collective_ready->wait(cuda_.stream());
    if (execution_.parallel) execution_.parallel->sum(roots_.output, cuda_);
    ops::hc_combine_norm(roots_.residual, roots_.output, roots_.injection, p.moe_hc.norm,
                         config.norm_epsilon, roots_.residual, roots_.normalized, cuda_);
    {
        auto scratch = arena.scope();
        ops::hc_mix(roots_.normalized, p.moe_hc.mix, arena, roots_.input, roots_.injection, cuda_);
    }
}

void TextRun::resident_moe(std::uint32_t i) {
    const auto& p = parameters_.text.layers[i].moe;
    const bool probe = debug::enabled("NINFER_MOE_DEBUG") && i < 2;
    if (probe) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_.stream()));
        debug::Sample in;
        debug::sample_tensor(roots_.input, 0, in);
        debug::print("moe_in", i, batch_.tokens, in);
    }
    moe::execute(roots_.input, p, p.resident, ops::MoeEpilogue::Write, roots_.output,
                 execution_.workspace, cuda_);
    if (probe) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_.stream()));
        debug::Sample out;
        debug::sample_tensor(roots_.output, 0, out);
        debug::print("moe_out", i, batch_.tokens, out);
    }
}

void TextRun::gpu_moe(const moe::GpuCall& call) {
    auto scratch = execution_.workspace.scope();
    call(roots_.input, roots_.output, execution_.workspace, {});
}

void TextRun::combine(std::uint32_t i) {
    if (execution_.parallel) execution_.parallel->sum(roots_.output, cuda_);
    const auto& config = parameters_.profile->config.text;
    if (i + 1 == config.ple.layer) {
        if (debug::enabled("NINFER_LAYER_DEBUG") && i < 2) {
            CUDA_CHECK(cudaStreamSynchronize(cuda_.stream()));
            debug::Sample out, inj, res;
            debug::sample_tensor(roots_.output, 0, out);
            debug::sample_tensor(roots_.residual, 0, res);
            debug::print("pre_hccombine_block", i, batch_.tokens, out);
            debug::print("pre_hccombine_residual", i, batch_.tokens, res);
            const auto* inj_ptr = static_cast<const float*>(roots_.injection.data);
            const auto n = roots_.injection.ne[0] * roots_.injection.ne[1];
            std::vector<float> host(static_cast<std::size_t>(n));
            cudaMemcpy(host.data(), inj_ptr, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::uint32_t bad = 0;
            for (auto v : host) bad += !std::isfinite(v);
            std::fprintf(stderr, "LAYERDBG pre_hccombine_injection layer=%u n=%d nonfinite=%u "
                                 "first=%.6g %.6g %.6g %.6g\n",
                         i, n, bad, host[0], host.size() > 1 ? host[1] : 0.f,
                         host.size() > 2 ? host[2] : 0.f, host.size() > 3 ? host[3] : 0.f);
        }
        ops::hc_combine(roots_.residual, roots_.output, roots_.injection, roots_.residual, cuda_);
    } else {
        const auto& norm = i + 1 < config.layers.size()
                               ? parameters_.text.layers[i + 1].attention_hc.norm
                               : parameters_.text.final_hc.norm;
        ops::hc_combine_norm(roots_.residual, roots_.output, roots_.injection, norm,
                             config.norm_epsilon, roots_.residual, roots_.normalized, cuda_);
    }
}

void TextRun::consume_ple(const Tensor& embedding) {
    const auto& p    = parameters_.text;
    const auto start = parameters_.profile->config.text.ple.layer;
    ops::PleConvRecordWindowView records;
    if (!std::holds_alternative<SequenceState>(update_))
        records = {execution_.ple_records.slice(2, 0, batch_.batch), batch_.record_offsets};
    const auto after = std::holds_alternative<VerifyRecords>(update_)
                           ? std::nullopt
                           : std::optional{execution_.states.ple(batch_.destination_offsets)};
    ple(embedding, parameters_.profile->config.text, p.ple, batch_,
        execution_.states.ple(batch_.source_offsets), after, records, p.layers[start].attention_hc,
        roots_.residual, roots_.normalized, execution_.workspace, cuda_);
}

void TextRun::finish(TextOutputs output, Tensor logits) {
    const auto& p = parameters_.text;
    if (output.conditioning.data)
        copy_bytes_async({output.conditioning.data, output.conditioning.bytes(), cuda_.device()},
                         {roots_.residual.data, roots_.residual.bytes(), cuda_.device()}, cuda_);
    auto sample = output.sample.view(
        {dimension(parameters_.profile->config.text.hidden), batch_.tokens * batch_.batch});
    if (debug::enabled("NINFER_LAYER_DEBUG") && batch_.tokens == 1) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_.stream()));
        debug::Sample pre_normalized, pre_residual;
        debug::sample_tensor(roots_.normalized, 0, pre_normalized);
        debug::sample_tensor(roots_.residual, 0, pre_residual);
        debug::print("pre_hcmix_normalized", parameters_.text.layers.size(), batch_.tokens,
                     pre_normalized);
        debug::print("pre_hcmix_residual", parameters_.text.layers.size(), batch_.tokens,
                     pre_residual);
    }
    ops::hc_mix(roots_.normalized, p.final_hc.mix, execution_.workspace, sample, cuda_);
    if (debug::enabled("NINFER_LAYER_DEBUG") && batch_.tokens == 1) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_.stream()));
        debug::Sample hidden;
        debug::sample_tensor(sample, 0, hidden);
        debug::print("final_sample", parameters_.text.layers.size(), batch_.tokens, hidden);
    }
    if (logits.data) {
        logits = logits.view({logits.ne[0], batch_.tokens * batch_.batch});
        project_head(sample, p.head, logits, execution_.workspace, cuda_);
        if (debug::enabled("NINFER_LAYER_DEBUG") && batch_.tokens == 1) {
            CUDA_CHECK(cudaStreamSynchronize(cuda_.stream()));
            debug::Sample head;
            debug::sample_tensor(logits, 0, head);
            debug::print("logits", parameters_.text.layers.size(), batch_.tokens, head);
        }
    }
}

void project_head(const Tensor& sample, const LinearParameters& head, Tensor& logits,
                  MemoryArena& arena, CudaExecutionView cuda) {
    auto scope = arena.scope();
    ops::linear(sample, head.weight, logits, head.policy, arena, cuda);
}

} // namespace ninfer::models::qwen4_exp::execution
