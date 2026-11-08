#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/execution/debug_probe.h"

#include <cstdio>

namespace ninfer::models::qwen4_exp::detail {

Tensor RankProgramImpl::device_embedding(std::int32_t tokens, std::int32_t batch) {
    return packed_tensor(*resources.plan.persistent.buffers.ple_embedding,
                         {dimension(parameters.profile->config.text.ple.embedding), tokens, batch});
}

void RankProgramImpl::prepare_device_ple(const execution::BatchView& batch) {
    const auto& table    = std::get<DevicePleTable>(parameters.model.ple_table());
    const auto& controls = parameters.model.definition().ple_parameters().hash;
    ops::NgramHashParameters hash;
    for (std::size_t i = 0; i < hash.multipliers.size(); ++i)
        hash.multipliers[i] = static_cast<std::uint64_t>(controls.multipliers[i]);
    for (std::size_t i = 0; i < hash.offsets.size(); ++i) {
        hash.vocab_sizes[i] = controls.vocab_sizes[i];
        hash.offsets[i]     = controls.offsets[i];
    }
    const auto& io     = resources.plan.persistent.buffers.batch;
    const auto tail    = packed_tensor(io.raw_tail, {2, batch.batch});
    const auto offsets = device_tensor(io.raw_tail_offsets).slice(0, 0, batch.batch);
    const StateFieldView tail_field{
        {{tail.data, tail.bytes(), tail.device}, offsets},
        {{{0, 2 * sizeof(TokenId), alignof(TokenId)}, DType::I32, {2, 1, 1, 1}},
         1,
         2 * sizeof(TokenId)}};
    auto embedding      = device_embedding(batch.tokens, batch.batch);
    const auto producer = resources.ple_transfer();
    ple_inputs.record(cuda.stream());
    ple_inputs.wait(producer.stream());
    ops::ngram_embedding_lookup(batch.ids, batch.valid_columns, {tail_field}, hash,
                                parameters.profile->config.text.ple.eos, table.shards, embedding,
                                producer);
    if (parallel) parallel->merge_unique(embedding, producer);
    ple_ready.record(producer.stream());
    if (execution::debug::enabled("NINFER_PLE_DEBUG")) {
        CUDA_CHECK(cudaStreamSynchronize(producer.stream()));
        probe_ple("device", batch);
    }
}

// Diagnostic only: dumps the PLE tail words, the ngram hash row per head, and the first
// embedding words for batch row 0. Gated by NINFER_PLE_DEBUG and capped.
void RankProgramImpl::probe_ple(const char* placement, const execution::BatchView& batch) {
    static std::uint32_t budget = 0;
    if (budget >= 32) return;
    ++budget;
    const auto& io     = resources.plan.persistent.buffers.batch;
    const auto* tail   = static_cast<const TokenId*>(device_tensor(io.raw_tail).data);
    const auto* ids    = static_cast<const TokenId*>(batch.ids.data);
    const auto* valid  = static_cast<const TokenId*>(batch.valid_columns.data);
    const auto* tail_offset = static_cast<const std::int64_t*>(device_tensor(io.raw_tail_offsets).data);
    TokenId tail_words[2]{};
    TokenId first_ids[4]{};
    TokenId valid_columns = 0;
    std::int64_t tail_bytes = 0;
    cudaMemcpy(tail_words, tail, sizeof(tail_words), cudaMemcpyDeviceToHost);
    cudaMemcpy(&valid_columns, valid, sizeof(valid_columns), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tail_bytes, tail_offset, sizeof(tail_bytes), cudaMemcpyDeviceToHost);
    cudaMemcpy(first_ids, ids, sizeof(first_ids), cudaMemcpyDeviceToHost);
    std::fprintf(stderr,
                 "PLEDBG %s tokens=%d batch=%d valid=%d tail_bytes=%lld tail=[%d %d] ids=[%d %d "
                 "%d %d]\n",
                 placement, batch.tokens, batch.batch, valid_columns,
                 static_cast<long long>(tail_bytes), tail_words[0], tail_words[1], first_ids[0],
                 first_ids[1], first_ids[2], first_ids[3]);
    auto embedding = device_embedding(batch.tokens, batch.batch);
    execution::debug::Sample sample;
    execution::debug::sample_tensor(embedding, 0, sample);
    execution::debug::print("ple_embedding", 0, batch.tokens, sample);
}

} // namespace ninfer::models::qwen4_exp::detail
