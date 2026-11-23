#pragma once

// The forward pass of Inkling 2 over a loaded Model: 48 visits of its 24 stored layers (the
// prelude, the looped core three times, the coda), each visit with its own keys and values and
// its own short-convolution history. The caches are F16, or — with kv_cache rk8v4 / rk4v4 — the
// eight full-attention visits' planes are quantized (kv_format.h) while every windowed visit
// keeps its F16 ring. A sliding-window visit keeps a ring of the last window
// (and a round's drafts) per sequence, a full-attention visit every position. Every visit's
// convolution sites keep a ring of their last kHistory raw inputs (FP32) per sequence. Prompt
// chunks run up to prefill_chunk tokens of one sequence; decode steps run one token of each of
// several sequences at once, replayed from a CUDA graph per batch size.
//
// With draft_tokens, verify() runs a sequence's next token (its anchor) and up to draft_tokens
// proposed tokens after it as decode rows, and commit() keeps the first ones. With a DSpine
// drafter (ExecutorOptions::dspine) every pass stores the drafter's taps of the rows it computes,
// the rows a sequence keeps go into the drafter's context ring, and draft() proposes. With the
// model's DSpark drafter (ExecutorOptions::dspark) every pass writes the drafter's context keys and
// values and its recurrent head's context for each row it computes from the visits the drafter taps
// (a verification's rejected rows lie past the next anchor, where the next block overwrites them);
// draft() runs the drafter's block (the anchor and masks, every row seeing the anchor's window and
// the whole block), then its recurrent head row by row, a draft at a time.
//
// With the model's vision tower (vision_max_patches), encode_image() turns an image's patches into
// its span's rows, and a prompt chunk's image rows take the place of their tokens' embeddings.

#include "core/device.h"
#include "core/expert_disk.h"
#include "core/tensor.h"
#include "models/expert_profile.h"
#include "models/inkling2/kv_format.h"
#include "models/inkling2/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::inkling2 {

// The device lacks the memory the options need; the message names the amounts.
struct InsufficientMemory : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A call needed a sequence's full-attention rows backed further than the device has room for (the
// growing caches without an elastic expert cache to give way). Nothing was mapped or run: free a
// sequence (preempt it) and retry.
struct KvExhausted : InsufficientMemory {
    using InsufficientMemory::InsufficientMemory;
};

struct ExecutorOptions {
    std::uint32_t max_context   = 32768;
    std::uint32_t prefill_chunk = 2048; // prompt tokens per forward()
    std::uint32_t sequences     = 1;
    bool cuda_graphs            = true;
    // The KV cache's storage (kv_format.h): F16 (bf16, the default, or fp16), or the quantized
    // full-attention visits of rk8v4 / rk4v4 — the windowed visits' rings stay F16 whatever this
    // says.
    KvCacheStorage kv_cache     = KvCacheStorage::BFloat16;
    // Proposed tokens a verification may carry per sequence after its anchor (0: no
    // verification); a verification's rows, the anchors and drafts of all its sequences, are at
    // most 8.
    std::uint32_t draft_tokens = 0;
    // One checkpoint of every sequence's rings (snapshot/restore), allocated with the caches.
    bool checkpoints = true;
    // Drafts with the model's DSpine drafter (Model::dspine(), draft_tokens of them at most).
    bool dspine = false;
    // Drafts with the model's DSpark drafter (Model::dspark(); its block of drafts a round).
    bool dspark = false;
    // The full-attention visits' K and V in E4M3 with static scales (--kv-dtype fp8): a byte a
    // value instead of two. The sliding rings and the drafter's caches stay F16.
    bool kv_fp8 = false;
    // Nonzero: the DSpark drafter's full-attention layers attend over the last draft_window
    // positions, kept in a ring, instead of the whole context. Drafts are verified, so this
    // changes how many survive, never the output; it bounds the drafter's per-sequence caches.
    std::uint32_t draft_window = 0;
    // Prompt calls may carry images (a model loaded with its vision tower): the most patches an
    // image has, which sizes the tower's scratch. 0: text only.
    std::uint32_t vision_max_patches = 0;
    std::optional<std::uint64_t> expert_cache_bytes;
    // Host or disk experts: the cache's first contents, the most routed of these counts
    // ([expert layer][expert]) once warm-up is done; none leaves the cache empty then.
    std::optional<ExpertRouteCounts> expert_profile;
    // The share 0..1 of a decode or verification call's missing experts the GPU copies; the CPU
    // computes the rest (cpu_experts.h). Unset: each call splits its misses by measured CPU and
    // bus times. CPU threads: 0 for one per physical core.
    std::optional<float> expert_dma_share;
    std::uint32_t expert_cpu_threads = 0;
    // Disk experts: a drive mirror and the direct path (core/expert_disk.h).
    ExpertDiskReads expert_disk{};
};

struct ExecutorMemory {
    std::uint64_t kv_bytes        = 0;
    // Beside an elastic expert cache, the full-attention caches grow with their sequences: the
    // bytes backed now, of kv_bytes at most.
    std::uint64_t kv_backed_bytes = 0;
    bool kv_grows                 = false;
    // The short convolutions' histories (and their checkpoints), inside kv_bytes.
    std::uint64_t history_bytes   = 0;
    std::uint64_t workspace_bytes = 0;
    // The vision tower: the scratch of one encode within workspace_bytes, and the device bytes the
    // tower's image buffers own beside the model's weights.
    std::uint64_t vision_encode_bytes = 0;
    std::uint64_t vision_owned_bytes  = 0;
    // A drafter's rings, table, features and workspace (its weights are the model's).
    std::uint64_t drafter_bytes = 0;
    ExpertDeliveryMemory experts;
};

// A sequence's checkpoint: its rings (sliding-window caches and convolution histories), kept in the
// executor's checkpoint storage of that sequence, and its position, which restore() resumes from.
struct SequenceSnapshot {
    std::uint32_t position = 0;
};

// A sequence's state in host memory, laid out as image_bytes() sizes it: its rings, its
// full-attention rows below `position` and a drafter's ring slots of the positions before it.
struct SequenceImage {
    std::uint32_t position = 0;
};

// An image's rows in a prompt call: the call's rows from `row` on, as many as `embeddings` holds
// (BF16 [rows][hidden] in host memory, a slice of the span encode_image() wrote).
struct ImageRows {
    std::uint32_t row = 0;
    std::span<const std::uint16_t> embeddings;
};

class Executor {
public:
    // Logit rows forward() may leave at once.
    static constexpr std::uint32_t kMaxLogitRows = 256;
    // Raw inputs each convolution site keeps per sequence.
    static constexpr int kHistory = 32;

    Executor(const Model& model, DeviceContext& device, ExecutorOptions options);
    ~Executor();
    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

    [[nodiscard]] const ExecutorOptions& options() const noexcept;
    [[nodiscard]] ExecutorMemory memory() const noexcept;
    // Logit rows of a column: the unpadded vocabulary.
    [[nodiscard]] std::uint32_t vocab() const noexcept;
    void reset(std::uint32_t sequence);
    [[nodiscard]] std::uint32_t position(std::uint32_t sequence) const;
    // Growing caches: the device bytes `sequence`'s full-attention rows still need backed before a
    // call reaches position `end` (0 when they are, or the caches do not grow), and the bytes the
    // device can still back for them. Without an elastic expert cache a call whose growth exceeds
    // the room throws KvExhausted; the scheduler checks first and preempts a sequence instead.
    [[nodiscard]] std::uint64_t kv_growth(std::uint32_t sequence, std::uint32_t end) const;
    [[nodiscard]] std::uint64_t kv_room() const;
    // A sequence's full-attention bytes at most: what one sequence at the full context backs.
    [[nodiscard]] std::uint64_t kv_sequence_bytes() const noexcept;
    // Loads the kernels every route reaches, and captures the decode graphs.
    void warm_up();
    void check_experts() const;
    [[nodiscard]] ExpertDeliveryStats expert_stats() const;
    [[nodiscard]] ExpertRouteCounts expert_totals() const;
    [[nodiscard]] std::string expert_cpu_description() const;
    // Runs up to prefill_chunk tokens at the sequence's next positions and leaves the logits of the
    // last `logit_rows` (at most kMaxLogitRows) in logits(). `images` (in row order) put encoded
    // image rows in place of their tokens' embeddings.
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows, std::span<const ImageRows> images = {});
    // Branches of a prompt in one pass, each continuing the sequence from its position: branch b
    // runs `lengths[b]` of `tokens` (in branch order) at the positions after the sequence's,
    // seeing the sequence's keys and its own. Leaves the logits of each branch's last token in
    // logits(), one column per branch. The sequence keeps its position and state.
    void forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                          std::span<const std::uint32_t> lengths);
    // A prompt's last chunk and the first branches of a read in one pass (Flint's contract).
    void forward_prompt_branches(std::uint32_t sequence, std::span<const std::int32_t> prompt,
                                 std::span<const std::int32_t> tokens,
                                 std::span<const std::uint32_t> lengths);
    // One token of each of `sequences` (distinct) at its next position; logits one column each.
    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // BF16 [vocab, rows], ready on head_stream().
    [[nodiscard]] Tensor logits(std::uint32_t rows) const;

    [[nodiscard]] std::uint32_t draft_tokens() const noexcept;
    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens,
                std::span<const std::uint32_t> widths = {});
    void commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> kept);
    // The drafter's greedy drafts of the `k` positions after each sequence's anchor and each
    // draft's probability under it, sequence-major; returns once they are on the host. DSpark runs
    // one block per sequence.
    void draft(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> anchors,
               std::uint32_t k, std::span<std::int32_t> drafts, std::span<float> confidence);
    // DSpark: the drafts its block proposes a round, its block less one (0 without the drafter);
    // and the drafter's greedy drafts of the drafts.size() positions after the sequence's anchor
    // (at most that many) with their probabilities. A block needs a committed token before it.
    [[nodiscard]] std::uint32_t dspark_block() const noexcept;
    void draft(std::uint32_t sequence, std::int32_t anchor, std::span<std::int32_t> drafts,
               std::span<float> confidence);
    [[nodiscard]] cudaStream_t head_stream() const noexcept;
    // Prompt chunks beside decoding (NINFER_INKLING_OVERLAP=0: off): between side_begin() and
    // side_end() the executor's calls run on a second stream with their own workspace and staging,
    // after everything enqueued before side_begin(); side_idle() tells when that work is done and
    // side_join() orders later calls after it. Only one side call runs at a time, and it must not
    // touch a sequence the main stream uses until joined.
    [[nodiscard]] bool overlap() const noexcept;
    void side_begin();
    void side_end();
    [[nodiscard]] bool side_idle() const;
    void side_join();
    void snapshot(std::uint32_t sequence, SequenceSnapshot& out);
    void restore(std::uint32_t sequence, const SequenceSnapshot& from);
    // Takes the sequence back `tokens` positions: false, changing nothing, when its rings no
    // longer hold what the earlier position needs (not before its last prompt, checkpoint or
    // image, and within the slots its rings keep past the window, the convolutions' taps and a
    // round's drafts).
    bool rewind(std::uint32_t sequence, std::uint32_t tokens);
    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const;
    SequenceImage save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes);
    // save_image on a copy stream of its own, beside the work queued after it (a write-through
    // of a finished conversation); the sequence's next state write waits for it. copy_stream()
    // is that stream (null until the first such copy).
    SequenceImage save_image_async(std::uint32_t sequence, const SequenceSnapshot* at,
                                   std::span<std::byte* const> pages, std::size_t page_bytes);
    [[nodiscard]] cudaStream_t copy_stream() const noexcept;
    void load_image(std::uint32_t sequence, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes);

    // Whether prompt calls may carry images.
    [[nodiscard]] bool vision() const noexcept;
    // The span rows of an image of grid_w x grid_h patches (BF16 [grid_h * grid_w][patch
    // features], row-major) into `out` (BF16 [rows][hidden], host memory); returns the rows.
    std::uint32_t encode_image(std::span<const std::uint16_t> patches, int grid_w, int grid_h,
                               std::span<std::uint16_t> out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::inkling2
