#pragma once

// Session-owned seam between the context directory (checkpoint decisions) and the
// DiskContextArchive (NVMe bytes). The session maps each immutable CheckpointState onto an
// archive SaveRequest — frontier, digest-chain prefix key, per-page digest keys and the
// identity/state images — and stages page bytes from the Host tier once a demotion or pause
// migration has made the checkpoint's pages Host-resident.
//
// Wedge rules (2026-10-30/11-01 incidents): every call here is made from the single control
// thread and returns immediately or performs bounded work. The archive's write budget gates
// saves — a declined save is skipped and retried from a later flush, never queued, and the
// demotion path never waits on archive progress. No filesystem access happens on this thread;
// the archive's own writer and reader threads own all IO.

#include "core/disk_context_archive.h"
#include "core/memory.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::models::context {

class ContextArchiveLink {
public:
    // Bytes of one checkpoint's identity and state images, plus the digest-derived keys the
    // archive indexes. Page keys are the rolling prefix digests at page boundaries, so shared
    // prefixes dedup and keys stay stable across restarts. `retain` keeps the manifest exempt
    // from supersede (restore points whose whole-state image is valid only at their frontier).
    // `lineage` maps an ancestor frontier to that ancestor's prefix key from this checkpoint's
    // own digest chain; it drives supersede-on-commit and must return a distinct key for any
    // frontier outside the chain.
    struct CheckpointEntry {
        std::uint32_t frontier         = 0;
        std::uint32_t backend_frontier = 0;
        ArchiveKey prefix_key{};
        std::array<std::vector<ArchiveKey>, kArchivePageKinds> page_keys;
        std::vector<std::byte> identity;
        std::vector<std::byte> state;
        bool retain = false;
        using LineageKeyAt = std::function<ArchiveKey(std::uint32_t)>;
        LineageKeyAt lineage;
    };

    // One Host-resident page offered to the save: the registered key it belongs to and its CPU
    // bytes (the full Host record, matching the page kind's configured byte count).
    struct StagedPage {
        ArchiveKey key;
        ConstMemorySpan bytes;
    };

    struct Options {
        std::filesystem::path directory;
        std::uint64_t capacity_bytes         = 0;
        std::uint64_t write_bytes_per_second = 0;
        std::array<std::size_t, kArchivePageKinds> page_bytes{};
        std::uint64_t signature              = 0;
        std::size_t staging_bytes            = 0;
    };

    enum class FlushResult { Committed, Incomplete, Declined };

    ContextArchiveLink() noexcept = default;
    explicit ContextArchiveLink(Options options);
    ~ContextArchiveLink();

    ContextArchiveLink(const ContextArchiveLink&)            = delete;
    ContextArchiveLink& operator=(const ContextArchiveLink&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return archive_ != nullptr; }

    // Registers one published checkpoint and arms its save. `entry.lineage` must be set.
    // Cheap: metadata and image bytes only; page staging happens at flush. A later registration
    // under the same id replaces the earlier one (a lane's newer checkpoint supersedes it before
    // it ever committed).
    void register_checkpoint(std::uint64_t id, CheckpointEntry entry);

    // Drops a registration whose checkpoint was retired before its save committed. Nothing was
    // archived, so nothing to unreference.
    void discard_checkpoint(std::uint64_t id);

    // Attempts the armed save for one checkpoint. The caller stages pages only after the
    // directory reports every page of the checkpoint Host-resident, and passes the checkpoint's
    // full page set; pages already in the archive may be omitted. Incomplete: the offer does not
    // cover the pages the archive is missing. Declined: write budget or staging exhausted —
    // retry from a later flush. Committed: durable handoff done (writer thread owns the IO).
    [[nodiscard]] FlushResult flush(std::uint64_t id, std::span<const StagedPage> staged);

    [[nodiscard]] DiskArchiveStats stats() const;

    // Blocks until all committed saves are durable. Orderly shutdown and tests only.
    void flush_all();

    // Archive lookup surface for the resume side. Longest archived checkpoint with
    // min_frontier <= frontier <= max_frontier whose prefix key matches key_at(frontier).
    using KeyAt     = DiskContextArchive::KeyAt;
    using Candidate = DiskContextArchive::Candidate;
    [[nodiscard]] std::optional<Candidate>
    find(std::uint32_t min_frontier, std::uint32_t max_frontier, const KeyAt& key_at) const;
    [[nodiscard]] std::shared_ptr<DiskContextArchive::Prefetch>
    prefetch(const Candidate& candidate);

private:
    struct Pending {
        CheckpointEntry entry;
        bool armed = false;
    };

    Options options_;
    std::unique_ptr<DiskContextArchive> archive_;
    std::vector<std::byte> staging_;
    std::unordered_map<std::uint64_t, Pending> pending_;
};

} // namespace ninfer::models::context
