#include "models/common/context/disk_archive_link.h"

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ninfer::models::context {

namespace {

constexpr std::uint64_t kDefaultStagingBytes = 512ULL << 20;

} // namespace

ContextArchiveLink::ContextArchiveLink(Options options) : options_(std::move(options)) {
    if (options_.directory.empty()) {
        throw std::invalid_argument("context archive requires a directory");
    }
    if (options_.capacity_bytes == 0) {
        throw std::invalid_argument("context archive requires a positive capacity");
    }
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        if (options_.page_bytes[kind] != 0 && options_.page_bytes[kind] < 64) {
            throw std::invalid_argument("context archive page size is implausibly small");
        }
    }
    if (options_.staging_bytes == 0) options_.staging_bytes = kDefaultStagingBytes;
    staging_.resize(options_.staging_bytes);
    DiskArchiveOptions archive;
    archive.directory              = options_.directory;
    archive.capacity_bytes         = options_.capacity_bytes;
    archive.write_bytes_per_second = options_.write_bytes_per_second;
    archive.page_bytes             = options_.page_bytes;
    archive.signature              = options_.signature;
    archive_ = std::make_unique<DiskContextArchive>(
        std::move(archive), std::span<std::byte>(staging_.data(), staging_.size()));
}

ContextArchiveLink::~ContextArchiveLink() = default;

void ContextArchiveLink::register_checkpoint(std::uint64_t id, CheckpointEntry entry) {
    if (!archive_) return;
    if (entry.frontier == 0) throw std::invalid_argument("context archive entry has no frontier");
    if (!entry.lineage) {
        throw std::invalid_argument("context archive entry has no lineage mapping");
    }
    if (entry.page_keys[0].empty() && options_.page_bytes[0] != 0) {
        throw std::invalid_argument("context archive entry has no pages");
    }
    auto& pending = pending_[id];
    pending.entry = std::move(entry);
    pending.armed = true;
}

void ContextArchiveLink::discard_checkpoint(std::uint64_t id) { pending_.erase(id); }

ContextArchiveLink::FlushResult ContextArchiveLink::flush(std::uint64_t id,
                                                          std::span<const StagedPage> staged,
                                                          std::span<const ConstMemorySpan> state_parts) {
    if (!archive_) return FlushResult::Declined;
    const auto found = pending_.find(id);
    if (found == pending_.end() || !found->second.armed) return FlushResult::Declined;
    auto& entry = found->second.entry;

    // Same checkpoint already durable (superseded save, duplicate flush): retire the pending
    // entry; begin_save declining on an exact frontier/prefix match means it is present.
    DiskContextArchive::SaveRequest request;
    request.frontier         = entry.frontier;
    request.backend_frontier = entry.backend_frontier;
    request.prefix_key       = entry.prefix_key;
    request.page_keys        = entry.page_keys;
    request.identity_bytes   = entry.identity.size();
    std::size_t state_image_bytes = 0;
    for (const auto& part : state_parts) state_image_bytes += part.bytes;
    request.state_bytes      = entry.state.size() + state_image_bytes;
    request.retain           = entry.retain;
    const auto& lineage      = entry.lineage;
    const auto before        = archive_->stats();

    auto job = archive_->begin_save(std::move(request), lineage);
    if (!job) {
        // Distinguish "budget/staging declined" (retry later) from "already archived" (done):
        // the archive counts the former in dedicated skip counters.
        const auto after = archive_->stats();
        if (after.saves_skipped_budget == before.saves_skipped_budget &&
            after.saves_skipped_staging == before.saves_skipped_staging) {
            pending_.erase(found);
            return FlushResult::Committed;
        }
        return FlushResult::Declined;
    }

    // Per-kind offered maps: pools share the digest table (main and MTP frontiers overlap),
    // so one flat key map would let a kind-0 page satisfy a kind-1 lookup with the wrong
    // byte size (observed live: bytes_found=1 bytes_match=0 on every kind-1 page).
    std::array<std::unordered_map<ArchiveKey, ConstMemorySpan, ArchiveKeyHash>, 2> offered;
    for (auto& map : offered) map.reserve(staged.size());
    for (const auto& page : staged) offered[page.kind].emplace(page.key, page.bytes);

    for (std::size_t kind = 0; kind < 2; ++kind) {
        const auto page_bytes = options_.page_bytes[kind];
        const auto missing    = job->missing(kind);
        for (std::size_t i = 0; i < missing.size(); ++i) {
            const auto& key = entry.page_keys[kind][missing[i]];
            const auto& map = offered[kind];
            const auto bytes = map.find(key);
            if (bytes == map.end() || bytes->second.bytes != page_bytes) {
                // Host tier cannot supply this page yet; keep the save armed.
                static const bool trace = [] {
                    const char* v = ::getenv("NINFER_ARCHIVE_TRACE");
                    return v && v[0] == '1';
                }();
                if (trace)
                    std::fprintf(stderr,
                                 "archive-link: INCOMPLETE kind=%zu missing_idx=%zu "
                                 "entry_key=%016lx/%016lx bytes_found=%d bytes_match=%d "
                                 "offered=%zu\n",
                                 kind, i, static_cast<unsigned long long>(key[0]),
                                 static_cast<unsigned long long>(key[1]),
                                 bytes != map.end() ? 1 : 0,
                                 bytes != map.end()
                                     ? static_cast<int>(bytes->second.bytes == page_bytes)
                                     : -1,
                                 map.size());
                return FlushResult::Incomplete;
            }
            auto destination = job->page(kind, i);
            std::memcpy(destination.data(), bytes->second.data, destination.size());
        }
    }
    auto identity = job->identity();
    if (identity.size() != entry.identity.size()) {
        throw std::logic_error("context archive identity staging size differs");
    }
    std::memcpy(identity.data(), entry.identity.data(), identity.size());
    auto state = job->state();
    if (state.size() != entry.state.size() + state_image_bytes) {
        throw std::logic_error("context archive state staging size differs");
    }
    std::memcpy(state.data(), entry.state.data(), entry.state.size());
    auto cursor = state.data() + entry.state.size();
    for (const auto& part : state_parts) {
        std::memcpy(cursor, part.data, part.bytes);
        cursor += part.bytes;
    }
    archive_->commit(std::move(*job));
    pending_.erase(found);
    return FlushResult::Committed;
}

DiskArchiveStats ContextArchiveLink::stats() const {
    return archive_ ? archive_->stats() : DiskArchiveStats{};
}

void ContextArchiveLink::flush_all() {
    if (archive_) archive_->flush();
}

std::optional<DiskContextArchive::Candidate>
ContextArchiveLink::find(std::uint32_t min_frontier, std::uint32_t max_frontier,
                         const KeyAt& key_at) const {
    return archive_ ? archive_->find(min_frontier, max_frontier, key_at) : std::nullopt;
}

std::shared_ptr<DiskContextArchive::Prefetch>
ContextArchiveLink::prefetch(const Candidate& candidate) {
    return archive_ ? archive_->prefetch(candidate) : nullptr;
}

} // namespace ninfer::models::context
