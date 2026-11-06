#include "models/qwen3_5/session/internal.h"

#include "models/common/frontend/media.h"
#include "runtime/contract/placement.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

constexpr std::uint64_t kStateSentinel = 0x4E494E4645522131ULL; // "NINFER!1"

// ArchiveKey is a std::array alias; the link header pulls it in. The context alias below keeps
// the session-side mapping readable against the directory types it stages from.
namespace context = ninfer::models::context;

// The text pool's pages carry the digest chain; the backend pool's pages are a derived view of
// the same tokens, so both kinds key from the same boundary digests and stay dedupable. Each
// kind is sized by its own frontier; a session without a backend pool registers no kind-1 keys.
std::array<std::vector<ArchiveKey>, kArchivePageKinds>
page_keys(const qwen::PrefixShortlistDigests& digests, std::uint32_t text_frontier,
          std::uint32_t backend_frontier, bool backend) {
    std::array<std::vector<ArchiveKey>, kArchivePageKinds> keys;
    const auto boundaries = [&digests](std::uint32_t frontier) {
        std::vector<ArchiveKey> result;
        const std::uint32_t pages = context::kv_pages_for_frontier(frontier);
        result.reserve(pages);
        for (std::uint32_t i = 0; i < pages; ++i) {
            result.push_back(digests.at(std::min<std::uint32_t>((i + 1) * kPagedKVPageSize,
                                                                frontier)));
        }
        return result;
    };
    keys[0] = boundaries(text_frontier);
    if (backend) keys[1] = boundaries(backend_frontier);
    return keys;
}

void put_u32(std::vector<std::byte>& out, std::uint32_t value) {
    for (std::size_t i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::byte>(value >> (8U * i)));
    }
}

// identity image: [u32 positions bytes][positions image][u32 digests bytes][digests image]
//                 [u64 ledger tokens][TokenId x tokens]
// state image:    [ContinuationMetadata][u64 role][u64 sentinel]

context::ContextArchiveLink::CheckpointEntry
archive_entry(const CheckpointState& record) {
    const auto& identity = record.snapshot.identity();
    context::ContextArchiveLink::CheckpointEntry entry;
    entry.frontier         = record.frontier();
    entry.backend_frontier = record.backend_frontier();
    entry.prefix_key       = record.key.digests;
    entry.page_keys        = page_keys(identity.digests, record.frontier(),
                                       record.backend_frontier(),
                                       record.backend_prefix() != nullptr);
    auto positions = identity.prefix_identity.image();
    put_u32(entry.identity, static_cast<std::uint32_t>(positions.size()));
    entry.identity.insert(entry.identity.end(), positions.begin(), positions.end());
    auto digests_image = identity.digests.image();
    put_u32(entry.identity, static_cast<std::uint32_t>(digests_image.size()));
    entry.identity.insert(entry.identity.end(), digests_image.begin(), digests_image.end());
    const auto tokens = static_cast<std::uint64_t>(identity.ledger.size());
    for (std::size_t i = 0; i < 8; ++i) {
        entry.identity.push_back(static_cast<std::byte>(tokens >> (8U * i)));
    }
    for (const auto token : identity.ledger) {
        for (std::size_t i = 0; i < 4; ++i) {
            entry.identity.push_back(static_cast<std::byte>(
                static_cast<std::uint32_t>(token) >> (8U * i)));
        }
    }
    entry.state.resize(sizeof(ContinuationMetadata) + 16);
    std::memcpy(entry.state.data(), &record.snapshot.continuation().metadata,
                sizeof(ContinuationMetadata));
    const auto role = static_cast<std::uint64_t>(record.role);
    std::memcpy(entry.state.data() + sizeof(ContinuationMetadata), &role, sizeof(role));
    std::memcpy(entry.state.data() + sizeof(ContinuationMetadata) + 8, &kStateSentinel,
                sizeof(kStateSentinel));
    entry.retain = record.role == runtime::CheckpointRole::LongAnchor;
    const auto& digests = identity.digests;
    entry.lineage       = [&digests](std::uint32_t frontier) -> ArchiveKey {
        if (frontier == 0 || frontier > digests.size()) {
            // Outside this chain: a key that cannot match any checkpoint of this lineage.
            return {0x9E3779B97F4A7C15ULL ^ frontier, kArchiveIdentityScheme};
        }
        return digests.at(frontier);
    };
    return entry;
}

} // namespace

void SessionImpl::archive_register(CheckpointHandle handle, const CheckpointState& record) {
    if (!archive_ || !archive_->enabled()) return;
    if (record.frontier() == 0 || !record.history().text.valid()) return;
    try {
        // A slot's earlier pending registration (if any) is replaced by the newer checkpoint.
        const auto existing = archive_ids.find(handle.index);
        if (existing != archive_ids.end()) archive_->discard_checkpoint(existing->second);
        // Pin-budget eviction: see qwen4_exp archive_register (archive_save_pins caps the
        // plain-release intervention; drop the oldest pending save beyond the cap). The
        // incoming slot is exempt.
        if (archive_ids.size() >= kArchiveSavePins) {
            auto evict = archive_ids.begin();
            if (evict->first == handle.index) evict = std::next(evict);
            if (evict != archive_ids.end()) {
                archive_->discard_checkpoint(evict->second);
                archive_ids.erase(evict);
            }
        }
        const auto id = next_archive_id++;
        archive_->register_checkpoint(id, archive_entry(record));
        archive_ids[handle.index] = id;
        archive_dirty             = true;
    } catch (...) {
        // The archive is an accelerator; registration failures must not fail a capture.
        archive_ids.erase(handle.index);
    }
}

void SessionImpl::archive_discard(CheckpointHandle handle) noexcept {
    if (!archive_) return;
    const auto found = archive_ids.find(handle.index);
    if (found == archive_ids.end()) return;
    archive_->discard_checkpoint(found->second);
    archive_ids.erase(found);
}

std::span<const CheckpointHandle> SessionImpl::archive_save_pins(
    std::span<const CheckpointHandle> seed) const {
    if (!archive_enabled() || archive_ids.empty()) return {};
    if (seed.size() >= kArchiveSavePins) return {};
    const auto room = kArchiveSavePins - seed.size();
    archive_pin_cache.clear();
    for (const auto [index, id] : archive_ids) {
        const auto handle = checkpoints.handle(index);
        if (!checkpoints.valid(handle)) continue;
        if (std::find(seed.begin(), seed.end(), handle) != seed.end()) continue;
        archive_pin_cache.push_back(handle);
        if (archive_pin_cache.size() == room) break;
    }
    return archive_pin_cache;
}

void SessionImpl::archive_flush() {
    if (!archive_ || !archive_->enabled() || !archive_dirty) return;
    if (!archive_host_arena.data) return;
    bool work_remains = false;
    for (auto entry = archive_ids.begin(); entry != archive_ids.end();) {
        const auto id     = entry->second;
        const auto handle = checkpoints.handle(entry->first);
        if (!checkpoints.valid(handle)) {
            // Retired before its save committed: nothing was archived, nothing to unreference.
            archive_->discard_checkpoint(id);
            entry = archive_ids.erase(entry);
            continue;
        }
        const auto& record = checkpoint(handle);
        // The save commits only from a fully Host-resident checkpoint; device-resident pages
        // have no stable CPU bytes to stage yet. A page that lost both replicas can never
        // stage again - abandon the save instead of retrying until retirement.
        const auto scan = [&](const context::HistoryDirectory& histories,
                              const context::PageDirectory& page_directory,
                              const context::HistoryPrefix& prefix, bool& resident,
                              bool& abandonable) {
            const std::uint32_t count = context::kv_pages_for_frontier(prefix.frontier());
            if (histories.mapped_pages(prefix.history()) < count) {
                abandonable = true;
                return;
            }
            for (std::uint32_t i = 0; i < count; ++i) {
                const auto page = histories.logical_page(prefix.history(), i);
                if (page_directory.host_replica_current(page)) continue;
                resident = false;
                if (!page_directory.valid(page) || !page_directory.device_resident(page)) {
                    abandonable = true;
                    return;
                }
            }
        };
        const auto& text_prefix   = record.text_prefix();
        bool resident             = state_store->host_resident(record.continuation().image);
        bool abandonable          = false;
        scan(*text_kv_histories, *text_kv_pages, text_prefix, resident, abandonable);
        if (!abandonable) {
            if (const auto* backend = record.backend_prefix())
                scan(*backend_kv_histories, *backend_kv_pages, *backend, resident, abandonable);
        }
        if (abandonable) {
            archive_->discard_checkpoint(id);
            entry = archive_ids.erase(entry);
            continue;
        }
        if (!resident) {
            work_remains = true;
            ++entry;
            continue;
        }
        std::vector<context::ContextArchiveLink::StagedPage> staged;
        staged.reserve(2 * context::kv_pages_for_frontier(record.frontier()) +
                       (record.backend_prefix()
                            ? 2 * context::kv_pages_for_frontier(record.backend_frontier())
                            : 0) +
                       2);
        const auto& digests = record.snapshot.identity().digests;
        const auto stage    = [&](const context::HistoryDirectory& histories,
                               const context::PageDirectory& page_directory,
                               const context::HistoryPrefix& prefix) {
            const std::uint32_t count = context::kv_pages_for_frontier(prefix.frontier());
            for (std::uint32_t i = 0; i < count; ++i) {
                const auto page = histories.logical_page(prefix.history(), i);
                const auto key  = digests.at(
                    std::min<std::uint32_t>((i + 1) * kPagedKVPageSize, prefix.frontier()));
                for (const auto& part : page_directory.layout().parts) {
                    const auto binding = page_directory.host_binding(page, part.rank);
                    staged.push_back(
                        {key, archive_host_arena.subspan(static_cast<std::size_t>(binding.offset),
                                                         binding.bytes)});
                }
            }
        };
        stage(*text_kv_histories, *text_kv_pages, text_prefix);
        if (const auto* backend = record.backend_prefix()) {
            stage(*backend_kv_histories, *backend_kv_pages, *backend);
        }
        try {
            const auto result = archive_->flush(id, staged);
            if (result == context::ContextArchiveLink::FlushResult::Committed) {
                entry = archive_ids.erase(entry);
                continue;
            }
            // Incomplete (host tier not ready) or Declined (budget/staging): retry later.
            work_remains = true;
        } catch (...) {
            work_remains = true;
        }
        ++entry;
    }
    archive_dirty = work_remains;
}

std::optional<context::ContextArchiveLink::Candidate>
SessionImpl::archive_find(std::uint32_t min_frontier, std::uint32_t max_frontier,
                          const qwen::PrefixShortlistDigests& digests) const {
    if (!archive_ || !archive_->enabled()) return std::nullopt;
    return archive_->find(min_frontier, max_frontier,
                          [&digests](std::uint32_t frontier) -> ArchiveKey {
                              if (frontier == 0 || frontier > digests.size()) return {};
                              return digests.at(frontier);
                          });
}

std::shared_ptr<ninfer::DiskContextArchive::Prefetch>
SessionImpl::archive_prefetch(const context::ContextArchiveLink::Candidate& candidate) const {
    return archive_ ? archive_->prefetch(candidate) : nullptr;
}

} // namespace ninfer::models::qwen3_5::detail
