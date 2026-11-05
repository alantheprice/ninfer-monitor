#include "models/qwen4_exp/session/internal.h"

#include "core/paged_kv_geometry.h"
#include "models/common/context/work.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::detail {
namespace {

constexpr std::uint64_t kStateSentinel = 0x4E494E4645522132ULL; // "NINFER!2"

// The main pool's pages carry the digest chain; the MTP pool's pages are a derived view of
// the same tokens, so both kinds key from the same boundary digests and stay dedupable. Each
// kind is sized by its own frontier; a session without an MTP pool registers no kind-1 keys.
std::array<std::vector<ArchiveKey>, kArchivePageKinds>
page_keys(const qwen::PrefixShortlistDigests& digests, std::uint32_t main_frontier,
          std::uint32_t mtp_frontier, bool mtp) {
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
    keys[0] = boundaries(main_frontier);
    if (mtp) keys[1] = boundaries(mtp_frontier);
    return keys;
}

ArchiveKey key_at(const qwen::PrefixShortlistDigests& digests, std::uint32_t frontier) {
    if (frontier == 0 || frontier > digests.size()) {
        // Outside this chain: a key that cannot match any checkpoint of this lineage.
        return {0x9E3779B97F4A7C15ULL ^ frontier, kArchiveIdentityScheme};
    }
    return digests.at(frontier);
}

void put_u32(std::vector<std::byte>& out, std::uint32_t value) {
    for (std::size_t i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::byte>(value >> (8U * i)));
    }
}

void put_u64(std::vector<std::byte>& out, std::uint64_t value) {
    for (std::size_t i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::byte>(value >> (8U * i)));
    }
}

// identity image: [u32 positions bytes][positions image][u32 digests bytes][digests image]
//                 [u64 tag][u64 token count][TokenId x tokens]
// state image:    [ContinuationMetadata][u64 role][u64 sentinel]

context::ContextArchiveLink::CheckpointEntry archive_entry(const Checkpoint& record) {
    const auto& identity = record.snapshot.identity();
    const auto& metadata = record.snapshot.continuation().metadata;
    const bool mtp       = metadata.mtp_frontier.has_value() && *metadata.mtp_frontier > 0;
    context::ContextArchiveLink::CheckpointEntry entry;
    entry.frontier         = record.frontier();
    entry.backend_frontier = mtp ? *metadata.mtp_frontier : 0;
    entry.prefix_key       = key_at(identity.digests, record.frontier());
    entry.page_keys        = page_keys(identity.digests, record.frontier(),
                                       entry.backend_frontier, mtp);
    auto positions = identity.positions.image();
    put_u32(entry.identity, static_cast<std::uint32_t>(positions.size()));
    entry.identity.insert(entry.identity.end(), positions.begin(), positions.end());
    auto digests_image = identity.digests.image();
    put_u32(entry.identity, static_cast<std::uint32_t>(digests_image.size()));
    entry.identity.insert(entry.identity.end(), digests_image.begin(), digests_image.end());
    put_u64(entry.identity, identity.tag);
    put_u64(entry.identity, identity.tokens.size());
    for (const auto token : identity.tokens) {
        put_u32(entry.identity, static_cast<std::uint32_t>(token));
    }
    entry.state.resize(sizeof(ContinuationMetadata) + 16);
    std::memcpy(entry.state.data(), &metadata, sizeof(ContinuationMetadata));
    const auto role = static_cast<std::uint64_t>(record.role);
    std::memcpy(entry.state.data() + sizeof(ContinuationMetadata), &role, sizeof(role));
    std::memcpy(entry.state.data() + sizeof(ContinuationMetadata) + 8, &kStateSentinel,
                sizeof(kStateSentinel));
    entry.retain = record.role == runtime::CheckpointRole::LongAnchor;
    const auto& digests = identity.digests;
    entry.lineage       = [&digests](std::uint32_t frontier) -> ArchiveKey {
        return key_at(digests, frontier);
    };
    return entry;
}

} // namespace

void SessionImpl::archive_register(CheckpointHandle handle, const Checkpoint& record) {
    if (!archive_ || !archive_->enabled()) return;
    if (record.frontier() == 0 || record.snapshot.prefixes().empty() ||
        !record.snapshot.prefix(0).valid())
        return;
    try {
        // A slot's earlier pending registration (if any) is replaced by the newer checkpoint.
        const auto existing = archive_ids.find(handle.index);
        if (existing != archive_ids.end()) archive_->discard_checkpoint(existing->second);
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

void SessionImpl::archive_flush() {
    if (!archive_ || !archive_->enabled() || !archive_dirty) return;
    if (!archive_host_arena.data) return;
    if (archive_ids.empty()) { archive_dirty = false; return; }
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
        const auto& record = checkpoints.at(handle);
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
        bool resident    = directory.states().host_resident(record.snapshot.continuation().image);
        bool abandonable = false;
        const auto pool_count = directory.pool_count();
        for (std::uint16_t pool = 0; pool < pool_count && !abandonable; ++pool) {
            const auto& prefix = record.snapshot.prefix(pool);
            if (prefix.valid() && prefix.frontier() != 0)
                scan(directory.pool(pool).histories(), directory.pool(pool).pages(), prefix,
                     resident, abandonable);
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
        const auto& digests = record.snapshot.identity().digests;
        std::vector<context::ContextArchiveLink::StagedPage> staged;
        { // One staging slot per page per rank part across pools.
            std::size_t slots = 0;
            for (std::uint16_t pool = 0; pool < pool_count; ++pool) {
                const auto& prefix = record.snapshot.prefix(pool);
                if (prefix.valid() && prefix.frontier() != 0)
                    slots += std::size_t{context::kv_pages_for_frontier(prefix.frontier())} *
                             directory.pool(pool).pages().layout().parts.size();
            }
            staged.reserve(slots);
        }
        for (std::uint16_t pool = 0; pool < pool_count; ++pool) {
            const auto& prefix = record.snapshot.prefix(pool);
            if (!prefix.valid() || prefix.frontier() == 0) continue;
            const std::uint32_t count = context::kv_pages_for_frontier(prefix.frontier());
            for (std::uint32_t i = 0; i < count; ++i) {
                const auto page = directory.pool(pool).histories().logical_page(
                    prefix.history(), i);
                const auto key =
                    digests.at(std::min<std::uint32_t>((i + 1) * kPagedKVPageSize,
                                                       prefix.frontier()));
                for (const auto& part : directory.pool(pool).pages().layout().parts) {
                    const auto binding =
                        directory.pool(pool).pages().host_binding(page, part.rank);
                    staged.push_back(
                        {key, archive_host_arena.subspan(static_cast<std::size_t>(binding.offset),
                                                         binding.bytes)});
                }
            }
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
            // The archive is an accelerator; a failed save retries on a later sweep.
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
                              return key_at(digests, frontier);
                          });
}

std::shared_ptr<ninfer::DiskContextArchive::Prefetch>
SessionImpl::archive_prefetch(const context::ContextArchiveLink::Candidate& candidate) const {
    return archive_ ? archive_->prefetch(candidate) : nullptr;
}

} // namespace ninfer::models::qwen4_exp::detail
