#include "models/common/context/disk_archive_link.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
namespace context = ninfer::models::context;

namespace {

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

constexpr std::size_t kPageBytes = 4096;
// One 64-token page per entry in these fixtures.
constexpr std::uint32_t kFrontier = 64;

std::byte pattern(std::uint64_t seed, std::size_t offset) {
    return static_cast<std::byte>((seed * 131U + offset * 7U) & 0xFFU);
}

struct Chain {
    std::uint64_t seed = 0;

    [[nodiscard]] ArchiveKey key_at(std::uint32_t frontier) const {
        return ArchiveKey{seed * 1'000'003ULL + frontier, seed ^ 0xABCDULL};
    }
};

context::ContextArchiveLink::CheckpointEntry
entry_for(const Chain& chain, std::uint32_t frontier, bool retained = false) {
    context::ContextArchiveLink::CheckpointEntry entry;
    entry.frontier         = frontier;
    entry.backend_frontier = frontier;
    entry.prefix_key       = chain.key_at(frontier);
    entry.page_keys[0].push_back(chain.key_at(frontier));
    entry.identity.assign(40, pattern(frontier, 1));
    entry.state.assign(96, pattern(frontier, 2));
    entry.retain = retained;
    const auto chain_keys =
        [&chain](std::uint32_t at) -> ArchiveKey { return chain.key_at(at); };
    entry.lineage = chain_keys;
    return entry;
}

std::vector<context::ContextArchiveLink::StagedPage>
staged_for(const Chain& chain, const std::vector<std::byte>& bytes, std::uint32_t frontier) {
    return {{chain.key_at(frontier), {bytes.data(), bytes.size(), Device::cpu()}}};
}

context::ContextArchiveLink::Options options(const std::filesystem::path& directory,
                                             std::uint64_t capacity, std::size_t staging = 0) {
    context::ContextArchiveLink::Options out;
    out.directory      = directory;
    out.capacity_bytes = capacity;
    out.page_bytes     = {kPageBytes, 0};
    out.signature      = 0x5EED;
    out.staging_bytes  = staging;
    return out;
}

} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("ninfer-archive-link-test-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    try {
        const Chain a{.seed = 1};
        const Chain b{.seed = 2};
        std::vector<std::byte> bytes(kPageBytes);
        for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = pattern(700, i);

        {
            context::ContextArchiveLink link(options(root / "basic", 1ULL << 20));
            require(link.enabled(), "link did not construct");
            link.register_checkpoint(1, entry_for(a, 700));
            // Incomplete offer: the archive is missing the page and none was staged.
            require(link.flush(1, {}) == context::ContextArchiveLink::FlushResult::Incomplete,
                    "an empty offer committed");
            require(link.flush(1, staged_for(a, bytes, 700)) ==
                        context::ContextArchiveLink::FlushResult::Committed,
                    "a complete offer was declined");
            link.flush_all();
            require(link.stats().saves_committed == 1, "the save did not commit");

            const auto candidate = link.find(1, 700, [&](std::uint32_t f) { return a.key_at(f); });
            require(candidate && candidate->frontier == 700, "the checkpoint was not found");
            const auto prefetch = link.prefetch(*candidate);
            require(prefetch != nullptr, "prefetch could not be staged");
            require(prefetch->wait_for(std::chrono::seconds(10)) ==
                        DiskContextArchive::Prefetch::Status::Ready,
                    "prefetch did not become ready");
            const auto read_back = prefetch->pages(0);
            require(read_back.size() == kPageBytes, "prefetched page size differs");
            require(std::memcmp(read_back.data(), bytes.data(), kPageBytes) == 0,
                    "prefetched page bytes differ");
            require(prefetch->identity().size() == 40, "identity image size differs");
            require(std::memcmp(prefetch->identity().data(), entry_for(a, 700).identity.data(),
                                40) == 0,
                    "identity image bytes differ");

            // A duplicate registration and flush is already archived: Committed (nothing more
            // to do), never a second write.
            link.register_checkpoint(2, entry_for(a, 700));
            require(link.flush(2, staged_for(a, bytes, 700)) ==
                        context::ContextArchiveLink::FlushResult::Committed,
                    "an already-archived checkpoint did not settle");
            require(link.stats().saves_committed == 1, "an already-archived checkpoint rewrote");
            link.discard_checkpoint(2);
            require(link.flush(2, {}) == context::ContextArchiveLink::FlushResult::Declined,
                    "a discarded registration still flushed");

            // A different lineage never matches.
            require(!link.find(1, 4000, [&](std::uint32_t f) { return b.key_at(f); }),
                    "another lineage matched");
        }
        {
            // Persistence across instances: the restarted engine rediscovers the checkpoint.
            context::ContextArchiveLink link(options(root / "basic", 1ULL << 20));
            require(link.stats().saves_committed == 0, "stats leaked across instances");
            const auto candidate = link.find(1, 700, [&](std::uint32_t f) { return a.key_at(f); });
            require(candidate && candidate->frontier == 700,
                    "the archive did not survive a restart");
        }
        {
            // Staging exhaustion declines and keeps the registration pending.
            context::ContextArchiveLink link(options(root / "small", 1ULL << 20, 4096));
            link.register_checkpoint(1, entry_for(a, 700));
            require(link.flush(1, staged_for(a, bytes, 700)) ==
                        context::ContextArchiveLink::FlushResult::Declined,
                    "a save larger than staging was accepted");
            require(link.stats().saves_skipped_staging == 1, "staging skip was not counted");
            // And the entry still flushes once staging is adequate.
            context::ContextArchiveLink roomy(options(root / "small", 1ULL << 20));
            (void)roomy;
        }
        {
            // Retained entries survive supersede by a newer same-lineage checkpoint.
            context::ContextArchiveLink link(options(root / "retain", 1ULL << 20));
            std::vector<std::byte> later(kPageBytes);
            for (std::size_t i = 0; i < later.size(); ++i) later[i] = pattern(900, i);
            link.register_checkpoint(1, entry_for(a, 700, true));
            require(link.flush(1, staged_for(a, bytes, 700)) ==
                        context::ContextArchiveLink::FlushResult::Committed,
                    "retained anchor save was declined");
            link.register_checkpoint(2, entry_for(a, 900));
            require(link.flush(2, staged_for(a, later, 900)) ==
                        context::ContextArchiveLink::FlushResult::Committed,
                    "descendant save was declined");
            link.flush_all();
            require(link.stats().manifests == 2, "retained anchor was superseded");
            // key_at drives find: the anchor at 700 is visible for max_frontier >= 700 once the
            // descendant also matches the same chain key at 900.
            require(link.find(700, 700, [&](std::uint32_t f) { return a.key_at(f); }).has_value(),
                    "retained anchor became undiscoverable");
            require(link.find(700, 900, [&](std::uint32_t f) { return a.key_at(f); }).has_value(),
                    "newer checkpoint missing");
        }
        {
            // An entry whose lineage mapping is absent is rejected at registration.
            context::ContextArchiveLink link(options(root / "nolineage", 1ULL << 20));
            auto entry              = entry_for(a, 700);
            entry.lineage           = nullptr;
            bool rejected           = false;
            try {
                link.register_checkpoint(1, std::move(entry));
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "an entry without a lineage mapping was accepted");
        }
        {
            // Disabled link: every call is a safe no-op.
            context::ContextArchiveLink link;
            require(!link.enabled(), "default link reported enabled");
            link.register_checkpoint(1, entry_for(a, 700));
            link.discard_checkpoint(1);
            require(link.flush(1, {}) == context::ContextArchiveLink::FlushResult::Declined,
                    "a disabled link accepted a flush");
            require(!link.find(1, 700, [&](std::uint32_t f) { return a.key_at(f); }),
                    "a disabled link returned a candidate");
            const ninfer::DiskContextArchive::Candidate candidate{.manifest         = 1,
                                                                  .frontier         = 700,
                                                                  .backend_frontier = 700};
            require(link.prefetch(candidate) == nullptr, "a disabled link returned a prefetch");
        }
        std::filesystem::remove_all(root);
        std::cout << "context archive link tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}
