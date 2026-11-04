#include "core/disk_context_archive.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

constexpr std::size_t kTextPage    = 4096 + 128;
constexpr std::size_t kBackendPage = 1024;

struct Lineage {
    std::uint64_t seed = 0;

    [[nodiscard]] ArchiveKey key_at(std::uint32_t frontier) const {
        return ArchiveKey{seed * 1'000'003ULL + frontier, seed ^ 0xABCDULL};
    }

    [[nodiscard]] ArchiveKey page_key(std::uint32_t kind, std::uint32_t end) const {
        return ArchiveKey{seed * 7919ULL + end, (static_cast<std::uint64_t>(kind) << 32) | 0x77ULL};
    }
};

std::byte pattern(std::uint64_t key, std::size_t offset) {
    return static_cast<std::byte>((key * 131U + offset * 7U) & 0xFFU);
}

DiskArchiveOptions options(const std::filesystem::path& directory, std::uint64_t capacity,
                           std::uint64_t rate = 0) {
    DiskArchiveOptions out;
    out.directory              = directory;
    out.capacity_bytes         = capacity;
    out.write_bytes_per_second = rate;
    out.page_bytes             = {kTextPage, kBackendPage};
    out.signature              = 0x5EED;
    return out;
}

DiskContextArchive::SaveRequest request_for(const Lineage& lineage, std::uint32_t frontier,
                                            std::size_t state_bytes = 8192) {
    DiskContextArchive::SaveRequest request;
    request.frontier         = frontier;
    request.backend_frontier = frontier - 1;
    request.prefix_key       = lineage.key_at(frontier);
    for (std::uint32_t end = 64; end - 64 < frontier; end += 64) {
        const std::uint32_t covered = std::min(end, frontier);
        request.page_keys[0].push_back(lineage.page_key(0, covered));
        request.page_keys[1].push_back(lineage.page_key(1, covered));
    }
    request.identity_bytes = 100 + frontier;
    request.state_bytes    = state_bytes;
    return request;
}

std::uint64_t fill_and_commit(DiskContextArchive& archive, const Lineage& lineage,
                              std::uint32_t frontier, std::size_t state_bytes = 8192) {
    const auto request = request_for(lineage, frontier, state_bytes);
    auto job = archive.begin_save(request, [&](std::uint32_t f) { return lineage.key_at(f); });
    require(job.has_value(), "save was unexpectedly declined");
    std::uint64_t staged = 0;
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        const auto missing = job->missing(kind);
        for (std::size_t i = 0; i < missing.size(); ++i) {
            const std::uint64_t key = request.page_keys[kind][missing[i]][0];
            auto bytes              = job->page(kind, i);
            for (std::size_t b = 0; b < bytes.size(); ++b) { bytes[b] = pattern(key, b); }
            ++staged;
        }
    }
    auto identity = job->identity();
    for (std::size_t b = 0; b < identity.size(); ++b) { identity[b] = pattern(frontier, b); }
    auto state = job->state();
    for (std::size_t b = 0; b < state.size(); ++b) { state[b] = pattern(frontier + 1, b); }
    archive.commit(std::move(*job));
    return staged;
}

void verify_prefetch(DiskContextArchive& archive, const Lineage& lineage, std::uint32_t frontier) {
    const auto candidate =
        archive.find(1, frontier, [&](std::uint32_t f) { return lineage.key_at(f); });
    require(candidate && candidate->frontier == frontier, "archived checkpoint was not found");
    const auto prefetch = archive.prefetch(*candidate);
    require(prefetch != nullptr, "prefetch could not be staged");
    require(prefetch->wait_for(std::chrono::seconds(10)) ==
                DiskContextArchive::Prefetch::Status::Ready,
            "prefetch did not become ready");
    const auto request = request_for(lineage, frontier);
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        const std::size_t page_bytes = kind == 0 ? kTextPage : kBackendPage;
        require(prefetch->page_count(kind) == request.page_keys[kind].size(),
                "prefetched page count differs");
        const auto pages = prefetch->pages(kind);
        for (std::size_t page = 0; page < prefetch->page_count(kind); ++page) {
            const std::uint64_t key = request.page_keys[kind][page][0];
            for (std::size_t b = 0; b < page_bytes; b += 97) {
                require(pages[page * page_bytes + b] == pattern(key, b),
                        "prefetched page bytes differ");
            }
        }
    }
    const auto identity = prefetch->identity();
    require(identity.size() == 100 + frontier, "prefetched identity size differs");
    for (std::size_t b = 0; b < identity.size(); ++b) {
        require(identity[b] == pattern(frontier, b), "prefetched identity bytes differ");
    }
    const auto state = prefetch->state();
    for (std::size_t b = 0; b < state.size(); b += 13) {
        require(state[b] == pattern(frontier + 1, b), "prefetched state bytes differ");
    }
}

std::size_t count_files(const std::filesystem::path& root, const std::string& extension) {
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == extension) { ++count; }
    }
    return count;
}

} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("ninfer-disk-archive-test-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    std::vector<std::byte> staging(64U << 20);
    try {
        const Lineage a{.seed = 1};
        const Lineage b{.seed = 2};
        {
            DiskContextArchive archive(options(root / "basic", 1ULL << 30), staging);
            require(fill_and_commit(archive, a, 300) == 10, "first save staged the wrong pages");
            archive.flush();
            verify_prefetch(archive, a, 300);

            // A later turn of the same conversation shares every full page; only the old partial
            // tail and the new pages are written, and the ancestor checkpoint is retired.
            require(fill_and_commit(archive, a, 700) == 2 * (11 - 4),
                    "descendant save restaged shared pages");
            archive.flush();
            require(archive.stats().manifests == 1, "ancestor checkpoint was not superseded");
            require(!archive.find(1, 699, [&](std::uint32_t f) { return a.key_at(f); }),
                    "superseded ancestor remained discoverable");
            verify_prefetch(archive, a, 700);

            require(!archive.begin_save(request_for(a, 700),
                                        [&](std::uint32_t f) { return a.key_at(f); }),
                    "duplicate checkpoint was archived twice");
            require(!archive.find(1, 650, [&](std::uint32_t f) { return a.key_at(f); }),
                    "a checkpoint beyond the requested frontier was returned");
            require(!archive.find(1, 4000, [&](std::uint32_t f) { return b.key_at(f); }),
                    "another lineage matched");
        }
        {
            // Persistence: a new instance sees the archive written by the previous one.
            DiskContextArchive archive(options(root / "basic", 1ULL << 30), staging);
            require(archive.stats().manifests == 1, "archive did not survive a restart");
            verify_prefetch(archive, a, 700);
        }
        {
            // Retained restore points (turn closures, long anchors) survive a newer same-lineage
            // checkpoint and stay discoverable at their own frontier, on disk and after restart.
            DiskContextArchive archive(options(root / "retained", 1ULL << 30), staging);
            auto anchor = request_for(a, 300);
            anchor.retain = true;
            auto job = archive.begin_save(anchor, [&](std::uint32_t f) { return a.key_at(f); });
            require(job.has_value(), "retained anchor save was declined");
            for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
                const auto missing = job->missing(kind);
                for (std::size_t i = 0; i < missing.size(); ++i) {
                    const std::uint64_t key = anchor.page_keys[kind][missing[i]][0];
                    auto bytes = job->page(kind, i);
                    for (std::size_t b = 0; b < bytes.size(); ++b) {
                        bytes[b] = static_cast<std::byte>(pattern(key, b));
                    }
                }
            }
            for (std::size_t b = 0; b < job->identity().size(); ++b) {
                job->identity()[b] = static_cast<std::byte>(pattern(300, b));
            }
            for (std::size_t b = 0; b < job->state().size(); ++b) {
                job->state()[b] = static_cast<std::byte>(pattern(301, b));
            }
            archive.commit(std::move(*job));
            archive.flush();

            auto later = request_for(a, 700);
            auto later_job = archive.begin_save(later, [&](std::uint32_t f) { return a.key_at(f); });
            require(later_job.has_value(), "descendant save was declined");
            for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
                const auto missing = later_job->missing(kind);
                for (std::size_t i = 0; i < missing.size(); ++i) {
                    const std::uint64_t key = later.page_keys[kind][missing[i]][0];
                    auto bytes = later_job->page(kind, i);
                    for (std::size_t b = 0; b < bytes.size(); ++b) {
                        bytes[b] = static_cast<std::byte>(pattern(key, b));
                    }
                }
            }
            for (std::size_t b = 0; b < later_job->identity().size(); ++b) {
                later_job->identity()[b] = static_cast<std::byte>(pattern(700, b));
            }
            for (std::size_t b = 0; b < later_job->state().size(); ++b) {
                later_job->state()[b] = static_cast<std::byte>(pattern(701, b));
            }
            archive.commit(std::move(*later_job));
            archive.flush();

            require(archive.stats().manifests == 2,
                    "retained anchor was superseded by the newer checkpoint");
            require(archive.find(1, 699, [&](std::uint32_t f) { return a.key_at(f); }).has_value(),
                    "retained anchor became undiscoverable");
            require(archive.find(1, 4000, [&](std::uint32_t f) { return a.key_at(f); }).has_value(),
                    "newer checkpoint missing");
        }
        {
            DiskContextArchive archive(options(root / "retained", 1ULL << 30), staging);
            require(archive.stats().manifests == 2,
                    "retained flag did not survive a restart");
            require(archive.find(1, 699, [&](std::uint32_t f) { return a.key_at(f); }).has_value(),
                    "retained anchor missing after restart");
        }
        {
            // Startup removes temporary files and orphan pages, and drops manifests whose pages
            // are gone.
            const auto directory = root / "basic";
            std::ofstream(directory / "manifests" / "deadbeef.ckpt.tmp") << "partial";
            std::filesystem::create_directories(directory / "pages" / "k0" / "ff");
            std::ofstream(directory / "pages" / "k0" / "ff" / "ff00000000000000ff00000000000000.pg")
                << "orphan";
            DiskContextArchive archive(options(directory, 1ULL << 30), staging);
            require(count_files(directory, ".tmp") == 0, "temporary files survived startup");
            require(!std::filesystem::exists(directory / "pages" / "k0" / "ff" /
                                             "ff00000000000000ff00000000000000.pg"),
                    "orphan page survived startup");
            require(archive.stats().manifests == 1, "valid manifest was dropped");
        }
        {
            const auto directory = root / "basic";
            std::filesystem::path victim;
            for (const auto& entry :
                 std::filesystem::recursive_directory_iterator(directory / "pages" / "k1")) {
                if (entry.is_regular_file()) {
                    victim = entry.path();
                    break;
                }
            }
            std::filesystem::remove(victim);
            DiskContextArchive archive(options(directory, 1ULL << 30), staging);
            require(archive.stats().manifests == 0, "manifest with a missing page was kept");
            require(count_files(directory / "pages", ".pg") == 0,
                    "pages of a dropped manifest were kept");
        }
        {
            // Corruption is detected on read and the checkpoint is retired.
            const auto directory = root / "corrupt";
            {
                DiskContextArchive archive(options(directory, 1ULL << 30), staging);
                fill_and_commit(archive, a, 200);
                archive.flush();
            }
            for (const auto& entry :
                 std::filesystem::recursive_directory_iterator(directory / "pages" / "k0")) {
                if (!entry.is_regular_file()) { continue; }
                std::fstream file(entry.path(), std::ios::in | std::ios::out | std::ios::binary);
                file.seekp(200);
                file.put('\x5A');
                break;
            }
            DiskContextArchive archive(options(directory, 1ULL << 30), staging);
            const auto candidate =
                archive.find(1, 200, [&](std::uint32_t f) { return a.key_at(f); });
            require(candidate.has_value(), "corrupt checkpoint was not indexed before reading");
            {
                const auto prefetch = archive.prefetch(*candidate);
                require(prefetch->wait_for(std::chrono::seconds(10)) ==
                            DiskContextArchive::Prefetch::Status::Failed,
                        "corrupt page was accepted");
            }
            require(archive.stats().manifests == 0, "corrupt checkpoint was kept");
        }
        {
            // The least recently used checkpoint is evicted when the capacity is exceeded.
            const auto directory = root / "evict";
            DiskContextArchive archive(options(directory, 150'000), staging);
            const Lineage c{.seed = 3};
            fill_and_commit(archive, a, 640);
            archive.flush();
            fill_and_commit(archive, b, 640);
            archive.flush();
            verify_prefetch(archive, a, 640);
            fill_and_commit(archive, c, 640);
            archive.flush();
            require(archive.stats().manifests_evicted != 0, "capacity was not enforced");
            require(archive.stats().stored_bytes <= 150'000, "stored bytes exceed capacity");
            require(!archive.find(1, 640, [&](std::uint32_t f) { return b.key_at(f); }),
                    "least recently used checkpoint survived eviction");
            verify_prefetch(archive, c, 640);
        }
        {
            // The write budget declines saves instead of queueing them.
            DiskContextArchive archive(options(root / "budget", 1ULL << 30, 100), staging);
            require(!archive.begin_save(request_for(a, 640, 1U << 20),
                                        [&](std::uint32_t f) { return a.key_at(f); }),
                    "save beyond the write budget was accepted");
            require(archive.stats().saves_skipped_budget == 1, "budget skip was not counted");
        }
        {
            // Staging exhaustion declines a save and leaves no page references behind.
            std::vector<std::byte> small(64U << 10);
            DiskContextArchive archive(options(root / "staging", 1ULL << 30), small);
            require(!archive.begin_save(request_for(a, 640, 1U << 20),
                                        [&](std::uint32_t f) { return a.key_at(f); }),
                    "save larger than staging was accepted");
            require(archive.stats().saves_skipped_staging == 1, "staging skip was not counted");
        }
        {
            // An abandoned job releases its staging and page references.
            DiskContextArchive archive(options(root / "abandon", 1ULL << 30), staging);
            {
                auto job = archive.begin_save(request_for(a, 300),
                                              [&](std::uint32_t f) { return a.key_at(f); });
                require(job.has_value(), "save was declined");
            }
            require(fill_and_commit(archive, a, 300) == 10, "abandoned job left pages behind");
            archive.flush();
            verify_prefetch(archive, a, 300);
        }
        std::filesystem::remove_all(root);
        std::cout << "disk context archive tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}
