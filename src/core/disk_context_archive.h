#pragma once

// Persistent NVMe tier below the Host context tier. The archive owns its directory, an in-memory
// index, a staging allocator over caller-provided memory, and one writer and one
// reader thread. It stores opaque page and blob bytes: the Program decides what a page or a state
// blob means, stages Device data into a SaveJob, and consumes a Prefetch to restore a checkpoint.
//
// Durability: every file is written to a temporary name, flushed with fdatasync and renamed. Pages
// become durable before the manifest that references them, so a crash leaves at most unreferenced
// pages or temporary files, which the startup scan removes. A manifest whose pages are missing or
// whose payload checksum fails is never restored.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ninfer {

using ArchiveKey = std::array<std::uint64_t, 2>;

struct ArchiveKeyHash {
    std::size_t operator()(const ArchiveKey& key) const noexcept {
        return static_cast<std::size_t>(key[0] ^ (key[1] * 0x9E3779B97F4A7C15ULL));
    }
};

[[nodiscard]] std::uint64_t archive_checksum(std::span<const std::byte> bytes,
                                             std::uint64_t seed = 0) noexcept;

inline constexpr std::size_t kArchivePageKinds = 2;

struct DiskArchiveOptions {
    std::filesystem::path directory;
    std::uint64_t capacity_bytes = 0;
    // Sustained archive write budget. Writes beyond it are skipped rather than queued; a burst of
    // kArchiveWriteBurstSeconds worth of budget is available after an idle period.
    std::uint64_t write_bytes_per_second = 0;
    // Bytes of one page for each page kind; an unused kind has zero bytes.
    std::array<std::size_t, kArchivePageKinds> page_bytes{};
    // Identity of everything that gives archived bytes their meaning (model, layouts, dtypes).
    std::uint64_t signature = 0;
};

inline constexpr std::uint64_t kArchiveWriteBurstSeconds = 120;
inline constexpr std::size_t kArchiveReaderThreads       = 4;

/** First-fit allocator over one caller-owned byte range; safe across threads. */
class ArchiveStagingAllocator {
public:
    struct Region {
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    explicit ArchiveStagingAllocator(std::span<std::byte> storage);

    [[nodiscard]] std::optional<Region> allocate(std::size_t bytes);
    void release(Region region) noexcept;

    [[nodiscard]] std::byte* data(Region region) const noexcept {
        return storage_.data() + region.offset;
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return storage_.size(); }

    [[nodiscard]] std::size_t used() const;

private:
    std::span<std::byte> storage_;
    mutable std::mutex mutex_;
    std::vector<Region> free_;
};

struct DiskArchiveStats {
    std::uint64_t saves_committed       = 0;
    std::uint64_t saves_skipped_budget  = 0;
    std::uint64_t saves_skipped_staging = 0;
    std::uint64_t saves_failed          = 0;
    std::uint64_t bytes_written         = 0;
    std::uint64_t prefetches_started    = 0;
    std::uint64_t prefetches_failed     = 0;
    std::uint64_t bytes_read            = 0;
    double read_seconds                 = 0.0;
    std::uint64_t manifests_evicted     = 0;
    std::uint64_t manifests             = 0;
    std::uint64_t stored_bytes          = 0;
};

class DiskContextArchive {
public:
    struct SaveRequest {
        std::uint32_t frontier         = 0;
        std::uint32_t backend_frontier = 0;
        ArchiveKey prefix_key{};
        std::array<std::vector<ArchiveKey>, kArchivePageKinds> page_keys;
        std::size_t identity_bytes = 0;
        std::size_t state_bytes    = 0;
        // Restore points (turn closures, long anchors) are spared by the supersede pass when a
        // newer checkpoint of the same lineage commits: their whole-state images are valid only
        // at their own frontier, so the newer checkpoint's prefix cannot replace them.
        bool retain = false;
    };

    // Staging for one checkpoint. Only pages absent from the archive need data; missing(kind)
    // lists their indices into SaveRequest::page_keys in ascending order and page(kind, i) is the
    // staging for the i-th missing page.
    class SaveJob {
    public:
        SaveJob() = default;
        SaveJob(SaveJob&&) noexcept;
        SaveJob& operator=(SaveJob&&) noexcept;
        SaveJob(const SaveJob&)            = delete;
        SaveJob& operator=(const SaveJob&) = delete;
        ~SaveJob();

        [[nodiscard]] std::span<const std::uint32_t> missing(std::size_t kind) const {
            return missing_[kind];
        }

        [[nodiscard]] std::span<std::byte> page(std::size_t kind, std::size_t missing_index);
        [[nodiscard]] std::span<std::byte> identity();
        [[nodiscard]] std::span<std::byte> state();

    private:
        friend class DiskContextArchive;

        DiskContextArchive* owner_ = nullptr;
        SaveRequest request_;
        std::array<std::vector<std::uint32_t>, kArchivePageKinds> missing_;
        ArchiveStagingAllocator::Region staging_;
        std::array<std::size_t, kArchivePageKinds> page_offsets_{};
        std::size_t identity_offset_ = 0;
        std::size_t state_offset_    = 0;
        std::uint64_t write_bytes_   = 0;
        std::vector<std::uint64_t> supersedes_;
    };

    struct Candidate {
        std::uint64_t manifest         = 0;
        std::uint32_t frontier         = 0;
        std::uint32_t backend_frontier = 0;
    };

    // Archived checkpoint read into staging. Pages of each kind are contiguous in key order.
    class Prefetch {
    public:
        enum class Status : std::uint8_t { Pending, Ready, Failed };

        ~Prefetch();
        Prefetch(const Prefetch&)            = delete;
        Prefetch& operator=(const Prefetch&) = delete;

        [[nodiscard]] Status status() const noexcept { return status_.load(); }

        [[nodiscard]] Status wait_for(std::chrono::microseconds timeout) const;

        [[nodiscard]] const Candidate& candidate() const noexcept { return candidate_; }

        [[nodiscard]] std::span<const std::byte> identity() const;
        [[nodiscard]] std::span<const std::byte> state() const;
        [[nodiscard]] std::span<const std::byte> pages(std::size_t kind) const;

        [[nodiscard]] std::size_t page_count(std::size_t kind) const noexcept {
            return page_counts_[kind];
        }

    private:
        friend class DiskContextArchive;

        explicit Prefetch(DiskContextArchive& owner) : owner_(&owner) {}

        DiskContextArchive* owner_ = nullptr;
        Candidate candidate_;
        ArchiveStagingAllocator::Region staging_;
        std::array<std::size_t, kArchivePageKinds> page_counts_{};
        std::array<std::size_t, kArchivePageKinds> page_offsets_{};
        std::size_t identity_offset_ = 0;
        std::size_t identity_bytes_  = 0;
        std::size_t state_offset_    = 0;
        std::size_t state_bytes_     = 0;
        std::atomic<Status> status_{Status::Pending};
        mutable std::mutex mutex_;
        mutable std::condition_variable ready_;
    };

    using KeyAt = std::function<ArchiveKey(std::uint32_t frontier)>;

    DiskContextArchive(DiskArchiveOptions options, std::span<std::byte> staging);
    ~DiskContextArchive();

    DiskContextArchive(const DiskContextArchive&)            = delete;
    DiskContextArchive& operator=(const DiskContextArchive&) = delete;

    // Returns no job when the checkpoint is already archived, the write budget is exhausted, or
    // staging is full. key_at describes the new checkpoint's prefix so archived ancestors of the
    // same lineage can be superseded once it is durable.
    [[nodiscard]] std::optional<SaveJob> begin_save(SaveRequest request, const KeyAt& key_at);
    void commit(SaveJob&& job);

    // Longest archived checkpoint with min_frontier <= frontier <= max_frontier whose prefix key
    // matches key_at(frontier).
    [[nodiscard]] std::optional<Candidate>
    find(std::uint32_t min_frontier, std::uint32_t max_frontier, const KeyAt& key_at) const;

    // Starts reading a candidate on the reader thread; empty when staging is exhausted or the
    // candidate is gone.
    [[nodiscard]] std::shared_ptr<Prefetch> prefetch(const Candidate& candidate);

    [[nodiscard]] DiskArchiveStats stats() const;

    // Blocks until all committed saves are durable. Used by tests and orderly shutdown.
    void flush();

private:
    struct ManifestRecord {
        std::uint64_t id               = 0;
        std::uint32_t frontier         = 0;
        std::uint32_t backend_frontier = 0;
        ArchiveKey prefix_key{};
        std::array<std::vector<ArchiveKey>, kArchivePageKinds> page_keys;
        std::size_t identity_bytes = 0;
        std::size_t state_bytes    = 0;
        std::uint64_t file_bytes   = 0;
        std::uint64_t last_used    = 0;
        std::uint32_t pins         = 0;
        bool superseded            = false;
        // Retained restore point: exempt from supersede, still LRU-evictable.
        bool retained              = false;
    };

    struct PageRecord {
        std::uint32_t references = 0;
        bool durable             = false;
    };

    void writer_loop();
    void reader_loop();
    void write_job(SaveJob& job);
    void read_prefetch(Prefetch& prefetch);
    void evict_locked(std::unique_lock<std::mutex>& lock);
    void delete_manifest_locked(std::uint64_t id);
    void scan();
    void refill_budget_locked();

    [[nodiscard]] std::filesystem::path manifest_path(std::uint64_t id) const;
    [[nodiscard]] std::filesystem::path page_path(std::size_t kind, const ArchiveKey& key) const;
    [[nodiscard]] std::uint64_t page_file_bytes(std::size_t kind) const;

    DiskArchiveOptions options_;
    ArchiveStagingAllocator staging_;

    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, ManifestRecord> manifests_;
    std::array<std::unordered_map<ArchiveKey, PageRecord, ArchiveKeyHash>, kArchivePageKinds>
        pages_;
    std::uint64_t stored_bytes_ = 0;
    std::uint64_t next_id_      = 1;
    std::uint64_t use_clock_    = 0;
    double budget_bytes_        = 0.0;
    std::chrono::steady_clock::time_point budget_refilled_;
    DiskArchiveStats stats_;

    std::condition_variable writer_cv_;
    std::condition_variable flushed_cv_;
    std::deque<SaveJob> write_queue_;
    bool writing_ = false;

    std::condition_variable reader_cv_;
    std::deque<std::shared_ptr<Prefetch>> read_queue_;

    bool stopping_ = false;
    std::thread writer_;
    std::thread reader_;
};

} // namespace ninfer
