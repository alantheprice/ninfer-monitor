#include "core/disk_context_archive.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ninfer {

namespace {

constexpr std::uint64_t kPrime1 = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t kPrime2 = 0xC2B2AE3D27D4EB4FULL;
constexpr std::uint64_t kPrime3 = 0x165667B19E3779F9ULL;
constexpr std::uint64_t kPrime4 = 0x85EBCA77C2B2AE63ULL;
constexpr std::uint64_t kPrime5 = 0x27D4EB2F165667C5ULL;

constexpr std::uint64_t rotl(std::uint64_t value, int bits) noexcept {
    return (value << bits) | (value >> (64 - bits));
}

std::uint64_t read64(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

std::uint32_t read32(const std::byte* data) noexcept {
    std::uint32_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

constexpr std::uint64_t mix_round(std::uint64_t accumulator, std::uint64_t input) noexcept {
    accumulator += input * kPrime2;
    accumulator = rotl(accumulator, 31);
    return accumulator * kPrime1;
}

constexpr std::uint64_t merge_round(std::uint64_t accumulator, std::uint64_t value) noexcept {
    accumulator ^= mix_round(0, value);
    return accumulator * kPrime1 + kPrime4;
}

constexpr std::array<char, 8> kManifestMagic{'N', 'I', 'N', 'F', 'C', 'K', 'P', '1'};
constexpr std::array<char, 8> kPageMagic{'N', 'I', 'N', 'F', 'P', 'A', 'G', '1'};

struct ManifestHeader {
    std::array<char, 8> magic{};
    std::uint64_t signature        = 0;
    std::uint64_t id               = 0;
    std::uint32_t frontier         = 0;
    std::uint32_t backend_frontier = 0;
    std::uint64_t prefix_key[2]{};
    std::uint64_t page_counts[kArchivePageKinds]{};
    std::uint64_t identity_bytes   = 0;
    std::uint64_t state_bytes      = 0;
    std::uint64_t payload_checksum = 0;
};

struct PageHeader {
    std::array<char, 8> magic{};
    std::uint64_t signature = 0;
    std::uint64_t key[2]{};
    std::uint32_t kind     = 0;
    std::uint32_t reserved = 0;
    std::uint64_t bytes    = 0;
    std::uint64_t checksum = 0;
    std::uint64_t padding  = 0;
};

static_assert(sizeof(PageHeader) == 64);

class FileDescriptor {
public:
    explicit FileDescriptor(int fd) noexcept : fd_(fd) {}

    ~FileDescriptor() {
        if (fd_ >= 0) { ::close(fd_); }
    }

    FileDescriptor(const FileDescriptor&)            = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    [[nodiscard]] int get() const noexcept { return fd_; }

    [[nodiscard]] bool close() noexcept {
        const int fd = fd_;
        fd_          = -1;
        return fd < 0 || ::close(fd) == 0;
    }

private:
    int fd_ = -1;
};

bool write_all(int fd, const void* data, std::size_t bytes) noexcept {
    const auto* cursor = static_cast<const char*>(data);
    while (bytes != 0) {
        const std::size_t chunk = std::min<std::size_t>(bytes, std::size_t{1} << 30);
        const ssize_t written   = ::write(fd, cursor, chunk);
        if (written < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        cursor += written;
        bytes -= static_cast<std::size_t>(written);
    }
    return true;
}

bool read_all(int fd, void* data, std::size_t bytes) noexcept {
    auto* cursor = static_cast<char*>(data);
    while (bytes != 0) {
        const std::size_t chunk = std::min<std::size_t>(bytes, std::size_t{1} << 30);
        const ssize_t got       = ::read(fd, cursor, chunk);
        if (got < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        if (got == 0) { return false; }
        cursor += got;
        bytes -= static_cast<std::size_t>(got);
    }
    return true;
}

bool sync_directory(const std::filesystem::path& directory) noexcept {
    const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) { return false; }
    FileDescriptor guard(fd);
    return ::fsync(fd) == 0;
}

// Writes the parts to a temporary sibling, makes it durable, drops it from the page cache and
// renames it into place. The caller syncs the directory once for a batch of renames.
bool write_durable(const std::filesystem::path& path,
                   std::initializer_list<std::span<const std::byte>> parts) noexcept {
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) { return false; }
    FileDescriptor guard(fd);
    for (std::span<const std::byte> part : parts) {
        if (!write_all(fd, part.data(), part.size())) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return false;
        }
    }
    if (::fdatasync(fd) != 0) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
#if defined(POSIX_FADV_DONTNEED)
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    if (!guard.close()) { return false; }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    return true;
}

std::span<const std::byte> as_bytes_of(const auto& value) noexcept {
    return std::as_bytes(std::span(&value, 1));
}

std::string hex16(std::uint64_t value) {
    std::array<char, 17> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx", static_cast<unsigned long long>(value));
    return std::string(buffer.data(), 16);
}

std::optional<std::uint64_t> parse_hex16(std::string_view text) noexcept {
    std::uint64_t value = 0;
    const auto result   = std::from_chars(text.data(), text.data() + text.size(), value, 16);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

std::uint64_t now_ns() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::system_clock::now().time_since_epoch())
                                          .count());
}

constexpr std::size_t kStagingAlignment = 4096;

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

std::uint64_t archive_checksum(std::span<const std::byte> bytes, std::uint64_t seed) noexcept {
    const std::byte* cursor = bytes.data();
    const std::byte* end    = cursor + bytes.size();
    std::uint64_t hash      = 0;
    if (bytes.size() >= 32) {
        std::uint64_t v1       = seed + kPrime1 + kPrime2;
        std::uint64_t v2       = seed + kPrime2;
        std::uint64_t v3       = seed;
        std::uint64_t v4       = seed - kPrime1;
        const std::byte* limit = end - 32;
        do {
            v1 = mix_round(v1, read64(cursor));
            v2 = mix_round(v2, read64(cursor + 8));
            v3 = mix_round(v3, read64(cursor + 16));
            v4 = mix_round(v4, read64(cursor + 24));
            cursor += 32;
        } while (cursor <= limit);
        hash = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        hash = merge_round(hash, v1);
        hash = merge_round(hash, v2);
        hash = merge_round(hash, v3);
        hash = merge_round(hash, v4);
    } else {
        hash = seed + kPrime5;
    }
    hash += static_cast<std::uint64_t>(bytes.size());
    while (cursor + 8 <= end) {
        hash ^= mix_round(0, read64(cursor));
        hash = rotl(hash, 27) * kPrime1 + kPrime4;
        cursor += 8;
    }
    if (cursor + 4 <= end) {
        hash ^= static_cast<std::uint64_t>(read32(cursor)) * kPrime1;
        hash = rotl(hash, 23) * kPrime2 + kPrime3;
        cursor += 4;
    }
    while (cursor < end) {
        hash ^= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(*cursor)) * kPrime5;
        hash = rotl(hash, 11) * kPrime1;
        ++cursor;
    }
    hash ^= hash >> 33;
    hash *= kPrime2;
    hash ^= hash >> 29;
    hash *= kPrime3;
    hash ^= hash >> 32;
    return hash;
}

ArchiveStagingAllocator::ArchiveStagingAllocator(std::span<std::byte> storage) : storage_(storage) {
    if (!storage_.empty()) { free_.push_back(Region{.offset = 0, .bytes = storage_.size()}); }
}

std::optional<ArchiveStagingAllocator::Region>
ArchiveStagingAllocator::allocate(std::size_t bytes) {
    if (bytes == 0) { return std::nullopt; }
    const std::size_t rounded = align_up(bytes, kStagingAlignment);
    std::scoped_lock lock(mutex_);
    for (auto it = free_.begin(); it != free_.end(); ++it) {
        if (it->bytes < rounded) { continue; }
        Region region{.offset = it->offset, .bytes = rounded};
        it->offset += rounded;
        it->bytes -= rounded;
        if (it->bytes == 0) { free_.erase(it); }
        return region;
    }
    return std::nullopt;
}

void ArchiveStagingAllocator::release(Region region) noexcept {
    if (region.bytes == 0) { return; }
    std::scoped_lock lock(mutex_);
    auto it =
        std::lower_bound(free_.begin(), free_.end(), region,
                         [](const Region& a, const Region& b) { return a.offset < b.offset; });
    it = free_.insert(it, region);
    if (it + 1 != free_.end() && it->offset + it->bytes == (it + 1)->offset) {
        it->bytes += (it + 1)->bytes;
        free_.erase(it + 1);
    }
    if (it != free_.begin() && (it - 1)->offset + (it - 1)->bytes == it->offset) {
        (it - 1)->bytes += it->bytes;
        free_.erase(it);
    }
}

std::size_t ArchiveStagingAllocator::used() const {
    std::scoped_lock lock(mutex_);
    std::size_t available = 0;
    for (const Region& region : free_) { available += region.bytes; }
    return storage_.size() - available;
}

DiskContextArchive::SaveJob::SaveJob(SaveJob&& other) noexcept { *this = std::move(other); }

DiskContextArchive::SaveJob& DiskContextArchive::SaveJob::operator=(SaveJob&& other) noexcept {
    if (this == &other) { return *this; }
    owner_           = std::exchange(other.owner_, nullptr);
    request_         = std::move(other.request_);
    missing_         = std::move(other.missing_);
    staging_         = std::exchange(other.staging_, {});
    page_offsets_    = other.page_offsets_;
    identity_offset_ = other.identity_offset_;
    state_offset_    = other.state_offset_;
    write_bytes_     = other.write_bytes_;
    supersedes_      = std::move(other.supersedes_);
    return *this;
}

DiskContextArchive::SaveJob::~SaveJob() {
    if (owner_ == nullptr) { return; }
    owner_->staging_.release(staging_);
    std::scoped_lock lock(owner_->mutex_);
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        auto& pages = owner_->pages_[kind];
        for (const ArchiveKey& key : request_.page_keys[kind]) {
            auto it = pages.find(key);
            if (it == pages.end() || --it->second.references != 0) { continue; }
            if (it->second.durable) {
                std::error_code ignored;
                std::filesystem::remove(owner_->page_path(kind, key), ignored);
                owner_->stored_bytes_ -= owner_->page_file_bytes(kind);
            }
            pages.erase(it);
        }
    }
}

std::span<std::byte> DiskContextArchive::SaveJob::page(std::size_t kind,
                                                       std::size_t missing_index) {
    const std::size_t bytes = owner_->options_.page_bytes[kind];
    return std::span(owner_->staging_.data(staging_) + page_offsets_[kind] + missing_index * bytes,
                     bytes);
}

std::span<std::byte> DiskContextArchive::SaveJob::identity() {
    return std::span(owner_->staging_.data(staging_) + identity_offset_, request_.identity_bytes);
}

std::span<std::byte> DiskContextArchive::SaveJob::state() {
    return std::span(owner_->staging_.data(staging_) + state_offset_, request_.state_bytes);
}

DiskContextArchive::Prefetch::~Prefetch() {
    owner_->staging_.release(staging_);
    std::scoped_lock lock(owner_->mutex_);
    const auto it = owner_->manifests_.find(candidate_.manifest);
    if (it == owner_->manifests_.end()) { return; }
    if (--it->second.pins == 0 && it->second.superseded) {
        owner_->delete_manifest_locked(candidate_.manifest);
    }
}

DiskContextArchive::Prefetch::Status
DiskContextArchive::Prefetch::wait_for(std::chrono::microseconds timeout) const {
    std::unique_lock lock(mutex_);
    ready_.wait_for(lock, timeout, [&] { return status_.load() != Status::Pending; });
    return status_.load();
}

std::span<const std::byte> DiskContextArchive::Prefetch::identity() const {
    return std::span(owner_->staging_.data(staging_) + identity_offset_, identity_bytes_);
}

std::span<const std::byte> DiskContextArchive::Prefetch::state() const {
    return std::span(owner_->staging_.data(staging_) + state_offset_, state_bytes_);
}

std::span<const std::byte> DiskContextArchive::Prefetch::pages(std::size_t kind) const {
    return std::span(owner_->staging_.data(staging_) + page_offsets_[kind],
                     page_counts_[kind] * owner_->options_.page_bytes[kind]);
}

DiskContextArchive::DiskContextArchive(DiskArchiveOptions options, std::span<std::byte> staging)
    : options_(std::move(options)), staging_(staging) {
    if (options_.directory.empty() || options_.capacity_bytes == 0) {
        throw std::invalid_argument("disk context archive needs a directory and a capacity");
    }
    std::filesystem::create_directories(options_.directory / "manifests");
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        if (options_.page_bytes[kind] == 0) { continue; }
        for (unsigned prefix = 0; prefix < 256; ++prefix) {
            std::array<char, 3> name{};
            std::snprintf(name.data(), name.size(), "%02x", prefix);
            std::filesystem::create_directories(options_.directory / "pages" /
                                                ("k" + std::to_string(kind)) / name.data());
        }
    }
    scan();
    budget_refilled_ = std::chrono::steady_clock::now();
    budget_bytes_    = static_cast<double>(options_.write_bytes_per_second) *
                       static_cast<double>(kArchiveWriteBurstSeconds);
    writer_          = std::thread([this] { writer_loop(); });
    reader_          = std::thread([this] { reader_loop(); });
}

DiskContextArchive::~DiskContextArchive() {
    {
        std::scoped_lock lock(mutex_);
        stopping_ = true;
    }
    writer_cv_.notify_all();
    reader_cv_.notify_all();
    if (writer_.joinable()) { writer_.join(); }
    if (reader_.joinable()) { reader_.join(); }
}

std::filesystem::path DiskContextArchive::manifest_path(std::uint64_t id) const {
    return options_.directory / "manifests" / (hex16(id) + ".ckpt");
}

std::filesystem::path DiskContextArchive::page_path(std::size_t kind, const ArchiveKey& key) const {
    const std::string name = hex16(key[0]) + hex16(key[1]);
    return options_.directory / "pages" / ("k" + std::to_string(kind)) / name.substr(0, 2) /
           (name + ".pg");
}

std::uint64_t DiskContextArchive::page_file_bytes(std::size_t kind) const {
    return sizeof(PageHeader) + options_.page_bytes[kind];
}

void DiskContextArchive::scan() {
    std::error_code error;
    for (auto it = std::filesystem::recursive_directory_iterator(options_.directory, error);
         !error && it != std::filesystem::recursive_directory_iterator(); it.increment(error)) {
        if (it->is_regular_file(error) && it->path().extension() == ".tmp") {
            std::filesystem::remove(it->path(), error);
        }
    }

    std::vector<std::filesystem::path> invalid;
    for (const auto& entry :
         std::filesystem::directory_iterator(options_.directory / "manifests", error)) {
        if (entry.path().extension() != ".ckpt") { continue; }
        const int fd = ::open(entry.path().c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            invalid.push_back(entry.path());
            continue;
        }
        FileDescriptor guard(fd);
        ManifestHeader header;
        struct stat info{};
        bool valid = ::fstat(fd, &info) == 0 && read_all(fd, &header, sizeof(header)) &&
                     header.magic == kManifestMagic && header.signature == options_.signature;
        ManifestRecord record;
        if (valid) {
            record.id               = header.id;
            record.frontier         = header.frontier;
            record.backend_frontier = header.backend_frontier;
            record.prefix_key       = {header.prefix_key[0], header.prefix_key[1]};
            record.identity_bytes   = header.identity_bytes;
            record.state_bytes      = header.state_bytes;
            record.file_bytes       = static_cast<std::uint64_t>(info.st_size);
            record.last_used = static_cast<std::uint64_t>(info.st_mtim.tv_sec) * 1'000'000'000ULL +
                               static_cast<std::uint64_t>(info.st_mtim.tv_nsec);
            std::uint64_t expected = sizeof(header) + header.identity_bytes + header.state_bytes;
            for (std::size_t kind = 0; kind < kArchivePageKinds && valid; ++kind) {
                const std::uint64_t count = header.page_counts[kind];
                if (count != 0 && options_.page_bytes[kind] == 0) { valid = false; }
                if (count > (std::uint64_t{1} << 24)) { valid = false; }
                expected += count * sizeof(ArchiveKey);
            }
            valid =
                valid && expected == record.file_bytes && manifest_path(header.id) == entry.path();
            for (std::size_t kind = 0; kind < kArchivePageKinds && valid; ++kind) {
                record.page_keys[kind].resize(header.page_counts[kind]);
                valid = read_all(fd, record.page_keys[kind].data(),
                                 record.page_keys[kind].size() * sizeof(ArchiveKey));
            }
        }
        if (!valid) {
            invalid.push_back(entry.path());
            continue;
        }
        next_id_ = std::max(next_id_, record.id + 1);
        for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
            for (const ArchiveKey& key : record.page_keys[kind]) { ++pages_[kind][key].references; }
        }
        stored_bytes_ += record.file_bytes;
        manifests_.emplace(record.id, std::move(record));
    }
    for (const auto& path : invalid) { std::filesystem::remove(path, error); }

    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        if (options_.page_bytes[kind] == 0) { continue; }
        const auto root = options_.directory / "pages" / ("k" + std::to_string(kind));
        for (auto it = std::filesystem::recursive_directory_iterator(root, error);
             !error && it != std::filesystem::recursive_directory_iterator(); it.increment(error)) {
            if (!it->is_regular_file(error)) { continue; }
            const std::string name = it->path().stem().string();
            std::optional<std::uint64_t> high;
            std::optional<std::uint64_t> low;
            if (name.size() == 32 && it->path().extension() == ".pg") {
                high = parse_hex16(std::string_view(name).substr(0, 16));
                low  = parse_hex16(std::string_view(name).substr(16, 16));
            }
            const auto record =
                high && low ? pages_[kind].find(ArchiveKey{*high, *low}) : pages_[kind].end();
            if (record == pages_[kind].end() || it->file_size(error) != page_file_bytes(kind)) {
                std::filesystem::remove(it->path(), error);
                continue;
            }
            record->second.durable = true;
            stored_bytes_ += page_file_bytes(kind);
        }
    }

    std::vector<std::uint64_t> incomplete;
    for (const auto& [id, record] : manifests_) {
        for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
            const bool missing =
                std::any_of(record.page_keys[kind].begin(), record.page_keys[kind].end(),
                            [&](const ArchiveKey& key) { return !pages_[kind][key].durable; });
            if (missing) {
                incomplete.push_back(id);
                break;
            }
        }
    }
    for (const std::uint64_t id : incomplete) { delete_manifest_locked(id); }
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        std::erase_if(pages_[kind], [](const auto& entry) { return entry.second.references == 0; });
    }
    std::unique_lock lock(mutex_);
    evict_locked(lock);
}

void DiskContextArchive::delete_manifest_locked(std::uint64_t id) {
    const auto it = manifests_.find(id);
    if (it == manifests_.end()) { return; }
    std::error_code ignored;
    std::filesystem::remove(manifest_path(id), ignored);
    stored_bytes_ -= std::min(stored_bytes_, it->second.file_bytes);
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        auto& pages = pages_[kind];
        for (const ArchiveKey& key : it->second.page_keys[kind]) {
            auto page = pages.find(key);
            if (page == pages.end() || --page->second.references != 0) { continue; }
            if (page->second.durable) {
                std::filesystem::remove(page_path(kind, key), ignored);
                stored_bytes_ -= std::min(stored_bytes_, page_file_bytes(kind));
            }
            pages.erase(page);
        }
    }
    manifests_.erase(it);
}

void DiskContextArchive::evict_locked(std::unique_lock<std::mutex>&) {
    while (stored_bytes_ > options_.capacity_bytes) {
        std::uint64_t victim = 0;
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        for (const auto& [id, record] : manifests_) {
            if (record.pins == 0 && record.last_used < oldest) {
                oldest = record.last_used;
                victim = id;
            }
        }
        if (victim == 0) { return; }
        delete_manifest_locked(victim);
        ++stats_.manifests_evicted;
    }
}

void DiskContextArchive::refill_budget_locked() {
    if (options_.write_bytes_per_second == 0) { return; }
    const auto now      = std::chrono::steady_clock::now();
    const double rate   = static_cast<double>(options_.write_bytes_per_second);
    const double cap    = rate * static_cast<double>(kArchiveWriteBurstSeconds);
    const double earned = std::chrono::duration<double>(now - budget_refilled_).count() * rate;
    budget_bytes_       = std::min(cap, budget_bytes_ + earned);
    budget_refilled_    = now;
}

std::optional<DiskContextArchive::SaveJob> DiskContextArchive::begin_save(SaveRequest request,
                                                                          const KeyAt& key_at) {
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        if (!request.page_keys[kind].empty() && options_.page_bytes[kind] == 0) {
            throw std::invalid_argument("disk archive save uses an unconfigured page kind");
        }
    }
    std::unique_lock lock(mutex_);
    for (auto& [id, record] : manifests_) {
        if (record.frontier == request.frontier && record.prefix_key == request.prefix_key) {
            record.last_used = now_ns();
            return std::nullopt;
        }
    }
    SaveJob job;
    std::size_t staging_bytes = 0;
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        auto& pages = pages_[kind];
        for (std::size_t index = 0; index < request.page_keys[kind].size(); ++index) {
            if (!pages.contains(request.page_keys[kind][index])) {
                job.missing_[kind].push_back(static_cast<std::uint32_t>(index));
            }
        }
        job.page_offsets_[kind] = staging_bytes;
        staging_bytes += job.missing_[kind].size() * options_.page_bytes[kind];
    }
    job.identity_offset_ = staging_bytes;
    staging_bytes        = align_up(staging_bytes + request.identity_bytes, 256);
    job.state_offset_    = staging_bytes;
    staging_bytes += request.state_bytes;
    job.write_bytes_ = staging_bytes;

    refill_budget_locked();
    if (options_.write_bytes_per_second != 0 &&
        budget_bytes_ < static_cast<double>(job.write_bytes_)) {
        ++stats_.saves_skipped_budget;
        return std::nullopt;
    }
    const std::optional<ArchiveStagingAllocator::Region> region = staging_.allocate(staging_bytes);
    if (!region) {
        ++stats_.saves_skipped_staging;
        return std::nullopt;
    }
    if (options_.write_bytes_per_second != 0) {
        budget_bytes_ -= static_cast<double>(job.write_bytes_);
    }
    job.owner_   = this;
    job.staging_ = *region;
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        for (const ArchiveKey& key : request.page_keys[kind]) { ++pages_[kind][key].references; }
    }
    // Archived ancestors of the same lineage are retired once this checkpoint is durable: their
    // prefixes remain reachable through it, so they only cost space.
    for (const auto& [id, record] : manifests_) {
        if (record.frontier < request.frontier && key_at(record.frontier) == record.prefix_key) {
            job.supersedes_.push_back(id);
        }
    }
    job.request_ = std::move(request);
    return job;
}

void DiskContextArchive::commit(SaveJob&& job) {
    if (job.owner_ != this) { throw std::invalid_argument("disk archive job has another owner"); }
    {
        std::scoped_lock lock(mutex_);
        write_queue_.push_back(std::move(job));
    }
    writer_cv_.notify_one();
}

void DiskContextArchive::flush() {
    std::unique_lock lock(mutex_);
    flushed_cv_.wait(lock, [&] { return write_queue_.empty() && !writing_; });
}

void DiskContextArchive::writer_loop() {
    for (;;) {
        std::unique_lock lock(mutex_);
        writer_cv_.wait(lock, [&] { return stopping_ || !write_queue_.empty(); });
        if (write_queue_.empty()) { return; }
        SaveJob job = std::move(write_queue_.front());
        write_queue_.pop_front();
        writing_ = true;
        lock.unlock();
        write_job(job);
        { SaveJob finished = std::move(job); }
        lock.lock();
        writing_ = false;
        if (write_queue_.empty()) { flushed_cv_.notify_all(); }
    }
}

void DiskContextArchive::write_job(SaveJob& job) {
    const SaveRequest& save = job.request_;
    bool ok                 = true;
    std::vector<std::filesystem::path> touched;
    for (std::size_t kind = 0; kind < kArchivePageKinds && ok; ++kind) {
        for (std::size_t index = 0; index < job.missing_[kind].size() && ok; ++index) {
            const ArchiveKey& key            = save.page_keys[kind][job.missing_[kind][index]];
            const std::span<std::byte> bytes = job.page(kind, index);
            PageHeader header;
            header.magic     = kPageMagic;
            header.signature = options_.signature;
            header.key[0]    = key[0];
            header.key[1]    = key[1];
            header.kind      = static_cast<std::uint32_t>(kind);
            header.bytes     = bytes.size();
            header.checksum  = archive_checksum(bytes);
            const auto path  = page_path(kind, key);
            ok               = write_durable(path, {as_bytes_of(header), bytes});
            if (!ok) { break; }
            touched.push_back(path.parent_path());
            std::scoped_lock lock(mutex_);
            PageRecord& page = pages_[kind][key];
            if (!page.durable) {
                page.durable = true;
                stored_bytes_ += page_file_bytes(kind);
            }
        }
    }
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (const auto& directory : touched) { ok = ok && sync_directory(directory); }

    std::uint64_t id = 0;
    {
        std::scoped_lock lock(mutex_);
        for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
            for (const ArchiveKey& key : save.page_keys[kind]) {
                const auto page = pages_[kind].find(key);
                ok              = ok && page != pages_[kind].end() && page->second.durable;
            }
        }
        id = next_id_++;
    }

    std::vector<ArchiveKey> keys;
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        keys.insert(keys.end(), save.page_keys[kind].begin(), save.page_keys[kind].end());
    }
    const auto key_bytes = std::as_bytes(std::span(keys));
    ManifestHeader header;
    header.magic            = kManifestMagic;
    header.signature        = options_.signature;
    header.id               = id;
    header.frontier         = save.frontier;
    header.backend_frontier = save.backend_frontier;
    header.prefix_key[0]    = save.prefix_key[0];
    header.prefix_key[1]    = save.prefix_key[1];
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        header.page_counts[kind] = save.page_keys[kind].size();
    }
    header.identity_bytes   = save.identity_bytes;
    header.state_bytes      = save.state_bytes;
    header.payload_checksum = archive_checksum(
        job.state(), archive_checksum(job.identity(), archive_checksum(key_bytes)));
    const auto path = manifest_path(id);
    ok = ok && write_durable(path, {as_bytes_of(header), key_bytes, job.identity(), job.state()}) &&
         sync_directory(path.parent_path());

    std::unique_lock lock(mutex_);
    if (!ok) {
        ++stats_.saves_failed;
        return;
    }
    ManifestRecord record;
    record.id               = id;
    record.frontier         = save.frontier;
    record.backend_frontier = save.backend_frontier;
    record.prefix_key       = save.prefix_key;
    record.page_keys        = save.page_keys;
    record.identity_bytes   = save.identity_bytes;
    record.state_bytes      = save.state_bytes;
    record.file_bytes = sizeof(header) + key_bytes.size() + save.identity_bytes + save.state_bytes;
    record.last_used  = now_ns();
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        for (const ArchiveKey& key : save.page_keys[kind]) { ++pages_[kind][key].references; }
    }
    stored_bytes_ += record.file_bytes;

    manifests_.emplace(id, std::move(record));
    for (const std::uint64_t ancestor : job.supersedes_) {
        const auto existing = manifests_.find(ancestor);
        if (existing == manifests_.end()) { continue; }
        if (existing->second.pins != 0) {
            existing->second.superseded = true;
        } else {
            delete_manifest_locked(ancestor);
        }
    }
    ++stats_.saves_committed;
    stats_.bytes_written += job.write_bytes_;
    evict_locked(lock);
}

std::optional<DiskContextArchive::Candidate> DiskContextArchive::find(std::uint32_t min_frontier,
                                                                      std::uint32_t max_frontier,
                                                                      const KeyAt& key_at) const {
    std::scoped_lock lock(mutex_);
    std::optional<Candidate> best;
    for (const auto& [id, record] : manifests_) {
        if (record.superseded || record.frontier < min_frontier || record.frontier > max_frontier ||
            (best && record.frontier <= best->frontier)) {
            continue;
        }
        if (key_at(record.frontier) != record.prefix_key) { continue; }
        best = Candidate{.manifest         = id,
                         .frontier         = record.frontier,
                         .backend_frontier = record.backend_frontier};
    }
    return best;
}

std::shared_ptr<DiskContextArchive::Prefetch>
DiskContextArchive::prefetch(const Candidate& candidate) {
    std::unique_lock lock(mutex_);
    const auto it = manifests_.find(candidate.manifest);
    if (it == manifests_.end() || it->second.superseded) { return nullptr; }
    ManifestRecord& record = it->second;
    std::shared_ptr<Prefetch> prefetch(new Prefetch(*this));
    prefetch->candidate_ = candidate;
    std::size_t bytes    = 0;
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        prefetch->page_offsets_[kind] = bytes;
        prefetch->page_counts_[kind]  = record.page_keys[kind].size();
        bytes += prefetch->page_counts_[kind] * options_.page_bytes[kind];
    }
    prefetch->identity_offset_ = bytes;
    prefetch->identity_bytes_  = record.identity_bytes;
    bytes                      = align_up(bytes + record.identity_bytes, 256);
    prefetch->state_offset_    = bytes;
    prefetch->state_bytes_     = record.state_bytes;
    bytes += record.state_bytes;
    const std::optional<ArchiveStagingAllocator::Region> region = staging_.allocate(bytes);
    if (!region) { return nullptr; }
    prefetch->staging_ = *region;
    ++record.pins;
    ++stats_.prefetches_started;
    read_queue_.push_back(prefetch);
    lock.unlock();
    reader_cv_.notify_one();
    return prefetch;
}

void DiskContextArchive::reader_loop() {
    for (;;) {
        std::unique_lock lock(mutex_);
        reader_cv_.wait(lock, [&] { return stopping_ || !read_queue_.empty(); });
        if (stopping_) {
            std::deque<std::shared_ptr<Prefetch>> abandoned = std::move(read_queue_);
            read_queue_.clear();
            lock.unlock();
            for (const auto& pending : abandoned) {
                {
                    std::scoped_lock ready(pending->mutex_);
                    pending->status_.store(Prefetch::Status::Failed);
                }
                pending->ready_.notify_all();
            }
            return;
        }
        std::shared_ptr<Prefetch> prefetch = std::move(read_queue_.front());
        read_queue_.pop_front();
        lock.unlock();
        read_prefetch(*prefetch);
        { std::scoped_lock ready(prefetch->mutex_); }
        prefetch->ready_.notify_all();
    }
}

void DiskContextArchive::read_prefetch(Prefetch& prefetch) {
    ManifestRecord record;
    {
        std::scoped_lock lock(mutex_);
        const auto it = manifests_.find(prefetch.candidate_.manifest);
        if (it == manifests_.end()) {
            ++stats_.prefetches_failed;
            prefetch.status_.store(Prefetch::Status::Failed);
            return;
        }
        record = it->second;
    }
    const auto started = std::chrono::steady_clock::now();
    std::byte* staging = staging_.data(prefetch.staging_);
    std::atomic<bool> ok{true};
    std::atomic<std::uint64_t> read{0};

    // Page files are independent: several readers keep an NVMe queue busy, which a single
    // sequential reader cannot. The manifest (identity and StateImage) is read alongside them.
    std::vector<std::pair<std::size_t, std::size_t>> pages;
    for (std::size_t kind = 0; kind < kArchivePageKinds; ++kind) {
        for (std::size_t index = 0; index < record.page_keys[kind].size(); ++index) {
            pages.emplace_back(kind, index);
        }
    }
    std::atomic<std::size_t> next_page{0};
    const auto read_pages = [&] {
        for (std::size_t task = next_page++; task < pages.size() && ok.load(); task = next_page++) {
            const auto [kind, index] = pages[task];
            const std::size_t bytes  = options_.page_bytes[kind];
            const ArchiveKey& key    = record.page_keys[kind][index];
            const auto path          = page_path(kind, key);
            const int fd             = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            FileDescriptor guard(fd);
            PageHeader header;
            std::byte* destination = staging + prefetch.page_offsets_[kind] + index * bytes;
            const bool page_ok =
                fd >= 0 && read_all(fd, &header, sizeof(header)) && header.magic == kPageMagic &&
                header.signature == options_.signature && header.key[0] == key[0] &&
                header.key[1] == key[1] && header.kind == kind && header.bytes == bytes &&
                read_all(fd, destination, bytes) &&
                archive_checksum(std::span<const std::byte>(destination, bytes)) == header.checksum;
            if (!page_ok) { ok.store(false); }
            read += page_file_bytes(kind);
        }
    };
    std::vector<std::thread> readers;
    const std::size_t reader_count = std::min<std::size_t>(kArchiveReaderThreads, pages.size());
    for (std::size_t reader = 0; reader < reader_count; ++reader) {
        readers.emplace_back(read_pages);
    }
    {
        const auto path = manifest_path(record.id);
        const int fd    = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        FileDescriptor guard(fd);
        ManifestHeader header;
        std::vector<ArchiveKey> keys;
        for (const auto& kind_keys : record.page_keys) {
            keys.resize(keys.size() + kind_keys.size());
        }
        bool manifest_ok =
            fd >= 0 && read_all(fd, &header, sizeof(header)) && header.magic == kManifestMagic &&
            header.signature == options_.signature && header.id == record.id &&
            read_all(fd, keys.data(), keys.size() * sizeof(ArchiveKey)) &&
            read_all(fd, staging + prefetch.identity_offset_, prefetch.identity_bytes_) &&
            read_all(fd, staging + prefetch.state_offset_, prefetch.state_bytes_);
        if (manifest_ok) {
            std::size_t cursor = 0;
            for (std::size_t kind = 0; kind < kArchivePageKinds && manifest_ok; ++kind) {
                manifest_ok =
                    std::equal(record.page_keys[kind].begin(), record.page_keys[kind].end(),
                               keys.begin() + static_cast<std::ptrdiff_t>(cursor));
                cursor += record.page_keys[kind].size();
            }
            const std::uint64_t checksum = archive_checksum(
                prefetch.state(),
                archive_checksum(prefetch.identity(),
                                 archive_checksum(std::as_bytes(std::span(keys)))));
            manifest_ok = manifest_ok && checksum == header.payload_checksum;
            read += record.file_bytes;
        }
        if (!manifest_ok) { ok.store(false); }
    }
    read_pages();
    for (std::thread& reader : readers) { reader.join(); }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    std::scoped_lock lock(mutex_);
    if (!ok.load()) {
        ++stats_.prefetches_failed;
        // A manifest that cannot be read back is never useful; retire it once unpinned.
        const auto it = manifests_.find(record.id);
        if (it != manifests_.end()) { it->second.superseded = true; }
        prefetch.status_.store(Prefetch::Status::Failed);
        return;
    }
    stats_.bytes_read += read.load();
    stats_.read_seconds += seconds;
    const auto it = manifests_.find(record.id);
    if (it != manifests_.end()) {
        it->second.last_used = now_ns();
        std::error_code ignored;
        std::filesystem::last_write_time(manifest_path(record.id),
                                         std::filesystem::file_time_type::clock::now(), ignored);
    }
    prefetch.status_.store(Prefetch::Status::Ready);
}

DiskArchiveStats DiskContextArchive::stats() const {
    std::scoped_lock lock(mutex_);
    DiskArchiveStats out = stats_;
    out.manifests        = manifests_.size();
    out.stored_bytes     = stored_bytes_;
    return out;
}

} // namespace ninfer
