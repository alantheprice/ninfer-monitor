// Persistent NVMe tier below the Host context tier.
//
// A finished turn archives the checkpoint its next turn is most likely to resume from: the rewrite
// checkpoint when present, otherwise the endpoint. Only pages absent from the archive are copied
// out; a page is keyed by the shortlist digest of the token prefix it completes, so later turns of
// a conversation share the pages of earlier ones.
//
// A request whose archived prefix extends its planned reuse by at least the configured minimum
// starts reading that checkpoint when its prefill is staged. The first prefill step then waits
// briefly for the read, verifies the exact prompt identity, and writes the archived pages and
// StateImage into the request's own freshly mapped (exclusive) pages and state slot before the
// ordinary prefill resumes from the archived frontier. Nothing here changes the resource plan:
// the restore replaces prefill work inside capacity the plan already granted.

#include "models/qwen3_5/program/program_impl.h"

#include "core/device.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace ninfer::models::qwen3_5::detail {

namespace {

constexpr std::uint32_t kPageTokens = static_cast<std::uint32_t>(kPagedKVPageSize);
constexpr std::size_t kTextPages    = 0;
constexpr std::size_t kBackendPages = 1;
constexpr auto kRestoreWaitSlice    = std::chrono::milliseconds(2);
constexpr auto kRestoreDeadline     = std::chrono::seconds(10);

std::uint32_t pages_for(std::uint32_t tokens) noexcept {
    return (tokens + kPageTokens - 1U) / kPageTokens;
}

ArchiveKey page_key(std::size_t kind, const std::array<std::uint64_t, 2>& prefix,
                    std::uint32_t columns) noexcept {
    const std::array<std::uint64_t, 4> words{prefix[0], prefix[1], kind, columns};
    const auto bytes = std::as_bytes(std::span(words));
    return ArchiveKey{archive_checksum(bytes, 0x6E696E6670673031ULL),
                      archive_checksum(bytes, 0x6E696E6670673032ULL)};
}

void mix_layout(std::string& signature, const HostKVPageLayout& layout) {
    signature += "kv:" + std::to_string(layout.page_stride) + ':' +
                 std::to_string(layout.geometry.page_tokens) + ':' +
                 std::to_string(static_cast<int>(layout.geometry.device_plane_order));
    for (const KVPlaneGeometry& plane : layout.geometry.planes) {
        signature += ':' + std::to_string(static_cast<int>(plane.dtype)) + ',' +
                     std::to_string(plane.leading_extent) + ',' + std::to_string(plane.head_extent);
    }
    signature += '\n';
}

} // namespace

void ProgramImpl::initialize_disk_tier(const ContextCacheOptions& options) {
    if (options.disk_directory.empty()) { return; }
    auto tier         = std::make_unique<DiskTier>();
    tier->min_tokens  = options.disk_min_tokens;
    tier->text_layout = plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry());
    if (const qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
        tier->backend_layout = plan_host_kv_page_layout(backend->page_pool().geometry());
    }

    std::string signature =
        "ninfer-disk-tier-1\nartifact:" + std::to_string(options.disk_artifact_identity) +
        "\nspec:" + std::to_string(static_cast<int>(speculative_backend)) +
        "\nstate:" + std::to_string(state_images->host_layout().image_bytes) + '\n';
    mix_layout(signature, tier->text_layout);
    if (tier->backend_layout) { mix_layout(signature, *tier->backend_layout); }
    const std::uint64_t digest = archive_checksum(std::as_bytes(std::span(signature)));
    std::array<char, 17> name{};
    std::snprintf(name.data(), name.size(), "%016llx", static_cast<unsigned long long>(digest));

    DiskArchiveOptions archive;
    archive.directory              = options.disk_directory / name.data();
    archive.capacity_bytes         = options.disk_capacity_bytes;
    archive.write_bytes_per_second = options.disk_write_bytes_per_second;
    archive.page_bytes[kTextPages] = tier->text_layout.page_stride;
    archive.page_bytes[kBackendPages] =
        tier->backend_layout ? tier->backend_layout->page_stride : 0U;
    archive.signature = digest;

    tier->staging       = std::unique_ptr<std::byte[]>(new std::byte[options.disk_staging_bytes]);
    tier->staging_bytes = options.disk_staging_bytes;
    tier->archive       = std::make_unique<DiskContextArchive>(
        std::move(archive), std::span(tier->staging.get(), tier->staging_bytes));
    disk_tier_ = std::move(tier);
}

std::optional<std::uint32_t>
ProgramImpl::disk_backend_frontier(std::uint32_t frontier) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) {
        return frontier == 0 ? 0U : frontier - 1U;
    }
    if (speculative_backend == SpeculativeBackend::DFlash && backend_kv_cache() != nullptr) {
        return frontier;
    }
    return std::nullopt;
}

void ProgramImpl::archive_finished_sequence(const SequenceState& sequence) noexcept {
    if (!disk_tier_ || !sequence.kv) { return; }
    std::uint32_t frontier = 0;
    StateImageHandle image;
    if (sequence.rewrite_checkpoint.valid && sequence.rewrite_state) {
        frontier = sequence.rewrite_checkpoint.frontier;
        image    = *sequence.rewrite_state;
    } else if (sequence.endpoint_valid) {
        frontier = sequence.execution_frontier;
        image    = sequence.state.read;
    } else {
        return;
    }
    if (frontier < disk_tier_->min_tokens) { return; }
    if (frontier > sequence.ledger.size() || frontier > sequence.prefix_digests.size() ||
        frontier > sequence.text_kv_valid) {
        ++disk_tier_->save_skips;
        return;
    }
    std::optional<DiskContextArchive::SaveJob> job;
    try {
        if (!state_store->valid(image) ||
            state_store->residency(image) == StateReplicaResidency::None) {
            ++disk_tier_->save_skips;
            return;
        }
        const std::optional<std::uint32_t> backend = disk_backend_frontier(frontier);
        const bool archive_backend = backend.has_value() && sequence.kv->backend.has_value();
        if (archive_backend && backend_kv_valid(sequence) < *backend) {
            ++disk_tier_->save_skips;
            return;
        }
        // MTP KV at position p is computed from token p+1, so a backend page is keyed by the
        // prefix one token past its own end.
        const std::uint32_t backend_lookahead =
            speculative_backend == SpeculativeBackend::Mtp ? 1U : 0U;

        std::vector<std::byte> identity;
        if (!sequence.prefix_identity.serialize_text_prefix(sequence.ledger, frontier, identity)) {
            return;
        }

        // A checkpoint may already be partly demoted: each page is archived from its Device
        // replica when present, otherwise from a Host replica covering the needed columns.
        const auto readable = [](const LogicalKVPageStore& pages, LogicalKVPageHandle logical,
                                 std::uint32_t columns) {
            return pages.device_resident(logical) ||
                   (pages.host_resident(logical) && pages.host_replica_current(logical) &&
                    pages.host_replica(logical).committed_columns >= columns);
        };

        DiskContextArchive::SaveRequest request;
        request.frontier               = frontier;
        request.backend_frontier       = archive_backend ? *backend : 0U;
        request.prefix_key             = sequence.prefix_digests.at(frontier);
        const std::uint32_t text_pages = pages_for(frontier);
        for (std::uint32_t page = 0; page < text_pages; ++page) {
            const std::uint32_t end     = std::min(frontier, (page + 1U) * kPageTokens);
            const std::uint32_t columns = end - page * kPageTokens;
            if (!readable(*text_kv_pages, text_kv_addresses->logical_page(sequence.kv->text, page),
                          columns)) {
                ++disk_tier_->save_skips;
                return;
            }
            request.page_keys[kTextPages].push_back(
                page_key(kTextPages, sequence.prefix_digests.at(end), columns));
        }
        if (archive_backend) {
            const std::uint32_t backend_pages = pages_for(*backend);
            for (std::uint32_t page = 0; page < backend_pages; ++page) {
                const std::uint32_t end     = std::min(*backend, (page + 1U) * kPageTokens);
                const std::uint32_t columns = end - page * kPageTokens;
                if (!readable(*backend_kv_pages,
                              backend_kv_addresses->logical_page(*sequence.kv->backend, page),
                              columns)) {
                    ++disk_tier_->save_skips;
                    return;
                }
                request.page_keys[kBackendPages].push_back(page_key(
                    kBackendPages, sequence.prefix_digests.at(end + backend_lookahead), columns));
            }
        }
        request.identity_bytes = identity.size();
        request.state_bytes    = state_images->host_layout().image_bytes;

        job = disk_tier_->archive->begin_save(
            std::move(request), [&](std::uint32_t at) { return sequence.prefix_digests.at(at); });
        if (!job) { return; }

        const auto stage_pages = [&](std::size_t kind, KVAddressSpaceStore& addresses,
                                     LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                     DeviceKVPagePool& pool, const HostKVPageLayout& layout) {
            const std::span<const std::uint32_t> missing = job->missing(kind);
            std::vector<DeviceKVPageHandle> run;
            std::size_t run_begin = 0;
            const auto flush_run  = [&] {
                if (run.empty()) { return; }
                pool.copy_to_host(run, job->page(kind, run_begin).data(), layout, device.stream);
                run.clear();
            };
            for (std::size_t index = 0; index < missing.size(); ++index) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, missing[index]);
                if (pages.device_resident(logical)) {
                    if (run.empty()) { run_begin = index; }
                    run.push_back(pages.physical(logical));
                    continue;
                }
                flush_run();
                const HostKVPageReplica replica = pages.host_replica(logical);
                const HostKVAllocationConstView view =
                    host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
                std::memcpy(job->page(kind, index).data(), view.data(), layout.page_stride);
            }
            flush_run();
        };
        stage_pages(kTextPages, *text_kv_addresses, *text_kv_pages, sequence.kv->text,
                    decoder->text_kv.page_pool(), disk_tier_->text_layout);
        if (archive_backend) {
            stage_pages(kBackendPages, *backend_kv_addresses, *backend_kv_pages,
                        *sequence.kv->backend, backend_kv_cache()->page_pool(),
                        *disk_tier_->backend_layout);
        }
        if (state_store->residency(image) == StateReplicaResidency::HostOnly) {
            const HostStateImageConstView view = state_store->host_view(image);
            std::memcpy(job->state().data(), view.data, job->state().size());
        } else {
            state_images->copy_to_host(state_store->physical_slot(image),
                                       HostStateImageView{.data   = job->state().data(),
                                                          .layout = &state_images->host_layout()},
                                       device.stream);
        }
        std::memcpy(job->identity().data(), identity.data(), identity.size());
        device.synchronize();
        disk_tier_->archive->commit(std::move(*job));
    } catch (...) {
        // An archive is an optimization: a failure leaves the request's own state untouched. Any
        // staged copy must finish before the job releases its staging.
        try {
            device.synchronize();
        } catch (...) {}
        ++disk_tier_->save_failures;
    }
}

void ProgramImpl::begin_disk_restore(const SequenceState& sequence, RequestControl::Prefill& staged,
                                     ReusePath reuse) noexcept {
    if (!disk_tier_ || staged.vision || !staged.prompt.vision_items.empty() ||
        (reuse != ReusePath::Root && reuse != ReusePath::SharedStablePrefix)) {
        return;
    }
    if (speculative_backend == SpeculativeBackend::Mtp && !staged.prepare_mtp) { return; }
    if (staged.prompt_tokens == 0 || sequence.prefix_digests.size() < staged.prompt_tokens) {
        return;
    }
    const std::uint64_t minimum = static_cast<std::uint64_t>(staged.base) + disk_tier_->min_tokens;
    const std::uint32_t maximum = staged.prompt_tokens - 1U;
    if (minimum > maximum) { return; }
    try {
        ++disk_tier_->restore_attempts;
        const std::optional<DiskContextArchive::Candidate> candidate = disk_tier_->archive->find(
            static_cast<std::uint32_t>(minimum), maximum,
            [&](std::uint32_t at) { return sequence.prefix_digests.at(at); });
        if (!candidate) { return; }
        const std::optional<std::uint32_t> backend = disk_backend_frontier(candidate->frontier);
        if (backend && sequence.kv && sequence.kv->backend &&
            candidate->backend_frontier != *backend) {
            return;
        }
        staged.disk_restore          = disk_tier_->archive->prefetch(*candidate);
        staged.disk_restore_deadline = std::chrono::steady_clock::now() + kRestoreDeadline;
    } catch (...) { staged.disk_restore.reset(); }
}

ProgramImpl::DiskRestoreStep ProgramImpl::advance_disk_restore(SequenceState& sequence,
                                                               RequestControl::Prefill& staged,
                                                               std::uint32_t& restored_tokens) {
    restored_tokens = 0;
    if (!staged.disk_restore) { return DiskRestoreStep::None; }
    const DiskContextArchive::Prefetch::Status status =
        staged.disk_restore->wait_for(kRestoreWaitSlice);
    if (status == DiskContextArchive::Prefetch::Status::Pending &&
        std::chrono::steady_clock::now() < staged.disk_restore_deadline) {
        return DiskRestoreStep::Waiting;
    }
    const std::shared_ptr<DiskContextArchive::Prefetch> prefetch = std::move(staged.disk_restore);
    staged.disk_restore.reset();
    if (status != DiskContextArchive::Prefetch::Status::Ready) {
        ++disk_tier_->restore_failures;
        if (status == DiskContextArchive::Prefetch::Status::Pending) {
            // Still reading when the deadline expired; the request falls back to prefill.
            ++disk_tier_->restore_deadline_misses;
        }
        return DiskRestoreStep::None;
    }
    const auto started = std::chrono::steady_clock::now();
    restored_tokens    = restore_from_disk(sequence, staged, *prefetch);
    if (restored_tokens == 0) {
        ++disk_tier_->restore_failures;
        return DiskRestoreStep::None;
    }
    ++disk_tier_->restores;
    disk_tier_->restored_tokens += restored_tokens;
    disk_tier_->restore_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return DiskRestoreStep::Restored;
}

std::uint32_t ProgramImpl::restore_from_disk(SequenceState& sequence,
                                             RequestControl::Prefill& staged,
                                             const DiskContextArchive::Prefetch& prefetch) {
    const std::uint32_t frontier = prefetch.candidate().frontier;
    if (!sequence.kv || staged.cursor != staged.base || frontier <= staged.cursor ||
        frontier >= staged.prompt_tokens || staged.mtp_bridge == MtpBridgeMode::AfterExactHit ||
        staged.pending_capture_offer != 0) {
        return 0;
    }
    std::vector<TokenId> tokens;
    ResidentPrefixIdentity identity;
    if (!identity.deserialize_text_prefix(prefetch.identity(), tokens) ||
        tokens.size() != frontier || !prefix_matches(staged.prompt, tokens, identity, frontier) ||
        prefetch.state().size() != state_images->host_layout().image_bytes) {
        return 0;
    }

    // Archived pages may only overwrite pages this request owns exclusively. Pages below the
    // current frontier that are shared with a source prefix are never written: the copy starts at
    // the page holding the current frontier, which a prefix fork has already made private.
    const auto exclusive_pages =
        [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages, KVAddressSpaceHandle address,
            std::uint32_t current, std::uint32_t target, std::vector<DeviceKVPageHandle>& handles) {
            handles.clear();
            for (std::uint32_t page = current / kPageTokens; page < pages_for(target); ++page) {
                if (pages.address_references(addresses.logical_page(address, page)) != 1U) {
                    return false;
                }
                handles.push_back(addresses.physical_page(address, page));
            }
            return true;
        };

    const std::uint32_t text_pages = pages_for(frontier);
    if (prefetch.page_count(kTextPages) != text_pages) { return 0; }
    std::vector<DeviceKVPageHandle> text_handles;
    if (!exclusive_pages(*text_kv_addresses, *text_kv_pages, sequence.kv->text, staged.cursor,
                         frontier, text_handles)) {
        return 0;
    }

    const std::optional<std::uint32_t> backend = disk_backend_frontier(frontier);
    const bool restore_backend = backend.has_value() && sequence.kv->backend.has_value();
    std::vector<DeviceKVPageHandle> backend_handles;
    std::uint32_t backend_current = 0;
    if (restore_backend) {
        backend_current = backend_kv_valid(sequence);
        if (prefetch.candidate().backend_frontier != *backend ||
            prefetch.page_count(kBackendPages) != pages_for(*backend) ||
            backend_current > *backend ||
            !exclusive_pages(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                             backend_current, *backend, backend_handles)) {
            return 0;
        }
    }

    // The whole StateImage is replaced, so a pending Fork from a source checkpoint is settled
    // first; its destination becomes the slot the archived image is written to.
    settle_state_fork(sequence);
    const std::int32_t slot = state_store->physical_slot(sequence.state.write);

    const std::size_t text_stride = disk_tier_->text_layout.page_stride;
    decoder->text_kv.page_pool().copy_from_host(
        prefetch.pages(kTextPages).data() + (staged.cursor / kPageTokens) * text_stride,
        disk_tier_->text_layout, text_handles, device.stream);
    if (restore_backend && !backend_handles.empty()) {
        const std::size_t backend_stride = disk_tier_->backend_layout->page_stride;
        backend_kv_cache()->page_pool().copy_from_host(
            prefetch.pages(kBackendPages).data() + (backend_current / kPageTokens) * backend_stride,
            *disk_tier_->backend_layout, backend_handles, device.stream);
    }
    state_images->copy_from_host(HostStateImageConstView{.data   = prefetch.state().data(),
                                                         .layout = &state_images->host_layout()},
                                 slot, device.stream);
    // Staging is released with the Prefetch; every copy from it must have completed.
    device.synchronize();

    const std::uint32_t restored = frontier - staged.cursor;
    sequence.text_kv_valid       = frontier;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        sequence.mtp_kv_valid = frontier - 1U;
    } else if (is_masked_draft_backend(speculative_backend)) {
        sequence.dflash_context_frontier = frontier;
    }
    commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    refresh_state_views(sequence);
    staged.cursor          = frontier;
    staged.bridge_frontier = frontier;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        staged.mtp_bridge = MtpBridgeMode::BeforeSuffix;
    }
    while (staged.next_capture < staged.capture_groups.size() &&
           staged.capture_groups[staged.next_capture].frontier <= frontier) {
        ++staged.next_capture;
    }
    return restored;
}

void ProgramImpl::populate_disk_usage(qwen3_5::PhysicalUsageSnapshot& out) const noexcept {
    if (!disk_tier_) { return; }
    const DiskArchiveStats archive = disk_tier_->archive->stats();
    out.disk_restores              = disk_tier_->restores;
    out.disk_restored_tokens       = disk_tier_->restored_tokens;
    out.disk_restore_failures      = disk_tier_->restore_failures;
    out.disk_restore_seconds       = disk_tier_->restore_seconds;
    out.disk_read_seconds          = archive.read_seconds;
    out.disk_saves                 = archive.saves_committed;
    out.disk_saves_skipped =
        archive.saves_skipped_budget + archive.saves_skipped_staging + disk_tier_->save_skips;
    out.disk_save_failures = archive.saves_failed + disk_tier_->save_failures;
    out.disk_bytes_written = archive.bytes_written;
    out.disk_bytes_read    = archive.bytes_read;
    out.disk_stored_bytes  = archive.stored_bytes;
    out.disk_manifests     = archive.manifests;
    out.disk_restore_attempts        = disk_tier_->restore_attempts;
    out.disk_restore_deadline_misses = disk_tier_->restore_deadline_misses;
    out.disk_saves_evicted           = archive.manifests_evicted;
    out.disk_saves_skipped_budget    = archive.saves_skipped_budget;
    out.disk_saves_skipped_staging   = archive.saves_skipped_staging;
}

} // namespace ninfer::models::qwen3_5::detail
