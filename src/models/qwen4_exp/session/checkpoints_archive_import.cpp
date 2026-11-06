#include "models/qwen4_exp/session/internal.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen4_exp {
namespace detail {

// Mirrors kStateSentinel in checkpoints_archive.cpp.
constexpr std::uint64_t kImportStateSentinel = 0x414E4142494E3141ULL;

// Identity image layout (checkpoints_archive.cpp:archive_entry):
// [u32 positions bytes][positions image][u32 digests bytes][digests image][u64 tag]
// [u64 token count][TokenId x tokens]
struct ArchivedIdentity {
    std::vector<TokenId> tokens;
    qwen::ResidentPrefixIdentity positions;
    qwen::PrefixShortlistDigests digests;
    std::uint32_t tag = 0;
};

ArchivedIdentity parse_identity(std::span<const std::byte> bytes) {
    if (bytes.size() < 24) throw std::invalid_argument("archived identity is truncated");
    ArchivedIdentity out;
    std::size_t cursor = 0;
    const auto need = [&](std::size_t count) {
        if (cursor + count > bytes.size())
            throw std::invalid_argument("archived identity is truncated");
    };
    const auto read_u32 = [&] {
        need(4);
        std::uint32_t value = 0;
        for (std::size_t i = 0; i < 4; ++i)
            value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[cursor + i]))
                     << (8U * i);
        cursor += 4;
        return value;
    };
    const auto read_u64 = [&] {
        need(8);
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < 8; ++i)
            value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[cursor + i]))
                     << (8U * i);
        cursor += 8;
        return value;
    };
    const auto positions_bytes = read_u32();
    need(positions_bytes);
    out.positions.assign_image(bytes.subspan(cursor, positions_bytes));
    cursor += positions_bytes;
    const auto digests_bytes = read_u32();
    need(digests_bytes);
    out.digests.assign_image(bytes.subspan(cursor, digests_bytes));
    cursor += digests_bytes;
    const auto tag = read_u64();
    const auto count = read_u64();
    if (count > (std::uint64_t{1} << 30) || count * sizeof(std::uint32_t) > bytes.size() - cursor)
        throw std::invalid_argument("archived identity token count overflows the image");
    out.tokens.resize(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < count; ++i) {
        const auto token = read_u32();
        out.tokens[i] = static_cast<TokenId>(static_cast<std::int32_t>(token));
    }
    if (cursor != bytes.size()) throw std::invalid_argument("archived identity has trailing bytes");
    out.tag = static_cast<std::uint32_t>(tag);
    return out;
}

struct PoolImport {
    std::uint16_t pool = 0;
    std::size_t kind = 0;
    std::uint32_t pages = 0;
    std::uint32_t frontier = 0;
    context::ContextDemandId device{};
    context::ContextDemandId host{};
};

} // namespace detail

// NVMe checkpoint import: builds a Host-resident checkpoint (host-only KV pages + host-only
// state image) from an archived manifest, so the next admission decision binds it and the
// ordinary materialization path fills device memory instead of re-prefilling. Everything here
// is CPU-side record surgery inside one context transaction; the only GPU work is the empty
// record allocations the device-born allocators force, and no compute frame is used (the
// transaction rides a Context frame like release submission does).
std::optional<CheckpointHandle> Session::archive_import(const RequestBasePlan& base) {
    auto& p = *impl_;
    if (!p.archive_enabled() || p.operation || p.compute || p.pending_maintenance ||
        p.release_ticket)
        return std::nullopt;
    const auto& plan_identity = base.plan_->identity;
    const auto prompt = base.summary().prompt_tokens;

    try {
        const auto candidate = p.archive_find(1, prompt, plan_identity->digests);
        if (!candidate) return std::nullopt;
        // One import per manifest per boot; imported checkpoints retire through the normal
        // reclaim paths, and a retired import's lineage is proven stale for this session.
        if (const auto found = p.import_memo.find(candidate->manifest);
            found != p.import_memo.end())
            return p.checkpoints.valid(found->second) ? std::optional{found->second}
                                                      : std::nullopt;
        p.import_memo.emplace(candidate->manifest, CheckpointHandle{});

        auto prefetch = p.archive_prefetch(*candidate);
        if (!prefetch) return std::nullopt;
        if (prefetch->wait_for(std::chrono::milliseconds(500)) !=
            ninfer::DiskContextArchive::Prefetch::Status::Ready)
            return std::nullopt;

        // --- Manifest payload ---
        auto parsed = detail::parse_identity(prefetch->identity());
        const auto state_blob = prefetch->state();
        if (state_blob.size() < sizeof(ContinuationMetadata) + 16)
            throw std::invalid_argument("archived state blob is truncated");
        ContinuationMetadata metadata;
        std::memcpy(&metadata, state_blob.data(), sizeof(metadata));
        std::uint64_t role_raw = 0, sentinel = 0;
        std::memcpy(&role_raw, state_blob.data() + sizeof(metadata), sizeof(role_raw));
        std::memcpy(&sentinel, state_blob.data() + sizeof(metadata) + 8, sizeof(sentinel));
        if (sentinel != detail::kImportStateSentinel)
            throw std::invalid_argument("archived state blob has a bad sentinel");
        const auto state_image = state_blob.subspan(sizeof(metadata) + 16);
        if (parsed.tag != plan_identity->tag)
            throw std::invalid_argument("archived identity tag mismatch");
        if (metadata.main_frontier != candidate->frontier ||
            parsed.tokens.size() != candidate->frontier)
            throw std::invalid_argument("archived checkpoint is inconsistent");

        const bool mtp = metadata.mtp_frontier.has_value() && *metadata.mtp_frontier > 0;
        const auto main_frontier = candidate->frontier;
        const auto backend_frontier = mtp ? *metadata.mtp_frontier : 0;

        // --- Reserve: device-born page working set per pool (dropped in-transaction), Host
        // page replicas per pool, device-born state shell + its Host replica.
        context::ContextDemand demand;
        std::vector<detail::PoolImport> pool_plans;
        {
            const auto main_pages = context::kv_pages_for_frontier(main_frontier);
            detail::PoolImport main_plan{.pool = 0, .kind = 0, .pages = main_pages,
                                         .frontier = main_frontier};
            main_plan.device = demand.add_history(0, main_pages, context::ContextMemory::Device);
            main_plan.host = demand.add_history(0, main_pages, context::ContextMemory::Host);
            pool_plans.push_back(main_plan);
            if (mtp) {
                const auto backend_pages = context::kv_pages_for_frontier(backend_frontier);
                detail::PoolImport backend_plan{.pool = 1, .kind = 1, .pages = backend_pages,
                                                .frontier = backend_frontier};
                backend_plan.device =
                    demand.add_history(1, backend_pages, context::ContextMemory::Device);
                backend_plan.host =
                    demand.add_history(1, backend_pages, context::ContextMemory::Host);
                pool_plans.push_back(backend_plan);
            }
        }
        auto state_device_demand = context::StateDemand{};
        state_device_demand.count = 1;
        state_device_demand.variant = 0;
        const auto state_device_id = demand.add_state(state_device_demand);
        auto state_host_demand = context::StateDemand{};
        state_host_demand.count = 1;
        state_host_demand.variant = 0;
        state_host_demand.logical_count = 1;
        state_host_demand.memory = context::ContextMemory::Host;
        const auto state_host_id = demand.add_state(state_host_demand);
        auto reserved = p.directory.reserve(demand);
        if (!reserved.reservation || !reserved.reservation->ready()) return std::nullopt;

        context::ContextTransaction migration(p.directory);
        auto& targets = *reserved.reservation;

        // --- State image via the donor pattern: device-born shell, a D2H transfer to it as
        // the Host record allocation, archived bytes written into those records, publish
        // keeping the (never-copied) source, then the empty device replica dropped. Result:
        // a Host-only CheckpointImmutable image with archived content.
        auto& states = p.directory.states();
        const auto shell = states.reserve_destination(targets.take(state_device_id), 0);
        if (!shell) throw std::invalid_argument("import state device reservation failed");
        states.freeze(*shell);
        auto transfer = states.reserve_device_to_host(*shell, targets.take(state_host_id));
        if (!transfer) throw std::invalid_argument("import state host reservation failed");
        {
            const auto ranks = states.ranks(0);
            if (ranks.size() != 1)
                throw std::invalid_argument("import expects a single-rank state layout");
            const auto rank = ranks.front();
            const auto bytes = states.image_bytes(rank, 0);
            if (state_image.size() != bytes)
                throw std::invalid_argument("archived state image size differs from the layout");
            const auto destination = transfer->description(rank).destination_record;
            if (destination.bytes != bytes)
                throw std::invalid_argument("state host record size differs from the layout");
            if (destination.offset < 0 ||
                static_cast<std::size_t>(destination.offset) + bytes >
                    p.archive_host_arena_mut.bytes)
                throw std::invalid_argument("state host record is outside the context arena");
            std::memcpy(p.archive_host_arena_mut.data + destination.offset, state_image.data(),
                        bytes);
        }
        transfer->publish(true);
        if (!states.drop_device_replica(*shell))
            throw std::invalid_argument("import state donor drop failed");

        // --- Pages, pool by pool: device-born working set bound on the fresh-history path,
        // grown to the archived page count through the ordinary write preparation, Host
        // replicas filled from the contiguous per-kind archive span, device replicas dropped.
        std::vector<context::HistoryPrefix> prefixes;
        for (const auto& plan : pool_plans) {
            auto& pool = p.directory.pool(plan.pool);
            auto& histories = pool.histories();
            auto& pages = pool.pages();
            auto binding = histories.prepare_binding(nullptr, 0, plan.frontier, 0,
                                                     targets.take(plan.device), false, {});
            const auto history = binding.history();
            auto owner = binding.commit();
            histories.prepare_write(owner, 0, plan.frontier, migration);
            auto host_records = targets.take(plan.host);
            const auto staged_pages = prefetch->pages(plan.kind);
            const auto part_bytes = pages.layout().parts.empty()
                                        ? std::size_t{0}
                                        : pages.layout().parts.front().record.bytes;
            const auto frontier_pages = context::kv_pages_for_frontier(plan.frontier);
            if (plan.pages != frontier_pages ||
                staged_pages.size() < std::size_t{part_bytes} * frontier_pages)
                throw std::invalid_argument("archived page count differs from the frontier");
            for (std::uint32_t i = 0; i < frontier_pages; ++i) {
                const auto page = histories.logical_page(history, i);
                for (const auto& part : pages.layout().parts) {
                    auto backup = pages.prepare_host_replica(
                        page, part.rank, host_records.materialize(part.rank));
                    const auto span = pages.host_binding(page, part.rank);
                    if (span.bytes != part_bytes)
                        throw std::invalid_argument("page record size differs from the layout");
                    if (span.offset < 0 ||
                        static_cast<std::size_t>(span.offset) + part_bytes >
                            p.archive_host_arena_mut.bytes)
                        throw std::invalid_argument("page host record is outside the arena");
                    std::memcpy(p.archive_host_arena_mut.data + span.offset,
                                staged_pages.data() + std::size_t{i} * part_bytes, part_bytes);
                    // publish() validates the device coverage the write preparation set up;
                    // afterwards the empty device replica is droppable (writer released by
                    // the deactivate below, no readers exist during import).
                    backup.publish();
                    if (!pages.drop_device_replica(page, part.rank))
                        throw std::invalid_argument("import page donor drop failed");
                }
            }
            prefixes.push_back(histories.retain_prefix(history, plan.frontier));
            histories.deactivate(history, migration);
        }

        // --- Publish the checkpoint and hand it to the cache like a captured one ---
        auto continuation = context::Continuation{.image = *shell, .metadata = metadata};
        auto snapshot_identity = std::make_shared<detail::Identity>();
        snapshot_identity->tokens = std::move(parsed.tokens);
        snapshot_identity->positions = std::move(parsed.positions);
        snapshot_identity->digests = std::move(parsed.digests);
        snapshot_identity->tag = parsed.tag;
        auto slot = p.checkpoints.reserve_owned();
        if (!slot.valid()) throw std::invalid_argument("checkpoint slot unavailable");
        auto handle = slot.publish(
            {context::ContextSnapshot<ContinuationMetadata, detail::Identity>(
                 states, std::move(continuation), std::move(prefixes),
                 std::move(snapshot_identity)),
             static_cast<runtime::CheckpointRole>(role_raw)});
        migration.prepare_actions();
        migration.mark_submitted();
        migration.mark_completed();
        migration.publish();
        p.import_memo.at(candidate->manifest) = handle;
        static const bool trace = [] {
            const char* v = ::getenv("NINFER_ARCHIVE_TRACE");
            return v && v[0] == '1';
        }();
        if (trace)
            std::fprintf(stderr, "archive-import: manifest=%llu frontier=%u mtp=%d\n",
                         static_cast<unsigned long long>(candidate->manifest), main_frontier,
                         mtp ? 1 : 0);
        return handle;
    } catch (const std::exception& error) {
        static const bool trace = [] {
            const char* v = ::getenv("NINFER_ARCHIVE_TRACE");
            return v && v[0] == '1';
        }();
        if (trace) std::fprintf(stderr, "archive-import: declined: %s\n", error.what());
        return std::nullopt;
    }
}

} // namespace ninfer::models::qwen4_exp
