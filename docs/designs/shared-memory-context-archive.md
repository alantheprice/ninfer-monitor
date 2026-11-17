# Shared-Memory Context Archive

Status: **proposal** — not implemented. Scoped 2026-11-16.
Branch context: `port/fleet-features` @ `f6fdbe42` (file-backed archive + qwen3_5 import path live).

---

## 1. What exists today

`DiskContextArchive` (`src/core/disk_context_archive.{h,cpp}`) is a persistent tier below the
Host context tier. It stores opaque page bytes and per-checkpoint identity/state blobs, so a
cold engine can rebuild a checkpoint instead of re-prefilling the prompt.

On-disk layout:

```
<dir>/.archive.lock                              flock(LOCK_EX) target
<dir>/manifests/<16-hex-id>.ckpt                 ManifestHeader + keys + identity + state
<dir>/pages/k<kind>/<2-hex>/<32-hex-key>.pg      PageHeader + page bytes
```

Durability comes from `write → fdatasync → rename → fsync(dir)`, pages before the manifest that
references them. Cross-process sharing was added in `b30e517a`: `flock` around `scan()` and
`write_job()`, manifest ids re-read from disk under that lock, `stored_bytes_` re-measured every
60 s. The import side landed in `f6fdbe42` for `qwen3_5`, so the 27B can now restore what another
process saved.

Measured cost of the file model, from production use:

| observation | value |
|---|---|
| cache at capacity | 101 GiB, **47,232 files** (~2.1 MB average) |
| one 27B manifest | 157 MB (115 MB of it the full-MTP state blob) |
| steady state | at cap, so every archive forces an eviction |

## 2. The honest framing

Two different things are called "shared memory". Only one of them is worth building.

**Already free: one copy of the bytes in RAM.** Point the archive at tmpfs and the kernel page
cache maps the same physical pages into every process that opens a file. No second copy. This
needs **zero code changes** — verified: the archive uses no `O_DIRECT`, and its `fsync`/
`fdatasync` calls are no-ops that return success on tmpfs.

**Not free, and not achievable: sharing GPU KV.** Each engine process has its own CUDA context
and its own device KV. You cannot map another process's VRAM. Import reconstructs the checkpoint
into the importer's own host arena and device memory. The memcpy into the host arena stays in
every design.

So the win this spec is chasing is **not** "avoid a copy". It is:

1. **No per-file overhead.** 47,232 inodes, dentries, and path lookups at capacity. A shm
   segment has none of that.
2. **No syscalls per page.** `open`/`read`/`write`/`rename`/`fsync` per page becomes pointer
   arithmetic into a mapping.
3. **One fewer copy on the save side.** Today: device → staging buffer → disk. With a shared
   arena the save can stage **directly into the target slot**.
4. **Eviction stops being a filesystem operation.** This is the prime suspect in the disk-tier
   wedges (`std::filesystem::remove` in a loop under the global archive mutex,
   `evict_locked`/`delete_manifest_locked`). In shm, eviction is free-list pointer work.

Item 4 alone may justify the work, independently of performance.

## 3. Goals and non-goals

**Goals**

- Two or more engine processes on one host share archived checkpoint bytes with a single
  in-RAM copy and no filesystem round-trip.
- Restore latency and save cost no longer scale with file count.
- Eviction and commit never issue syscalls.
- Crash of one process cannot corrupt the segment for the others.

**Non-goals**

- Sharing GPU memory, CUDA contexts, or device KV. Never.
- Eliminating the copy into the importer's host arena. Not possible.
- Persistence across reboot. tmpfs and shm are both volatile; the file archive remains the
  durable option.
- Cross-host sharing. This is one machine's RAM.
- Dynamic growth. A segment is sized at creation (see §7).

## 4. Decision gate — measure before building

Phase 0 is mandatory and costs no code:

```bash
--context-disk-archive /dev/shm/ninfer-archive
```

Run the existing two-process import test against tmpfs and compare restore latency and throughput
against NVMe. That isolates "no disk I/O" from "no per-file overhead".

- If tmpfs already meets the target, **stop**. Document it and move on.
- If tmpfs is still slow at capacity (file count, dentry pressure), the gap is the per-file
  overhead and §5 is justified.
- Independently of the numbers, if disk-tier wedges recur, item 4 in §2 is reason enough.

Capacity constraint on this box: `/dev/shm` is **32 GB**, RAM is 62 GB total (~55 GB available),
and the host cache already takes 12–16 GiB. A tmpfs archive competes with the host cache for the
same pool. Size it deliberately.

## 5. Design

### 5.1 Segment layout

One POSIX shm object per archive signature, mapped by every participating process.

```
+---------------------------------------------------------------+
| SegmentHeader   magic, version, signature, capacity,          |
|                 attached[] (pid), lock, generation            |
+---------------------------------------------------------------+
| PageIndex       open-addressing hash: ArchiveKey -> PageSlot  |
+---------------------------------------------------------------+
| ManifestTable   id -> ManifestSlot, plus free list            |
+---------------------------------------------------------------+
| PageArena       fixed-size slots, page_bytes[kind] each       |
+---------------------------------------------------------------+
| BlobArena       identity + state payloads (variable size)     |
+---------------------------------------------------------------+
```

Name: `/ninfer-archive-<signature-hex>`. The signature already encodes model, layouts, and
dtypes, so flash-next and the 27B naturally get separate segments and can never collide — the
same namespacing the file archive relies on.

### 5.2 Page arena

`page_bytes[kind]` is fixed per kind, so slots are uniform and the allocator is a free-list of
`uint32_t` slot indices. No fragmentation, no size classes, O(1) alloc and free.

`PageSlot` carries `references`, `generation`, `checksum`, and the key. Pages are shared across
manifests (that is the dedup), so `references` is the eviction refcount.

### 5.3 Blob arena

Identity and state blobs are variable and large — a full-MTP 27B state blob is **115 MB**. A
naive first-fit allocator will fragment over a long run.

Recommended: size-class allocation in powers of two from 64 KiB up, with a per-class free list.
Blob sizes are stable per model, so the classes settle quickly and fragmentation stays bounded.
Reject a blob larger than the largest class rather than growing the segment.

### 5.4 Index

`ArchiveKey` is 16 bytes (`std::array<std::uint64_t, 2>`). Open addressing with linear probing,
power-of-two table size, load factor 0.7, tombstones for deletion. Sized once at creation from the
expected page count.

### 5.5 Locking

Replace `flock` with a process-shared **robust** mutex in the header:

```
pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
```

`flock` needs a file and cannot guard a mapping. A robust mutex also handles the case the file
lock handled only by accident: a process dying while holding it. On `EOWNERDEAD` the acquiring
process must repair the invariant (reclaim any slot with an odd generation, §5.6) and call
`pthread_mutex_consistent`. If it cannot, it poisons the segment and every process detaches —
better than limping on a corrupt index.

Keep the existing lock order discipline: segment lock, then any in-process mutex. Never reverse.

### 5.6 Crash consistency

The file archive gets atomicity from `rename(2)`. A mapping has no rename, so use generation
counters:

- Writer sets `generation` to an **odd** value before touching the payload.
- Writes payload, computes checksum.
- Sets `generation` to the next **even** value with release semantics.
- Reader accepts a slot only if `generation` is even **and** the checksum matches.

A process killed mid-write leaves an odd generation; the next locker reclaims it. A torn write
that somehow lands an even generation is caught by the checksum.

Manifest commit ordering is unchanged in spirit: pages valid first, then the manifest slot's
generation bump. A manifest referencing an invalid page is discarded on attach.

### 5.7 Lifecycle

- First process: `shm_open(O_CREAT|O_EXCL)`, `ftruncate(capacity)`, `mmap(MAP_SHARED)`, format the
  header and indexes.
- Later processes: `shm_open(O_RDWR)`, validate magic/version/signature, `mmap`, register their
  pid in `attached[]` under the lock.
- Detach: clear the pid slot. The last one out `shm_unlink`s.
- A process that dies without detaching leaves a stale pid. Prune on attach with `kill(pid, 0)`.
  If the segment has no live pids and no valid manifests, reformat it.

### 5.8 Sizing

Fixed at creation, no growth. Inputs: `--context-shm-capacity` (total), plus the existing
`page_bytes` and signature. Split it between page and blob arenas — a starting ratio of 3:1 is
reasonable for the 27B, where the state blob is 115 MB per checkpoint but pages dominate the
total.

Refuse to start if the requested capacity exceeds what the box can back, and surface the actual
`/dev/shm` and free-RAM numbers in the error.

### 5.9 Save and restore paths

**Save** — the staging buffer disappears. `begin_save` reserves page slots; the caller stages
device bytes **directly into the slot**; `commit` validates checksums and bumps generations. This
removes one full copy of the page set versus the file path.

**Restore** — `prefetch` resolves slots and copies from the mapping into the caller's staging for
the existing import path to consume. Still one copy, but no syscalls and no reader threads needed;
the 4-thread reader pool can go away entirely.

## 6. Integration

`ContextArchiveLink` (`src/models/common/context/disk_archive_link.{h,cpp}`) is **concrete** today
— it holds `std::unique_ptr<DiskContextArchive>` directly. Two steps:

1. **Extract an interface.** `ContextArchive` with `enabled`, `begin_save`/`commit`, `find`,
   `prefetch`, `stats`, `flush`. `FileContextArchive` wraps the existing `DiskContextArchive`
   unchanged; `ShmContextArchive` implements the interface directly.
2. **Select the backend** from options. `Options` gains `shm_name` alongside `directory`;
   `directory` set means file, `shm_name` set means shm, both set is an error.

Callers do not change: `archive_find` / `archive_prefetch` / `archive_register` in
`src/models/qwen3_5/session/checkpoints_archive.cpp` and the `qwen4_exp` equivalents all go
through the link. The `qwen3_5` import path from `f6fdbe42` is backend-agnostic.

`write_bytes_per_second` is meaningless for RAM (its purpose is SSD wear and I/O pacing). Keep it
accepted and ignored, or drop it for the shm backend.

## 7. Phasing

| phase | work | gate |
|---|---|---|
| 0 | tmpfs run, measure, compare against NVMe | decides whether to continue |
| 1 | extract `ContextArchive`; `FileContextArchive` wraps current code | no behaviour change, tests green |
| 2 | `ShmContextArchive`: layout, allocators, index, robust lock, save/restore | two-process import works |
| 3 | crash recovery, pid pruning, poison path | kill -9 tests pass |
| 4 | eviction under the new allocator; delete the reader thread pool | capacity tests pass |

Phase 1 is worth doing on its own merits even if shm is never built — it makes the archive
swappable and testable.

## 8. Testing

Extend `tests/test_disk_context_archive.cpp` and `tests/test_disk_archive_link.cpp`, which already
carry `test_shared_directory_across_processes` (4 forked writers, one directory, every checkpoint
survives byte-correct; fails 5/5 with the lock disabled).

New cases:

- **Two-process import.** A saves, B (cold) imports, bytes match. The existing end-to-end check
  (`archive-import: manifest=N frontier=M`, then `cached_tokens`) is the acceptance signal.
- **N writers, M readers concurrently.** No corruption, no lost manifest.
- **Writer killed mid-write.** Odd generation reclaimed; segment still usable.
- **Holder killed while holding the lock.** `EOWNERDEAD` path repairs and continues.
- **Capacity exhaustion.** Eviction frees slots; no leak after a long churn run (assert free-list
  length returns to baseline).
- **Stale pid pruning.** Kill a process without detaching; next attach prunes it.
- **Signature isolation.** Two different signatures cannot see each other's segments.
- **Parity.** Same workload, file vs shm: identical restored bytes, and a measured restore-latency
  delta.

## 9. Risks

| risk | mitigation |
|---|---|
| Robust-mutex recovery is subtle; a bug corrupts every process at once | Fuzz the kill-during-write path; poison rather than limp |
| Fixed capacity cannot grow | Size deliberately; refuse to start rather than silently truncate |
| Blob-arena fragmentation over long runs | Size classes, not first-fit; assert on free-list drift |
| A segment outlives a crashed creator | pid pruning + reformat when empty |
| `/dev/shm` is only 32 GB here | Enforce the cap; report real numbers in the error |
| Two copies of the same data (host cache + segment) | Document the RAM budget; this is a sizing choice, not a bug |

## 10. Open questions

1. **Does the save path really avoid a copy?** The device→host staging is driven by the caller,
   which currently stages into a `ContextArchiveLink`-owned buffer. Staging straight into the
   mapping requires the caller to accept a destination span from the archive. Small API change,
   needs confirming against `checkpoints_archive.cpp`.
2. **Should the file archive stay the default?** It is the only durable option and it already
   works. Recommendation: yes, shm is opt-in.
3. **Is eviction actually the wedge cause?** Unproven. Building shm would remove the suspect by
   construction, but a stack trace from a live wedge would confirm it first — see
   `disk-tier-wedge-3rd-2026-11-01`, which notes `gdb -p` needs sudo and the opportunity is lost
   once the process is killed.
4. **Worth it for one GPU?** On a single card there is no second engine to share with, so the
   only wins are syscalls and eviction cost. The multi-GPU case is where sharing actually pays.

## 11. Summary

The copy is not the problem and never will be — GPU KV stays per-process. The problems are 47,232
files, a syscall per page, a redundant staging copy on save, and `unlink()` in a loop under a
global lock. A shared segment removes all four. tmpfs removes the first two for free, so measure
that first and only build §5 if the numbers or the wedges demand it.
