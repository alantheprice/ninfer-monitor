# B200 Platform Bring-Up Plan

Branch: `port/fleet-features` @ `13cbb7a9` (pushed, clean)
Build: `build-b200/apps/ninfer-serve` (sm_100a, links clean, emulation seed for FP4)
Goal: new platform, ~100 concurrent users at coding context (64-128K), model `qwen3.8-flash-next`

---

## Phase 0 — HERE (no GPU needed)

| # | Task | Status |
|---|---|---|
| 0.1 | b200 target scaffold (platform, 23 kernel subtrees, lane ceiling 128) | ✅ ee287c45 |
| 0.2 | FP8/nvfp4 emulation seed (mma.cuh, arch-keyed) | ✅ ee287c45 |
| 0.3 | ISA probes (block-scale MMA / tcgen05 / e2m1 cvt absent on sm_100a) | ✅ probed+recorded |
| 0.4 | Artifact staging (see §Artifact below) | ✅ done |
| 0.5 | **Rent-box provisioning script** — one-shot: apt deps + CUDA driver check + pull artifact + pull engine binary + smoke `--help` | TODO before rental |
| 0.6 | **Bench harness port** — the 4×512-tok prose + 64-tool agent protocol runs over plain HTTP; works anywhere. Add `--base-url` so it can target the rented box | TODO (small) |
| 0.7 | **Reference-diff pack** — frozen 50-prompt set + expected-token JSON captured from the 6000 Pro, to prove emulation numerics on the rental before kernel work starts | TODO (½ day here) |
| 0.8 | tcgen05 study pass — read the 3 PTX docs, sketch the tmem alloc/ldst layout for our tile shapes | can start now, free |

**Everything in Phase 0 is doable on this box.** The more that lands here, the fewer rented hours get burned on non-GPU work.

## Phase 1 — RENTED BOX, functional validation (~4-8 rented hours)

Objective: prove the emulation seed boots, serves correct output, and measure its honest speed.

1. Provision via 0.5 script (artifact streaming starts BEFORE first hour ideally — pre-warm)
2. Boot engine: 1 lane first (correctness), then 16, then 128 (graphs capture behavior at big lane counts is unknown #1)
3. Run reference-diff pack → expect byte-or-near parity vs 6000 Pro (emulation preserves per-fragment contract; accumulation order differs → assert max-token-match ≥ 99%, not byte parity)
4. Bench: C=1/4/16/128 prose + 64-tool agent shape → this is the BASELINE the tcgen05 work must beat
5. Capture /usage + ncu profiles of the emulated GEMMs (the ncu sections name exactly which inner ops dominate → tells us what tcgen05 must replace first)
6. Decision gate: emulation baseline ≥ acceptable? If emulated throughput already serves 100 users at target latency, tcgen05 becomes ROI-optional, not blocking

## Phase 2 — RENTED BOX, tcgen05 native kernels (the real work, 15-40 rented hours)

Order chosen by profile data from Phase 1, expected roughly:

1. `tcgen05.alloc`/`relinquish`/tmem lifetime wrapper + tests (sm_100a only; guard per-target like mma.cuh)
2. tcgen05.mma nvfp4 block-scale for the **MoE grouped GEMM** (biggest win: 512 experts × small tiles)
3. tcgen05.mma for dense nvfp4 projection/add/split/swiglu shape kernels
4. tcgen05 fp8 path (attention QKV/proj + KV ops) if profiles justify
5. CUDA-graph capture at 128 lanes with the new kernels (watch tmem alloc across graph nodes — known trap area)
6. Full reference-diff + bench re-run; compare against Phase 1 baseline table

Each step: implement → unit test on-box → integrate → same reference-diff gate. The per-target kernel-tree pattern (rtx5090/b200 dirs) means zero cmake churn per step.

## Phase 3 — SWAP TO OWNED HARDWARE (days, not rented)

When B200s are bought: image box, run same Phase 1 validation locally, then fleet cutover.
The rented box's only remaining role: regression CI if we keep it reserved by the hour.

## Pitfalls already banked (from tonight)

- `#if defined(__CUDA_ARCH__)` hides includes from the host pass → hoist includes above guards
- sm_100a has NO warp-level FP4 MMA and NO e2m1 cvt — tcgen05 is the only native FP4 path
- public headers must not include generated platform.h (compile-definition instead)
- lane ceiling rides as plain -D define; rtx5090/dgx_spark builds byte-identical to before

---

## Artifact staging (DONE)

- Source of truth: `/home/aprice/models/flash-next-ninfer/` (108.4 GB: 32 GB × 3 + 12.4 GB parts + manifest)
- artifact_id `57cf9cd010e3475fac9481e8fc66fc0f`, container v3, components text+vision+mtp
- Local sha256 manifest: `/home/aprice/models/flash-next-ninfer/sha256-manifest.txt` (print with `cd dir && sha256sum *.part-* qwen3_8_flash_next.ninfer > sha256-manifest.txt`)
- Staged to `gdrive:ninfer-artifacts/flash-next/` via rclone (see commands below) — verify with `rclone lsl gdrive:ninfer-artifacts/flash-next | wc -l` = 6 files

### Streaming commands for the rented box (paste at provisioning)

```bash
# on rented box (Ubuntu 22.04/24.04 x86_64, NVIDIA driver >= 570 for sm_100a):
curl -sO https://downloads.rclone.org/rclone-current-linux-amd64.zip && unzip -o rclone*.zip && sudo mv rclone*/rclone /usr/local/bin/
rclone config create gdrive drive    # then paste token: (rclone will print a URL — do it interactively)
mkdir -p /srv/ninfer-artifact && cd /srv/ninfer-artifact
rclone copy gdrive:ninfer-artifacts/flash-next . --transfers 8 --checkers 16 --drive-chunk-size 512M --progress
sha256sum -c sha256-manifest.txt
```

Alternative if gdrive throttles (DRIVE daily quota ~750 GB single-user): pre-stage to any object store reachable at ≥100 MB/s; the 108 GB is split into 4 files so `aria2c -x8` per file also works from any HTTPS host.

**Rule of thumb: 108 GB at ~200 MB/s = ~10 min; at gdrive-throttled 50 MB/s = ~35 min ≈ $4-5 of wasted rental. Stage before the clock starts.**

## Rental target specs

- 1× B200 (SXM or PCIe), 180 GB HBM3e, CUDA driver ≥ 570.x, Ubuntu 22.04+, root ok
- 16 vCPU / 100 GB+ RAM preferred (engine host-pins KV staging; exact host footprint TBD in Phase 1)
- 1 TB+ NVMe scratch (artifact 108 GB + disk-archive tiers)
- Plain HTTP reachability from your laptop for the bench harness (SSH tunnel fine)
- Do NOT rent 8-GPU nodes for Phase 1-2 — engine is single-rank; waste of money
