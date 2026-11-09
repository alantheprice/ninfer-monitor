data: H200 MMA-path deep-dive (2026-11-09, from GPU session on pod qlmbft2xsaqyoz)

The vector-NVFP4 unblock (532c83fd) leaves perf on the table. Proper MMA-path fix has
TWO defects to solve, not one:

1. stage_inputs stages raw e2m1 codes; the software mma_nvfp4_e4m3 (mma.cuh arch<1200)
   reinterprets fragment words as f16x2 -> NaN. (Known.)

2. k-extent/scale contract: contract_tile calls mma_nvfp4_e4m3 ONCE per (ki, m, n) with
   fragments loaded for a 64-code span (a col = ki*32 raw bytes, b col stride 16 raw
   bytes, as[m] = ONE word = 4 scale bytes = hardware scale_vec::4X semantics of
   m16n8k64). The software path is a single mma_bf16 m16n8k16 = k16, folding all 4
   groups' scales (fa*fb product of one byte pair) onto one group's product.
   A correct software path must unroll g=0..3 per (ki,n): ldmatrix_x4 a at widened
   bytes g*32 + (lane>>4)*16, ldmatrix_x2 b at g*32 + ((lane>>3)&1)*16, mma_bf16, fold
   with scale BYTE g extracted from as[m]/bs words (staged scale bytes are LINEAR in g
   within the 8-byte copy: byte j = group (kt*8 + k64*4 + j) of that row).

3. OPEN QUESTION (blocks implementation): staged scale tile covers 32 gate + 32 up rows
   per k64 chunk (staging loop row = task % 32, part selects gate/up), but tile rows are
   128 (Gate: 64 gate + 64 up). Consumer word (ki*128 + (sr%32)*4 + sr/32) for sr in
   32..63 reads bytes 4..7 of row (sr-32) — i.e. groups 4..7 of a DIFFERENT row. Either
   the hardware scale_vec::4X consumes scales at a coarser row granularity than per-row
   (plausible: 4X = 4 rows share?), or rows 32..63 reuse — needs a GPU probe: quantize
   a known input, dump acc for a single-group case, compare per-row scales.

4. Empirical reference for the widened contract: softmax_attention
   kernels/causal/nvfp4/{tiled_mma,grouped_mma}.cuh pre-dequantize BOTH operands to f16
   smem (scales folded INTO the f16 values via nvfp4_dequant_f16x16) then run plain
   mma_f16 with no scale fold. That pattern is PROVEN on this codebase (attention path
   works). Option: mirror it for MoE — widen with scale-fold into f16, plain mma_f16,
   skip mma_nvfp4_e4m3 entirely. Requires the widened tile to hold SCALED f16 (no
   separate scale fold at mma time) — sidesteps the whole per-group scale pairing.
   Cost: f16 (11-bit mantissa) vs bf16-fold — numerics need refdiff validation, but the
   attention path already proves the pattern acceptable.

Recommendation: implement option 4 on a worktree, GPU-probe the scale-row question with
a single-expert single-group kernel, refdiff vs tests/goldens/h200-refdiff-pack.json
(secured, 30 cases, 100% self-consistency baseline), then throughput.

## H200 validation results (2026-11-09, pod qlmbft2xsaqyoz, GPU-verified)

- 45f3f933's widening builds on sm_90a and serves correctly: coherent greedy output,
  clean 1024-token generation finish=stop, 17x23->391. No crash, no corruption.
- Measured on H200 (128-tok streams, greedy): C=1 long-gen 88-98 tok/s across runs
  (98.0 first, 87.6 repeat — real run variance), short answers ~100 tok/s,
  C=2 ~110-122, C=4 ~153, C=8 aggregate 170-173 tok/s. Vector-unblock baseline was
  C=1 88 / C=8 152, so the MMA path adds ~0-11% C=1 (within variance) and ~+13% C=8.
- MMA-path replay vs the vector-unblock h200 goldens: exact=1/30, prefix12 5/30 (17%)
  — expected: different accumulation numerics. Recapture goldens on the MMA build
  before using them as a regression gate.
- Do NOT expect the 45f3f933 approach to close more of the B200 gap: the single-stage
  schedule + software bf16 MMA on 4x-widened tiles is inherently modest. The measured
  numbers confirm: the remaining C=8 gap to B200 (173 vs 641) is batching/scheduler
  and Hopper-native kernel tuning, per the README optimization list.
