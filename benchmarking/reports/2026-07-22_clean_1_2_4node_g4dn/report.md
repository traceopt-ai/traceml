# 1/2/4-Node Scaling Campaign — Clean Rerun (`perf_benchmark`, on-demand g4dn.xlarge)

- **Date:** 2026-07-22
- **Supersedes:** [`../2026-07-21_ondemand_1_2_4node_g4dn/report.md`](../2026-07-21_ondemand_1_2_4node_g4dn/report.md).
  That campaign's headline numbers were contaminated — `run_benchmark.py` enables
  `--gil-probe` by default (an unthrottled, continuously-running CPU thread that
  competes for the GIL in every cell, including `never_init`), and none of the
  configs used for that campaign overrode it. Confirmed via
  [issue #233](https://github.com/traceopt-ai/traceml/issues/233) and PR #230
  discussion with @abhinavsriva. This rerun has `gil_probe: false` explicitly set
  in every config.
- **Scope:** Same three topologies (1/2/4 nodes, 1× T4 per node, on-demand,
  `eu-central-1`) as the superseded campaign, plus a batch-size sweep (256/512/1024)
  at every topology — the original only tested bs=256 at 1-node and had a
  path-collision bug that silently dropped bs=512/1024 at 2-node on the first
  attempt (fixed by using a separate config/run-id per batch size, matching
  Abhinav's own `run_clean_single_gpu_campaign.sh` pattern).
- **Workload:** `tiny_mlp`, `step` timing mode throughout (no phase-vs-step
  mismatch this time — see caveat below on the separate phase-mode-only run
  used for the issue #233 validation).
- **Repeats:** 10 per cell (up from 3 in the superseded campaign), matching
  Abhinav's own clean methodology.
- **Aggregation:** uses the harness's per-repeat-median statistics (added in
  the `dbfdf2d` commit) as the primary unit — median of 10 independent
  process-repeat medians, not pooled per-step samples.

## Results

Percentages alone are misleading here: the **absolute** overhead stays in a
narrow band (~0.17-0.58ms) across every topology and batch size tested, while
the percentage swings by an order of magnitude (4-42%) purely because the
baseline step time itself grows 1.4ms → 9.3ms as node count increases (more
cross-node gradient sync). A bare percentage table reads as "overhead shrinks
dramatically at scale" when the more accurate story is "the fixed per-step
cost stays roughly the same; it's just a shrinking fraction of a growing
baseline." Leading with ms for that reason, percentage as secondary context.

| Batch size | Topology | Baseline (ms) | `trace_auto` overhead | `trace_manual` overhead |
|---|---|---:|---|---|
| 256 | 1-node | 1.3775 | **+0.578 ms (42.0%)** — real signal | +0.244 ms (17.7%) — real signal |
| 256 | 2-node | 5.0724 | +0.396 ms (7.8%) — within noise | +0.115 ms (2.3%) — within noise |
| 256 | 4-node | 9.2348 | +0.378 ms (4.1%) — within noise | +0.080 ms (0.9%) — within noise |
| 512 | 1-node | 1.5060 | **+0.520 ms (34.5%)** — real signal | +0.222 ms (14.8%) — real signal |
| 512 | 2-node | 5.0185 | +0.447 ms (8.9%) — within noise | +0.195 ms (3.9%) — within noise |
| 512 | 4-node | 8.8797 | +0.431 ms (4.9%) — within noise | +0.277 ms (3.1%) — within noise |
| 1024 | 1-node | 2.4439 | +0.172 ms (7.0%) — real signal | +0.049 ms (2.0%) — within noise |
| 1024 | 2-node | 5.3442 | +0.390 ms (7.3%) — within noise | +0.128 ms (2.4%) — within noise |
| 1024 | 4-node | 9.2670 | +0.396 ms (4.3%) — within noise | +0.183 ms (2.0%) — within noise |

"Real signal" / "within noise" is the harness's own `within_baseline_noise`
computation (delta vs. that cell's noise floor from repeated `never_init`
measurements), not a subjective call — see the `within_baseline_noise` column
in each CSV in this folder.

## Key findings

1. **`trace_auto`'s absolute overhead is roughly constant, ~0.17-0.58ms**,
   across all 9 conditions — 2-node and 4-node cluster tightly around
   0.38-0.45ms; 1-node shows more spread (0.17ms at bs=1024 up to 0.58ms at
   bs=256). This looks like a largely fixed per-step instrumentation cost,
   not something that scales with node count or batch size.
2. **Baseline step time scales cleanly with node count** — ~1.4-2.4ms
   (1-node) → ~5.0-5.3ms (2-node) → ~9.2-9.3ms (4-node), roughly doubling
   each time. This is the real cost of cross-node gradient synchronization
   growing with more ranks — not a TraceML cost, and it's *why* the same
   absolute overhead reads as a shrinking percentage at larger scale.
3. **Every multi-node measurement (2-node and 4-node, all batch sizes,
   both cells) falls within the noise floor** — cross-machine timing jitter
   is large enough at N=10 repeats to swallow the ~0.4ms signal entirely.
   Only 1-node's `trace_auto` resolves as real signal at every batch size
   (`trace_manual` resolves at bs=256/512 but not bs=1024, where the
   absolute overhead shrinks to ~0.05ms).

## Possible mechanism (hypothesis, not confirmed)

The pattern above is consistent with a specific explanation, offered here as
a hypothesis worth testing further — not as an established conclusion.

**`trace_auto`'s overhead looks like a fixed, CPU-side bookkeeping cost per
step** (trace-context enter/exit, sample recording), largely independent of
how much compute or communication happens inside that step. This is the
usual behavior of hook/tracer overhead: proportional to call-site count, not
to the amount of work between calls.

**This would explain the 1-node batch-size trend.** At 1-node, `trace_auto`
overhead shrinks monotonically as batch size grows (256→512→1024:
0.578→0.520→0.172ms). CUDA execution is asynchronous, so a larger batch
means more GPU compute time per step, which gives more "slack" for that
fixed CPU-side cost to overlap with an already-busy GPU stream instead of
sitting exposed on the wall-clock critical path. At bs=256 the GPU finishes
quickly, so the same fixed CPU cost has less compute to hide behind.

**This also explains why the 2-node/4-node numbers don't show as clean a
batch-size trend.** At those topologies the baseline step time is dominated
by cross-node gradient-sync wait, not local compute — there's less "compute
headroom" for the overlap effect to hide behind, so the fixed cost shows up
more directly regardless of batch size (hence the tighter 0.38-0.45ms
clustering across all three batch sizes at 2-node and 4-node, versus 1-node's
wider 0.17-0.58ms spread).

**Independent corroboration:** Abhinav made the same observation from his own
data in the PR #230 discussion — `tiny_mlp`@256 is "intentionally a
fixed-overhead stress case," and the *absolute* (not just percentage) cost
drops substantially at bs=1024 in his clean single-node numbers too.

**Important: "within noise" at 2-node/4-node is not evidence the overhead is
absent at scale.** It means the ~0.4ms signal isn't currently resolvable
against a noise floor that grew to 1.4-4.6ms from cross-machine timing
jitter — not that the underlying cost shrank to zero. The absolute-ms
numbers are consistent with the same roughly-fixed cost persisting at 2 and
4 nodes; there just isn't enough statistical power (10 repeats, noisy
cross-machine timing) to confirm that the way 1-node's signal is confirmed.
Absence of a resolvable signal is not the same claim as evidence of no
signal, and neither this report nor anyone citing it should conflate the two.

**A real test of this hypothesis** would be a heavier workload than
`tiny_mlp` (more forward/backward compute per step, same instrumentation).
If the hypothesis is right, overhead should keep shrinking as compute per
step grows. If it stays flat regardless of workload weight, the cost is
more likely something synchronous that blocks the GPU stream rather than
hideable CPU-side work — a meaningfully different and more concerning
finding for TraceML's design. Not run as part of this campaign.

## Caveats (read before citing further)

1. **Cell order is still block-ordered, not rotated**, for the 2-node and
   4-node runs (`never_init` ×10, then `trace_manual` ×10, then
   `trace_auto` ×10) — this campaign only fixed `gil_probe`; a per-repeat
   cell-rotation script (matching Abhinav's single-GPU methodology) doesn't
   exist yet for multi-node. Time-of-run/system-drift bias isn't ruled out
   for the multi-node rows.
2. **The issue #233 phase-attribution validation used a separate, single-node,
   phase-timing-mode run** (`aws_clean_phase_mode_20260722_191149`, bs=256
   only) — not part of the topology comparison above, since phase mode isn't
   tested at 2/4-node here. See the issue and PR #240 for that data.
3. **10 repeats per cell, one campaign per topology/batch-size** — same
   sample-size caveat as the superseded report: enough to resolve 1-node's
   signal clearly, not enough (given multi-node's larger noise floor) to
   rule out multi-node's overhead being real but currently unresolved,
   versus genuinely negligible at that noise scale.

## Conclusion

The GIL-probe contamination fully explains why the superseded campaign's
numbers didn't match a sane mental model of TraceML overhead. With it fixed,
`trace_auto`'s cost looks like a small, roughly-constant per-step tax
(sub-millisecond) that becomes proportionally smaller as the baseline step
time grows — whether from bigger batches or more nodes. Only 1-node currently
gives a clean, resolvable signal; the multi-node numbers are consistent with
that same fixed cost but aren't yet statistically distinguishable from
cross-machine timing noise at this repeat count.
