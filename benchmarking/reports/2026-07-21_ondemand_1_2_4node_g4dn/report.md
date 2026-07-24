# 1/2/4-Node Scaling Campaign (`perf_benchmark`, on-demand g4dn.xlarge)

> **Superseded 2026-07-22.** `run_benchmark.py` enables `--gil-probe` by
> default — an unthrottled, continuously-running CPU thread that competes
> for the GIL in every cell, including `never_init` — and none of the
> configs below overrode it. Per @abhinavsriva (issue #233, PR #230) and
> confirmed independently on a clean rerun, the numbers in this report are
> GIL-stress diagnostic data, not normal TraceML overhead evidence. See
> [`../2026-07-22_clean_1_2_4node_g4dn/report.md`](../2026-07-22_clean_1_2_4node_g4dn/report.md)
> for the clean rerun with `gil_probe: false`.

- **Date:** 2026-07-21
- **Scope:** First real run of Abhinav's `perf_benchmark` harness (added
  2026-07-20) across three topologies — 1 node, 2 nodes, 4 nodes — each
  1× T4 per node, `eu-central-1`, **on-demand** (not spot; see note below).
- **Workload:** `tiny_mlp` (bs=256, synthetic loader), the only workload
  common to all three runs (single-node also covered `tiny_transformer`
  under phase-mode; not repeated at 2/4-node).
- **Run IDs:** `aws_full_t4_phaseonly_20260721_074055` (1-node, phase
  timing-mode), `aws_2node-ondemand_t4_20260721_171556` (2-node, step
  timing-mode), `aws_4node-ondemand_t4_20260721_173544` (4-node, step
  timing-mode). Raw per-rank data + full summaries: `1node_summary.csv`,
  `2node_summary.csv`, `4node_summary.csv` in this folder; full raw
  traces available on request (multi-GB, not checked into git — ask for
  the S3 paths).

## Executive summary — total step time overhead vs `never_init`, `tiny_mlp`

| Topology | Baseline median (ms) | `trace_auto` overhead | `trace_manual` overhead |
|---|---:|---|---|
| 1 node | 246.39 | **+29.56%** (real signal) | +10.46% (**within noise floor** — not distinguishable from run-to-run variance) |
| 2 node | 247.98 | **+27.96%** (real signal) | +6.66% (**within noise floor**) |
| 4 node | 216.16 | **+34.43%** (real signal) | **+12.72%** (real signal — first topology where `trace_manual` clears the noise floor) |

"Real signal" / "within noise floor" is the harness's own
`within_baseline_noise` computation (delta vs. baseline compared against
that cell's noise floor from repeated `never_init` runs), not a subjective
call — see the `within_baseline_noise` column in each CSV.

**Read before treating this as a scaling trend:** each topology is a
*single* campaign (3 repeats per cell). `trace_auto` overhead is
28-34% at every scale tested with no clean monotonic pattern (2-node is
slightly *lower* than 1-node) — flat-ish, not a clear function of node
count from these 3 points. `trace_manual` crossing the noise floor
between 2-node and 4-node is a real observation but a sample size of one
campaign per topology; it would need repeat campaigns per topology to
call it a confirmed trend rather than run-to-run variance landing on
either side of the noise floor.

## Methodology differences (read before comparing 1-node to 2/4-node directly)

1. **Timing mode differs.** The 1-node run used `timing_mode: phase`
   (decomposes each step into `dataloader_ms` / `h2d_ms` / `forward_ms` /
   `backward_ms` / `optimizer_step_ms` / etc., each independently
   instrumented). The 2-node and 4-node runs used `timing_mode: step`
   (single wall-clock measurement per step, no phase decomposition). This
   means the 1-node `trace_auto`/`trace_manual` numbers include the cost
   of phase-boundary instrumentation itself, which the 2/4-node numbers
   never pay — the three overhead percentages are not a clean apples-to-
   apples comparison of "the same instrumentation at different scale."
   The `never_init` (no tracing at all) baselines are close across all
   three (246-248ms for 1/2-node) which is consistent with this — the
   timing-mode difference should mainly affect the *traced* cells, not
   the untraced baseline.
2. **The 4-node `never_init` baseline is ~13% faster** (216.16ms vs
   ~247ms at 1/2-node) with no confirmed root cause. Could be instance-
   to-instance hardware variance, or a genuine effect of running 4 ranks
   instead of 1-2 (e.g. some fixed cost amortizing differently) — not
   isolated in this campaign. Worth a repeat run if this matters for a
   published number, since it changes the reference point the 4-node
   percentages are measured against.
3. **Launched on-demand, not spot** — a change made mid-session after
   repeated spot capacity/reclaim failures (see `AWS_INFRA.md` §2a for the
   full incident history). Not expected to affect the timing numbers
   themselves, but worth noting since prior campaigns in this repo used
   spot.
4. **No plots this campaign** — same gap the 2026-07-19 rerun's own
   report flagged for 2026-06-11: those plots came from a Jupyter
   notebook (in a different repo, `traceopt-viewer/study/`) whose
   markdown output and images were hand-extracted into that campaign's
   folder, not the notebook itself. That notebook targets the core
   TraceML `final_summary.json` schema; this `perf_benchmark` harness
   (added 2026-07-20) has its own, different `summary.json`/`summary.csv`
   schema, so the existing notebook wouldn't plot it without rework
   anyway. Raw data for charts is in `1node_summary.csv` / `2node_summary.csv`
   / `4node_summary.csv` for whoever picks this up next.

## Known artifact: phase-attribution percentages near a zero baseline

This was flagged internally during the session as a methodology concern
*before* any GPU data existed, specifically deferred pending confirmation
on real hardware ("let's open an issue on this later after confirming it
reproduces on GPU"). **It reproduces**, so it should become an actual
issue now.

`1node_summary.csv`'s phase-attribution rows show entries like:

| Phase | `never_init` baseline (ms) | `trace_auto` (ms) | Overhead |
|---|---:|---:|---|
| `trace_context_enter_ms` | 0.0114 | 10.32 | **90,085%** |
| `trace_context_exit_ms` | 0.0122 | 10.59 | **86,591%** |

The *absolute* overhead (~10.3ms of real phase-boundary instrumentation
cost) is a legitimate number worth reporting. The *percentage* is
meaningless — it's an artifact of dividing by a near-zero baseline
(0.01ms), not a 900x slowdown. Recommend: report absolute ms for any
phase whose baseline is below some floor (e.g. the existing noise-floor
threshold already computed per-cell) instead of a percentage, or flag
those rows as "N/A (baseline too small for a percentage)" in the
generated report.

## Rank skew and GIL probe (2/4-node only — not measured at 1-node)

| Topology | Median rank skew (ms), `trace_auto` | GIL victim probe median (ms) |
|---|---:|---:|
| 2 node | 15.94 | 0.109 |
| 4 node | 29.31 | 0.104 |

Skew roughly doubles from 2 to 4 nodes (more ranks, more chance of one
straggler) — expected direction, not surprising. GIL victim probe stays
flat (~0.1ms) across both topologies and across cells within each — no
GIL-contention signal visible at this scale/workload. Not conclusive
either way for the Uber GIL-contention question; `tiny_mlp` on a single
T4 per rank is a lightweight workload, this may need a heavier model or
more ranks per node to stress the GIL meaningfully.

## Conclusion

TraceML's measured overhead (`trace_auto`) sits in the high-20s to
mid-30s percent range across 1, 2, and 4 nodes on this synthetic
`tiny_mlp` workload, with no clean scaling trend from 3 single-campaign
data points. `trace_manual` overhead is smaller and sits right at the
noise floor at low node counts, crossing it at 4 nodes. Two things worth
resolving before treating any of these numbers as final: the phase-vs-step
timing-mode mismatch between the 1-node and 2/4-node runs (methodology
difference, not a scaling effect), and the unexplained faster baseline at
4 nodes. The phase-attribution percentage artifact is confirmed
reproducing on real GPU hardware and should be filed as an issue against
the report-generation code in `aggregate_results.py`.
