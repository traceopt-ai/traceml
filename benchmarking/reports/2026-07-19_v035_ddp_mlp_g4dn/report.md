# v0.3.5 Overhead Rerun: TraceML Multi-Node Aggregator (`ddp_mlp_e2e`)

- **Date:** 2026-07-19 · **Version:** TraceML v0.3.5
- **Scope:** Rerun of the PR #153 base multi-node telemetry/aggregation
  layer overhead study (see
  [`../2026-06-11_pr153_ddp_mlp_g4dn/report.md`](../2026-06-11_pr153_ddp_mlp_g4dn/report.md))
  on a newer TraceML version and different hardware, to confirm the
  overhead figure still holds.
- **Workload:** [`benchmarking/wallclock_overhead/ddp_mlp_e2e.py`](../../wallclock_overhead/ddp_mlp_e2e.py)
- **Hardware:** 1× AWS `g4dn.12xlarge` (4× Tesla T4), `eu-central-1`,
  spot instance.

## Executive summary

| Topology | Throughput overhead | Wall-clock overhead |
|---|---|---|
| **1× T4** | **+0.95% ± 0.09** | +2.16% ± 0.34 |
| **4× T4 (single-node DDP)** | **+0.41% ± 0.07** | +1.76% ± 0.36 |

Consistent with the 2026-06-11 v0.3.1 study (+1.02% single-GPU, ≈0% DDP
network-bound floor). Wall-clock overhead is higher than throughput
overhead here because these runs use a fixed 300s duration with a
non-trivial fixed startup cost (aggregator spin-up + finalize) relative
to that shorter window; see the 2026-06-11 report's caveat #1 for why
wall and throughput diverge on fixed-duration runs. Steady-state per-step
cost is the throughput row.

## Methodology

5 paired runs per topology (native vs TraceML), alternated order,
identical args, 300s per trial. Per-run detail in
[`per_run_breakdown.csv`](per_run_breakdown.csv) (TraceML-mode rows only,
same convention as the 2026-06-11 campaign — native-mode aggregate wall
times below).

**Native-mode aggregates** (from the same paired campaign, not present in
the CSV — see this campaign's `per_run_breakdown.csv` note):

| Topology | Native avg wall (s) | TraceML avg wall (s) | Wall overhead |
|---|---|---|---|
| 1× T4 | 313.97 | 320.76 | 2.16% |
| 4× T4 DDP | 314.90 | 320.44 | 1.76% |

## Methodology differences from the 2026-06-11 campaign (read before comparing directly)

1. **Hardware differs:** single `g4dn.12xlarge` (4 GPUs, one host) here vs
   two separate `g4dn.xlarge` hosts (1 GPU each) in 2026-06-11 — this
   campaign's "4× T4" is single-node DDP, not the earlier campaign's
   2-node DDP. Not directly comparable topology-for-topology.
2. **Telemetry schema evolved (v0.3.1 → v0.3.5):** the per-run summary
   card in this version reports an aggregated `Compute` phase rather than
   the separate `fwd_ms`/`bwd_ms`/`opt_ms` breakdown the 2026-06-11 CSV
   has, and does not expose `gpu_power_avg_w` or
   `gpu_mem_reserved_gb` in the same place. `per_run_breakdown.csv`'s
   columns reflect what this version's telemetry card actually exposes;
   they are not a 1:1 match to the 2026-06-11 CSV's columns.
3. **No plots regenerated.** The 2026-06-11 campaign's plots came from an
   internal notebook not checked into this repo; reproducing that exact
   plot set for this campaign was out of scope for this pass. The raw
   per-run data needed to build them is in `per_run_breakdown.csv` for
   whoever picks this up next.

## Conclusion

Overhead stays within the same low-single-digit-percent range on v0.3.5
that it was on v0.3.1 — no regression across versions or the hardware
change.
