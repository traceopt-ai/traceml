# TraceML overhead benchmarking

TraceML measures its own runtime cost through two independent tracks, both
living under this `benchmarking/` folder — follow the links below for the
actual instructions.

```mermaid
flowchart LR
    A["Wall-clock overhead<br/>benchmarking/<br/>(coarse, paired repeats)"] -->|"answers:<br/>is there overhead?"| C[Overhead story]
    B["Attribution harness<br/>perf_benchmark/<br/>(fine, per-hook, per-baseline)"] -->|"answers:<br/>where does it come from?"| C
```

## Wall-clock overhead — `benchmarking/`

**What it measures:** end-to-end training-time cost of the full multi-node
telemetry/aggregation layer, as a single number per topology.

**Granularity:** coarse. One wall-clock and one throughput percentage per
topology; no breakdown of which internal component contributes the cost.

**Method:** paired repeats — `time torchrun ...` (native) vs
`time traceml run ...` (TraceML), alternated order to cancel thermal/
environment bias, identical args each time. Requires no TraceML-specific
tooling to reproduce; anyone can run the same two `time` commands themselves.

**Campaigns so far:**

| Date | TraceML version | Hardware | 1-GPU throughput overhead | Multi-GPU/DDP throughput overhead |
|---|---|---|---|---|
| 2026-06-11 | v0.3.1 | 2× AWS g4dn.xlarge (T4) | +1.02% | ≈0% (network-bound floor) |
| 2026-07-19 | v0.3.5 | AWS g4dn.12xlarge (T4) | +0.95% ± 0.09 | +0.41% ± 0.07 |

```mermaid
xychart-beta
    title "Throughput overhead by campaign (%)"
    x-axis ["06-11 1xT4", "06-11 2-node DDP", "07-19 1xT4", "07-19 4xT4 DDP"]
    y-axis "Overhead %" 0 --> 1.5
    bar [1.02, 0, 0.95, 0.41]
```

![Throughput across the 2026-06-11 campaign](analysis/2026-06-11_pr153_ddp_mlp_g4dn/plots/06_throughput.png)

Full write-ups: [`README.md`](README.md),
[`analysis/2026-06-11_pr153_ddp_mlp_g4dn/report.md`](analysis/2026-06-11_pr153_ddp_mlp_g4dn/report.md),
[`analysis/2026-07-19_v035_ddp_mlp_g4dn/report.md`](analysis/2026-07-19_v035_ddp_mlp_g4dn/report.md).

## Attribution / automated harness — `perf_benchmark/`

**What it measures:** where overhead comes from — which instrumentation
hook (dataloader, forward, backward, H2D, optimizer, background sampler
thread) contributes, and whether that hook runs on the training-critical
path or off-thread.

**Granularity:** fine. Three baselines (`never_init` / `trace_manual` /
`trace_auto`) × two timing modes (`step` headline / `phase` attribution),
plus a static source audit tagging every hook `critical_path: true/false`.

**Method:** JSON-config-driven automated runner + aggregator; see
[`perf_benchmark/README.md`](perf_benchmark/README.md) for the full
methodology and every reproduction command.

**Campaigns so far:**

| Date | TraceML version | Hardware | Headline |
|---|---|---|---|
| 2026-07-22 | benchmarking branch @ `58a177c` | 1/2/4× AWS g4dn.xlarge (1× T4 each), on-demand | `trace_auto` adds **< 1 ms per rank per step** (0.17–0.58 ms measured) across every topology and batch size (256/512/1024) tested |

The absolute cost stays roughly constant while the baseline step time grows
with node count, so the same overhead reads as 42% of a 1.4 ms single-node
step but only ~4% of a 9.3 ms 4-node step — lead with the ms figure, not
the percentage. Full write-up:
[`analysis/2026-07-22_clean_1_2_4node_g4dn/report.md`](analysis/2026-07-22_clean_1_2_4node_g4dn/report.md).
(An earlier 2026-07-21 campaign is retained as GIL-stress diagnostic data
only — its configs unknowingly ran the GIL stress probe in every cell; see
that report's superseded banner.)

## Reading this as an outside visitor

If you only need one number: the wall-clock track's latest campaign
(2026-07-19 row above) is the current headline overhead figure. The
attribution track exists to explain *why* that number is what it is, not
to replace it.

The main repository README's measured-overhead line links here.
