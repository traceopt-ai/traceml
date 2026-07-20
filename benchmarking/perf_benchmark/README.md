# TraceML Overhead Benchmark

Public, reproducible scripts for measuring TraceML runtime overhead. All code
is under `perf_benchmark/`; TraceML source is not modified.

## Purpose

Measure TraceML cost where it can appear:

- training-thread overhead,
- background collector/GIL tail effects,
- per-rank skew,
- TraceML artifact and SQLite growth.

The benchmark reports distributions and bounds. It never reports "zero
overhead"; deltas inside baseline noise are reported as `within noise <= X ms`.

## Methodology

- `step` timing: headline result, one CUDA sync before/after the full step.
- `phase` timing: attribution result, CUDA sync around each phase.
- Baseline: `never_init`.
- TraceML cells: `trace_manual`, `trace_auto`.
- Warmup steps are discarded.
- CPU runs are smoke tests only.

## Layout

```text
phase1_cost_model/   static source audit
phase2_design/       methodology and configs
phase3_benchmark/    runnable benchmark and aggregation
common/              shared helpers
results/             generated outputs
```

## Requirements

- Same repo commit on every node.
- Same Python/PyTorch/CUDA environment on every node.
- CUDA GPU for publishable numbers.
- Node 0 ports open: default `29500` and `29765`.
- Node-local output storage; merge `runs/` on node 0 after a multi-node run.

## Phase 1: Static Cost Audit

```bash
python perf_benchmark/phase1_cost_model/static_cost_audit.py \
  --output-dir perf_benchmark/results/phase1_static
```

Outputs:

```text
static_cost_audit.json
static_cost_audit.md
```

## Phase 2: Configs

Review or edit:

```text
perf_benchmark/phase2_design/methodology.md
perf_benchmark/phase2_design/experiment_matrix.md
perf_benchmark/phase2_design/configs/
```

Tune batch sizes for the target GPU before publishing.

## Phase 3: Local Smoke

```bash
python perf_benchmark/phase3_benchmark/run_benchmark.py \
  --config perf_benchmark/phase2_design/configs/local_smoke.json \
  --aggregate-after
```

## Phase 3: Single GPU

```bash
python perf_benchmark/phase3_benchmark/run_benchmark.py \
  --config perf_benchmark/phase2_design/configs/single_gpu.json \
  --aggregate-after
```

## Phase 3: 2 Nodes x 1 GPU

Choose a fresh ID once and use it on every node. Result directories are never
reused.

Run once per node, at the same time.

Node 0:

```bash
python perf_benchmark/phase3_benchmark/run_benchmark.py \
  --config perf_benchmark/phase2_design/configs/multinode_2x1.json \
  --run-id <UNIQUE_RUN_ID> \
  --node-rank 0 \
  --master-addr <NODE0_IP>
```

Node 1:

```bash
python perf_benchmark/phase3_benchmark/run_benchmark.py \
  --config perf_benchmark/phase2_design/configs/multinode_2x1.json \
  --run-id <UNIQUE_RUN_ID> \
  --node-rank 1 \
  --master-addr <NODE0_IP>
```

Before the full run, add `--quick` to both node commands. It verifies the
fixed-port rendezvous and rank artifacts with `never_init` and `trace_auto`;
do not publish its timings.

## Phase 3: 4 Nodes x 1 GPU

Run once per node:

```bash
python perf_benchmark/phase3_benchmark/run_benchmark.py \
  --config perf_benchmark/phase2_design/configs/multinode_4x1.json \
  --run-id <UNIQUE_RUN_ID> \
  --node-rank <0|1|2|3> \
  --master-addr <NODE0_IP>
```

Multi-node ports are fixed by config and varied deterministically per cell. A
120-second pre-launch barrier stops the matrix if nodes desynchronize. The
multi-node cell cap is 15 minutes, separate from that barrier timeout.

## Aggregate

If each node has local storage, copy every node's `runs/` subtree back under
the same result directory on node 0. Then run:

```bash
python perf_benchmark/phase3_benchmark/aggregate_results.py \
  --results-dir perf_benchmark/results/<run_id>
```

This writes:

```text
summary.json
summary.csv
report.md
```

## Ray Train

Optional dependency check:

```bash
python perf_benchmark/phase3_benchmark/run_ray_train.py \
  --config perf_benchmark/phase2_design/configs/multinode_2x1.json \
  --dry-run
```

This is a dependency-only stub. Use `torchrun` for publishable runs.

## Output Files

```text
environment.json
suite_manifest.json
runs/<cell>/repeat_<n>/rank_<rank>.json
summary.json
summary.csv
report.md
```

## Troubleshooting

- Missing CUDA: use `local_smoke.json`; do not publish CPU overhead.
- Multi-node hang: check `<NODE0_IP>`, firewall, `master_port`, `aggregator_port`, `barrier_port`.
- Existing result directory: pass a new shared `--run-id`; reuse is refused.
- Missing rank files: copy all node result folders before aggregation.
- Missing TraceML summary: inspect `launcher_stderr.txt`.
- Network counters: Linux default-route NIC counters are captured when
  available. They include DDP and TraceML traffic, so attribute by the
  `never_init` delta.
