# Wall-clock overhead harness

The coarse overhead track: paired `time torchrun` vs `time traceml run`
runs of [`ddp_mlp_e2e.py`](ddp_mlp_e2e.py), no TraceML-specific tooling
required to reproduce. For how this track relates to the attribution
harness (`perf_benchmark/`), see the [benchmarking overview](../README.md).
Campaign write-ups live in [`../analysis/`](../analysis/README.md).

## Simple End-To-End Timing

For the simplest user-facing overhead check, run the same normal DDP MLP
training script with and without TraceML and compare shell wall time.

Single GPU native:

```bash
time torchrun --nproc_per_node=1 benchmarking/workloads/ddp_mlp_e2e.py \
  --duration-sec 600 \
  --batch-size 256 \
  --hidden-dim 4096
```

Single GPU with TraceML:

```bash
time traceml run benchmarking/workloads/ddp_mlp_e2e.py \
  --mode=summary \
  --run-name=e2e_1gpu_traceml_r01 \
  --nproc-per-node=1 \
  --args \
  --duration-sec 600 \
  --batch-size 256 \
  --hidden-dim 4096
```

Repeat both commands several times and alternate order:

```text
repeat 1: native, TraceML
repeat 2: TraceML, native
repeat 3: native, TraceML
```

Use the shell `real` time:

```text
overhead_pct = ((traceml_real_time / native_real_time) - 1) * 100
```

Increase `--batch-size`, `--hidden-dim`, or `--hidden-layers` if the training
step is too fast. Keep the exact same arguments for native and TraceML runs.

## What To Publish

Use the end-to-end overhead percentage:

```text
overhead_pct = ((traceml_real_time / native_real_time) - 1) * 100
```

Publish average and median overhead across successful paired repeats, plus
min/max or IQR. Keep the raw shell `real` times.

For publishable numbers, use at least:

```text
5 repeats x 2 modes x each topology
```

Two or three repeats are fine for smoke tests.

## Results: first campaign (2026-06-11, `ddp_mlp_e2e`)

5 paired repeats × 2 modes × 2 topologies, 600 s/trial, alternating order,
identical args (`--duration-sec 600 --batch-size 256 --hidden-dim 4096`).
20/20 trials, 0 failures. Hardware: 2× AWS `g4dn.xlarge` (1× Tesla T4 each),
`eu-central-1`; torch 2.11.0+cu130, CUDA 13.0, NCCL 2.28.9.

| Topology | Native real (s) | TraceML real (s) | Wall overhead | Throughput overhead |
|---|---|---|---|---|
| Single GPU (1× T4)   | 613.43 | 618.98 | **+0.90%** | **+1.02%** |
| DDP (2 nodes × 1 T4) | 614.67 | 618.62 | **+0.64%** | **≈0%** (network-bound) |

- **Single GPU:** ~5.5 s fixed startup + ~1% per-step, stable across repeats.
- **DDP:** ~92% of the step is backward/all-reduce over TCP, so TraceML's
  per-step cost is not measurable. (Repeat 1 ran on g4dn network burst
  credits; repeats 2-5 sat at the stable floor, throughput is reported as
  the median to exclude that confound.)
- Wall overhead is the metric defined in "What To Publish" above; throughput
  (`native_steps_per_s / traceml_steps_per_s − 1`) is shown alongside because
  for a fixed-duration workload, wall time mainly captures fixed startup, not
  per-step cost.

Full write-up with plots:
[`analysis/2026-06-11_pr153_ddp_mlp_g4dn/report.md`](../analysis/2026-06-11_pr153_ddp_mlp_g4dn/report.md).

## Results: second campaign (2026-07-19, `ddp_mlp_e2e`, v0.3.5)

Rerun on TraceML v0.3.5 to confirm the overhead figure held across
versions. 5 paired repeats × 2 modes × 2 topologies (single GPU, 4-GPU
single-node DDP), 300 s/trial. Hardware: 1× AWS `g4dn.12xlarge` (4×
Tesla T4), `eu-central-1`.

| Topology | Throughput overhead | Wall-clock overhead |
|---|---|---|
| Single GPU (1× T4)          | **+0.95% ± 0.09** | +2.16% ± 0.34 |
| 4-GPU single-node DDP        | **+0.41% ± 0.07** | +1.76% ± 0.36 |

Full write-up:
[`analysis/2026-07-19_v035_ddp_mlp_g4dn/report.md`](../analysis/2026-07-19_v035_ddp_mlp_g4dn/report.md).

## Multi-Node Method

Multi-node needs one launcher per node. Use the same `--run-name` on every
node. First run the native command on both nodes with `torchrun`, then run the
TraceML command on both nodes with `traceml run`.

TraceML node 0:

```bash
time traceml run benchmarking/workloads/ddp_mlp_e2e.py \
  --mode=summary \
  --logs-dir=benchmarking/results/logs \
  --run-name=e2e_2x1_traceml_r01 \
  --nnodes=2 \
  --node-rank=0 \
  --nproc-per-node=1 \
  --master-addr=<node0-ip> \
  --master-port=29500 \
  --aggregator-port=29765 \
  --args \
  --duration-sec 600 \
  --batch-size 256 \
  --hidden-dim 4096
```

TraceML node 1:

```bash
time traceml run benchmarking/workloads/ddp_mlp_e2e.py \
  --mode=summary \
  --logs-dir=benchmarking/results/logs \
  --run-name=e2e_2x1_traceml_r01 \
  --nnodes=2 \
  --node-rank=1 \
  --nproc-per-node=1 \
  --master-addr=<node0-ip> \
  --master-port=29500 \
  --aggregator-port=29765 \
  --args \
  --duration-sec 600 \
  --batch-size 256 \
  --hidden-dim 4096
```

For Slurm, keep the existing TraceML pattern: allocate one task per node, run
one `traceml run` launcher per node, and let TraceML spawn one worker per GPU.
