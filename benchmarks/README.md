# TraceML Benchmarks

## Simple End-To-End Timing

For the simplest user-facing overhead check, run the same normal DDP MLP
training script with and without TraceML and compare shell wall time.

Single GPU native:

```bash
time torchrun --nproc_per_node=1 benchmarks/workloads/ddp_mlp_e2e.py \
  --duration-sec 600 \
  --batch-size 256 \
  --hidden-dim 4096
```

Single GPU with TraceML:

```bash
time traceml run benchmarks/workloads/ddp_mlp_e2e.py \
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

## Steady-State Paired Runner

This folder starts with one practical overhead benchmark: a synthetic DDP MLP
that does enough GPU work to make TraceML overhead measurable without needing a
dataset download.

The workload reserves persistent GPU memory after AdamW optimizer state is
initialized. By default it targets 30% allocated memory per GPU. On a 16 GiB
T4, that is roughly 4.8 GiB. Use `--target-gpu-mem-frac 0.50` if you want a
heavier memory fill.

## Single-Node Paired Overhead Run

Run a quick 1 GPU smoke benchmark:

```bash
python benchmarks/run_overhead.py \
  --nproc-per-node 1 \
  --repeats 3 \
  --target-gpu-mem-frac 0.30
```

Run the first serious single-node DDP cell:

```bash
python benchmarks/run_overhead.py \
  --nproc-per-node 4 \
  --repeats 5 \
  --target-gpu-mem-frac 0.30
```

The runner launches paired trials:

```text
native:  traceml run --disable-traceml ...
traceml: traceml run ...
```

Both modes use the same TraceML launcher, same workload, same model size, same
step count, and same GPU memory target. The workload ignores warmup steps and
writes steady-state step timing metrics.

Results land in:

```text
benchmarks/results/results.md
benchmarks/results/pairs.csv
benchmarks/results/runs.json
benchmarks/results/<run-name>/workload_metrics.json
benchmarks/results/logs/<run-name>/final_summary.json
```

`final_summary.json` exists only for TraceML-enabled runs.

## What To Publish

Use the paired overhead percentage:

```text
overhead_pct = ((traceml_step_ms / native_step_ms) - 1) * 100
```

Publish the median overhead across successful pairs, plus min/max or IQR. Do
not publish a single average runtime as the headline number.

For publishable numbers, use at least:

```text
5 repeats x 2 modes x each topology
```

Two or three repeats are fine for smoke tests.

## Multi-Node Method

Multi-node needs one launcher per node. Use the same `--run-name` on every
node. First run the native pair with `--disable-traceml` on both nodes, then
the TraceML-enabled pair without it.

Node 0:

```bash
traceml run benchmarks/workloads/ddp_mlp_overhead.py \
  --mode=summary \
  --logs-dir=benchmarks/results/logs \
  --run-name=overhead_2x1_traceml_r01 \
  --nnodes=2 \
  --node-rank=0 \
  --nproc-per-node=1 \
  --master-addr=<node0-ip> \
  --master-port=29500 \
  --aggregator-port=29765 \
  --args \
  --steps 220 \
  --warmup-steps 20 \
  --target-gpu-mem-frac 0.30 \
  --metrics-file benchmarks/results/overhead_2x1_traceml_r01_metrics.json
```

Node 1:

```bash
traceml run benchmarks/workloads/ddp_mlp_overhead.py \
  --mode=summary \
  --logs-dir=benchmarks/results/logs \
  --run-name=overhead_2x1_traceml_r01 \
  --nnodes=2 \
  --node-rank=1 \
  --nproc-per-node=1 \
  --master-addr=<node0-ip> \
  --master-port=29500 \
  --aggregator-port=29765 \
  --args \
  --steps 220 \
  --warmup-steps 20 \
  --target-gpu-mem-frac 0.30 \
  --metrics-file benchmarks/results/overhead_2x1_traceml_r01_metrics.json
```

For the native paired run, add `--disable-traceml` to both node commands and
change the run name, for example `overhead_2x1_native_r01`.

Only global rank 0 writes the workload metrics file, but the JSON contains
gathered timing metrics for every rank.

For Slurm, keep the existing TraceML pattern: allocate one task per node, run
one `traceml run` launcher per node, and let TraceML spawn one worker per GPU.
