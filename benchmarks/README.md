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

## Multi-Node Method

Multi-node needs one launcher per node. Use the same `--run-name` on every
node. First run the native command on both nodes with `torchrun`, then run the
TraceML command on both nodes with `traceml run`.

TraceML node 0:

```bash
time traceml run benchmarks/workloads/ddp_mlp_e2e.py \
  --mode=summary \
  --logs-dir=benchmarks/results/logs \
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
time traceml run benchmarks/workloads/ddp_mlp_e2e.py \
  --mode=summary \
  --logs-dir=benchmarks/results/logs \
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
