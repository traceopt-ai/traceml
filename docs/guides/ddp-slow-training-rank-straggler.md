# Debug slow DDP training and rank stragglers

Use this guide when PyTorch DDP training is slow, uneven, or controlled by one
slow rank.

TraceML compares worst-rank and median-rank timing in distributed runs. It can
surface input stragglers, compute stragglers, mixed stragglers, and residual-heavy
windows.

For whole-run triage, start with
[Find why PyTorch training is slow](slow-pytorch-training.md). This page stays
focused on DDP rank skew.

## Run DDP with TraceML

For single-node DDP:

```bash
traceml run train.py --nproc-per-node=4
```

For multi-node DDP, use the same `--run-name`, `--nnodes`,
`--nproc-per-node`, `--master-addr`, `--master-port`, and script args on every
node. See
[Distributed Training](../user_guide/distributed-training.md).

On Slurm, use the [Slurm guide](../user_guide/slurm.md) instead of writing
the node flags by hand.

## Reproduce rank stragglers

The repo includes a small DDP demo with a compute-heavy baseline and two
rank-local straggler modes.

Balanced baseline:

```bash
traceml run examples/distributed/ddp_rank_straggler_demo.py --mode=summary --nproc-per-node=2 --run-name ddp_balanced --args --scenario balanced
```

Input straggler:

```bash
traceml run examples/distributed/ddp_rank_straggler_demo.py --mode=summary --nproc-per-node=2 --run-name ddp_input_straggler --args --scenario input-straggler --straggler-rank 0 --input-sleep-ms 200
```

Compute straggler:

```bash
traceml run examples/distributed/ddp_rank_straggler_demo.py --mode=summary --nproc-per-node=2 --run-name ddp_compute_straggler --args --scenario compute-straggler --straggler-rank 0 --compute-extra-matmuls 8
```

For two nodes with one GPU each, run the same script on both nodes with
`--nnodes=2 --nproc-per-node=1`, the same `--master-addr`,
`--master-port`, and `--run-name`, and a different `--node-rank` on each node.

Example node 0:

```bash
traceml run examples/distributed/ddp_rank_straggler_demo.py --mode=summary --nnodes=2 --nproc-per-node=1 --node-rank=0 --master-addr <NODE_0_PRIVATE_IP> --master-port 29546 --run-name ddp_compute_straggler --args --scenario compute-straggler --straggler-rank 0 --compute-extra-matmuls 8
```

Example node 1:

```bash
traceml run examples/distributed/ddp_rank_straggler_demo.py --mode=summary --nnodes=2 --nproc-per-node=1 --node-rank=1 --master-addr <NODE_0_PRIVATE_IP> --master-port 29546 --run-name ddp_compute_straggler --args --scenario compute-straggler --straggler-rank 0 --compute-extra-matmuls 8
```

## Demo fingerprints

These fingerprints came from the DDP demo on two nodes with one GPU per node.

| Scenario | Diagnosis | Key signal |
|---|---|---|
| Balanced | `COMPUTE-BOUND` | total `124.6/124.6ms`, input `1.4/1.4ms`, compute `122.4/122.4ms` |
| Input straggler | `INPUT STRAGGLER` | r0 dataloader `201.6ms` vs r1 `1.4ms` |
| Compute straggler | `COMPUTE STRAGGLER` | r0 optimizer `33.1ms` vs r1 `14.5ms` |

Read the balanced run as the control: both ranks have similar total step time,
input time is small, and the run is mostly compute.

Read the input-straggler run as a rank-local input issue: rank 0 reaches compute
late because its dataloader path is slower.

Read the compute-straggler run as a rank-local compute issue: rank 0 spends more
time in optimizer work than the peer rank.

## What to look for

Start with the Step Time diagnosis.

| Diagnosis | Meaning |
|---|---|
| `INPUT STRAGGLER` | one rank has meaningfully more dataloader burden than a typical rank |
| `COMPUTE STRAGGLER` | one rank has meaningfully more clean compute burden than a typical rank |
| `H2D STRAGGLER` | one rank has meaningfully more host-to-device transfer burden than a typical rank |
| `RESIDUAL STRAGGLER` | one rank has meaningfully more rank-local residual `residual_proxy` than a typical rank |
| `STRAGGLER` | one rank is slower, but dataloader, clean compute, H2D, and residual excess are mixed |
| `INPUT-BOUND` | input work is broad, not just one bad rank |
| `RESIDUAL-HEAVY` | meaningful residual time is not attributed to input wait, H2D, forward, backward, or optimizer work |

For rank-local stragglers, TraceML uses clean-step attribution before blaming a
component. It discounts backward time that can be explained by another rank's
non-backward work, then compares clean step across ranks. The component label is
the largest worst-rank excess over peer median among input wait, clean compute,
H2D, and residual, and it must dominate the next-largest excess by `1.25x`.

Then inspect:

- `Worst Rank`
- worst vs median timing
- `Input Wait`
- `Forward`
- `Backward`
- `Optimizer Step`
- `Residual`

## How to triage the worst rank

If the diagnosis is `INPUT STRAGGLER`, inspect input loading on the worst rank:

- uneven shards or batch construction
- rank-local preprocessing jitter
- slow storage path on one host
- custom collation or tokenization on one rank

If the diagnosis is `COMPUTE STRAGGLER`, inspect model-side work on the worst
rank:

- uneven input shapes
- rank-local branching
- extra forward, backward, or optimizer work
- framework hooks or callbacks that run on one rank

If the diagnosis is `H2D STRAGGLER`, inspect host-to-device transfer on the
worst rank: batch shapes, transfer placement, pinned memory, and transfer
jitter.

If the diagnosis is `RESIDUAL STRAGGLER`, inspect rank-local host-side work on the
worst rank: logging, checkpointing, validation, callbacks, CPU stalls, or
unobserved transfer work.

If the diagnosis is `STRAGGLER`, reduce the problem one phase at a time. Start
with the largest clean-step component gap.

If the diagnosis is `RESIDUAL-HEAVY`, remember that TraceML reports residual
time as a derived bucket. It is not direct NCCL or all-reduce timing. Inspect logging,
checkpointing, validation, CPU stalls, framework orchestration, and unobserved
transfer paths before assuming the cause.

## FSDP note

TraceML supports single-node multi-GPU DDP and FSDP in the current public docs.
Multi-node DDP summary reports are supported. Multi-node FSDP uses the same
distributed launch path as DDP, but should be validated in your environment.

Use this guide for rank-skew symptoms in FSDP runs, but do not treat it as
FSDP-specific collective timing.

## Compare a fix

After changing data loading, batching, model logic, or host placement, compare
the old and new summaries:

```bash
traceml compare old_run/final_summary.json new_run/final_summary.json
```

Look for changes in total step time, worst-rank skew, input time, compute time,
residual time, and diagnosis.

## Related

- [Find why PyTorch training is slow](slow-pytorch-training.md)
- [Find DataLoader Bottlenecks](pytorch-dataloader-bottleneck.md)
- [Distributed Training](../user_guide/distributed-training.md)
- [Running on Slurm](../user_guide/slurm.md)
- [How to Read TraceML Output](../user_guide/reading-output.md)
