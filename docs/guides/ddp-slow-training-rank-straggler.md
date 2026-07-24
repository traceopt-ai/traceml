# Debug slow DDP training and rank stragglers

Use this guide when PyTorch DDP training is slow, uneven, or controlled by one
culprit rank.

TraceML looks for visible rank skew in distributed runs, identifies a culprit
rank and a victim/reference rank, then attributes material culprit excess to
input, H2D, DDP forward compute, or sync/unattributed work.

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
| Input straggler | `INPUT STRAGGLER` | r0 input wait `201.6ms` vs r1 `1.4ms` |
| Compute straggler | `COMPUTE STRAGGLER` | r0 optimizer `33.1ms` vs r1 `14.5ms` |

Read the balanced run as the control: both ranks have similar total step time,
input time is small, and the run is mostly compute.

Read the input-straggler run as a rank-local input issue: rank 0 reaches compute
late because its input path is slower.

Read the compute-straggler run as a rank-local compute issue: rank 0 spends more
time in optimizer work than the peer rank.

## What to look for

Start with the Step Time diagnosis.

| Diagnosis | Meaning |
|---|---|
| `INPUT STRAGGLER` | one rank spends much longer waiting for input than peer ranks |
| `COMPUTE STRAGGLER` | in DDP, the culprit rank has materially more forward-time burden than the victim rank |
| `H2D STRAGGLER` | the culprit rank has meaningfully more host-to-device transfer burden than the victim rank |
| `STRAGGLER` | visible rank skew exists, but input wait, H2D, and DDP forward do not explain it |
| `INPUT-BOUND` | input work is broad, not just one bad rank |
| `RESIDUAL-HEAVY` | meaningful residual time is not attributed to input wait, H2D, forward, backward, or optimizer work |

For rank stragglers, TraceML first looks for visible wait cost: in DDP this is
the gap between the upper-median backward time and the rank with the smallest
backward time. The smallest-backward rank is the likely culprit because it
arrived late and waited least. TraceML then compares that culprit with the
victim rank to see whether input wait, H2D, or DDP forward time explains the
cost.

Then inspect:

- culprit rank
- victim/reference rank
- visible rank skew
- `Input Wait`
- `Forward`
- `Backward`
- `Optimizer Step`
- `Residual`

## How to triage the culprit rank

If the diagnosis is `INPUT STRAGGLER`, inspect input loading on the culprit
rank:

- uneven shards or batch construction
- rank-local preprocessing jitter
- slow storage path on one host
- custom collation or tokenization on one rank

If the diagnosis is `COMPUTE STRAGGLER`, inspect DDP forward work on the
culprit rank:

- uneven input shapes
- rank-local branching
- extra forward work
- framework hooks or callbacks that run on one rank

If the diagnosis is `H2D STRAGGLER`, inspect host-to-device transfer on the
culprit rank: batch shapes, transfer placement, pinned memory, and transfer
jitter.

If the diagnosis is `STRAGGLER`, inspect synchronization, collectives, and
unattributed work around the culprit rank. Explicit collective timing is not
available yet, so this is the honest fallback when input, H2D, and DDP forward
do not explain the visible wait cost.

If the diagnosis is `RESIDUAL-HEAVY`, remember that TraceML reports residual
time as a derived bucket. It is not direct NCCL or all-reduce timing. Inspect logging,
checkpointing, validation, CPU stalls, framework orchestration, and unobserved
transfer paths before assuming the cause.

## FSDP note

TraceML supports single-node multi-GPU DDP and FSDP in the current public docs.
Multi-node DDP summary reports are supported. Multi-node FSDP uses the same
distributed launch path as DDP, but should be validated in your environment.

TraceML uses backward as the visible rank-skew phase for DDP/default strategy.
For FSDP, it uses forward + backward because sharding communication can appear
in both phases. FSDP rank stragglers can still be attributed to input wait or
H2D, but TraceML does not emit `COMPUTE STRAGGLER` from the FSDP rank-skew
rule without explicit all-gather or reduce-scatter timing.

## Compare a fix

After changing data loading, batching, model logic, or host placement, compare
the old and new summaries:

```bash
traceml compare old_run/final_summary.json new_run/final_summary.json
```

Look for changes in total step time, visible rank skew, input time, compute
time, residual time, and diagnosis.

## Related

- [Find why PyTorch training is slow](slow-pytorch-training.md)
- [Find Input Pipeline Bottlenecks](pytorch-input-pipeline-bottleneck.md)
- [Distributed Training](../user_guide/distributed-training.md)
- [Running on Slurm](../user_guide/slurm.md)
- [How to Read TraceML Output](../user_guide/reading-output.md)
