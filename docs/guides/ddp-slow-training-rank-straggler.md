# Debug slow DDP training and rank stragglers

Use this guide when PyTorch DDP training is slow, uneven, or controlled by one
slow rank.

TraceML compares worst-rank and median-rank timing in distributed runs. It can
surface input stragglers, compute stragglers, mixed stragglers, and overhead-heavy
windows.

## Run DDP with TraceML

For single-node DDP:

```bash
traceml run train.py --nproc-per-node=4
```

For multi-node DDP, use the same `--run-name`, `--nnodes`,
`--nproc-per-node`, and `--master-addr` on every node. See
[Distributed Training](../user_guide/distributed-training.md).

On Slurm, use the [Slurm guide](../user_guide/slurm.md) instead of writing
the node flags by hand.

## What to look for

Start with the Step Time diagnosis.

| Diagnosis | Meaning |
|---|---|
| `INPUT STRAGGLER` | one rank has meaningfully more dataloader burden than a typical rank |
| `COMPUTE STRAGGLER` | one rank has meaningfully more forward, backward, or optimizer burden than a typical rank |
| `STRAGGLER` | input and compute are both materially uneven |
| `INPUT-BOUND` | input work is broad, not just one bad rank |
| `OVERHEAD-HEAVY` | meaningful step overhead is not attributed to dataloader, H2D, forward, backward, or optimizer work |

Then inspect:

- `Worst Rank`
- worst vs median timing
- `Dataloader Fetch`
- `Forward`
- `Backward`
- `Optimizer Step`
- `Overhead`

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

If the diagnosis is `STRAGGLER`, reduce the problem one phase at a time. Start
with the phase that has the largest worst-vs-median gap.

If the diagnosis is `OVERHEAD-HEAVY`, remember that TraceML reports step
overhead. It is not proof of GPU idle time or direct NCCL/all-reduce timing.
Inspect logging, checkpointing, validation, CPU stalls, framework
orchestration, and unobserved transfer paths before assuming the cause.

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
step overhead, and diagnosis.

## Related

- [Find why PyTorch training is slow](slow-pytorch-training.md)
- [Find DataLoader Bottlenecks](pytorch-dataloader-bottleneck.md)
- [Distributed Training](../user_guide/distributed-training.md)
- [Running on Slurm](../user_guide/slurm.md)
- [How to Read TraceML Output](../user_guide/reading-output.md)
