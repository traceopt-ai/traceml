# Find why PyTorch training is slow

Use this guide when PyTorch training is slow and you do not yet know whether
the bottleneck is input loading, GPU utilization, distributed rank skew, memory
growth, residual time, or a run-to-run regression.

This is a triage guide. It points to focused guides instead of repeating every
TraceML diagnosis.

## Run one summary

If your script is not instrumented yet, start with the
[Quickstart](../user_guide/quickstart.md).

Run your training script:

```bash
traceml run train.py
```

TraceML writes:

```text
logs/<run_name>/final_summary.json
logs/<run_name>/final_summary.txt
```

Start with the final summary, then follow the branch that matches your
diagnosis or symptom.

## Choose the right path

| What you see | Start here |
|---|---|
| Step Time says `INPUT-BOUND` | [Find DataLoader Bottlenecks](pytorch-dataloader-bottleneck.md) |
| System says `LOW_GPU_UTILIZATION` or `MODERATE_GPU_UTILIZATION` | [Debug Low GPU Utilization](low-gpu-utilization-pytorch.md) |
| Step Time says `INPUT STRAGGLER`, `COMPUTE STRAGGLER`, `H2D STRAGGLER`, `RESIDUAL STRAGGLER`, or `STRAGGLER` | [Debug DDP Rank Stragglers](ddp-slow-training-rank-straggler.md) |
| Step Memory says `MEMORY CREEP` or `MEMORY RISING` | [Find PyTorch Memory Creep](pytorch-memory-creep.md) |
| A recent change made the run slower | [Compare Runs](../user_guide/compare.md) |
| Step Time says `COMPUTE-BOUND` or `RESIDUAL-HEAVY` | [How to Read TraceML Output](../user_guide/reading-output.md) |

## Quick interpretation

`INPUT-BOUND` means dataloader or input work is taking a large share of the
typical step. Confirm the input path before tuning model compute.

`LOW_GPU_UTILIZATION` and `MODERATE_GPU_UTILIZATION` are system-level symptoms.
They say the GPU was not fully busy, not why it was not fully busy. Pair them
with Step Time.

`INPUT STRAGGLER`, `COMPUTE STRAGGLER`, `H2D STRAGGLER`, `RESIDUAL STRAGGLER`, and
`STRAGGLER` are distributed clean-step signals. Inspect the called-out worst
rank and compare it with the median rank.

`MEMORY CREEP` and `MEMORY RISING` are step-memory trend signals. Inspect
retained tensors, caches, and per-step state that may stay alive.

`COMPUTE-BOUND` means forward, backward, or optimizer time dominates the
observed step. Use TraceML to choose the hot phase, then use an operator-level
profiler if you need kernel or operator detail.

`RESIDUAL-HEAVY` is residual time not attributed to dataloader, H2D, forward,
backward, or optimizer work. Inspect logging, checkpointing, validation,
CPU-side stalls, framework orchestration, or unobserved transfers.

## What not to assume

- Low GPU utilization alone does not prove a DataLoader bottleneck.
- A DDP slowdown is not always NCCL. TraceML currently reports rank skew and
  residual residual, not explicit collective timing.
- `BALANCED` means no clear bottleneck in the observed window. It does not
  prove the run is globally optimal.
- A single slow run is easier to trust after comparing it with a known good
  `final_summary.json`.

## Related

- [Quickstart](../user_guide/quickstart.md)
- [How to Read TraceML Output](../user_guide/reading-output.md)
- [Compare Runs](../user_guide/compare.md)
- [Distributed Training](../user_guide/distributed-training.md)
