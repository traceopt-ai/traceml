# Debug low GPU utilization in PyTorch training

Use this guide when PyTorch training is slow and GPU utilization is low,
moderate, uneven, or bursty.

Low GPU utilization is a symptom. TraceML helps decide whether the likely cause
is input loading, host-to-device transfer, wait time, rank skew, memory
pressure, or model-side compute behavior.

## Confirm the symptom

Run your training script:

```bash
traceml run train.py
```

In the System section, TraceML can report GPU utilization symptoms as:

| Structured kind | Text status | Meaning |
|---|---|---|
| `LOW_GPU_UTILIZATION` | `LOW GPU UTIL` | average GPU utilization was below 30% |
| `MODERATE_GPU_UTILIZATION` | `MODERATE GPU UTIL` | average GPU utilization was from 30% through 70% |

Above 70%, TraceML does not emit a GPU-utilization issue unless another System
rule fires.

## Explain it with Step Time

After confirming low or moderate GPU utilization, read the Step Time diagnosis.

| Step Time diagnosis | What to do |
|---|---|
| `INPUT-BOUND` | Inspect the DataLoader and input path. |
| `INPUT STRAGGLER` | Inspect the slow input rank. |
| `WAIT-HEAVY` | Inspect work outside traced phases, such as logging, checkpointing, validation, CPU stalls, framework orchestration, or unobserved transfers. |
| `COMPUTE-BOUND` | Inspect forward, backward, and optimizer time before changing the DataLoader. |
| `COMPUTE STRAGGLER` or `STRAGGLER` | Inspect rank skew and the called-out worst rank. |
| `BALANCED` | Compare against a known good run or use a heavier profiler for lower-level detail. |

Low GPU utilization plus `INPUT-BOUND` is a strong signal to start with the
[DataLoader bottleneck guide](pytorch-dataloader-bottleneck.md). Low GPU
utilization by itself is not enough.

## Checks that match TraceML evidence

If input time is high:

- inspect `DataLoader(num_workers=...)`
- inspect CPU preprocessing, decoding, tokenization, and collation
- check slow storage or network filesystems
- compare with a synthetic-data run

If H2D time is high:

- check host-to-device transfer behavior
- for CUDA training, check whether the input path uses `pin_memory=True` and
  non-blocking transfer where appropriate

If wait time is high:

- inspect logging, checkpointing, validation, and framework work around the
  traced training step
- inspect CPU or RAM pressure in the System and Process sections

If one rank is worse than the others:

- use the [DDP rank straggler guide](ddp-slow-training-rank-straggler.md)
- compare the worst rank with the median rank

## Compare a fix

After changing the suspected cause, compare the before and after summaries:

```bash
traceml compare old_run/final_summary.json new_run/final_summary.json
```

Check whether GPU utilization, total step time, input time, wait time, or the
primary diagnosis changed.

## Related

- [Find why PyTorch training is slow](slow-pytorch-training.md)
- [Find DataLoader Bottlenecks](pytorch-dataloader-bottleneck.md)
- [Debug DDP Rank Stragglers](ddp-slow-training-rank-straggler.md)
- [How to Read TraceML Output](../user_guide/reading-output.md)
