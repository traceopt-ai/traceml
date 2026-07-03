# Find PyTorch DataLoader bottlenecks

Use this guide when PyTorch training feels slow, GPU utilization is low, or
distributed workers spend time waiting for an input rank to catch up.

A DataLoader bottleneck means the input path is taking enough time to affect
training throughput. In TraceML, start by checking whether step time is going
to dataloader fetch, host-to-device transfer, compute, residual time, or rank skew.

For whole-run triage, start with
[Find why PyTorch training is slow](slow-pytorch-training.md). This page stays
focused on the DataLoader and input-fetch path.

## Symptoms

A PyTorch DataLoader bottleneck can look like:

- low or uneven GPU utilization
- long gaps between training steps
- real data is slower than synthetic data
- batch collation, decoding, tokenization, or preprocessing feels expensive
- one DDP or FSDP rank reaches compute later than the others
- changing `num_workers` or input preprocessing changes throughput

Low GPU utilization alone is not proof of a DataLoader bottleneck. Confirm
where step time goes before changing the input pipeline.

## Run TraceML

If your script is not instrumented yet, start with the
[Quickstart](../user_guide/quickstart.md).

Run your training script in the default summary mode:

```bash
traceml run train.py
```

TraceML writes:

```text
logs/<run_name>/final_summary.json
logs/<run_name>/final_summary.txt
```

You can re-print the saved text summary later:

```bash
traceml view logs/<run_name>/final_summary.json
```

## What to look for

Start with the `Step Time` section.

For DataLoader problems, the most relevant diagnoses are:

- `INPUT-BOUND`: dataloader or input work is taking a large share of the
  typical step
- `INPUT STRAGGLER`: one rank has meaningfully more dataloader burden than a
  typical rank

Example:

```text
Step Time
- Diagnosis: INPUT STRAGGLER
- Stats: total 303.7ms | input 254.5ms | compute 259.5ms
- Residual: 40.5ms
- Why: r0 input was slower than median global rank (254.5/3.8ms).
```

Read this as:

- input time is large enough to affect training speed
- rank 0 is slower in the input path than the typical rank
- other ranks may wait because distributed training follows the slowest rank

If the diagnosis is `INPUT-BOUND`, inspect the whole input path. If the
diagnosis is `INPUT STRAGGLER`, inspect the called-out rank first.

## Check the input path

Change one thing at a time, then rerun TraceML.

Good first checks:

- increase `DataLoader(num_workers=...)` gradually
- reduce expensive CPU transforms, decoding, tokenization, or collation
- move repeated preprocessing out of the training loop
- check slow storage, network filesystems, or uneven dataset shards
- compare against a synthetic-data run
- in DDP or FSDP, inspect the worst input rank for host-side jitter or uneven
  batches

For CUDA training, also check whether your existing input path uses
`pin_memory=True` and non-blocking host-to-device transfer where appropriate.
TraceML separates dataloader fetch, H2D, compute, and residual time. Use that split
to avoid treating a transfer, compute, or residual issue as a DataLoader issue.

## Compare before and after

After changing the input path, compare the old and new final summaries:

```bash
traceml compare old_run/final_summary.json new_run/final_summary.json
```

Use the compare output to check whether total step time, input time, residual time,
or the diagnosis changed.

## When this is not the right guide

Use a different guide when the primary symptom is not DataLoader fetch time:

- low or moderate GPU utilization without an input diagnosis:
  [Debug Low GPU Utilization](low-gpu-utilization-pytorch.md)
- distributed rank skew beyond the input path:
  [Debug DDP Rank Stragglers](ddp-slow-training-rank-straggler.md)
- rising GPU memory:
  [Find PyTorch Memory Creep](pytorch-memory-creep.md)
- run-to-run slowdown after a change:
  [Compare Runs](../user_guide/compare.md)

## Custom loaders and Ray Data

TraceML automatically instruments `torch.utils.data.DataLoader` when initialized
with `traceml.init(mode="auto")`.

If your input iterator is not a PyTorch `DataLoader`, wrap the fetch path:

```python
train_loader = traceml.wrap_dataloader_fetch(train_loader)
```

This is the pattern used for Ray Data iterators, because Ray
`iter_torch_batches(...)` is not a PyTorch `DataLoader`.

## When to use a heavier profiler

Use TraceML first to decide whether the input path is likely the problem.

Use `torch.profiler` when you need operator-level or timeline detail for a
specific window. Use Nsight Systems when you need lower-level CUDA or system
timeline detail.

TraceML does not replace those tools. It tells you whether the next profiler
run should focus on DataLoader fetches, H2D copies, a specific rank, or a
specific slow window.

## Related

- [Quickstart](../user_guide/quickstart.md)
- [Find why PyTorch training is slow](slow-pytorch-training.md)
- [How to Read TraceML Output](../user_guide/reading-output.md)
- [Compare Runs](../user_guide/compare.md)
- [Ray Train](../user_guide/integrations/ray.md)
