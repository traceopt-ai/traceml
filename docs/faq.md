# FAQ

Short answers to common questions before or during adoption.

If you are new to TraceML, start with:

- [Quickstart](quickstart.md)
- [How to Read TraceML Output](how-to-read-output)

---

## Do I need to replace W&B, MLflow, or TensorBoard?

No.

TraceML is designed to work alongside your existing stack.

Use your current tools for:

- experiment tracking
- artifacts
- dashboards
- reporting

Use TraceML for:

- bottleneck diagnosis
- stragglers
- wait-heavy behavior
- memory creep

See:

- [Use TraceML with W&B / MLflow](use-with-wandb-mlflow.md)

---

## How is TraceML different from `torch.profiler`?

`torch.profiler` is a deeper profiling tool.

TraceML is a lighter-weight bottleneck finder for real training runs.

A simple rule:

- use TraceML to find where the problem is
- use `torch.profiler` when you need deeper low-level analysis

---

## How much code do I need to change?

Usually just this:

```python
with trace_step(model):
    ...
```

For supported integrations:

- Hugging Face: use `TraceMLTrainer`
- Lightning: add `TraceMLCallback()`

---

## Does TraceML work with Hugging Face Trainer?

Yes.

See:

- [Hugging Face Trainer](huggingface.md)

---

## Does TraceML work with PyTorch Lightning?

Yes.

See:

- [PyTorch Lightning](lightning.md)

---

## Does TraceML support DDP?

Yes, for single-node DDP.

TraceML can surface:

- input stragglers
- compute stragglers
- rank imbalance
- worst-rank vs median-rank skew

---

## Does TraceML support multi-node?

Not yet.

Today the main distributed target is single-node DDP.

---

## Does TraceML support FSDP?

Partially / early, depending on setup.

If FSDP matters for your workload, test it on a representative run and open an issue if something does not behave as expected.

---

## Does TraceML support tensor parallel or pipeline parallel?

Not yet.

---

## What is the difference between `watch`, `run`, and `deep`?

`watch`
- zero-code system and process visibility

`run`
- the default mode
- step-aware bottleneck diagnosis

`deep`
- optional deeper layer-level inspection
- best for short follow-up diagnostic runs

Start with `run`.

---

## Is there a local UI?

Yes.

Run:

```bash
traceml run train.py --mode=dashboard
```

The local UI runs at:

```text
http://localhost:8765
```

---

## Can I run without TraceML telemetry for a baseline?

Yes.

Use:

```bash
traceml run train.py --disable-traceml
```

---

## What does `MEMORY CREEP` usually mean?

It usually means memory is rising over time instead of staying stable.

A common cause is retaining tensors across steps, for example by storing graph-backed tensors in a persistent cache or list.

See:

- [How to Read TraceML Output](how-to-read-output)

---

## What does `INPUT STRAGGLER` mean?

It means one rank is slower in the input path than the typical rank.

Common causes:

- uneven data loading
- preprocessing imbalance
- host-side jitter

See:

- [How to Read TraceML Output](how-to-read-output)

---

## What does `COMPUTE STRAGGLER` mean?

It means one rank is slower in compute than the typical rank.

Common causes:

- uneven forward/backward/optimizer work
- shape differences
- rank-local extra work

See:

- [How to Read TraceML Output](how-to-read-output)

---

## Is TraceML heavy?

TraceML is intended to be much lighter than a heavyweight profiler.

If you want the lowest-overhead normal mode, use:

```bash
traceml run train.py
```

Use `deep` only when you need deeper inspection.

---

## Why does TraceML launch through `torchrun`?

TraceML uses:

```bash
python -m torch.distributed.run
```

so launch behavior stays consistent for both single-process and DDP workflows.

---

## What if the terminal gets noisy?

Good ways to keep things clean:

- disable `tqdm`
- reduce extra console logging
- use the local UI with `--mode=dashboard`

---

## What should I include in an issue?

Please include:

- hardware / CUDA / PyTorch versions
- single GPU or multi-GPU
- whether you used `watch`, `run`, or `deep`
- the end-of-run summary
- a minimal repro if possible

GitHub issues:

`https://github.com/traceopt-ai/traceml/issues`
