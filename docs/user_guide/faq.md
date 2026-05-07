# FAQ

Short answers to common questions before or during adoption.

If you are new to TraceML, start with:

- [Quickstart](quickstart.md)
- [How to Read TraceML Output](reading-output.md)
- [Compare Runs](compare.md)

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
- run-to-run bottleneck comparison from saved TraceML summary JSON files

See:

- [Use TraceML with W&B / MLflow](integrations/wandb-mlflow.md)

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
import traceml

traceml.init(mode="auto")

with traceml.trace_step(model):
    ...
```

For supported integrations:

- Hugging Face: use `TraceMLTrainer`
- Lightning: add `TraceMLCallback()`

`from traceml.decorators import trace_step` still works for backward
compatibility, but the preferred public API is now the top-level
`traceml.*`.

---

## Should I use `traceml.trace_step()` or `trace_step()`?

Prefer:

```python
import traceml

traceml.init(mode="auto")

with traceml.trace_step(model):
    ...
```

TraceML still supports:

```python
from traceml.sdk.decorators_compat import trace_step
```

for backward compatibility, but new examples and docs use the top-level
`traceml.*` API. Legacy decorator imports are planned for deprecation
starting in `v0.3.0`.

---

## What is the difference between `auto`, `manual`, and `selective`?

Use:

- `traceml.init(mode="auto")` for the default TraceML workflow
- `traceml.init(mode="manual")` when you want fully explicit wrappers
- `traceml.init(mode="selective", ...)` when you want some automatic patching
  and some explicit wrapping

Start with `auto` unless you already know you need more control.

---

## When should I use the wrapper APIs?

Use wrappers when you do not want the default automatic patching path or when
part of your training loop is custom.

The main wrapper entrypoints are:

- `traceml.wrap_dataloader_fetch(...)`
- `traceml.wrap_forward(...)`
- `traceml.wrap_backward(...)`
- `traceml.wrap_optimizer(...)`

This is most relevant in `manual` or `selective` mode. Most users should start
with `mode="auto"` and only move to wrappers if they need explicit control.

---

## Does TraceML work with Hugging Face Trainer?

Yes.

See:

- [Hugging Face Trainer](integrations/huggingface.md)

---

## Does TraceML work with PyTorch Lightning?

Yes.

See:

- [PyTorch Lightning](integrations/lightning.md)

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

Yes, for single-node FSDP.

If you hit an issue on your setup, please open an issue with a minimal repro and environment details.

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
- the best place to start for most users

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

## Is there a summary-only mode?

Yes.

Run:

```bash
traceml run train.py --mode=summary
```

This skips the live UI and focuses on the final end-of-run summary. It is a
good fit when you want lower terminal noise or want to forward TraceML summary
fields into W&B or MLflow.

---

## Can TraceML compare two runs?

Yes.

Use:

```bash
traceml compare run_a.json run_b.json
```

`traceml compare` is designed to consume TraceML `final_summary.json`
artifacts.


It writes:

- a structured compare JSON
- a compact text report

A good workflow is:

1. run each job with TraceML
2. retain `final_summary.json` for each run
3. compare the two runs with `traceml compare`

See:

- [Compare Runs](compare.md)

---

## Can I log TraceML output into W&B or MLflow?

Yes.

TraceML is designed to work alongside your existing tracking stack. The
recommended low-noise path is:

1. launch with `traceml run train.py --mode=summary`
2. call `traceml.final_summary()` near the end of your script
3. log selected fields from the returned dict into W&B or MLflow

See:

- [Use TraceML with W&B / MLflow](integrations/wandb-mlflow.md)

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

- [How to Read TraceML Output](reading-output.md)

---

## What does `INPUT STRAGGLER` mean?

It means one rank is slower in the input path than the typical rank.

Common causes:

- uneven data loading
- preprocessing imbalance
- host-side jitter

See:

- [How to Read TraceML Output](reading-output.md)

---

## What does `COMPUTE STRAGGLER` mean?

It means one rank is slower in compute than the typical rank.

Common causes:

- uneven shapes or data
- rank-local branching or extra work
- compute imbalance in forward, backward, or optimizer

See:

- [How to Read TraceML Output](reading-output.md)

---

## When should I use compare instead of live output?

Use live output when you want to understand the current run while it is still in progress.

Use compare when you already have final summary JSON files and want to answer:

- did the run get slower or faster?
- did the diagnosis change?
- did memory or wait behavior regress?

Live output is for in-run diagnosis.

Compare is for run-to-run review after the runs have finished.
