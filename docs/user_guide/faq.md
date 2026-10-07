# FAQ

Common questions about running TraceML and reading its diagnosis. For your
first run, start with the [Quickstart](quickstart.md).

## How much code do I need to change?

For a standard Hugging Face Trainer, PyTorch Lightning `Trainer.fit()`, or
RF-DETR `model.train()` script, launch your existing script with:

```bash
traceml run train.py
```

TraceML attaches the matching integration automatically. Plain PyTorch and
custom loops need explicit setup. See [Integrations](integrations.md) to
choose your path.

## Does TraceML work with Hugging Face Trainer?

Yes. Automatic attachment supports Trainer subclasses that use Hugging Face's
standard inner training loop. See the [Hugging Face guide](integrations/huggingface.md).

## Does TraceML work with PyTorch Lightning?

Yes, with both `lightning.pytorch` and `pytorch_lightning`. See the
[Lightning guide](integrations/lightning.md) for automatic attachment and
supported strategies.

## Does TraceML work with RF-DETR?

Yes, for its supported object-detection `model.train()` path. See the
[RF-DETR guide](integrations/rfdetr.md).

## What is the default run mode?

`traceml run train.py` uses summary mode. It prints the diagnosis when training
finishes and saves `final_summary.json` and `final_summary.txt` in the run
folder. Summary mode is the default for single-node and multi-node runs.

See [Reading the Output](reading-output.md) for the report fields.

## What is the difference between `watch` and `run`?

Use `run` for step timing and training bottleneck diagnosis. It automatically
instruments supported trainers; other training paths need explicit setup.

Use `watch` for system and process visibility without automatic training
instrumentation. It does not provide step measurements or a performance
verdict.

## Can I trace only a small number of steps?

Yes. Record the first 100 TraceML steps while letting training continue:

```bash
traceml run train.py --trace-max-steps 100 --args --epochs 5
```

The limit counts reported TraceML steps, not necessarily individual batches.
For automatic trainers, see the integration guide's accumulation semantics.

## Is there a local UI?

Yes. For a single-node run, including multiple GPUs, use:

```bash
pip install "traceml-ai[dashboard]"
traceml run train.py --mode=dashboard
```

Open `http://127.0.0.1:8765`. Use `--mode=cli` for a live terminal view instead.
Multi-node runs use summary mode.

On a remote server, start the dashboard there and forward its port from your
local terminal:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Then open the same URL locally.

## Does TraceML support DDP?

Yes. TraceML reports per-rank timing and can identify input or compute
stragglers. Single-node DDP supports summary, CLI, and dashboard modes;
multi-node DDP uses summary mode.

Use the [Distributed Training guide](distributed-training.md) and your
framework's integration guide for launch settings.

Capacity-relative GPU memory diagnoses assume equal GPU memory capacity
across ranks. Mixed-capacity runs still collect per-rank telemetry, but
aggregate memory-pressure and imbalance diagnoses may be inaccurate.

## Does TraceML support multi-node?

Yes, for summary-mode DDP runs. Use matching launch settings and a shared
`--logs-dir` on every node, with a different `--node-rank` on each node.
Node 0 owns the aggregator and final reports.

Choose a fresh `--run-name` for each launch. TraceML refuses to overwrite an
existing run folder. See [Distributed Training](distributed-training.md) for
the per-node commands, or [Slurm](slurm.md) for a cluster template.

## Does TraceML support FSDP?

TraceML supports timing and rank-skew reporting for explicitly instrumented
PyTorch FSDP training. This does not imply automatic FSDP support in every
framework integration. Check the [integration coverage](integrations.md)
before choosing a setup.

FSDP communication is not reported in separate collective buckets: forward
and backward can include all-gather or reduce-scatter work. Multi-node FSDP
should be validated in your environment.

## Does TraceML support tensor parallel or pipeline parallel?

Not yet.

## Do I need to replace W&B, MLflow, or TensorBoard?

No. Keep your existing experiment tracker. TraceML adds training timing,
bottleneck diagnosis, and saved evidence for comparison.

See [W&B / MLflow](integrations/wandb-mlflow.md) if you want to export the
TraceML summary to your tracker.

## How is TraceML different from `torch.profiler`?

TraceML reports training phases and likely bottlenecks. `torch.profiler`
provides operator-level traces for a closer investigation. Use the TraceML
diagnosis to decide which part of training to profile next.

## Can TraceML compare two runs?

Yes. Compare their saved final summaries:

```bash
traceml compare logs/reference/final_summary.json logs/candidate/final_summary.json
```

The comparison shows changes in timing, memory, and diagnosis, and writes
JSON and text reports. See [Compare Runs](compare.md).

## Can I catch training regressions in CI?

Yes, with the experimental [Regression Guard](regression-guard.md). Declare
the same workload for the reference and candidate runs, then compare them
with a Step Time threshold. A slower candidate beyond the threshold returns
a failing exit code; missing or incompatible evidence returns an inconclusive
result.

The guard compares aggregate Step Time from the saved summaries. Its declared
measurement range does not select the analyzed steps.

## When should I use compare instead of live output?

Use live output to inspect a run while it is in progress. Use compare after
both runs finish to see whether timing, memory, or the diagnosis changed.
Use Regression Guard when that comparison should determine a CI result.

## Can I log TraceML output into W&B or MLflow?

Yes. After training finishes, call `traceml.summary()` to get a flat dictionary
for tracker logging, or `traceml.final_summary()` for the full report.
Keep the tracker run open until you export it.

These APIs return the finalized report, not live metrics, and return `None`
on non-primary ranks by default. See [W&B / MLflow](integrations/wandb-mlflow.md)
for examples and finalization requirements.

## Can I run without TraceML telemetry for a baseline?

For an automatically instrumented script, use your usual `python` or
`torchrun` command. If the script contains explicit TraceML setup, or you
want to keep the same TraceML launcher settings, use:

```bash
traceml run train.py --disable-traceml
```

Keep the training workload and process count the same when comparing timings.

## Where are training stdout and stderr saved?

`run` and `watch` save both streams by default:

```text
logs/<run-name>/nodes/node_<node-rank>/training.stdout.log
logs/<run-name>/nodes/node_<node-rank>/training.stderr.log
```

Use `--no-save-training-output` to let the training command inherit the
terminal directly. See [Training Crashes](training-crashes.md) for failure
output and the artifacts available after a crash.

## What does `MEMORY CREEP` usually mean?

Memory is rising across measured steps. Retaining tensors in a list or cache
is one possible cause; the diagnosis alone does not prove a memory leak.
See [Reading the Output](reading-output.md) for the supporting evidence.

## What does `INPUT STRAGGLER` mean?

TraceML found rank imbalance with evidence that excess input waiting on one
rank is making another rank wait. Uneven loading, preprocessing, or host
jitter are common causes. Inspect the per-rank evidence in
[Reading the Output](reading-output.md).

## What does `COMPUTE STRAGGLER` mean?

In DDP, the likely culprit rank spends materially more time in forward than
the waiting rank. Uneven shapes or rank-local work can cause this.

FSDP forward can include communication, so unexplained skew is reported as
`STRAGGLER` rather than attributed to compute. See
[Reading the Output](reading-output.md).

## Should I use `traceml.trace_step()` or `trace_step()`?

For explicit loop setup, prefer the top-level API:

```python
import traceml_ai as traceml

traceml.init(mode="auto")

# Inside your training loop:
with traceml.trace_step(model):
    ...  # training work through optimizer.step()
```

Automatic trainer integrations manage their own step boundaries. See the
[Public API](public-api.md) for custom-loop setup and reference details.

## What is the difference between `auto`, `manual`, and `selective`?

These are `init()` instrumentation modes for explicit setup, separate from
the launcher's `summary`, `cli`, and `dashboard` display modes.

- `auto`: patch supported PyTorch operations inside your explicit step boundary.
- `manual`: use explicit timing wrappers.
- `selective`: enable selected patches and wrap other phases yourself.

For a framework integration, use its own setup instructions rather than adding
an extra generic `init()`. For custom loops, call `init()` before creating
TraceML wrappers; models and DataLoaders may already exist.

## When should I use the wrapper APIs?

Use wrappers for explicit timing in `manual` or `selective` mode. Each phase
must have one timing owner: its automatic patch or its wrapper.

The wrappers cover DataLoader fetch, forward, backward, and optimizer work.
In `selective` mode, disable the matching patch before wrapping that phase.
In `auto` mode, standard phase wrappers raise a configuration error; the
exception is fetch wrapping for a custom iterator that the PyTorch DataLoader
patch cannot observe, such as Ray Data.

See the [Public API](public-api.md) for signatures and examples.
