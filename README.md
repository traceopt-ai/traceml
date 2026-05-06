<div align="center">

# TraceML

**Catch PyTorch training slowdowns early, while the job is still running.**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/traceopt-ai/traceml?style=social)](https://github.com/traceopt-ai/traceml)

[**Quickstart**](docs/quickstart.md) • [**Compare Runs**](docs/compare.md) • [**How to Read Output**](docs/how-to-read-output.md) • [**FAQ**](docs/faq.md) • [**Use with W&B / MLflow**](docs/use-with-wandb-mlflow.md) • [**Issues**](https://github.com/traceopt-ai/traceml/issues)

</div>

TraceML is an open-source tool for catching PyTorch training slowdowns early, so bad runs do not quietly waste costly compute.

It gives you lightweight step-level signals while the job is still running, so you can quickly tell whether the slowdown looks input-bound, compute-bound, wait-heavy, imbalanced across ranks, or memory-related.

Use TraceML when you want a fast answer before reaching for a heavyweight profiler.

**⭐ If TraceML helps you, please consider starring the repo.**

> **Upcoming rename:** TraceML will transition to **TraceOpt** in a future release.
> For now, the active package remains `traceml-ai` and Python imports remain `traceml`.
> The future PyPI package name [`traceopt-ai`](https://pypi.org/project/traceopt-ai/) is now in place as we prepare the migration.

---

## The fastest way to try it

Install:

```bash
pip install traceml-ai
```

Initialize TraceML and wrap your training step:

```python
import traceml

traceml.init(mode="auto")

for batch in dataloader:
    with traceml.trace_step(model):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()
```

Run:

```bash
traceml run train.py
```

During training, TraceML opens a live terminal view alongside your logs.

![TraceML terminal dashboard](docs/assets/cli_demo_v1.png)

At the end of the run, it prints a compact summary you can review or share.

![TraceML summary](docs/assets/end-of-run-summary.png)

Start with `traceml run train.py`. Most users do not need `watch` or `deep` first.

> Legacy imports from `traceml.decorators` still work for backward compatibility.
> The preferred interface is now the top-level `traceml.*`.
> Legacy decorator imports are planned for deprecation starting in `v0.3.0`.

TraceML supports three initialization modes:

- `traceml.init(mode="auto")` for the default patch-based workflow
- `traceml.init(mode="manual")` for fully explicit wrapper-based instrumentation
- `traceml.init(mode="selective", ...)` when you want part automatic and part explicit

Manual and selective flows can use:

- `traceml.wrap_dataloader_fetch(...)`
- `traceml.wrap_forward(...)`
- `traceml.wrap_backward(...)`
- `traceml.wrap_optimizer(...)`

---

## Core workflows

### 1. Live diagnosis

Use the default workflow when you want live step-aware diagnosis during training plus the end-of-run summary.

```bash
traceml run train.py
```

### 2. Low-noise summary runs

Use summary mode when you mainly want the structured final summary for logging into W&B or MLflow.

```bash
traceml run train.py --mode=summary
```

Then call `traceml.final_summary()` near the end of your script.

TraceML also writes canonical summary artifacts for the run, including `final_summary.json`, which is the intended machine-readable output for downstream logging and later run comparison.


### 3. Compare two runs

If you have `final_summary.json` from two runs, compare them directly:

```bash
traceml compare run_a.json run_b.json
```

TraceML writes both a structured compare JSON and a compact text report.

See [docs/compare.md](docs/compare.md).

---

## What TraceML helps you see

TraceML is currently strongest at surfacing:

- step-time slowdowns while training is still running
- whether the pattern looks input-bound, compute-bound, or wait-heavy
- whether work is uneven across distributed ranks
- whether memory is drifting upward over time
- where time is showing up across dataloader, forward, backward, and optimizer phases

It is designed to help you decide quickly whether a run looks healthy or whether it is worth digging deeper.

---

## Overhead

TraceML adds fixed per-step instrumentation overhead, so the relative cost is highest when training steps are very short. In larger or distributed workloads, that fixed cost is amortized over a longer end-to-end step. In our early DDP benchmarks, TraceML did not produce a measurable slowdown beyond normal run-to-run variation.

---

## When to use TraceML

Use TraceML when training feels:

- slower than expected
- unstable from step to step
- imbalanced across distributed ranks
- fine in dashboards but still underperforming

Start with TraceML when you need a fast answer in the terminal.
Reach for `torch.profiler` once you know where to dig deeper.

---

## How it fits with your stack

TraceML is designed to work alongside tools like W&B, MLflow, and TensorBoard.

Use those for:

- experiment tracking
- artifacts
- dashboards
- team reporting

Use TraceML for:

- bottleneck diagnosis while a run is still in progress
- spotting throughput drift during a run
- checking for rank imbalance or straggler patterns
- checking for memory creep or pressure signals
- structured final summaries you can forward into W&B or MLflow
- simple run-to-run comparison from saved TraceML summary JSON files

See [Use TraceML with W&B / MLflow](docs/use-with-wandb-mlflow.md).

---

## Current support

**Works today:**

- single GPU
- single-node DDP/FSDP

**Not yet:**

- multi-node
- tensor parallel
- pipeline parallel

`deep` remains available for deeper follow-up inspection. If `deep` is important
for your workflow, please let us know in [GitHub issues](https://github.com/traceopt-ai/traceml/issues).

---

## Roadmap

TraceML is focused on making distributed training slowdowns easier to diagnose while the job is still running.

Current priorities:

- stronger barrier and communication diagnostics for DDP/FSDP, with clearer separation of compute, communication, and synchronization wait
- multi-node training support, including common launch environments such as Slurm and Ray
- more structured final summary JSON for experiment trackers, CI workflows, and run-to-run comparisons
- broader coverage across common training stacks, including Hugging Face Trainer, PyTorch Lightning, FSDP, and potentially DeepSpeed
- streaming telemetry export for live consumers, starting with simple JSONL-style events and keeping OpenTelemetry compatibility in mind

---

## Learn more

- [Quickstart](docs/quickstart.md)
- [Compare Runs](docs/compare.md)
- [Examples](examples/README.md)
- [How to Read TraceML Output](docs/how-to-read-output.md)
- [FAQ](docs/faq.md)
- [Use TraceML with W&B / MLflow](docs/use-with-wandb-mlflow.md)
- Hugging Face integration: `docs/huggingface.md`
- PyTorch Lightning integration: `docs/lightning.md`

Need a lighter zero-code first look or a deeper follow-up run? See the Quickstart and FAQ for `watch` and `deep`.

---

## Feedback

If TraceML helped you catch a slowdown, please open an issue and include:

- hardware / CUDA / PyTorch versions
- single GPU or multi-GPU
- whether you used `run`, `watch`, or `deep`
- the end-of-run summary
- a minimal repro if possible

GitHub issues: https://github.com/traceopt-ai/traceml/issues

Email: support@traceopt.ai

---

## Contributing

Contributions are welcome, especially:

- reproducible slowdown cases
- bug reports
- docs improvements
- integrations
- examples

---

## License

Apache 2.0. See [LICENSE](LICENSE).

TraceOpt is a trademark of OptAI UG (haftungsbeschränkt).
