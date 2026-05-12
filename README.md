<div align="center">

# TraceML

**Find where PyTorch training time is going.**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/traceopt-ai/traceml?style=social)](https://github.com/traceopt-ai/traceml)

[**Quickstart**](docs/user_guide/quickstart.md) • [**Compare Runs**](docs/user_guide/compare.md) • [**How to Read Output**](docs/user_guide/reading-output.md) • [**W&B / MLflow**](docs/user_guide/integrations/wandb-mlflow.md) • [**FAQ**](docs/user_guide/faq.md) • [**Issues**](https://github.com/traceopt-ai/traceml/issues)

</div>

TraceML is an open-source runtime diagnostic layer for PyTorch training.

It records lightweight step-level signals during a run, then writes a compact final summary you can review, share, log, or compare against another run. TraceML focuses on the questions that usually come before deep profiling:

- Is the run input-bound, compute-bound, wait-heavy, rank-imbalanced, or memory-related?
- Where is time going across dataloader, forward, backward, and optimizer work?
- Are some distributed ranks consistently slower than others?
- Did memory behavior change during the run?
- Did a code, data, config, or hardware change cause a regression?

**TraceML is not a replacement for PyTorch Profiler, W&B, MLflow, or TensorBoard.** It is the lightweight first pass: find the likely bottleneck, keep a useful artifact, and decide whether deeper investigation is worth it.

**If TraceML helps you catch a slowdown or regression, please consider starring the repo.**

> **Upcoming rename:** TraceML will transition to **TraceOpt** in a future release.
> For now, the active package remains `traceml-ai` and Python imports remain `traceml`.

---

## Quickstart

Install:

```bash
pip install traceml-ai
```

Initialize TraceML and wrap your training step:

```python
import traceml

traceml.init()

for batch in dataloader:
    with traceml.trace_step(model):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()
```

Run your script with TraceML:

```bash
traceml run train.py
```

By default, TraceML keeps the run low-noise and writes an end-of-run summary. Use that summary to review the run, attach evidence to an issue, log selected fields to W&B or MLflow, or compare against another run.

![TraceML summary](docs/assets/end-of-run-summary.png)

---

## Compare Runs

TraceML is most useful when you compare a normal run against a slow or suspicious run.

```bash
traceml compare run_a/final_summary.json run_b/final_summary.json
```

TraceML writes both a structured compare JSON and a compact text report.

See [Compare Runs](docs/user_guide/compare.md).

---

## Common Workflows

### Diagnose One Run

```bash
traceml run train.py
```

Use this when you want the default TraceML workflow: a compact final diagnosis and summary artifacts.

### Zero-Code First Look

```bash
traceml watch train.py
```

Use this when you want system and process telemetry without adding step instrumentation.

Live terminal and dashboard views are available with `--mode=cli` and `--mode=dashboard` when you want to watch a run while it is active.

---

## When To Use TraceML

Use TraceML when:

- training is slower than expected
- GPU dashboards look fine but throughput is still poor
- a recent code, data, config, or hardware change caused a regression
- distributed ranks may be imbalanced
- memory usage grows over time
- you want a small artifact to share with another engineer
- you want a fast first answer before opening PyTorch Profiler or Nsight

TraceML is especially useful for repeated PyTorch training and fine-tuning workflows where small regressions can waste GPU hours.

---

## How It Fits With Your Stack

TraceML works alongside your existing tools.

Use:

- **W&B / MLflow / TensorBoard** for experiment tracking, dashboards, metrics, and artifacts.
- **PyTorch Profiler / Nsight** for deep operator and kernel-level investigation.
- **TraceML** for lightweight runtime diagnosis and run-to-run efficiency comparison.

A common workflow:

```text
Training feels slow
        ↓
Run TraceML
        ↓
Check the final summary
        ↓
Identify the likely bottleneck
        ↓
Compare against a previous run
        ↓
Open PyTorch Profiler only if deeper investigation is needed
```

See [Use TraceML with W&B / MLflow](docs/user_guide/integrations/wandb-mlflow.md).

---

## Current Support

Works today:

- single GPU
- single-node DDP
- single-node FSDP
- custom PyTorch training loops
- Hugging Face training workflows
- PyTorch Lightning workflows

Next:

- multi-node training support
- stronger distributed rank attribution
- richer W&B / MLflow logging examples
- CI-oriented regression checks

---

## Overhead

TraceML adds fixed per-step instrumentation overhead, so relative overhead is highest when training steps are very short. In larger training jobs, that fixed cost is amortized over longer step time.

TraceML is intended as a low-overhead first pass, not a full kernel-level profiler. For detailed profiling, use PyTorch Profiler or Nsight after TraceML has identified where to look.

---

## Learn More

- [Quickstart](docs/user_guide/quickstart.md)
- [Compare Runs](docs/user_guide/compare.md)
- [How to Read TraceML Output](docs/user_guide/reading-output.md)
- [Examples](examples/README.md)
- [FAQ](docs/user_guide/faq.md)
- [Use TraceML with W&B / MLflow](docs/user_guide/integrations/wandb-mlflow.md)
- [Hugging Face integration](docs/user_guide/integrations/huggingface.md)
- [PyTorch Lightning integration](docs/user_guide/integrations/lightning.md)

---

## Feedback

If TraceML helped you catch a slowdown or regression, please open an issue and include:

- hardware / CUDA / PyTorch versions
- single GPU or multi-GPU setup
- training framework
- the end-of-run summary
- whether the issue looked input-bound, compute-bound, wait/rank-skewed, or memory-related
- a minimal repro if possible

GitHub issues: https://github.com/traceopt-ai/traceml/issues

Email: support@traceopt.ai


---

## Contributing

Contributions are welcome, especially:

- real slowdown examples
- distributed training edge cases
- reproducible regression cases
- docs improvements
- framework integrations
- W&B / MLflow examples

---

## License

Apache 2.0. See [LICENSE](LICENSE).

TraceOpt is a trademark of OptAI UG (haftungsbeschränkt).
