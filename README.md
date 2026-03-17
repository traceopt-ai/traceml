<div align="center">

# TraceML

**Find why PyTorch training got slow — while the run is still live.**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/traceopt-ai/traceml?style=social)](https://github.com/traceopt-ai/traceml)
[![GitHub issues](https://img.shields.io/github/issues/traceopt-ai/traceml)](https://github.com/traceopt-ai/traceml/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)

[**Quickstart**](docs/quickstart.md) • [**Examples**](src/examples) • [**Contributing**](#contributing)

</div>

TraceML is a lightweight, step-aware bottleneck finder for PyTorch training. It helps you catch:

- input stalls
- unstable or drifting step times
- DDP rank stragglers
- memory creep over time

without jumping straight to heavyweight profiling.

**The gap it fills:** system dashboards show utilization over time. TraceML shows what happened **per training step** and, in DDP, **which rank is slowing the run down**.

**Works today:** Single GPU, Single-node DDP

**Not yet:** Multi-node DDP, FSDP, TP, PP

```bash
pip install traceml-ai
traceml run train.py
```

---

## When to use TraceML

Use it when training feels:

- slower than expected
- jittery from step to step
- imbalanced across DDP ranks
- stable in dashboards but still underperforming

Start with TraceML when you need a fast answer. Reach for `torch.profiler` after you know where to dig.

---

## Quick start

Wrap your training step:

```python
from traceml.decorators import trace_step

for batch in dataloader:
    with trace_step(model):
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

Run through TraceML:

```bash
traceml run train.py
```

During training, TraceML opens a live terminal view alongside your logs.

![TraceML terminal dashboard](docs/assets/cli_demo.png)

At the end of the run, it prints a compact summary.

![TraceML summary](docs/assets/end-of-run-summary.png)

For local review and comparison, TraceML also includes a local UI.

![TraceML local UI](docs/assets/local_ui.png)

See [`docs/quickstart.md`](docs/quickstart.md) for setup details.

---

## What TraceML shows

- step time and its breakdown
- dataloader / input wait
- forward / backward / optimizer / overhead timing
- step jitter and drift
- GPU memory trend
- CPU / RAM / GPU signals
- in single-node DDP: worst-rank vs median-rank timing and skew

This helps you tell whether the slowdown is coming from input, compute, optimizer work, or rank imbalance.

---

## Supported stacks

### Plain PyTorch
Use `trace_step(model)` around your training step.

### Hugging Face Trainer
```python
from traceml.hf_decorators import TraceMLTrainer

trainer = TraceMLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    traceml_enabled=True,
)
```

See [`docs/huggingface.md`](docs/huggingface.md).

### PyTorch Lightning
```python
import lightning as L
from traceml.utils.lightning import TraceMLCallback

trainer = L.Trainer(callbacks=[TraceMLCallback()])
```

See the Lightning docs for the full setup.

---

## Run modes

#### `traceml run train.py`
Default mode for live bottleneck diagnosis.

#### `traceml deep train.py`
Adds per-layer timing and memory signals for deeper inspection.

Use `run` by default. Use `deep` only when you need layer-level root cause.

---

## Optional model hooks

```python
from traceml.decorators import trace_model_instance

trace_model_instance(model)
```

Use this with `trace_step(model)` when you want optional per-layer timing and memory signals. The core step-level view works without it.

---

## Scope

TraceML is for lightweight diagnosis during real PyTorch training runs.

It is **not**:

- a kernel-level tracer
- an auto-tuner
- a replacement for deep profilers
- a full observability platform

---

## Example cases

Start with examples such as:

- basic example
- input / dataloader stall
- DDP straggler / rank skew

See [**Examples**](src/examples) for runnable cases.

---

## Feedback

If TraceML caught a slowdown for you, please open an issue and include:

- hardware / CUDA / PyTorch versions
- single GPU or DDP
- whether you used core tracing only or model hooks
- the end-of-run summary
- a minimal repro if possible

📧 Email: abhinav@traceopt.ai

📋 User Survey: https://forms.gle/KwPSLaPmJnJjoVXSA

---

## Contributing

Contributions are welcome, especially:

- reproducible slowdown cases
- integrations
- bug reports
- examples

---

## License

Apache 2.0. See [`LICENSE`](LICENSE).
