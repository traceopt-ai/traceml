# TraceML

**Find why PyTorch training is slow, while it’s still running**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/traceopt-ai/traceml?style=social)](https://github.com/traceopt-ai/traceml)

TraceML is a lightweight bottleneck finder for PyTorch training.
It helps you catch dataloader stalls, DDP stragglers, unstable step times, and GPU memory creep during real runs, with minimal code changes.

**Works today:** Single GPU, single-node DDP, Hugging Face Trainer, PyTorch Lightning
**Not yet:** Multi-node DDP, FSDP / TP / PP

---

## Why people use TraceML

When a run feels slow or unstable, TraceML helps answer:

- **Is the dataloader the bottleneck?**
- **Are step times drifting or jittering?**
- **Is GPU memory creeping up over time?**
- **How much time is spent in forward, backward, optimizer, and overhead?**
- **Is one DDP rank lagging behind the others?**

TraceML is designed to stay lightweight enough for real training runs, so you can spot issues while the job is still in progress instead of digging only after it finishes.

---


## Quick Start

Install:

```bash
pip install traceml-ai
```

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

Run your script through TraceML:

```bash
traceml run train.py
```

During training, TraceML opens a live terminal view alongside your logs. At run end, it also prints a compact summary you can review, paste into an issue, or share with others.

![TraceML terminal dashboard](cli_demo_v1.png)

**Want more setup details?** See [docs/quickstart.md](docs/quickstart.md).

---

## Works with your training stack

### Plain PyTorch
Use `trace_step(model)` around your training step.

### Hugging Face Trainer
Replace `Trainer` with `TraceMLTrainer`:

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

See [docs/huggingface.md](docs/huggingface.md).

### PyTorch Lightning
Add `TraceMLCallback()` to your trainer:

```python
import lightning as L
from traceml.utils.lightning import TraceMLCallback

trainer = L.Trainer(callbacks=[TraceMLCallback()])
```

See the Lightning docs for the full setup.

---

## What TraceML shows

### Per-step breakdown

TraceML tracks:

- `dataloader → forward → backward → optimizer → overhead`
- step time
- GPU memory (allocated + peak)
- CPU / RAM / GPU signals

### Across ranks in DDP

TraceML surfaces:

- **median rank** for typical behavior
- **worst rank** for the slowest or most memory-heavy rank
- **skew (%)** to make imbalance easy to spot

This makes straggler behavior and rank imbalance visible without extra instrumentation.

---


## Optional: layer-level diagnostics

If you want additional model-level context, enable lightweight hooks:

```python
from traceml.decorators import trace_model_instance

trace_model_instance(model)
```

Use this together with `trace_step(model)` to add optional per-layer timing and memory signals.
The core step-level view works without it.

---

## Optional dashboard view

You can also use the dashboard mode:

```bash
traceml run train.py --mode=dashboard
```

![TraceML web dashboard](web_demo_v1.png)

---


## Supported today

| Area | Status |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.5+ |
| OS | Linux, macOS |
| Single GPU | ✅ |
| Single-node DDP | ✅ |
| Hugging Face Trainer | ✅ |
| PyTorch Lightning | ✅ |
| Multi-node DDP | ❌ |
| FSDP / TP / PP | ❌ |

---

## Safe to try on real runs

TraceML is built for practical training workflows:

- lightweight enough to use during real runs
- compact terminal output during training
- end-of-run summary for quick review and sharing
- fail-open behavior in normal usage so instrumentation does not become the center of your training script

---

## Scope

TraceML focuses on lightweight step-level diagnosis during real training runs.

It is not:

- a kernel-level tracer
- an auto-tuner
- a replacement for deep profiling tools

---

## Feedback

If TraceML helps you find a bottleneck, open an issue and include:

- a minimal repro if possible
- hardware / CUDA / PyTorch versions
- single GPU or DDP
- whether you used core step tracing only or layer-level hooks
- the TraceML summary output

Useful feedback, example scripts, and bug reports are especially valuable right now.

- 📧 Email: abhinav@traceopt.ai
- 📋 User Survey: https://forms.gle/KwPSLaPmJnJjoVXSA

---

## Contributing

Contributions are welcome.

If you want to help, examples, bug reports, reproducible performance cases, and integration feedback are especially useful.

---

## License

TraceML is released under the **Apache 2.0**.
See [LICENSE](./LICENSE) for details.
