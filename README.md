<div align="center">

# TraceML

**Find training bottlenecks live, while the run is still running.**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/traceopt-ai/traceml?style=social)](https://github.com/traceopt-ai/traceml)
[![GitHub issues](https://img.shields.io/github/issues/traceopt-ai/traceml)](https://github.com/traceopt-ai/traceml/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)

[**Quickstart**](docs/quickstart.md) • [**Examples**](src/examples) • [**Contributing**](#contributing)

</div>

TraceML is a lightweight bottleneck finder for PyTorch training. It helps you catch:

- input stalls
- unstable or drifting step times
- DDP rank stragglers
- memory creep over time

without jumping straight to heavyweight profiling.

Think of it as **`htop` for GPU training** — always-on, lightweight, and focused on what's slowing you down.

**The gap it fills:** system dashboards show utilization over time. TraceML shows what happened **during training steps** and, in DDP, **which rank is slowing the run down**.

**Works today:** Single GPU, Single-node DDP

**Not yet:** Multi-node DDP, FSDP, TP, PP

Minimal setup with system and process behaviour during training

```bash
pip install "traceml-ai[torch]"
traceml watch train.py
```

---

## When to use TraceML

Use it when training feels:

- slower than expected
- jittery from step to step
- imbalanced across DDP ranks
- stable in dashboards but still underperforming

Start with TraceML when you need a fast answer in the terminal. Reach for `torch.profiler` after you know where to dig.

---

## Quick start

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| PyTorch | 2.5+ | [Install guide](https://pytorch.org/get-started/locally/) |
| CUDA toolkit | optional | For GPU support |

### Install

```bash
# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate
# or: conda create -n traceml python=3.10 && conda activate traceml

# Install TraceML with PyTorch
pip install "traceml-ai[torch]"
```

If PyTorch 2.5+ is already installed in your environment:

```bash
pip install traceml-ai
```

#### Optional extras

```bash
pip install "traceml-ai[hf]"        # Hugging Face Trainer (transformers, accelerate)
pip install "traceml-ai[lightning]"  # PyTorch Lightning
```

#### Development (editable install from source)

```bash
git clone https://github.com/traceopt-ai/traceml.git
cd traceml
pip install -e ".[dev,torch]"
```

#### Verify

```bash
traceml --help
python -c "import traceml; import torch; print('Ready')"
```

### Zero-code first look

```bash
traceml watch train.py
```

Use `watch` for a zero-code live view of system and process behavior during training.

### Step-aware bottleneck diagnosis

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

During training, TraceML opens a live CLI view alongside your logs.

![TraceML terminal dashboard](docs/assets/cli_demo_v1.png)

At the end of the run, it prints a compact summary.

![TraceML summary](docs/assets/end-of-run-summary.png)

For local review and comparison, TraceML also includes a local UI. See [`docs/quickstart.md`](docs/quickstart.md) for setup details.

![TraceML local UI](docs/assets/local_ui.png)

---

## Run modes

#### `traceml watch train.py`
Zero-code live visibility for system and process behavior.

#### `traceml run train.py`
Default mode for live bottleneck diagnosis.

#### `traceml deep train.py`
Adds per-layer timing and memory signals for deeper inspection.

Start with `watch` for fast visibility. Use `run` when you need step-aware diagnosis. Use `deep` only when you need layer-level root cause.

---

## Features

| Feature | `watch` | `run` | `deep` |
|---|:---:|:---:|:---:|
| CPU / RAM / GPU utilization | &#x2705; | &#x2705; | &#x2705; |
| Step time breakdown | | &#x2705; | &#x2705; |
| Dataloader / input stall detection | | &#x2705; | &#x2705; |
| Forward / backward / optimizer timing | | &#x2705; | &#x2705; |
| Step jitter and drift | | &#x2705; | &#x2705; |
| GPU memory trend | | &#x2705; | &#x2705; |
| DDP rank imbalance (single-node) | | &#x2705; | &#x2705; |
| Per-layer timing and memory | | | &#x2705; |
| End-of-run summary | | &#x2705; | &#x2705; |
| Local UI dashboard | | &#x2705; | &#x2705; |

| Integration | Status |
|---|:---:|
| Single GPU | &#x2705; |
| Single-node DDP | &#x2705; |
| Hugging Face Trainer | &#x2705; |
| PyTorch Lightning | &#x2705; |
| Multi-node DDP | planned |
| FSDP / TP / PP | planned |

---

## Supported stacks

### Plain PyTorch
Use `trace_step(model)` around your training step.

### Hugging Face Trainer
```python
from traceml.integrations.huggingface import TraceMLTrainer

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
from traceml.integrations.lightning import TraceMLCallback

trainer = L.Trainer(callbacks=[TraceMLCallback()])
```

See [`docs/lightning.md`](docs/lightning.md) for the full setup.

---

## Optional model hooks

```python
from traceml.decorators import trace_model_instance

trace_model_instance(model)
```

Use this with `trace_step(model)` when you want optional per-layer timing and memory signals. The core step-level view works without it.

---

## TraceML vs alternatives

| Capability | TraceML | W&B / Neptune | TensorBoard Profiler |
|---|:---:|:---:|:---:|
| Step phase breakdown (fwd/bwd/opt) | Auto | Manual logging | Manual logging |
| Dataloader stall detection | Auto | Not available | Not available |
| DDP rank imbalance | Auto | Not available | Not available |
| Per-layer timing and memory | Auto | Not available | Via profiler plugin |
| CPU / RAM / GPU utilization | Auto | Auto | Not available |
| Runs fully offline, no account | &#x2705; | &#x274C; | &#x2705; |
| Live terminal view (~1s latency) | &#x2705; | &#x274C; | &#x274C; |
| Safe for full training runs | &#x2705; | &#x2705; | &#x274C; (heavy) |
| Experiment tracking & comparison | &#x274C; | &#x2705; | &#x2705; |
| Hyperparameter sweeps | &#x274C; | &#x2705; | &#x274C; |
| Team collaboration | &#x274C; | &#x2705; | &#x274C; |

**TraceML is not a replacement for W&B or Neptune** — it's a complement. Use TraceML to find where training time is wasted. Use W&B/Neptune to track experiments and collaborate.

---

## Examples

| Example | Requires | Description |
|---|---|---|
| [`basic_example.py`](src/examples/basic_example.py) | `traceml-ai[torch]` | Single GPU, minimal setup |
| [`input-stall.py`](src/examples/input-stall.py) | `traceml-ai[torch]` | Detect dataloader bottlenecks |
| [`ddp_example.py`](src/examples/ddp_example.py) | `traceml-ai[torch]` | Single-node DDP |
| [`straggler_ddp_example.py`](src/examples/straggler_ddp_example.py) | `traceml-ai[torch]` | DDP rank imbalance |
| [`hf-trainer-minimal.py`](src/examples/hf-trainer-minimal.py) | `traceml-ai[hf]` | Hugging Face Trainer integration |

See [`src/examples/advanced/`](src/examples/advanced) for more (BERT, ViT, LLaMA fine-tuning, Lightning, etc.).

---

## Feedback

If TraceML caught a slowdown for you, please open an issue and include:

- hardware / CUDA / PyTorch versions
- single GPU or DDP
- whether you used `watch`, `run`, or `deep`
- whether you used core tracing only or model hooks
- the end-of-run summary
- a minimal repro if possible

📧 Email: support@traceopt.ai

📋 User Survey: https://forms.gle/KwPSLaPmJnJjoVXSA

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

PyTorch is not included with the base `traceml-ai` install. Install with extras:

```bash
pip install "traceml-ai[torch]"
```

### `ModuleNotFoundError: No module named 'transformers'`

Install the Hugging Face extras:

```bash
pip install "traceml-ai[hf]"
```

### `torchrun: command not found`

Check if `torchrun` is available via module:

```bash
python -m torch.distributed.run --help
```

If that works but `torchrun` does not, fix your PATH or reinstall PyTorch.

### No GPU detected

TraceML works on CPU too. GPU memory signals will show `N/A`, but step timing still works.

See [`docs/quickstart.md`](docs/quickstart.md) for more troubleshooting tips.

---

## Contributing

Contributions are welcome, especially:

- reproducible slowdown cases
- integrations
- bug reports
- examples

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## License

Apache 2.0. See [`LICENSE`](LICENSE).
