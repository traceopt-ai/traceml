# TraceML

**Runtime observability and failure attribution for PyTorch training: step-aware, low-overhead, and always-on.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![Python 3.9-3.13](https://img.shields.io/badge/python-3.9–3.13-blue)](https://www.python.org/) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/src/examples/tracing_bert_notebook.ipynb)


TraceML is a lightweight PyTorch runtime observability tool that makes training behavior *visible while it runs*. 
It focuses on **semantic, step-level signals** that are missing from infrastructure metrics and too expensive to extract with full profilers.

TraceML helps you understand *why* a training step is slow, memory-heavy, or failing without stopping or heavily perturbing your training job.

---

## The Problem TraceML Solves

Training deep learning models often feels like debugging a black box especially once you move beyond toy workloads.

Common pain points include:

- **CUDA OOM errors** with no attribution to the responsible layer
- **Slow or unstable training steps** without knowing whether the bottleneck is data loading, compute, communication, or the optimizer
- **Layer-level opacity** unclear memory and compute hotspots
- **Heavy profilers** that are too intrusive to keep enabled during real training

TraceML addresses these problems by providing **continuous, low-overhead, step-aware observability** during training.

---

## What TraceML Does

TraceML answers the questions you actually need answered:

| Question                              | TraceML Answer                                                 |
|---------------------------------------|----------------------------------------------------------------|
| Which **layer caused OOM** ?          | Automatic detection of the failing layer during forward or backward pass        |
| What's slowing down my training step? | Step-level timing: dataloader → forward → backward → optimizer |
| Where did that memory spike happen?   | Step-level (batch-level) memory tracking with peak attribution     |
| Which layer is eating my GPU memory?  | Per-layer memory breakdown (params + forward + backward)       |
| Which layer is slow?                  | Per-layer compute time (forward + backward)                    |

**Three ways to view the results:**
- 🖥️ **Terminal** — live updates in your console
- 🌐 **Web UI** — local browser at `localhost:8765`
- 📓 **Jupyter notebooks** — inline visualizations

---

## Tracking Profiles

TraceML supports two tracking profiles so you can choose the right trade-off between insight and overhead.

### ESSENTIAL mode (always-on runtime signals)

Designed for day-to-day training and long-running jobs.

Tracks:
- Dataloader fetch time
- Training step time (GPU-aware)
- Step-level GPU memory (allocated and peak)
- System metrics (CPU, RAM, GPU)
- OOM layer attribution

This mode is designed to run **continuously during real training**, not just short profiling sessions.


### DEEP-DIVE mode (diagnostic)

Designed for debugging performance pathologies and OOM failures.

Includes everything in **ESSENTIAL**, plus:
- Per-layer memory (parameters, activations, gradients)
- Per-layer forward and backward compute time


## Optional: Timed Regions

TraceML supports **explicit, semantic timing regions** via decorators. 
These are executed once per step per decorated function and incur minimal overhead.

Tracks:
- User-defined timing blocks (e.g. dataloader, backward, optimizer)
- CPU wall-clock or GPU time (CUDA events)

---


## Installation

```bash
pip install traceml-ai
```

For development:

```bash
git clone https://github.com/traceopt-ai/traceml.git
cd traceml
pip install -e '.[dev]'
```

**Requirements:** Python 3.9–3.13, PyTorch 1.12+  
**Platform support:** macOS (Intel/ARM), Linux  
**Training support:** Single-GPU training supported. Multi-GPU (DDP/FSDP) support is under active development.

---

## Quick Start 

### Step-level tracking (required)

```python
from traceml.decorators import trace_step

for batch in dataloader:
    with trace_step():
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

Without `trace_step(...)`:
- Step timing is not computed
- Step memory is not recorded
- Live dashboards will not update



### Optional: Timing Specific Code Regions

Use `@trace_time` to time specific functions.
This works in **all modes** and has **low overhead**.

```python
from traceml.decorators import trace_time

@trace_time("backward", use_gpu=True)
def backward_pass(loss, scaler=None):
    loss.backward()
```

Notes:
- `use_gpu=True` uses CUDA events (correct for async GPU work)
- `use_gpu=False` uses CPU wall-clock time

#### Deprecation (⚠️ **Breaking change**)

- `@trace_timestep` is deprecated, use `@trace_time` instead


### Deep-Dive: Model Registration

Required **only for Deep-Dive mode**.

```python
from traceml.decorators import trace_model_instance

trace_model_instance(model)
```

- Enables forward/backward hooks
- Required for per-layer memory and timing
- Required for OOM layer attribution

---

### Running TraceML

```python
traceml run train.py 
```

You'll immediately see a live terminal dashboard tracking:
- System resources (CPU, RAM, GPU)
- Dataloader fetch time, training step time and training step GPU memory
- (Deep-Dive only) Per-layer memory and compute time

![TraceML CLI Demo](demo_gc.png)

---

### 🌐 Web Dashboard

```bash
traceml run train.py --mode=dashboard
```

Opens `http://localhost:8765` with interactive charts and real-time updates.

<img src="web_demo.png" width="1200" alt="TraceML CLI Demo">


### 📓 Jupyter Notebooks

Please see the [notebook example](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/src/examples/tracing_bert_notebook.ipynb) for inline visualizations.


---

## Roadmap


TraceML prioritizes **clear attribution and low overhead** over exhaustive tracing.

Planned directions include:

- Performance and stability improvements
- Distributed training support (DDP / FSDP, later multi-node)
- Framework integrations (PyTorch Lightning, Hugging Face Accelerate)
- Advanced diagnostics (memory leaks, regression attribution)



## Contributing

We welcome contributions! Here's how to help:

1. ⭐ **Star the repo** to show support
2. 🐛 **Report bugs** via [GitHub Issues](https://github.com/traceopt-ai/traceml/issues)
3. 💡 **Request features** we should prioritize
4. 🔧 **Submit PRs** for improvements


## Community & Support

- 📧 Email: abhinav@traceopt.ai
- 🐙 LinkedIn: [Abhinav Srivastav](https://www.linkedin.com/in/abhinavsriva/)
- 📋 User Survey: [Help shape the roadmap](https://forms.gle/vaDQao8L81oAoAkv9) (2 minutes)

---

## License

TraceML is released under the **MIT License with Commons Clause**.

**What this means:**
- ✅ Free for personal use
- ✅ Free for research and academic use
- ✅ Free for internal company use
- ❌ Not allowed for resale or SaaS products

For commercial licensing inquiries, contact abhinav@traceopt.ai.

See [LICENSE](./LICENSE) for full details.

---

## Citation

If TraceML helps your research, please cite:

```bibtex
@software{traceml2024,
  author = {TraceOpt AI},
  title = {TraceML: Real-time Training Observability for PyTorch},
  year = {2024},
  url = {https://github.com/traceopt-ai/traceml}
}
```

---

<div align="center">


**TraceML — Stop guessing. Start attributing.**

Made with ❤️ by [TraceOpt AI](https://traceopt.ai)

</div>
