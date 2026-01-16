# TraceML

**Real-time training observability and failure attribution tool for PyTorch — lightweight, always-on, and actionable.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/traceml-ai?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/traceml-ai)
[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![Python 3.9-3.13](https://img.shields.io/badge/python-3.9–3.13-blue)](https://www.python.org/) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/src/examples/tracing_bert_notebook.ipynb)

---

## The Problem TraceML Solves

Training deep learning models shouldn't feel like debugging a black box. Yet we constantly face:

- **💥 CUDA OOM errors** with no insight into which layer caused the memory spike
- **🐌 Slow training** without knowing if the bottleneck is data loading, forward pass, backward pass, or optimizer
- **🔍 Layer-level mysteries** — which layers consume the most memory? Which are slowest?
- **📊 Heavy profilers** that are impractical to keep running during actual training

TraceML changes this with continuous, low-overhead visibility while your training runs. 

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

## Tracking Profiles (New)

TraceML supports two tracking profiles so you can choose the right trade-off between insight and overhead.

### ESSENTIAL mode (lightweight, always-on)

Best for day-to-day training and long runs.

Tracks:
- Dataloader fetch time
- Training step time (GPU-aware)
- Step GPU memory (allocated + peak)
- System stats (CPU, RAM, GPU)

### DEEP-DIVE mode (diagnostic)

Best for debugging OOMs and performance pathologies.

Tracks everything in **Essential**, plus:
- Per-layer memory (parameters, activations, gradients)
- Per-layer forward and backward time


### Timed Regions (optional, very low overhead)

**Optional instrumentation for specific code blocks.**  
Executed **once per step per decorated function**.

Tracks:
- Custom timing blocks (e.g. dataloader, forward, backward, optimizer)
- CPU or GPU time (via CUDA events)
- Low (one timing measurement per step)



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

**Requirements:** Python 3.9-3.13, PyTorch 1.12+

**Platform support:** macOS (Intel/ARM), Linux. Single-GPU training (DDP support coming soon).

---

## Quick Start (Important)

### Step-level tracking (required for all modes)

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

Notes:
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


- ***Performance & Stability Improvements***: 
Continuous reduction of tracing overhead, improved robustness for long-running training jobs, and better defaults for production-scale workloads.

- ***Distributed Training Support***:
Support for multi-GPU training (DDP / FSDP) and, over time, multi-node distributed setups with clear failure and performance attribution.

- ***Framework Integrations***:
Native integrations with popular training frameworks such as PyTorch Lightning and Hugging Face Accelerate.

- ***Advanced Diagnostics***:
Memory leak detection, clearer attribution of performance regressions, and richer debugging signals for complex training runs.

- ***Actionable Insights & Automation***:
Smarter summaries and recommendations to help users identify bottlenecks and optimize training configurations.

---



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

**TraceML — Stop guessing. Start profiling.**

Made with ❤️ by [TraceOpt AI](https://traceopt.ai)

</div>
