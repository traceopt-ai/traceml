# TraceML

**Always-on, live observability and failure attribution for distributed PyTorch training**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![Downloads](https://static.pepy.tech/badge/traceml-ai)](https://pepy.tech/project/traceml-ai)
[![GitHub stars](https://img.shields.io/github/stars/traceopt-ai/traceml?style=social)](https://github.com/traceopt-ai/traceml)
[![Python 3.9-3.13](https://img.shields.io/badge/python-3.9–3.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20Commons%20Clause-yellow)](./LICENSE)


TraceML is a lightweight **runtime observability** tool for **distributed PyTorch training** (Single Node Multi-GPU).
It makes training behavior visible *while it runs* using **semantic, step-level signals** aligned with model execution.

> Current focus: **single-node DDP**.
> We are working on Multi-node distributed training.

---

## Why TraceML

Training deep learning models often becomes a systems-level black box once you scale beyond toy workloads.

Common pain points:
- **Slow or unstable steps** without knowing whether the bottleneck is **data loading, compute, or communication**
- **Distributed blind spots**: unclear where time is lost across ranks, phases, or synchronization points
- **Limited always-on visibility** during real training runs

TraceML is designed to be **always on**, providing actionable attribution during long-running jobs.

---

## What TraceML shows (core signals)

TraceML focuses on the signals that explain training behavior at runtime:

### Step-aware signals (synchronized across ranks)
For each training step (in single-node DDP):
- **Dataloader fetch time**
- **Step time**
- **Forward / backward / optimizer** timings (DEEP-DIVE) are **CUDA-event estimates** on the current CUDA stream.
- GPU timings are best for **relative comparisons and trend/bottleneck detection**; they may not sum to wall time due to overlap (compute/comm) and multi-stream execution.

Across ranks, TraceML reports:
- **Median rank** (typical behavior)
- **Worst rank** (straggler / bottleneck)

This makes it easy to catch cases like “8 GPUs slower than 1” *as it happens*, and understand whether you are bottlenecked by input pipeline, compute, or rank-level stragglers.


---

## What TraceML is not

TraceML is **NOT** an auto-tuner or a profiler replacement.

- It does not automatically optimize your batch size
- It does not always “find a problem”
- It does not replace Nsight or PyTorch Profiler

Instead, TraceML answers a core question:

> “Where is time and memory actually going in each training step and is that expected?”


---

## Views

TraceML supports two ways to consume runtime signals:

- 🖥️ **Terminal dashboard** — live updates in your console
- 🌐 **Web dashboard** — local browser at `http://localhost:8765`

Note: The notebook is temporarily unavailable and will be restored shortly.

---

## Tracking Profiles

TraceML provides two tracking profiles so you can choose the right trade-off between insight and overhead.

### ESSENTIAL mode (always-on runtime signals)
Designed for day-to-day training and long-running jobs.

Tracks:
- Dataloader fetch time
- Training step time (GPU-aware)
- Step-level GPU memory (allocated and peak)
- System metrics (CPU, RAM, GPU)
- Basic failure signals

This mode is intended to run **continuously during real training**.

### DEEP-DIVE mode (diagnostic)
Designed for performance pathology debugging and OOM investigations.

Includes everything in **ESSENTIAL**, plus:
- Per-layer memory (parameters, activations, gradients)
- Per-layer forward and backward compute time
- OOM layer attribution (forward/backward)

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
**Training support:** Single GPU and **single-node DDP (alpha)**

---

## Quick Start

### 1) Step-level tracking (required)

TraceML computes step timing / memory only inside a `trace_step()` scope.

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

Without `trace_step()`:
- Step timing is not computed
- Step memory is not recorded
- Live dashboards will not update

---

### 2) Optional: Time specific code regions

Use `@trace_time` to time specific functions.
This works in **all modes** and is designed to have **low overhead**.

```python
from traceml.decorators import trace_time

@trace_time("backward", use_gpu=True)
def backward_pass(loss):
    loss.backward()
```

Notes:
- `use_gpu=True` uses CUDA events (correct for async GPU work)
- `use_gpu=False` uses CPU wall-clock time

#### Deprecation (Breaking change)
- `@trace_timestep` is deprecated — use `@trace_time` instead

---

### 3) Deep-Dive: model registration (only for Deep-Dive)

```python
from traceml.decorators import trace_model_instance

trace_model_instance(model)
```

Enables forward/backward hooks required for:
- per-layer memory and timing (layerwise worst across ranks)
- OOM layer attribution (experimental, work-in-progress)

---

## Running TraceML

```bash
traceml run train.py --nproc-per-node=2
```

You’ll see a live terminal dashboard tracking:
- System resources (CPU, RAM, GPU)
- Dataloader fetch time, step time, step GPU memory
- (Deep-Dive only) per-layer memory + compute time

> Tip: for **DDP**, run TraceML on rank 0 and collect rank signals via the TraceML runtime.

---

## Web Dashboard

```bash
traceml run train.py --nproc-per-node=2 --mode=dashboard
```

Opens `http://localhost:8765` with interactive charts and real-time updates.

---

## Roadmap

TraceML prioritizes **clear attribution and low overhead** over exhaustive tracing.

Near-term:
- **Optimize single-node DDP**: reduce overhead, improve rank synchronization accuracy, improve comm + GIL behavior
- **Broaden workload coverage**: validated examples + benchmarks for representative workloads:
  - CV (e.g., ResNet / ViT)
  - NLP / LLM fine-tuning (e.g., BERT / small decoder models)
  - Diffusion / vision-language (as time permits)
- **Documentation improvements**: clearer docs + examples (targeting beta)

Next:
- **Multi-node distributed support** (DDP → FSDP)
- Integrations: PyTorch Lightning / Hugging Face Accelerate (as optional wrappers)
- Advanced diagnostics: leak detection, regression attribution, and automated “why is my step slower?” summaries

---

## Contributing

Contributions are welcome.

1. ⭐ Star the repo
2. 🐛 Report bugs via GitHub Issues
3. 💡 Request features / workloads you want supported
4. 🔧 Submit PRs (small focused PRs are ideal)

If you hit an issue, please open a GitHub Issue with:
- minimal repro script
- hardware + CUDA + PyTorch versions
- whether you used ESSENTIAL or DEEP-DIVE
- single GPU vs DDP

We’ll try to respond and resolve quickly.

---

## Community & Support

- 📧 Email: abhinav@traceopt.ai
- 🐙 LinkedIn:  [Abhinav Srivastav](https://www.linkedin.com/in/abhinavsriva/)
- 📋 User Survey: Help shape the roadmap (2 minutes) https://forms.gle/KwPSLaPmJnJjoVXSA
- Stars help the project grow and makes it easier for other to find our work.🌟

<a href="https://www.star-history.com/#traceopt-ai/traceml&type=date&legend=top-left">
  <img src="https://api.star-history.com/svg?repos=traceopt-ai/traceml&type=date&legend=top-left" width="50%">
</a>
---

## License

TraceML is released under the **MIT License with Commons Clause**.

**Summary:**
- ✅ Free for personal use
- ✅ Free for research and academic use
- ✅ Free for internal company use
- ❌ Not allowed for resale or SaaS products

See [LICENSE](./LICENSE) for full details.
For commercial licensing, contact: abhinav@traceopt.ai

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

Made with ❤️ by TraceOpt AI

</div>
