# TraceML

**Know what’s slowing your (PyTorch) training, while it runs**


[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![Downloads](https://static.pepy.tech/badge/traceml-ai)](https://pepy.tech/project/traceml-ai)
[![GitHub stars](https://img.shields.io/github/stars/traceopt-ai/traceml?style=social)](https://github.com/traceopt-ai/traceml)
[![Python 3.9-3.13](https://img.shields.io/badge/python-3.9–3.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)

TraceML provides step-level training visibility for PyTorch workloads. It shows where time and memory go inside each training step so you can
quickly understand performance behavior across single-GPU and  single-node DDP runs.

**Current support**
- ✅ Single GPU
- ✅ Single-node multi-GPU (**DDP**)
- ❌ Multi-node DDP (not yet)
- ❌ FSDP / TP / PP (not yet)

---

## What You See in Minutes

-   System signals (CPU, RAM, GPU)
-   Breakdown of each training step:
    -   `dataloader → forward → backward → optimizer → overhead`
-   Median vs worst rank (in case of DDP)
-   Skew (%) to surface imbalance
-   GPU memory (allocated + peak)


Healthy runs are clearly stable. Unstable runs reveal drift, imbalance, or memory creep early.

---
## Quick Start

Install:

``` bash
pip install traceml-ai
```

Wrap your training step:

``` python
from traceml.decorators import trace_step

for batch in dataloader:
    with trace_step(model):
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

Run with cli:

``` bash
traceml run train.py 
```

The terminal dashboard opens alongside your logs.
![TraceML terminal dashboard](cli_demo_v1.png)



Optional web UI:

``` bash
traceml run train.py --mode=dashboard
```

![TraceML web dashboard](web_demo_v1.png)

---

## What TraceML Surfaces

### Step-Level Signals

-   Dataloader fetch time
-   Step time (low-overhead, GPU-aware)
-   Step GPU memory (allocated + peak)

Across ranks:

-   Median (typical behavior)
-   Worst rank (slowest / highest memory)
-   Skew (% difference)

This makes rank imbalance and straggler behavior immediately visible.

---

## Deep-Dive Mode (Optional)

Enable model-level hooks for diagnostic context:

``` python
from traceml.decorators import trace_model_instance
trace_model_instance(model)
```

Use together with `trace_step(model)` to enable:

-   Per-layer memory signals
-   Per-layer forward/backward timing
-   Lightweight failure attribution (experimental)

If not enabled, ESSENTIAL signals remain unchanged.

---

## What It Is Not

-   Not a replacement for PyTorch Profiler or Nsight
-   Not an auto-tuner
-   Not a kernel-level tracer

TraceML focuses on step-level visibility that is practical during real
training runs.

---

## Supported Environments

-   Python 3.9--3.13
-   PyTorch 1.12+
-   macOS (Intel/ARM), Linux
-   Single GPU
-   Single-node DDP

---

## Hugging Face Integration

TraceML provides a seamless integration with Hugging Face `transformers` via `TraceMLTrainer`.

### Usage

Replace `transformers.Trainer` with `traceml.hf_decorators.TraceMLTrainer`.

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

---

## Roadmap

Near-term: - Single-node DDP hardening - Disk run logging -
Compatibility validation (gradient accumulation, torch.compile) -
Accelerate / Lightning wrappers

Next: - Multi-node DDP - Initial FSDP support

Later: - Tensor / Pipeline parallel awareness




---

## Contributing


Contributions are welcome.

When opening issues, include: - Minimal repro script - Hardware + CUDA +
PyTorch versions - ESSENTIAL vs DEEP-DIVE - Single GPU vs DDP

---

## Community & Support

Founding Engineer / Co-Founder track (Berlin/Germany): We are looking 
for a senior systems+ML builder to help grow TraceML into a sustainable AI 
infra product. See the GitHub Discussion https://github.com/traceopt-ai/traceml/discussions/36

- 📧 Email: abhinav@traceopt.ai
- 🐙 LinkedIn: [Abhinav Srivastav](https://www.linkedin.com/in/abhinavsriva/)
- 📋 User Survey (2 min): https://forms.gle/KwPSLaPmJnJjoVXSA

Stars help more teams find the project. 🌟

<a href="https://www.star-history.com/#traceopt-ai/traceml&type=date&legend=top-left">
  <img src="https://api.star-history.com/svg?repos=traceopt-ai/traceml&type=date&legend=top-left" width="50%">
</a>

---

## License

TraceML is released under the **Apache 2.0**.

See [LICENSE](./LICENSE) for details.

---

## Citation

If TraceML helps your research, please cite:

```bibtex
@software{traceml2024,
  author = {TraceOpt},
  title = {TraceML: Real-time Training Observability for PyTorch},
  year = {2024},
  url = {https://github.com/traceopt-ai/traceml}
}
```

---

<div align="center">

Made with ❤️ by TraceOpt

</div>
