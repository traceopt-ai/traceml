# TraceML

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![GitHub Stars](https://img.shields.io/github/stars/abhinavsriva/trace_ml?style=social)](https://github.com/traceml-ai/traceml/stargazers)


A simple CLI tool to automatically trace PyTorch training memory usage.

## The Problem

Training large machine learning models often feels like a black box. One minute everything's running — the next, you're staring at a cryptic **"CUDA out of memory"** error.

Pinpointing which part of the model is consuming too much memory or slowing things down is frustrating and time-consuming. Traditional profiling tools can be overly complex or lack the granularity deep learning developers need.

## Why TraceML?

`traceml` is a lightweight CLI tool to instrument your PyTorch training scripts and get real-time, granular insights into:

- System and process-level **CPU, GPU & RAM usage**
- PyTorch **layer-level memory allocation** (via GC or decorator/instance tracing)
- All shown live in your terminal — no config, no setup, just plug-and-trace.

## 📦 Installation

```bash
pip install -e .
```

## 🚀 Usage

TraceML wraps your training script and prints memory insights to the terminal as your model trains:

```bash
traceml run <your_training_script.py>
```

### Registering your model for tracing

To capture **memory usage**, you need to register your model with TraceML. There are two simple ways:

#### 1. With a class decorator (recommended)

```python
import torch.nn as nn
from traceml.utils.patch import trace_model

@trace_model()
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)
```

✅ Any instance of `TinyNet` will now be automatically traced.

#### 2. With an explicit model instance

```python
import torch.nn as nn
from traceml.utils.patch import trace_model_instance

model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
).to("cuda")

# Attach hooks so TraceML can see memory events
trace_model_instance(model)
```

✅ Best when you build models dynamically or don't want to decorate the class.

## Examples

```bash
# Trace an explicitly defined model instance
traceml run src/examples/tracing_with_model_instance

# Trace a model using a class decorator (recommended)
traceml run src/examples/tracing_with_class_decorator
```

## Current Features

- Live **CPU, RAM, and GPU** usage (System + Current Process)
- PyTorch **layer-level memory tracking**
  - Via `@trace_model` class decorator
  - Via `trace_model_instance()` for manual model instance tracing
- Model memory summaries (per-layer + total)
- Activation memory tracking
- Gradient memory tracking

## Coming Soon

- Notebook support
- Export logs as JSON / CSV

## 🙌 Contribute & Feedback

Found it useful? Please ⭐ the repo! Got ideas, feedback, or feature requests? I'd love to hear from you.

📧 **Email**: traceml.ai@gmail.com

---

*TraceML - Making PyTorch memory usage visible, one trace at a time.*
