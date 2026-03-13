# PyTorch Lightning Integration

TraceML provides `TraceMLCallback`, an official callback for PyTorch Lightning that automatically adds step-level timing and memory visibility to your `Trainer`.

> **Prerequisites:** Follow the [Quickstart](quickstart.md) first to confirm `traceml run` works with a plain PyTorch loop.

---

## Table of Contents

- [PyTorch Lightning Integration](#pytorch-lightning-integration)
  - [Table of Contents](#table-of-contents)
  - [1. Install](#1-install)
  - [2. How it works](#2-how-it-works)
  - [3. Basic usage](#3-basic-usage)
  - [4. Enable Deep-Dive mode](#4-enable-deep-dive-mode)
  - [5. Complete Example: MNIST CNN](#5-complete-example-mnist-cnn)
  - [6. Gradient Accumulation](#6-gradient-accumulation)
  - [7. Trainer tips](#7-trainer-tips)
  - [Next steps](#next-steps)

---

## 1. Install

Install TraceML and PyTorch Lightning:

```bash
pip install "traceml-ai[lightning]"
```

---

## 2. How it works

`TraceMLCallback` hooks into Lightning's lifecycle events (`on_train_batch_start`, `on_before_backward`, `on_train_batch_end`, etc.) to automatically time the forward, backward, and optimizer phases, and to sample GPU memory.

You don't need to wrap your code with `trace_step(model)` because the callback manages the tracing context for you.

---

## 3. Basic usage

Just import `TraceMLCallback` and add it to your `Trainer` callbacks list. Everything else stays the same.

```python
import lightning as L
from traceml.integrations.lightning import TraceMLCallback

model = MyLightningModule()

trainer = L.Trainer(
    max_steps=500,
    accelerator="auto",
    devices=1,
    enable_progress_bar=False,  # Lets the TraceML dashboard own the terminal output
    callbacks=[TraceMLCallback()],
)

trainer.fit(model, train_dataloaders=loader)
```

Then launch with the CLI:

```bash
traceml run train.py
```

Or with the web dashboard:

```bash
traceml run train.py --mode=dashboard
```

---

## 4. Enable Deep-Dive mode

To enable Deep-Dive mode (per-layer memory and timing), you need to attach hooks to your model instance. Since the callback only handles the step-level lifecycle, call `trace_model_instance` directly inside your `LightningModule`'s `__init__`.

```python
from traceml.decorators import trace_model_instance
import lightning as L

class MyLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyCoreModel()
        self.loss_fn = nn.CrossEntropyLoss()

        # Enable Deep-Dive hooks
        trace_model_instance(
            self,
            sample_layer_memory=True,
            trace_layer_forward_memory=True,
            trace_layer_backward_memory=True,
            trace_layer_forward_time=True,
            trace_layer_backward_time=True,
        )
```

> **Note:** TraceML attaches hooks to the modules that exist when `trace_model_instance` is called. It's often best to pass `self` (the `LightningModule` itself) so that it traces the entire forward pass.

---

## 5. Complete Example: MNIST CNN

A fully runnable example.

**Save as `train_lightning.py`:**

```python
import os
import sys

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from traceml.decorators import trace_model_instance
from traceml.integrations.lightning import TraceMLCallback


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class TraceLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MNISTCNN()
        self.loss_fn = nn.CrossEntropyLoss()

        # Enable Deep-Dive per-layer signals
        trace_model_instance(
            self,
            trace_layer_forward_memory=True,
            trace_layer_backward_memory=True,
            trace_layer_forward_time=True,
            trace_layer_backward_time=True,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)

        if batch_idx % 50 == 0:
            print(f"Step {batch_idx}, loss={loss.item():.4f}")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():
    transform = transforms.Compose([transforms.ToTensor()])

    try:
        dataset = datasets.MNIST(
            root="./mnist", train=True, download=True, transform=transform
        )
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = TraceLightningModule()

    trainer = L.Trainer(
        max_steps=300,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        callbacks=[TraceMLCallback()],
    )

    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    main()
```

**Run it:**
```bash
traceml run train_lightning.py
```

---

## 6. Gradient Accumulation

`TraceMLCallback` safely handles gradient accumulation out of the box.

When you set `accumulate_grad_batches=N` in your `Trainer`, Lightning skips the optimizer step for N-1 micro-batches. TraceML treats every micro-batch as a "step" to preserve fine-grained forward/backward times. On the aggregating micro-batches, TraceML emits a 0-duration dummy optimizer event.

This ensures the dashboard's step alignment remains perfectly intact without crashing or showing misaligned phases.

---

## 7. Trainer tips

A few `Trainer` settings interact with TraceML.

| Setting | Recommended value | Why |
|---------|-------------------|-----|
| `enable_progress_bar=False` | Yes | Prevents Lightning's `tqdm` from fighting with the Rich terminal dashboard for output. |
| `enable_model_summary=False`| Optional | You can disable Lightning's textual model summary to keep terminal output cleaner. |
| `logger=False` | Yes (for local debug) | Disables TensorBoard/CSV loggers if you only need TraceML. |

---

## Next steps

- Read the [Quickstart](quickstart.md) for plain PyTorch training loops
- [Open an issue](https://github.com/traceopt-ai/traceml/issues) if you hit a problem not covered here
