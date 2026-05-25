# PyTorch Lightning

Use TraceML AI with PyTorch Lightning to find training bottlenecks without changing your training loop.

`TraceMLCallback` adds step-aware diagnosis so you can quickly see whether a run is input-bound, compute-bound, straggler-heavy, wait-heavy, or showing memory drift.

> Start with the [Quickstart](../quickstart.md) if you have not used TraceML AI yet.

---

## Install

Install TraceML AI with Lightning support:

```bash
pip install "traceml-ai[lightning]"
```

---

## Basic usage

Add `TraceMLCallback` to your Lightning `Trainer`. Everything else stays the same.

```python
import lightning as L
from traceml_ai.integrations.lightning import TraceMLCallback

model = MyLightningModule()

trainer = L.Trainer(
    max_steps=500,
    accelerator="auto",
    devices=1,
    enable_progress_bar=False,
    callbacks=[TraceMLCallback()],
)

trainer.fit(model, train_dataloaders=loader)
```

Run with:

```bash
traceml run train.py
```

Or open the local UI:

```bash
traceml run train.py --mode=dashboard
```

Dashboard mode is intended for single-node runs, including single-node
multi-GPU.

---

## What TraceML AI will show

In Lightning runs, TraceML AI helps you spot:

- input-bound training
- compute-bound steps
- wait-heavy behavior
- rank imbalance and stragglers
- memory creep over time

You keep the normal Lightning workflow. TraceML AI adds diagnosis around the training step.

---

## How it works

`TraceMLCallback` hooks into Lightning’s training lifecycle automatically.

That means you do not need to wrap your code with `tml.trace_step(...)`
manually in Lightning.

---

## Use with Lightning loggers

TraceML AI works alongside Lightning loggers such as:

- W&B
- TensorBoard
- CSVLogger

For the cleanest terminal experience during diagnosis runs, it helps to use:

```python
enable_progress_bar=False
```

You do not need to replace your existing logger stack to use TraceML AI.

---

## Optional: local UI

If you want a richer browser-based view, run:

```bash
traceml run train.py --mode=dashboard
```

Dashboard mode is intended for single-node runs.

The local UI is useful when you want:

- a richer run review experience
- easier local comparison
- less terminal clutter

---

## Trainer tips

These settings usually give the cleanest experience with TraceML AI:

| Setting | Recommended value | Why |
|---|---|---|
| `enable_progress_bar=False` | Yes | Prevents Lightning progress output from fighting with the TraceML AI CLI |
| `enable_model_summary=False` | Optional | Keeps terminal output cleaner |
| `logger=False` | Optional | Useful for local diagnosis runs if you want minimal output |

---

## Full example

Save as `train_lightning.py`:

```python
import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from traceml_ai.integrations.lightning import TraceMLCallback

SEED = 42
INPUT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 10
NUM_SAMPLES = 4096
BATCH_SIZE = 64
MAX_STEPS = 200


class SyntheticClassificationDataset(Dataset):
    def __init__(self, num_samples: int):
        self.x = torch.randn(num_samples, INPUT_DIM)
        self.y = torch.randint(0, NUM_CLASSES, (num_samples,))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TinyLightningModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        if batch_idx % 50 == 0:
            print(f"Step {batch_idx} | loss={loss.item():.4f}")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    torch.manual_seed(SEED)

    dataset = SyntheticClassificationDataset(NUM_SAMPLES)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = TinyLightningModel()

    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        callbacks=[TraceMLCallback()],
        logger=False,
    )

    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    main()
```

Run with:

```bash
traceml run train_lightning.py
```

---

## Gradient accumulation

`TraceMLCallback` supports gradient accumulation.

When Lightning uses `accumulate_grad_batches=N`, TraceML AI still preserves step alignment so the dashboard and summaries stay consistent.

---

## Troubleshooting

### Terminal output overlaps with TraceML AI

Set:

```python
enable_progress_bar=False
```

This gives the TraceML AI CLI cleaner terminal control.

### I still want W&B or TensorBoard

That is fine. TraceML AI is designed to work alongside them.

If terminal output gets noisy, use:

```bash
traceml run train.py --mode=dashboard
```

Dashboard mode is intended for single-node runs. For multi-node runs, use the
default final summary path.

### I want a baseline without TraceML AI

Run:

```bash
traceml run train_lightning.py --disable-traceml
```

This launches your script natively through `torchrun` without TraceML AI telemetry.

---

## Next steps

- Read the [Quickstart](../quickstart.md) for plain PyTorch loops
- Read [huggingface.md](huggingface.md) for Hugging Face Trainer integration
- Open an issue if you hit a problem: https://github.com/traceopt-ai/traceml/issues
