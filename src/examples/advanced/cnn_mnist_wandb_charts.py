"""
CNN-MNIST with TraceML + W&B: Summary *and* Charts
====================================================

Shows the **correct** way to push TraceML metrics to W&B when using
``traceml run``.

Why ``TRACEML_WANDB_AUTO=1`` does not work for charts
-------------------------------------------------------
``traceml run`` launches two **separate processes**:

* **Executor** (your script, this process) — ``wandb.run`` is active here.
* **Aggregator** (separate subprocess) — collects telemetry via TCP.
  ``wandb.run`` is *always* ``None`` in the aggregator process; setting
  ``TRACEML_WANDB_AUTO=1`` cannot reach it.

The fix: call ``upload_traceml_summary()`` from inside your script (the
executor process), BEFORE ``wandb.finish()``.  It reads the session DB the
aggregator has been writing, generates the summary cards, and uploads them
while the W&B run is still active.

Where to look in W&B after the run
------------------------------------
1. **Charts tab** — a ``traceml/`` section with panels for each metric.
2. **Overview → Summary** — the same keys as queryable scalars.
3. **Artifacts tab** — ``traceml_summary`` artifact with full JSON.

Prerequisites
-------------
::

    pip install "traceml-ai[wandb]" wandb
    wandb login

Run
---
::

    traceml run src/examples/advanced/cnn_mnist_wandb_charts.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from traceml.decorators import trace_model_instance, trace_step
from traceml.integrations.wandb import upload_traceml_summary

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # ── W&B init ─────────────────────────────────────────────────────────────
    # Must be called BEFORE training so the run is active when we upload.
    wandb_run = wandb.init(
        project="traceml-examples",
        name="cnn-mnist-traceml-charts",
        config={
            "architecture": "MNISTCNN",
            "dataset": "MNIST",
            "batch_size": 64,
            "optimizer": "Adam",
            "lr": 1e-3,
        },
    )

    # ── Dataset & model ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [transforms.RandomRotation(10), transforms.ToTensor()]
    )
    dataset = datasets.MNIST(
        root="./mnist", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = MNISTCNN().to(device)
    trace_model_instance(
        model,
        trace_layer_forward_memory=True,
        trace_layer_backward_memory=True,
        trace_layer_forward_time=True,
        trace_layer_backward_time=True,
    )

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    print("Starting training...")
    for step, (xb, yb) in enumerate(loader):
        with trace_step(model):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

        if step % 50 == 0:
            print(f"Step {step}, loss={loss.item():.4f}")
            # Live loss → Charts tab (time series, one point per step)
            wandb.log({"train/loss": loss.item()}, step=step)

        if step == 500:
            break

    # ── Upload TraceML summary to W&B ─────────────────────────────────────────
    # Call BEFORE wandb.finish() while the run is still active.
    #
    # upload_traceml_summary():
    #   - Reads the session DB written by the TraceML aggregator
    #   - Generates summary cards in this process (where wandb.run is alive)
    #   - Updates run.summary  → visible in Overview tab
    #   - Calls wandb.log()    → visible in Charts tab  (log_as_charts=True)
    #   - Uploads full JSON as a W&B Artifact
    success = upload_traceml_summary(
        run=wandb_run,
        log_as_charts=True,
    )

    if success:
        print(
            "[example] TraceML summary uploaded to W&B ✓\n"
            "          → Charts tab: look for 'traceml/' metric section\n"
            "          → Overview tab: Summary panel lists all keys\n"
            "          → Artifacts: 'traceml_summary' contains full JSON"
        )
    else:
        print(
            "[example] TraceML W&B upload skipped or failed — see logs above."
        )

    wandb.finish()


if __name__ == "__main__":
    main()
