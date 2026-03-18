"""
CNN-MNIST with TraceML + Weights & Biases (W&B)
================================================

Shows how to combine TraceML step-level profiling with W&B experiment
tracking.  At the end of training, ``upload_traceml_summary()`` reads the
session DB written by TraceML, generates the end-of-run summary, and logs:

  1. **Flat scalar metrics** to ``run.summary`` (queryable in runs table /
     Overview tab)
  2. **A W&B Artifact** ``traceml_summary`` with the full JSON detail

For charts panels in the W&B Charts tab, see ``cnn_mnist_wandb_charts.py``
which passes ``log_as_charts=True``.

Prerequisites
-------------
::

    pip install "traceml-ai[wandb]" wandb
    wandb login

Run
---
::

    traceml run src/examples/advanced/cnn_mnist_wandb.py

Why ``upload_traceml_summary()`` (not ``TRACEML_WANDB_AUTO=1``)
----------------------------------------------------------------
``traceml run`` launches the **aggregator as a separate subprocess**;
``wandb.run`` is always ``None`` there.  ``upload_traceml_summary()`` runs
inside the executor process (your script) where ``wandb.run`` is alive, so
it can actually reach W&B.  Call it BEFORE ``wandb.finish()``.
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
    # W&B init
    # Call BEFORE training so the run is active when we upload.
    wandb_run = wandb.init(
        project="traceml-examples",
        name="cnn-mnist-traceml",
        config={
            "architecture": "MNISTCNN",
            "dataset": "MNIST",
            "batch_size": 64,
            "optimizer": "Adam",
            "lr": 1e-3,
        },
    )

    # Dataset & model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [transforms.RandomRotation(10), transforms.ToTensor()]
    )
    dataset = datasets.MNIST(
        root="./mnist", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = MNISTCNN().to(device)

    # Per-layer memory + timing hooks (remove for speed in production)
    trace_model_instance(
        model,
        trace_layer_forward_memory=True,
        trace_layer_backward_memory=True,
        trace_layer_forward_time=True,
        trace_layer_backward_time=True,
    )

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
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
            # Live training loss → W&B Charts tab (time series)
            wandb.log({"train/loss": loss.item()}, step=step)

        if step == 500:
            break

    # Upload TraceML summary to W&B
    # Must be called BEFORE wandb.finish() while the run is still active.
    # Reads the TraceML session DB and uploads metrics + artifact in one call.
    success = upload_traceml_summary(run=wandb_run)

    if success:
        print(
            "[example] TraceML summary uploaded to W&B ✓\n"
            "          → Overview tab → Summary: all traceml/ scalar metrics\n"
            "          → Artifacts tab: 'traceml_summary' JSON artifact"
        )
    else:
        print(
            "[example] TraceML W&B upload skipped or failed (see logs above).\n"
            "          Are you running via `traceml run`?"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
