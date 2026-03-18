"""
CNN-MNIST with TraceML + Weights & Biases (W&B) Integration
============================================================

This example shows how to combine TraceML step-level profiling with W&B
experiment tracking.

At the end of training TraceML writes an end-of-run summary JSON file.
``log_traceml_summary_to_wandb`` uploads that JSON to your W&B run as:

  1. **Flat scalar metrics** logged to ``run.summary``  (queryable across runs)
  2. **A W&B Artifact** named ``traceml_summary`` containing the full JSON

Prerequisites
-------------
::

    pip install "traceml-ai[wandb]" wandb
    wandb login   # one-time API-key setup

Run
---
::

    # Plain Python (TraceML passive — only the W&B upload is active):
    python src/examples/advanced/cnn_mnist_wandb.py

    # Full TraceML live dashboard *and* W&B upload:
    traceml run src/examples/advanced/cnn_mnist_wandb.py

When ``traceml run`` is used, TraceML writes the summary JSON automatically
after the run finishes.  This example then loads that JSON and uploads it.

For the automatic path (no manual ``log_traceml_summary_to_wandb`` call),
set the environment variable::

    TRACEML_WANDB_AUTO=1 traceml run src/examples/advanced/cnn_mnist_wandb.py

and make sure ``wandb.init()`` is called before training starts.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from traceml.decorators import trace_model_instance, trace_step

# ── Optional W&B import (graceful if not installed) ────────────────────────
try:
    import wandb

    from traceml.integrations.wandb import log_traceml_summary_to_wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print(
        "[example] wandb not installed — skipping W&B upload.\n"
        "          Install with: pip install 'traceml-ai[wandb]'"
    )


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
# TraceML wrappers (unchanged from cnn_mnist.py)
# ---------------------------------------------------------------------------


def forward_step(model, x):
    return model(x)


def backward_step(loss):
    loss.backward()


def optimizer_step(opt):
    opt.step()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # ── W&B init ────────────────────────────────────────────────────────────
    # Call wandb.init() *before* training so the run is active when
    # generate_summary() fires at shutdown (needed for TRACEML_WANDB_AUTO=1).
    wandb_run = None
    if HAS_WANDB:
        wandb_run = wandb.init(
            project="traceml-examples",
            name="cnn-mnist-traceml",
            config={
                "architecture": "MNISTCNN",
                "dataset": "MNIST",
                "epochs": 1,
                "batch_size": 64,
                "optimizer": "Adam",
                "lr": 1e-3,
            },
        )

    # ── Dataset & model ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.MNIST(
        root="./mnist", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = MNISTCNN().to(device)

    # ⚠️ Deep instrumentation (use only for detailed debugging / profiling)
    # Enables per-layer memory + timing hooks.
    trace_model_instance(
        model,
        trace_layer_forward_memory=True,
        trace_layer_backward_memory=True,
        trace_layer_forward_time=True,
        trace_layer_backward_time=True,
    )

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # ── Training loop ────────────────────────────────────────────────────────
    print("Starting training...")
    for step, (xb, yb) in enumerate(loader):
        with trace_step(model):
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)

            out = forward_step(model, xb)
            loss = loss_fn(out, yb)

            backward_step(loss)
            optimizer_step(opt)

        if step % 50 == 0:
            print(f"Step {step}, loss={loss.item():.4f}")

            # Also log live training loss to W&B (optional)
            if wandb_run is not None:
                wandb.log({"train/loss": loss.item()}, step=step)

        if step == 500:
            break

    # ── W&B summary export ───────────────────────────────────────────────────
    # TraceML writes the summary JSON when the aggregator shuts down.
    # Since we're running WITHOUT `traceml run` here, we call generate_summary
    # manually to demonstrate the API.  When using `traceml run`, this happens
    # automatically — just set TRACEML_WANDB_AUTO=1.
    if HAS_WANDB and wandb_run is not None:
        # Locate the summary JSON.
        # When launched via `traceml run`, the path is logged to stdout as:
        #   [TraceML] summary written to <path>_summary_card.json
        # For this standalone demo we check a common default location.
        summary_path = os.environ.get(
            "TRACEML_SUMMARY_JSON",
            "",  # Set this to the actual path in your workflow
        )

        if summary_path:
            success = log_traceml_summary_to_wandb(
                summary_json_path=summary_path,
                run=wandb_run,
            )
            if success:
                print("[example] TraceML summary uploaded to W&B ✓")
            else:
                print("[example] W&B upload skipped (see logs above).")
        else:
            print(
                "[example] Set TRACEML_SUMMARY_JSON=<path>_summary_card.json "
                "to upload the TraceML summary to W&B.\n"
                "          When using `traceml run`, set TRACEML_WANDB_AUTO=1 "
                "instead."
            )

        wandb.finish()


if __name__ == "__main__":
    main()
