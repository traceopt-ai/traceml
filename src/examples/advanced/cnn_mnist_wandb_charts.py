"""
CNN-MNIST with TraceML + W&B: Summary *and* Charts
====================================================

Extends ``cnn_mnist_wandb.py`` with ``log_as_charts=True`` so TraceML
metrics appear in **two places** in the W&B UI:

  - **Overview → Summary panel** (scalar values, queryable across runs)
  - **Charts tab** (visual panels, comparable side-by-side in Reports)

Both destinations are populated from the same ``*_summary_card.json``.
The difference is just one extra keyword argument.

Where to look in W&B
---------------------
After the run completes:

1. **Charts tab** → scroll past ``train/`` section → look for a
   ``traceml/`` section with panels like ``traceml/system/cpu_avg_percent``,
   ``traceml/step_time/worst_avg_step_ms``, etc.
2. **Overview tab → Summary** → all ``traceml/`` keys listed as scalars.
3. **Artifacts tab** (left sidebar) → ``traceml_summary`` artifact →
   open ``traceml_summary.json`` for the full per-rank breakdown.

Prerequisites
-------------
::

    pip install "traceml-ai[wandb]" wandb
    wandb login

Run
---
::

    # Standalone (no live TraceML dashboard):
    TRACEML_SUMMARY_JSON=<path>_summary_card.json \\
        python src/examples/advanced/cnn_mnist_wandb_charts.py

    # Full TraceML live dashboard + W&B:
    TRACEML_WANDB_AUTO=1 \\
        traceml run src/examples/advanced/cnn_mnist_wandb_charts.py
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from traceml.decorators import trace_model_instance, trace_step

# ── Optional W&B import ────────────────────────────────────────────────────
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
# Model (same as cnn_mnist.py)
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
    wandb_run = None
    if HAS_WANDB:
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

            # Live training loss → Charts tab (time series)
            if wandb_run is not None:
                wandb.log({"train/loss": loss.item()}, step=step)

        if step == 500:
            break

    # ── W&B summary export — BOTH summary panel AND charts ───────────────────
    if HAS_WANDB and wandb_run is not None:
        summary_path = os.environ.get("TRACEML_SUMMARY_JSON", "")

        if summary_path:
            success = log_traceml_summary_to_wandb(
                summary_json_path=summary_path,
                run=wandb_run,
                log_as_charts=True,  # ← THIS is the new bit vs cnn_mnist_wandb.py
                #
                # log_as_charts=True does two things:
                #   1. run.summary.update(flat)   → Overview tab  (always done)
                #   2. run.log(flat, commit=True)  → Charts tab   (new)
                #
                # After the run finishes, go to the Charts tab and look for a
                # "traceml/" section with bars/columns for each metric.
            )
            if success:
                print(
                    "[example] TraceML summary uploaded to W&B ✓\n"
                    "          → Charts tab: look for 'traceml/' section\n"
                    "          → Overview tab: Summary panel has all keys\n"
                    "          → Artifacts tab: full JSON in 'traceml_summary'"
                )
            else:
                print("[example] W&B upload skipped (see logs above).")
        else:
            print(
                "[example] Set TRACEML_SUMMARY_JSON=<path>_summary_card.json\n"
                "          to upload the TraceML summary to W&B.\n"
                "          Or use: TRACEML_WANDB_AUTO=1 traceml run <script>"
            )

        wandb.finish()


if __name__ == "__main__":
    main()
