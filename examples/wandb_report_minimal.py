# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Minimal example for publishing TraceML's diagnosis into a W&B run.

Requires W&B:

    pip install "traceml-ai[wandb]"

Run with:

    traceml run examples/wandb_report_minimal.py

Use ``--steps`` to change the number of optimizer steps::

    traceml run examples/wandb_report_minimal.py --args --steps 20

Set ``WANDB_MODE=offline`` to try it without a W&B account::

    WANDB_MODE=offline traceml run examples/wandb_report_minimal.py

The loop reads its batches from a ``DataLoader``, so TraceML measures input
wait as well as forward, backward and optimizer time. At the end of the run,
``log_training_report()`` logs the compact diagnosis card to the run as
``traceml/diagnosis_card``: the verdict, coverage, the phase bar and the step
time, input wait and compute averages. It also writes
``traceml/primary/status``, ``traceml/step_time/step_time_ms`` and
``traceml/step_time/input_wait_ms`` to the run summary, where they appear as
runs-table columns.
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import traceml_ai as traceml
from traceml_ai.integrations.wandb import log_training_report

NUM_STEPS = 128
BATCH_SIZE = 64
FEATURES = 512
HIDDEN = 4096


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=NUM_STEPS,
        help="Number of optimizer steps to run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a small traced loop and publish the TraceML diagnosis to W&B."""
    args = parse_args()
    try:
        import wandb
    except ImportError:
        print(
            "This example requires W&B. Install it with:\n"
            '  pip install "traceml-ai[wandb]"'
        )
        sys.exit(0)

    traceml.init()
    wandb.init(project="traceml-examples")

    torch.manual_seed(0)
    # Enough compute per step that untraced loop overhead stays a small
    # share of Step Time, even on CPU.
    model = nn.Sequential(
        nn.Linear(FEATURES, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, FEATURES),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # One pass over the loader yields exactly ``--steps`` batches.
    num_samples = args.steps * BATCH_SIZE
    dataset = TensorDataset(
        torch.randn(num_samples, FEATURES),
        torch.randint(0, FEATURES, (num_samples,)),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    for x, y in loader:
        with traceml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    log_training_report()
    wandb.finish()


if __name__ == "__main__":
    main()
