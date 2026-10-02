# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Minimal example for logging TraceML's compact summary.

Run with:

    traceml run examples/summary_logging_minimal.py

Use ``--steps`` to change the number of optimizer steps::

    traceml run examples/summary_logging_minimal.py --args --steps 20

At the end of the run, ``traceml.summary()`` returns a flat dict designed for
W&B, MLflow, and other experiment trackers.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

import traceml_ai as traceml

NUM_STEPS = 128


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
    """Run a tiny traced loop and print the compact TraceML summary."""
    args = parse_args()
    traceml.init()

    torch.manual_seed(0)
    model = nn.Linear(8, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(32, 8)
    y = torch.randint(0, 2, (32,))

    for _ in range(args.steps):
        with traceml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        time.sleep(0.04)

    summary = traceml.summary(print_text=True)
    if summary is None:
        return

    print("\nCompact TraceML summary:")
    for key, value in sorted(summary.items()):
        print(f"{key}: {value}")

    # W&B:
    # import wandb
    # wandb.init(project="my-project")
    # wandb.log(summary)

    # MLflow:
    # import mlflow
    # numeric = {
    #     k: v for k, v in summary.items()
    #     if isinstance(v, (int, float)) and not isinstance(v, bool)
    # }
    # tags = {k.replace("/", "."): v for k, v in summary.items() if isinstance(v, str)}
    # mlflow.log_metrics(numeric)
    # mlflow.set_tags(tags)


if __name__ == "__main__":
    main()
