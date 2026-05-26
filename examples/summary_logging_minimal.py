# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Minimal example for logging TraceML's compact summary.

Run with:

    traceml run examples/summary_logging_minimal.py

At the end of the run, ``tml.summary()`` returns a flat dict designed for
W&B, MLflow, and other experiment trackers.
"""

from __future__ import annotations

import time

import torch
from torch import nn

import traceml_ai as tml


def main() -> None:
    """Run a tiny traced loop and print the compact TraceML summary."""
    tml.init()

    torch.manual_seed(0)
    model = nn.Linear(8, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(32, 8)
    y = torch.randint(0, 2, (32,))

    for _ in range(128):
        with tml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        time.sleep(0.04)

    summary = tml.summary(print_text=True)
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
