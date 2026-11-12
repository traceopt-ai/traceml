# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Minimal example for publishing TraceML's diagnosis into an MLflow run.

Requires MLflow:

    pip install "traceml-ai[mlflow]" mlflow

The full ``mlflow`` package provides the local SQLite store this example
writes to. With only ``mlflow-skinny``, pass ``--tracking-uri`` with the
address of a tracking server.

Run with:

    traceml run examples/mlflow_report_minimal.py

Use ``--steps`` to change the number of optimizer steps::

    traceml run examples/mlflow_report_minimal.py --args --steps 20

The run is stored in ``mlflow.db`` in the current directory. View it with::

    mlflow ui --backend-store-uri sqlite:///mlflow.db

The loop reads its batches from a ``DataLoader``, so TraceML measures input
wait as well as forward, backward and optimizer time. At the end of the run,
``log_training_report()`` logs the compact diagnosis card to the run as the
artifact ``traceml/training-report.html``: the verdict, coverage, the phase
bar and the step time, input wait and compute averages. It also logs
``traceml/step_time/step_time_ms`` and ``traceml/step_time/input_wait_ms``
as metrics and ``traceml/primary/status`` as a tag, so runs can be compared
by them.
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import traceml_ai as traceml
from traceml_ai.integrations.mlflow import log_training_report

NUM_STEPS = 128
TRACKING_URI = "sqlite:///mlflow.db"
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
    parser.add_argument(
        "--tracking-uri",
        default=TRACKING_URI,
        help="MLflow tracking URI; the default is a local SQLite file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a small traced loop and publish the TraceML diagnosis to MLflow."""
    args = parse_args()
    try:
        import mlflow
    except ImportError:
        print(
            "This example requires MLflow. Install it with:\n"
            '  pip install "traceml-ai[mlflow]" mlflow'
        )
        sys.exit(0)

    traceml.init()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("traceml-examples")

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

    with mlflow.start_run():
        for x, y in loader:
            with traceml.trace_step(model):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        log_training_report()


if __name__ == "__main__":
    main()
