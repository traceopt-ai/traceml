
"""Minimal example for logging TraceML's summary to MLflow.

Requires MLflow:

    pip install mlflow

Run with:

    traceml run examples/mlflow_summary_minimal.py

Use ``--steps`` to change the number of optimizer steps::

    traceml run examples/mlflow_summary_minimal.py --args --steps 20

Use ``--log-final-summary`` to also attach the full report::

    traceml run examples/mlflow_summary_minimal.py --args --log-final-summary

At the end of the run, numeric ``traceml.summary()`` values are logged as
MLflow metrics and diagnosis strings as MLflow tags.
"""

from __future__ import annotations

import argparse
import sys
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
    parser.add_argument(
        "--log-final-summary",
        action="store_true",
        help="Also attach the full final summary JSON as an MLflow artifact.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a tiny traced loop and log the TraceML summary to MLflow."""
    args = parse_args()
    try:
        import mlflow
    except ImportError:
        print(
            "This example requires MLflow. Install it with:\n"
            "  pip install mlflow"
        )
        sys.exit(0)

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

    metrics = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    tags = {
        key.replace("/", "."): value
        for key, value in summary.items()
        if isinstance(value, str)
    }

    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)

        if args.log_final_summary:
            final = traceml.final_summary()
            if final is not None:
                mlflow.log_dict(final, "traceml/final_summary.json")


if __name__ == "__main__":
    main()
