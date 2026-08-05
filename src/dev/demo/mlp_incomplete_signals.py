"""
Incomplete-signals scenario for the Step Time INCOMPLETE_DATA diagnosis.

This keeps the healthy single-process MLP loop shape but calls
``model.forward(features)`` directly instead of ``model(features)``. The
forward auto-timer hooks ``nn.Module.__call__`` only, so forward timing is
never recorded: compute and residual become underivable and TraceML must
report ``INCOMPLETE DATA`` naming ``forward`` - not a confident BALANCED or
RESIDUAL-HEAVY verdict built on a fake zero.

Control twin: ``mlp_ddp_healthy_baseline.py`` (fully instrumented; must not
report INCOMPLETE DATA).

Run on any machine (CPU is fine):

    traceml run python -m dev.demo.mlp_incomplete_signals
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import traceml_ai as traceml

SEED = 42
NUM_SAMPLES = 4096
INPUT_DIM = 512
HIDDEN_DIM = 1024
NUM_CLASSES = 100

BATCH_SIZE = 64
EPOCHS = 2
LR = 1e-4


class BaselineMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    traceml.init(mode="auto")

    features = torch.randn(NUM_SAMPLES, INPUT_DIM)
    labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = BaselineMLP().to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(EPOCHS):
        for batch_features, batch_labels in loader:
            with traceml.trace_step(model):
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                # Deliberate: bypasses nn.Module.__call__, so forward
                # timing is never recorded for this run.
                logits = model.forward(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()


if __name__ == "__main__":
    main()
