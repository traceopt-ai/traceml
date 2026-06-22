"""Smallest TraceML example for a plain PyTorch training loop.

Run with:

    traceml run examples/quickstart.py --mode=summary

The example uses 128 completed steps so the end-of-run Step Time and Step
Memory summaries clear TraceML's 50-step diagnosis threshold.
"""

from __future__ import annotations

import torch
from torch import nn

import traceml_ai as traceml


SEED = 42
NUM_STEPS = 128
BATCH_SIZE = 64
INPUT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 10


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    traceml.init(mode="auto")

    model = TinyMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, NUM_STEPS + 1):
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)

        with traceml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        if step % 32 == 0:
            print(f"step {step:03d}/{NUM_STEPS} | loss: {loss.item():.4f}")

    traceml.summary(print_text=True)
    print("Done.")


if __name__ == "__main__":
    main()
