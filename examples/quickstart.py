"""Smallest TraceML example for a plain PyTorch training loop.

Run with:

    traceml run examples/quickstart.py

The example uses 128 completed steps so the end-of-run Step Time and Step
Memory summaries clear TraceML's 50-step diagnosis threshold.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


def build_dataloader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Return one deterministic, CPU-friendly epoch of synthetic batches.

    A real PyTorch DataLoader is intentional: TraceML's automatic mode records
    its fetch timing, just as it would for a normal training script. Keep this
    loader fast; the diagnosis demos are the place to inject artificial delay.
    """
    generator = torch.Generator().manual_seed(SEED)
    dataset = TensorDataset(
        torch.randn(NUM_STEPS * BATCH_SIZE, INPUT_DIM, generator=generator),
        torch.randint(
            NUM_CLASSES,
            (NUM_STEPS * BATCH_SIZE,),
            generator=generator,
        ),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    traceml.init(mode="auto")

    model = TinyMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    dataloader = build_dataloader()

    model.train()
    for step, (x, y) in enumerate(dataloader, start=1):
        x = x.to(device)
        y = y.to(device)

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
