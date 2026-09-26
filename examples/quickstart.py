"""Smallest TraceML example for a plain PyTorch training loop.

Run with:

    traceml run examples/quickstart.py

Use ``--steps`` to change the number of optimizer steps::

    traceml run examples/quickstart.py --args --steps 20

The example defaults to 128 completed steps so the end-of-run summaries
clear every diagnosis gate. Step Memory needs at least 50 completed steps
before it diagnoses at all. Step Time needs 2 steps for warning-only
bottleneck diagnoses and 20 steps for critical ones.
"""

from __future__ import annotations

import argparse

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


def build_dataloader(
    num_steps: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Return one deterministic, CPU-friendly epoch of synthetic batches.

    A real PyTorch DataLoader is intentional: TraceML's automatic mode records
    its fetch timing, just as it would for a normal training script. Keep this
    loader fast; the diagnosis demos are the place to inject artificial delay.
    """
    generator = torch.Generator().manual_seed(SEED)
    dataset = TensorDataset(
        torch.randn(num_steps * BATCH_SIZE, INPUT_DIM, generator=generator),
        torch.randint(
            NUM_CLASSES,
            (num_steps * BATCH_SIZE,),
            generator=generator,
        ),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    traceml.init(mode="auto")

    model = TinyMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    dataloader = build_dataloader(args.steps)

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
            print(f"step {step:03d}/{args.steps} | loss: {loss.item():.4f}")

    traceml.summary(print_text=True)
    print("Done.")


if __name__ == "__main__":
    main()
