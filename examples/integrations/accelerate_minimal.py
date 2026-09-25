"""Minimal Hugging Face Accelerate loop wrapped with traceml.trace_step.

Run with:

    traceml run examples/integrations/accelerate_minimal.py

Use ``--epochs`` or ``--steps`` to change the run length::

    traceml run examples/integrations/accelerate_minimal.py --args --steps 20
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

import traceml_ai as traceml

SEED = 42
INPUT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 10
NUM_SAMPLES = 8192
BATCH_SIZE = 64
EPOCHS = 4


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
    length = parser.add_mutually_exclusive_group()
    length.add_argument(
        "--epochs",
        type=positive_int,
        default=EPOCHS,
        help="Number of full epochs to run.",
    )
    length.add_argument(
        "--steps",
        type=positive_int,
        default=None,
        help=(
            "Number of optimizer steps per rank. Replaces --epochs; by "
            "default, run --epochs full epochs."
        ),
    )
    return parser.parse_args()


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def main():
    args = parse_args()
    torch.manual_seed(SEED)

    x = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TinyMLP()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Build the Accelerator before calling traceml.init() so TraceML's
    # instrumentation installs after the distributed/device context already
    # exists.
    accelerator = Accelerator()
    traceml.init(mode="auto")

    # accelerator.prepare() moves the model/data to the right device and,
    # under a distributed launch, wraps the model (e.g. in DDP).
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    # Pass the unwrapped model to trace_step, exactly like model.module for
    # DDP and base_model for FSDP: TraceML keys instrumentation off id(model)
    # and reads device placement from model.parameters(), so it needs the
    # real underlying module, not a distributed wrapper. A no-op here
    # (single process); strips the wrapper under --nproc-per-node.
    traced_model = accelerator.unwrap_model(model)

    model.train()
    global_step = 0
    epochs = args.epochs
    if args.steps is not None:
        epochs = (args.steps + len(dataloader) - 1) // len(dataloader)

    for epoch in range(epochs):
        running_loss = torch.zeros((), device=accelerator.device)

        for batch_x, batch_y in dataloader:
            global_step += 1

            with traceml.trace_step(traced_model):
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                accelerator.backward(loss)
                optimizer.step()

                # Stay on-device inside trace_step: accumulating a raw
                # tensor avoids a per-step host sync. Only convert to a
                # Python float in the print block below, which runs
                # outside trace_step and on a fixed cadence.
                running_loss += loss.detach()

            if global_step % 25 == 0:
                accelerator.print(
                    f"Epoch {epoch + 1} | Step {global_step} | "
                    f"loss: {float(running_loss) / 25:.4f}"
                )
                running_loss.zero_()

            if args.steps is not None and global_step >= args.steps:
                break

    accelerator.print("Done.")


if __name__ == "__main__":
    main()
