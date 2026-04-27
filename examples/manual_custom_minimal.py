import random
import time
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

import traceml

SEED = 42
INPUT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 10
BATCH_SIZE = 64
STEPS = 250
PAUSE_BETWEEN_STEPS = 0.03


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


class CustomBatchSource:
    """
    Minimal custom batch source to demonstrate manual TraceML wrapping.

    This is intentionally not a torch DataLoader. It mimics a custom input
    pipeline that yields batches directly.
    """

    def __init__(
        self,
        *,
        steps: int,
        batch_size: int,
        input_dim: int,
        num_classes: int,
        sleep_s: float = 0.0,
    ) -> None:
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.sleep_s = float(sleep_s)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(self.steps):
            if self.sleep_s > 0.0:
                time.sleep(self.sleep_s)

            batch_x = torch.randn(self.batch_size, self.input_dim)
            batch_y = torch.randint(0, self.num_classes, (self.batch_size,))
            yield batch_x, batch_y


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Manual mode disables the default automatic patching path.
    traceml.init(mode="manual")

    batch_source = CustomBatchSource(
        steps=STEPS,
        batch_size=BATCH_SIZE,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        sleep_s=0.004,
    )

    # Manual wrappers are the explicit instrumentation path.
    batch_source = traceml.wrap_dataloader_fetch(batch_source)

    model = TinyMLP().to(device)
    model = traceml.wrap_forward(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = traceml.wrap_optimizer(optimizer)

    criterion = nn.CrossEntropyLoss()
    model.train()

    for step, (batch_x, batch_y) in enumerate(batch_source, start=1):

        with traceml.trace_step(model):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            # Backward timing is wrapped explicitly in manual mode.
            traceml.wrap_backward(loss).backward()
            optimizer.step()

        if step % 50 == 0:
            print(f"Step {step} | loss: {loss.item():.4f}")

        time.sleep(PAUSE_BETWEEN_STEPS)

    print("Done.")

    summary = traceml.final_summary(print_text=True)
    print(summary is not None)


if __name__ == "__main__":
    main()
