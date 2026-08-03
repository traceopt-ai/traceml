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

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for batch_x, batch_y in dataloader:
            global_step += 1

            with traceml.trace_step(traced_model):
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                accelerator.backward(loss)
                optimizer.step()

                running_loss += float(loss.detach())

            if global_step % 25 == 0:
                accelerator.print(
                    f"Epoch {epoch + 1} | Step {global_step} | "
                    f"loss: {running_loss / 25:.4f}"
                )
                running_loss = 0.0

    accelerator.print("Done.")


if __name__ == "__main__":
    main()
