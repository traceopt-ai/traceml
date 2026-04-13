import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import traceml

SEED = 42
INPUT_DIM = 1024
HIDDEN_DIM = 2048
NUM_CLASSES = 10
BATCH_SIZE = 256
NUM_SAMPLES = 50000
NUM_EPOCHS = 2
PAUSE_BETWEEN_STEPS = 0.05  # keeps the run visible a bit longer


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Create a simple in-memory dataset
    x = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(x, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # keep minimal and portable
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = TinyMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    total_steps = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        for batch_x, batch_y in loader:
            total_steps += 1

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with traceml.trace_step(model):
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if total_steps % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {total_steps} | loss: {loss.item():.4f}"
                )

            time.sleep(PAUSE_BETWEEN_STEPS)

    print("Done.")

    summary = traceml.final_summary(print_text=True)
    print(summary is not None)


if __name__ == "__main__":
    main()
