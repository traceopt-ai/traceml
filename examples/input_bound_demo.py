import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import traceml

SEED = 42
INPUT_DIM = 1024
HIDDEN_DIM = 2048
NUM_CLASSES = 10

BATCH_SIZE = 128
NUM_SAMPLES = 12000
NUM_EPOCHS = 4

# This is the key knob for the example.
# It makes input loading deliberately slow so TraceML can surface it clearly.
SLEEP_PER_SAMPLE = 0.02


class SlowDataset(Dataset):
    def __init__(self, num_samples, input_dim, num_classes, sleep_s):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sleep_s = sleep_s

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        time.sleep(self.sleep_s)  # simulate slow input pipeline
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
        return x, y


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
    traceml.init(mode="auto")

    dataset = SlowDataset(
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        sleep_s=SLEEP_PER_SAMPLE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # keep stall visible in the main process
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

            if total_steps % 25 == 0:
                print(
                    f"Epoch {epoch} | Step {total_steps} | "
                    f"loss: {loss.item():.4f}"
                )

    print("Done.")


if __name__ == "__main__":
    main()
