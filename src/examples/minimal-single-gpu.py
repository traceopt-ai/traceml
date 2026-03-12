import time

import torch
import torch.nn as nn
import torch.optim as optim

from traceml.decorators import trace_step

# Chosen so the run lasts long enough to see TraceML clearly.
# Increase NUM_STEPS or PAUSE_BETWEEN_STEPS if it still feels too fast.
SEED = 42
INPUT_DIM = 1024
HIDDEN_DIM = 2048
NUM_CLASSES = 10
BATCH_SIZE = 256
NUM_STEPS = 600
PAUSE_BETWEEN_STEPS = 0.15  # seconds


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

    model = TinyMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for step in range(1, NUM_STEPS + 1):
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
        y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)

        with trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:4d}/{NUM_STEPS} | loss: {loss.item():.4f}")

        # Keeps the run visible long enough for users to watch TraceML update.
        # This pause is outside trace_step, so it does not pollute TraceML timings.
        time.sleep(PAUSE_BETWEEN_STEPS)

    print("Done.")


if __name__ == "__main__":
    main()
