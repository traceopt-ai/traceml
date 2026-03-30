import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from traceml.decorators import trace_step

SEED = 42
INPUT_DIM = 1024
HIDDEN_DIM = 2048
NUM_CLASSES = 10

BATCH_SIZE = 512
NUM_SAMPLES = 60000
NUM_EPOCHS = 10

# 🚨 TRIGGERS CUDNN_BENCHMARK_MISSING
# (Notice we don't have torch.backends.cudnn.benchmark = True anywhere)


class DummyDataset(Dataset):
    def __init__(self, num_samples, input_dim, num_classes):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
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

    dataset = DummyDataset(
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    model = TinyMLP().to(device)

    # TRIGGERS: HALF_DTYPE_WITHOUT_AUTOCAST
    # Manually casting to half without using torch.autocast
    if torch.cuda.is_available():
        model = model.half()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    total_steps = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        for batch_x, batch_y in loader:
            total_steps += 1

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if torch.cuda.is_available():
                batch_x = batch_x.half()

            with trace_step(model):
                optimizer.zero_grad(set_to_none=True)

                # TRIGGERS: AMP_NOT_USED (if the GPU natively supports it, but .half() is used instead)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if total_steps % 25 == 0:
                print(
                    f"Epoch {epoch} | Step {total_steps} | "
                    f"loss: {loss.item():.4f}"
                )

    print("Done precision diag trigger.")


if __name__ == "__main__":
    main()
