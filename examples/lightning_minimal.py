import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from traceml.integrations.lightning import TraceMLCallback

SEED = 42
INPUT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 10
NUM_SAMPLES = 4096
BATCH_SIZE = 64
MAX_STEPS = 200


class SyntheticClassificationDataset(Dataset):
    def __init__(self, num_samples: int):
        self.x = torch.randn(num_samples, INPUT_DIM)
        self.y = torch.randint(0, NUM_CLASSES, (num_samples,))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TinyLightningModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        if batch_idx % 50 == 0:
            print(f"Step {batch_idx} | loss={loss.item():.4f}")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    torch.manual_seed(SEED)

    dataset = SyntheticClassificationDataset(NUM_SAMPLES)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = TinyLightningModel()

    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        callbacks=[TraceMLCallback()],
        logger=False,
    )

    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    main()
