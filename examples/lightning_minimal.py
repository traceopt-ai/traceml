import argparse
import time

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from traceml_ai.integrations import lightning as traceml_lightning

SEED = 42
MODEL_INPUT_DIM = 128
TRANSFER_INPUT_DIM = 131072
HIDDEN_DIM = 256
NUM_CLASSES = 10
NUM_SAMPLES = 512
BATCH_SIZE = 64
MAX_STEPS = 200


class SyntheticClassificationDataset(Dataset):
    def __init__(self, num_samples: int):
        # Transfer a wider CPU batch so Lightning H2D timing is visible, while
        # the model below only consumes MODEL_INPUT_DIM features for compute.
        self.x = torch.randn(num_samples, TRANSFER_INPUT_DIM)
        self.y = torch.randint(0, NUM_CLASSES, (num_samples,))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TinyLightningModel(L.LightningModule):
    def __init__(self, *, delay_rank: int = -1, delay_ms: float = 0.0):
        super().__init__()
        self.delay_rank = int(delay_rank)
        self.delay_ms = float(delay_ms)
        self.net = nn.Sequential(
            nn.Linear(MODEL_INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x[..., :MODEL_INPUT_DIM].contiguous()
        return self.net(x)

    def training_step(self, batch, batch_idx):
        if self.delay_ms > 0 and int(self.global_rank) == self.delay_rank:
            time.sleep(self.delay_ms / 1000.0)

        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        if self.global_step % 50 == 0:
            print(f"Step {self.global_step} | loss={loss.item():.4f}")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--delay-rank", type=int, default=-1)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    traceml_lightning.init()

    dataset = SyntheticClassificationDataset(NUM_SAMPLES)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = TinyLightningModel(
        delay_rank=args.delay_rank,
        delay_ms=args.delay_ms,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    strategy = "ddp" if int(args.devices) > 1 else "auto"

    trainer = L.Trainer(
        max_steps=int(args.max_steps),
        accelerator=accelerator,
        devices=int(args.devices),
        strategy=strategy,
        enable_progress_bar=False,
        callbacks=[traceml_lightning.TraceMLCallback()],
        logger=False,
    )

    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    main()
