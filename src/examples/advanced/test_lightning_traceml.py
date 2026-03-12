import sys

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from traceml.decorators import trace_model_instance
from traceml.utils.lightning import TraceMLCallback


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


class TraceLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MNISTCNN()
        self.loss_fn = nn.CrossEntropyLoss()

        trace_model_instance(
            self,
            trace_layer_forward_memory=True,
            trace_layer_backward_memory=True,
            trace_layer_forward_time=True,
            trace_layer_backward_time=True,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The TraceMLCallback automatically wraps standard operations
        # (forward, backward, optimizer) based on Lightning's lifecycle hooks.

        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)

        if batch_idx % 50 == 0:
            print(f"Step {batch_idx}, loss={loss.item():.4f}")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():
    transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    try:
        dataset = datasets.MNIST(
            root="./mnist", train=True, download=True, transform=transform
        )
    except Exception as e:
        print(f"Failed to download/load dataset: {e}")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    model = TraceLightningModule()

    # Lightning Trainer execution context
    trainer = L.Trainer(
        max_steps=500,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        callbacks=[TraceMLCallback()],
    )

    print("\nStarting Lightning Auto-Instrumented Training...")
    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    main()
