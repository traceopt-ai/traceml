import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightning as L
from traceml.decorators import trace_model_instance


# Lightning Module


class MNISTLightning(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # Attach TraceML deep instrumentation
        # This gives you per-layer memory and timing in the dashboard.
        trace_model_instance(
            self,
            trace_layer_forward_memory=True,
            trace_layer_backward_memory=True,
            trace_layer_forward_time=True,
            trace_layer_backward_time=True,
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # NOTE: No more 'with trace_step(self)' here
        # The new built-in TraceMLCallback handles the step boundary automatically.
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


# Main Execution


def main():
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.MNIST(root="./mnist", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = MNISTLightning()
    
    # TraceML automatically patches this Trainer to add the TraceMLCallback
    trainer = L.Trainer(
        max_steps=100,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )

    print("Starting Lightning training with Built-in TraceML Support...")
    trainer.fit(model, train_dataloaders=loader)

if __name__ == "__main__":
    main()