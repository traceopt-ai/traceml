import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class DummyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        print(f"  training_step: batch_idx={batch_idx}")
        return self(batch[0]).sum()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)


class TestCallback(L.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        print(f"on_train_batch_start: {batch_idx}")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        print(">>> on_before_optimizer_step")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        print(f"on_train_batch_end: {batch_idx}\n")


ds = TensorDataset(torch.randn(8, 10))
dl = DataLoader(ds, batch_size=2)  # 4 batches total

trainer = L.Trainer(
    max_epochs=1,
    accumulate_grad_batches=2,
    callbacks=[TestCallback()],
    enable_progress_bar=False,
    logger=False,
)
trainer.fit(DummyModel(), dl)
