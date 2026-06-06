import argparse
import os
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


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _infer_num_nodes() -> int:
    traceml_nnodes = _env_int("TRACEML_NNODES")
    if traceml_nnodes is not None:
        return max(1, traceml_nnodes)

    world_size = _env_int("WORLD_SIZE", 1) or 1
    local_world_size = _env_int("LOCAL_WORLD_SIZE", 1) or 1
    return max(1, world_size // max(1, local_world_size))


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
        self._debug_printed = False
        self._delay_printed = False
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
        if not self._debug_printed:
            print(
                "[TraceML Lightning Example] "
                f"training_step rank={self.global_rank} "
                f"delay_rank={self.delay_rank} delay_ms={self.delay_ms}",
                flush=True,
            )
            self._debug_printed = True

        if self.delay_ms > 0 and int(self.global_rank) == self.delay_rank:
            if not self._delay_printed:
                print(
                    "[TraceML Lightning Example] "
                    f"applying delay rank={self.global_rank} "
                    f"batch_idx={batch_idx} delay_ms={self.delay_ms}",
                    flush=True,
                )
                self._delay_printed = True
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
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--delay-rank", type=int, default=-1)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    args = parser.parse_args()

    devices = max(1, int(args.devices))
    num_nodes = max(
        1,
        (
            int(args.num_nodes)
            if args.num_nodes is not None
            else _infer_num_nodes()
        ),
    )
    world_size = devices * num_nodes

    print(
        "[TraceML Lightning Example] "
        f"parsed args devices={devices} num_nodes={num_nodes} "
        f"max_steps={args.max_steps} "
        f"delay_rank={args.delay_rank} delay_ms={args.delay_ms}",
        flush=True,
    )

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
    strategy = "ddp" if world_size > 1 else "auto"
    print(
        "[TraceML Lightning Example] "
        f"trainer accelerator={accelerator} devices={devices} "
        f"num_nodes={num_nodes} strategy={strategy} "
        f"env_rank={os.environ.get('RANK', 'unset')} "
        f"env_world_size={os.environ.get('WORLD_SIZE', 'unset')} "
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )

    trainer = L.Trainer(
        max_steps=int(args.max_steps),
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        enable_progress_bar=False,
        callbacks=[traceml_lightning.TraceMLCallback()],
        logger=False,
    )

    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    main()
