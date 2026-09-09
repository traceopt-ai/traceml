"""Find a data-loading bottleneck in a PyTorch Lightning run.

Trains ResNet-18 on the 320px Imagenette train split under one ``Trainer``.
The ``--profile`` flag changes the DataLoader settings and nothing else, so
two runs measure the loader change alone. TraceML reports whether each run
waited on input or on compute; ``traceml compare`` shows where the
difference occurred.

Launch through ``traceml run`` (a bare ``python`` run trains untraced):

    traceml run --mode summary --logs-dir logs --run-name lightning_baseline \\
        lightning_dataloading_bottleneck.py \\
        --args --profile baseline --max-steps 300 --batch-size 64
    traceml run --mode summary --logs-dir logs --run-name lightning_optimized \\
        lightning_dataloading_bottleneck.py \\
        --args --profile optimized --max-steps 300 --batch-size 64
    traceml compare logs/lightning_baseline/final_summary.json \\
        logs/lightning_optimized/final_summary.json

CPU-only check without the dataset (what the notebook smoke job runs):

    traceml run --mode summary --logs-dir logs --run-name lightning_smoke \\
        lightning_dataloading_bottleneck.py \\
        --args --smoke --max-steps 8 --batch-size 4

The 320px Imagenette archive (326 MB) is downloaded into ``--data-dir`` on
first use through ``torchvision.datasets.Imagenette``.
"""

from __future__ import annotations

import argparse
import os
import time

import lightning as L
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from traceml_ai.integrations import lightning as traceml_lightning

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SEED = 42
NUM_CLASSES = 10
IMAGENETTE_SIZE = "320px"
SMOKE_SAMPLES = 32
SMOKE_DELAY_S = 0.05


class SlowSyntheticImages(Dataset):
    """Small CPU-only dataset with deliberate fetch latency for smoke mode."""

    classes = ("zero", "one")

    def __init__(self, samples=SMOKE_SAMPLES, delay_s=SMOKE_DELAY_S):
        self.images = torch.zeros(samples, 3, 32, 32)
        self.labels = torch.arange(samples) % len(self.classes)
        self.delay_s = delay_s

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        time.sleep(self.delay_s)
        return self.images[index], self.labels[index]


def loader_settings(
    profile,
    smoke=False,
    num_workers=None,
    persistent_workers=None,
):
    """
    Resolve the DataLoader knobs for one run.

    The profile is the one experimental change: ``baseline`` decodes every
    batch in the training process, ``optimized`` lets up to four workers
    decode ahead, pins batches and keeps the workers alive between epochs.
    The two explicit arguments override the profile for extra runs. Smoke
    mode always runs single-process on CPU.
    """
    optimized = profile == "optimized"
    if smoke:
        workers = 0
    elif num_workers is not None:
        workers = int(num_workers)
    else:
        # Match workers to CPU cores; more workers than cores thrash
        # instead of overlapping (free Colab has two).
        workers = min(4, os.cpu_count() or 2) if optimized else 0
    if persistent_workers is None:
        persistent = optimized
    else:
        persistent = bool(persistent_workers)
    return {
        "num_workers": workers,
        "pin_memory": optimized and not smoke,
        "persistent_workers": persistent and workers > 0,
    }


class ImagenetteDataModule(L.LightningDataModule):
    """Imagenette train loader whose only knobs are the DataLoader settings."""

    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        smoke=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None

    @property
    def num_classes(self):
        if self.hparams.smoke:
            return len(SlowSyntheticImages.classes)
        return NUM_CLASSES

    def prepare_data(self):
        # Download only; Lightning calls this once per node before setup().
        if not self.hparams.smoke:
            torchvision.datasets.Imagenette(
                self.hparams.data_dir,
                split="train",
                size=IMAGENETTE_SIZE,
                download=True,
            )

    def setup(self, stage=None):
        if self.hparams.smoke:
            self.dataset = SlowSyntheticImages()
            return
        transform = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ]
        )
        self.dataset = torchvision.datasets.Imagenette(
            self.hparams.data_dir,
            split="train",
            size=IMAGENETTE_SIZE,
            transform=transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=True,
        )


class LitResNet18(L.LightningModule):
    """ResNet-18 from scratch; the model is not the experimental variable."""

    def __init__(self, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(
            weights=None, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        return F.cross_entropy(self(images), labels)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "ResNet-18 on Imagenette under Lightning; only the DataLoader "
            "profile changes between runs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        choices=["baseline", "optimized"],
        default="baseline",
        help="DataLoader profile, the only experimental change.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Where torchvision stores the Imagenette archive.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the profile's worker count.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the profile's persistent_workers setting.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU-only synthetic verification path; downloads nothing.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    settings = loader_settings(
        args.profile,
        smoke=args.smoke,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
    )
    accelerator = "cpu" if args.smoke else "auto"
    L.seed_everything(SEED, workers=True)

    traceml_lightning.init()  # TraceML line 1: fetch and H2D timers.

    datamodule = ImagenetteDataModule(
        args.data_dir, args.batch_size, smoke=args.smoke, **settings
    )
    model = LitResNet18(datamodule.num_classes)
    print(
        f"[demo] profile={args.profile} smoke={args.smoke} "
        f"accelerator={accelerator} num_workers={settings['num_workers']} "
        f"pin_memory={settings['pin_memory']} "
        f"persistent_workers={settings['persistent_workers']} "
        f"batch={args.batch_size} max_steps={args.max_steps} amp=off",
        flush=True,
    )
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator=accelerator,
        devices=1,
        # AMP stays off in both profiles: it would change compute time and
        # confound a comparison whose only variable is the loader.
        precision="32-true",
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[traceml_lightning.TraceMLCallback()],  # TraceML line 2
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
