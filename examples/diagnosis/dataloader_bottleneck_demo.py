"""PyTorch DataLoader bottleneck demo.

This example runs the same tiny PyTorch training loop in two scenarios:

* ``fast``: no artificial input delay.
* ``slow``: each dataset sample sleeps before returning, simulating slow
  decoding, tokenization, storage reads, or other expensive input work.

Trace the contrast with:

    traceml run examples/diagnosis/dataloader_bottleneck_demo.py --args --scenario fast
    traceml run examples/diagnosis/dataloader_bottleneck_demo.py --args --scenario slow

Use ``--sleep-ms`` and ``--num-workers`` to make the bottleneck stronger or to
test whether adding DataLoader workers reduces input wait. On a fast GPU, use
``--hidden-dim`` or ``--depth`` to increase model compute.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import traceml_ai as traceml

SEED = 42
INPUT_DIM = 256
HIDDEN_DIM = 512
NUM_CLASSES = 10
MODEL_DEPTH = 2

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_SAMPLES = 960
DEFAULT_NUM_EPOCHS = 1
DEFAULT_NUM_WORKERS = 0
DEFAULT_SLOW_SLEEP_MS = 8.0
DEFAULT_PRINT_EVERY = 10


@dataclass(frozen=True)
class DemoConfig:
    scenario: str
    sleep_ms: float
    batch_size: int
    num_workers: int
    num_samples: int
    epochs: int
    input_dim: int
    hidden_dim: int
    depth: int
    seed: int
    print_every: int

    @property
    def sleep_s(self) -> float:
        return self.sleep_ms / 1000.0


class SyntheticInputDataset(Dataset):
    """Synthetic classification data with optional per-sample input delay."""

    def __init__(
        self,
        num_samples: int,
        input_dim: int,
        num_classes: int,
        sleep_s: float,
    ):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sleep_s = sleep_s

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        # This is the synthetic bottleneck: in a real job this might be image
        # decode, tokenization, feature lookup, or slow storage I/O.
        if self.sleep_s > 0.0:
            time.sleep(self.sleep_s)
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
        return x, y


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(
        description="Run a tiny PyTorch job with a DataLoader bottleneck.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        choices=["fast", "slow"],
        default="fast",
        help="Preset input behavior for the demo.",
    )
    parser.add_argument(
        "--sleep-ms",
        type=non_negative_float,
        default=None,
        help=(
            "Per-sample input delay. Defaults to 0 for fast and "
            f"{DEFAULT_SLOW_SLEEP_MS:g} for slow."
        ),
    )
    parser.add_argument(
        "--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument(
        "--num-workers",
        type=non_negative_int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument(
        "--num-samples", type=positive_int, default=DEFAULT_NUM_SAMPLES
    )
    parser.add_argument(
        "--epochs", type=positive_int, default=DEFAULT_NUM_EPOCHS
    )
    parser.add_argument("--input-dim", type=positive_int, default=INPUT_DIM)
    parser.add_argument("--hidden-dim", type=positive_int, default=HIDDEN_DIM)
    parser.add_argument("--depth", type=positive_int, default=MODEL_DEPTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--print-every", type=positive_int, default=DEFAULT_PRINT_EVERY
    )
    args = parser.parse_args()

    if args.num_samples < args.batch_size:
        parser.error(
            "--num-samples must be greater than or equal to --batch-size."
        )

    sleep_ms = args.sleep_ms
    if sleep_ms is None:
        sleep_ms = 0.0 if args.scenario == "fast" else DEFAULT_SLOW_SLEEP_MS

    return DemoConfig(
        scenario=args.scenario,
        sleep_ms=sleep_ms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        epochs=args.epochs,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        seed=args.seed,
        print_every=args.print_every,
    )


def build_fast_dataset(config: DemoConfig) -> TensorDataset:
    x = torch.randn(config.num_samples, config.input_dim)
    y = torch.randint(0, NUM_CLASSES, (config.num_samples,))
    return TensorDataset(x, y)


def build_slow_dataset(config: DemoConfig) -> SyntheticInputDataset:
    return SyntheticInputDataset(
        num_samples=config.num_samples,
        input_dim=config.input_dim,
        num_classes=NUM_CLASSES,
        sleep_s=config.sleep_s,
    )


def build_dataloader(config: DemoConfig, device: torch.device) -> DataLoader:
    dataset = (
        build_fast_dataset(config)
        if config.scenario == "fast"
        else build_slow_dataset(config)
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=generator,
    )


def print_run_config(config: DemoConfig, device: torch.device) -> None:
    steps_per_epoch = config.num_samples // config.batch_size
    print("[TraceML DataLoader bottleneck demo]")
    print(f"Device: {device}")
    print(
        "Config: "
        f"scenario={config.scenario} "
        f"sleep_ms={config.sleep_ms:g} "
        f"batch_size={config.batch_size} "
        f"num_workers={config.num_workers} "
        f"num_samples={config.num_samples} "
        f"epochs={config.epochs} "
        f"input_dim={config.input_dim} "
        f"hidden_dim={config.hidden_dim} "
        f"depth={config.depth} "
        f"steps={steps_per_epoch * config.epochs}"
    )


def train(config: DemoConfig) -> None:
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_run_config(config, device)

    # Initialize before creating the DataLoader so auto mode can time input
    # fetches.
    traceml.init(mode="auto")

    loader = build_dataloader(config, device)
    model = TinyMLP(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    total_steps = 0
    for epoch in range(1, config.epochs + 1):
        for batch_x, batch_y in loader:
            total_steps += 1

            with traceml.trace_step(model):
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if total_steps % config.print_every == 0:
                print(
                    f"Epoch {epoch} | Step {total_steps} | "
                    f"loss: {loss.item():.4f}"
                )

    print(f"Done. Trained {total_steps} steps.")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
