"""Minimal single-node DDP training example for CPU or CUDA.

Use ``--steps`` to cap optimizer steps per rank for a short smoke run::

    traceml run examples/distributed/ddp_minimal.py \
        --nproc-per-node=2 --args --steps 20
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import traceml_ai as traceml

SEED = 42
INPUT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 10
NUM_SAMPLES = 8192
BATCH_SIZE = 64
EPOCHS = 4


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=None,
        help=(
            "Maximum optimizer steps per rank. By default, run all four "
            "epochs."
        ),
    )
    return parser.parse_args()


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(rank: int, world_size: int):
    x = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(x, y)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return loader, sampler


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    traceml.init(mode="auto")

    set_seed(SEED + rank)

    train_loader, train_sampler = prepare_data(rank, world_size)

    model = TinyMLP().to(device)

    if use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    global_step = 0
    epochs = EPOCHS
    if args.steps is not None:
        epochs = (args.steps + len(train_loader) - 1) // len(train_loader)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            global_step += 1

            with traceml.trace_step(model.module):
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.detach())

                if rank == 0 and global_step % 25 == 0:
                    print(
                        f"Epoch {epoch + 1} | Step {global_step} | "
                        f"loss: {running_loss / 25:.4f}"
                    )
                    running_loss = 0.0

            if args.steps is not None and global_step >= args.steps:
                break

    if rank == 0:
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
