"""
Input-bound synthetic-MLP DDP scenario for TraceOpt demo data.

This keeps the healthy baseline model and training loop shape, but injects a
uniform per-batch delay inside the DataLoader path. TraceML's summary
input-bound rule expects dataloader time to be a large share of the step while
cross-rank input skew stays low, so every rank receives the same delay.
"""

from __future__ import annotations

import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import traceml_ai as traceml

SEED = 42
NUM_SAMPLES = 65536
INPUT_DIM = 2048
HIDDEN_DIM = 8192
NUM_CLASSES = 1000

BATCH_SIZE = 128
EPOCHS = 2
LR = 1e-3
LOG_EVERY_STEPS = 50

# Dataloader share must clear the summary input-bound threshold (30%) while
# staying uniform across ranks so this does not become an input straggler.
INPUT_DELAY_S = 0.16


class BaselineMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean()


def slow_input_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    time.sleep(INPUT_DELAY_S)
    features, labels = zip(*batch)
    return torch.stack(features, dim=0), torch.stack(labels, dim=0)


def prepare_data(rank: int, world_size: int):
    generator = torch.Generator().manual_seed(SEED)
    features = torch.randn(
        NUM_SAMPLES,
        INPUT_DIM,
        dtype=torch.float32,
        generator=generator,
    )
    labels = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(NUM_SAMPLES,),
        dtype=torch.long,
        generator=generator,
    )
    dataset = TensorDataset(features, labels)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=slow_input_collate,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        drop_last=True,
    )

    return loader, sampler


def main() -> None:
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
        amp_dtype = torch.float16
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32

    traceml.init(mode="auto")
    set_seed(SEED + rank)

    train_loader, train_sampler = prepare_data(rank, world_size)
    model = BaselineMLP().to(device)

    if use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        enabled=use_cuda,
        device="cuda" if use_cuda else "cpu",
    )

    model.train()
    total_steps = EPOCHS * len(train_loader)
    global_step = 0
    running_loss = 0.0
    running_acc = 0.0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        for features, labels in train_loader:
            global_step += 1

            with traceml.trace_step(model.module):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type="cuda" if use_cuda else "cpu",
                    enabled=use_cuda,
                    dtype=amp_dtype,
                ):
                    logits = model(features)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                acc = accuracy_from_logits(logits.detach(), labels)
                running_loss += float(loss.detach())
                running_acc += float(acc.detach())

            if rank == 0 and global_step % LOG_EVERY_STEPS == 0:
                print(
                    f"[input-bound] epoch={epoch + 1} "
                    f"step={global_step}/{total_steps} "
                    f"loss={running_loss / LOG_EVERY_STEPS:.4f} "
                    f"acc={running_acc / LOG_EVERY_STEPS:.4f} "
                    f"input_delay_s={INPUT_DELAY_S:.2f}"
                )
                running_loss = 0.0
                running_acc = 0.0

    if rank == 0:
        print("[input-bound] done")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
