"""
Memory-creep synthetic-MLP DDP scenario for TraceOpt demo data.

This keeps the healthy baseline data, model, loss, and DDP loop, but retains a
small CUDA tensor every step on every rank. The leak is intentionally uniform
across ranks and below high-pressure thresholds, so TraceML should report
MEMORY CREEP rather than rank imbalance or near-OOM pressure.
"""

from __future__ import annotations

import os
import random

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

# The summary creep rule needs a sustained rise across the analyzed window.
# 8 MiB x ~256 steps grows by ~2 GiB per rank on the demo A100, large enough
# for confirmed creep while staying far below high-pressure memory thresholds.
LEAK_MIB_PER_STEP = 8
LEAK_BYTES_PER_STEP = LEAK_MIB_PER_STEP * 1024 * 1024


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


def retain_step_memory(
    retained_tensors: list[torch.Tensor],
    *,
    device: torch.device,
    rank: int,
) -> None:
    if device.type != "cuda":
        return

    # Keep live references so allocated/reserved CUDA memory rises step by
    # step. Using the same allocation on every rank avoids creating a rank
    # imbalance demo by accident.
    with torch.no_grad():
        leak = torch.empty(
            LEAK_BYTES_PER_STEP,
            dtype=torch.uint8,
            device=device,
        )
        leak.fill_(rank % 251)
        retained_tensors.append(leak)


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
    retained_tensors: list[torch.Tensor] = []

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

                retain_step_memory(
                    retained_tensors,
                    device=device,
                    rank=rank,
                )

                acc = accuracy_from_logits(logits.detach(), labels)
                running_loss += float(loss.detach())
                running_acc += float(acc.detach())

            if rank == 0 and global_step % LOG_EVERY_STEPS == 0:
                retained_gib = (
                    len(retained_tensors) * LEAK_BYTES_PER_STEP / (1024**3)
                )
                print(
                    f"[memory-creep] epoch={epoch + 1} "
                    f"step={global_step}/{total_steps} "
                    f"loss={running_loss / LOG_EVERY_STEPS:.4f} "
                    f"acc={running_acc / LOG_EVERY_STEPS:.4f} "
                    f"retained={retained_gib:.2f}GiB"
                )
                running_loss = 0.0
                running_acc = 0.0

    if rank == 0:
        print("[memory-creep] done")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
