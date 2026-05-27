"""
Compute-straggler synthetic-MLP DDP scenario for TraceOpt demo data.

This keeps the healthy baseline data, model output, optimizer, and DDP loop,
but makes one global rank do extra GPU matrix work inside forward(). The extra
work is multiplied by zero before it touches logits, so the training semantics
stay the same while TraceML sees one rank with higher model-compute cost.
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

# One rank does extra forward-side GPU work. If your GPU is much faster/slower,
# tune only EXTRA_FORWARD_MATMULS while keeping the model/data unchanged.
STRAGGLER_RANK = 2
EXTRA_FORWARD_DIM = 4096
EXTRA_FORWARD_MATMULS = 24


class BaselineMLP(nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = int(rank)
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

        generator = torch.Generator().manual_seed(SEED)
        probe_a = torch.randn(
            EXTRA_FORWARD_DIM,
            EXTRA_FORWARD_DIM,
            dtype=torch.float32,
            generator=generator,
        )
        probe_b = torch.randn(
            EXTRA_FORWARD_DIM,
            EXTRA_FORWARD_DIM,
            dtype=torch.float32,
            generator=generator,
        )
        self.register_buffer("probe_a", probe_a.mul_(0.01))
        self.register_buffer("probe_b", probe_b.mul_(0.01))

    def _extra_forward_compute(self) -> torch.Tensor:
        work = self.probe_a
        other = self.probe_b
        with torch.no_grad():
            for _ in range(EXTRA_FORWARD_MATMULS):
                work = torch.matmul(work, other)
        return work.flatten()[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        if self.rank == STRAGGLER_RANK:
            extra = self._extra_forward_compute().to(dtype=logits.dtype)
            logits = logits + extra * 0.0
        return logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean()


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
    model = BaselineMLP(rank=rank).to(device)

    traceml.trace_model_instance(model)

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
                    f"[compute-straggler] epoch={epoch + 1} "
                    f"step={global_step}/{total_steps} "
                    f"loss={running_loss / LOG_EVERY_STEPS:.4f} "
                    f"acc={running_acc / LOG_EVERY_STEPS:.4f} "
                    f"straggler_rank={STRAGGLER_RANK} "
                    f"extra_matmuls={EXTRA_FORWARD_MATMULS}"
                )
                running_loss = 0.0
                running_acc = 0.0

    if rank == 0:
        print("[compute-straggler] done")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
