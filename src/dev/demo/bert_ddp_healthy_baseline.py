"""
Healthy BERT DDP baseline for TraceOpt demo data.

Run this with TraceML/torchrun to generate a balanced final_summary.json.
The script intentionally avoids rank-specific sleeps, extra compute branches,
checkpointing, and validation so the expected diagnosis is NORMAL/BALANCED
when the host and GPUs are healthy.
"""

from __future__ import annotations

import os
import random

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import traceml_ai as traceml

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
MAX_TRAIN_EXAMPLES = 32768
VOCAB_SIZE = 30522

BATCH_SIZE = 32
EPOCHS = 2
LR = 2e-6
WARMUP_RATIO = 0.06
LOG_EVERY_STEPS = 50


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
    input_ids = torch.randint(
        low=100,
        high=VOCAB_SIZE,
        size=(MAX_TRAIN_EXAMPLES, MAX_LENGTH),
        dtype=torch.long,
        generator=generator,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(
        low=0,
        high=4,
        size=(MAX_TRAIN_EXAMPLES,),
        dtype=torch.long,
        generator=generator,
    )
    train_ds = TensorDataset(input_ids, attention_mask, labels)

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        drop_last=True,
    )

    return train_loader, train_sampler


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

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
    ).to(device)

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
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        enabled=use_cuda,
        device="cuda" if use_cuda else "cpu",
    )

    model.train()
    global_step = 0
    running_loss = 0.0
    running_acc = 0.0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        for input_ids, attention_mask, labels in train_loader:
            global_step += 1

            with traceml.trace_step(model.module):
                batch = {
                    "input_ids": input_ids.to(device, non_blocking=True),
                    "attention_mask": attention_mask.to(
                        device, non_blocking=True
                    ),
                    "labels": labels.to(device, non_blocking=True),
                }

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type="cuda" if use_cuda else "cpu",
                    enabled=use_cuda,
                    dtype=amp_dtype,
                ):
                    out = model(**batch)

                loss = out.loss
                logits = out.logits

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                acc = accuracy_from_logits(logits.detach(), batch["labels"])
                running_loss += float(loss.detach())
                running_acc += float(acc.detach())

            if rank == 0 and global_step % LOG_EVERY_STEPS == 0:
                print(
                    f"[healthy-baseline] epoch={epoch + 1} "
                    f"step={global_step}/{total_steps} "
                    f"loss={running_loss / LOG_EVERY_STEPS:.4f} "
                    f"acc={running_acc / LOG_EVERY_STEPS:.4f}"
                )
                running_loss = 0.0
                running_acc = 0.0

    if rank == 0:
        print("[healthy-baseline] done")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
