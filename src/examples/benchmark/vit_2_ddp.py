import os
import random
import time
from typing import Dict

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.models import vit_l_16  # <-- heavier model

# ============================================================
# TraceML imports
# ============================================================
from traceml.decorators import trace_step

# ============================================================
# CONFIG (TUNE THESE FOR MEMORY WALL)
# ============================================================
SEED = 42

IMAGE_SIZE = 224         # 224 → 384 → 448 (push this)
PER_GPU_BATCH = 16        # increase until ~45GB peak
NUM_WORKERS = 8

LR = 3e-4
WEIGHT_DECAY = 0.05

MAX_STEPS = 5000
LOG_EVERY = 50


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Data
# ============================================================
def prepare_dataloader(rank: int, world_size: int):

    if rank == 0:
        load_dataset(
            "ljnlonoljpiljm/places365-256px",
            split="train[:20%]"
        )
    dist.barrier()

    dataset = load_dataset(
        "ljnlonoljpiljm/places365-256px",
        split="train[:20%]"
    )

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    def preprocess(batch):
        images = [transform(img.convert("RGB")) for img in batch["image"]]
        return {
            "pixel_values": images,
            "labels": batch["label"],
        }

    dataset = dataset.with_transform(preprocess)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=PER_GPU_BATCH,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    return loader, sampler


# ============================================================
# Training Components
# ============================================================
def load_batch_to_device(batch: Dict, device: torch.device):
    return {
        "images": batch["pixel_values"].to(device, non_blocking=True),
        "labels": torch.tensor(batch["labels"]).to(device, non_blocking=True),
    }


def forward_pass(model, images):
    return model(images)


def compute_loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels)


def backward_pass(loss, scaler: GradScaler):
    scaler.scale(loss).backward()


def optimizer_step(optimizer, scaler):
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


# ============================================================
# MAIN
# ============================================================
def main():

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group("nccl")
    set_seed(SEED + rank)

    train_loader, train_sampler = prepare_dataloader(rank, world_size)

    # -----------------------------
    # Model (heavier for memory wall)
    # -----------------------------
    model = vit_l_16(num_classes=365).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = GradScaler()

    model.train()
    global_step = 0
    start_time = time.time()

    torch.cuda.reset_peak_memory_stats(device)

    # ========================================================
    # TRAIN LOOP
    # ========================================================
    while global_step < MAX_STEPS:
        train_sampler.set_epoch(global_step)

        for batch in train_loader:
            if global_step >= MAX_STEPS:
                break

            with trace_step(model.module):

                batch = load_batch_to_device(batch, device)

                with autocast(dtype=torch.float16):
                    logits = forward_pass(model, batch["images"])
                    loss = compute_loss(logits, batch["labels"])

                backward_pass(loss, scaler)
                optimizer_step(optimizer, scaler)

            global_step += 1

            if rank == 0 and global_step % LOG_EVERY == 0:
                elapsed = (time.time() - start_time) / 60
                peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                print(
                    f"step {global_step:6d} | "
                    f"loss {loss.item():.4f} | "
                    f"peak_mem {peak_gb:.1f} GB | "
                    f"elapsed {elapsed:.1f} min"
                )

    if rank == 0:
        print("Training complete")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
