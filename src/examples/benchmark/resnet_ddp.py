import os
import random

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as T
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from traceml.decorators import trace_model_instance, trace_step


# =========================
# CONFIG
# =========================
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
NUM_WORKERS = 1      # <-- change to 8 to fix jitter
IMAGE_SIZE = 224


# =========================
# Utilities
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(rank: int, world_size: int):
    """
    Heavy CPU transforms → real dataloader pressure.
    """
    transform = T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.GaussianBlur(5),
        T.RandomRotation(30),
        T.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

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
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    return loader, sampler


# =========================
# Step Components
# =========================
def load_batch_to_device(batch, device):
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def forward_pass(model, images):
    return model(images)


def backward_pass(loss):
    loss.backward()


def optimizer_step(optimizer):
    optimizer.step()


# =========================
# MAIN
# =========================
def main():

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

    set_seed(SEED + rank)

    # Data
    train_loader, train_sampler = prepare_data(rank, world_size)

    # Model
    model = torchvision.models.resnet50(num_classes=10).to(device)

    # Attach TraceML BEFORE DDP
    # trace_model_instance(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if use_cuda else None,
        output_device=local_rank if use_cuda else None,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    model.train()
    global_step = 0

    for epoch in range(EPOCHS):

        train_sampler.set_epoch(epoch)

        running_loss = 0.0

        for batch in train_loader:

            with trace_step(model.module):

                images, labels = load_batch_to_device(batch, device)

                optimizer.zero_grad(set_to_none=True)

                outputs = forward_pass(model, images)
                loss = criterion(outputs, labels)

                backward_pass(loss)
                optimizer_step(optimizer)

                running_loss += loss.detach()
                global_step += 1

                if rank == 0 and global_step % 50 == 0:
                    print(
                        f"[Train] epoch {epoch+1} step {global_step} "
                        f"| loss {(running_loss/50).item():.4f}"
                    )
                    running_loss.zero_()

    if rank == 0:
        print("Training finished.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
