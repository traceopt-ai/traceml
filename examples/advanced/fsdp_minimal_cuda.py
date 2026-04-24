import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import traceml

SEED = 42
INPUT_DIM = 1024
HIDDEN_DIM = 2048
NUM_CLASSES = 10

NUM_SAMPLES = 100000
BATCH_SIZE = 256
EPOCHS = 6


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(rank: int, world_size: int):
    x = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(x, y)

    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, train_sampler


def load_batch_to_device(batch, device):
    batch_x, batch_y = batch
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)
    return batch_x, batch_y


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not torch.cuda.is_available():
        raise RuntimeError("This minimal FSDP example expects CUDA GPUs.")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    traceml.init(mode="auto")

    set_seed(SEED + rank)

    train_loader, train_sampler = prepare_data(rank, world_size)

    # Build the real model first
    base_model = TinyMLP().to(device)

    # Attach TraceML hooks to the real model before FSDP wrapping
    traceml.trace_model_instance(base_model)

    # Wrap with FSDP
    model = FSDP(
        base_model,
        device_id=torch.cuda.current_device(),
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    global_step = 0
    running_loss = 0.0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            with traceml.trace_step(base_model):
                batch_x, batch_y = load_batch_to_device(batch, device)

                optimizer.zero_grad(set_to_none=True)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

                running_loss += loss.detach()
                global_step += 1

                if rank == 0 and global_step % 50 == 0:
                    print(
                        f"[Train] epoch {epoch+1} step {global_step} "
                        f"| loss {(running_loss / 50).item():.4f}"
                    )
                    running_loss.zero_()

    if rank == 0:
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
