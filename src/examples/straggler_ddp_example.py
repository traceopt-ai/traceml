import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from traceml.decorators import trace_model_instance, trace_step

SEED = 42
INPUT_DIM = 1024
HIDDEN_DIM = 2048
NUM_CLASSES = 10

BATCH_SIZE = 128
NUM_SAMPLES = 12000
NUM_EPOCHS = 40

# Main knob for the demo:
# rank 0 does extra compute every step, making it the straggler.
STRAGGLER_RANK = 0
EXTRA_MATMUL_ITERS = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FastDataset(Dataset):
    def __init__(self, num_samples, input_dim, num_classes):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
        return x, y


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


def inject_straggler_compute(device: torch.device) -> None:
    if device.type != "cuda":
        return

    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    for _ in range(4):
        x = x @ y


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

    dataset = FastDataset(
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
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
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=True,
    )

    model = TinyMLP().to(device)
    trace_model_instance(model)

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
    total_steps = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        sampler.set_epoch(epoch)

        for batch_x, batch_y in loader:
            total_steps += 1

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with trace_step(model.module):
                optimizer.zero_grad(set_to_none=True)

                logits = model(batch_x)

                # Deliberately make one rank slower during traced compute.
                if rank == STRAGGLER_RANK:
                    inject_straggler_compute(device)

                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if rank == 0 and total_steps % 25 == 0:
                print(
                    f"Epoch {epoch} | Step {total_steps} | "
                    f"loss: {loss.item():.4f}"
                )

    if rank == 0:
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
