import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from traceml.decorators import trace_step

SEED = 42
INPUT_DIM = 1024
HIDDEN_DIM = 2048
NUM_CLASSES = 10

BATCH_SIZE = 128
NUM_SAMPLES = 12000
NUM_EPOCHS = 20

# Main knobs for the demo:
# rank 0 becomes slow in the input path, not compute.
STRAGGLER_RANK = 0
STRAGGLER_SLEEP_S = 0.25
STRAGGLER_EVERY_N_BATCHES = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FastDataset(Dataset):
    def __init__(self, num_samples: int, input_dim: int, num_classes: int):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
        return x, y


class SlowRankCollate:
    """
    Inject delay on one rank during batch collation.

    This makes one rank slow before the traced compute step starts, which helps
    TraceML surface an INPUT STRAGGLER diagnosis.
    """

    def __init__(
        self,
        *,
        rank: int,
        straggler_rank: int,
        sleep_s: float,
        every_n_batches: int,
    ) -> None:
        self.rank = rank
        self.straggler_rank = straggler_rank
        self.sleep_s = float(sleep_s)
        self.every_n_batches = max(1, int(every_n_batches))
        self.batch_idx = 0

    def __call__(self, batch):
        self.batch_idx += 1

        if (
            self.rank == self.straggler_rank
            and self.batch_idx % self.every_n_batches == 0
        ):
            time.sleep(self.sleep_s)

        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


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

    collate_fn = SlowRankCollate(
        rank=rank,
        straggler_rank=STRAGGLER_RANK,
        sleep_s=STRAGGLER_SLEEP_S,
        every_n_batches=STRAGGLER_EVERY_N_BATCHES,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=True,
    )

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
                loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

            if rank == 0 and total_steps % 25 == 0:
                print(
                    f"Epoch {epoch} | Step {total_steps} | "
                    f"loss: {loss.item():.4f} | "
                    f"input_straggler_rank={STRAGGLER_RANK}"
                )

    if rank == 0:
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
