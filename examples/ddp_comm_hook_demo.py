"""
DDP gradient-sync timing demo.

Shows ``traceml.wrap_ddp(model)`` instrumenting the DDP comm hook to
capture per-step NCCL all_reduce timing as ``_traceml_comm:ddp_grad_sync``
events.

Run with torchrun::

    torchrun --nproc_per_node=2 examples/ddp_comm_hook_demo.py

Or via the TraceML CLI::

    traceml run examples/ddp_comm_hook_demo.py

Convention
----------
- Pass the DDP **wrapper** to ``trace_step(model)`` — this triggers
  auto-install of the comm hook.
- Pass ``model.module`` (the inner module) to ``trace_model_instance()``
  for layer hooks.

Notes
-----
- During ``ddp_model.no_sync()`` accumulation steps, DDP suppresses the
  hook — those steps produce zero ``ddp_grad_sync`` events (correct:
  no communication happened).
- Comm timing overlaps with backward compute for early buckets. Only the
  last bucket's all_reduce is pure communication (backward is done).
  See ``deep_dive_bucket_ordering_and_overlap.md`` for details.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import traceml_ai as traceml

NUM_STEPS = 10
BATCH_SIZE = 32


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    traceml.init(mode="auto")

    inner_model = TinyMLP().to(device)

    if use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            inner_model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(inner_model)

    # Explicit wrap_ddp — alternatively, just pass `model` (the DDP
    # wrapper) to trace_step() and auto-install handles it.
    traceml.wrap_ddp(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(256, 128)
    y = torch.randint(0, 10, (256,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    model.train()

    for step, (batch_x, batch_y) in enumerate(loader):
        if step >= NUM_STEPS:
            break

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        with traceml.trace_step(model):
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    if rank == 0:
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
