"""Minimal DDP example exercising the TRA-16 all_reduce comm-timing patch.

What this example demonstrates
------------------------------
- A trivial DDP training loop that runs entirely on synthetic tensors (no
  real dataset, no tokenizer) so the demo is small enough to read in one
  sitting and runs end-to-end on a 1- or 2-rank torchrun (gloo on CPU,
  nccl on CUDA).
- An EXPLICIT user-issued ``dist.all_reduce(loss.detach())`` inside the
  ``trace_step`` block. This is the in-scope path the v0 patch site
  catches, and it surfaces a ``_traceml_comm:all_reduce`` event in each
  step's timing batch.
- DDP's per-bucket gradient sync also runs each step but is dispatched by
  the C++ Reducer and does NOT route through the Python ``all_reduce``
  symbol. v0 intentionally does not capture those events; v1 will via
  ``register_comm_hook``.

How to run
----------
Single rank::

    torchrun --standalone --nproc_per_node=1 examples/all_reduce_demo.py

Two ranks (gloo on CPU, fast)::

    torchrun --standalone --nproc_per_node=2 examples/all_reduce_demo.py

Two ranks with NCCL on a multi-GPU host::

    torchrun --standalone --nproc_per_node=2 examples/all_reduce_demo.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import traceml

INPUT_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 10

BATCH_SIZE = 32
NUM_STEPS = 50


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def synthetic_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one batch of random inputs and labels."""
    x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
    y = torch.randint(
        0, OUTPUT_DIM, (BATCH_SIZE,), dtype=torch.long, device=device
    )
    return x, y


def main() -> None:
    # ----------------------------------------------------------------
    # DDP environment (provided automatically by torchrun)
    # ----------------------------------------------------------------
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size
    )

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # ----------------------------------------------------------------
    # TraceML init -- mode='auto' enables the all_reduce comm-timing
    # patch, which records `_traceml_comm:all_reduce` events for every
    # user-issued dist.all_reduce(...) call inside trace_step.
    # ----------------------------------------------------------------
    traceml.init(mode="auto")

    # ----------------------------------------------------------------
    # Model + optimizer
    # ----------------------------------------------------------------
    torch.manual_seed(42 + rank)
    model = TinyMLP().to(device)

    # Attach TraceML hooks to the real model BEFORE wrapping with DDP.
    traceml.trace_model_instance(model)

    if use_cuda:
        ddp_model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    else:
        ddp_model = nn.parallel.DistributedDataParallel(model)

    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------------------
    # Training loop on synthetic data
    # ----------------------------------------------------------------
    ddp_model.train()

    for step in range(1, NUM_STEPS + 1):
        # `trace_step(model.module)` opens the per-step boundary that
        # activates forward / backward / all_reduce auto-timers.
        with traceml.trace_step(ddp_model.module):
            x, y = synthetic_batch(device)

            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            # Explicit user-issued all_reduce on the detached loss: this
            # is the in-scope path the v0 patch records as the
            # `_traceml_comm:all_reduce` event.
            loss_for_log = loss.detach().clone()
            dist.all_reduce(loss_for_log)
            loss_for_log /= world_size

            if rank == 0 and step % 10 == 0:
                print(
                    f"[step {step:>3}] mean-loss across ranks: "
                    f"{loss_for_log.item():.4f}"
                )

    if rank == 0:
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
