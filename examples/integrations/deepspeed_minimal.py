"""Minimal TraceML + DeepSpeed example.

TraceML does not own the DeepSpeed training loop. You keep the standard
DeepSpeed loop (forward, ``model_engine.backward(loss)``,
``model_engine.step()``) and wrap each step with ``traceml.trace_step(...)``,
exactly like the plain PyTorch, DDP, and FSDP examples.

Run (single GPU, end-of-run summary):

    traceml run examples/integrations/deepspeed_minimal.py --mode=summary

Single-node multi-GPU (e.g. 2 GPUs):

    traceml run examples/integrations/deepspeed_minimal.py \\
        --nproc-per-node=2 --mode=summary

DeepSpeed is optional and is not a TraceML dependency. Install it with:

    pip install "traceml-ai[deepspeed]"

or follow https://www.deepspeed.ai/getting-started/. DeepSpeed requires a
CUDA GPU; this example exits cleanly when DeepSpeed or a GPU is unavailable.
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import traceml_ai as traceml

SEED = 42
INPUT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 10
NUM_SAMPLES = 8192
BATCH_SIZE = 64
EPOCHS = 4

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "deepspeed_config_minimal.json",
)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(rank: int, world_size: int):
    x = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(x, y)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    # batch_size must match train_micro_batch_size_per_gpu in the config.
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )

    return loader, sampler


def main() -> None:
    try:
        import deepspeed
    except ImportError:
        print(
            "This example requires DeepSpeed. Install it with:\n"
            '  pip install "traceml-ai[deepspeed]"\n'
            "or follow https://www.deepspeed.ai/getting-started/. "
            "DeepSpeed also requires a CUDA GPU."
        )
        sys.exit(0)

    if not torch.cuda.is_available():
        print(
            "DeepSpeed requires a CUDA GPU; none is available, so this "
            "minimal example has nothing to run. Exiting cleanly."
        )
        sys.exit(0)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # deepspeed.init_distributed() replaces torch.distributed's
    # init_process_group. It reads RANK/LOCAL_RANK/WORLD_SIZE from the
    # torchrun environment, which is how `traceml run` launches this script.
    deepspeed.init_distributed()

    set_seed(SEED + rank)

    model = TinyMLP()
    train_loader, train_sampler = prepare_data(rank, world_size)

    # DeepSpeed builds the optimizer from the "optimizer" block in the config
    # and returns (engine, optimizer, dataloader, lr_scheduler).
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=CONFIG_PATH,
    )

    # Enable TraceML auto instrumentation (dataloader, forward, backward,
    # optimizer, and H2D timing) for this process.
    traceml.init(mode="auto")

    criterion = nn.CrossEntropyLoss()
    device = model_engine.device

    model_engine.train()
    global_step = 0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            # Wrap the DeepSpeed step. Pass model_engine.module (the unwrapped
            # model) so TraceML's forward timer targets your model, exactly
            # like model.module for DDP and base_model for FSDP.
            with traceml.trace_step(model_engine.module):
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                logits = model_engine(batch_x)
                loss = criterion(logits, batch_y)

                model_engine.backward(loss)
                model_engine.step()

                running_loss += float(loss.detach())
                global_step += 1

                if rank == 0 and global_step % 25 == 0:
                    print(
                        f"Epoch {epoch + 1} | Step {global_step} | "
                        f"loss: {running_loss / 25:.4f}"
                    )
                    running_loss = 0.0

    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()
