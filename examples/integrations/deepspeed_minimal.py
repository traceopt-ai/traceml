"""Minimal DeepSpeed loop wrapped with traceml.trace_step.

Requires ``deepspeed`` and a CUDA GPU; exits cleanly without them. Run with:

    traceml run examples/integrations/deepspeed_minimal.py --mode=summary

Use ``--epochs`` or ``--steps`` to change the run length::

    traceml run examples/integrations/deepspeed_minimal.py \
        --mode=summary --args --steps 20
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import traceml_ai as traceml

SEED = 42
INPUT_DIM = 1024
HIDDEN_DIM = 2048
NUM_CLASSES = 10
NUM_SAMPLES = 100000
BATCH_SIZE = 256
EPOCHS = 2

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "deepspeed_config_minimal.json",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    length = parser.add_mutually_exclusive_group()
    length.add_argument(
        "--epochs",
        type=positive_int,
        default=EPOCHS,
        help="Number of full epochs to run.",
    )
    length.add_argument(
        "--steps",
        type=positive_int,
        default=None,
        help=(
            "Number of optimizer steps per rank. Replaces --epochs; by "
            "default, run --epochs full epochs."
        ),
    )
    return parser.parse_args()


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
    args = parse_args()
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
    epochs = args.epochs
    if args.steps is not None:
        epochs = (args.steps + len(train_loader) - 1) // len(train_loader)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        running_loss = torch.zeros((), device=device)

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

            # Keep loss logging OUTSIDE trace_step so it never inflates the
            # measured step, and accumulate on-device so there is no per-step
            # device-to-host sync (only when we actually print), mirroring the
            # quickstart example.
            running_loss += loss.detach()
            global_step += 1

            if rank == 0 and global_step % 50 == 0:
                print(
                    f"Epoch {epoch + 1} | Step {global_step} | "
                    f"loss: {float(running_loss) / 50:.4f}"
                )
                running_loss.zero_()

            if args.steps is not None and global_step >= args.steps:
                break

    if rank == 0:
        print("Done.")

    # Match the DDP / FSDP examples: tear down the process group that
    # deepspeed.init_distributed() set up.
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
