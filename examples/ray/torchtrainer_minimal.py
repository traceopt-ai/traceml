"""Minimal TraceML + Ray Train example.

Ray Data iterators are not torch DataLoader objects, so TraceML cannot see
their input fetches through the PyTorch DataLoader patch. Wrap Ray
``iter_torch_batches(...)`` with ``traceml.wrap_dataloader_fetch(...)`` when
you want Ray input timing in the Step Time summary.

Run locally:
    python examples/ray/torchtrainer_minimal.py

Run with GPUs:
    python examples/ray/torchtrainer_minimal.py --use-gpu --num-workers 4
"""

from __future__ import annotations

import argparse

INPUT_DIM = 32
NUM_CLASSES = 4


def build_dataset(*, num_samples: int, seed: int):
    import numpy as np
    import ray

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, INPUT_DIM), dtype=np.float32)
    y = rng.integers(0, NUM_CLASSES, size=(num_samples,), dtype=np.int64)
    return ray.data.from_items(
        [{"x": x[idx], "y": int(y[idx])} for idx in range(num_samples)]
    )


def train_loop_per_worker(config):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from ray import train

    import traceml_ai as traceml

    ctx = train.get_context()
    device = torch.device(
        "cuda"
        if bool(config["use_gpu"]) and torch.cuda.is_available()
        else "cpu"
    )

    model = nn.Sequential(
        nn.Linear(INPUT_DIM, 64),
        nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_ds = train.get_dataset_shard("train")
    train_loader = train_ds.iter_torch_batches(
        batch_size=int(config["batch_size"]),
        prefetch_batches=1,
    )

    delay_s = float(config.get("input_delay_ms", 0.0)) / 1000.0
    if delay_s > 0.0:
        train_loader = _delay_batches(train_loader, delay_s)

    train_loader = traceml.wrap_dataloader_fetch(train_loader)

    for step, batch in enumerate(train_loader):
        if step >= int(config["steps"]):
            break

        with traceml.trace_step(model):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        if ctx.get_world_rank() == 0:
            print(f"step={step + 1} loss={loss.detach().item():.4f}")


def _delay_batches(iterator, delay_s: float):
    import time

    for batch in iterator:
        time.sleep(delay_s)
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-delay-ms",
        type=float,
        default=0.0,
        help="Optional per-batch delay to make Ray input timing visible.",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--ray-address", default=None)
    args = parser.parse_args()

    import ray
    from ray.train import ScalingConfig

    from traceml_ai.integrations.ray import (
        TraceMLRayConfig,
        TraceMLTorchTrainer,
    )

    ray.init(address=args.ray_address)

    train_dataset = build_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
    )

    trainer = TraceMLTorchTrainer(
        train_loop_per_worker,
        train_loop_config={
            "steps": args.steps,
            "batch_size": args.batch_size,
            "input_delay_ms": args.input_delay_ms,
            "use_gpu": args.use_gpu,
        },
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
        ),
        datasets={"train": train_dataset},
        traceml_config=TraceMLRayConfig(mode="summary"),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
