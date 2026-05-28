"""Minimal TraceML + Ray Train example.

Run locally:
    python examples/ray_torchtrainer_minimal.py

Run with GPUs:
    python examples/ray_torchtrainer_minimal.py --use-gpu --num-workers 4
"""

from __future__ import annotations

import argparse


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
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 4),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    traceml.trace_model_instance(model)

    for step in range(int(config["steps"])):
        with traceml.trace_step(model):
            x = torch.randn(64, 32, device=device)
            y = torch.randint(0, 4, (64,), device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        if ctx.get_world_rank() == 0:
            print(f"step={step + 1} loss={loss.detach().item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
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

    trainer = TraceMLTorchTrainer(
        train_loop_per_worker,
        train_loop_config={
            "steps": args.steps,
            "use_gpu": args.use_gpu,
        },
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
        ),
        traceml_config=TraceMLRayConfig(mode="summary"),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
