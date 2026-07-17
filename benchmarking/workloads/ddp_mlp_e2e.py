"""Simple DDP MLP workload for end-to-end TraceML timing.

This is intentionally close to ``examples/ddp_minimal.py`` but parameterized
for benchmark runs. Use it to compare whole-command wall time:

    time torchrun --nproc_per_node=1 benchmarking/workloads/ddp_mlp_e2e.py
    time traceml run benchmarking/workloads/ddp_mlp_e2e.py --nproc-per-node=1

The script only enables TraceML when it is launched by ``traceml run``.
Plain ``python``/``torchrun`` runs execute normal PyTorch training.
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import traceml_ai as traceml


class BenchmarkMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(max(0, hidden_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duration-based DDP MLP training workload."
    )
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=262144)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.input_dim <= 0 or args.hidden_dim <= 0:
        raise SystemExit("--input-dim and --hidden-dim must be positive")
    if args.hidden_layers <= 0:
        raise SystemExit("--hidden-layers must be positive")
    if args.num_classes <= 1:
        raise SystemExit("--num-classes must be greater than 1")
    if args.num_samples < args.batch_size:
        raise SystemExit("--num-samples must be at least --batch-size")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def traceml_active() -> bool:
    if os.environ.get("TRACEML_DISABLED", "0") == "1":
        return False
    return bool(os.environ.get("TRACEML_SESSION_ID"))


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    return rank, local_rank, world_size


def prepare_data(args: argparse.Namespace, rank: int, world_size: int):
    generator = torch.Generator().manual_seed(args.seed)
    features = torch.randn(
        args.num_samples,
        args.input_dim,
        dtype=torch.float32,
        generator=generator,
    )
    labels = torch.randint(
        low=0,
        high=args.num_classes,
        size=(args.num_samples,),
        dtype=torch.long,
        generator=generator,
    )
    dataset = TensorDataset(features, labels)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader, sampler


def should_stop(
    start_s: float, duration_s: float, device: torch.device
) -> bool:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start_s >= duration_s


def main() -> None:
    args = parse_args()
    validate_args(args)

    rank, local_rank, world_size = setup_distributed()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        amp_dtype = torch.float16
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32

    trace_enabled = traceml_active()
    if trace_enabled:
        traceml.init(mode="auto")

    set_seed(args.seed + rank)
    loader, sampler = prepare_data(args, rank, world_size)

    base_model = BenchmarkMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        num_classes=args.num_classes,
    ).to(device)

    if use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(base_model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        enabled=use_cuda,
        device="cuda" if use_cuda else "cpu",
    )

    model.train()
    start_s = time.perf_counter()
    global_step = 0
    epoch = 0
    running_loss = 0.0

    while True:
        sampler.set_epoch(epoch)
        epoch += 1

        for batch_x, batch_y in loader:
            if should_stop(start_s, args.duration_sec, device):
                elapsed = time.perf_counter() - start_s
                if rank == 0:
                    print(
                        f"Done. steps={global_step} "
                        f"elapsed_s={elapsed:.2f} "
                        f"steps_per_s={global_step / elapsed:.3f} "
                        f"traceml_active={trace_enabled}"
                    )
                dist.destroy_process_group()
                return

            context = (
                traceml.trace_step(base_model)
                if trace_enabled
                else nullcontext()
            )
            with context:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda" if use_cuda else "cpu",
                    enabled=use_cuda,
                    dtype=amp_dtype,
                ):
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            global_step += 1
            running_loss += float(loss.detach())
            if (
                rank == 0
                and args.log_every > 0
                and (global_step % args.log_every == 0)
            ):
                elapsed = time.perf_counter() - start_s
                print(
                    f"step={global_step} "
                    f"elapsed_s={elapsed:.1f} "
                    f"loss={running_loss / args.log_every:.4f}"
                )
                running_loss = 0.0


if __name__ == "__main__":
    main()
