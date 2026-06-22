"""
Single-node DDP rank-straggler demo for TraceML.

Run the same DDP training loop in three modes:

  balanced           every rank does the same input and compute work
  input-straggler    one rank sleeps during batch collation
  compute-straggler  one rank does extra optimizer-side GPU work

The baseline uses precomputed tensors and a reasonably large MLP so the normal
case is compute-heavy on GPUs such as T4 or L4. The straggler scenarios then
change one phase at a time, which makes the TraceML summary easier to read.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import traceml_ai as traceml


class MLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SlowRankCollate:
    """Inject deterministic input delay on one global rank."""

    def __init__(
        self,
        *,
        rank: int,
        straggler_rank: int,
        sleep_ms: float,
        every_n_batches: int,
    ) -> None:
        self.rank = int(rank)
        self.straggler_rank = int(straggler_rank)
        self.sleep_s = max(0.0, float(sleep_ms)) / 1000.0
        self.every_n_batches = max(1, int(every_n_batches))
        self.batch_idx = 0

    def __call__(self, batch):
        self.batch_idx += 1
        if (
            self.sleep_s > 0.0
            and self.rank == self.straggler_rank
            and self.batch_idx % self.every_n_batches == 0
        ):
            time.sleep(self.sleep_s)

        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


class RankStragglerAdamW(AdamW):
    """AdamW with optional extra optimizer-side matrix work on one rank."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        rank: int,
        straggler_rank: int,
        extra_matmuls: int,
        extra_dim: int,
        device: torch.device,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self.rank = int(rank)
        self.straggler_rank = int(straggler_rank)
        self.extra_matmuls = max(0, int(extra_matmuls))
        self.extra_dim = max(1, int(extra_dim))
        self.work_a: torch.Tensor | None = None
        self.work_b: torch.Tensor | None = None

        if self.extra_matmuls > 0:
            self.work_a = torch.randn(
                self.extra_dim,
                self.extra_dim,
                device=device,
                dtype=torch.float32,
            ).mul_(0.01)
            self.work_b = torch.randn(
                self.extra_dim,
                self.extra_dim,
                device=device,
                dtype=torch.float32,
            ).mul_(0.01)

    def step(self, closure=None):
        result = super().step(closure=closure)
        if self.rank == self.straggler_rank and self.extra_matmuls > 0:
            self._extra_optimizer_compute()
        return result

    def _extra_optimizer_compute(self) -> None:
        if self.work_a is None or self.work_b is None:
            return

        work = self.work_a
        other = self.work_b
        with torch.no_grad():
            for _ in range(self.extra_matmuls):
                work = torch.matmul(work, other)

        # Keep one value alive so eager execution cannot trivially discard work.
        self.work_a = work


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DDP rank-straggler demo for TraceML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        choices=("balanced", "input-straggler", "compute-straggler"),
        default="balanced",
        help="Which rank-skew pattern to create.",
    )
    parser.add_argument("--num-samples", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--straggler-rank",
        type=int,
        default=0,
        help="Global rank to slow down in straggler scenarios.",
    )
    parser.add_argument(
        "--input-sleep-ms",
        type=float,
        default=200.0,
        help="Per-batch collate delay for the input-straggler scenario.",
    )
    parser.add_argument(
        "--input-every-n-batches",
        type=int,
        default=1,
        help="Apply input delay every N batches on the straggler rank.",
    )
    parser.add_argument(
        "--compute-extra-matmuls",
        type=int,
        default=8,
        help="Extra optimizer-side matrix multiplications on the straggler rank.",
    )
    parser.add_argument(
        "--compute-extra-dim",
        type=int,
        default=2048,
        help="Matrix dimension for compute-straggler extra work.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler]:
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

    sleep_ms = (
        args.input_sleep_ms if args.scenario == "input-straggler" else 0.0
    )
    collate_fn = SlowRankCollate(
        rank=rank,
        straggler_rank=args.straggler_rank,
        sleep_ms=sleep_ms,
        every_n_batches=args.input_every_n_batches,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        drop_last=True,
    )
    return loader, sampler


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if args.scenario != "balanced" and args.straggler_rank >= world_size:
        raise ValueError(
            f"--straggler-rank={args.straggler_rank} is outside WORLD_SIZE={world_size}"
        )

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    traceml.init(mode="auto")
    set_seed(args.seed + rank)

    train_loader, train_sampler = prepare_data(args, rank, world_size)
    model = MLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_classes=args.num_classes,
    ).to(device)
    traceml.trace_model_instance(model)

    if use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    extra_matmuls = (
        args.compute_extra_matmuls
        if args.scenario == "compute-straggler"
        else 0
    )
    optimizer = RankStragglerAdamW(
        model.parameters(),
        rank=rank,
        straggler_rank=args.straggler_rank,
        extra_matmuls=extra_matmuls,
        extra_dim=args.compute_extra_dim,
        device=device,
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    running_loss = 0.0

    if rank == 0:
        print(
            "[ddp-rank-straggler] "
            f"scenario={args.scenario} world_size={world_size} "
            f"batch_size={args.batch_size} hidden_dim={args.hidden_dim} "
            f"depth={args.depth} straggler_rank={args.straggler_rank}"
        )

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        for features, labels in train_loader:
            global_step += 1

            with traceml.trace_step(model.module):
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.detach())

            if (
                rank == 0
                and args.print_every > 0
                and global_step % args.print_every == 0
            ):
                print(
                    f"[ddp-rank-straggler] epoch={epoch + 1} "
                    f"step={global_step}/{total_steps} "
                    f"loss={running_loss / args.print_every:.4f}"
                )
                running_loss = 0.0

    if rank == 0:
        print("[ddp-rank-straggler] done")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
