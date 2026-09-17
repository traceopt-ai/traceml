"""Train RF-DETR Nano and compare DataLoader settings with TraceML."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Roboflow COCO directory containing train/, valid/, and test/",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for RF-DETR checkpoints and training configuration",
    )
    parser.add_argument("--epochs", type=_positive_int, default=2)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto selects CUDA when available, otherwise CPU",
    )
    return parser


def launch_topology(env: Mapping[str, str]) -> tuple[int, int, int]:
    """Return devices per node, node count, and global rank from torchrun."""
    world_size = int(env.get("WORLD_SIZE", "1"))
    devices = int(env.get("LOCAL_WORLD_SIZE", "1"))
    if devices < 1 or world_size < 1 or world_size % devices:
        raise ValueError(
            "WORLD_SIZE must be a positive multiple of LOCAL_WORLD_SIZE"
        )
    num_nodes = int(env.get("TRACEML_NNODES", str(world_size // devices)))
    if num_nodes < 1 or num_nodes * devices != world_size:
        raise ValueError(
            "TRACEML_NNODES * LOCAL_WORLD_SIZE must equal WORLD_SIZE"
        )
    rank = int(env.get("RANK", "0"))
    if not 0 <= rank < world_size:
        raise ValueError("RANK must be between 0 and WORLD_SIZE - 1")
    return devices, num_nodes, rank


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        devices, num_nodes, rank = launch_topology(os.environ)
    except ValueError as exc:
        parser.error(str(exc))

    dataset_dir = args.dataset_dir.expanduser().resolve()
    for split in ("train", "valid", "test"):
        annotation = dataset_dir / split / "_annotations.coco.json"
        if not annotation.is_file():
            parser.error(f"missing COCO annotation file: {annotation}")

    # Keep --help and argument validation usable without the optional stack.
    import torch
    from rfdetr import RFDETRNano

    from traceml_ai.integrations import rfdetr as traceml_rfdetr

    device = args.accelerator
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--accelerator=cuda requires an available CUDA device")

    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        # Only the writer checks/creates this path: other DDP ranks may start
        # after it exists. mkdir also prevents two runs sharing checkpoints.
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(
                f"output directory already exists; choose a new path: {output_dir}"
            )

    torch.manual_seed(42)
    traceml_rfdetr.init()  # Every rank initializes its own instrumentation.
    model = RFDETRNano(device=device, resolution=384, compile=False)
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        devices=devices,
        num_nodes=num_nodes,
        strategy="ddp" if devices * num_nodes > 1 else "auto",
        seed=42,
        grad_accum_steps=1,
        multi_scale=False,
        expanded_scales=False,
    )


if __name__ == "__main__":
    main()
