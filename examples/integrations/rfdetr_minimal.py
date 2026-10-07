"""Try RF-DETR Nano with sample data or your own COCO export.

Run from the repository root with::

    traceml run examples/integrations/rfdetr_minimal.py --args \\
      --demo --output-dir checkpoints/rfdetr-demo --epochs 1

The demo generates temporary images locally; RF-DETR may download pretrained
weights on first use. For real data, replace ``--demo`` with
``--dataset-dir data/coco``. Training follows RF-DETR's standard Python API:
https://rfdetr.roboflow.com/learn/train/
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument(
        "--demo",
        action="store_true",
        help="Generate temporary sample images; no dataset download needed",
    )
    data.add_argument(
        "--dataset-dir",
        type=Path,
        help="Roboflow COCO directory containing train/, valid/, and test/",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for RF-DETR checkpoints and training config",
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


def write_demo_dataset(root: Path) -> None:
    """Write small, deterministic COCO splits for a functional training demo."""
    from PIL import Image, ImageDraw

    size = 384
    for split, count in (("train", 32), ("valid", 4), ("test", 4)):
        folder = root / split
        folder.mkdir()
        images, annotations = [], []
        for index in range(count):
            x, y = 32 + (index * 17) % 192, 32 + (index * 29) % 192
            side = 96
            image = Image.new("RGB", (size, size), (40, 60, 80))
            ImageDraw.Draw(image).rectangle(
                (x, y, x + side - 1, y + side - 1), fill=(220, 160, 60)
            )
            filename = f"sample_{index:03d}.jpg"
            image.save(folder / filename)
            images.append(
                {
                    "id": index + 1,
                    "file_name": filename,
                    "width": size,
                    "height": size,
                }
            )
            annotations.append(
                {
                    "id": index + 1,
                    "image_id": index + 1,
                    "category_id": 1,
                    "bbox": [x, y, side, side],
                    "area": side * side,
                    "iscrowd": 0,
                }
            )
        (folder / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": [{"id": 1, "name": "rectangle"}],
                }
            ),
            encoding="utf-8",
        )


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

    # Each demo process owns identical temporary data, including under DDP.
    dataset_context = (
        TemporaryDirectory(prefix="traceml-rfdetr-")
        if args.demo
        else nullcontext(args.dataset_dir.expanduser().resolve())
    )
    with dataset_context as dataset_root:
        dataset_dir = Path(dataset_root)
        if args.demo:
            write_demo_dataset(dataset_dir)
            if rank == 0:
                print(
                    f"Sample data: {dataset_dir} (removed after training). "
                    "Synthetic demo only; timings are not benchmark evidence.",
                    flush=True,
                )
        for split in ("train", "valid", "test"):
            annotation = dataset_dir / split / "_annotations.coco.json"
            if not annotation.is_file():
                parser.error(f"missing COCO annotation file: {annotation}")

        # Keep --help and argument validation usable without the optional stack.
        import torch
        from rfdetr import RFDETRNano

        device = args.accelerator
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            parser.error(
                "--accelerator=cuda requires an available CUDA device"
            )

        output_dir = args.output_dir.expanduser().resolve()
        if rank == 0:
            # Only the writer checks/creates this path: other DDP ranks may start
            # after it exists. mkdir also prevents two runs sharing checkpoints.
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                parser.error(
                    "output directory already exists; choose a new path: "
                    f"{output_dir}"
                )

        torch.manual_seed(42)
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
