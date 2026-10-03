"""Generate a deterministic COCO dataset with large PNG or BMP images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-format", choices=("png", "bmp"), default="png"
    )
    parser.add_argument("--image-size", type=int, default=4096)
    parser.add_argument("--train-images", type=int, default=32)
    parser.add_argument("--val-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1544)
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_image(size: int, image_id: int, seed: int) -> Image.Image:
    """Create compressible but nonuniform pixels without external inputs."""
    rng = random.Random((seed << 32) + image_id)
    background = tuple(rng.randrange(24, 208) for _ in range(3))
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    stripe = max(32, size // 32)
    for offset in range(0, size, stripe * 2):
        color = tuple((channel + 37) % 256 for channel in background)
        draw.rectangle(
            (0, offset, size, min(size, offset + stripe)), fill=color
        )
    margin = size // 5
    box = (margin, margin, size - margin, size - margin)
    accent = tuple(255 - channel for channel in background)
    draw.rectangle(box, outline=accent, width=max(4, size // 256))
    draw.line((0, 0, size, size), fill=accent, width=max(2, size // 512))
    return image


def coco_payload(split: str, count: int, size: int, extension: str) -> dict:
    images = []
    annotations = []
    margin = size // 5
    side = size - 2 * margin
    for index in range(1, count + 1):
        image_id = index if split == "train2017" else 1_000_000 + index
        images.append(
            {
                "id": image_id,
                "file_name": f"{image_id:012d}.{extension}",
                "width": size,
                "height": size,
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [margin, margin, side, side],
                "area": side * side,
                "iscrowd": 0,
            }
        )
    return {
        "info": {
            "description": "Deterministic non-JPEG RF-DETR timing fixture",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
    }


def generate(
    output: Path,
    *,
    image_format: str,
    image_size: int,
    train_images: int,
    val_images: int,
    seed: int,
) -> dict:
    if image_size < 384:
        raise ValueError(
            "image-size must be at least the 384-pixel model input"
        )
    if train_images < 4 or val_images < 1:
        raise ValueError(
            "require at least four training and one validation image"
        )
    if output.exists():
        raise FileExistsError(
            f"refusing to replace existing dataset: {output}"
        )

    annotations_dir = output / "annotations"
    annotations_dir.mkdir(parents=True)
    files = []
    for split, count in (("train2017", train_images), ("val2017", val_images)):
        split_dir = output / split
        split_dir.mkdir()
        payload = coco_payload(split, count, image_size, image_format)
        for row in payload["images"]:
            path = split_dir / row["file_name"]
            image = make_image(image_size, row["id"], seed)
            image.save(path, format=image_format.upper())
            files.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        annotation_path = annotations_dir / f"instances_{split}.json"
        annotation_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files.append(
            {
                "path": str(annotation_path.relative_to(output)),
                "bytes": annotation_path.stat().st_size,
                "sha256": sha256(annotation_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "generator": {
            "image_format": image_format,
            "image_size": image_size,
            "train_images": train_images,
            "val_images": val_images,
            "seed": seed,
        },
        "files": sorted(files, key=lambda row: row["path"]),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        manifest = generate(
            args.output_dir.expanduser().resolve(),
            image_format=args.image_format,
            image_size=args.image_size,
            train_images=args.train_images,
            val_images=args.val_images,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"dataset generation failed: {exc}") from exc
    total = sum(row["bytes"] for row in manifest["files"])
    print(
        f"Generated {args.train_images + args.val_images} {args.image_format.upper()} "
        f"images in {args.output_dir} ({total / 1024**2:.1f} MiB including annotations)."
    )


if __name__ == "__main__":
    main()
