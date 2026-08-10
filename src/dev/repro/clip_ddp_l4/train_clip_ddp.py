# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Representative CLIP + Hugging Face Trainer + DDP reproduction.

This configurable workload is based on the training and launch scripts in
Hugging Face Transformers issue #41615:
https://github.com/huggingface/transformers/issues/41615

It follows the reported model, Trainer, optimizer, precision, DataLoader, DDP,
torch.compile, and attention configuration while adding explicit controls for
dataset source, hardware count, workers, compilation, and rank delay.

Use the sibling ``launch.sh`` so TraceML starts its aggregator before the DDP
workers. The default 4-GPU configuration keeps the reported global batch size
of 1024 by using two gradient-accumulation steps. On 8 GPUs it automatically
uses one, matching the issue.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

ISSUE_URL = "https://github.com/huggingface/transformers/issues/41615"
DEFAULT_TARGET_MODEL = "openai/clip-vit-base-patch32"


def is_main_process() -> bool:
    """Return whether this is global rank zero."""
    return int(os.environ.get("RANK", "0")) == 0


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable and frozen parameters, matching the issue helper."""
    return sum(parameter.numel() for parameter in model.parameters())


def package_version(name: str) -> str:
    """Return an installed package version without making it a dependency."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def rank_delay(delay_ms: float, delayed_rank: int) -> None:
    """Optionally delay one rank once per collated batch."""
    rank = int(os.environ.get("RANK", "0"))
    if delay_ms > 0.0 and rank == delayed_rank:
        time.sleep(delay_ms / 1000.0)


def _item_text(item: Mapping[str, Any], text_column: str = "text") -> str:
    """Read the text shapes used by the issue's CC, COCO, and LAION paths."""
    if text_column in item:
        return str(item[text_column])
    if "text" in item:
        return str(item["text"])
    if "txt" in item:
        return str(item["txt"])
    if "conversations" in item:
        conversations = item["conversations"]
        if len(conversations) > 1:
            return str(conversations[1]["value"])
    raise KeyError(
        "Could not find text. Configure --text-column or provide one of "
        "text, txt, or conversations."
    )


def _item_image(item: Mapping[str, Any], image_column: str = "image") -> Any:
    """Read the image shapes used by the issue's CC, COCO, and LAION paths."""
    if image_column in item:
        return item[image_column]
    if "image" in item:
        return item["image"]
    if "jpg" in item:
        return item["jpg"]
    raise KeyError(
        "Could not find an image. Configure --image-column or provide image "
        "or jpg."
    )


def collate_fn(
    batch: Sequence[Mapping[str, Any]],
    processor: Any,
    *,
    image_column: str = "image",
    text_column: str = "text",
    max_length: int = 77,
) -> dict[str, Any]:
    """Process raw text and images, following the issue's collator."""
    texts = [_item_text(item, text_column) for item in batch]
    images = [_item_image(item, image_column) for item in batch]
    output = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    # CLIPModel only calculates the contrastive training loss when requested.
    # Set it explicitly so the plain Trainer receives a scalar training loss.
    output["return_loss"] = True
    return dict(output)


def collate_fn_laion(
    batch: Sequence[Mapping[str, Any]],
    processor: Any,
    *,
    max_length: int = 77,
) -> dict[str, Any]:
    """Preserve the issue's LAION ``txt``/``jpg`` collator shape."""
    return collate_fn(
        batch,
        processor,
        image_column="jpg",
        text_column="txt",
        max_length=max_length,
    )


class RawClipCollator:
    """Pickle-friendly raw-input collator with an optional rank delay."""

    def __init__(
        self,
        processor: Any,
        *,
        image_column: str,
        text_column: str,
        max_length: int,
        rank_delay_ms: float,
        delayed_rank: int,
    ) -> None:
        self.processor = processor
        self.image_column = image_column
        self.text_column = text_column
        self.max_length = max_length
        self.rank_delay_ms = rank_delay_ms
        self.delayed_rank = delayed_rank

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rank_delay(self.rank_delay_ms, self.delayed_rank)
        return collate_fn(
            batch,
            self.processor,
            image_column=self.image_column,
            text_column=self.text_column,
            max_length=self.max_length,
        )


class PreprocessedClipCollator:
    """Stack already-tokenized synthetic samples with minimal CPU work."""

    def __init__(self, *, rank_delay_ms: float, delayed_rank: int) -> None:
        self.rank_delay_ms = rank_delay_ms
        self.delayed_rank = delayed_rank

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> dict[str, Any]:
        rank_delay(self.rank_delay_ms, self.delayed_rank)
        output = {
            key: torch.stack([sample[key] for sample in batch])
            for key in ("pixel_values", "input_ids", "attention_mask")
        }
        output["return_loss"] = True
        return output


class GeneratedRawClipDataset(Dataset[dict[str, Any]]):
    """Generate deterministic PIL images and captions without a download."""

    def __init__(
        self,
        num_samples: int,
        image_size: int,
        seed: int,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        generator = np.random.default_rng(self.seed + index)
        pixels = generator.integers(
            0,
            256,
            size=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        return {
            "image": Image.fromarray(pixels),
            "text": f"a generated training image with sample id {index}",
        }


class PreprocessedClipDataset(Dataset[dict[str, torch.Tensor]]):
    """Return fixed CLIP-shaped CPU tensors to isolate model-side behavior."""

    def __init__(
        self,
        num_samples: int,
        image_size: int,
        max_length: int,
        vocab_size: int,
        seed: int,
    ) -> None:
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed)
        self.pixel_values = torch.rand(
            3,
            image_size,
            image_size,
            generator=generator,
            dtype=torch.float32,
        )
        self.input_ids = torch.randint(
            0,
            vocab_size,
            (max_length,),
            generator=generator,
            dtype=torch.long,
        )
        self.attention_mask = torch.ones(max_length, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        return {
            "pixel_values": self.pixel_values,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }


class JsonlImageTextDataset(Dataset[dict[str, Any]]):
    """Load a CC/LAION-like local JSONL manifest and decode images lazily."""

    _IMAGE_KEYS = ("image", "jpg", "file_name", "path")

    def __init__(
        self,
        metadata_file: Path,
        image_root: Path,
        num_samples: int,
        text_column: str,
    ) -> None:
        self.image_root = image_root
        self.text_column = text_column
        self.records: list[dict[str, Any]] = []
        with metadata_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.records.append(json.loads(line))
                if 0 < num_samples <= len(self.records):
                    break
        if not self.records:
            raise ValueError(f"No records found in {metadata_file}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        record = self.records[index]
        relative_path = next(
            (record[key] for key in self._IMAGE_KEYS if key in record),
            None,
        )
        if relative_path is None:
            raise KeyError(
                f"Record {index} has no image path in {self._IMAGE_KEYS}"
            )
        image_path = Path(str(relative_path))
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        return {
            "image": rgb_image,
            "text": _item_text(record, self.text_column),
        }


class ColumnMappedDataset(Dataset[dict[str, Any]]):
    """Normalize arbitrary Hugging Face image/text column names."""

    def __init__(
        self,
        dataset: Any,
        *,
        image_column: str,
        text_column: str,
    ) -> None:
        self.dataset = dataset
        self.image_column = image_column
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        return {
            "image": item[self.image_column],
            "text": item[self.text_column],
        }


def load_huggingface_dataset(args: argparse.Namespace) -> Dataset[Any]:
    """Load ImageFolder or a named non-streaming Hugging Face dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "This dataset source requires `datasets`. Install the project's "
            "development or Hugging Face dependencies."
        ) from exc

    if args.dataset_source == "imagefolder":
        if not args.data_path:
            raise ValueError("--data-path is required for imagefolder")
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.data_path,
            split=args.dataset_split,
        )
    else:
        if not args.dataset_name:
            raise ValueError(
                "--dataset-name is required for --dataset-source hf"
            )
        positional = [args.dataset_name]
        if args.dataset_config:
            positional.append(args.dataset_config)
        dataset = load_dataset(*positional, split=args.dataset_split)

    if 0 < args.num_data < len(dataset):
        dataset = dataset.select(range(args.num_data))
    return ColumnMappedDataset(
        dataset,
        image_column=args.image_column,
        text_column=args.text_column,
    )


def build_dataset_and_collator(
    args: argparse.Namespace,
    config: Any,
    processor: Any | None,
) -> tuple[Dataset[Any], Any, str]:
    """Create one of the controlled input-pipeline variants."""
    image_size = int(config.vision_config.image_size)
    vocab_size = int(config.text_config.vocab_size)

    if args.dataset_source == "preprocessed":
        dataset = PreprocessedClipDataset(
            args.num_data,
            image_size,
            args.max_length,
            vocab_size,
            args.seed,
        )
        collator = PreprocessedClipCollator(
            rank_delay_ms=args.rank_delay_ms,
            delayed_rank=args.delayed_rank,
        )
        return dataset, collator, "preprocessed"

    if processor is None:
        raise RuntimeError("A CLIPProcessor is required for raw input modes")

    if args.dataset_source == "generated":
        dataset = GeneratedRawClipDataset(
            args.num_data,
            image_size,
            args.seed,
        )
        dataset_name = "generated"
    elif args.dataset_source in {"imagefolder", "hf"}:
        dataset = load_huggingface_dataset(args)
        dataset_name = (
            "imagefolder"
            if args.dataset_source == "imagefolder"
            else args.dataset_name.replace("/", "-")
        )
    elif args.dataset_source == "jsonl":
        if not args.metadata_file or not args.data_path:
            raise ValueError(
                "--metadata-file and --data-path are required for jsonl"
            )
        dataset = JsonlImageTextDataset(
            Path(args.metadata_file),
            Path(args.data_path),
            args.num_data,
            args.text_column,
        )
        dataset_name = "jsonl"
    else:  # pragma: no cover - argparse enforces the choices.
        raise ValueError(f"Unknown dataset source: {args.dataset_source}")

    collator = RawClipCollator(
        processor,
        image_column="image",
        text_column="text",
        max_length=args.max_length,
        rank_delay_ms=args.rank_delay_ms,
        delayed_rank=args.delayed_rank,
    )
    return dataset, collator, dataset_name


def resolve_gradient_accumulation(
    args: argparse.Namespace,
    world_size: int,
) -> int:
    """Preserve the issue's target global batch across 4 or 8 GPUs."""
    if args.gradient_accumulation_steps > 0:
        return args.gradient_accumulation_steps

    data_parallel_batch = args.per_device_train_batch_size * world_size
    if args.target_global_batch_size % data_parallel_batch != 0:
        raise ValueError(
            "target global batch size must be divisible by per-device batch "
            f"times world size: {args.target_global_batch_size} % "
            f"{data_parallel_batch} != 0. Set "
            "--gradient-accumulation-steps explicitly or change the batch."
        )
    return args.target_global_batch_size // data_parallel_batch


def precision_flags(precision: str) -> tuple[bool, bool]:
    """Return the fp16 and bf16 TrainingArguments flags."""
    return precision == "fp16", precision == "bf16"


def configure_device(local_rank: int) -> None:
    """Bind each DDP process to its local CUDA device before setup."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def print_environment(args: argparse.Namespace, world_size: int) -> None:
    """Print enough environment metadata to interpret a reproduction."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(local_rank)
    print(
        f"[Rank {rank}] world_size={world_size} local_rank={local_rank} "
        f"device={device_name}",
        flush=True,
    )
    if not is_main_process():
        return
    print("==== Reproduction environment ====")
    print(f"Source issue: {ISSUE_URL}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {package_version('transformers')}")
    print(f"Accelerate: {package_version('accelerate')}")
    print(f"Datasets: {package_version('datasets')}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print("==== Script arguments ====")
    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}")


def train(args: argparse.Namespace) -> None:
    """Run the issue-inspired CLIP training workload."""
    try:
        from transformers import (
            CLIPConfig,
            CLIPModel,
            CLIPProcessor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the Hugging Face dependencies before running this "
            "reproduction: pip install -e '.[hf]'"
        ) from exc

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    configure_device(local_rank)
    print_environment(args, world_size)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    config = CLIPConfig.from_pretrained(
        args.target_model,
        _attn_implementation=args.attention_impl,
    )
    model = CLIPModel(config=config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    processor = None
    if args.dataset_source != "preprocessed":
        processor = CLIPProcessor.from_pretrained(
            args.target_model,
            use_fast=args.use_fast_processor,
        )

    train_dataset, data_collator, dataset_name = build_dataset_and_collator(
        args,
        config,
        processor,
    )
    gradient_accumulation = resolve_gradient_accumulation(args, world_size)
    effective_global_batch = (
        args.per_device_train_batch_size * world_size * gradient_accumulation
    )

    if is_main_process():
        print(f"Model parameters: {count_parameters(model):,}")
        print(f"Dataset: {dataset_name} ({len(train_dataset):,} samples)")
        print(f"Target global batch: {args.target_global_batch_size}")
        print(f"Effective global batch: {effective_global_batch}")
        print(f"Per-device batch: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {gradient_accumulation}")

    fp16, bf16 = precision_flags(args.precision)
    persistent_workers = (
        args.dataloader_persistent_workers and args.num_workers > 0
    )
    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None
    save_strategy = "steps" if args.save_steps > 0 else "no"
    report_to: str | list[str] = (
        args.report_to if args.report_to != "none" else []
    )

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        run_name=args.run_name or f"{dataset_name}-clip-ddp",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        fp16=fp16,
        bf16=bf16,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy=save_strategy,
        save_steps=max(args.save_steps, 1),
        save_total_limit=3,
        load_best_model_at_end=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_prefetch_factor=prefetch_factor,
        dataloader_persistent_workers=persistent_workers,
        ddp_backend=(
            "nccl"
            if world_size > 1 and torch.cuda.is_available()
            else "gloo" if world_size > 1 else None
        ),
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
        # torchrun supplies LOCAL_RANK=0 even for rank zero; direct smoke runs
        # do not initialize a process group and therefore need the HF sentinel.
        local_rank=local_rank if world_size > 1 else -1,
        seed=args.seed,
        data_seed=args.seed,
        report_to=report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        save_on_each_node=False,
        torch_compile=args.torch_compile,
        torch_compile_backend=args.torch_compile_backend,
        torch_compile_mode=args.torch_compile_mode,
        tf32=args.tf32,
        ddp_broadcast_buffers=False,
        disable_tqdm=args.disable_tqdm,
    )

    callbacks: list[Any] = []
    if os.environ.get("TRACEML_DISABLED") != "1":
        from traceml_ai.integrations import huggingface as traceml_hf

        traceml_hf.init()
        callbacks.append(traceml_hf.TraceMLTrainerCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    model.train()
    trainer.train(resume_from_checkpoint=args.resume or None)

    if is_main_process() and args.save_final_model:
        final_path = Path(args.save_dir) / args.save_name
        trainer.save_model(str(final_path))
        print(f"Final model saved to: {final_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the configurable argument parser for this reproduction."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the CLIP/DDP utilization reproduction inspired by "
            "Transformers issue #41615."
        )
    )
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument(
        "--dataset-source",
        choices=("generated", "preprocessed", "imagefolder", "hf", "jsonl"),
        default="generated",
    )
    parser.add_argument("--data-path", default="")
    parser.add_argument("--metadata-file", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--num-data", type=int, default=100_000)
    parser.add_argument("--max-length", type=int, default=77)

    parser.add_argument("--per-device-train-batch-size", type=int, default=128)
    parser.add_argument("--target-global-batch-size", type=int, default=1024)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=0,
        help="0 computes it from target global batch and world size.",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=4.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=100)

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--dataloader-persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--attention-impl",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-mode", default="default")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-fast-processor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=25)

    parser.add_argument("--rank-delay-ms", type=float, default=0.0)
    parser.add_argument("--delayed-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--save-dir", default="checkpoints/clip_ddp_l4")
    parser.add_argument("--save-name", default="final_model")
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument(
        "--save-final-model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--report-to", default="none")
    parser.add_argument(
        "--disable-tqdm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject configurations that would make results misleading."""
    positive = {
        "num_data": args.num_data,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "target_global_batch_size": args.target_global_batch_size,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    if args.rank_delay_ms < 0:
        raise ValueError("--rank-delay-ms cannot be negative")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        train(args)
    except (ImportError, KeyError, RuntimeError, ValueError) as exc:
        print(f"[clip-ddp-repro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
