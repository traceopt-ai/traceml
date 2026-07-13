"""Run one fixed BERT workload on one GPU and compare machines with TraceML.

This example is for the first hardware-comparison blog post:

1. Run this same script on one GPU machine.
2. Run it again on a different one GPU machine.
3. Compare the two ``final_summary.json`` files with ``traceml compare``.

Example:

    traceml run examples/advanced/bert_single_gpu_compare.py \\
        --mode=summary \\
        --run-name bert_l4_bs16_seq128 \\
        --args --batch-size 16 --max-length 128 --max-steps 200

Then compare two saved runs:

    traceml compare run_a/final_summary.json run_b/final_summary.json

The printed tokens/sec number is padded token throughput:

    batch_size * max_length / measured_step_time

Fixed-length padded tokens make the workload shape stable across machines. For
a fair comparison, keep the model, batch size, sequence length, precision, step
count, and software stack the same. Run once first to populate the Hugging Face
model and dataset cache, then use a fresh measured run for blog numbers.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import traceml_ai as traceml


SEED = 42
NUM_LABELS = 4
DEFAULT_DATASET_NAME = "fancyzhx/ag_news"
DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 128
DEFAULT_MAX_STEPS = 200
DEFAULT_MEASUREMENT_WARMUP_STEPS = 20
DEFAULT_NUM_SAMPLES = 12000
DEFAULT_NUM_WORKERS = 2
DEFAULT_LR = 2e-5
DEFAULT_PRINT_EVERY = 25


class MissingDependencyError(RuntimeError):
    """Raised when optional Hugging Face example dependencies are missing."""


@dataclass(frozen=True)
class RunConfig:
    """Workload knobs that should stay fixed across machines."""

    model_name: str
    dataset_name: str
    batch_size: int
    max_length: int
    max_steps: int
    measurement_warmup_steps: int
    num_samples: int
    num_workers: int
    precision: str
    learning_rate: float
    seed: int
    print_every: int


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args() -> RunConfig:
    """Read command-line options for the single-GPU comparison workload."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed BERT text-classification workload for single-GPU "
            "hardware comparison with TraceML."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size for this one GPU.",
    )
    parser.add_argument(
        "--max-length",
        type=positive_int,
        default=DEFAULT_MAX_LENGTH,
        help="Fixed padded sequence length.",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=DEFAULT_MAX_STEPS,
        help="Training steps to run, including measurement warmup.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=non_negative_int,
        default=DEFAULT_MEASUREMENT_WARMUP_STEPS,
        help="Initial steps excluded from the printed throughput estimate.",
    )
    parser.add_argument(
        "--num-samples",
        type=positive_int,
        default=DEFAULT_NUM_SAMPLES,
        help="Training examples to load from the dataset split.",
    )
    parser.add_argument(
        "--num-workers",
        type=non_negative_int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="Training precision. auto uses fp16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--print-every",
        type=positive_int,
        default=DEFAULT_PRINT_EVERY,
    )
    args = parser.parse_args()

    if args.warmup_steps >= args.max_steps:
        parser.error("--warmup-steps must be smaller than --max-steps.")

    return RunConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_steps=args.max_steps,
        measurement_warmup_steps=args.warmup_steps,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        precision=args.precision,
        learning_rate=args.learning_rate,
        seed=args.seed,
        print_every=args.print_every,
    )


def check_huggingface_dependencies() -> None:
    """Fail early with a clear message if optional HF packages are missing."""
    try:
        import datasets  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            "This example requires Hugging Face dependencies. Install them "
            "with: pip install datasets transformers"
        ) from exc


def set_seed(seed: int) -> None:
    """Make the workload repeatable enough for a simple comparison."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Use the first CUDA GPU when available; otherwise fall back to CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def autocast_settings(
    precision: str, device: torch.device
) -> tuple[bool, Optional[torch.dtype]]:
    """Return autocast settings for the requested precision."""
    if device.type != "cuda":
        if precision not in {"auto", "fp32"}:
            raise ValueError("fp16 and bf16 precision require CUDA.")
        return False, None

    if precision == "auto":
        precision = "fp16"
    if precision == "fp32":
        return False, None
    if precision == "fp16":
        return True, torch.float16
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError(
                "bf16 was requested, but this GPU does not support it."
            )
        return True, torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def build_dataloader(config: RunConfig, device: torch.device) -> DataLoader:
    """Tokenize AG News and build a fixed-shape training DataLoader."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    raw = load_dataset(config.dataset_name)
    train_raw = raw["train"].select(
        range(min(config.num_samples, len(raw["train"])))
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )

    train_ds = train_raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )
    train_ds = train_ds.rename_column("label", "labels")
    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a tokenized batch to the training device."""
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA so local wall-clock throughput is meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def print_run_header(config: RunConfig, device: torch.device) -> None:
    """Print the exact workload shape used for this run."""
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda or "unknown"
    else:
        device_name = "CPU"
        cuda_version = "not available"

    padded_tokens_per_step = config.batch_size * config.max_length

    print("[TraceML BERT single-GPU compare]")
    print(f"Device: {device_name}")
    print(f"PyTorch: {torch.__version__} | CUDA: {cuda_version}")
    print(
        "Workload: "
        f"model={config.model_name} "
        f"dataset={config.dataset_name} "
        f"precision={config.precision} "
        f"batch_size={config.batch_size} "
        f"max_length={config.max_length} "
        f"padded_tokens_per_step={padded_tokens_per_step} "
        f"max_steps={config.max_steps} "
        f"measurement_warmup_steps={config.measurement_warmup_steps} "
        f"num_workers={config.num_workers}"
    )


def train(config: RunConfig) -> None:
    """Run the fixed BERT training loop and print throughput numbers."""
    check_huggingface_dependencies()

    from transformers import (
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    set_seed(config.seed)
    device = select_device()
    use_autocast, autocast_dtype = autocast_settings(config.precision, device)

    # Initialize before building the DataLoader so TraceML can time input fetches.
    traceml.init(mode="auto")

    print_run_header(config, device)
    loader = build_dataloader(config, device)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=NUM_LABELS,
    ).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(config.max_steps * 0.06)),
        num_training_steps=config.max_steps,
    )
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(device.type == "cuda" and autocast_dtype == torch.float16),
    )

    measured_tokens = 0.0
    measured_samples = 0.0
    measured_seconds = 0.0
    running_loss = 0.0
    step = 0

    while step < config.max_steps:
        for batch in loader:
            if step >= config.max_steps:
                break

            sync_if_cuda(device)
            start_s = time.perf_counter()

            with traceml.trace_step(model):
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_autocast,
                    dtype=autocast_dtype,
                ):
                    out = model(**batch)
                    loss = out.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            sync_if_cuda(device)
            elapsed_s = time.perf_counter() - start_s

            step += 1
            if step > config.measurement_warmup_steps:
                measured_tokens += float(batch["input_ids"].numel())
                measured_samples += float(batch["labels"].shape[0])
                measured_seconds += elapsed_s

            running_loss += float(loss.detach())

            if step % config.print_every == 0:
                tokens_per_s = (
                    measured_tokens / measured_seconds
                    if measured_seconds > 0.0
                    else 0.0
                )
                samples_per_s = (
                    measured_samples / measured_seconds
                    if measured_seconds > 0.0
                    else 0.0
                )
                print(
                    f"step {step:04d}/{config.max_steps} "
                    f"| loss {running_loss / config.print_every:.4f} "
                    f"| padded_tok/s {tokens_per_s:,.0f} "
                    f"| samples/s {samples_per_s:,.1f}"
                )
                running_loss = 0.0

    measured_steps = config.max_steps - config.measurement_warmup_steps
    tokens_per_s = measured_tokens / max(measured_seconds, 1e-9)
    samples_per_s = measured_samples / max(measured_seconds, 1e-9)
    print(
        "Benchmark result: "
        f"measured_steps={measured_steps} "
        f"padded_tokens_per_second={tokens_per_s:,.0f} "
        f"samples_per_second={samples_per_s:,.1f}"
    )
    traceml.summary(print_text=True)


def main() -> None:
    """Entry point used by ``traceml run`` or direct Python execution."""
    config = parse_args()
    try:
        train(config)
    except MissingDependencyError as exc:
        print(
            f"[TraceML BERT single-GPU compare] ERROR: {exc}", file=sys.stderr
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
