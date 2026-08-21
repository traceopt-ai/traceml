"""Run one production-shaped TRL LoRA gradient-accumulation workload."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from traceml_ai.integrations import huggingface as traceml_hf


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
DATASET_SPLIT = "train_sft"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Qwen3-8B TRL LoRA configuration while holding the "
            "effective batch and token workload constant."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Physical per-device batch size.",
    )
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--dataset-samples",
        type=int,
        default=50_000,
        help="Conversation rows selected before TRL tokenization and packing.",
    )
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("logs/qwen3_8b_lora_ga/manual_runs"),
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def gradient_accumulation_steps(batch_size: int, effective_batch: int) -> int:
    if batch_size <= 0 or effective_batch <= 0:
        raise ValueError("batch_size and effective_batch must be positive.")
    if effective_batch % batch_size != 0:
        raise ValueError(
            f"effective_batch ({effective_batch}) must be divisible by "
            f"batch_size ({batch_size})."
        )
    return effective_batch // batch_size


def package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def environment_record() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    record: dict[str, object] = {
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
    }
    for distribution in [
        "torch",
        "transformers",
        "trl",
        "peft",
        "accelerate",
        "datasets",
        "traceml-ai",
    ]:
        record[distribution] = package_version(distribution)
    return record


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This workload requires one CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This workload requires a GPU with BF16 support.")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if args.max_length <= 0:
        raise ValueError("max_length must be positive.")
    if args.dataset_samples <= 0:
        raise ValueError("dataset_samples must be positive.")

    accumulation = gradient_accumulation_steps(
        args.batch_size, args.effective_batch
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or (
        f"bs{args.batch_size}_ga{accumulation}_{timestamp}"
    )
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = environment_record()
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
    )

    token_capacity = args.max_steps * args.effective_batch * args.max_length
    print("\nWorkload")
    print(f"  GPU:                     {environment['gpu']}")
    print(f"  Model:                   {MODEL_ID}")
    print(f"  Dataset:                 {DATASET_ID}/{DATASET_SPLIT}")
    print(f"  Physical batch:          {args.batch_size}")
    print(f"  Accumulation steps:      {accumulation}")
    print(f"  Effective batch:         {args.effective_batch}")
    print(f"  Packed sequence length:  {args.max_length}")
    print(f"  Optimizer steps:         {args.max_steps}")
    print(f"  Packed-token capacity:   {token_capacity:,}")
    print(f"  Source conversations:    {args.dataset_samples:,}")
    print(f"  Output:                   {output_dir}\n", flush=True)

    set_seed(SEED)
    dataset = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    if args.dataset_samples > len(dataset):
        raise ValueError(
            f"Requested {args.dataset_samples} rows, but {DATASET_ID}/"
            f"{DATASET_SPLIT} contains {len(dataset)}."
        )
    dataset = dataset.shuffle(seed=SEED).select(range(args.dataset_samples))
    if "messages" not in dataset.column_names:
        raise ValueError(
            f"Expected a 'messages' column in {DATASET_ID}/{DATASET_SPLIT}, "
            f"but found {dataset.column_names}."
        )
    # UltraChat also carries a metadata column named ``prompt``. TRL uses the
    # presence of that name to identify prompt-completion datasets, which must
    # also contain ``completion``. This workload uses the conversational
    # language-modeling format, so pass only its structured messages.
    dataset = dataset.select_columns(["messages"])

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        model_init_kwargs={
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "use_cache": False,
            "attn_implementation": "sdpa",
        },
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        packing=True,
        packing_strategy="bfd",
        assistant_only_loss=True,
        max_steps=args.max_steps,
        warmup_steps=min(20, max(1, args.max_steps // 10)),
        learning_rate=2e-4,
        optim="adamw_torch",
        logging_steps=max(1, min(25, args.max_steps // 10)),
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        bf16=True,
        fp16=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataset_num_proc=args.dataset_num_proc,
        seed=SEED,
        data_seed=SEED,
    )

    traceml_hf.init()
    trainer = SFTTrainer(
        model=MODEL_ID,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        callbacks=[traceml_hf.TraceMLTrainerCallback()],
    )

    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    reported_num_tokens = next(
        (
            entry["num_tokens"]
            for entry in reversed(trainer.state.log_history)
            if "num_tokens" in entry
        ),
        None,
    )
    metrics = {
        **train_result.metrics,
        "run_name": run_name,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": args.effective_batch,
        "max_length": args.max_length,
        "optimizer_steps": train_result.global_step,
        "packed_token_capacity": token_capacity,
        "reported_num_tokens": reported_num_tokens,
        "source_conversations": args.dataset_samples,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    metrics_path = output_dir / "trainer_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nTrainer result")
    print(f"  Runtime:                  {metrics['train_runtime']:.2f} s")
    print(
        f"  Optimizer steps/second:   "
        f"{metrics['train_steps_per_second']:.3f}"
    )
    print(
        f"  Samples/second:           "
        f"{metrics['train_samples_per_second']:.3f}"
    )
    if reported_num_tokens is not None:
        print(f"  Tokens reported by TRL:   {reported_num_tokens:,.0f}")
    print(
        f"  Peak reserved memory:     {metrics['peak_reserved_gib']:.2f} GiB"
    )
    print(f"  Metrics saved to:         {metrics_path}\n", flush=True)


if __name__ == "__main__":
    main()
