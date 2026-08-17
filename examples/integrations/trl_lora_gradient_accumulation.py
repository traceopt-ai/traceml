"""Compare LoRA microbatch layouts with one fixed effective batch size.

Designed for the companion Colab notebook. Each invocation runs one isolated
TRL ``SFTTrainer`` configuration so model state, CUDA peak counters, and
TraceML instrumentation cannot leak between comparison lanes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from traceml_ai.integrations import huggingface as traceml_hf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset-id", default="trl-lib/Capybara")
    parser.add_argument("--dataset-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--per-device-batch-size", type=int, required=True)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment requires a CUDA GPU. In Colab, select "
            "Runtime -> Change runtime type -> T4 GPU."
        )

    set_seed(args.seed)
    effective_batch = (
        args.per_device_batch_size * args.gradient_accumulation_steps
    )
    run_name = (
        f"bs{args.per_device_batch_size}_"
        f"ga{args.gradient_accumulation_steps}"
    )
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[experiment] run={run_name} gpu={torch.cuda.get_device_name(0)!r} "
        f"physical_batch={args.per_device_batch_size} "
        f"gradient_accumulation={args.gradient_accumulation_steps} "
        f"effective_batch={effective_batch} max_length={args.max_length} "
        f"optimizer_steps={args.max_steps}",
        flush=True,
    )

    dataset = load_dataset(args.dataset_id, split="train")
    sample_count = min(args.dataset_samples, len(dataset))
    dataset = dataset.shuffle(seed=args.seed).select(range(sample_count))

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        model_init_kwargs={"dtype": "float16", "use_cache": False},
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        # With max_length truncation, this pads every sequence to exactly the
        # same length and controls token work across microbatch layouts.
        pad_to_multiple_of=args.max_length,
        packing=False,
        max_steps=args.max_steps,
        warmup_steps=5,
        learning_rate=2e-4,
        optim="adamw_torch",
        logging_steps=5,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        fp16=True,
        bf16=False,
        dataloader_num_workers=0,
        dataset_num_proc=1,
        seed=args.seed,
        data_seed=args.seed,
    )

    # TraceML records one step per optimizer update. All forward/backward
    # calls from the accumulation group are summed into that step.
    traceml_hf.init()
    trainer = SFTTrainer(
        model=args.model_id,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[traceml_hf.TraceMLTrainerCallback()],
    )

    train_output = trainer.train()
    metrics = dict(train_output.metrics)
    metrics.update(
        {
            "run_name": run_name,
            "physical_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": (args.gradient_accumulation_steps),
            "effective_batch_size": effective_batch,
            "max_length": args.max_length,
            "optimizer_steps": args.max_steps,
            "gpu_name": torch.cuda.get_device_name(0),
        }
    )
    metrics_path = output_dir / "trainer_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[experiment] Trainer metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
