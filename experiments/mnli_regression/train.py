# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune BERT on MNLI with the affected or fixed Transformers collator.

Trainer owns training, DDP, logging and evaluation. Use ``traceml run`` to
collect telemetry without adding instrumentation to this workload.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from contextlib import nullcontext
from importlib import metadata
from pathlib import Path

MODEL = "google-bert/bert-base-cased"
MODEL_REVISION = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
DATASET = "nyu-mll/glue"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
TRANSFORMERS_REVISIONS = {
    "affected": "50e35cc0ca18ec19b80653690de553e6decf74f5",
    "fixed": "4ec6e54f0457942a89928097d47e220fcb635450",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=TRANSFORMERS_REVISIONS, required=True
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-to", choices=("none", "trackio"), default="none"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Cache model/data; do not train.",
    )
    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be greater than 0 and at most 1")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        parser.error("--epochs must be finite and positive")
    if min(args.batch_size, args.max_length) < 1 or args.num_workers < 0:
        parser.error(
            "batch size/length must be positive; workers cannot be negative"
        )
    if not args.prepare_only and args.output_dir is None:
        parser.error("--output-dir is required for training")

    # Check the actual installation, not just the label given to the run.
    distribution = metadata.distribution("transformers")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = origin.get("vcs_info", {}).get("commit_id")
    if commit != TRANSFORMERS_REVISIONS[args.variant]:
        parser.error(
            f"Install the pinned {args.variant} Transformers revision from README.md; "
            f"installed commit is {commit!r}."
        )

    import numpy as np
    import torch
    from datasets import DatasetDict, load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not args.prepare_only and not torch.cuda.is_available():
        parser.error(
            "Training requires a CUDA GPU; --prepare-only can run on CPU."
        )

    def tokenize(batch):
        # Leave examples unpadded: the collator under investigation pads batches.
        encoded = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=args.max_length,
        )
        encoded["labels"] = batch["label"]
        return encoded

    training_args = (
        None
        if args.prepare_only
        else TrainingArguments(
            output_dir=str(args.output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=2e-5,
            seed=args.seed,
            data_seed=args.seed,
            fp16=True,
            bf16=False,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=args.num_workers,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            logging_first_step=True,
            report_to=args.report_to,
            run_name=args.output_dir.parent.name,
            disable_tqdm=True,
        )
    )

    set_seed(args.seed)
    # Prepare once per node; the other local ranks reuse the Arrow cache.
    preparation = (
        training_args.main_process_first(local=True, desc="MNLI preparation")
        if training_args is not None
        else nullcontext()
    )
    with preparation:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=MODEL_REVISION
        )
        raw = load_dataset(DATASET, "mnli", revision=DATASET_REVISION)
        count = max(1, int(len(raw["train"]) * args.train_fraction))
        selected = DatasetDict(
            train=raw["train"].shuffle(seed=args.seed).select(range(count)),
            matched=raw["validation_matched"],
            mismatched=raw["validation_mismatched"],
        )
        encoded = selected.map(
            tokenize, batched=True, remove_columns=raw["train"].column_names
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=MODEL_REVISION, num_labels=3
    )
    if args.prepare_only:
        print(
            f"Cached {count} training examples, both validation splits and {MODEL}."
        )
        return

    def accuracy(prediction):
        return {
            "accuracy": float(
                np.mean(
                    prediction.predictions.argmax(axis=-1)
                    == prediction.label_ids
                )
            )
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset={
            name: encoded[name] for name in ("matched", "mismatched")
        },
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=accuracy,
    )
    if trainer.is_world_process_zero():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        experiment = {
            "variant": args.variant,
            "transformers_commit": commit,
            "model": {"name": MODEL, "revision": MODEL_REVISION},
            "dataset": {"name": DATASET, "revision": DATASET_REVISION},
            "samples": {name: len(split) for name, split in selected.items()},
            "fingerprints": {
                name: split._fingerprint for name, split in selected.items()
            },
            "world_size": training_args.world_size,
            "arguments": {**vars(args), "output_dir": str(args.output_dir)},
            "training_arguments": training_args.to_dict(),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "node_0_gpus": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ],
                "traceml": metadata.version("traceml-ai"),
            },
        }
        (args.output_dir / "experiment.json").write_text(
            json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
        )
        with (args.output_dir / "installed-packages.txt").open("w") as output:
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                stdout=output,
                check=True,
            )

    result = trainer.train()
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model()  # Final model only; no repeated checkpoint writes.


if __name__ == "__main__":
    main()
