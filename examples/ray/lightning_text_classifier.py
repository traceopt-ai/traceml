"""TraceML + Ray Train + PyTorch Lightning text-classifier demo.

This follows Ray's Lightning TorchTrainer pattern while adding TraceML's Ray
runtime wrapper and Lightning callback:

    python examples/ray/lightning_text_classifier.py --ray-address=auto

On a two-node Ray cluster with one GPU per node:

    python examples/ray/lightning_text_classifier.py \\
      --ray-address=auto \\
      --use-gpu \\
      --num-workers=2 \\
      --max-steps=150

The dataset is synthetic and fixed-size so the example runs without downloading
CoLA, Hugging Face models, or tokenizers. It still exercises the same important
systems as the Ray example: Ray Data shards, Lightning DDP, and TraceML
telemetry across Ray workers.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

import numpy as np


def build_synthetic_text_dataset(
    *,
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
):
    """Create a Ray Dataset with fixed-size token ids and binary labels."""
    import ray

    rng = np.random.default_rng(seed)
    input_ids = rng.integers(
        low=0,
        high=vocab_size,
        size=(num_samples, seq_len),
        dtype=np.int64,
    )
    labels = (
        (input_ids[:, : seq_len // 2].sum(axis=1) % 2)
        ^ (input_ids[:, seq_len // 2 :].sum(axis=1) % 2)
    ).astype(np.int64)

    rows = [
        {"input_ids": input_ids[idx], "labels": labels[idx]}
        for idx in range(num_samples)
    ]
    return ray.data.from_items(rows)


def train_loop_per_worker(config: Dict[str, Any]) -> None:
    import lightning.pytorch as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import ray.train
    import traceml_ai as traceml
    from ray.train.lightning import (
        RayDDPStrategy,
        RayLightningEnvironment,
        prepare_trainer,
    )

    from traceml_ai.integrations.lightning import TraceMLCallback

    class TextClassifier(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.save_hyperparameters(dict(config))
            vocab_size = int(config["vocab_size"])
            embed_dim = int(config["embed_dim"])
            hidden_dim = int(config["hidden_dim"])
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            pooled = self.embedding(input_ids).mean(dim=1)
            return self.net(pooled)

        def training_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
        ):
            delay_rank = int(config.get("delay_rank", -1))
            delay_ms = float(config.get("delay_ms", 0.0))
            rank = int(os.environ.get("RANK", "0"))
            if delay_ms > 0 and rank == delay_rank:
                time.sleep(delay_ms / 1000.0)

            logits = self(batch["input_ids"].long())
            loss = F.cross_entropy(logits, batch["labels"].long())
            acc = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            self.log("train_acc", acc, prog_bar=True, sync_dist=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.parameters(),
                lr=float(config["lr"]),
            )

    train_ds = ray.train.get_dataset_shard("train")
    train_loader = train_ds.iter_torch_batches(
        batch_size=int(config["batch_size"]),
        prefetch_batches=1,
    )

    input_delay_s = float(config.get("input_delay_ms", 0.0)) / 1000.0
    if input_delay_s > 0.0:
        train_loader = _delay_batches(train_loader, input_delay_s)

    train_loader = traceml.wrap_dataloader_fetch(train_loader)

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_steps=int(config["max_steps"]),
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[TraceMLCallback()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(TextClassifier(), train_dataloaders=train_loader)


def _delay_batches(iterator, delay_s: float):
    for batch in iterator:
        time.sleep(delay_s)
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--num-samples", type=int, default=65536)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-delay-ms",
        type=float,
        default=0.0,
        help="Optional per-batch delay to make Ray input timing visible.",
    )
    parser.add_argument(
        "--delay-rank",
        type=int,
        default=-1,
        help="Optional global rank to slow down for straggler demos.",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=0.0,
        help="Optional sleep injected into training_step on --delay-rank.",
    )
    args = parser.parse_args()

    import ray
    from ray.train import RunConfig, ScalingConfig

    from traceml_ai.integrations.ray import (
        TraceMLRayConfig,
        TraceMLTorchTrainer,
    )

    ray.init(address=args.ray_address)

    train_dataset = build_synthetic_text_dataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )

    trainer = TraceMLTorchTrainer(
        train_loop_per_worker,
        train_loop_config={
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "vocab_size": args.vocab_size,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "input_delay_ms": args.input_delay_ms,
            "delay_rank": args.delay_rank,
            "delay_ms": args.delay_ms,
        },
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
        ),
        run_config=RunConfig(name="traceml-ray-lightning-text"),
        datasets={"train": train_dataset},
        traceml_config=TraceMLRayConfig(
            mode="summary",
            init_mode="selective",
            patch_dataloader=True,
            patch_h2d=True,
            session_id="ray-lightning-text",
        ),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
