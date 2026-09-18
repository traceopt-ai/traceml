"""Offline RF-DETR training fixture; also runnable under ``traceml run``.

The detector and data are deliberately tiny. RF-DETR's module, data module,
optimizer, EMA, and Lightning training loop remain real.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def forward(self, samples, targets=None):
        return {"scores": self.projection(samples.tensors.mean((2, 3)))}


class TinyCriterion(nn.Module):
    weight_dict = {"loss_ce": 1.0}

    def forward(self, outputs, targets):
        return {"loss_ce": (outputs["scores"] - 0.25).square().mean()}


class DetectionDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        return torch.full((3, 32, 32), (index + 1) / self.rows), {
            "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(index),
            "orig_size": torch.tensor([32, 32]),
            "size": torch.tensor([32, 32]),
            "area": torch.tensor([64.0]),
            "iscrowd": torch.tensor([0]),
        }


def postprocess(outputs, sizes):
    return [
        {
            "boxes": torch.tensor([[12.0, 12.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        }
        for _ in sizes
    ]


@contextmanager
def components(output_dir, *, accumulation=2, rows=8):
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule

    model_config = RFDETRNanoConfig(
        pretrain_weights=None, device="cpu", num_classes=2, compile=False
    )
    train_config = TrainConfig(
        dataset_dir=str(output_dir),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        grad_accum_steps=accumulation,
        num_workers=0,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        drop_path=0.0,
        tensorboard=False,
        use_ema=True,
        accelerator="cpu",
        progress_bar=None,
        seed=13,
    )
    with (
        patch(
            "rfdetr.training.module_model.build_model_from_config",
            side_effect=lambda *args: TinyDetector(),
        ),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            side_effect=lambda *args: (TinyCriterion(), postprocess),
        ),
        patch(
            "rfdetr.training.module_model.get_param_dict",
            side_effect=lambda args, model: [{"params": model.parameters()}],
        ),
        patch(
            "rfdetr.training.module_data.build_dataset",
            side_effect=lambda split, *args: DetectionDataset(
                rows if split == "train" else 2
            ),
        ),
    ):
        yield (
            RFDETRModelModule(model_config, train_config),
            RFDETRDataModule(model_config, train_config),
        )


def trainer(callbacks, *, accumulation=2, **kwargs):
    options = dict(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=accumulation,
        num_sanity_val_steps=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
    )
    options.update(kwargs)
    return pl.Trainer(**options)


def main():
    import rfdetr.training
    from rfdetr.training.callbacks.ema import RFDETREMACallback

    from traceml_ai.integrations import rfdetr as tracing

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(13)
    tracing.init()
    with components(args.output, rows=32) as (module, data):
        fit = rfdetr.training.build_trainer(
            module.train_config,
            module.model_config,
            callbacks=[RFDETREMACallback()],
            devices=2,
            strategy="ddp",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            num_sanity_val_steps=1,
            # RF-DETR versions differ in padding very small DDP datasets.
            # Fix the work count so the rank-coverage assertion is stable.
            limit_train_batches=8,
            limit_val_batches=1,
        )
        assert (
            sum(
                isinstance(callback, tracing._callback_class())
                for callback in fit.callbacks
            )
            == 1
        )
        fit.fit(module, datamodule=data)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank-{fit.global_rank}.json").write_text(
            json.dumps({"rank": fit.global_rank, "steps": fit.global_step})
        )


if __name__ == "__main__":
    main()
