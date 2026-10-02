"""Run one bounded RF-DETR release-regression measurement."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SUPPORTED_VERSIONS = ("1.10.1", "1.11.0", "1.11.1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rfdetr-version", required=True)
    parser.add_argument("--rfdetr-wheel-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--precision", choices=("auto", "fp16", "bf16"), default="auto"
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1544)
    return parser


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def command(*args: str) -> str:
    return subprocess.check_output(
        args, text=True, stderr=subprocess.PIPE
    ).strip()


def source_identity(module, distribution: str) -> dict:
    """Record a Git source identity or the installed distribution version."""
    source = Path(module.__file__).resolve()
    for root in source.parents:
        if not (root / ".git").exists():
            continue
        try:
            command(
                "git",
                "-C",
                str(root),
                "ls-files",
                "--error-unmatch",
                str(source.relative_to(root)),
            )
        except subprocess.CalledProcessError:
            break
        diff = command("git", "-C", str(root), "diff", "HEAD", "--", ".")
        return {
            "commit": command("git", "-C", str(root), "rev-parse", "HEAD"),
            "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "dirty": bool(diff),
        }
    return {
        "version": importlib.metadata.version(distribution),
        "commit": None,
        "dirty": None,
    }


def package_tree_sha256(module) -> str:
    """Fingerprint installed Python sources without cache or metadata files."""
    root = Path(module.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dependency_fingerprint(frozen: str) -> str:
    """Hash the shared environment while excluding the intentional RF-DETR delta."""
    controlled = []
    for line in frozen.splitlines():
        normalized = line.strip().lower()
        if normalized.startswith("rfdetr==") or normalized.startswith(
            "rfdetr @"
        ):
            continue
        controlled.append(line.strip())
    payload = "\n".join(sorted(controlled)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def cpu_description() -> str:
    try:
        rows = json.loads(command("lscpu", "--json"))["lscpu"]
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ):
        return platform.processor() or "unknown"
    fields = {
        "Architecture:",
        "CPU(s):",
        "On-line CPU(s) list:",
        "Vendor ID:",
        "Model name:",
        "Thread(s) per core:",
        "Core(s) per socket:",
        "Socket(s):",
        "NUMA node(s):",
    }
    return "\n".join(
        f"{row['field']} {row['data']}"
        for row in rows
        if row["field"] in fields
    )


def validate_dataset(dataset: Path) -> tuple[dict, str]:
    manifest_path = dataset / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"missing or invalid dataset manifest: {manifest_path}"
        ) from exc
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported dataset manifest schema")
    image_format = manifest.get("generator", {}).get("image_format")
    if image_format not in {"png", "bmp"}:
        raise ValueError("case study requires a PNG or BMP dataset")
    for row in manifest.get("files", []):
        path = dataset / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"dataset file differs from manifest: {path}")
    for split in ("train2017", "val2017"):
        if (
            not (dataset / split).is_dir()
            or not (
                dataset / "annotations" / f"instances_{split}.json"
            ).is_file()
        ):
            raise ValueError(f"dataset is missing the COCO {split} split")
    return manifest, sha256(manifest_path)


def make_window_callback(warmup_steps: int, steps: int):
    """Time one complete post-warm-up window, including input loading."""
    import torch
    from pytorch_lightning import Callback

    class Window(Callback):
        def __init__(self):
            self.started = None
            self.elapsed_s = None
            self.completed_steps = 0
            self.last_loss = None

        @staticmethod
        def synchronize(module) -> None:
            if module.device.type == "cuda":
                torch.cuda.synchronize(module.device)

        def start(self, trainer, module) -> None:
            self.synchronize(module)
            trainer.strategy.barrier()
            self.started = time.perf_counter()

        def on_train_start(self, trainer, pl_module) -> None:
            if warmup_steps == 0:
                self.start(trainer, pl_module)

        def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx
        ) -> None:
            step = int(trainer.global_step)
            if step != self.completed_steps + 1:
                raise RuntimeError(
                    "expected exactly one optimizer-step attempt per batch"
                )
            self.completed_steps = step
            loss = (
                outputs.get("loss") if isinstance(outputs, dict) else outputs
            )
            self.last_loss = loss.detach() if loss is not None else None
            if step == warmup_steps:
                self.start(trainer, pl_module)
            elif step == steps:
                self.synchronize(pl_module)
                self.elapsed_s = time.perf_counter() - self.started

        def result(self, trainer) -> dict:
            if self.completed_steps != steps or self.elapsed_s is None:
                raise RuntimeError(
                    "training ended before the measurement completed"
                )
            loss = (
                float(self.last_loss)
                if self.last_loss is not None
                else float("nan")
            )
            if not math.isfinite(loss):
                raise RuntimeError(
                    "final training loss is missing or non-finite"
                )
            return {
                "completed_steps": self.completed_steps,
                "measured_steps": steps - warmup_steps,
                "elapsed_s": self.elapsed_s,
                "final_loss": loss,
                "precision": trainer.precision,
                "gpu": torch.cuda.get_device_name(
                    trainer.strategy.root_device
                ),
            }

    return Window()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_rfdetr_version not in SUPPORTED_VERSIONS:
        parser.error(f"expected version must be one of {SUPPORTED_VERSIONS}")
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("require 0 <= warmup-steps < steps")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    try:
        int(args.rfdetr_wheel_sha256, 16)
    except ValueError:
        parser.error("rfdetr-wheel-sha256 must be hexadecimal")
    if len(args.rfdetr_wheel_sha256) != 64:
        parser.error(
            "rfdetr-wheel-sha256 must contain 64 hexadecimal characters"
        )

    disabled = os.environ.get("TRACEML_DISABLED") == "1"
    if not disabled and not all(
        os.environ.get(name)
        for name in ("TRACEML_SESSION_ID", "TRACEML_LOGS_DIR")
    ):
        parser.error("launch through traceml run as documented in README.md")

    dataset = args.dataset_dir.expanduser().resolve()
    try:
        manifest, manifest_sha256 = validate_dataset(dataset)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output_dir.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"choose a new output directory: {output}")

    import rfdetr
    import torch
    import traceml_ai
    from pytorch_lightning import seed_everything
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    from traceml_ai.integrations import rfdetr as tracing
    from traceml_ai.integrations.lightning import TraceMLCallback

    installed_version = importlib.metadata.version("rfdetr")
    if installed_version != args.expected_rfdetr_version:
        parser.error(
            f"expected RF-DETR {args.expected_rfdetr_version}, found {installed_version}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("this experiment requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    precision = args.precision
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if precision == "auto":
        precision = "bf16" if bf16 else "fp16"
    if precision == "bf16" and not bf16:
        parser.error("BF16 is not supported natively on this GPU; use fp16")

    seed_everything(args.seed, workers=True)
    tracing.init()
    from rfdetr.training import (
        RFDETRDataModule,
        RFDETRModelModule,
        build_trainer,
    )

    model_kwargs = {"device": "cuda:0", "compile": False, "resolution": 384}
    if "cuda_graphs" in RFDETRNanoConfig.model_fields:
        model_kwargs["cuda_graphs"] = False
    model_config = RFDETRNanoConfig(**model_kwargs)
    train_config = TrainConfig(
        dataset_file="coco",
        dataset_dir=str(dataset),
        output_dir=str(output / "training"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        grad_accum_steps=1,
        amp_dtype=precision,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        augmentation_backend="torchvision",
        devices=1,
        num_nodes=1,
        strategy="auto",
        accelerator="gpu",
        tensorboard=False,
        wandb=False,
        mlflow=False,
        progress_bar=None,
        run_test=False,
        save_dataset_grids=False,
    )
    trainer_overrides = {
        "max_steps": args.steps,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "logger": False,
        "enable_model_summary": False,
    }
    module = RFDETRModelModule(model_config, train_config)
    data = RFDETRDataModule(model_config, train_config)
    timer = make_window_callback(args.warmup_steps, args.steps)
    trainer = build_trainer(train_config, model_config, **trainer_overrides)
    trace_callbacks = [
        callback
        for callback in trainer.callbacks
        if isinstance(callback, TraceMLCallback)
    ]
    if not disabled and len(trace_callbacks) != 1:
        raise RuntimeError(
            "TraceML did not install exactly one RF-DETR training callback"
        )
    if disabled and trace_callbacks:
        raise RuntimeError(
            "native control unexpectedly contains a TraceML callback"
        )
    trainer.callbacks.append(timer)

    frozen = command(sys.executable, "-m", "pip", "freeze", "--all")
    (output / "installed-packages.txt").write_text(
        frozen + "\n", encoding="utf-8"
    )
    trace_db = (
        None
        if disabled
        else os.path.relpath(
            Path(os.environ["TRACEML_LOGS_DIR"]).resolve()
            / os.environ["TRACEML_SESSION_ID"]
            / "aggregator"
            / "telemetry",
            output,
        )
    )
    run = {
        "schema_version": 1,
        "mode": "native" if disabled else "traced",
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "trace_db": trace_db,
        "workload": {
            "model": "RF-DETR Nano",
            "resolution": 384,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "precision_requested": args.precision,
            "precision_resolved": trainer.precision,
            "augmentation_backend": "torchvision",
            "multi_scale": False,
            "compile": False,
            "cuda_graphs": False,
        },
        "dataset": {
            "manifest_sha256": manifest_sha256,
            "generator": manifest["generator"],
            "annotations": {
                split: sha256(
                    dataset / "annotations" / f"instances_{split}.json"
                )
                for split in ("train2017", "val2017")
            },
        },
        "sources": {
            "rfdetr": {
                "version": installed_version,
                "wheel_sha256": args.rfdetr_wheel_sha256.lower(),
                "python_tree_sha256": package_tree_sha256(rfdetr),
            },
            "traceml": source_identity(traceml_ai, "traceml-ai"),
            "script_sha256": sha256(__file__),
        },
        "weights_sha256": sha256(model_config.pretrain_weights),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "controlled_packages_sha256": dependency_fingerprint(frozen),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "cpu": cpu_description(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_driver": command(
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ),
        },
    }
    write_json(output / "run.json", run)
    trainer.fit(module, datamodule=data)
    write_json(output / "result.json", timer.result(trainer))


if __name__ == "__main__":
    main()
