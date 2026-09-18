"""Bounded, native RF-DETR Nano training for upstream issue #1410."""

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
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from unittest.mock import patch

RFDETR_COMMIT = "0ed5be8e8d6762c4978a11671cbf34cfc0595e25"
PROFILE_WAIT, PROFILE_WARMUP, PROFILE_ACTIVE = 20, 5, 10
CRITERION_PROFILE_SCOPES = (
    "rfdetr/criterion_including_matcher",
    "rfdetr/matcher",
)
TRAINING_PROFILE_SCOPES = (
    "rfdetr/optimizer_step",
    "rfdetr/ema_update",
    "rfdetr/lr_scheduler_step",
)
PROFILE_SCOPES = CRITERION_PROFILE_SCOPES + TRAINING_PROFILE_SCOPES
COLLECTIVE_TOKENS = (
    "nccl",
    "allreduce",
    "all_reduce",
    "reduce_scatter",
    "reducescatter",
    "all_gather",
    "allgather",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--precision", choices=("auto", "fp16", "bf16"), default="auto"
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-rfdetr-commit",
        default=RFDETR_COMMIT,
        help="Require this clean RF-DETR source revision",
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Use RF-DETR's native multi-scale and expanded-scale defaults",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Separate, untraced PyTorch profiler run",
    )
    return parser


def sha256(path):
    with Path(path).open("rb") as handle:
        return _hash_stream(handle)


def _hash_stream(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def command(*args):
    return subprocess.check_output(
        args, text=True, stderr=subprocess.PIPE
    ).strip()


def cpu_description():
    # lscpu's instantaneous/scaling MHz changes between otherwise paired runs.
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
    rows = json.loads(command("lscpu", "--json"))["lscpu"]
    return "\n".join(
        f"{row['field']} {row['data']}"
        for row in rows
        if row["field"] in fields
    )


def source_identity(module, distribution):
    """Identify editable checkouts or pip installs made from a Git commit."""
    source = Path(module.__file__).resolve()
    for root in source.parents:
        if (root / ".git").exists():
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
                break  # A virtualenv inside a repo does not belong to that repo.
            diff = command("git", "-C", str(root), "diff", "HEAD", "--", ".")
            status = command(
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            )
            return {
                "commit": command("git", "-C", str(root), "rev-parse", "HEAD"),
                "tracked_diff_sha256": hashlib.sha256(
                    diff.encode()
                ).hexdigest(),
                "dirty": bool(status),
            }
    installed = importlib.metadata.distribution(distribution)
    direct = installed.read_text("direct_url.json")
    vcs = json.loads(direct or "{}").get("vcs_info", {})
    installed_source = Path(
        installed.locate_file(f"{module.__name__}/__init__.py")
    ).resolve()
    if source == installed_source and vcs.get("commit_id"):
        return {"commit": vcs["commit_id"], "dirty": None}
    return {"commit": None, "dirty": None}


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )


def config_changes(config, defaults):
    """Record changed native settings; paths remain in the full run config."""
    original = defaults.model_dump(mode="json")
    return {
        key: {"default": original[key], "value": value}
        for key, value in config.model_dump(mode="json").items()
        if key not in ("dataset_dir", "output_dir") and value != original[key]
    }


def _collective_summary(trace_events):
    """Summarize explicit collective CUDA kernels without inferring gaps."""
    kernels = []
    for event in trace_events:
        name = str(event.get("name", ""))
        category = str(event.get("cat", "")).lower()
        duration = event.get("dur")
        if (
            event.get("ph") == "X"
            and ("kernel" in category or "gpu" in category)
            and any(token in name.lower() for token in COLLECTIVE_TOKENS)
            and isinstance(duration, (int, float))
            and math.isfinite(duration)
            and duration >= 0
        ):
            kernels.append((name, float(duration)))
    if not kernels:
        return {
            "available": False,
            "calls": None,
            "cuda_total_ms": None,
            "names": [],
        }
    return {
        "available": True,
        "calls": len(kernels),
        "cuda_total_ms": sum(duration for _, duration in kernels) / 1000,
        "names": sorted({name for name, _ in kernels}),
    }


def export_profile(profiler, output, rank, scopes=PROFILE_SCOPES):
    """Save the raw trace and inclusive scope totals from its active window."""
    from torch.profiler import ProfilerActivity

    trace_path = output / f"profile-rank-{rank}.json"
    profiler.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    cpu_events = {
        name: [
            event
            for event in trace["traceEvents"]
            if event.get("name") == name
            and event.get("cat") == "user_annotation"
            and event.get("ph") == "X"
        ]
        for name in scopes
    }
    events = {event.key: event for event in profiler.key_averages()}
    missing = {
        name for name in scopes if name not in events or not cpu_events[name]
    }
    if missing:
        raise RuntimeError(f"Profiler is missing RF-DETR scopes: {missing}")
    cuda = ProfilerActivity.CUDA in profiler.activities
    write_json(
        output / f"profile-summary-rank-{rank}.json",
        {
            "rank": rank,
            "start_step": PROFILE_WAIT + PROFILE_WARMUP + 1,
            "end_step": PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE,
            "active_steps": PROFILE_ACTIVE,
            "collectives": _collective_summary(trace["traceEvents"]),
            "scopes": {
                name: {
                    "calls": events[name].count,
                    # PyTorch 2.9 reports zero cpu_time_total for these user
                    # annotations even though their Chrome events have duration.
                    "cpu_total_ms": sum(
                        event["dur"] for event in cpu_events[name]
                    )
                    / 1000,
                    "cuda_total_ms": (
                        events[name].device_time_total / 1000 if cuda else None
                    ),
                }
                for name in scopes
            },
        },
    )


def make_window_callback(warmup_steps, steps, profiler=None):
    """Time a contiguous window including the fetch after the warmup batch."""
    import torch
    from pytorch_lightning import Callback

    class Window(Callback):
        def __init__(self):
            self.started = None
            self.elapsed_s = None
            self.completed_steps = 0
            self.last_loss = None
            self.scopes = None
            self.peak_allocated_bytes = None
            self.peak_reserved_bytes = None

        def synchronize(self, module):
            if module.device.type == "cuda":
                torch.cuda.synchronize(module.device)

        def start(self, trainer, module):
            self.synchronize(module)
            trainer.strategy.barrier()
            if module.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(module.device)
            self.started = time.perf_counter()

        def on_train_start(self, trainer, pl_module):
            if profiler is not None:
                # Install after EMA copies the module so its copy stays unwrapped.
                self.scopes = profiler_scopes(trainer, pl_module)
                self.scopes.__enter__()
            if warmup_steps == 0:
                self.start(trainer, pl_module)

        def close_scopes(self):
            if self.scopes is not None:
                self.scopes.__exit__(None, None, None)
                self.scopes = None

        def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx
        ):
            step = int(trainer.global_step)
            if step != self.completed_steps + 1:
                raise RuntimeError(
                    "Expected exactly one optimizer-step attempt per batch"
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
                if pl_module.device.type == "cuda":
                    self.peak_allocated_bytes = (
                        torch.cuda.max_memory_allocated(pl_module.device)
                    )
                    self.peak_reserved_bytes = torch.cuda.max_memory_reserved(
                        pl_module.device
                    )
            if profiler is not None:
                profiler.step()

        def result(self, trainer):
            if self.completed_steps != steps or self.elapsed_s is None:
                raise RuntimeError(
                    "Training ended before the complete measurement window"
                )
            loss = (
                float(self.last_loss)
                if self.last_loss is not None
                else float("nan")
            )
            if not math.isfinite(loss):
                raise RuntimeError(
                    "Final training loss is missing or non-finite"
                )
            return {
                "rank": int(trainer.global_rank),
                "completed_steps": self.completed_steps,
                "measured_steps": steps - warmup_steps,
                "elapsed_s": self.elapsed_s,
                "final_loss": loss,
                "precision": trainer.precision,
                "peak_allocated_bytes": self.peak_allocated_bytes,
                "peak_reserved_bytes": self.peak_reserved_bytes,
                "gpu": (
                    torch.cuda.get_device_name(trainer.strategy.root_device)
                    if trainer.strategy.root_device.type == "cuda"
                    else "CPU (smoke test only)"
                ),
            }

    return Window()


@contextmanager
def criterion_scopes(module):
    """Label the existing criterion and nested matcher only in profiler mode."""
    from torch.profiler import record_function

    criterion_forward = module.criterion.forward
    matcher_forward = module.criterion.matcher.forward

    def criterion(*args, **kwargs):
        with record_function("rfdetr/criterion_including_matcher"):
            return criterion_forward(*args, **kwargs)

    def matcher(*args, **kwargs):
        with record_function("rfdetr/matcher"):
            return matcher_forward(*args, **kwargs)

    matcher_type = type(module.criterion.matcher)
    match_many = getattr(matcher_type, "_match_many", None)

    def batched_matcher(self, *args, **kwargs):
        with record_function("rfdetr/matcher"):
            return match_many(self, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(module.criterion, "forward", criterion)
        )
        stack.enter_context(
            patch.object(module.criterion.matcher, "forward", matcher)
        )
        # SetCriterion calls _match_many on the class. Leave class forward and
        # hooks intact to preserve its fast-path gate and fallback behavior.
        if match_many is not None:
            stack.enter_context(
                patch.object(matcher_type, "_match_many", batched_matcher)
            )
        yield


def _patch_profiled_method(stack, target, attribute, name):
    """Wrap one bound method in a PyTorch Profiler annotation."""
    from torch.profiler import record_function

    original = getattr(target, attribute)

    @wraps(original)
    def profiled(*args, **kwargs):
        with record_function(name):
            return original(*args, **kwargs)

    stack.enter_context(patch.object(target, attribute, profiled))


@contextmanager
def profiler_scopes(trainer, module):
    """Label criterion and post-backward work only for profiler runs."""
    with ExitStack() as stack:
        stack.enter_context(criterion_scopes(module))
        if len(trainer.optimizers) != 1:
            raise RuntimeError("Expected exactly one optimizer for profiling")
        _patch_profiled_method(
            stack,
            trainer.optimizers[0],
            "step",
            "rfdetr/optimizer_step",
        )

        ema_callbacks = [
            callback
            for callback in trainer.callbacks
            if callable(getattr(callback, "_update_ema_for_step", None))
        ]
        if len(ema_callbacks) != 1:
            raise RuntimeError("Expected exactly one RF-DETR EMA callback")
        _patch_profiled_method(
            stack,
            ema_callbacks[0],
            "_update_ema_for_step",
            "rfdetr/ema_update",
        )

        scheduler_configs = list(trainer.lr_scheduler_configs)
        if len(scheduler_configs) != 1:
            raise RuntimeError("Expected exactly one learning-rate scheduler")
        _patch_profiled_method(
            stack,
            scheduler_configs[0].scheduler,
            "step",
            "rfdetr/lr_scheduler_step",
        )
        yield


def validate_rfdetr_source(source, expected_commit):
    """Require the requested clean RF-DETR worktree revision."""
    if source.get("commit") != expected_commit or source.get("dirty"):
        raise ValueError(
            f"use the clean RF-DETR commit {expected_commit}; found {source}"
        )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("require 0 <= warmup-steps < steps")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error(
            "batch-size must be positive; num-workers must be nonnegative"
        )
    if (
        args.profile
        and args.steps < PROFILE_WAIT + PROFILE_WARMUP + PROFILE_ACTIVE
    ):
        parser.error("the profiler schedule requires at least 35 steps")
    disabled = os.environ.get("TRACEML_DISABLED") == "1"
    if args.profile and not disabled:
        parser.error(
            "use traceml run --disable-traceml for the separate profiler run"
        )
    if not disabled and not os.environ.get("TRACEML_SESSION_ID"):
        parser.error("launch through traceml run (see README)")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != local_world or not 0 <= rank < world:
        parser.error("this case study supports single-node torchrun only")
    dataset = args.dataset_dir.expanduser().resolve()
    for split in ("train2017", "val2017"):
        if (
            not (dataset / split).is_dir()
            or not (
                dataset / "annotations" / f"instances_{split}.json"
            ).is_file()
        ):
            parser.error(
                f"missing COCO {split} images or annotations under {dataset}"
            )

    import rfdetr
    import torch
    import traceml_ai
    from pytorch_lightning import seed_everything
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    from traceml_ai.integrations import rfdetr as tracing

    if (
        not torch.cuda.is_available()
        or local_world > torch.cuda.device_count()
    ):
        parser.error("the requested number of CUDA GPUs is unavailable")
    torch.cuda.set_device(local_rank)
    precision = args.precision
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if precision == "auto":
        precision = "bf16" if bf16 else "fp16"
    if precision == "bf16" and not bf16:
        parser.error("BF16 is not supported natively on this GPU; use fp16")
    source = source_identity(rfdetr, "rfdetr")
    try:
        validate_rfdetr_source(source, args.expected_rfdetr_commit)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output_dir.expanduser().resolve()
    if rank == 0:
        for split, count in (("train2017", 118287), ("val2017", 5000)):
            found = sum(1 for _ in (dataset / split).glob("*.jpg"))
            if found != count:
                parser.error(
                    f"full COCO {split} requires {count} JPEGs; found {found}"
                )
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f"choose a new output directory: {output}")

    seed_everything(args.seed, workers=True)
    tracing.init()
    # Import the patched factory only after opt-in initialization.
    from rfdetr.training import (
        RFDETRDataModule,
        RFDETRModelModule,
        build_trainer,
    )

    mc = RFDETRNanoConfig(
        device=f"cuda:{local_rank}", compile=False, resolution=384
    )
    training_options = {}
    if not args.multi_scale:
        training_options.update(
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
        )
    tc = TrainConfig(
        dataset_file="coco",
        dataset_dir=str(dataset),
        output_dir=str(output / "training"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        grad_accum_steps=1,
        amp_dtype=precision,
        augmentation_backend="torchvision",
        devices=world,
        num_nodes=1,
        strategy="ddp" if world > 1 else "auto",
        accelerator="gpu",
        tensorboard=False,
        wandb=False,
        mlflow=False,
        progress_bar=None,
        run_test=False,
        save_dataset_grids=False,
        **training_options,
    )
    trainer_overrides = dict(
        max_steps=args.steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_model_summary=False,
    )
    overrides = {
        "model": config_changes(mc, RFDETRNanoConfig()),
        "training": config_changes(tc, TrainConfig(dataset_dir=str(dataset))),
        "trainer": trainer_overrides,
    }
    module = RFDETRModelModule(mc, tc)
    data = RFDETRDataModule(mc, tc)
    profiler = None
    if args.profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=PROFILE_WAIT,
                warmup=PROFILE_WARMUP,
                active=PROFILE_ACTIVE,
                repeat=1,
            ),
            on_trace_ready=lambda prof: export_profile(prof, output, rank),
        )
    timer = make_window_callback(args.warmup_steps, args.steps, profiler)
    trainer = build_trainer(tc, mc, **trainer_overrides)
    # Preserve upstream EMA/checkpoint callbacks and the RF-DETR TraceML callback.
    trainer.callbacks.append(timer)
    if rank == 0:
        frozen = command(sys.executable, "-m", "pip", "freeze", "--all")
        (output / "environment.txt").write_text(frozen + "\n")
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
        write_json(
            output / "run.json",
            {
                "mode": (
                    "profiler"
                    if args.profile
                    else ("baseline" if disabled else "traced")
                ),
                "steps": args.steps,
                "warmup_steps": args.warmup_steps,
                "world_size": world,
                "model_config": mc.model_dump(mode="json"),
                "train_config": tc.model_dump(mode="json"),
                "configuration_overrides": overrides,
                "trainer_precision": trainer.precision,
                "trace_db": trace_db,
                "sources": {
                    "rfdetr": source,
                    "traceml": source_identity(traceml_ai, "traceml-ai"),
                    "script_sha256": sha256(__file__),
                },
                "annotations": {
                    split: sha256(
                        dataset / "annotations" / f"instances_{split}.json"
                    )
                    for split in ("train2017", "val2017")
                },
                "weights_sha256": sha256(mc.pretrain_weights),
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "packages_sha256": hashlib.sha256(
                        frozen.encode()
                    ).hexdigest(),
                    "hostname": platform.node(),
                    "platform": platform.platform(),
                    "cpu": cpu_description(),
                    "cpu_count": os.cpu_count(),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    ),
                    "gpu_driver": command(
                        "nvidia-smi",
                        "--query-gpu=name,uuid,driver_version,memory.total",
                        "--format=csv,noheader",
                    ),
                },
            },
        )
    with profiler if profiler is not None else nullcontext():
        try:
            trainer.fit(module, datamodule=data)
        finally:
            timer.close_scopes()
    write_json(output / f"rank-{rank}.json", timer.result(trainer))


if __name__ == "__main__":
    main()
