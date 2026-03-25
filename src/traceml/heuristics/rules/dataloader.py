"""DataLoader heuristic rules.

Each function returns a list of Recommendation objects.
All functions are pure: no I/O, no torch imports.
"""

from __future__ import annotations

from typing import Any, Dict, List

from traceml.heuristics._types import Recommendation


def check_dataloader(
    code: Dict[str, Any], system: Dict[str, Any]
) -> List[Recommendation]:
    recs: List[Recommendation] = []
    dl = code.get("dataloader", {})
    dist = code.get("distributed", {})
    is_distributed = dl.get("distributed_sampler") or any(
        dist.get(k) for k in ("ddp", "fsdp")
    )

    num_workers = dl.get("num_workers")
    pin_memory = dl.get("pin_memory")
    persistent_workers = dl.get("persistent_workers")
    set_epoch_called = dl.get("set_epoch_called", False)
    dist_sampler = dl.get("distributed_sampler", False)

    logical_cores = system.get("cpu", {}).get("logical_cores") or 0
    gpu_count = len(system.get("gpus", []))
    has_gpu = gpu_count > 0

    if num_workers == 0:
        recs.append(
            Recommendation(
                kind="NUM_WORKERS_ZERO",
                severity="crit",
                category="dataloader",
                reason=(
                    "num_workers=0: data loading is synchronous and blocks the GPU "
                    "every step while the CPU fetches the next batch"
                ),
                action=(
                    "Set num_workers≥4; a good starting point is "
                    "logical_cores // 4"
                    + (f" = {logical_cores // 4}" if logical_cores else "")
                ),
            )
        )
    elif num_workers is not None and logical_cores > 0:
        recommended = max(logical_cores // 4, 2)
        if num_workers < recommended:
            recs.append(
                Recommendation(
                    kind="NUM_WORKERS_LOW",
                    severity="warn",
                    category="dataloader",
                    reason=(
                        f"num_workers={num_workers} but {logical_cores} logical cores "
                        f"are available; prefetch parallelism is underutilized"
                    ),
                    action=f"Set num_workers={recommended} (logical_cores // 4)",
                )
            )

    if pin_memory is False and has_gpu:
        recs.append(
            Recommendation(
                kind="PIN_MEMORY_OFF",
                severity="warn",
                category="dataloader",
                reason=(
                    "pin_memory=False: every batch requires an extra pageable→pinned "
                    "copy on the CPU before the H2D DMA transfer can start"
                ),
                action="Set pin_memory=True in your DataLoader when training on GPU",
            )
        )

    if (
        num_workers is not None
        and num_workers > 0
        and persistent_workers is not True
    ):
        recs.append(
            Recommendation(
                kind="PERSISTENT_WORKERS_OFF",
                severity="info",
                category="dataloader",
                reason=(
                    "persistent_workers not enabled: worker processes are torn down "
                    "and re-spawned each epoch, adding startup latency"
                ),
                action="Set persistent_workers=True when num_workers > 0",
            )
        )

    if (
        num_workers is not None
        and num_workers > 0
        and pin_memory is True
        and not dl.get("prefetch_factor_set")
    ):
        recs.append(
            Recommendation(
                kind="PREFETCH_FACTOR_DEFAULT",
                severity="info",
                category="dataloader",
                reason=(
                    "prefetch_factor defaults to 2; with pin_memory=True workers can "
                    "pipeline more batches and reduce GPU idle time between steps"
                ),
                action=(
                    "Try prefetch_factor=4 and monitor GPU utilization; "
                    "raise further if gaps between steps are still visible"
                ),
            )
        )

    if dist_sampler and not set_epoch_called and is_distributed:
        recs.append(
            Recommendation(
                kind="DIST_SAMPLER_NO_SET_EPOCH",
                severity="crit",
                category="dataloader",
                reason=(
                    "DistributedSampler detected but sampler.set_epoch() was not "
                    "found; every epoch uses the same shuffled order, breaking "
                    "data diversity across epochs"
                ),
                action=(
                    "Call sampler.set_epoch(epoch) at the start of each epoch "
                    "before iterating the DataLoader"
                ),
            )
        )

    if is_distributed and not dist_sampler:
        recs.append(
            Recommendation(
                kind="DIST_SAMPLER_MISSING",
                severity="warn",
                category="dataloader",
                reason=(
                    "DDP/FSDP run without DistributedSampler detected; ranks may "
                    "receive overlapping data, inflating effective batch size silently"
                ),
                action=(
                    "Wrap your dataset with "
                    "torch.utils.data.distributed.DistributedSampler"
                ),
            )
        )

    return recs
