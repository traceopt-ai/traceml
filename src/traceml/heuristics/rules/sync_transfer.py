"""Sync call and device transfer heuristic rules."""

from __future__ import annotations

from typing import Any, Dict, List

from traceml.heuristics._types import Recommendation


def check_sync_transfer(
    code: Dict[str, Any], system: Dict[str, Any]
) -> List[Recommendation]:
    recs: List[Recommendation] = []
    sync = code.get("sync_calls_in_train_loop", {})
    transfer = code.get("device_transfer", {})
    dl = code.get("dataloader", {})

    item_calls = sync.get("item_calls", 0)
    cpu_calls = sync.get("cpu_calls", 0)
    numpy_calls = sync.get("numpy_calls", 0)
    cuda_sync_calls = sync.get("cuda_synchronize_calls", 0)

    to_device = transfer.get("to_device_detected", False)
    non_blocking = transfer.get("non_blocking_used", False)
    pin_memory = dl.get("pin_memory", False)

    total_sync = item_calls + cpu_calls + numpy_calls
    if total_sync >= 3:
        parts = []
        if item_calls:
            parts.append(f"{item_calls}x .item()")
        if cpu_calls:
            parts.append(f"{cpu_calls}x .cpu()")
        if numpy_calls:
            parts.append(f"{numpy_calls}x .numpy()")
        recs.append(
            Recommendation(
                kind="SYNC_CALLS_HIGH",
                severity="warn",
                category="sync",
                reason=(
                    f"{total_sync} CPU-sync call(s) detected in training loop "
                    f"({', '.join(parts)}); each forces a full GPU pipeline drain "
                    "and can dominate step time at high throughput"
                ),
                action=(
                    "Accumulate metrics as tensors; call .item() once per "
                    "log interval, not every step. Use torch.no_grad() for "
                    "any in-loop metric computation"
                ),
            )
        )

    if cuda_sync_calls >= 1:
        recs.append(
            Recommendation(
                kind="CUDA_SYNCHRONIZE_IN_LOOP",
                severity="crit",
                category="sync",
                reason=(
                    f"torch.cuda.synchronize() detected {cuda_sync_calls} time(s) "
                    "in or near the training loop; this drains the entire GPU "
                    "pipeline and serializes CPU and GPU execution every call"
                ),
                action=(
                    "Remove cuda.synchronize() from the training loop. "
                    "For timing, use CUDA events: "
                    "start = torch.cuda.Event(enable_timing=True)"
                ),
            )
        )

    if to_device and not non_blocking and pin_memory:
        recs.append(
            Recommendation(
                kind="NON_BLOCKING_NOT_USED",
                severity="warn",
                category="sync",
                reason=(
                    "Tensors are moved to device without non_blocking=True even "
                    "though pin_memory=True is set; the CPU stalls waiting for the "
                    "H2D DMA to complete instead of overlapping with compute"
                ),
                action=(
                    "Use tensor.to(device, non_blocking=True) to allow the CPU "
                    "to continue preparing the next batch while the GPU receives data"
                ),
            )
        )

    if to_device and non_blocking and not pin_memory:
        recs.append(
            Recommendation(
                kind="NON_BLOCKING_WITHOUT_PIN_MEMORY",
                severity="warn",
                category="sync",
                reason=(
                    "non_blocking=True is used but pin_memory=False; non-pinned "
                    "memory requires an extra CPU-side copy before DMA, making "
                    "non_blocking ineffective"
                ),
                action=(
                    "Enable pin_memory=True in your DataLoader to make "
                    "non_blocking=True transfers genuinely asynchronous"
                ),
            )
        )

    return recs
