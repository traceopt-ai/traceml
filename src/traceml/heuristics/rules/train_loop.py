"""Train-loop anti-pattern heuristic rules."""

from __future__ import annotations

from typing import Any, Dict, List

from traceml.heuristics._types import Recommendation


def check_train_loop(
    code: Dict[str, Any], system: Dict[str, Any]
) -> List[Recommendation]:
    recs: List[Recommendation] = []
    loop = code.get("train_loop", {})
    sync = code.get("sync_calls_in_train_loop", {})

    logging_in_loop = loop.get("logging_in_loop", False)
    checkpoint_in_loop = loop.get("checkpoint_in_loop", False)
    validation_in_loop = loop.get("validation_in_loop", False)
    backward = loop.get("backward_detected", False)

    total_sync = (
        sync.get("item_calls", 0)
        + sync.get("cpu_calls", 0)
        + sync.get("numpy_calls", 0)
    )

    if logging_in_loop and total_sync >= 1:
        recs.append(
            Recommendation(
                kind="LOGGING_SYNC_IN_LOOP",
                severity="info",
                category="train_loop",
                reason=(
                    f"{total_sync} CPU-sync call(s) (e.g. .item()) inside the "
                    "training loop cause a GPU pipeline drain on every step where "
                    "logging fires"
                ),
                action=(
                    "Accumulate loss as a tensor; call .item() only at the log "
                    "boundary. Use torch.no_grad() and avoid .item() inside the "
                    "hot path to keep the GPU pipeline unblocked"
                ),
            )
        )

    if checkpoint_in_loop:
        recs.append(
            Recommendation(
                kind="CHECKPOINT_IN_LOOP",
                severity="info",
                category="train_loop",
                reason=(
                    "Checkpointing (torch.save / save_checkpoint) detected inside "
                    "the training loop; synchronous serialization adds periodic I/O "
                    "stalls that pause GPU utilization"
                ),
                action=(
                    "Move checkpointing to epoch boundaries or use a background "
                    "thread / async checkpoint library to overlap I/O with compute"
                ),
            )
        )

    if validation_in_loop and backward:
        # Only flag if we also see backward — validates this is inside a training loop
        recs.append(
            Recommendation(
                kind="VALIDATION_IN_LOOP",
                severity="info",
                category="train_loop",
                reason=(
                    "Validation (model.eval() / torch.no_grad()) appears to run "
                    "inside the same loop as the training backward pass; frequent "
                    "eval can significantly reduce effective training throughput"
                ),
                action=(
                    "Run validation at epoch boundaries or at a fixed step interval. "
                    "Ensure torch.no_grad() or torch.inference_mode() wraps all "
                    "validation forward passes to avoid retaining activation memory"
                ),
            )
        )

    return recs
