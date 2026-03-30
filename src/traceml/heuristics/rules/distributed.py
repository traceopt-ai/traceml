"""Distributed training correctness heuristic rules."""

from __future__ import annotations

from typing import Any, Dict, List

from traceml.heuristics._types import Recommendation


def check_distributed(
    code: Dict[str, Any], system: Dict[str, Any]
) -> List[Recommendation]:
    recs: List[Recommendation] = []
    dist = code.get("distributed", {})
    prec = code.get("precision", {})
    model = code.get("model", {})

    is_ddp = dist.get("ddp", False)
    is_fsdp = dist.get("fsdp", False)
    has_init_pg = dist.get("init_process_group", False)
    autocast = prec.get("autocast", False)
    grad_accum = model.get("gradient_accumulation_steps")
    grad_checkpoint = model.get("gradient_checkpointing", False)

    if is_ddp and not has_init_pg:
        recs.append(
            Recommendation(
                kind="DDP_WITHOUT_INIT_PG",
                severity="warn",
                category="distributed",
                reason=(
                    "DDP wrapper detected but dist.init_process_group() was not "
                    "found; relying on torchrun auto-init makes backend selection "
                    "implicit and harder to debug"
                ),
                action=(
                    "Explicitly call dist.init_process_group(backend='nccl') "
                    "at the start of your script for reproducible multi-node behavior"
                ),
            )
        )

    if is_fsdp and not autocast:
        recs.append(
            Recommendation(
                kind="FSDP_WITHOUT_AMP",
                severity="info",
                category="distributed",
                reason=(
                    "FSDP without mixed precision: parameters are kept in float32, "
                    "missing the memory and throughput gains from FSDP's built-in "
                    "mixed-precision sharding"
                ),
                action=(
                    "Pass MixedPrecision(param_dtype=torch.bfloat16, "
                    "reduce_dtype=torch.float32) to FSDP for optimal throughput"
                ),
            )
        )

    if grad_accum is not None and grad_accum > 1 and (is_ddp or is_fsdp):
        recs.append(
            Recommendation(
                kind="GRAD_ACCUM_WITHOUT_NO_SYNC",
                severity="warn",
                category="distributed",
                reason=(
                    f"gradient_accumulation_steps={grad_accum} with DDP/FSDP but "
                    "model.no_sync() was not detected; all-reduce fires every "
                    "micro-step instead of every accumulation boundary, wasting "
                    "communication bandwidth"
                ),
                action=(
                    "Wrap all accumulation steps except the last in "
                    "`with model.no_sync():` to batch the all-reduce communication"
                ),
            )
        )

    if grad_checkpoint and not autocast:
        recs.append(
            Recommendation(
                kind="GRAD_CHECKPOINT_WITHOUT_AMP",
                severity="info",
                category="distributed",
                reason=(
                    "Gradient checkpointing is enabled but autocast is not; "
                    "recomputed activations will be in float32, partially negating "
                    "the activation memory savings"
                ),
                action=(
                    "Combine gradient checkpointing with torch.autocast('cuda') "
                    "so that recomputed activations stay in the lower precision dtype"
                ),
            )
        )

    return recs
