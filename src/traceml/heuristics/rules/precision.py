"""Precision / AMP heuristic rules."""

from __future__ import annotations

from typing import Any, Dict, List

from traceml.heuristics._types import Recommendation


def _min_compute_capability(system: Dict[str, Any]) -> float:
    """Return the lowest GPU SM compute capability, or 0.0 if unknown."""
    caps = []
    for gpu in system.get("gpus", []):
        raw = gpu.get("compute_capability", "")
        try:
            caps.append(float(str(raw)))
        except (ValueError, TypeError):
            pass
    return min(caps) if caps else 0.0


def check_precision(
    code: Dict[str, Any], system: Dict[str, Any]
) -> List[Recommendation]:
    recs: List[Recommendation] = []
    prec = code.get("precision", {})
    dist = code.get("distributed", {})

    autocast = prec.get("autocast", False)
    grad_scaler = prec.get("grad_scaler", False)
    torch_compile = prec.get("torch_compile", False)
    cudnn_benchmark = prec.get("cudnn_benchmark", False)
    dtype = prec.get("dtype")

    gpu_count = len(system.get("gpus", []))
    has_gpu = gpu_count > 0
    sm_cap = _min_compute_capability(system)
    gpu_names = [g.get("name", "") for g in system.get("gpus", [])]
    gpu_label = gpu_names[0] if gpu_names else "detected GPU"

    is_hf = dist.get("hf_trainer", False)
    is_deepspeed = dist.get("deepspeed", False)
    is_fsdp = dist.get("fsdp", False)

    # Tensor Cores require SM ≥ 7.0 (Volta+)
    if (
        has_gpu
        and sm_cap >= 7.0
        and not autocast
        and not grad_scaler
        and not is_hf
        and not is_deepspeed
    ):
        recs.append(
            Recommendation(
                kind="AMP_NOT_USED",
                severity="warn",
                category="precision",
                reason=(
                    f"No mixed precision detected; Tensor Cores on {gpu_label} "
                    f"(SM {sm_cap:.1f}) are unused, leaving significant throughput "
                    "on the table"
                ),
                action=(
                    "Wrap forward pass with torch.autocast('cuda') and add "
                    "torch.cuda.amp.GradScaler() for float16, or switch to "
                    "bfloat16 (GradScaler not required)"
                ),
            )
        )

    # autocast without scaler — risky unless using bfloat16
    if autocast and not grad_scaler and dtype != "bfloat16" and not is_hf:
        recs.append(
            Recommendation(
                kind="GRAD_SCALER_MISSING",
                severity="warn",
                category="precision",
                reason=(
                    "torch.autocast is used but GradScaler is absent; float16 "
                    "gradients can underflow to zero, silently corrupting training"
                ),
                action=(
                    "Add torch.cuda.amp.GradScaler() unless you are using bfloat16, "
                    "which does not require loss scaling"
                ),
            )
        )

    # Manual .half() without autocast
    if dtype == "float16" and not autocast:
        recs.append(
            Recommendation(
                kind="HALF_DTYPE_WITHOUT_AUTOCAST",
                severity="warn",
                category="precision",
                reason=(
                    ".half() detected without torch.autocast; manual mixed-precision "
                    "misses operator-level dtype selection and is error-prone"
                ),
                action=(
                    "Replace model.half() with torch.autocast('cuda') "
                    "for automatic and safe dtype management"
                ),
            )
        )

    # torch.compile with FSDP/DeepSpeed — known compatibility surface
    if torch_compile and (is_fsdp or is_deepspeed):
        recs.append(
            Recommendation(
                kind="TORCH_COMPILE_COMPAT",
                severity="info",
                category="precision",
                reason=(
                    "torch.compile with FSDP/DeepSpeed has known compatibility "
                    "caveats; some module types and hooks are not yet fully supported"
                ),
                action=(
                    "Verify with a short smoke run; if errors occur try "
                    "torch.compile(..., fullgraph=False)"
                ),
            )
        )

    # cudnn.benchmark missing for fixed-shape non-distributed training
    is_distributed = any(dist.get(k) for k in ("ddp", "fsdp", "deepspeed"))
    if has_gpu and not cudnn_benchmark and not is_distributed and not is_hf:
        recs.append(
            Recommendation(
                kind="CUDNN_BENCHMARK_MISSING",
                severity="info",
                category="precision",
                reason=(
                    "torch.backends.cudnn.benchmark is not enabled; cuDNN will not "
                    "auto-select the fastest convolution algorithm for your input shapes"
                ),
                action=(
                    "Add torch.backends.cudnn.benchmark = True at script startup "
                    "if input shapes are fixed (avoid with dynamic shapes)"
                ),
            )
        )

    return recs
