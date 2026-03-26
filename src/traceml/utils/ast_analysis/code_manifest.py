"""Distills CodeFindings from the AST scanner into the code_manifest.json schema.

The manifest is a flat, version-stamped JSON document that other components
(heuristics engine, aggregator, renderers) can read without importing torch or
running any ML code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from traceml.utils.ast_analysis.scanner import CodeFindings


def _first_dataloader(findings: CodeFindings) -> Optional[Any]:
    return findings.dataloaders[0] if findings.dataloaders else None


def build_code_manifest(findings: CodeFindings) -> Dict[str, Any]:
    """Return the code manifest dict for *findings*.

    All fields are present; unknown/undetected values are None or False.
    """
    dl = _first_dataloader(findings)

    dist_kinds = {d.kind for d in findings.distributed}
    prec_kinds = {p.kind for p in findings.precision}
    prec_dtype = next(
        (p.dtype_str for p in findings.precision if p.dtype_str), None
    )

    return {
        "schema_version": 1,
        "script_path": findings.script_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parse_errors": findings.parse_errors,
        "dataloader": {
            "num_workers": dl.num_workers if dl else None,
            "pin_memory": dl.pin_memory if dl else None,
            "persistent_workers": dl.persistent_workers if dl else None,
            "prefetch_factor_set": findings.prefetch_factor_set,
            "batch_size": dl.batch_size if dl else None,
            "distributed_sampler": findings.distributed_sampler_used,
            "set_epoch_called": findings.set_epoch_called,
        },
        "precision": {
            "autocast": "autocast" in prec_kinds,
            "grad_scaler": "grad_scaler" in prec_kinds,
            "torch_compile": "torch_compile" in dist_kinds,
            "dtype": prec_dtype,
            "cudnn_benchmark": any(
                m.kind == "cudnn_benchmark" for m in findings.models
            ),
        },
        "distributed": {
            "ddp": "ddp" in dist_kinds,
            "fsdp": "fsdp" in dist_kinds,
            "deepspeed": "deepspeed" in dist_kinds,
            "accelerate": "accelerate" in dist_kinds,
            "hf_trainer": "hf_trainer" in dist_kinds,
            "lightning": "lightning" in dist_kinds,
            "init_process_group": "init_process_group" in dist_kinds,
        },
        "sync_calls_in_train_loop": {
            "item_calls": findings.sync_calls_item,
            "cpu_calls": findings.sync_calls_cpu,
            "numpy_calls": findings.sync_calls_numpy,
            "cuda_synchronize_calls": findings.sync_calls_cuda_synchronize,
        },
        "device_transfer": {
            "to_device_detected": findings.to_device_detected,
            "non_blocking_used": findings.non_blocking_used,
        },
        "train_loop": {
            "zero_grad_detected": findings.zero_grad_detected,
            "backward_detected": findings.backward_detected,
            "optimizer_step_detected": findings.optimizer_step_detected,
            "logging_in_loop": findings.logging_in_loop,
            "checkpoint_in_loop": findings.checkpoint_in_loop,
            "validation_in_loop": findings.validation_in_loop,
        },
        "model": {
            "gradient_checkpointing": findings.has_gradient_checkpointing,
            "gradient_accumulation_steps": findings.gradient_accumulation_steps,
            "from_pretrained": any(
                m.kind == "from_pretrained" for m in findings.models
            ),
        },
    }
