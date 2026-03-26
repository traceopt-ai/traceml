"""Build a versioned manifest from AST findings.

The manifest is intentionally simple and resilient: all expected top-level
sections are always present, and failures in one summarisation step do not
prevent the rest of the manifest from being produced.

This module performs summarisation only.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from traceml.utils.ast_analysis.models import (
    CodeFindings,
    DataLoaderFinding,
    HFTrainingArgumentsFinding,
)


def _safe(callable_, default):
    """Execute *callable_* and return *default* if it raises."""
    try:
        return callable_()
    except Exception:
        return default


def _pick_primary_dataloader(
    findings: CodeFindings,
) -> Optional[DataLoaderFinding]:
    """Return the most likely primary training dataloader.

    Heuristic order:
    1. Dataloader whose variable name contains 'train'
    2. Dataloader found in a train-like phase/function
    3. First dataloader found
    """
    if not findings.dataloaders:
        return None

    for dl in findings.dataloaders:
        v = (dl.variable_name or "").lower()
        if "train" in v:
            return dl

    for dl in findings.dataloaders:
        ph = (dl.phase_hint or "").lower()
        if ph == "train":
            return dl

    return findings.dataloaders[0]


def _pick_primary_hf_args(
    findings: CodeFindings,
) -> Optional[HFTrainingArgumentsFinding]:
    """Return the most relevant Hugging Face TrainingArguments finding."""
    if not findings.hf_training_args:
        return None
    return findings.hf_training_args[0]


def _summarize_framework(findings: CodeFindings) -> Dict[str, Optional[bool]]:
    dist_kinds = {d.kind for d in findings.distributed}
    return {
        "pytorch": (
            True
            if (
                findings.has_training_loop
                or len(findings.dataloaders) > 0
                or len(findings.optimizers) > 0
                or len(findings.distributed) > 0
            )
            else None
        ),
        "huggingface": (
            True
            if ("hf_trainer" in dist_kinds or findings.hf_training_args)
            else False
        ),
        "lightning": True if "lightning" in dist_kinds else False,
        "distributed_wrapper_detected": (
            True
            if dist_kinds.intersection(
                {"ddp", "fsdp", "hf_trainer", "lightning"}
            )
            else False
        ),
    }


def _summarize_dataloader(
    findings: CodeFindings,
    primary_dl: Optional[DataLoaderFinding],
    primary_hf_args: Optional[HFTrainingArgumentsFinding],
) -> Dict[str, Any]:
    if primary_dl is None and primary_hf_args is None:
        return {
            "primary_source": None,
            "count": len(findings.dataloaders),
            "num_workers": None,
            "pin_memory": None,
            "persistent_workers": None,
            "prefetch_factor": None,
            "batch_size": None,
            "per_device_train_batch_size": None,
            "per_device_eval_batch_size": None,
            "distributed_sampler": findings.distributed_sampler_used,
            "set_epoch_called": findings.set_epoch_called,
        }

    if primary_dl is not None:
        return {
            "primary_source": "dataloader",
            "count": len(findings.dataloaders),
            "num_workers": primary_dl.num_workers,
            "pin_memory": primary_dl.pin_memory,
            "persistent_workers": primary_dl.persistent_workers,
            "prefetch_factor": primary_dl.prefetch_factor,
            "batch_size": primary_dl.batch_size,
            "per_device_train_batch_size": (
                primary_hf_args.per_device_train_batch_size
                if primary_hf_args
                else None
            ),
            "per_device_eval_batch_size": (
                primary_hf_args.per_device_eval_batch_size
                if primary_hf_args
                else None
            ),
            "distributed_sampler": findings.distributed_sampler_used,
            "set_epoch_called": findings.set_epoch_called,
        }

    return {
        "primary_source": "hf_training_args",
        "count": len(findings.dataloaders),
        "num_workers": primary_hf_args.dataloader_num_workers,
        "pin_memory": primary_hf_args.dataloader_pin_memory,
        "persistent_workers": primary_hf_args.dataloader_persistent_workers,
        "prefetch_factor": None,
        "batch_size": None,
        "per_device_train_batch_size": primary_hf_args.per_device_train_batch_size,
        "per_device_eval_batch_size": primary_hf_args.per_device_eval_batch_size,
        "distributed_sampler": findings.distributed_sampler_used,
        "set_epoch_called": findings.set_epoch_called,
    }


def _summarize_precision(
    findings: CodeFindings,
    primary_hf_args: Optional[HFTrainingArgumentsFinding],
) -> Dict[str, Any]:
    prec_kinds = {p.kind for p in findings.precision}
    dtype = next(
        (p.dtype_str for p in findings.precision if p.dtype_str), None
    )

    if dtype is None and primary_hf_args is not None:
        if primary_hf_args.fp16 is True:
            dtype = "float16"
        elif primary_hf_args.bf16 is True:
            dtype = "bfloat16"

    return {
        "autocast": "autocast" in prec_kinds,
        "grad_scaler": "grad_scaler" in prec_kinds,
        "dtype": dtype,
        "half_calls_detected": "half" in prec_kinds,
        "bfloat16_calls_detected": "bfloat16" in prec_kinds,
        "cudnn_benchmark": any(
            m.kind == "cudnn_benchmark" for m in findings.models
        ),
        "float32_matmul_precision": any(
            m.kind == "float32_matmul_precision" for m in findings.models
        ),
    }


def _summarize_execution(
    findings: CodeFindings,
    primary_hf_args: Optional[HFTrainingArgumentsFinding],
) -> Dict[str, Any]:
    dist_kinds = {d.kind for d in findings.distributed}
    return {
        "torch_compile": (
            True
            if "torch_compile" in dist_kinds
            or (
                primary_hf_args is not None
                and primary_hf_args.torch_compile is True
            )
            else False
        ),
        "gradient_accumulation_steps": (
            findings.gradient_accumulation_steps
            if findings.gradient_accumulation_steps is not None
            else (
                primary_hf_args.gradient_accumulation_steps
                if primary_hf_args
                else None
            )
        ),
    }


def _summarize_distributed(findings: CodeFindings) -> Dict[str, Any]:
    dist_kinds = {d.kind for d in findings.distributed}
    init_pg = next(
        (d for d in findings.distributed if d.kind == "init_process_group"),
        None,
    )

    return {
        "ddp": "ddp" in dist_kinds,
        "fsdp": "fsdp" in dist_kinds,
        "accelerate": "accelerate" in dist_kinds,
        "hf_trainer": "hf_trainer" in dist_kinds,
        "lightning": "lightning" in dist_kinds,
        "init_process_group": "init_process_group" in dist_kinds,
        "backend": init_pg.backend if init_pg is not None else None,
        "single_node_focus": True,
    }


def _summarize_train_loop(findings: CodeFindings) -> Dict[str, Any]:
    return {
        "has_training_loop": findings.has_training_loop,
        "phase_hints_detected": sorted(findings.phase_hints),
        "zero_grad_detected": findings.zero_grad_detected,
        "backward_detected": findings.backward_detected,
        "optimizer_step_detected": findings.optimizer_step_detected,
        "logging_in_loop": findings.logging_in_loop,
        "checkpoint_in_loop": findings.checkpoint_in_loop,
        "validation_in_loop": findings.validation_in_loop,
        "trainer_train_called": findings.trainer_train_called,
    }


def _summarize_model(
    findings: CodeFindings,
    primary_hf_args: Optional[HFTrainingArgumentsFinding],
) -> Dict[str, Any]:
    from_pretrained = next(
        (m.model_name for m in findings.models if m.kind == "from_pretrained"),
        None,
    )
    return {
        "from_pretrained": from_pretrained is not None,
        "from_pretrained_name": from_pretrained,
        "gradient_checkpointing": (
            findings.has_gradient_checkpointing
            or (
                primary_hf_args.gradient_checkpointing is True
                if primary_hf_args
                else False
            )
        ),
    }


def _summarize_optimizer(
    findings: CodeFindings,
    primary_hf_args: Optional[HFTrainingArgumentsFinding],
) -> Dict[str, Any]:
    opt = findings.optimizers[0] if findings.optimizers else None
    return {
        "optimizer_type": opt.optimizer_type if opt else None,
        "learning_rate": (
            opt.learning_rate
            if opt and opt.learning_rate is not None
            else (primary_hf_args.learning_rate if primary_hf_args else None)
        ),
        "weight_decay": (
            opt.weight_decay
            if opt and opt.weight_decay is not None
            else (primary_hf_args.weight_decay if primary_hf_args else None)
        ),
        "hf_optim": primary_hf_args.optim if primary_hf_args else None,
    }


def _summarize_sync_calls(findings: CodeFindings) -> Dict[str, int]:
    return {
        "item_calls": findings.sync_calls_item,
        "cpu_calls": findings.sync_calls_cpu,
        "numpy_calls": findings.sync_calls_numpy,
        "cuda_synchronize_calls": findings.sync_calls_cuda_synchronize,
    }


def _summarize_device_transfer(findings: CodeFindings) -> Dict[str, bool]:
    return {
        "to_device_detected": findings.to_device_detected,
        "non_blocking_used": findings.non_blocking_used,
    }


def _summarize_hf(
    primary_hf_args: Optional[HFTrainingArgumentsFinding],
) -> Dict[str, Any]:
    if primary_hf_args is None:
        return {
            "training_arguments_detected": False,
            "fp16": None,
            "bf16": None,
            "per_device_train_batch_size": None,
            "per_device_eval_batch_size": None,
            "gradient_accumulation_steps": None,
            "dataloader_num_workers": None,
            "dataloader_pin_memory": None,
            "dataloader_persistent_workers": None,
            "learning_rate": None,
            "weight_decay": None,
            "optim": None,
            "logging_steps": None,
            "save_steps": None,
            "eval_steps": None,
            "max_steps": None,
            "num_train_epochs": None,
            "gradient_checkpointing": None,
            "torch_compile": None,
        }

    return {
        "training_arguments_detected": True,
        "fp16": primary_hf_args.fp16,
        "bf16": primary_hf_args.bf16,
        "per_device_train_batch_size": primary_hf_args.per_device_train_batch_size,
        "per_device_eval_batch_size": primary_hf_args.per_device_eval_batch_size,
        "gradient_accumulation_steps": primary_hf_args.gradient_accumulation_steps,
        "dataloader_num_workers": primary_hf_args.dataloader_num_workers,
        "dataloader_pin_memory": primary_hf_args.dataloader_pin_memory,
        "dataloader_persistent_workers": primary_hf_args.dataloader_persistent_workers,
        "learning_rate": primary_hf_args.learning_rate,
        "weight_decay": primary_hf_args.weight_decay,
        "optim": primary_hf_args.optim,
        "logging_steps": primary_hf_args.logging_steps,
        "save_steps": primary_hf_args.save_steps,
        "eval_steps": primary_hf_args.eval_steps,
        "max_steps": primary_hf_args.max_steps,
        "num_train_epochs": primary_hf_args.num_train_epochs,
        "gradient_checkpointing": primary_hf_args.gradient_checkpointing,
        "torch_compile": primary_hf_args.torch_compile,
    }


def build_code_manifest(findings: CodeFindings) -> Dict[str, Any]:
    """Return a resilient manifest dict for *findings*.

    All major sections are always present. If a summarisation step fails, that
    section falls back to a schema-compatible value instead of raising.

    The manifest is intended for downstream heuristics/recommendation engines,
    UI renderers, and telemetry enrichment.
    """
    primary_dl = _safe(lambda: _pick_primary_dataloader(findings), None)
    primary_hf_args = _safe(lambda: _pick_primary_hf_args(findings), None)

    return {
        "schema_version": 2,
        "script_path": findings.script_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parse_errors": list(findings.parse_errors),
        "framework": _safe(
            lambda: _summarize_framework(findings),
            {
                "pytorch": None,
                "huggingface": None,
                "lightning": None,
                "distributed_wrapper_detected": None,
            },
        ),
        "dataloader": _safe(
            lambda: _summarize_dataloader(
                findings, primary_dl, primary_hf_args
            ),
            {
                "primary_source": None,
                "count": 0,
                "num_workers": None,
                "pin_memory": None,
                "persistent_workers": None,
                "prefetch_factor": None,
                "batch_size": None,
                "per_device_train_batch_size": None,
                "per_device_eval_batch_size": None,
                "distributed_sampler": False,
                "set_epoch_called": False,
            },
        ),
        "precision": _safe(
            lambda: _summarize_precision(findings, primary_hf_args),
            {
                "autocast": False,
                "grad_scaler": False,
                "dtype": None,
                "half_calls_detected": False,
                "bfloat16_calls_detected": False,
                "cudnn_benchmark": False,
                "float32_matmul_precision": False,
            },
        ),
        "execution": _safe(
            lambda: _summarize_execution(findings, primary_hf_args),
            {
                "torch_compile": False,
                "gradient_accumulation_steps": None,
            },
        ),
        "distributed": _safe(
            lambda: _summarize_distributed(findings),
            {
                "ddp": False,
                "fsdp": False,
                "accelerate": False,
                "hf_trainer": False,
                "lightning": False,
                "init_process_group": False,
                "backend": None,
                "single_node_focus": True,
            },
        ),
        "sync_calls_in_train_loop": _safe(
            lambda: _summarize_sync_calls(findings),
            {
                "item_calls": 0,
                "cpu_calls": 0,
                "numpy_calls": 0,
                "cuda_synchronize_calls": 0,
            },
        ),
        "device_transfer": _safe(
            lambda: _summarize_device_transfer(findings),
            {
                "to_device_detected": False,
                "non_blocking_used": False,
            },
        ),
        "train_loop": _safe(
            lambda: _summarize_train_loop(findings),
            {
                "has_training_loop": False,
                "phase_hints_detected": [],
                "zero_grad_detected": False,
                "backward_detected": False,
                "optimizer_step_detected": False,
                "logging_in_loop": False,
                "checkpoint_in_loop": False,
                "validation_in_loop": False,
                "trainer_train_called": False,
            },
        ),
        "model": _safe(
            lambda: _summarize_model(findings, primary_hf_args),
            {
                "from_pretrained": False,
                "from_pretrained_name": None,
                "gradient_checkpointing": False,
            },
        ),
        "optimizer": _safe(
            lambda: _summarize_optimizer(findings, primary_hf_args),
            {
                "optimizer_type": None,
                "learning_rate": None,
                "weight_decay": None,
                "hf_optim": None,
            },
        ),
        "huggingface": _safe(
            lambda: _summarize_hf(primary_hf_args),
            {
                "training_arguments_detected": False,
                "fp16": None,
                "bf16": None,
                "per_device_train_batch_size": None,
                "per_device_eval_batch_size": None,
                "gradient_accumulation_steps": None,
                "dataloader_num_workers": None,
                "dataloader_pin_memory": None,
                "dataloader_persistent_workers": None,
                "learning_rate": None,
                "weight_decay": None,
                "optim": None,
                "logging_steps": None,
                "save_steps": None,
                "eval_steps": None,
                "max_steps": None,
                "num_train_epochs": None,
                "gradient_checkpointing": None,
                "torch_compile": None,
            },
        ),
        "summary": {
            "dataloader_count": len(findings.dataloaders),
            "optimizer_count": len(findings.optimizers),
            "precision_signal_count": len(findings.precision),
            "distributed_signal_count": len(findings.distributed),
            "model_signal_count": len(findings.models),
            "fine_tuning_signal_count": len(findings.fine_tuning),
        },
    }
