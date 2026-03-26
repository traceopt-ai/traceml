"""Public orchestration layer for AST-based training-script analysis.

This module exposes the stable public API:
- analyze_script
- detect_strategy_hint
- scan_for_optimizer

It also re-exports the public result dataclasses
"""

import ast
from pathlib import Path
from typing import Optional, Set

from traceml.utils.ast_analysis.helpers import (
    build_import_map,
    build_parent_map,
    collect_string_constants,
    fqn,
    has_kw,
    kw_const,
    location,
    read_python_source,
    safe_int,
)
from traceml.utils.ast_analysis.models import (
    CodeFindings,
    DistributedFinding,
    ModelFinding,
    PrecisionFinding,
)
from traceml.utils.ast_analysis.visitor import PatternVisitor

_QLORA_SIGNALS = {"prepare_model_for_kbit_training", "BitsAndBytesConfig"}

_HF_TRAINING_ARG_CLASSES = {
    "TrainingArguments",
    "Seq2SeqTrainingArguments",
    "transformers.TrainingArguments",
    "transformers.Seq2SeqTrainingArguments",
}

_HF_TRAINER_CLASSES = {
    "Trainer",
    "Seq2SeqTrainer",
    "transformers.Trainer",
    "transformers.Seq2SeqTrainer",
    "trl.SFTTrainer",
    "trl.DPOTrainer",
    "trl.PPOTrainer",
    "trl.RewardTrainer",
    "SFTTrainer",
    "DPOTrainer",
    "PPOTrainer",
    "RewardTrainer",
}


def analyze_script(script_path: str) -> CodeFindings:
    """Return best-effort findings for *script_path*.

    Never raises for expected file/parse/analysis issues. Returns a populated
    ``CodeFindings`` object with ``parse_errors`` on failure.
    """
    path = Path(script_path).resolve()
    result = CodeFindings(script_path=str(path))

    if not path.exists() or not path.is_file():
        result.parse_errors.append(f"File not found: {path}")
        return result

    try:
        source = read_python_source(path)
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        result.parse_errors.append(f"Syntax error: {exc}")
        return result
    except OSError as exc:
        result.parse_errors.append(f"Cannot read file: {exc}")
        return result
    except Exception as exc:
        result.parse_errors.append(f"Unexpected parse failure: {exc}")
        return result

    try:
        result.imports = build_import_map(tree)
    except Exception as exc:
        result.parse_errors.append(f"Import-map build failed: {exc}")
        result.imports = {}

    try:
        str_consts = collect_string_constants(tree)
    except Exception as exc:
        result.parse_errors.append(f"String-constant collection failed: {exc}")
        str_consts = {}

    try:
        parent_map = build_parent_map(tree)
    except Exception as exc:
        result.parse_errors.append(f"Parent-map build failed: {exc}")
        parent_map = {}

    try:
        visitor = PatternVisitor(
            script_path=str(path),
            lines=lines,
            imports=result.imports,
            str_consts=str_consts,
            parent_map=parent_map,
        )
        visitor.visit(tree)
    except Exception as exc:
        result.parse_errors.append(f"AST visitor failed: {exc}")
        return result

    result.dataloaders = visitor.dataloaders
    result.optimizers = visitor.optimizers
    result.precision = visitor.precision
    result.distributed = visitor.distributed
    result.models = visitor.models
    result.fine_tuning = visitor.fine_tuning
    result.hf_training_args = visitor.hf_training_args
    result.gradient_accumulation_steps = visitor.gradient_accumulation_steps

    result.phase_hints = set(visitor.phase_hints)
    result.trainer_train_called = visitor.trainer_train_called

    result.sync_calls_item = visitor.sync_calls_item
    result.sync_calls_cpu = visitor.sync_calls_cpu
    result.sync_calls_numpy = visitor.sync_calls_numpy
    result.sync_calls_cuda_synchronize = visitor.sync_calls_cuda_synchronize
    result.to_device_detected = visitor.to_device_detected
    result.non_blocking_used = visitor.non_blocking_used
    result.zero_grad_detected = visitor.zero_grad_detected
    result.backward_detected = visitor.backward_detected
    result.optimizer_step_detected = visitor.optimizer_step_detected
    result.logging_in_loop = visitor.logging_in_loop
    result.checkpoint_in_loop = visitor.checkpoint_in_loop
    result.validation_in_loop = visitor.validation_in_loop
    result.distributed_sampler_used = visitor.distributed_sampler_used
    result.set_epoch_called = visitor.set_epoch_called
    result.prefetch_factor_set = visitor.prefetch_factor_set

    try:
        _apply_qlora_upgrade(result)
    except Exception as exc:
        result.parse_errors.append(f"QLoRA post-processing failed: {exc}")

    try:
        _absorb_hf_training_args(tree, result, lines, str(path))
    except Exception as exc:
        result.parse_errors.append(f"HF TrainingArguments pass failed: {exc}")

    try:
        result.has_training_loop = _has_training_loop(tree, result)
    except Exception as exc:
        result.parse_errors.append(f"Training-loop detection failed: {exc}")
        result.has_training_loop = False

    result.has_gradient_checkpointing = any(
        m.kind == "gradient_checkpointing" for m in result.models
    )

    try:
        _ingest_local_imports(tree, result, str(path))
    except Exception as exc:
        result.parse_errors.append(f"Local-import ingestion failed: {exc}")

    return result


def detect_strategy_hint(script_path: str) -> Optional[str]:
    """Return a conservative distributed strategy hint for single-node setups.

    Returns one of:
    - "fsdp"
    - "ddp"
    - None
    """
    try:
        findings = analyze_script(script_path)
        kinds: Set[str] = {d.kind for d in findings.distributed}
        for strategy in ("fsdp", "ddp"):
            if strategy in kinds:
                return strategy
        return None
    except Exception:
        return None


def scan_for_optimizer(script_path: str) -> Optional[str]:
    """Return the first optimizer type found in *script_path*, or ``None``."""
    findings = analyze_script(script_path)
    if findings.optimizers:
        return findings.optimizers[0].optimizer_type
    return None


def _apply_qlora_upgrade(result: CodeFindings) -> None:
    """Upgrade LoRA findings to QLoRA when k-bit quantization signals exist."""
    src = Path(result.script_path).resolve()
    try:
        source = read_python_source(src)
        tree = ast.parse(source)
    except Exception:
        return

    has_signal = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name in _QLORA_SIGNALS:
            has_signal = True
            break

    if has_signal:
        for ft in result.fine_tuning:
            if ft.method == "lora":
                ft.method = "qlora"


def _absorb_hf_training_args(
    tree: ast.AST,
    result: CodeFindings,
    lines: list[str],
    script_path: str,
) -> None:
    """Backfill key HF Trainer / TrainingArguments signals into the result."""
    imports = result.imports

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = fqn(node, imports)
        if name is None:
            continue
        leaf = name.rsplit(".", 1)[-1]

        if name in _HF_TRAINER_CLASSES or leaf in {
            n.rsplit(".", 1)[-1] for n in _HF_TRAINER_CLASSES
        }:
            if not any(d.kind == "hf_trainer" for d in result.distributed):
                result.distributed.append(
                    DistributedFinding(
                        location=location(script_path, node, lines),
                        kind="hf_trainer",
                        backend=None,
                    )
                )
            continue

        if name not in _HF_TRAINING_ARG_CLASSES and leaf not in {
            n.rsplit(".", 1)[-1] for n in _HF_TRAINING_ARG_CLASSES
        }:
            continue

        loc = location(script_path, node, lines)

        if kw_const(node, "fp16") is True:
            result.precision.append(
                PrecisionFinding(
                    location=loc,
                    kind="autocast",
                    dtype_str="float16",
                    is_deprecated_api=False,
                    phase_hint="train",
                )
            )
        if kw_const(node, "bf16") is True:
            result.precision.append(
                PrecisionFinding(
                    location=loc,
                    kind="autocast",
                    dtype_str="bfloat16",
                    is_deprecated_api=False,
                    phase_hint="train",
                )
            )

        ga = kw_const(node, "gradient_accumulation_steps")
        if ga is not None and result.gradient_accumulation_steps is None:
            result.gradient_accumulation_steps = safe_int(ga)

        if has_kw(node, "deepspeed") and not any(
            d.kind == "deepspeed" for d in result.distributed
        ):
            result.distributed.append(
                DistributedFinding(
                    location=loc,
                    kind="deepspeed",
                    backend=None,
                )
            )

        if kw_const(node, "gradient_checkpointing") is True and not any(
            m.kind == "gradient_checkpointing" for m in result.models
        ):
            result.models.append(
                ModelFinding(
                    location=loc,
                    kind="gradient_checkpointing",
                    model_name=None,
                )
            )


def _has_training_loop(tree: ast.AST, result: CodeFindings) -> bool:
    """Return whether the script likely contains a training path."""
    if any(d.kind == "hf_trainer" for d in result.distributed):
        return True
    if any(d.kind == "lightning" for d in result.distributed):
        return True
    if result.trainer_train_called:
        return True

    saw_backward = saw_step = saw_loop = saw_train = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            saw_loop = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "backward":
                saw_backward = True
            elif attr == "step":
                saw_step = True
            elif attr == "train":
                saw_train = True

    return (saw_backward and saw_step) or (saw_loop and saw_train)


def _ingest_local_imports(
    tree: ast.AST,
    result: CodeFindings,
    script_path: str,
) -> None:
    """Scan sibling Python modules referenced by shallow local imports."""
    base_dir = Path(script_path).resolve().parent
    visited: Set[Path] = set()

    for node in ast.walk(tree):
        candidates: list[Path] = []

        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod and "." not in mod:
                candidates.append(base_dir / f"{mod}.py")

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "." not in alias.name:
                    candidates.append(base_dir / f"{alias.name}.py")

        for cand in candidates:
            cand = cand.resolve()
            if not cand.exists() or not cand.is_file() or cand in visited:
                continue
            visited.add(cand)
            _merge_file_into(cand, result)


def _merge_file_into(path: Path, result: CodeFindings) -> None:
    """Parse *path* and append partial findings into *result*."""
    try:
        source = read_python_source(path)
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, OSError):
        return

    imports = build_import_map(tree)
    str_consts = collect_string_constants(tree)
    parent_map = build_parent_map(tree)

    visitor = PatternVisitor(
        script_path=str(path),
        lines=lines,
        imports=imports,
        str_consts=str_consts,
        parent_map=parent_map,
    )
    visitor.visit(tree)

    result.models.extend(visitor.models)
    result.distributed.extend(visitor.distributed)
    result.precision.extend(visitor.precision)
    result.optimizers.extend(visitor.optimizers)
    result.fine_tuning.extend(visitor.fine_tuning)
    result.dataloaders.extend(visitor.dataloaders)
    result.hf_training_args.extend(visitor.hf_training_args)
    result.phase_hints.update(visitor.phase_hints)
    result.trainer_train_called = (
        result.trainer_train_called or visitor.trainer_train_called
    )

    if (
        visitor.gradient_accumulation_steps is not None
        and result.gradient_accumulation_steps is None
    ):
        result.gradient_accumulation_steps = (
            visitor.gradient_accumulation_steps
        )
