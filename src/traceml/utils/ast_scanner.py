"""AST-based code analyzer — extract ML patterns from Python training scripts.

Walks Python AST to find:
  - DataLoader instantiations with kwargs (num_workers, pin_memory, etc.)
  - Optimizer usage (Adam, AdamW, SGD)
  - Precision patterns (autocast, .half(), GradScaler)
  - Distributed patterns (DDP, FSDP, init_process_group, torch.compile)
  - Model patterns (from_pretrained, gradient_checkpointing, cudnn.benchmark)

Never crashes. Returns empty CodeFindings on parse errors.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SourceLocation:
    file_path: str
    line_number: int
    col_offset: int
    source_line: str


@dataclass
class DataLoaderFinding:
    location: SourceLocation
    batch_size: Optional[int]
    num_workers: Optional[int]
    pin_memory: Optional[bool]
    persistent_workers: Optional[bool]
    prefetch_factor: Optional[int]
    raw_kwargs: Dict[str, str]


@dataclass
class OptimizerFinding:
    location: SourceLocation
    optimizer_type: str
    learning_rate: Optional[float]
    weight_decay: Optional[float]


@dataclass
class PrecisionFinding:
    location: SourceLocation
    kind: str
    dtype_str: Optional[str]
    is_deprecated_api: bool


@dataclass
class DistributedFinding:
    location: SourceLocation
    kind: str
    backend: Optional[str]


@dataclass
class ModelFinding:
    location: SourceLocation
    kind: str
    model_name: Optional[str]


@dataclass
class FineTuningFinding:
    location: SourceLocation
    method: str  # "lora", "qlora", "peft_generic", "adapter"
    details: Dict[str, str] = field(default_factory=dict)


@dataclass
class CodeFindings:
    script_path: str
    dataloaders: List[DataLoaderFinding] = field(default_factory=list)
    optimizers: List[OptimizerFinding] = field(default_factory=list)
    precision: List[PrecisionFinding] = field(default_factory=list)
    distributed: List[DistributedFinding] = field(default_factory=list)
    models: List[ModelFinding] = field(default_factory=list)
    fine_tuning: List[FineTuningFinding] = field(default_factory=list)
    imports: Dict[str, str] = field(default_factory=dict)
    has_training_loop: bool = False
    has_gradient_checkpointing: bool = False
    gradient_accumulation_steps: Optional[int] = None
    parse_errors: List[str] = field(default_factory=list)


def analyze_script(script_path: str) -> CodeFindings:
    """Analyze a Python script and return structured ML pattern findings.

    Never crashes — returns CodeFindings with parse_errors on failure.
    """
    findings = CodeFindings(script_path=script_path)

    if not os.path.isfile(script_path):
        findings.parse_errors.append(f"File not found: {script_path}")
        return findings

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            source = f.read()
        lines = source.splitlines()
        tree = ast.parse(source, filename=script_path)
    except SyntaxError as e:
        findings.parse_errors.append(f"Syntax error: {e}")
        return findings
    except OSError as e:
        findings.parse_errors.append(f"Read error: {e}")
        return findings

    findings.imports = _walk_imports(tree)
    findings.dataloaders = _find_dataloaders(
        tree, findings.imports, lines, script_path
    )
    findings.optimizers = _find_optimizers(
        tree, findings.imports, lines, script_path
    )
    findings.precision = _find_precision(
        tree, findings.imports, lines, script_path
    )
    findings.distributed = _find_distributed(
        tree, findings.imports, lines, script_path
    )
    findings.models = _find_models(tree, findings.imports, lines, script_path)
    findings.fine_tuning = _find_fine_tuning(
        tree, findings.imports, lines, script_path
    )
    _extract_training_args(tree, findings, lines, script_path)
    findings.has_training_loop = _detect_training_loop(tree, findings)
    findings.has_gradient_checkpointing = any(
        m.kind == "gradient_checkpointing" for m in findings.models
    )

    # Follow local imports one level deep
    _follow_local_imports(tree, findings, script_path)

    return findings


def detect_strategy_hint(script_path: str) -> Optional[str]:
    """Lightweight AST check: return strategy kind if detectable, else None.

    Returns one of: 'fsdp', 'ddp', 'deepspeed', 'data_parallel', or None.
    Never crashes — returns None on any error.
    """
    try:
        if not os.path.isfile(script_path):
            return None
        with open(script_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=script_path)
        imports = _walk_imports(tree)
        distributed = _find_distributed(
            tree, imports, source.splitlines(), script_path
        )
        # Priority: fsdp > deepspeed > ddp
        # data_parallel is single-process (not a distributed strategy) — ignored.
        kinds = {d.kind for d in distributed}
        if "fsdp" in kinds:
            return "fsdp"
        if "deepspeed" in kinds:
            return "deepspeed"
        if "ddp" in kinds:
            return "ddp"
        return None
    except Exception:
        return None


# Import resolution


def _walk_imports(tree: ast.AST) -> Dict[str, str]:
    """Build alias -> full module path map from import statements."""
    imports = {}  # type: Dict[str, str]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                imports[name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                full = f"{module}.{alias.name}" if module else alias.name
                imports[name] = full
    return imports


# Helper: resolve a call node to a fully-qualified name


def _resolve_call_name(
    node: ast.Call, imports: Dict[str, str]
) -> Optional[str]:
    """Resolve a Call node's function to a fully-qualified name using imports."""
    func = node.func

    # Simple name: DataLoader(...)
    if isinstance(func, ast.Name):
        return imports.get(func.id, func.id)

    # Attribute chain: torch.utils.data.DataLoader(...) or data.DataLoader(...)
    if isinstance(func, ast.Attribute):
        parts = []
        current = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            parts.reverse()
            # Resolve the root name through imports
            root = parts[0]
            resolved_root = imports.get(root, root)
            return (
                resolved_root + "." + ".".join(parts[1:])
                if len(parts) > 1
                else resolved_root
            )

    return None


def _get_kwarg_constant(node: ast.Call, name: str):
    """Extract a constant keyword argument value from a Call node. Returns None if not found or dynamic."""
    for kw in node.keywords:
        if kw.arg == name:
            if isinstance(kw.value, ast.Constant):
                return kw.value.value
            return None
    return None


def _get_kwarg_str(node: ast.Call, name: str) -> Optional[str]:
    """Extract string representation of a keyword argument for raw_kwargs."""
    for kw in node.keywords:
        if kw.arg == name:
            if isinstance(kw.value, ast.Constant):
                return repr(kw.value.value)
            if isinstance(kw.value, ast.Name):
                return kw.value.id
            return "<dynamic>"
    return None


def _has_kwarg(node: ast.Call, name: str) -> bool:
    """Check if a keyword argument is present."""
    return any(kw.arg == name for kw in node.keywords)


def _source_line(lines: List[str], lineno: int) -> str:
    """Get source line (1-indexed), returning empty string if out of range."""
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1]
    return ""


def _loc(script_path: str, node: ast.AST, lines: List[str]) -> SourceLocation:
    """Build a SourceLocation from an AST node."""
    lineno = getattr(node, "lineno", 0)
    col = getattr(node, "col_offset", 0)
    return SourceLocation(
        file_path=script_path,
        line_number=lineno,
        col_offset=col,
        source_line=_source_line(lines, lineno),
    )


# DataLoader detection

_DATALOADER_NAMES = {
    "torch.utils.data.DataLoader",
    "torch.utils.data.dataloader.DataLoader",
}


def _is_dataloader_call(fqn: Optional[str]) -> bool:
    """Check if a fully-qualified name matches DataLoader."""
    if fqn is None:
        return False
    return fqn in _DATALOADER_NAMES or fqn.endswith(".DataLoader")


def _find_dataloaders(
    tree: ast.AST,
    imports: Dict[str, str],
    lines: List[str],
    script_path: str,
) -> List[DataLoaderFinding]:
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fqn = _resolve_call_name(node, imports)
        if not _is_dataloader_call(fqn):
            continue

        num_workers = _get_kwarg_constant(node, "num_workers")
        if num_workers is not None:
            try:
                num_workers = int(num_workers)
            except (ValueError, TypeError):
                num_workers = None

        batch_size = _get_kwarg_constant(node, "batch_size")
        if batch_size is not None:
            try:
                batch_size = int(batch_size)
            except (ValueError, TypeError):
                batch_size = None

        pin_memory = _get_kwarg_constant(node, "pin_memory")
        if pin_memory is not None:
            pin_memory = bool(pin_memory)

        persistent_workers = _get_kwarg_constant(node, "persistent_workers")
        if persistent_workers is not None:
            persistent_workers = bool(persistent_workers)

        prefetch_factor = _get_kwarg_constant(node, "prefetch_factor")
        if prefetch_factor is not None:
            try:
                prefetch_factor = int(prefetch_factor)
            except (ValueError, TypeError):
                prefetch_factor = None

        raw_kwargs = {}  # type: Dict[str, str]
        for kw_name in (
            "num_workers",
            "batch_size",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
        ):
            val = _get_kwarg_str(node, kw_name)
            if val is not None:
                raw_kwargs[kw_name] = val

        results.append(
            DataLoaderFinding(
                location=_loc(script_path, node, lines),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                raw_kwargs=raw_kwargs,
            )
        )

    return results


# Optimizer detection

_OPTIMIZER_NAMES = {
    "torch.optim.Adam": "Adam",
    "torch.optim.AdamW": "AdamW",
    "torch.optim.SGD": "SGD",
    "torch.optim.RMSprop": "RMSprop",
    "torch.optim.Adagrad": "Adagrad",
    # Extended optimizer detection
    "lion_pytorch.Lion": "Lion",
    "transformers.optimization.Adafactor": "Adafactor",
    "torch.optim.LBFGS": "LBFGS",
    "bitsandbytes.optim.Adam8bit": "Adam8bit",
    "bitsandbytes.optim.AdamW8bit": "AdamW8bit",
    "apex.optimizers.FusedAdam": "FusedAdam",
    "apex.optimizers.FusedLAMB": "FusedLAMB",
    "deepspeed.ops.adam.FusedAdam": "FusedAdam",
    "deepspeed.ops.adam.DeepSpeedCPUAdam": "DeepSpeedCPUAdam",
}


def _find_optimizers(
    tree: ast.AST,
    imports: Dict[str, str],
    lines: List[str],
    script_path: str,
) -> List[OptimizerFinding]:
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fqn = _resolve_call_name(node, imports)
        if fqn is None:
            continue

        opt_type = None
        for known_fqn, name in _OPTIMIZER_NAMES.items():
            if fqn == known_fqn or fqn.endswith(f".{name}") or fqn == name:
                opt_type = name
                break

        if opt_type is None:
            continue

        lr = _get_kwarg_constant(node, "lr")
        if lr is not None:
            try:
                lr = float(lr)
            except (ValueError, TypeError):
                lr = None

        wd = _get_kwarg_constant(node, "weight_decay")
        if wd is not None:
            try:
                wd = float(wd)
            except (ValueError, TypeError):
                wd = None

        results.append(
            OptimizerFinding(
                location=_loc(script_path, node, lines),
                optimizer_type=opt_type,
                learning_rate=lr,
                weight_decay=wd,
            )
        )

    return results


# Precision detection


def _find_precision(
    tree: ast.AST,
    imports: Dict[str, str],
    lines: List[str],
    script_path: str,
) -> List[PrecisionFinding]:
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fqn = _resolve_call_name(node, imports)
        if fqn is None:
            continue

        # autocast patterns
        if "autocast" in fqn:
            is_deprecated = "cuda.amp.autocast" in fqn
            dtype_str = _extract_dtype_arg(node)
            results.append(
                PrecisionFinding(
                    location=_loc(script_path, node, lines),
                    kind="autocast",
                    dtype_str=dtype_str,
                    is_deprecated_api=is_deprecated,
                )
            )
            continue

        # GradScaler
        if "GradScaler" in fqn:
            results.append(
                PrecisionFinding(
                    location=_loc(script_path, node, lines),
                    kind="grad_scaler",
                    dtype_str=None,
                    is_deprecated_api="cuda.amp.GradScaler" in fqn,
                )
            )
            continue

    # Also check for .half() and .bfloat16() method calls
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "half":
                results.append(
                    PrecisionFinding(
                        location=_loc(script_path, node, lines),
                        kind="half",
                        dtype_str="float16",
                        is_deprecated_api=False,
                    )
                )
            elif func.attr == "bfloat16":
                results.append(
                    PrecisionFinding(
                        location=_loc(script_path, node, lines),
                        kind="bfloat16",
                        dtype_str="bfloat16",
                        is_deprecated_api=False,
                    )
                )

    return results


def _extract_dtype_arg(node: ast.Call) -> Optional[str]:
    """Extract dtype argument from autocast() call."""
    for kw in node.keywords:
        if kw.arg == "dtype":
            return _dtype_node_to_str(kw.value)
    if len(node.args) >= 2:
        return _dtype_node_to_str(node.args[1])
    return None


def _dtype_node_to_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


# Distributed detection


def _find_distributed(
    tree: ast.AST,
    imports: Dict[str, str],
    lines: List[str],
    script_path: str,
) -> List[DistributedFinding]:
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fqn = _resolve_call_name(node, imports)
        if fqn is None:
            continue

        if "DistributedDataParallel" in fqn or fqn.endswith(".DDP"):
            results.append(
                DistributedFinding(
                    location=_loc(script_path, node, lines),
                    kind="ddp",
                    backend=None,
                )
            )
            continue

        if (
            fqn
            and "DataParallel" in fqn
            and "Distributed" not in fqn
            and "FullyShard" not in fqn
        ):
            results.append(
                DistributedFinding(
                    location=_loc(script_path, node, lines),
                    kind="data_parallel",
                    backend=None,
                )
            )
            continue

        if "FullyShardedDataParallel" in fqn or fqn.endswith(".FSDP"):
            results.append(
                DistributedFinding(
                    location=_loc(script_path, node, lines),
                    kind="fsdp",
                    backend=None,
                )
            )
            continue

        if "init_process_group" in fqn:
            backend = None
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                backend = node.args[0].value
            else:
                backend_val = _get_kwarg_constant(node, "backend")
                if isinstance(backend_val, str):
                    backend = backend_val
            results.append(
                DistributedFinding(
                    location=_loc(script_path, node, lines),
                    kind="init_process_group",
                    backend=backend,
                )
            )
            continue

        if fqn == "torch.compile" or (
            fqn.endswith(".compile") and "torch" in fqn
        ):
            results.append(
                DistributedFinding(
                    location=_loc(script_path, node, lines),
                    kind="torch_compile",
                    backend=None,
                )
            )
            continue

        if "deepspeed" in fqn and "initialize" in fqn:
            results.append(
                DistributedFinding(
                    location=_loc(script_path, node, lines),
                    kind="deepspeed",
                    backend=None,
                )
            )
            continue

        if (
            fqn
            and ("Accelerator" in fqn)
            and "accelerate" in imports.get(fqn.split(".")[0], fqn).lower()
        ):
            results.append(
                DistributedFinding(
                    location=_loc(script_path, node, lines),
                    kind="accelerate",
                    backend=None,
                )
            )
            continue

    _lightning_prefixes = (
        "pytorch_lightning",
        "lightning.pytorch",
        "lightning",
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fqn = _resolve_call_name(node, imports)
        if fqn is None:
            continue
        if (
            any(fqn.startswith(pfx) for pfx in _lightning_prefixes)
            and "Trainer" in fqn
        ):
            if not any(d.kind == "lightning" for d in results):
                results.append(
                    DistributedFinding(
                        location=_loc(script_path, node, lines),
                        kind="lightning",
                        backend=None,
                    )
                )
            break
        if fqn == "Trainer" or fqn.endswith(".Trainer"):
            src = imports.get("Trainer", "")
            if any(src.startswith(pfx) for pfx in _lightning_prefixes):
                if not any(d.kind == "lightning" for d in results):
                    results.append(
                        DistributedFinding(
                            location=_loc(script_path, node, lines),
                            kind="lightning",
                            backend=None,
                        )
                    )
                break

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name == "LightningModule":
                src = imports.get("LightningModule", "")
                if not src or any(
                    src.startswith(pfx) for pfx in _lightning_prefixes
                ):
                    if not any(d.kind == "lightning" for d in results):
                        results.append(
                            DistributedFinding(
                                location=_loc(script_path, node, lines),
                                kind="lightning",
                                backend=None,
                            )
                        )
                    break

    return results


# Model detection


def _find_models(
    tree: ast.AST,
    imports: Dict[str, str],
    lines: List[str],
    script_path: str,
) -> List[ModelFinding]:
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fqn = _resolve_call_name(node, imports)
            if fqn and "from_pretrained" in fqn:
                model_name = None
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    model_name = node.args[0].value
                results.append(
                    ModelFinding(
                        location=_loc(script_path, node, lines),
                        kind="from_pretrained",
                        model_name=model_name,
                    )
                )
                continue

            if fqn and "gradient_checkpointing_enable" in fqn:
                results.append(
                    ModelFinding(
                        location=_loc(script_path, node, lines),
                        kind="gradient_checkpointing",
                        model_name=None,
                    )
                )
                continue

            if fqn and "set_float32_matmul_precision" in fqn:
                results.append(
                    ModelFinding(
                        location=_loc(script_path, node, lines),
                        kind="float32_matmul_precision",
                        model_name=None,
                    )
                )
                continue

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "benchmark"
                ):
                    if _is_cudnn_benchmark(target):
                        results.append(
                            ModelFinding(
                                location=_loc(script_path, node, lines),
                                kind="cudnn_benchmark",
                                model_name=None,
                            )
                        )

    return results


def _is_cudnn_benchmark(node: ast.Attribute) -> bool:
    parts = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    parts.reverse()
    full = ".".join(parts)
    return "cudnn" in full and "benchmark" in full


# Fine-tuning / PEFT detection

_PEFT_LORA_NAMES = {
    "peft.LoraConfig": "lora",
    "LoraConfig": "lora",
    "peft.get_peft_model": "peft_generic",
    "get_peft_model": "peft_generic",
    "peft.AdaLoraConfig": "lora",
    "AdaLoraConfig": "lora",
    "peft.IA3Config": "peft_generic",
    "IA3Config": "peft_generic",
    "peft.PrefixTuningConfig": "adapter",
    "PrefixTuningConfig": "adapter",
    "peft.PromptTuningConfig": "adapter",
    "PromptTuningConfig": "adapter",
}

_QLORA_MARKERS = {
    "prepare_model_for_kbit_training",
    "BitsAndBytesConfig",
}


def _find_fine_tuning(
    tree: ast.AST,
    imports: Dict[str, str],
    lines: List[str],
    script_path: str,
) -> List[FineTuningFinding]:
    results = []
    has_qlora_marker = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fqn = _resolve_call_name(node, imports)
        if fqn is None:
            continue

        for known_fqn, method in _PEFT_LORA_NAMES.items():
            if fqn == known_fqn or fqn.endswith(
                f".{known_fqn.split('.')[-1]}"
            ):
                details = {}
                r_val = _get_kwarg_constant(node, "r")
                if r_val is not None:
                    details["rank"] = str(r_val)
                lora_alpha = _get_kwarg_constant(node, "lora_alpha")
                if lora_alpha is not None:
                    details["lora_alpha"] = str(lora_alpha)
                results.append(
                    FineTuningFinding(
                        location=_loc(script_path, node, lines),
                        method=method,
                        details=details,
                    )
                )
                break

        for marker in _QLORA_MARKERS:
            if fqn and (fqn == marker or fqn.endswith(f".{marker}")):
                has_qlora_marker = True
                break

    if has_qlora_marker:
        for ft in results:
            if ft.method == "lora":
                ft.method = "qlora"

    return results


# HuggingFace TrainingArguments extraction


_TRAINING_ARGS_NAMES = {
    "transformers.TrainingArguments",
    "transformers.Seq2SeqTrainingArguments",
    "TrainingArguments",
    "Seq2SeqTrainingArguments",
}

_HF_TRAINER_NAMES = {
    "transformers.Trainer",
    "transformers.Seq2SeqTrainer",
    "Trainer",
    "Seq2SeqTrainer",
    "trl.SFTTrainer",
    "trl.DPOTrainer",
    "trl.PPOTrainer",
    "trl.RewardTrainer",
    "SFTTrainer",
    "DPOTrainer",
    "PPOTrainer",
    "RewardTrainer",
}


def _extract_training_args(
    tree: ast.AST,
    findings: CodeFindings,
    lines: List[str],
    script_path: str,
) -> None:
    imports = findings.imports

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fqn = _resolve_call_name(node, imports)
        if fqn is None:
            continue

        is_training_args = False
        for ta_name in _TRAINING_ARGS_NAMES:
            if fqn == ta_name or fqn.endswith(f".{ta_name.split('.')[-1]}"):
                is_training_args = True
                break

        if not is_training_args:
            is_trainer = False
            for t_name in _HF_TRAINER_NAMES:
                if fqn == t_name or fqn.endswith(f".{t_name.split('.')[-1]}"):
                    is_trainer = True
                    break
            if is_trainer:
                findings.distributed.append(
                    DistributedFinding(
                        location=_loc(script_path, node, lines),
                        kind="hf_trainer",
                        backend=None,
                    )
                )
            continue

        fp16 = _get_kwarg_constant(node, "fp16")
        bf16 = _get_kwarg_constant(node, "bf16")
        grad_accum = _get_kwarg_constant(node, "gradient_accumulation_steps")
        deepspeed_arg = _has_kwarg(node, "deepspeed")

        if fp16 is True:
            findings.precision.append(
                PrecisionFinding(
                    location=_loc(script_path, node, lines),
                    kind="autocast",
                    dtype_str="float16",
                    is_deprecated_api=False,
                )
            )

        if bf16 is True:
            findings.precision.append(
                PrecisionFinding(
                    location=_loc(script_path, node, lines),
                    kind="autocast",
                    dtype_str="bfloat16",
                    is_deprecated_api=False,
                )
            )

        if grad_accum is not None:
            try:
                findings.gradient_accumulation_steps = int(grad_accum)
            except (ValueError, TypeError):
                pass

        if deepspeed_arg:
            if not any(d.kind == "deepspeed" for d in findings.distributed):
                findings.distributed.append(
                    DistributedFinding(
                        location=_loc(script_path, node, lines),
                        kind="deepspeed",
                        backend=None,
                    )
                )


# Training loop detection


def _detect_training_loop(
    tree: ast.AST, findings: Optional[CodeFindings] = None
) -> bool:
    if findings and any(d.kind == "hf_trainer" for d in findings.distributed):
        return True

    has_backward = False
    has_step = False
    has_for_loop = False
    has_train_call = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "backward":
                has_backward = True
            elif attr == "step":
                has_step = True
            elif attr == "train":
                has_train_call = True

        if isinstance(node, ast.For):
            has_for_loop = True

    if has_backward and has_step:
        return True

    if has_for_loop and has_train_call:
        return True

    return False


# Multi-file import following (1-level deep)


def _follow_local_imports(
    tree: ast.AST,
    findings: CodeFindings,
    script_path: str,
) -> None:
    script_dir = os.path.dirname(os.path.abspath(script_path))
    followed = set()  # type: set

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue

        if isinstance(node, ast.ImportFrom):
            module_name = node.module
            if module_name is None:
                continue
            if "." in module_name:
                continue
            candidate = os.path.join(script_dir, module_name + ".py")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if "." in name:
                    continue
                candidate = os.path.join(script_dir, name + ".py")
                if os.path.isfile(candidate) and candidate not in followed:
                    followed.add(candidate)
                    _merge_imported_findings(candidate, findings)
            continue
        else:
            continue

        if not os.path.isfile(candidate):
            continue
        if candidate in followed:
            continue

        followed.add(candidate)
        _merge_imported_findings(candidate, findings)


def _merge_imported_findings(
    imported_path: str,
    main_findings: CodeFindings,
) -> None:
    try:
        with open(imported_path, "r", encoding="utf-8") as f:
            source = f.read()
        lines = source.splitlines()
        tree = ast.parse(source, filename=imported_path)
    except (SyntaxError, OSError):
        return

    imports = _walk_imports(tree)

    for model in _find_models(tree, imports, lines, imported_path):
        main_findings.models.append(model)

    for dist in _find_distributed(tree, imports, lines, imported_path):
        main_findings.distributed.append(dist)

    for prec in _find_precision(tree, imports, lines, imported_path):
        main_findings.precision.append(prec)

    for opt in _find_optimizers(tree, imports, lines, imported_path):
        main_findings.optimizers.append(opt)

    for ft in _find_fine_tuning(tree, imports, lines, imported_path):
        main_findings.fine_tuning.append(ft)

    sub_findings = CodeFindings(script_path=imported_path)
    sub_findings.imports = imports
    sub_findings.distributed = list(main_findings.distributed)
    _extract_training_args(tree, sub_findings, lines, imported_path)

    for prec in sub_findings.precision:
        if prec not in main_findings.precision:
            main_findings.precision.append(prec)

    if (
        sub_findings.gradient_accumulation_steps is not None
        and main_findings.gradient_accumulation_steps is None
    ):
        main_findings.gradient_accumulation_steps = (
            sub_findings.gradient_accumulation_steps
        )


# Added Adapter


def scan_for_optimizer(script_path: str) -> Optional[str]:
    """Adapter for traceml suggest-gpu."""
    findings = analyze_script(script_path)
    if findings.optimizers:
        return findings.optimizers[0].optimizer_type
    return None
