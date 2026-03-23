"""TraceML script inspector — static extraction of ML patterns from Python source.

Parses a training script's AST to surface:
  - DataLoader configuration (batch_size, num_workers, pin_memory, …)
  - Optimizer choice and hyperparameters
  - Mixed-precision usage (autocast, GradScaler, .half())
  - Parallelism strategy (DDP, FSDP, DeepSpeed, Lightning, Accelerate)
  - Model provenance (HuggingFace from_pretrained, gradient checkpointing)
  - Fine-tuning adapters (LoRA, QLoRA, PEFT variants)

Design principles
-----------------
* Zero-crash guarantee — every public entry point wraps failures; callers
  always receive a valid result object even for unparsable files.
* Single-pass import resolution — a pre-pass builds a flat alias→fqn map
  so individual detectors need no import-walking of their own.
* Visitor-based detection — the ``_PatternVisitor`` class subclasses
  ``ast.NodeVisitor`` and accumulates all findings in one tree walk,
  avoiding repeated ``ast.walk`` calls for different concerns.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass
class ScriptLocation:
    """Points to the exact line in the source file where a pattern was found."""

    file_path: str
    line: int
    col: int
    text: str  # the raw source line (may be empty)


@dataclass
class DataLoaderFinding:
    location: ScriptLocation
    batch_size: Optional[int]
    num_workers: Optional[int]
    pin_memory: Optional[bool]
    persistent_workers: Optional[bool]
    prefetch_factor: Optional[int]
    raw_kwargs: Dict[str, str]


@dataclass
class OptimizerFinding:
    location: ScriptLocation
    optimizer_type: str
    learning_rate: Optional[float]
    weight_decay: Optional[float]


@dataclass
class PrecisionFinding:
    location: ScriptLocation
    kind: str  # "autocast" | "grad_scaler" | "half" | "bfloat16"
    dtype_str: Optional[str]
    is_deprecated_api: bool


@dataclass
class DistributedFinding:
    location: ScriptLocation
    kind: str  # "ddp" | "fsdp" | "deepspeed" | "data_parallel" |
    #                      "accelerate" | "lightning" | "hf_trainer" |
    #                      "init_process_group" | "torch_compile"
    backend: Optional[str]


@dataclass
class ModelFinding:
    location: ScriptLocation
    kind: str  # "from_pretrained" | "gradient_checkpointing" |
    #                      "cudnn_benchmark" | "float32_matmul_precision"
    model_name: Optional[str]


@dataclass
class FineTuningFinding:
    location: ScriptLocation
    method: str  # "lora" | "qlora" | "peft_generic" | "adapter"
    details: Dict[str, str] = field(default_factory=dict)


@dataclass
class CodeFindings:
    """All ML patterns found in a single script (or an empty result on error)."""

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_script(script_path: str) -> CodeFindings:
    """Return ML pattern findings for *script_path*.

    Never raises — on any failure returns a ``CodeFindings`` with
    ``parse_errors`` populated.
    """
    result = CodeFindings(script_path=script_path)

    if not os.path.isfile(script_path):
        result.parse_errors.append(f"File not found: {script_path}")
        return result

    try:
        with open(script_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        lines = source.splitlines()
        tree = ast.parse(source, filename=script_path)
    except SyntaxError as exc:
        result.parse_errors.append(f"Syntax error: {exc}")
        return result
    except OSError as exc:
        result.parse_errors.append(f"Cannot read file: {exc}")
        return result

    result.imports = _build_import_map(tree)

    # String constants at module level (MODEL_NAME = "some/model")
    str_consts = _collect_string_constants(tree)

    visitor = _PatternVisitor(
        script_path=script_path,
        lines=lines,
        imports=result.imports,
        str_consts=str_consts,
    )
    visitor.visit(tree)

    result.dataloaders = visitor.dataloaders
    result.optimizers = visitor.optimizers
    result.precision = visitor.precision
    result.distributed = visitor.distributed
    result.models = visitor.models
    result.fine_tuning = visitor.fine_tuning
    result.gradient_accumulation_steps = visitor.gradient_accumulation_steps

    _apply_qlora_upgrade(result)
    _absorb_hf_training_args(tree, result, lines, script_path)

    result.has_training_loop = _has_training_loop(tree, result)
    result.has_gradient_checkpointing = any(
        m.kind == "gradient_checkpointing" for m in result.models
    )

    _ingest_local_imports(tree, result, script_path)
    return result


def detect_strategy_hint(script_path: str) -> Optional[str]:
    """Quick check: return the distributed strategy used, or ``None``.

    Returns one of ``'fsdp'``, ``'deepspeed'``, ``'ddp'``, or ``None``.
    Never raises.
    """
    try:
        findings = analyze_script(script_path)
        kinds: Set[str] = {d.kind for d in findings.distributed}
        for strategy in ("fsdp", "deepspeed", "ddp"):
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


# ---------------------------------------------------------------------------
# Import map construction
# ---------------------------------------------------------------------------


def _build_import_map(tree: ast.AST) -> Dict[str, str]:
    """Return a mapping of local name → fully-qualified module path.

    Examples::

        import torch                      → {"torch": "torch"}
        from torch.utils.data import DataLoader  → {"DataLoader": "torch.utils.data.DataLoader"}
        from torch import nn as neural    → {"neural": "torch.nn"}
    """
    mapping: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname if alias.asname else alias.name
                mapping[local] = alias.name

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname if alias.asname else alias.name
                fqn = f"{module}.{alias.name}" if module else alias.name
                mapping[local] = fqn

    return mapping


def _collect_string_constants(tree: ast.AST) -> Dict[str, str]:
    """Return all ``NAME = "string"`` assignments found anywhere in the file.

    Scans all scopes (module, function bodies, class methods) so that patterns
    like ``model_name = "bert-base"`` inside a ``main()`` function are still
    resolved when passed to ``from_pretrained(model_name)``.
    """
    consts: Dict[str, str] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            # Last assignment wins (later definitions override earlier ones)
            consts[node.targets[0].id] = node.value.value
    return consts


# ---------------------------------------------------------------------------
# Name resolution helpers
# ---------------------------------------------------------------------------


def _fqn(call: ast.Call, imports: Dict[str, str]) -> Optional[str]:
    """Resolve an ``ast.Call``'s function to a fully-qualified name string."""
    func = call.func

    if isinstance(func, ast.Name):
        return imports.get(func.id, func.id)

    if isinstance(func, ast.Attribute):
        chain: List[str] = []
        cur = func
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            chain.append(cur.id)
            chain.reverse()
            root_resolved = imports.get(chain[0], chain[0])
            tail = ".".join(chain[1:])
            return f"{root_resolved}.{tail}" if tail else root_resolved

    return None


def _kw_const(call: ast.Call, name: str):
    """Return the constant value of keyword argument *name*, or ``None``."""
    for kw in call.keywords:
        if kw.arg == name:
            return (
                kw.value.value if isinstance(kw.value, ast.Constant) else None
            )
    return None


def _kw_repr(call: ast.Call, name: str) -> Optional[str]:
    """Return a string representation of keyword argument *name* for display."""
    for kw in call.keywords:
        if kw.arg == name:
            if isinstance(kw.value, ast.Constant):
                return repr(kw.value.value)
            if isinstance(kw.value, ast.Name):
                return kw.value.id
            return "<dynamic>"
    return None


def _has_kw(call: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in call.keywords)


def _src_line(lines: List[str], lineno: int) -> str:
    return lines[lineno - 1] if 1 <= lineno <= len(lines) else ""


def _location(path: str, node: ast.AST, lines: List[str]) -> ScriptLocation:
    ln = getattr(node, "lineno", 0)
    col = getattr(node, "col_offset", 0)
    return ScriptLocation(
        file_path=path, line=ln, col=col, text=_src_line(lines, ln)
    )


def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Known patterns
# ---------------------------------------------------------------------------

_DATALOADER_SUFFIXES = {"DataLoader", "dataloader.DataLoader"}

_KNOWN_OPTIMIZERS: Dict[str, str] = {
    # PyTorch built-ins
    "torch.optim.Adam": "Adam",
    "torch.optim.AdamW": "AdamW",
    "torch.optim.SGD": "SGD",
    "torch.optim.RMSprop": "RMSprop",
    "torch.optim.Adagrad": "Adagrad",
    "torch.optim.LBFGS": "LBFGS",
    # Third-party
    "bitsandbytes.optim.Adam8bit": "Adam8bit",
    "bitsandbytes.optim.AdamW8bit": "AdamW8bit",
    "apex.optimizers.FusedAdam": "FusedAdam",
    "apex.optimizers.FusedLAMB": "FusedLAMB",
    "deepspeed.ops.adam.FusedAdam": "FusedAdam",
    "deepspeed.ops.adam.DeepSpeedCPUAdam": "DeepSpeedCPUAdam",
    "lion_pytorch.Lion": "Lion",
    "transformers.optimization.Adafactor": "Adafactor",
}

_PEFT_METHODS: Dict[str, str] = {
    "LoraConfig": "lora",
    "peft.LoraConfig": "lora",
    "AdaLoraConfig": "lora",
    "peft.AdaLoraConfig": "lora",
    "get_peft_model": "peft_generic",
    "peft.get_peft_model": "peft_generic",
    "IA3Config": "peft_generic",
    "peft.IA3Config": "peft_generic",
    "PrefixTuningConfig": "adapter",
    "peft.PrefixTuningConfig": "adapter",
    "PromptTuningConfig": "adapter",
    "peft.PromptTuningConfig": "adapter",
}

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

_LIGHTNING_PREFIXES = ("pytorch_lightning", "lightning.pytorch", "lightning")


# ---------------------------------------------------------------------------
# Visitor — single pass over the AST
# ---------------------------------------------------------------------------


class _PatternVisitor(ast.NodeVisitor):
    """Accumulates all ML pattern findings in a single AST traversal."""

    def __init__(
        self,
        script_path: str,
        lines: List[str],
        imports: Dict[str, str],
        str_consts: Dict[str, str],
    ) -> None:
        self._path = script_path
        self._lines = lines
        self._imports = imports
        self._str_consts = str_consts

        self.dataloaders: List[DataLoaderFinding] = []
        self.optimizers: List[OptimizerFinding] = []
        self.precision: List[PrecisionFinding] = []
        self.distributed: List[DistributedFinding] = []
        self.models: List[ModelFinding] = []
        self.fine_tuning: List[FineTuningFinding] = []
        self.gradient_accumulation_steps: Optional[int] = None

        # Dedup guards
        self._seen_lightning = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _loc(self, node: ast.AST) -> ScriptLocation:
        return _location(self._path, node, self._lines)

    def _resolve(self, call: ast.Call) -> Optional[str]:
        return _fqn(call, self._imports)

    def _add_dist(
        self, node: ast.AST, kind: str, backend: Optional[str] = None
    ) -> None:
        self.distributed.append(
            DistributedFinding(
                location=self._loc(node), kind=kind, backend=backend
            )
        )

    # ------------------------------------------------------------------
    # Call node — the main dispatch point
    # ------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = self._resolve(node)
        if name is not None:
            self._check_dataloader(node, name)
            self._check_optimizer(node, name)
            self._check_precision_call(node, name)
            self._check_distributed(node, name)
            self._check_model(node, name)
            self._check_peft(node, name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        # .half() / .bfloat16() appear as method calls but also as assignments
        # Check for cudnn.benchmark = True
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "benchmark"
            ):
                if _is_cudnn_attr(target):
                    self.models.append(
                        ModelFinding(
                            location=self._loc(node),
                            kind="cudnn_benchmark",
                            model_name=None,
                        )
                    )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        # Detect LightningModule subclasses
        if not self._seen_lightning:
            for base in node.bases:
                bname = (
                    base.id
                    if isinstance(base, ast.Name)
                    else base.attr if isinstance(base, ast.Attribute) else None
                )
                if bname == "LightningModule":
                    src = self._imports.get("LightningModule", "")
                    if not src or any(
                        src.startswith(p) for p in _LIGHTNING_PREFIXES
                    ):
                        self._seen_lightning = True
                        self._add_dist(node, "lightning")
                        break
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Per-pattern checks (called from visit_Call)
    # ------------------------------------------------------------------

    def _check_dataloader(self, node: ast.Call, name: str) -> None:
        if not (name.endswith(".DataLoader") or name in _DATALOADER_SUFFIXES):
            return

        raw: Dict[str, str] = {}
        for kw_name in (
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
        ):
            v = _kw_repr(node, kw_name)
            if v is not None:
                raw[kw_name] = v

        bs = _safe_int(_kw_const(node, "batch_size"))
        nw = _safe_int(_kw_const(node, "num_workers"))
        pm_raw = _kw_const(node, "pin_memory")
        pm = bool(pm_raw) if pm_raw is not None else None
        pw_raw = _kw_const(node, "persistent_workers")
        pw = bool(pw_raw) if pw_raw is not None else None
        pf = _safe_int(_kw_const(node, "prefetch_factor"))

        self.dataloaders.append(
            DataLoaderFinding(
                location=self._loc(node),
                batch_size=bs,
                num_workers=nw,
                pin_memory=pm,
                persistent_workers=pw,
                prefetch_factor=pf,
                raw_kwargs=raw,
            )
        )

    def _check_optimizer(self, node: ast.Call, name: str) -> None:
        opt_type: Optional[str] = None
        for fqn_key, label in _KNOWN_OPTIMIZERS.items():
            leaf = fqn_key.rsplit(".", 1)[-1]
            if name == fqn_key or name == leaf or name.endswith(f".{leaf}"):
                opt_type = label
                break
        if opt_type is None:
            return

        self.optimizers.append(
            OptimizerFinding(
                location=self._loc(node),
                optimizer_type=opt_type,
                learning_rate=_safe_float(_kw_const(node, "lr")),
                weight_decay=_safe_float(_kw_const(node, "weight_decay")),
            )
        )

    def _check_precision_call(self, node: ast.Call, name: str) -> None:
        if "autocast" in name:
            dtype_str = _extract_autocast_dtype(node)
            self.precision.append(
                PrecisionFinding(
                    location=self._loc(node),
                    kind="autocast",
                    dtype_str=dtype_str,
                    is_deprecated_api="cuda.amp.autocast" in name,
                )
            )
            return

        if "GradScaler" in name:
            self.precision.append(
                PrecisionFinding(
                    location=self._loc(node),
                    kind="grad_scaler",
                    dtype_str=None,
                    is_deprecated_api="cuda.amp.GradScaler" in name,
                )
            )
            return

        # .half() / .bfloat16() method calls
        if isinstance(node.func, ast.Attribute):
            meth = node.func.attr
            if meth == "half":
                self.precision.append(
                    PrecisionFinding(
                        location=self._loc(node),
                        kind="half",
                        dtype_str="float16",
                        is_deprecated_api=False,
                    )
                )
            elif meth == "bfloat16":
                self.precision.append(
                    PrecisionFinding(
                        location=self._loc(node),
                        kind="bfloat16",
                        dtype_str="bfloat16",
                        is_deprecated_api=False,
                    )
                )

    def _check_distributed(self, node: ast.Call, name: str) -> None:
        if "DistributedDataParallel" in name or name.endswith(".DDP"):
            self._add_dist(node, "ddp")
            return

        if (
            "DataParallel" in name
            and "Distributed" not in name
            and "FullyShard" not in name
        ):
            self._add_dist(node, "data_parallel")
            return

        if "FullyShardedDataParallel" in name or name.endswith(".FSDP"):
            self._add_dist(node, "fsdp")
            return

        if "init_process_group" in name:
            backend: Optional[str] = None
            if node.args and isinstance(node.args[0], ast.Constant):
                backend = str(node.args[0].value)
            else:
                raw = _kw_const(node, "backend")
                if isinstance(raw, str):
                    backend = raw
            self._add_dist(node, "init_process_group", backend)
            return

        if (
            name == "torch.compile" or name.endswith(".compile")
        ) and "torch" in name:
            self._add_dist(node, "torch_compile")
            return

        if "deepspeed" in name and "initialize" in name:
            self._add_dist(node, "deepspeed")
            return

        if (
            "Accelerator" in name
            and "accelerate"
            in self._imports.get(name.split(".")[0], name).lower()
        ):
            self._add_dist(node, "accelerate")
            return

        # Lightning Trainer (not a class def — instantiation)
        if not self._seen_lightning:
            if (
                any(name.startswith(p) for p in _LIGHTNING_PREFIXES)
                and "Trainer" in name
            ):
                self._seen_lightning = True
                self._add_dist(node, "lightning")
                return
            if name == "Trainer" or name.endswith(".Trainer"):
                src = self._imports.get("Trainer", "")
                if any(src.startswith(p) for p in _LIGHTNING_PREFIXES):
                    self._seen_lightning = True
                    self._add_dist(node, "lightning")
                    return

    def _check_model(self, node: ast.Call, name: str) -> None:
        if "from_pretrained" in name:
            model_id: Optional[str] = None
            if node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(
                    arg0.value, str
                ):
                    model_id = arg0.value
                elif (
                    isinstance(arg0, ast.Name) and arg0.id in self._str_consts
                ):
                    model_id = self._str_consts[arg0.id]
            self.models.append(
                ModelFinding(
                    location=self._loc(node),
                    kind="from_pretrained",
                    model_name=model_id,
                )
            )
            return

        if "gradient_checkpointing_enable" in name:
            self.models.append(
                ModelFinding(
                    location=self._loc(node),
                    kind="gradient_checkpointing",
                    model_name=None,
                )
            )
            return

        if "set_float32_matmul_precision" in name:
            self.models.append(
                ModelFinding(
                    location=self._loc(node),
                    kind="float32_matmul_precision",
                    model_name=None,
                )
            )

    def _check_peft(self, node: ast.Call, name: str) -> None:
        for key, method in _PEFT_METHODS.items():
            leaf = key.rsplit(".", 1)[-1]
            if name == key or name == leaf or name.endswith(f".{leaf}"):
                details: Dict[str, str] = {}
                r = _kw_const(node, "r")
                if r is not None:
                    details["rank"] = str(r)
                alpha = _kw_const(node, "lora_alpha")
                if alpha is not None:
                    details["lora_alpha"] = str(alpha)
                self.fine_tuning.append(
                    FineTuningFinding(
                        location=self._loc(node),
                        method=method,
                        details=details,
                    )
                )
                return

        # QLoRA signals (not PEFT configs themselves but mark the run as QLoRA)
        leaf = name.rsplit(".", 1)[-1]
        if leaf in _QLORA_SIGNALS or name in _QLORA_SIGNALS:
            # Recorded via _apply_qlora_upgrade; nothing to append here.
            pass


# ---------------------------------------------------------------------------
# Post-processing passes
# ---------------------------------------------------------------------------


def _apply_qlora_upgrade(result: CodeFindings) -> None:
    """Upgrade LoRA findings to QLoRA when k-bit quantisation signals are present."""
    src = result.script_path
    try:
        with open(src, "r", encoding="utf-8") as fh:
            source = fh.read()
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
    lines: List[str],
    script_path: str,
) -> None:
    """Extract precision and grad-accum from HuggingFace TrainingArguments."""
    imports = result.imports
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _fqn(node, imports)
        if name is None:
            continue
        leaf = name.rsplit(".", 1)[-1]

        # HF Trainer → mark as hf_trainer distributed strategy
        if name in _HF_TRAINER_CLASSES or leaf in {
            n.rsplit(".", 1)[-1] for n in _HF_TRAINER_CLASSES
        }:
            if not any(d.kind == "hf_trainer" for d in result.distributed):
                result.distributed.append(
                    DistributedFinding(
                        location=_location(script_path, node, lines),
                        kind="hf_trainer",
                        backend=None,
                    )
                )
            continue

        # TrainingArguments → extract fp16/bf16/grad_accum/deepspeed
        if name not in _HF_TRAINING_ARG_CLASSES and leaf not in {
            n.rsplit(".", 1)[-1] for n in _HF_TRAINING_ARG_CLASSES
        }:
            continue

        loc = _location(script_path, node, lines)
        if _kw_const(node, "fp16") is True:
            result.precision.append(
                PrecisionFinding(
                    location=loc,
                    kind="autocast",
                    dtype_str="float16",
                    is_deprecated_api=False,
                )
            )
        if _kw_const(node, "bf16") is True:
            result.precision.append(
                PrecisionFinding(
                    location=loc,
                    kind="autocast",
                    dtype_str="bfloat16",
                    is_deprecated_api=False,
                )
            )
        ga = _kw_const(node, "gradient_accumulation_steps")
        if ga is not None and result.gradient_accumulation_steps is None:
            result.gradient_accumulation_steps = _safe_int(ga)
        if _has_kw(node, "deepspeed") and not any(
            d.kind == "deepspeed" for d in result.distributed
        ):
            result.distributed.append(
                DistributedFinding(
                    location=loc, kind="deepspeed", backend=None
                )
            )


def _has_training_loop(tree: ast.AST, result: CodeFindings) -> bool:
    """Heuristic: True when the script contains a recognisable backward pass."""
    if any(d.kind == "hf_trainer" for d in result.distributed):
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


# ---------------------------------------------------------------------------
# Multi-file: follow relative imports one level deep
# ---------------------------------------------------------------------------


def _ingest_local_imports(
    tree: ast.AST,
    result: CodeFindings,
    script_path: str,
) -> None:
    """Scan sibling .py files referenced by ``import X`` / ``from X import ...``."""
    base_dir = os.path.dirname(os.path.abspath(script_path))
    visited: Set[str] = set()

    for node in ast.walk(tree):
        candidates: List[str] = []

        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod and "." not in mod:
                candidates.append(os.path.join(base_dir, mod + ".py"))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "." not in alias.name:
                    candidates.append(
                        os.path.join(base_dir, alias.name + ".py")
                    )

        for cand in candidates:
            if not os.path.isfile(cand) or cand in visited:
                continue
            visited.add(cand)
            _merge_file_into(cand, result)


def _merge_file_into(path: str, result: CodeFindings) -> None:
    """Parse *path* and append its findings into *result* (models, precision, etc.)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        lines = source.splitlines()
        tree = ast.parse(source, filename=path)
    except (SyntaxError, OSError):
        return

    imports = _build_import_map(tree)
    str_consts = _collect_string_constants(tree)
    visitor = _PatternVisitor(
        script_path=path,
        lines=lines,
        imports=imports,
        str_consts=str_consts,
    )
    visitor.visit(tree)

    result.models.extend(visitor.models)
    result.distributed.extend(visitor.distributed)
    result.precision.extend(visitor.precision)
    result.optimizers.extend(visitor.optimizers)
    result.fine_tuning.extend(visitor.fine_tuning)

    if (
        visitor.gradient_accumulation_steps is not None
        and result.gradient_accumulation_steps is None
    ):
        result.gradient_accumulation_steps = (
            visitor.gradient_accumulation_steps
        )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _extract_autocast_dtype(node: ast.Call) -> Optional[str]:
    """Pull the dtype name from an ``autocast(dtype=...)`` call."""
    for kw in node.keywords:
        if kw.arg == "dtype":
            return _dtype_attr_to_str(kw.value)
    if len(node.args) >= 2:
        return _dtype_attr_to_str(node.args[1])
    return None


def _dtype_attr_to_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_cudnn_attr(node: ast.Attribute) -> bool:
    """Return True if *node* is ``torch.backends.cudnn.benchmark``."""
    parts: List[str] = [node.attr]
    cur = node.value
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    joined = ".".join(parts)
    return "cudnn" in joined and "benchmark" in joined
