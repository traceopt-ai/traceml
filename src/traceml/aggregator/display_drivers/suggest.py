"""Suggest GPU Display Driver — Pure-AST static VRAM estimator.

This driver runs entirely offline: it never executes the user's training
script. Instead, it uses the traceml.utils.ast_analysis package to:

  1. Parse the script with Python's ast module
  2. Detect the optimizer (Adam, AdamW, SGD, …)
  3. Detect the precision (fp32/fp16/bf16)
  4. Estimate parameter count via:
       a. Registry lookup for known HuggingFace / common models
       b. Layer counting from nn.Linear / Conv2d / Embedding / … nodes
  5. Compute VRAM breakdown using standard ML memory formulas
  6. Render the Hardware Recommendation Card

No GPU is required. A 4 GB laptop can safely profile a 70 B parameter LLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings
from traceml.utils.ast_analysis import (
    ParamEstimate,
    analyze_script,
    estimate_params,
)
from traceml.utils.ast_analysis.hardware_catalog import recommend_hardware

# ---------------------------------------------------------------------------
# Optimizer VRAM multipliers (multiples of param memory)
# ---------------------------------------------------------------------------
#
# KEY: Adam/AdamW always store optimizer states in FP32 (m + v tensors),
#      regardless of the model's dtype. So for a fp16 model:
#        optimizer_bytes = 8 bytes/param  (2 × fp32 tensors)
#        param_gb        = 2 bytes/param
#        → multiplier = 4.0
#      For a fp32 model:
#        optimizer_bytes = 8 bytes/param
#        param_gb        = 4 bytes/param
#        → multiplier = 2.0

# Base multiplier assumes fp32 model (AdamW = 2×).
# For fp16/bf16, _optimizer_state_multiplier() doubles the relevant entries.
_OPTIMIZER_STATE_MULTIPLIERS_FP32 = {
    "adam": 2.0,
    "adamw": 2.0,
    "fusedadam": 2.0,
    "deepspeedcpuadam": 2.0,
    "fused lamb": 2.0,
    "adafactor": 0.5,  # only second moment (approx)
    "adam8bit": 0.25,  # 8-bit quantised states
    "adamw8bit": 0.25,
    "lion": 1.0,  # one momentum term, fp32
    "rmsprop": 1.0,
    "adagrad": 1.0,
    "sgdm": 1.0,  # SGD with momentum — fp32 state
    "sgd": 0.0,  # vanilla SGD — no state
    "lbfgs": 0.0,
}

# Optimizers that keep fp32 states regardless of model dtype
# (true for Adam/AdamW variants; not true for 8-bit or Adafactor)
_FP32_STATE_OPTIMIZERS = {
    "adam",
    "adamw",
    "fusedadam",
    "deepspeedcpuadam",
    "fused lamb",
}


def _optimizer_state_multiplier(opt_type: str, bytes_per_param: int) -> float:
    """Return optimizer VRAM as a multiple of param VRAM.

    Adam/AdamW always stores states in fp32 (8 bytes/param).
    For fp16 params (2 bytes/param) that's a 4× multiplier.
    For fp32 params (4 bytes/param) that's a 2× multiplier.
    """
    base = _OPTIMIZER_STATE_MULTIPLIERS_FP32.get(opt_type.lower(), 2.0)
    # Scale fp32-state optimizers by the inverse of param dtype ratio
    if opt_type.lower() in _FP32_STATE_OPTIMIZERS and bytes_per_param < 4:
        # fp32 states vs fp16/bf16 params → multiply by (4 / bytes_per_param)
        return base * (4 / bytes_per_param)
    return base


def _nice_optimizer_name(opt_type: str) -> str:
    """Return a human-readable optimizer label."""
    table = {
        "adamw": "AdamW",
        "adam": "Adam",
        "adam8bit": "Adam-8bit",
        "adamw8bit": "AdamW-8bit",
        "fusedadam": "FusedAdam",
        "sgd": "SGD",
        "sgdm": "SGD+momentum",
        "rmsprop": "RMSprop",
        "adagrad": "AdaGrad",
        "lion": "Lion",
        "adafactor": "Adafactor",
        "lbfgs": "L-BFGS",
        "deepspeedcpuadam": "DeepSpeed CPUAdam",
    }
    return table.get(opt_type.lower(), opt_type.upper())


# ---------------------------------------------------------------------------
# Activation heuristics
# ---------------------------------------------------------------------------


def _activation_heuristic(
    findings,
    param_estimate: ParamEstimate,
    target_batch_size: int,
) -> float:
    """Estimated activation memory (GB) for the target batch size.

    Activations scale linearly with batch size but NOT with param count the
    same way gradients do. Rule of thumb from empirical profiling:

      Transformer / LLM:  ~0.3 × param_gb per sample
                          (roughly 2 × hidden_dim × seq_len × num_layers,
                          expressed relative to param size)
      CNN:                ~0.15 × param_gb per sample
      Unknown:            ~0.25 × param_gb per sample

    These are conservative estimates; gradient checkpointing reduces this
    further but we don't account for it here (pessimistic = safer).
    """

    is_cnn = any("Conv" in str(m) for m in findings.models)
    is_transformer = any(m.kind == "from_pretrained" for m in findings.models)

    param_gb = param_estimate.total_gb

    if is_cnn and not is_transformer:
        per_sample = 0.15
    elif is_transformer or param_estimate.source == "registry":
        per_sample = 0.30
    else:
        per_sample = 0.25

    return param_gb * per_sample * target_batch_size


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class StaticAnalysisResult:
    optimizer: str
    param_estimate: "ParamEstimate"
    optimizer_gb: float
    activation_gb: float
    cuda_overhead_gb: float
    total_gb: float
    target_batch_size: int


class SuggestDisplayDriver:
    """Pure-AST static VRAM estimator — zero GPU execution required."""

    def __init__(
        self,
        logger: Any,
        store: RemoteDBStore,
        settings: TraceMLSettings,
    ) -> None:
        self._logger = logger
        self._store = store
        self._settings = settings
        self._console = Console()
        self._target_batch_size: Optional[int] = (
            None  # None = auto-detect from script
        )
        _explicit_bs = os.environ.get("TRACEML_TARGET_BATCH_SIZE", "")
        if _explicit_bs.strip():
            try:
                self._target_batch_size = int(_explicit_bs)
            except ValueError:
                pass
        self._script_path: Optional[str] = os.environ.get(
            "TRACEML_SCRIPT_PATH", ""
        )
        self._result: Optional[StaticAnalysisResult] = None

    # ------------------------------------------------------------------
    # BaseDisplayDriver interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._console.print(
            "[bold cyan]TraceML Suggest-GPU[/bold cyan] (Zero-Execution Mode) — analysing script…"
        )
        self._result = self._analyse()

    def tick(self) -> None:
        pass

    def stop(self) -> None:
        if self._result is None:
            self._console.print(
                "[bold red]Static analysis produced no result.[/bold red]"
            )
            return
        self._render(self._result)

    # ------------------------------------------------------------------
    # Core analysis (AST only — no GPU, no subprocess)
    # ------------------------------------------------------------------

    def _analyse(self) -> Optional[StaticAnalysisResult]:
        script_path = self._script_path
        if not script_path:
            self._console.print(
                "[bold red]No script path provided (TRACEML_SCRIPT_PATH unset).[/bold red]"
            )
            return None

        # Step 1 — full AST scan
        findings = analyze_script(script_path)
        if findings.parse_errors:
            for err in findings.parse_errors:
                self._console.print(f"[bold red]Parse error: {err}[/bold red]")
            return None

        # Step 2 — optimizer
        opt_type = os.environ.get("TRACEML_SUGGEST_OPTIMIZER", "")
        if not opt_type or opt_type.lower() == "auto":
            if findings.optimizers:
                opt_type = findings.optimizers[0].optimizer_type
            else:
                opt_type = "Adam"

        # Step 3 — parameter estimate
        param_est = estimate_params(script_path, findings)

        # Step 4 — auto-detect batch size from DataLoader if not explicitly set
        target_bs = self._target_batch_size
        if target_bs is None:
            # Try to find a DataLoader batch_size in the script
            for dl in findings.dataloaders:
                if dl.batch_size and dl.batch_size > 0:
                    target_bs = dl.batch_size
                    break
            if target_bs is None:
                target_bs = 1  # conservative fallback

        # Step 5 — VRAM math
        param_gb = param_est.total_gb
        grad_gb = param_gb  # same dtype as params

        opt_mult = _optimizer_state_multiplier(
            opt_type, param_est.bytes_per_param
        )
        optimizer_gb = param_gb * opt_mult

        activation_gb = _activation_heuristic(findings, param_est, target_bs)

        # Step 6 — CUDA / PyTorch / cuDNN framework overhead
        # Scales with model size: larger models need bigger cuDNN workspaces,
        # scratch buffers, and NCCL comms. Floor of 0.5 GB covers the minimum
        # CUDA context + PyTorch runtime regardless of model size.
        cuda_overhead_gb = max(0.5, 0.1 * param_gb)

        total_gb = (
            param_gb
            + grad_gb
            + optimizer_gb
            + activation_gb
            + cuda_overhead_gb
        )

        return StaticAnalysisResult(
            optimizer=opt_type,
            param_estimate=param_est,
            optimizer_gb=optimizer_gb,
            activation_gb=activation_gb,
            cuda_overhead_gb=cuda_overhead_gb,
            total_gb=total_gb,
            target_batch_size=target_bs,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(gb: float) -> str:
        """Adaptive number format: MB for tiny models, GB with 4dp for medium, 2dp for large."""
        if gb < 0.001:
            return f"{gb * 1024:.1f} MB"
        if gb < 1.0:
            return f"{gb:.4f}"
        return f"{gb:.2f}"

    def _render(self, r: StaticAnalysisResult) -> None:
        param_est = r.param_estimate
        param_gb = param_est.total_gb
        grad_gb = param_gb
        opt_label = _nice_optimizer_name(r.optimizer)
        fmt = self._fmt

        table = Table(
            title=f"Hardware Recommendation (Target Batch Size: {r.target_batch_size})"
        )
        table.add_column("Component", style="cyan")
        table.add_column("Estimated VRAM (GB)", justify="right", style="green")

        # Model info row
        if param_est.model_description and param_est.source == "registry":
            table.add_row(
                f"Model — {param_est.model_description}",
                fmt(param_gb),
            )
        else:
            table.add_row("Model Parameters", fmt(param_gb))

        table.add_row("Gradients", fmt(grad_gb))
        table.add_row(f"Optimizer States ({opt_label})", fmt(r.optimizer_gb))

        act_label = f"Activations (×{r.target_batch_size}, heuristic)"
        table.add_row(act_label, fmt(r.activation_gb))
        table.add_row("PyTorch / CUDA Overhead", fmt(r.cuda_overhead_gb))
        table.add_row(
            "Total Estimated VRAM",
            fmt(r.total_gb),
            style="bold yellow",
        )

        recommendation = recommend_hardware(r.total_gb)

        self._console.print()
        self._console.print(
            Panel(
                table,
                title="[bold magenta]TraceML Suggest GPU[/bold magenta]",
            )
        )
        self._console.print(
            f"\n[bold green]Recommended Hardware:[/bold green] {recommendation}"
        )

        source_detail = {
            "registry": f"model retrieved from TraceML registry ({param_est.model_description})",
            "layer_count": "parameter count estimated from AST layer analysis",
            "unknown": "model not identified — parameter estimate may be inaccurate",
        }.get(param_est.source, param_est.source)

        precision = "fp16/bf16" if param_est.bytes_per_param == 2 else "fp32"

        self._console.print(
            f"[dim]Zero-Execution Mode: {source_detail}. "
            f"Optimizer: {opt_label}. Precision: {precision}. "
            f"Activation memory is a conservative heuristic.[/dim]\n"
        )
