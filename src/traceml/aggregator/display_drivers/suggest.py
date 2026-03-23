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

from traceml.aggregator.hardware_catalog import recommend_hardware
from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings
from traceml.utils.ast_analysis import (
    ParamEstimate,
    analyze_script,
    estimate_params,
)

# ---------------------------------------------------------------------------
# Optimizer VRAM multipliers (multiples of param memory)
# ---------------------------------------------------------------------------

_OPTIMIZER_STATE_MULTIPLIERS = {
    "adam": 2.0,
    "adamw": 2.0,
    "fusedadam": 2.0,
    "deepspeedcpuadam": 2.0,
    "fused lamb": 2.0,
    "adafactor": 0.5,  # only second moment (approx)
    "adam8bit": 0.25,  # 8-bit quantised states
    "adamw8bit": 0.25,
    "lion": 1.0,  # one momentum term
    "rmsprop": 1.0,
    "adagrad": 1.0,
    "sgdm": 1.0,  # SGD with momentum
    "sgd": 0.0,  # vanilla SGD — no state
    "lbfgs": 0.0,
}


def _optimizer_state_multiplier(opt_type: str) -> float:
    return _OPTIMIZER_STATE_MULTIPLIERS.get(opt_type.lower(), 2.0)


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
    """Conservative heuristic for activation memory (GB).

    Transformer / LLM:  activations ≈ param_bytes × 2 per sample
    CNN:                activations ≈ param_bytes × 0.5 per sample
    Unknown:            activations ≈ param_bytes × 1.5 per sample
    """

    is_cnn = any("Conv" in str(m) for m in findings.models)
    is_transformer = any(m.kind == "from_pretrained" for m in findings.models)

    param_gb = param_estimate.total_gb

    if is_cnn and not is_transformer:
        multiplier = 0.5
    elif is_transformer or param_estimate.source == "registry":
        multiplier = 2.0
    else:
        multiplier = 1.5

    return param_gb * multiplier * target_batch_size


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class StaticAnalysisResult:
    optimizer: str
    param_estimate: "ParamEstimate"
    optimizer_gb: float
    activation_gb: float
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
        self._target_batch_size = int(
            os.environ.get("TRACEML_TARGET_BATCH_SIZE", "1")
        )
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

        # Step 4 — VRAM math
        param_gb = param_est.total_gb
        grad_gb = param_gb  # same dtype as params

        opt_mult = _optimizer_state_multiplier(opt_type)
        optimizer_gb = param_gb * opt_mult

        activation_gb = _activation_heuristic(
            findings, param_est, self._target_batch_size
        )

        total_gb = param_gb + grad_gb + optimizer_gb + activation_gb

        return StaticAnalysisResult(
            optimizer=opt_type,
            param_estimate=param_est,
            optimizer_gb=optimizer_gb,
            activation_gb=activation_gb,
            total_gb=total_gb,
            target_batch_size=self._target_batch_size,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, r: StaticAnalysisResult) -> None:
        param_est = r.param_estimate
        param_gb = param_est.total_gb
        grad_gb = param_gb
        opt_label = _nice_optimizer_name(r.optimizer)

        table = Table(
            title=f"Hardware Recommendation (Target Batch Size: {r.target_batch_size})"
        )
        table.add_column("Component", style="cyan")
        table.add_column("Estimated VRAM (GB)", justify="right", style="green")

        # Model info row
        if param_est.model_description and param_est.source == "registry":
            table.add_row(
                f"Model — {param_est.model_description}",
                f"{param_gb:.2f}",
            )
        else:
            table.add_row("Model Parameters", f"{param_gb:.2f}")

        table.add_row("Gradients", f"{grad_gb:.2f}")
        table.add_row(
            f"Optimizer States ({opt_label})", f"{r.optimizer_gb:.2f}"
        )

        act_label = f"Activations (×{r.target_batch_size}, {'heuristic'})"
        table.add_row(act_label, f"{r.activation_gb:.2f}")
        table.add_row(
            "Total Estimated VRAM",
            f"{r.total_gb:.2f}",
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
