"""
Summary-oriented system diagnosis rules.

These rules are intentionally conservative. System metrics are useful context,
but they are often less directly actionable than step-time diagnostics, so the
rules below favor precision and low noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..framework import DiagnosticIssue, DiagnosticRule
from .context import SystemSummarySignals


def _severity(value: float, crit_threshold: float) -> str:
    """
    Map a scalar signal to warn or crit severity.
    """
    return "crit" if float(value) >= float(crit_threshold) else "warn"


def _pct(value: Optional[float]) -> str:
    """
    Format one ratio-like value as a percentage string.
    """
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def _opt_pct(value: Optional[float]) -> Optional[float]:
    """
    Normalize an optional percentage-like value stored in `[0, 100]`.
    """
    if value is None:
        return None
    return max(0.0, float(value) / 100.0)


@dataclass(frozen=True)
class _BaseSystemRule(DiagnosticRule[SystemSummarySignals]):
    """
    Small shared helper for system rules.
    """

    name: str

    def _issue(
        self,
        *,
        kind: str,
        status: str,
        severity: str,
        summary: str,
        action: str,
        metric: Optional[str] = None,
        phase: Optional[str] = None,
        score: Optional[float] = None,
        ranks: Sequence[int] = (),
        evidence: Optional[dict] = None,
    ) -> DiagnosticIssue:
        return DiagnosticIssue(
            kind=kind,
            status=status,
            severity=severity,
            summary=summary,
            action=action,
            metric=metric,
            phase=phase,
            score=float(score) if score is not None else None,
            ranks=tuple(int(rank) for rank in ranks),
            evidence=dict(evidence or {}),
        )


@dataclass(frozen=True)
class LowGPUUtilizationRule(_BaseSystemRule):
    """
    Detect low overall GPU utilization over the analyzed summary window.
    """

    name: str = "low_gpu_utilization"
    warn_threshold_pct: float = 30.0
    crit_threshold_pct: float = 15.0

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        util = context.gpu_util_avg_percent
        if util is None or util >= self.warn_threshold_pct:
            return None

        lowest_gpu = context.lowest_util_gpu_idx
        severity = _severity(
            self.warn_threshold_pct - float(util),
            self.warn_threshold_pct - self.crit_threshold_pct,
        )
        return self._issue(
            kind="LOW_GPU_UTILIZATION",
            status="LOW GPU UTILIZATION",
            severity=severity,
            summary=f"Average GPU utilization is only {float(util):.1f}%.",
            action="Check whether host-side work or input throughput is limiting the run.",
            metric="gpu_util_avg_percent",
            phase="gpu",
            score=max(0.0, self.warn_threshold_pct - float(util)),
            ranks=(() if lowest_gpu is None else (lowest_gpu,)),
            evidence={
                "gpu_util_avg_percent": float(util),
                "lowest_util_gpu_idx": lowest_gpu,
            },
        )


@dataclass(frozen=True)
class HighCPUPressureRule(_BaseSystemRule):
    """
    Detect sustained high host CPU usage.
    """

    name: str = "high_cpu_pressure"
    warn_threshold_pct: float = 80.0
    crit_threshold_pct: float = 90.0

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        cpu = context.cpu_avg_percent
        if cpu is None or cpu < self.warn_threshold_pct:
            return None

        return self._issue(
            kind="HIGH_CPU_PRESSURE",
            status="HIGH CPU PRESSURE",
            severity=_severity(float(cpu), self.crit_threshold_pct),
            summary=f"Average host CPU usage is {float(cpu):.1f}%.",
            action="Inspect data loading, CPU-side preprocessing, or host contention.",
            metric="cpu_avg_percent",
            phase="cpu",
            score=float(cpu),
            evidence={"cpu_avg_percent": float(cpu)},
        )


@dataclass(frozen=True)
class HighRAMPressureRule(_BaseSystemRule):
    """
    Detect high host RAM pressure relative to total system memory.
    """

    name: str = "high_ram_pressure"
    warn_fraction: float = 0.85
    crit_fraction: float = 0.92

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        pressure = context.ram_pressure_frac
        if pressure is None or pressure < self.warn_fraction:
            return None

        return self._issue(
            kind="HIGH_RAM_PRESSURE",
            status="HIGH RAM PRESSURE",
            severity=_severity(float(pressure), self.crit_fraction),
            summary=f"Peak host RAM reached {_pct(pressure)} of system capacity.",
            action="Reduce host memory pressure or check for dataset/process growth.",
            metric="ram_peak_bytes",
            phase="ram",
            score=float(pressure),
            evidence={"ram_pressure_frac": float(pressure)},
        )


@dataclass(frozen=True)
class GPUUtilImbalanceRule(_BaseSystemRule):
    """
    Detect materially uneven average GPU utilization across devices.
    """

    name: str = "gpu_util_imbalance"
    warn_fraction: float = 0.20
    crit_fraction: float = 0.35

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        imbalance = context.gpu_util_imbalance_pct
        if imbalance is None or imbalance < self.warn_fraction:
            return None

        ranks: Tuple[int, ...] = tuple(
            gpu_idx
            for gpu_idx in (
                context.lowest_util_gpu_idx,
                context.highest_util_gpu_idx,
            )
            if gpu_idx is not None
        )
        return self._issue(
            kind="GPU_UTIL_IMBALANCE",
            status="GPU UTIL IMBALANCE",
            severity=_severity(float(imbalance), self.crit_fraction),
            summary=f"Average GPU utilization differs by {_pct(imbalance)} across devices.",
            action="Inspect device-level workload balance and rank placement.",
            metric="gpu_util_avg_percent",
            phase="gpu",
            score=float(imbalance),
            ranks=ranks,
            evidence={
                "gpu_util_imbalance_pct": float(imbalance),
                "lowest_util_gpu_idx": context.lowest_util_gpu_idx,
                "highest_util_gpu_idx": context.highest_util_gpu_idx,
            },
        )


DEFAULT_SYSTEM_RULES = (
    LowGPUUtilizationRule(),
    HighCPUPressureRule(),
    HighRAMPressureRule(),
    GPUUtilImbalanceRule(),
)


def run_system_rules(
    context: SystemSummarySignals,
    *,
    rules: Sequence[
        DiagnosticRule[SystemSummarySignals]
    ] = DEFAULT_SYSTEM_RULES,
) -> Tuple[DiagnosticIssue, ...]:
    """
    Run all registered system rules over one summary analysis context.
    """
    out = []
    for rule in rules:
        issue = rule.evaluate(context)
        if issue is not None:
            out.append(issue)
    return tuple(out)


__all__ = [
    "DEFAULT_SYSTEM_RULES",
    "run_system_rules",
]
