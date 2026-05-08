"""System diagnosis rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..common import DiagnosticIssue, DiagnosticRule
from .context import SystemSummarySignals
from .policy import DEFAULT_SYSTEM_POLICY, SystemDiagnosisPolicy


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.1f}%"


def _fmt_gpu_suffix(gpu_idx: Optional[int]) -> str:
    return "" if gpu_idx is None else f" on gpu{int(gpu_idx)}"


@dataclass(frozen=True)
class _BaseSystemRule(DiagnosticRule[SystemSummarySignals]):
    name: str
    policy: SystemDiagnosisPolicy = DEFAULT_SYSTEM_POLICY

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
class VeryHighGPUMemoryRule(_BaseSystemRule):
    name: str = "very_high_gpu_memory"

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        pct = context.gpu_mem_peak_percent
        if self.policy.gpu_memory_peak_percent.classify(pct) != "very_high":
            return None

        gpu_idx = context.highest_mem_pressure_gpu_idx
        return self._issue(
            kind="VERY_HIGH_GPU_MEMORY",
            status="VERY HIGH GPU MEMORY",
            severity="crit",
            summary=(
                "GPU memory was very high, peaking at "
                f"{_fmt_pct(pct)}{_fmt_gpu_suffix(gpu_idx)}."
            ),
            action="Reduce GPU memory pressure before scaling this run.",
            metric="gpu_mem_peak_percent",
            phase="gpu_memory",
            score=pct,
            ranks=(() if gpu_idx is None else (gpu_idx,)),
            evidence={
                "gpu_mem_peak_percent": pct,
                "gpu_idx": gpu_idx,
            },
        )


@dataclass(frozen=True)
class HighGPUMemoryRule(_BaseSystemRule):
    name: str = "high_gpu_memory"

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        pct = context.gpu_mem_peak_percent
        if self.policy.gpu_memory_peak_percent.classify(pct) != "high":
            return None

        gpu_idx = context.highest_mem_pressure_gpu_idx
        return self._issue(
            kind="HIGH_GPU_MEMORY",
            status="HIGH GPU MEMORY",
            severity="warn",
            summary=(
                "GPU memory was high, peaking at "
                f"{_fmt_pct(pct)}{_fmt_gpu_suffix(gpu_idx)}."
            ),
            action="Watch GPU memory headroom for larger batches or models.",
            metric="gpu_mem_peak_percent",
            phase="gpu_memory",
            score=pct,
            ranks=(() if gpu_idx is None else (gpu_idx,)),
            evidence={
                "gpu_mem_peak_percent": pct,
                "gpu_idx": gpu_idx,
            },
        )


@dataclass(frozen=True)
class HighGPUTemperatureRule(_BaseSystemRule):
    name: str = "high_gpu_temperature"

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        temp = context.gpu_temp_peak_c
        if self.policy.gpu_temp_peak_c.classify(temp) != "high":
            return None

        gpu_idx = context.highest_temp_gpu_idx
        return self._issue(
            kind="HIGH_GPU_TEMPERATURE",
            status="HIGH GPU TEMPERATURE",
            severity="crit",
            summary=(
                "GPU temperature was high, peaking at "
                f"{float(temp):.1f} C{_fmt_gpu_suffix(gpu_idx)}."
            ),
            action="Check cooling and thermal throttling risk.",
            metric="gpu_temp_peak_c",
            phase="gpu_temperature",
            score=float(temp),
            ranks=(() if gpu_idx is None else (gpu_idx,)),
            evidence={
                "gpu_temp_peak_c": float(temp),
                "gpu_idx": gpu_idx,
            },
        )


@dataclass(frozen=True)
class HighGPUPowerRule(_BaseSystemRule):
    name: str = "high_gpu_power"

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        pct = context.gpu_power_avg_limit_percent
        if self.policy.gpu_power_avg_limit_percent.classify(pct) != "high":
            return None

        gpu_idx = context.highest_power_gpu_idx
        return self._issue(
            kind="HIGH_GPU_POWER",
            status="HIGH GPU POWER",
            severity="warn",
            summary=(
                "GPU power was high, averaging "
                f"{_fmt_pct(pct)} of limit{_fmt_gpu_suffix(gpu_idx)}."
            ),
            action="Review power headroom if this run is unstable.",
            metric="gpu_power_avg_limit_percent",
            phase="gpu_power",
            score=pct,
            ranks=(() if gpu_idx is None else (gpu_idx,)),
            evidence={
                "gpu_power_avg_limit_percent": pct,
                "gpu_idx": gpu_idx,
            },
        )


@dataclass(frozen=True)
class HighHostMemoryRule(_BaseSystemRule):
    name: str = "high_host_memory"

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        pct = context.ram_peak_percent
        if self.policy.ram_peak_percent.classify(pct) != "high":
            return None

        return self._issue(
            kind="HIGH_HOST_MEMORY",
            status="HIGH HOST MEMORY",
            severity="warn",
            summary=(
                "Host RAM usage was high, peaking at "
                f"{_fmt_pct(pct)} of total."
            ),
            action="Reduce host memory pressure or inspect data workers.",
            metric="ram_peak_percent",
            phase="ram",
            score=pct,
            evidence={"ram_peak_percent": pct},
        )


@dataclass(frozen=True)
class HighCPURule(_BaseSystemRule):
    name: str = "high_cpu"

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        pct = context.cpu_avg_percent
        if self.policy.cpu_avg_percent.classify(pct) != "high":
            return None

        return self._issue(
            kind="HIGH_CPU",
            status="HIGH CPU",
            severity="warn",
            summary=f"CPU usage was high, averaging {_fmt_pct(pct)}.",
            action="Inspect CPU-side preprocessing or host contention.",
            metric="cpu_avg_percent",
            phase="cpu",
            score=pct,
            evidence={"cpu_avg_percent": pct},
        )


@dataclass(frozen=True)
class LowGPUUtilizationRule(_BaseSystemRule):
    name: str = "low_gpu_utilization"

    def evaluate(
        self, context: SystemSummarySignals
    ) -> Optional[DiagnosticIssue]:
        pct = context.gpu_util_avg_percent
        if self.policy.gpu_util_avg_percent.classify(pct) != "low":
            return None

        gpu_idx = context.lowest_util_gpu_idx
        return self._issue(
            kind="LOW_GPU_UTILIZATION",
            status="LOW GPU UTILIZATION",
            severity="info",
            summary=f"GPU utilization was low, averaging {_fmt_pct(pct)}.",
            action="Use step-time diagnostics to check host or input stalls.",
            metric="gpu_util_avg_percent",
            phase="gpu_utilization",
            score=100.0 - float(pct),
            ranks=(() if gpu_idx is None else (gpu_idx,)),
            evidence={
                "gpu_util_avg_percent": pct,
                "lowest_util_gpu_idx": gpu_idx,
            },
        )


DEFAULT_SYSTEM_RULES = (
    VeryHighGPUMemoryRule(),
    HighGPUTemperatureRule(),
    HighGPUMemoryRule(),
    HighGPUPowerRule(),
    HighHostMemoryRule(),
    HighCPURule(),
    LowGPUUtilizationRule(),
)


_ISSUE_PRIORITY = {
    "VERY_HIGH_GPU_MEMORY": 0,
    "HIGH_GPU_TEMPERATURE": 1,
    "HIGH_GPU_MEMORY": 2,
    "HIGH_GPU_POWER": 3,
    "HIGH_HOST_MEMORY": 4,
    "HIGH_CPU": 5,
    "LOW_GPU_UTILIZATION": 6,
}


def sort_system_issues(
    issues: Sequence[DiagnosticIssue],
) -> Tuple[DiagnosticIssue, ...]:
    return tuple(
        sorted(
            issues,
            key=lambda issue: (
                _ISSUE_PRIORITY.get(issue.kind, 999),
                -(float(issue.score or 0.0)),
            ),
        )
    )


def run_system_rules(
    context: SystemSummarySignals,
    *,
    rules: Sequence[
        DiagnosticRule[SystemSummarySignals]
    ] = DEFAULT_SYSTEM_RULES,
) -> Tuple[DiagnosticIssue, ...]:
    out = []
    for rule in rules:
        issue = rule.evaluate(context)
        if issue is not None:
            out.append(issue)
    return sort_system_issues(out)


__all__ = [
    "DEFAULT_SYSTEM_RULES",
    "HighCPURule",
    "HighGPUMemoryRule",
    "HighGPUPowerRule",
    "HighGPUTemperatureRule",
    "HighHostMemoryRule",
    "LowGPUUtilizationRule",
    "VeryHighGPUMemoryRule",
    "run_system_rules",
    "sort_system_issues",
]
