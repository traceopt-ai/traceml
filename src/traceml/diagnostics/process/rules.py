"""
Summary-oriented process diagnosis rules.

These rules are intentionally conservative. Process metrics are valuable
supporting context, but they are often less directly actionable than step-time
diagnostics, so the rules below favor precision and low noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..common import DiagnosticIssue, DiagnosticRule
from .context import ProcessSummarySignals


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


@dataclass(frozen=True)
class _BaseProcessRule(DiagnosticRule[ProcessSummarySignals]):
    """
    Small shared helper for process rules.
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
class HighCPUProcessPressureRule(_BaseProcessRule):
    """
    Detect sustained high traced-process CPU pressure.
    """

    name: str = "high_cpu_process_pressure"
    warn_fraction: float = 0.50
    crit_fraction: float = 0.80
    fallback_warn_percent: float = 200.0
    fallback_crit_percent: float = 400.0

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        if context.cpu_pressure_frac is not None:
            pressure = context.cpu_pressure_frac
            if pressure < self.warn_fraction:
                return None
            return self._issue(
                kind="HIGH_CPU_PROCESS_PRESSURE",
                status="HIGH CPU PROCESS PRESSURE",
                severity=_severity(float(pressure), self.crit_fraction),
                summary=(
                    "The traced workload used "
                    f"{_pct(pressure)} of logical CPU capacity on average."
                ),
                action="Inspect data loading, preprocessing, or host-side contention.",
                metric="cpu_avg_percent",
                phase="cpu",
                score=float(pressure),
                evidence={
                    "cpu_avg_percent": context.cpu_avg_percent,
                    "cpu_logical_core_count": context.cpu_logical_core_count,
                    "cpu_pressure_frac": float(pressure),
                },
            )

        cpu_avg = context.cpu_avg_percent
        if cpu_avg is None or cpu_avg < self.fallback_warn_percent:
            return None

        return self._issue(
            kind="HIGH_CPU_PROCESS_PRESSURE",
            status="HIGH CPU PROCESS PRESSURE",
            severity=_severity(float(cpu_avg), self.fallback_crit_percent),
            summary=f"Average traced-process CPU usage is {float(cpu_avg):.1f}%.",
            action="Inspect data loading, preprocessing, or host-side contention.",
            metric="cpu_avg_percent",
            phase="cpu",
            score=float(cpu_avg),
            evidence={"cpu_avg_percent": float(cpu_avg)},
        )


@dataclass(frozen=True)
class HighRSSPressureRule(_BaseProcessRule):
    """
    Detect high process RSS pressure relative to available host memory.
    """

    name: str = "high_rss_pressure"
    warn_fraction: float = 0.85
    crit_fraction: float = 0.92

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        pressure = context.ram_pressure_frac
        if pressure is None or pressure < self.warn_fraction:
            return None

        ranks: Tuple[int, ...] = ()
        if context.highest_rss_rank is not None:
            ranks = (context.highest_rss_rank,)

        return self._issue(
            kind="HIGH_RSS_PRESSURE",
            status="HIGH RSS PRESSURE",
            severity=_severity(float(pressure), self.crit_fraction),
            summary=(
                f"Peak traced-process RSS reached {_pct(pressure)} of host memory."
            ),
            action="Reduce host memory pressure or inspect memory growth in the traced workload.",
            metric="ram_peak_bytes",
            phase="ram",
            score=float(pressure),
            ranks=ranks,
            evidence={"ram_pressure_frac": float(pressure)},
        )


@dataclass(frozen=True)
class HighGPUMemoryPressureRule(_BaseProcessRule):
    """
    Detect high GPU memory pressure for the traced workload.
    """

    name: str = "high_gpu_memory_pressure"
    warn_fraction: float = 0.85
    crit_fraction: float = 0.92

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        pressure = context.gpu_mem_reserved_peak_frac
        metric = "gpu_mem_reserved_peak_bytes"
        rank = context.highest_reserved_rank

        if pressure is None:
            pressure = context.gpu_mem_used_peak_frac
            metric = "gpu_mem_used_peak_bytes"
            rank = context.highest_used_rank

        if pressure is None or pressure < self.warn_fraction:
            return None

        ranks: Tuple[int, ...] = ()
        if rank is not None:
            ranks = (rank,)

        return self._issue(
            kind="HIGH_GPU_MEMORY_PRESSURE",
            status="HIGH GPU MEMORY PRESSURE",
            severity=_severity(float(pressure), self.crit_fraction),
            summary=f"Peak traced GPU memory reached {_pct(pressure)} of device capacity.",
            action="Reduce peak memory demand or inspect the most memory-heavy rank.",
            metric=metric,
            phase="gpu_memory",
            score=float(pressure),
            ranks=ranks,
            evidence={
                "gpu_mem_used_peak_frac": context.gpu_mem_used_peak_frac,
                "gpu_mem_reserved_peak_frac": context.gpu_mem_reserved_peak_frac,
                "highest_used_rank": context.highest_used_rank,
                "highest_reserved_rank": context.highest_reserved_rank,
            },
        )


@dataclass(frozen=True)
class GPUMemoryReservedOverhangRule(_BaseProcessRule):
    """
    Detect materially higher reserved GPU memory than active use.
    """

    name: str = "gpu_memory_reserved_overhang"
    warn_ratio: float = 1.25
    crit_ratio: float = 1.50

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        ratio = context.gpu_mem_reserved_overhang_ratio
        if ratio is None or ratio < self.warn_ratio:
            return None

        ranks: Tuple[int, ...] = ()
        if context.highest_reserved_rank is not None:
            ranks = (context.highest_reserved_rank,)

        return self._issue(
            kind="GPU_MEMORY_RESERVED_OVERHANG",
            status="GPU MEMORY RESERVED OVERHANG",
            severity=_severity(float(ratio), self.crit_ratio),
            summary=(
                "Reserved GPU memory materially exceeds active use "
                f"(~{float(ratio):.2f}x at peak)."
            ),
            action="Inspect allocator behavior, fragmentation, or memory retention.",
            metric="gpu_mem_reserved_peak_bytes",
            phase="gpu_memory",
            score=float(ratio),
            ranks=ranks,
            evidence={
                "gpu_mem_reserved_overhang_ratio": float(ratio),
                "highest_reserved_rank": context.highest_reserved_rank,
            },
        )


@dataclass(frozen=True)
class RankGPUMemoryImbalanceRule(_BaseProcessRule):
    """
    Detect materially uneven GPU memory usage across traced ranks.
    """

    name: str = "rank_gpu_memory_imbalance"
    warn_fraction: float = 0.20
    crit_fraction: float = 0.35

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        imbalance = context.rank_gpu_reserved_imbalance_pct
        metric = "gpu_mem_reserved_peak_bytes"
        ranks: Tuple[int, ...] = tuple(
            rank
            for rank in (
                context.highest_reserved_rank,
                context.least_headroom_rank,
            )
            if rank is not None
        )

        if imbalance is None:
            imbalance = context.rank_gpu_used_imbalance_pct
            metric = "gpu_mem_used_peak_bytes"
            ranks = tuple(
                rank
                for rank in (
                    context.highest_used_rank,
                    context.least_headroom_rank,
                )
                if rank is not None
            )

        if imbalance is None or imbalance < self.warn_fraction:
            return None

        return self._issue(
            kind="RANK_GPU_MEMORY_IMBALANCE",
            status="RANK GPU MEMORY IMBALANCE",
            severity=_severity(float(imbalance), self.crit_fraction),
            summary=f"Peak traced GPU memory differs by {_pct(imbalance)} across ranks.",
            action="Inspect workload balance and per-rank memory behavior.",
            metric=metric,
            phase="gpu_memory",
            score=float(imbalance),
            ranks=ranks,
            evidence={
                "rank_gpu_used_imbalance_pct": context.rank_gpu_used_imbalance_pct,
                "rank_gpu_reserved_imbalance_pct": (
                    context.rank_gpu_reserved_imbalance_pct
                ),
                "highest_used_rank": context.highest_used_rank,
                "highest_reserved_rank": context.highest_reserved_rank,
                "least_headroom_rank": context.least_headroom_rank,
            },
        )


DEFAULT_PROCESS_RULES = (
    HighGPUMemoryPressureRule(),
    GPUMemoryReservedOverhangRule(),
    RankGPUMemoryImbalanceRule(),
    HighRSSPressureRule(),
    HighCPUProcessPressureRule(),
)


def run_process_rules(
    context: ProcessSummarySignals,
    *,
    rules: Sequence[
        DiagnosticRule[ProcessSummarySignals]
    ] = DEFAULT_PROCESS_RULES,
) -> Tuple[DiagnosticIssue, ...]:
    """
    Run all registered process rules over one summary analysis context.
    """
    out = []
    for rule in rules:
        issue = rule.evaluate(context)
        if issue is not None:
            out.append(issue)
    return tuple(out)


__all__ = [
    "DEFAULT_PROCESS_RULES",
    "GPUMemoryReservedOverhangRule",
    "HighCPUProcessPressureRule",
    "HighGPUMemoryPressureRule",
    "HighRSSPressureRule",
    "RankGPUMemoryImbalanceRule",
    "run_process_rules",
]
