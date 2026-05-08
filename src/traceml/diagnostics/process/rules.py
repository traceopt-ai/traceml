"""Process diagnosis rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..common import DiagnosticIssue, DiagnosticRule
from .context import ProcessSummarySignals
from .policy import DEFAULT_PROCESS_POLICY, ProcessDiagnosisPolicy


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.1f}%"


def _fmt_rank(rank: Optional[int]) -> str:
    return "" if rank is None else f" on rank {int(rank)}"


@dataclass(frozen=True)
class _BaseProcessRule(DiagnosticRule[ProcessSummarySignals]):
    name: str
    policy: ProcessDiagnosisPolicy = DEFAULT_PROCESS_POLICY

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


def _gpu_memory_peak(
    context: ProcessSummarySignals,
) -> tuple[Optional[float], Optional[str], Optional[int]]:
    if context.gpu_mem_reserved_peak_percent is not None:
        return (
            context.gpu_mem_reserved_peak_percent,
            "gpu_mem_reserved_peak_percent",
            context.highest_reserved_rank,
        )
    return (
        context.gpu_mem_used_peak_percent,
        "gpu_mem_used_peak_percent",
        context.highest_used_rank,
    )


@dataclass(frozen=True)
class VeryHighProcessGPUMemoryRule(_BaseProcessRule):
    name: str = "very_high_process_gpu_memory"

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        pct, metric, rank = _gpu_memory_peak(context)
        if self.policy.gpu_memory_peak_percent.classify(pct) != "very_high":
            return None

        return self._issue(
            kind="VERY_HIGH_PROCESS_GPU_MEMORY",
            status="VERY HIGH PROCESS GPU MEMORY",
            severity="crit",
            summary=(
                "Process GPU memory was very high, peaking at "
                f"{_fmt_pct(pct)}{_fmt_rank(rank)}."
            ),
            action="Reduce traced process GPU memory pressure.",
            metric=metric,
            phase="gpu_memory",
            score=pct,
            ranks=(() if rank is None else (rank,)),
            evidence={
                "gpu_mem_used_peak_percent": context.gpu_mem_used_peak_percent,
                "gpu_mem_reserved_peak_percent": (
                    context.gpu_mem_reserved_peak_percent
                ),
                "rank": rank,
            },
        )


@dataclass(frozen=True)
class HighProcessGPUMemoryRule(_BaseProcessRule):
    name: str = "high_process_gpu_memory"

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        pct, metric, rank = _gpu_memory_peak(context)
        if self.policy.gpu_memory_peak_percent.classify(pct) != "high":
            return None

        return self._issue(
            kind="HIGH_PROCESS_GPU_MEMORY",
            status="HIGH PROCESS GPU MEMORY",
            severity="warn",
            summary=(
                "Process GPU memory was high, peaking at "
                f"{_fmt_pct(pct)}{_fmt_rank(rank)}."
            ),
            action="Watch traced process GPU memory headroom.",
            metric=metric,
            phase="gpu_memory",
            score=pct,
            ranks=(() if rank is None else (rank,)),
            evidence={
                "gpu_mem_used_peak_percent": context.gpu_mem_used_peak_percent,
                "gpu_mem_reserved_peak_percent": (
                    context.gpu_mem_reserved_peak_percent
                ),
                "rank": rank,
            },
        )


@dataclass(frozen=True)
class GPUMemoryReservedOverhangRule(_BaseProcessRule):
    name: str = "gpu_memory_reserved_overhang"

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        ratio = context.gpu_mem_reserved_overhang_ratio
        if self.policy.gpu_reserved_overhang_ratio.classify(ratio) != "high":
            return None

        rank = context.highest_overhang_rank
        return self._issue(
            kind="GPU_MEMORY_RESERVED_OVERHANG",
            status="GPU MEMORY RESERVED OVERHANG",
            severity="warn",
            summary=f"Reserved GPU memory was {float(ratio):.2f}x active use.",
            action="Inspect allocator behavior or retained tensors.",
            metric="gpu_mem_reserved_peak_bytes",
            phase="gpu_memory",
            score=ratio,
            ranks=(() if rank is None else (rank,)),
            evidence={
                "gpu_mem_reserved_overhang_ratio": float(ratio),
                "highest_overhang_rank": rank,
            },
        )


@dataclass(frozen=True)
class RankGPUMemoryImbalanceRule(_BaseProcessRule):
    name: str = "rank_gpu_memory_imbalance"

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        pct = context.rank_gpu_reserved_imbalance_percent
        metric = "rank_gpu_reserved_imbalance_percent"
        ranks: Tuple[int, ...] = tuple(
            rank
            for rank in (
                context.highest_reserved_rank,
                context.least_headroom_rank,
            )
            if rank is not None
        )

        if pct is None:
            pct = context.rank_gpu_used_imbalance_percent
            metric = "rank_gpu_used_imbalance_percent"
            ranks = tuple(
                rank
                for rank in (
                    context.highest_used_rank,
                    context.least_headroom_rank,
                )
                if rank is not None
            )

        if (
            self.policy.rank_gpu_memory_imbalance_percent.classify(pct)
            != "high"
        ):
            return None

        return self._issue(
            kind="RANK_GPU_MEMORY_IMBALANCE",
            status="RANK GPU MEMORY IMBALANCE",
            severity="warn",
            summary=f"Process GPU memory differed by {_fmt_pct(pct)} across ranks.",
            action="Inspect per-rank workload and memory behavior.",
            metric=metric,
            phase="gpu_memory",
            score=pct,
            ranks=ranks,
            evidence={
                "rank_gpu_used_imbalance_percent": (
                    context.rank_gpu_used_imbalance_percent
                ),
                "rank_gpu_reserved_imbalance_percent": (
                    context.rank_gpu_reserved_imbalance_percent
                ),
                "highest_used_rank": context.highest_used_rank,
                "highest_reserved_rank": context.highest_reserved_rank,
                "least_headroom_rank": context.least_headroom_rank,
            },
        )


@dataclass(frozen=True)
class HighProcessRSSRule(_BaseProcessRule):
    name: str = "high_process_rss"

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        pct = context.ram_peak_percent
        if self.policy.rss_peak_percent.classify(pct) != "high":
            return None

        rank = context.highest_rss_rank
        return self._issue(
            kind="HIGH_PROCESS_RSS",
            status="HIGH PROCESS RSS",
            severity="warn",
            summary=f"Process RSS was high, peaking at {_fmt_pct(pct)}.",
            action="Reduce traced process host memory pressure.",
            metric="ram_peak_percent",
            phase="ram",
            score=pct,
            ranks=(() if rank is None else (rank,)),
            evidence={"ram_peak_percent": pct, "highest_rss_rank": rank},
        )


@dataclass(frozen=True)
class HighProcessCPURule(_BaseProcessRule):
    name: str = "high_process_cpu"

    def evaluate(
        self,
        context: ProcessSummarySignals,
    ) -> Optional[DiagnosticIssue]:
        pct = context.cpu_capacity_percent
        if self.policy.cpu_capacity_percent.classify(pct) != "high":
            return None

        return self._issue(
            kind="HIGH_PROCESS_CPU",
            status="HIGH PROCESS CPU",
            severity="warn",
            summary=f"Process CPU averaged {_fmt_pct(pct)} of capacity.",
            action="Inspect data loading, preprocessing, or host contention.",
            metric="cpu_capacity_percent",
            phase="cpu",
            score=pct,
            evidence={
                "cpu_avg_percent": context.cpu_avg_percent,
                "cpu_logical_core_count": context.cpu_logical_core_count,
                "cpu_capacity_percent": pct,
            },
        )


DEFAULT_PROCESS_RULES = (
    VeryHighProcessGPUMemoryRule(),
    HighProcessGPUMemoryRule(),
    GPUMemoryReservedOverhangRule(),
    RankGPUMemoryImbalanceRule(),
    HighProcessRSSRule(),
    HighProcessCPURule(),
)


_ISSUE_PRIORITY = {
    "VERY_HIGH_PROCESS_GPU_MEMORY": 0,
    "HIGH_PROCESS_GPU_MEMORY": 1,
    "GPU_MEMORY_RESERVED_OVERHANG": 2,
    "RANK_GPU_MEMORY_IMBALANCE": 3,
    "HIGH_PROCESS_RSS": 4,
    "HIGH_PROCESS_CPU": 5,
}


def sort_process_issues(
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


def run_process_rules(
    context: ProcessSummarySignals,
    *,
    rules: Sequence[
        DiagnosticRule[ProcessSummarySignals]
    ] = DEFAULT_PROCESS_RULES,
) -> Tuple[DiagnosticIssue, ...]:
    out = []
    for rule in rules:
        issue = rule.evaluate(context)
        if issue is not None:
            out.append(issue)
    return sort_process_issues(out)


HighCPUProcessPressureRule = HighProcessCPURule
HighGPUMemoryPressureRule = HighProcessGPUMemoryRule
HighRSSPressureRule = HighProcessRSSRule


__all__ = [
    "DEFAULT_PROCESS_RULES",
    "GPUMemoryReservedOverhangRule",
    "HighCPUProcessPressureRule",
    "HighGPUMemoryPressureRule",
    "HighProcessCPURule",
    "HighProcessGPUMemoryRule",
    "HighProcessRSSRule",
    "HighRSSPressureRule",
    "RankGPUMemoryImbalanceRule",
    "VeryHighProcessGPUMemoryRule",
    "run_process_rules",
    "sort_process_issues",
]
