# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Summary-oriented process diagnosis API.

Process diagnosis is intentionally conservative and primarily aimed at final
summary interpretation and JSON output. It is not used to change live runtime
rendering behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common import BaseDiagnosis, DiagnosticResult, Severity
from .context import (
    ProcessDiagnosisInput,
    ProcessSummarySignals,
    build_process_summary_signals,
)
from .rules import run_process_rules


@dataclass(frozen=True)
class ProcessDiagnosis(BaseDiagnosis):
    """
    Primary process diagnosis used in final-summary cards and JSON.
    """

    kind: str
    samples_used: int


def _mk_diag(
    *,
    kind: str,
    severity: Severity,
    status: str,
    reason: str,
    action: str,
    samples_used: int,
) -> ProcessDiagnosis:
    return ProcessDiagnosis(
        kind=str(kind),
        severity=severity,
        status=str(status),
        reason=str(reason),
        action=str(action),
        samples_used=int(samples_used),
    )


def _default_primary(signals: ProcessSummarySignals) -> ProcessDiagnosis:
    """Build the default non-issue process diagnosis."""
    if signals.samples <= 0:
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            status="NO DATA",
            reason="No traced process telemetry was recorded.",
            action="Collect process telemetry for workload-local context.",
            samples_used=signals.samples,
        )

    has_gpu_memory = any(
        value is not None
        for value in (
            signals.gpu_mem_used_peak_percent,
            signals.gpu_mem_reserved_peak_percent,
        )
    )
    reason = (
        "Process CPU, RSS, and GPU memory showed no pressure."
        if has_gpu_memory
        else "Process CPU and RSS showed no pressure."
    )
    return _mk_diag(
        kind="NORMAL",
        severity="info",
        status="NORMAL",
        reason=reason,
        action="Use training diagnostics for model-level bottlenecks.",
        samples_used=signals.samples,
    )


def diagnose_process(
    data: ProcessDiagnosisInput,
) -> DiagnosticResult[ProcessDiagnosis]:
    """
    Diagnose process pressure from a prepared diagnosis input.
    """
    signals = build_process_summary_signals(data)

    issues = run_process_rules(signals) if signals.samples > 0 else ()
    if issues:
        primary_issue = issues[0]
        primary = _mk_diag(
            kind=primary_issue.kind,
            severity=primary_issue.severity,
            status=primary_issue.status,
            reason=primary_issue.summary,
            action=primary_issue.action,
            samples_used=signals.samples,
        )
    else:
        primary = _default_primary(signals)

    return DiagnosticResult(
        primary=primary,
        issues=issues,
    )


__all__ = [
    "ProcessDiagnosis",
    "diagnose_process",
]
