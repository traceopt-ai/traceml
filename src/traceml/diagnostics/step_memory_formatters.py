"""
Formatting helpers for step-memory diagnosis.

These helpers intentionally keep CLI and dashboard output compact and
memory-specific.
"""

from __future__ import annotations

from .step_memory import StepMemoryDiagnosis


def _styled_status(diagnosis: StepMemoryDiagnosis) -> str:
    """Render a colored status label for Rich CLI output."""
    if diagnosis.kind == "BALANCED":
        style = "bold green"
    elif diagnosis.kind == "NO_DATA":
        style = "bold bright_black"
    elif diagnosis.kind == "CREEP_EARLY":
        style = "bold yellow"
    elif diagnosis.kind == "CREEP_CONFIRMED":
        style = "bold red"
    elif diagnosis.kind in {"HIGH_PRESSURE", "IMBALANCE"}:
        style = "bold red" if diagnosis.severity == "crit" else "bold yellow"
    else:
        style = {
            "crit": "bold red",
            "warn": "bold yellow",
            "info": "bold",
        }.get(diagnosis.severity, "bold")

    return f"[{style}]{diagnosis.status}[/{style}]"


def format_cli_diagnosis(diagnosis: StepMemoryDiagnosis) -> str:
    """Render a compact Rich-friendly diagnosis block for terminal output."""
    status = _styled_status(diagnosis)
    lines = [
        f"[bold]Issue:[/bold] {status}",
        f"[bold]Why:[/bold] {diagnosis.reason}",
        f"[bold]Next:[/bold] {diagnosis.action}",
    ]
    if diagnosis.note:
        lines.append(f"[bold]Stats:[/bold] {diagnosis.note}")
    return "\n".join(lines)


def format_dashboard_diagnosis(diagnosis: StepMemoryDiagnosis) -> str:
    """Render a compact diagnosis block for dashboard use."""
    meta = f"*Window: {diagnosis.steps_used} steps"
    if diagnosis.worst_rank is not None and diagnosis.kind != "BALANCED":
        meta += f" · Worst rank: r{diagnosis.worst_rank}"
    meta += "*"

    text = (
        f"**Status:** {diagnosis.status}  \n"
        f"**Why:** {diagnosis.reason}  \n"
        f"**Next:** {diagnosis.action}  \n"
        f"{meta}"
    )
    if diagnosis.note:
        text += f"  \n*Stats:* {diagnosis.note}"
    return text
