# /Users/abhinavsrivastav/Documents/projects/traceml/src/traceml/diagnostics/step_time_formatters.py
"""
Formatting helpers for step-time diagnosis.

Separated from core rule logic to keep diagnosis engine lean and reusable.
"""

from .step_time import StepDiagnosis


def _styled_status(status: str, severity: str) -> str:
    """Render a colored status label for Rich CLI output."""
    style = {
        "crit": "bold red",
        "warn": "bold yellow",
        "info": "bold green",
    }.get(severity, "bold")
    return f"[{style}]{status}[/{style}]"


def format_cli_diagnosis(diagnosis: StepDiagnosis) -> str:
    """Render a short Rich-friendly diagnosis block for terminal output."""
    status = _styled_status(diagnosis.status, diagnosis.severity)
    lines = [
        f"[bold]Issue:[/bold] {status}",
        f"[bold]Why:[/bold] {diagnosis.reason}",
        f"[bold]Hint:[/bold] {diagnosis.action}",
    ]
    if diagnosis.note:
        lines.append(f"[bold]Note:[/bold] {diagnosis.note}")
    return "\n".join(lines)


def format_dashboard_diagnosis(diagnosis: StepDiagnosis) -> str:
    """Render a short diagnosis block for dashboard use."""
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
        text += f"  \n*Note:* {diagnosis.note}"
    return text
