"""
Formatting helpers for step-memory diagnosis.
"""

from .step_memory import StepMemoryDiagnosis


def _styled_status(status: str, severity: str) -> str:
    style = {
        "crit": "bold red",
        "warn": "bold yellow",
        "info": "bold green",
    }.get(severity, "bold")
    return f"[{style}]{status}[/{style}]"


def format_cli_diagnosis(diagnosis: StepMemoryDiagnosis) -> str:
    status = _styled_status(diagnosis.status, diagnosis.severity)
    lines = [
        f"[bold]Issue:[/bold] {status}",
        f"[bold]Why:[/bold] {diagnosis.reason}",
        f"[bold]Hint:[/bold] {diagnosis.action}",
    ]
    if diagnosis.note:
        lines.append(f"[bold]Stats:[/bold] {diagnosis.note}")
    return "\n".join(lines)


def format_dashboard_diagnosis(diagnosis: StepMemoryDiagnosis) -> str:
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
