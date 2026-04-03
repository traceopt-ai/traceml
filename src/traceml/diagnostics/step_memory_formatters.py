"""
Formatting helpers for step-memory diagnosis.

These helpers intentionally keep CLI output compact and high-signal:
- short colored status
- concise reason/action lines
- compact stats line with only the most useful fields
"""

from __future__ import annotations

from .step_memory import StepMemoryDiagnosis


def _styled_status(diagnosis: StepMemoryDiagnosis) -> str:
    """
    Render a colored status label for Rich CLI output.

    Color policy
    ------------
    - confirmed/high-risk states: red or yellow
    - early creep / watch-like states: yellow
    - balanced: green
    - no data: dim
    """
    if diagnosis.kind == "BALANCED":
        style = "bold green"
    elif diagnosis.kind == "NO_DATA":
        style = "bold bright_black"
    elif diagnosis.kind in {"CREEP_EARLY", "IMBALANCE", "HIGH_PRESSURE"}:
        style = "bold yellow"
    elif diagnosis.kind == "CREEP_CONFIRMED":
        style = "bold red"
    else:
        style = {
            "crit": "bold red",
            "warn": "bold yellow",
            "info": "bold",
        }.get(diagnosis.severity, "bold")

    return f"[{style}]{diagnosis.status}[/{style}]"


def format_cli_diagnosis(diagnosis: StepMemoryDiagnosis) -> str:
    """
    Render a compact Rich-friendly diagnosis block for terminal output.
    """
    status = _styled_status(diagnosis)
    lines = [
        f"[bold]Issue:[/bold] {status}",
        f"[bold]Why:[/bold] {diagnosis.reason}",
        f"[bold]Hint:[/bold] {diagnosis.action}",
    ]

    stats = _compact_note(diagnosis.note)
    if stats:
        lines.append(f"[bold]Stats:[/bold] {stats}")

    return "\n".join(lines)


def format_dashboard_diagnosis(diagnosis: StepMemoryDiagnosis) -> str:
    """
    Render a compact diagnosis block for dashboard use.
    """
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

    stats = _compact_note(diagnosis.note)
    if stats:
        text += f"  \n*Stats:* {stats}"

    return text


def _compact_note(note: str | None) -> str | None:
    """
    Compress verbose diagnostic note text into a smaller CLI-friendly form.

    Examples
    --------
    Input:
      peak_allocated: window_Δ=3.30 GiB, worst_window_trend=20.5%, median_window_trend=20.5%

    Output:
      delta=3.30 GiB, trend=20.5%
    """
    if not note:
        return None

    value = str(note).strip()
    if not value:
        return None

    if ": " in value:
        _, value = value.split(": ", 1)

    parts = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue

        if item.startswith("window_Δ="):
            parts.append("delta=" + item.split("=", 1)[1])
        elif item.startswith("Δ="):
            parts.append("delta=" + item.split("=", 1)[1])
        elif item.startswith("worst_window_trend="):
            parts.append("trend=" + item.split("=", 1)[1])
        elif item.startswith("worst_trend="):
            parts.append("trend=" + item.split("=", 1)[1])
        elif item.startswith("median_window_trend="):
            continue
        elif item.startswith("median_trend="):
            continue
        elif item.startswith("worst_slope="):
            parts.append("slope=" + item.split("=", 1)[1])
        elif item.startswith("weak_recovery="):
            parts.append(item)
        elif item.startswith("median_slope="):
            continue

    if not parts:
        return value

    return ", ".join(parts[:3])
