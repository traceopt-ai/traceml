"""
End-of-run diagnosis presentation helpers.

This module adapts shared diagnosis objects into wording that fits a completed
run summary.

End-of-run summaries have a different job:
- explain what the run most likely showed
- tell the user what to do next
- avoid live-monitoring phrasing once the run is already over

To keep the architecture clean, this module does not change diagnosis truth or
threshold logic. It only rewrites wording for summary presentation.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SummaryDiagnosisPresentation:
    """
    Summary-ready diagnosis wording for end-of-run output.

    Fields
    ------
    status:
        Final diagnosis label shown to the user.
    reason:
        Concise explanation of why that diagnosis was selected.
    action:
        End-of-run next step phrasing. This is intentionally different from
        live-view wording when needed.
    note:
        Optional supporting context.
    """

    status: str
    reason: str
    action: str
    note: Optional[str] = None


def diagnosis_presentation_to_json(
    presentation: Optional[SummaryDiagnosisPresentation],
) -> Optional[Dict[str, Any]]:
    """
    Serialize a summary diagnosis presentation into JSON.
    """
    if presentation is None:
        return None
    return asdict(presentation)


def present_step_time_summary_diagnosis(
    diagnosis: Optional[Any],
) -> Optional[SummaryDiagnosisPresentation]:
    """
    Adapt a step-time diagnosis object for end-of-run summary display.

    This keeps the original diagnosis truth, but normalizes live-style actions
    into wording that makes sense after the run has finished.
    """
    if diagnosis is None:
        return None

    status = str(getattr(diagnosis, "status", "") or "").strip()
    reason = str(getattr(diagnosis, "reason", "") or "").strip()
    action = str(getattr(diagnosis, "action", "") or "").strip()
    note = getattr(diagnosis, "note", None)

    if status == "NO DATA":
        action = "Run longer for a stable timing diagnosis."
    elif action in {
        "Wait for a fuller window.",
        "Wait for more completed steps.",
    }:
        action = (
            "This run did not collect enough step data for a stable timing "
            "diagnosis."
        )

    return SummaryDiagnosisPresentation(
        status=status,
        reason=reason,
        action=action,
        note=note,
    )


def present_step_memory_summary_diagnosis(
    diagnosis: Optional[Any],
) -> Optional[SummaryDiagnosisPresentation]:
    """
    Adapt a step-memory diagnosis object for end-of-run summary display.

    Memory diagnoses are especially sensitive to live-window wording, so this
    function rewrites operational actions into end-of-run recommendations while
    preserving the original diagnosis label and reason.
    """
    if diagnosis is None:
        return None

    status = str(getattr(diagnosis, "status", "") or "").strip()
    reason = str(getattr(diagnosis, "reason", "") or "").strip()
    action = str(getattr(diagnosis, "action", "") or "").strip()
    note = getattr(diagnosis, "note", None)

    action_overrides = {
        "BALANCED": ("No pressure, skew, or creep signal."),
        "HIGH PRESSURE": ("Peak memory is close to capacity."),
        "IMBALANCE": ("Memory usage differs across ranks."),
        "MEMORY CREEP (EARLY)": ("Memory is rising in the tail window."),
        "MEMORY CREEP": ("Memory keeps rising over time."),
        "NO GPU": ("Step memory is not applicable for this run."),
        "NO DATA": ("Too little aligned memory data was collected."),
    }

    if status in action_overrides:
        action = action_overrides[status]
    elif action in {
        "Keep monitoring.",
        "Watch the next window.",
        "Wait for more completed steps.",
        "Wait for a fuller window.",
    }:
        action = (
            "Review the recorded memory trend if this behavior was unexpected."
        )

    return SummaryDiagnosisPresentation(
        status=status,
        reason=reason,
        action=action,
        note=note,
    )
