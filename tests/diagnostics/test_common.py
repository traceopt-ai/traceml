from dataclasses import dataclass

from traceml_ai.diagnostics.common import (
    BaseDiagnosis,
    DiagnosticIssue,
    DiagnosticResult,
    diagnosis_to_issue,
    ensure_primary_issue,
    sort_issues,
)
from traceml_ai.reporting.summaries.issue_summary import (
    diagnostic_result_to_json,
)


@dataclass(frozen=True)
class ExampleDiagnosis(BaseDiagnosis):
    kind: str
    samples_used: int = 0
    worst_rank: int | None = None


def _diagnosis(
    kind: str = "OK",
    *,
    samples_used: int = 0,
    worst_rank: int | None = None,
) -> ExampleDiagnosis:
    return ExampleDiagnosis(
        severity="info",
        status=kind,
        reason="example",
        action="none",
        kind=kind,
        samples_used=samples_used,
        worst_rank=worst_rank,
    )


def test_diagnostic_result_uses_primary_as_neutral_issue_when_empty() -> None:
    result: DiagnosticResult[ExampleDiagnosis] = DiagnosticResult(
        primary=_diagnosis(kind="NORMAL"),
    )

    assert result.issues[0].kind == "NORMAL"
    assert result.issues[0].status == "NORMAL"
    assert result.issues[0].summary == "example"
    assert result.issues[0].evidence == {"samples_used": 0}


def test_diagnostic_result_keeps_provided_primary_issue_first() -> None:
    issue = DiagnosticIssue(
        kind="HIGH_CPU",
        status="HIGH CPU",
        severity="warn",
        summary="CPU is high.",
        action="Inspect worker load.",
        score=0.5,
    )

    result: DiagnosticResult[ExampleDiagnosis] = DiagnosticResult(
        primary=_diagnosis(kind="HIGH_CPU"),
        issues=(issue,),
    )

    assert result.primary.kind == "HIGH_CPU"
    assert result.issues == (issue,)


def test_diagnostic_result_moves_matching_primary_issue_to_front() -> None:
    primary_issue = DiagnosticIssue(
        kind="HIGH_CPU",
        status="HIGH CPU",
        severity="warn",
        summary="CPU is high.",
        action="Inspect worker load.",
    )
    secondary_issue = DiagnosticIssue(
        kind="LOW_GPU_UTILIZATION",
        status="LOW GPU UTILIZATION",
        severity="info",
        summary="GPU utilization is low.",
        action="Inspect timing.",
    )

    result: DiagnosticResult[ExampleDiagnosis] = DiagnosticResult(
        primary=_diagnosis(kind="HIGH_CPU"),
        issues=(secondary_issue, primary_issue),
    )

    assert result.issues == (primary_issue, secondary_issue)


def test_diagnosis_to_issue_preserves_structured_evidence() -> None:
    issue = diagnosis_to_issue(
        _diagnosis(kind="WARMUP", samples_used=12, worst_rank=2)
    )

    assert issue.kind == "WARMUP"
    assert issue.summary == "example"
    assert issue.ranks == (2,)
    assert issue.evidence == {"samples_used": 12, "worst_rank": 2}


def test_ensure_primary_issue_prepends_primary_when_missing() -> None:
    secondary_issue = DiagnosticIssue(
        kind="LOW_GPU_UTILIZATION",
        status="LOW GPU UTILIZATION",
        severity="info",
        summary="GPU utilization is low.",
        action="Inspect timing.",
    )

    issues = ensure_primary_issue(
        _diagnosis(kind="HIGH_CPU"),
        (secondary_issue,),
    )

    assert issues[0].kind == "HIGH_CPU"
    assert issues[1] == secondary_issue


def test_diagnostic_result_json_uses_first_issue_as_diagnosis() -> None:
    issue = DiagnosticIssue(
        kind="HIGH_CPU",
        status="HIGH CPU",
        severity="warn",
        summary="CPU is high.",
        action="Inspect worker load.",
    )
    result: DiagnosticResult[ExampleDiagnosis] = DiagnosticResult(
        primary=_diagnosis(kind="HIGH_CPU"),
        issues=(issue,),
    )

    diagnosis, issues = diagnostic_result_to_json(result)

    assert diagnosis == issues[0]
    assert diagnosis["kind"] == "HIGH_CPU"


def test_sort_issues_orders_by_severity_score_and_rank_breadth() -> None:
    low = DiagnosticIssue(
        kind="LOW",
        status="LOW",
        severity="info",
        summary="low",
        action="none",
        score=1.0,
        ranks=(0, 1),
    )
    high_low_score = DiagnosticIssue(
        kind="HIGH_LOW_SCORE",
        status="HIGH LOW SCORE",
        severity="crit",
        summary="high",
        action="none",
        score=0.1,
        ranks=(0,),
    )
    high_high_score = DiagnosticIssue(
        kind="HIGH_HIGH_SCORE",
        status="HIGH HIGH SCORE",
        severity="crit",
        summary="high",
        action="none",
        score=0.9,
        ranks=(0,),
    )

    assert sort_issues((low, high_low_score, high_high_score)) == (
        high_high_score,
        high_low_score,
        low,
    )
