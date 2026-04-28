from dataclasses import dataclass

from traceml.diagnostics.common import (
    BaseDiagnosis,
    DiagnosticIssue,
    DiagnosticResult,
    sort_issues,
)


@dataclass(frozen=True)
class ExampleDiagnosis(BaseDiagnosis):
    kind: str


def _diagnosis(kind: str = "OK") -> ExampleDiagnosis:
    return ExampleDiagnosis(
        severity="info",
        status=kind,
        reason="example",
        action="none",
        kind=kind,
    )


def test_diagnostic_result_uses_issues_as_canonical_signals() -> None:
    issue = DiagnosticIssue(
        kind="HIGH_CPU",
        status="HIGH CPU",
        severity="warn",
        summary="CPU is high.",
        action="Inspect worker load.",
        score=0.5,
    )

    result: DiagnosticResult[ExampleDiagnosis] = DiagnosticResult(
        primary=_diagnosis(),
        issues=(issue,),
    )

    assert result.primary.kind == "OK"
    assert result.issues == (issue,)


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
