from __future__ import annotations

from traceml.reporting.summaries.step_time import (
    RankStepSummary,
)
from traceml.reporting.sections.step_time.builder import build_step_time_card


def _rank(
    *,
    steps: int = 64,
    dataloader: float = 5.0,
    forward: float = 30.0,
    backward: float = 50.0,
    optimizer: float = 10.0,
    step_cpu: float | None = None,
) -> RankStepSummary:
    compute = forward + backward + optimizer
    effective_step = max(
        step_cpu if step_cpu is not None else compute, compute
    )
    return RankStepSummary(
        steps_analyzed=steps,
        avg_dataloader_ms=dataloader,
        avg_forward_ms=forward,
        avg_backward_ms=backward,
        avg_optimizer_ms=optimizer,
        avg_step_cpu_ms=effective_step,
        avg_gpu_compute_ms=compute,
        avg_total_step_ms=dataloader + effective_step,
    )


def _summary(per_rank: dict[int, RankStepSummary]):
    _, payload = build_step_time_card(
        training_steps=100,
        latest_step_observed=99,
        per_rank_summary=per_rank,
        per_rank_step_metrics={},
        max_rows=64,
    )
    return payload


def _assert_compact_card(card: str) -> None:
    assert "- Issues:" not in card
    assert "- Note:" not in card
    assert "- Global:" not in card
    assert "- Dominant:" not in card


def test_step_time_no_data_card_is_compact() -> None:
    payload = _summary({})

    assert payload["primary_diagnosis"] is None
    assert "- Diagnosis: NO DATA" in payload["card"]
    assert "- Stats: n/a" in payload["card"]
    assert "- Why: Need more step-time samples." in payload["card"]
    _assert_compact_card(payload["card"])


def test_step_time_balanced_card_is_compact() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=20.0,
                forward=20.0,
                backward=35.0,
                optimizer=5.0,
                step_cpu=70.0,
            ),
            1: _rank(
                dataloader=20.0,
                forward=21.0,
                backward=34.0,
                optimizer=5.0,
                step_cpu=70.0,
            ),
        }
    )

    assert payload["primary_diagnosis"]["status"] == "BALANCED"
    assert "- Diagnosis: BALANCED" in payload["card"]
    assert "- Stats: median/worst |" in payload["card"]
    assert "- Ranks: median/worst |" in payload["card"]
    assert "- Why: No clear timing bottleneck." in payload["card"]
    _assert_compact_card(payload["card"])


def test_step_time_compute_bound_card_uses_short_reason() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=2.0,
                forward=20.0,
                backward=65.0,
                optimizer=5.0,
                step_cpu=95.0,
            )
        }
    )

    assert payload["primary_diagnosis"]["status"] == "COMPUTE-BOUND"
    assert "- Stats: step 97.0ms | compute 90.0ms" in payload["card"]
    assert (
        "- Why: Compute dominated (90.0ms/97.0ms); backward was largest."
        in payload["card"]
    )
    _assert_compact_card(payload["card"])


def test_step_time_input_bound_card_uses_short_reason() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=40.0,
                forward=20.0,
                backward=35.0,
                optimizer=5.0,
                step_cpu=100.0,
            )
        }
    )

    assert payload["primary_diagnosis"]["status"] == "INPUT-BOUND"
    assert (
        "- Why: Input loading took a large share (40.0ms/140.0ms)."
        in payload["card"]
    )
    _assert_compact_card(payload["card"])


def test_step_time_wait_heavy_card_uses_short_reason() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=2.0,
                forward=20.0,
                backward=45.0,
                optimizer=5.0,
                step_cpu=100.0,
            )
        }
    )

    assert payload["primary_diagnosis"]["status"] == "WAIT-HEAVY"
    assert "- Why: Wait time was high (30.0ms/102.0ms)." in payload["card"]
    _assert_compact_card(payload["card"])


def test_step_time_input_straggler_card_shows_rank_evidence() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=10.0,
                forward=40.0,
                backward=130.0,
                step_cpu=219.0,
            ),
            1: _rank(
                dataloader=70.0,
                forward=40.0,
                backward=130.0,
                step_cpu=219.0,
            ),
        }
    )

    assert payload["primary_diagnosis"]["status"] == "INPUT STRAGGLER"
    assert "- Ranks: median/worst |" in payload["card"]
    assert (
        "- Why: r1 input was slower than median rank (70.0/40.0ms)."
        in payload["card"]
    )
    assert {issue["kind"] for issue in payload["issues_by_rank"]["1"]} == {
        "INPUT_STRAGGLER"
    }
    _assert_compact_card(payload["card"])


def test_step_time_compute_straggler_card_shows_rank_evidence() -> None:
    payload = _summary(
        {
            0: _rank(dataloader=10.0, forward=40.0, backward=130.0),
            1: _rank(dataloader=10.0, forward=90.0, backward=160.0),
        }
    )

    assert payload["primary_diagnosis"]["status"] == "COMPUTE STRAGGLER"
    assert (
        "- Why: r1 compute was slower than median rank (260.0/220.0ms)."
        in payload["card"]
    )
    assert len(payload["issues_by_rank"]["1"]) == 1
    _assert_compact_card(payload["card"])


def test_step_time_combined_straggler_priority_keeps_all_rank_issues() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=10.0,
                forward=40.0,
                backward=130.0,
            ),
            1: _rank(
                dataloader=80.0,
                forward=90.0,
                backward=160.0,
            ),
        }
    )

    assert payload["primary_diagnosis"]["status"] == "STRAGGLER"
    assert {issue["kind"] for issue in payload["issues"]} >= {
        "STRAGGLER",
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
    }
    assert {issue["kind"] for issue in payload["issues_by_rank"]["1"]} == {
        "STRAGGLER",
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
    }
    assert "- Why: Input and compute varied across ranks." in payload["card"]
    _assert_compact_card(payload["card"])


def test_step_time_priority_prefers_straggler_over_wait_heavy() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=10.0,
                forward=40.0,
                backward=130.0,
                step_cpu=350.0,
            ),
            1: _rank(
                dataloader=80.0,
                forward=90.0,
                backward=160.0,
                step_cpu=350.0,
            ),
        }
    )

    assert payload["primary_diagnosis"]["status"] == "STRAGGLER"
    assert "WAIT_HEAVY" in {issue["kind"] for issue in payload["issues"]}
    assert "- Diagnosis: STRAGGLER" in payload["card"]
