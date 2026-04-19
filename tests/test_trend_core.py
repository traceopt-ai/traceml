from traceml.analytics.trends import (
    TrendComputationConfig,
    compute_trend_evidence,
    compute_trend_pct,
)


def test_compute_trend_evidence_detects_rising_series() -> None:
    series = [100.0 + float(i) for i in range(500)]
    evidence = compute_trend_evidence(series)
    assert evidence is not None
    assert evidence.delta_vs_baseline > 0.0
    assert evidence.delta_pct_vs_baseline is not None
    assert evidence.recent_avg > evidence.mid_avg > evidence.baseline_avg


def test_compute_trend_evidence_respects_history_limit() -> None:
    cfg = TrendComputationConfig(history_limit=200, min_points=50)
    series = [10.0] * 1000
    evidence = compute_trend_evidence(series, config=cfg)
    assert evidence is not None
    assert evidence.truncated is True
    assert evidence.points_used == 200


def test_compute_trend_pct_returns_none_for_short_series() -> None:
    cfg = TrendComputationConfig(min_points=50)
    assert compute_trend_pct([1.0, 2.0, 3.0], config=cfg) is None
