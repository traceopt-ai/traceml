"""
Tests for TraceML W&B integration.

All tests use mocks — no real W&B account or network connection required.
"""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from traceml.integrations.wandb import (
    WandbSummaryExporter,
    _flatten_summary,
    log_traceml_summary_to_wandb,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYSTEM_SUMMARY: Dict[str, Any] = {
    "duration_s": 120.5,
    "cpu_avg_percent": 45.2,
    "cpu_peak_percent": 92.1,
    "ram_avg_gb": 8.3,
    "ram_peak_gb": 12.1,
    "ram_total_gb": 32.0,
    "gpu_available": True,
    "gpu_count": 1,
    "gpu_util_avg_percent": 87.5,
    "gpu_util_peak_percent": 100.0,
    "gpu_mem_avg_gb": 14.2,
    "gpu_mem_peak_gb": 15.9,
    "gpu_temp_avg_c": 72.0,
    "gpu_temp_peak_c": 80.0,
    "gpu_power_avg_w": 280.0,
    "gpu_power_peak_w": 320.0,
    "card": "<rendered text>",
}

STEP_TIME_SUMMARY: Dict[str, Any] = {
    "training_steps": 501,
    "ranks_seen": 1,
    "worst_avg_step_ms": 148.3,
    "median_avg_step_ms": 148.3,
    "worst_vs_median_pct": None,
    "median_split_ms": {
        "dataloader": 10.5,
        "forward": 55.2,
        "backward": 62.1,
        "optimizer": 18.4,
    },
    "worst_split_ms": {
        "dataloader": 10.5,
        "forward": 55.2,
        "backward": 62.1,
        "optimizer": 18.4,
    },
    "card": "<rendered text>",
}

FULL_SUMMARY: Dict[str, Any] = {
    "system": SYSTEM_SUMMARY,
    "step_time": STEP_TIME_SUMMARY,
}


def _make_mock_run() -> MagicMock:
    """Return a mock object that behaves like ``wandb.Run``."""
    run = MagicMock()
    run.summary = MagicMock()
    run.log_artifact = MagicMock()
    return run


# ---------------------------------------------------------------------------
# 1. test_metric_keys_stable
#    Verify the exact set of flat metric keys matches the documented schema.
# ---------------------------------------------------------------------------

# Expected stable key set (must NOT change without a major-version bump)
EXPECTED_KEYS = {
    "traceml/system/duration_s",
    "traceml/system/cpu_avg_percent",
    "traceml/system/cpu_peak_percent",
    "traceml/system/ram_avg_gb",
    "traceml/system/ram_peak_gb",
    "traceml/system/ram_total_gb",
    "traceml/system/gpu_available",
    "traceml/system/gpu_count",
    "traceml/system/gpu_util_avg_percent",
    "traceml/system/gpu_util_peak_percent",
    "traceml/system/gpu_mem_avg_gb",
    "traceml/system/gpu_mem_peak_gb",
    "traceml/system/gpu_temp_avg_c",
    "traceml/system/gpu_temp_peak_c",
    "traceml/system/gpu_power_avg_w",
    "traceml/system/gpu_power_peak_w",
    "traceml/step_time/training_steps",
    "traceml/step_time/ranks_seen",
    "traceml/step_time/worst_avg_step_ms",
    "traceml/step_time/median_avg_step_ms",
    # worst_vs_median_pct is None in the fixture, so it must NOT appear
    "traceml/step_time/median_dataloader_ms",
    "traceml/step_time/median_forward_ms",
    "traceml/step_time/median_backward_ms",
    "traceml/step_time/median_optimizer_ms",
    "traceml/step_time/worst_dataloader_ms",
    "traceml/step_time/worst_forward_ms",
    "traceml/step_time/worst_backward_ms",
    "traceml/step_time/worst_optimizer_ms",
}


def test_metric_keys_stable():
    """Flat metrics must exactly match the documented stable schema."""
    flat = _flatten_summary(FULL_SUMMARY)
    assert set(flat.keys()) == EXPECTED_KEYS, (
        f"Extra keys: {set(flat.keys()) - EXPECTED_KEYS}\n"
        f"Missing keys: {EXPECTED_KEYS - set(flat.keys())}"
    )


def test_metric_none_values_excluded():
    """Keys whose source value is None must be silently dropped."""
    flat = _flatten_summary(FULL_SUMMARY)
    # worst_vs_median_pct is None in the fixture
    assert "traceml/step_time/worst_vs_median_pct" not in flat


def test_metric_values_are_numeric():
    """All emitted values must be int, float, or bool (no None/str)."""
    flat = _flatten_summary(FULL_SUMMARY)
    for key, val in flat.items():
        assert isinstance(
            val, (int, float, bool)
        ), f"Key '{key}' has non-numeric value: {val!r}"


# ---------------------------------------------------------------------------
# 2. test_export_success
#    Mock wandb; assert metrics are sent to run.summary and artifact is logged.
# ---------------------------------------------------------------------------


def test_export_success():
    """WandbSummaryExporter.export() logs metrics and uploads artifact."""
    mock_run = _make_mock_run()

    mock_wandb = MagicMock()
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact
    mock_wandb.run = None  # ensure we use the explicit run arg

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        result = WandbSummaryExporter.export(FULL_SUMMARY, run=mock_run)

    assert result is True
    mock_run.summary.update.assert_called_once()
    updated_dict = mock_run.summary.update.call_args[0][0]
    assert "traceml/system/cpu_avg_percent" in updated_dict
    assert "traceml/step_time/training_steps" in updated_dict
    mock_run.log_artifact.assert_called_once_with(mock_artifact)


def test_export_uses_global_run_when_none_provided():
    """WandbSummaryExporter.export() falls back to wandb.run when run=None."""
    mock_run = _make_mock_run()

    mock_wandb = MagicMock()
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact
    mock_wandb.run = mock_run

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        result = WandbSummaryExporter.export(FULL_SUMMARY, run=None)

    assert result is True
    mock_run.summary.update.assert_called_once()


# ---------------------------------------------------------------------------
# 3. test_export_graceful_no_wandb
#    ImportError for wandb → returns False, never raises.
# ---------------------------------------------------------------------------


def test_export_graceful_no_wandb():
    """If wandb is not installed, export() returns False silently."""
    import sys

    # Temporarily remove wandb from sys.modules and make import fail
    original = sys.modules.pop("wandb", None)
    try:
        with patch.dict("sys.modules", {"wandb": None}):
            result = WandbSummaryExporter.export(FULL_SUMMARY, run=None)
    finally:
        if original is not None:
            sys.modules["wandb"] = original

    assert result is False


# ---------------------------------------------------------------------------
# 4. test_export_graceful_exception
#    Exceptions inside wandb calls → returns False, never raises.
# ---------------------------------------------------------------------------


def test_export_graceful_exception():
    """Any exception from wandb API is caught; export() returns False."""
    mock_run = _make_mock_run()
    mock_run.summary.update.side_effect = RuntimeError("wandb API exploded")

    mock_wandb = MagicMock()
    mock_wandb.run = None

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        result = WandbSummaryExporter.export(FULL_SUMMARY, run=mock_run)

    assert result is False


# ---------------------------------------------------------------------------
# 5. test_export_no_active_run
#    Neither explicit run nor global wandb.run → returns False with warning.
# ---------------------------------------------------------------------------


def test_export_no_active_run():
    """Returns False when no run is available (explicit or global)."""
    mock_wandb = MagicMock()
    mock_wandb.run = None  # no active run

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        result = WandbSummaryExporter.export(FULL_SUMMARY, run=None)

    assert result is False


# ---------------------------------------------------------------------------
# 6. test_log_traceml_summary_from_file
#    High-level function reads JSON file and delegates to export().
# ---------------------------------------------------------------------------


def test_log_traceml_summary_from_file():
    """log_traceml_summary_to_wandb loads a JSON file and calls export."""
    mock_run = _make_mock_run()

    mock_wandb = MagicMock()
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact
    mock_wandb.run = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "summary.json"
        json_path.write_text(json.dumps(FULL_SUMMARY), encoding="utf-8")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = log_traceml_summary_to_wandb(
                summary_json_path=str(json_path),
                run=mock_run,
            )

    assert result is True
    mock_run.summary.update.assert_called_once()


def test_log_traceml_summary_missing_file():
    """Returns False gracefully when the JSON file does not exist."""
    result = log_traceml_summary_to_wandb(
        summary_json_path="/nonexistent/path/summary.json",
        run=_make_mock_run(),
    )
    assert result is False


# ---------------------------------------------------------------------------
# 7. test_generate_summary_wandb_integration
#    Verify final_summary.generate_summary() triggers W&B export when
#    wandb_run is provided.
# ---------------------------------------------------------------------------


def test_generate_summary_triggers_wandb_export(tmp_path):
    """
    generate_summary(db_path, wandb_run=...) should call
    log_traceml_summary_to_wandb after writing the JSON card.
    """
    # We patch the SQLite-based summary generators to avoid needing a real DB.
    # log_traceml_summary_to_wandb is imported lazily inside generate_summary(),
    # so we patch it at its definition site in traceml.integrations.wandb.
    with (
        patch("traceml.aggregator.final_summary.generate_system_summary_card"),
        patch(
            "traceml.aggregator.final_summary.generate_step_time_summary_card"
        ),
        patch(
            "traceml.integrations.wandb.log_traceml_summary_to_wandb"
        ) as mock_upload,
    ):
        from traceml.aggregator.final_summary import generate_summary

        mock_run = _make_mock_run()
        db_path = str(tmp_path / "session.db")

        generate_summary(db_path, wandb_run=mock_run)

        mock_upload.assert_called_once_with(
            summary_json_path=db_path + "_summary_card.json",
            run=mock_run,
        )


def test_generate_summary_no_wandb_run(tmp_path):
    """
    generate_summary() without wandb_run must NOT call upload at all.
    """
    with (
        patch("traceml.aggregator.final_summary.generate_system_summary_card"),
        patch(
            "traceml.aggregator.final_summary.generate_step_time_summary_card"
        ),
        patch(
            "traceml.integrations.wandb.log_traceml_summary_to_wandb"
        ) as mock_upload,
    ):
        from traceml.aggregator.final_summary import generate_summary

        db_path = str(tmp_path / "session.db")
        generate_summary(db_path)

        mock_upload.assert_not_called()
