import json

import pytest

from traceml_ai.sdk import summary_client
from traceml_ai.sdk.protocol import (
    FinalSummaryRequest,
    get_final_summary_json_path,
    get_final_summary_request_path,
    get_final_summary_txt_path,
    resolve_session_context_from_env,
)


def _configure_session(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACEML_SESSION_ID", "run-a")
    monkeypatch.setenv("TRACEML_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("TRACEML_HISTORY_ENABLED", "1")
    return tmp_path / "run-a"


def test_session_context_defaults_to_summary(monkeypatch, tmp_path):
    _configure_session(monkeypatch, tmp_path)
    monkeypatch.delenv("TRACEML_UI_MODE", raising=False)
    monkeypatch.delenv("TRACEML_MODE", raising=False)

    assert resolve_session_context_from_env().mode == "summary"


def test_final_summary_reuses_existing_artifact_without_request(
    monkeypatch,
    tmp_path,
):
    session_root = _configure_session(monkeypatch, tmp_path)
    session_root.mkdir(parents=True)
    payload = {
        "schema_version": "1",
        "duration_s": 12.0,
        "step_time": {
            "diagnosis": {"status": "BALANCED", "severity": "ok"},
            "global": {"average": {"total_step_ms": 10.0}},
        },
    }
    get_final_summary_json_path(session_root).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    result = summary_client.final_summary()

    assert result == payload
    assert not get_final_summary_request_path(session_root).exists()


def test_final_summary_prints_existing_text(monkeypatch, tmp_path, capsys):
    session_root = _configure_session(monkeypatch, tmp_path)
    session_root.mkdir(parents=True)
    get_final_summary_json_path(session_root).write_text(
        json.dumps({"schema_version": "1"}),
        encoding="utf-8",
    )
    get_final_summary_txt_path(session_root).write_text(
        "TraceML summary\n",
        encoding="utf-8",
    )

    summary_client.final_summary(print_text=True)

    assert "TraceML summary" in capsys.readouterr().out


def test_summary_projects_existing_final_payload(monkeypatch, tmp_path):
    session_root = _configure_session(monkeypatch, tmp_path)
    session_root.mkdir(parents=True)
    get_final_summary_json_path(session_root).write_text(
        json.dumps(
            {
                "schema_version": "1",
                "duration_s": 3.0,
                "primary_diagnosis": {
                    "kind": "NO_CLEAR_PERFORMANCE_BOTTLENECK",
                    "status": "NO CLEAR PERFORMANCE BOTTLENECK",
                    "severity": "info",
                    "section": "step_time",
                },
                "system": {
                    "diagnosis": {
                        "status": "NORMAL",
                        "severity": "ok",
                    },
                    "global": {
                        "average": {
                            "cpu_percent": 22.0,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = summary_client.summary()

    assert result == {
        "traceml/schema_version": "1",
        "traceml/duration_s": 3.0,
        "traceml/primary/kind": "NO_CLEAR_PERFORMANCE_BOTTLENECK",
        "traceml/primary/status": "NO CLEAR PERFORMANCE BOTTLENECK",
        "traceml/primary/severity": "info",
        "traceml/primary/section": "step_time",
        "traceml/system/status": "NORMAL",
        "traceml/system/severity": "ok",
        "traceml/system/cpu_percent": 22.0,
    }


def test_summary_projects_canonical_step_time_metrics(monkeypatch, tmp_path):
    session_root = _configure_session(monkeypatch, tmp_path)
    session_root.mkdir(parents=True)
    get_final_summary_json_path(session_root).write_text(
        json.dumps(
            {
                "schema_version": 1.8,
                "step_time": {
                    "diagnosis": {"status": "BALANCED", "severity": "info"},
                    "global": {
                        "window": {"diagnosis_clock": "gpu"},
                        "average": {
                            "step_time_ms": 120.0,
                            "traced_step_time_ms": 90.0,
                            "step_time_cpu_ms": 140.0,
                            "step_time_gpu_ms": 120.0,
                            "traced_step_time_cpu_ms": 100.0,
                            "traced_step_time_gpu_ms": 90.0,
                            "dataloader_fetch_cpu_ms": 12.0,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = summary_client.summary()

    assert result is not None
    assert result["traceml/step_time/diagnosis_clock"] == "gpu"
    assert result["traceml/step_time/step_time_ms"] == 120.0
    assert result["traceml/step_time/traced_step_time_ms"] == 90.0
    assert result["traceml/step_time/step_time_cpu_ms"] == 140.0
    assert result["traceml/step_time/step_time_gpu_ms"] == 120.0
    assert result["traceml/step_time/traced_step_time_cpu_ms"] == 100.0
    assert result["traceml/step_time/traced_step_time_gpu_ms"] == 90.0
    assert result["traceml/step_time/dataloader_fetch_cpu_ms"] == 12.0
    assert "traceml/step_time/total_step_ms" not in result


def test_final_summary_still_requests_generation_when_artifact_missing(
    monkeypatch,
    tmp_path,
):
    session_root = _configure_session(monkeypatch, tmp_path)
    request_ids = iter(["request-1"])

    monkeypatch.setattr(
        summary_client,
        "build_final_summary_request",
        lambda: FinalSummaryRequest(
            request_id=next(request_ids),
            created_at="",
            pid=1,
            rank=0,
            local_rank=0,
        ),
    )

    with pytest.raises(RuntimeError, match="Timed out"):
        summary_client.final_summary(timeout_sec=0.01, poll_interval_sec=0.01)

    assert get_final_summary_request_path(session_root).exists()
