import json

import pytest

from traceml_ai.runtime import lifecycle
from traceml_ai.sdk import summary_client
from traceml_ai.sdk.protocol import (
    FinalSummaryRequest,
    get_final_summary_json_path,
    get_final_summary_request_path,
    get_final_summary_response_path,
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


@pytest.mark.parametrize(
    "api", [summary_client.summary, summary_client.final_summary]
)
@pytest.mark.parametrize(
    ("mode", "history_enabled"), [("telemetry", "1"), ("cli", "0")]
)
def test_history_disabled_summary_fails_without_request_or_poll(
    monkeypatch, tmp_path, api, mode, history_enabled
):
    session_root = _configure_session(monkeypatch, tmp_path)
    monkeypatch.setenv("TRACEML_UI_MODE", mode)
    monkeypatch.setenv("TRACEML_HISTORY_ENABLED", history_enabled)
    monkeypatch.setattr(summary_client, "is_primary_rank", lambda: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled history must not request or poll")

    monkeypatch.setattr(summary_client, "write_json_atomic", forbidden)
    monkeypatch.setattr(summary_client, "load_json_or_none", forbidden)
    with pytest.raises(
        RuntimeError, match="summary APIs require local history"
    ):
        api()
    assert not get_final_summary_request_path(session_root).exists()

    monkeypatch.setattr(summary_client, "is_primary_rank", lambda: False)
    assert api() is None


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
                "schema_version": 1.7,
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
            global_rank=0,
            local_rank=0,
            node_rank=0,
        ),
    )

    with pytest.raises(RuntimeError, match="Timed out"):
        summary_client.final_summary(timeout_sec=0.01, poll_interval_sec=0.01)

    assert get_final_summary_request_path(session_root).exists()


class _FakeRuntimeHandle:
    def __init__(self, calls, *, is_live, flush_result=True):
        self.calls = calls
        self.is_live = is_live
        self.flush_result = flush_result

    def flush_pending_telemetry(self, timeout_sec):
        self.calls.append(("flush", timeout_sec))
        return self.flush_result


def _answer_requests_like_the_aggregator(
    monkeypatch, session_root, calls, warning=None
):
    """Answer each written request with a freshly generated summary."""
    real_write = summary_client.write_json_atomic

    def write_and_answer(path, payload):
        real_write(path, payload)
        calls.append(("request", payload["request_id"]))
        get_final_summary_json_path(session_root).write_text(
            json.dumps({"schema_version": "fresh"}),
            encoding="utf-8",
        )
        real_write(
            get_final_summary_response_path(session_root),
            {
                "request_id": payload["request_id"],
                "status": "ok",
                "summary_json_path": str(
                    get_final_summary_json_path(session_root)
                ),
                "warning": warning,
            },
        )

    monkeypatch.setattr(summary_client, "write_json_atomic", write_and_answer)


def test_live_runtime_flushes_then_requests_despite_existing_artifact(
    monkeypatch,
    tmp_path,
):
    session_root = _configure_session(monkeypatch, tmp_path)
    session_root.mkdir(parents=True)
    get_final_summary_json_path(session_root).write_text(
        json.dumps({"schema_version": "stale"}),
        encoding="utf-8",
    )
    calls = []
    handle = _FakeRuntimeHandle(calls, is_live=True)
    monkeypatch.setattr(lifecycle, "get_active_runtime_handle", lambda: handle)
    _answer_requests_like_the_aggregator(monkeypatch, session_root, calls)

    result = summary_client.final_summary(timeout_sec=5.0)

    assert result == {"schema_version": "fresh"}
    assert [name for name, _ in calls] == ["flush", "request"]
    assert 0.0 < calls[0][1] <= 5.0


def test_stopped_runtime_reuses_existing_artifact_without_flush(
    monkeypatch,
    tmp_path,
):
    session_root = _configure_session(monkeypatch, tmp_path)
    session_root.mkdir(parents=True)
    get_final_summary_json_path(session_root).write_text(
        json.dumps({"schema_version": "final"}),
        encoding="utf-8",
    )
    calls = []
    handle = _FakeRuntimeHandle(calls, is_live=False)
    monkeypatch.setattr(lifecycle, "get_active_runtime_handle", lambda: handle)

    result = summary_client.final_summary()

    assert result == {"schema_version": "final"}
    assert calls == []
    assert not get_final_summary_request_path(session_root).exists()


def test_failed_flush_warns_once_and_still_requests(
    monkeypatch,
    tmp_path,
    capsys,
):
    session_root = _configure_session(monkeypatch, tmp_path)
    session_root.mkdir(parents=True)
    calls = []
    handle = _FakeRuntimeHandle(calls, is_live=True, flush_result=False)
    monkeypatch.setattr(lifecycle, "get_active_runtime_handle", lambda: handle)
    _answer_requests_like_the_aggregator(monkeypatch, session_root, calls)

    result = summary_client.final_summary(timeout_sec=5.0)

    assert result == {"schema_version": "fresh"}
    assert [name for name, _ in calls] == ["flush", "request"]
    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("[TraceML]")
    ]
    assert len(lines) == 1
    assert "pending steps may be missing" in lines[0]


def test_response_warning_is_printed_once_and_summary_returned(
    monkeypatch,
    tmp_path,
    capsys,
):
    session_root = _configure_session(monkeypatch, tmp_path)
    session_root.mkdir(parents=True)
    calls = []
    handle = _FakeRuntimeHandle(calls, is_live=True)
    monkeypatch.setattr(lifecycle, "get_active_runtime_handle", lambda: handle)
    _answer_requests_like_the_aggregator(
        monkeypatch,
        session_root,
        calls,
        warning="Returned the previous summary because the refresh failed",
    )

    result = summary_client.final_summary(timeout_sec=5.0)

    assert result == {"schema_version": "fresh"}
    err = capsys.readouterr().err
    assert err.count("refresh failed") == 1
    assert "[TraceML]" in err


def test_completed_recording_requests_without_a_flush_warning(
    monkeypatch,
    tmp_path,
    capsys,
):
    from tests.runtime.test_sampler_registry import _completed_runtime
    from traceml_ai.runtime.state import configure_trace_recording

    session_root = _configure_session(monkeypatch, tmp_path)
    session_root.mkdir(parents=True)
    runtime = _completed_runtime(monkeypatch, tmp_path)
    try:
        handle = lifecycle.RuntimeHandle(runtime)
        monkeypatch.setattr(
            lifecycle, "get_active_runtime_handle", lambda: handle
        )
        calls = []
        _answer_requests_like_the_aggregator(monkeypatch, session_root, calls)
        capsys.readouterr()

        result = summary_client.final_summary(timeout_sec=5.0)

        assert result == {"schema_version": "fresh"}
        assert [name for name, _ in calls] == ["request"]
        assert "[TraceML]" not in capsys.readouterr().err
    finally:
        runtime.stop()
        configure_trace_recording()
