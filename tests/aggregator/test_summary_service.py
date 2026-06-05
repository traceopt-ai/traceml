import json

from traceml_ai.aggregator import summary_service
from traceml_ai.aggregator.summary_service import FinalSummaryService
from traceml_ai.sdk.protocol import (
    get_final_summary_json_path,
    get_final_summary_response_path,
    load_json_or_none,
)


class _Logger:
    def exception(self, *args, **kwargs):
        return None


def test_existing_final_summary_skips_generation(monkeypatch, tmp_path):
    calls = []
    get_final_summary_json_path(tmp_path).write_text(
        json.dumps({"schema_version": "1"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        summary_service,
        "generate_summary",
        lambda *args, **kwargs: calls.append("generate"),
    )

    service = FinalSummaryService(
        logger=_Logger(),
        session_root=tmp_path,
        db_path="db",
        flush_history=lambda timeout: calls.append("flush") or True,
    )

    service._handle_request(request_id="request-1")

    response = load_json_or_none(get_final_summary_response_path(tmp_path))
    assert response["status"] == "ok"
    assert calls == []


def test_first_summary_generation_settles_before_generate(
    monkeypatch,
    tmp_path,
):
    calls = []

    def fake_generate(*args, **kwargs):
        calls.append("generate")
        get_final_summary_json_path(tmp_path).write_text(
            json.dumps({"schema_version": "1"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(summary_service, "generate_summary", fake_generate)

    service = FinalSummaryService(
        logger=_Logger(),
        session_root=tmp_path,
        db_path="db",
        flush_history=lambda timeout: calls.append("flush") or True,
        settle_telemetry=lambda timeout: calls.append("settle") or True,
    )

    service._handle_request(request_id="request-1")

    response = load_json_or_none(get_final_summary_response_path(tmp_path))
    assert response["status"] == "ok"
    assert calls == ["settle", "generate"]


def test_failed_settle_returns_error(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        summary_service,
        "generate_summary",
        lambda *args, **kwargs: calls.append("generate"),
    )

    service = FinalSummaryService(
        logger=_Logger(),
        session_root=tmp_path,
        db_path="db",
        flush_history=lambda timeout: True,
        settle_telemetry=lambda timeout: False,
    )

    service._handle_request(request_id="request-1")

    response = load_json_or_none(get_final_summary_response_path(tmp_path))
    assert response["status"] == "error"
    assert "settle" in response["error"]
    assert calls == []
