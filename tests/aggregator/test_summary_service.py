import json

import pytest

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


def _counting_service(monkeypatch, tmp_path, *, summary_finalized):
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
        flush_history=lambda timeout: True,
        settle_telemetry=lambda timeout: calls.append("settle") or True,
        summary_finalized=summary_finalized,
    )
    return service, calls


def test_existing_summary_regenerates_until_the_final_one_is_written(
    monkeypatch,
    tmp_path,
):
    finalized = [False]
    service, calls = _counting_service(
        monkeypatch, tmp_path, summary_finalized=lambda: finalized[0]
    )

    service._handle_request(request_id="request-1")
    assert get_final_summary_json_path(tmp_path).is_file()
    # Every rank may have finished by now; only the end-of-run report
    # makes the artifact final.
    service._handle_request(request_id="request-2")
    assert calls == ["settle", "generate", "settle", "generate"]

    finalized[0] = True
    service._handle_request(request_id="request-3")

    response = load_json_or_none(get_final_summary_response_path(tmp_path))
    assert response["status"] == "ok"
    assert response["request_id"] == "request-3"
    assert calls == ["settle", "generate", "settle", "generate"]


def _raise_generation(*args, **kwargs):
    raise RuntimeError("generation boom")


@pytest.mark.parametrize(
    ("settle_ok", "generate", "reason"),
    [
        (False, None, "settle"),
        (True, _raise_generation, "generation boom"),
    ],
)
def test_failed_refresh_answers_with_existing_summary_and_warning(
    monkeypatch,
    tmp_path,
    settle_ok,
    generate,
    reason,
):
    get_final_summary_json_path(tmp_path).write_text(
        json.dumps({"schema_version": "1"}),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        summary_service,
        "generate_summary",
        generate or (lambda *args, **kwargs: calls.append("generate")),
    )
    service = FinalSummaryService(
        logger=_Logger(),
        session_root=tmp_path,
        db_path="db",
        flush_history=lambda timeout: True,
        settle_telemetry=lambda timeout: settle_ok,
        summary_finalized=lambda: False,
    )

    service._handle_request(request_id="request-1")

    response = load_json_or_none(get_final_summary_response_path(tmp_path))
    assert response["status"] == "ok"
    assert response["request_id"] == "request-1"
    assert response["summary_json_path"] == str(
        get_final_summary_json_path(tmp_path)
    )
    assert reason in response["warning"]
    assert calls == []
