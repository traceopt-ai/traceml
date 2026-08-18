import json

import pytest

from traceml_ai.reporting.view import view_summary


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_view_summary_prints_top_level_text(tmp_path, capsys) -> None:
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, {"text": "TraceML Run Summary\n- Status: OK\n"})

    text = view_summary(summary_path)

    assert text == "TraceML Run Summary\n- Status: OK"
    assert capsys.readouterr().out == "TraceML Run Summary\n- Status: OK\n"


def test_view_summary_falls_back_to_top_level_card(tmp_path, capsys) -> None:
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, {"card": "TraceML Section Summary\n- OK"})

    text = view_summary(summary_path)

    assert text == "TraceML Section Summary\n- OK"
    assert capsys.readouterr().out == "TraceML Section Summary\n- OK\n"


def test_view_summary_can_return_without_printing(tmp_path, capsys) -> None:
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, {"text": "TraceML Run Summary"})

    text = view_summary(summary_path, print_to_stdout=False)

    assert text == "TraceML Run Summary"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("file_name", "contents", "message"),
    [
        ("bad.json", "{bad json", "not valid JSON"),
        ("array.json", "[]", "must contain a JSON object"),
        ("empty.json", "{}", "does not contain printable text"),
        ("blank.json", '{"text": "   "}', "does not contain printable text"),
    ],
)
def test_view_summary_rejects_invalid_artifacts(
    tmp_path,
    file_name,
    contents,
    message,
) -> None:
    summary_path = tmp_path / file_name
    summary_path.write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        view_summary(summary_path, print_to_stdout=False)


def test_view_summary_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Summary file not found"):
        view_summary(tmp_path / "missing.json", print_to_stdout=False)


def test_view_summary_rejects_directory_path(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Summary path is not a file"):
        view_summary(tmp_path, print_to_stdout=False)


_OLD_CARD = (
    "+------------------------+\n"
    "|  TraceML Run Summary   |\n"
    "|  Section Status        |\n"
    "+------------------------+"
)


def _payload_with_stored_card(stored_card: str) -> dict:
    """An artifact whose stored card predates the current renderer."""
    return {
        "schema_version": 1.7,
        "duration_s": 52.4,
        "meta": {
            "run_name": "old_run",
            "mode": "single_node",
            "world_size": 1,
            "nodes_observed": 1,
            "gpus_observed": 1,
        },
        "primary_diagnosis": {
            "kind": "INPUT_BOUND",
            "status": "INPUT-BOUND",
            "severity": "crit",
            "section": "step_time",
            "summary": "Input Wait is 64.0% of the typical Step Time.",
            "action": "Increase workers, prefetch, or storage throughput.",
            "evidence": {"type": "phase_share", "score": 0.64},
        },
        "system": {"global": {"average": {"cpu_percent": 18.4}}},
        "process": {},
        "step_memory": {},
        "step_time": {
            "global": {
                "window": {"steps_analyzed": 256, "diagnosis_clock": "gpu"},
                "average": {
                    "step_time_ms": 200.4,
                    "input_wait_ms": 128.0,
                    "traced_step_time_ms": 72.0,
                    "compute_ms": 68.0,
                    "residual_ms": 3.6,
                },
            }
        },
        "text": stored_card,
    }


def test_view_summary_re_renders_an_older_card(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, _payload_with_stored_card(_OLD_CARD))

    text = view_summary(summary_path, print_to_stdout=False, re_render=True)

    assert "Section Status" not in text
    assert "Verdict: INPUT-BOUND" in text
    assert "├─ Input Wait" in text
    assert "256 steps analyzed" in text


@pytest.mark.parametrize("schema_version", [1.5, 1.6])
def test_view_summary_re_render_keeps_pre_1_7_card(
    tmp_path, schema_version
) -> None:
    summary_path = tmp_path / "summary.json"
    payload = _payload_with_stored_card(_OLD_CARD)
    payload["schema_version"] = schema_version
    _write_json(summary_path, payload)

    text = view_summary(summary_path, print_to_stdout=False, re_render=True)

    assert text == _OLD_CARD


def test_view_summary_default_still_prints_the_stored_card(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, _payload_with_stored_card(_OLD_CARD))

    text = view_summary(summary_path, print_to_stdout=False)

    assert text == _OLD_CARD


def test_view_summary_re_render_infers_the_watch_profile(tmp_path) -> None:
    stored = "+---+\n|  TraceML Watch Summary  |\n+---+"
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, _payload_with_stored_card(stored))

    text = view_summary(summary_path, print_to_stdout=False, re_render=True)

    assert "TraceML Watch Summary" in text
    assert "SYSTEM METRICS:" in text
    assert "PROCESS METRICS:" in text
    assert "Verdict:" not in text
    assert "Why:" not in text
    assert "Next:" not in text
    assert "Step Time" not in text


def test_view_summary_re_render_ignores_watch_title_in_run_name(
    tmp_path,
) -> None:
    stored = (
        "+---+\n"
        "|  TraceML Run Summary  |\n"
        "|  TraceML Watch Summary · 1 rank  |\n"
        "+---+"
    )
    summary_path = tmp_path / "summary.json"
    payload = _payload_with_stored_card(stored)
    payload["meta"]["run_name"] = "TraceML Watch Summary"
    _write_json(summary_path, payload)

    text = view_summary(summary_path, print_to_stdout=False, re_render=True)

    lines = text.splitlines()
    assert "TraceML Run Summary" in lines[1]
    assert "TraceML Watch Summary" in lines[2]
    assert "Verdict: INPUT-BOUND" in text


def test_view_summary_re_render_survives_a_bare_payload(tmp_path) -> None:
    """An unversioned payload keeps its stored card rather than raising."""
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, {"text": _OLD_CARD})

    text = view_summary(summary_path, print_to_stdout=False, re_render=True)

    assert text == _OLD_CARD


def test_view_summary_re_render_falls_back_when_rendering_fails(
    tmp_path, monkeypatch
) -> None:
    """A read-only command must never fail because a rebuild failed."""
    import traceml_ai.reporting.terminal_card.card as summary_card

    def _boom(*_args, **_kwargs):
        raise ValueError("renderer exploded")

    monkeypatch.setattr(summary_card, "build_card_from_payload", _boom)

    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, _payload_with_stored_card(_OLD_CARD))

    text = view_summary(summary_path, print_to_stdout=False, re_render=True)

    assert text == _OLD_CARD
