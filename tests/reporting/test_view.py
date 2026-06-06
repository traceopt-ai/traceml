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
