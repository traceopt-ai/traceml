import json

import pytest

from traceml_ai.reporting.html import (
    render_html_report,
    render_html_report_from_file,
    write_html_report,
)


def test_raw_json_block_is_present_and_escaped(make_payload) -> None:
    payload = make_payload(
        meta={"run_name": "</pre><script>x</script>", "mode": "single_node"}
    )
    out = render_html_report(payload)
    assert "Raw final_summary.json" in out
    # The embedded JSON must not break out of the <pre> or inject a script.
    assert "<script>x</script>" not in out


def test_write_html_report_writes_file_and_returns_path(
    make_payload, tmp_path
) -> None:
    out_path = tmp_path / "report.html"
    result = write_html_report(make_payload(), out_path)
    assert result == out_path
    assert out_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_from_file_defaults_to_sibling_html(make_payload, tmp_path) -> None:
    json_path = tmp_path / "final_summary.json"
    json_path.write_text(json.dumps(make_payload()), encoding="utf-8")

    result = render_html_report_from_file(json_path)

    assert result == tmp_path / "final_summary.html"
    assert result.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_from_file_honors_explicit_out_path(make_payload, tmp_path) -> None:
    json_path = tmp_path / "final_summary.json"
    json_path.write_text(json.dumps(make_payload()), encoding="utf-8")
    out_path = tmp_path / "custom.html"

    result = render_html_report_from_file(json_path, out_path)

    assert result == out_path
    assert out_path.exists()


def test_from_file_footer_carries_source_label(make_payload, tmp_path) -> None:
    json_path = tmp_path / "final_summary.json"
    json_path.write_text(json.dumps(make_payload()), encoding="utf-8")

    render_html_report_from_file(json_path)
    html = (tmp_path / "final_summary.html").read_text(encoding="utf-8")

    assert "final_summary.json" in html  # source label in footer


def test_from_file_rejects_malformed_json(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        render_html_report_from_file(bad)
