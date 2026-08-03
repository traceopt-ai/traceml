from html.parser import HTMLParser

from traceml_ai.reporting.html import render_html_report


def _parses(text: str) -> bool:
    HTMLParser().feed(text)
    return True


def test_render_returns_parseable_html_document(make_payload) -> None:
    out = render_html_report(make_payload())
    assert out.startswith("<!DOCTYPE html>")
    assert "<title>" in out
    assert _parses(out)


def test_header_shows_run_name(make_payload) -> None:
    out = render_html_report(make_payload())
    assert "demo-run" in out


def test_run_name_is_html_escaped(make_payload) -> None:
    payload = make_payload(
        meta={"run_name": "<script>alert(1)</script>", "mode": "single_node"}
    )
    out = render_html_report(payload)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out


def test_render_is_deterministic(make_payload) -> None:
    payload = make_payload()
    assert render_html_report(payload, source_label="x.json") == (
        render_html_report(payload, source_label="x.json")
    )


def test_missing_meta_does_not_crash_and_falls_back(make_payload) -> None:
    out = render_html_report(make_payload(meta=None))
    assert _parses(out)
    assert "run report" in out


def test_output_has_no_script_tags(make_payload) -> None:
    out = render_html_report(make_payload())
    assert "<script" not in out.lower()


def test_output_has_no_external_urls(make_payload) -> None:
    out = render_html_report(make_payload())
    assert "http://" not in out
    assert "https://" not in out


def test_footer_shows_schema_version_from_payload(make_payload) -> None:
    out = render_html_report(make_payload(schema_version=1.3))
    assert "1.3" in out
