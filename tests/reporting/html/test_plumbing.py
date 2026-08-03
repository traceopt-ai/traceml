from traceml_ai.aggregator.aggregator_main import read_traceml_env
from traceml_ai.runtime.settings import TraceMLSettings


def test_settings_html_report_defaults_false() -> None:
    assert TraceMLSettings().html_report is False


def test_read_env_html_report_true(monkeypatch) -> None:
    monkeypatch.setenv("TRACEML_HTML_REPORT", "1")
    assert read_traceml_env()["html_report"] is True


def test_read_env_html_report_defaults_false(monkeypatch) -> None:
    monkeypatch.delenv("TRACEML_HTML_REPORT", raising=False)
    assert read_traceml_env()["html_report"] is False
