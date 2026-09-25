"""Tests for aggregator_main.read_traceml_env configuration."""

from __future__ import annotations

from traceml_ai.aggregator.aggregator_main import read_traceml_env


def test_read_traceml_env_dashboard_defaults(monkeypatch) -> None:
    monkeypatch.delenv("TRACEML_UI_MODE", raising=False)
    monkeypatch.delenv("TRACEML_MODE", raising=False)
    monkeypatch.delenv("TRACEML_DASHBOARD_PORT", raising=False)
    monkeypatch.delenv("TRACEML_DASHBOARD_AUTO_OPEN", raising=False)
    monkeypatch.delenv("TRACEML_INTERVAL", raising=False)
    cfg = read_traceml_env()
    assert cfg["mode"] == "summary"
    assert cfg["dashboard_port"] == 8765
    assert cfg["dashboard_auto_open"] is True
    assert cfg["interval"] == 2.0


def test_read_traceml_env_dashboard_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TRACEML_DASHBOARD_PORT", "9100")
    monkeypatch.setenv("TRACEML_DASHBOARD_AUTO_OPEN", "0")
    cfg = read_traceml_env()
    assert cfg["dashboard_port"] == 9100
    assert cfg["dashboard_auto_open"] is False


def test_launcher_aggregator_enforces_session_and_run_nonce(
    monkeypatch,
) -> None:
    """The `traceml run` aggregator shares its run id with its own ranks."""
    from traceml_ai.aggregator import aggregator_main

    monkeypatch.setenv("TRACEML_SESSION_ID", "run-b")
    monkeypatch.setenv("TRACEML_RUN_NONCE", "abc123")
    captured = {}

    def _run(settings, *, logger=None):
        captured["settings"] = settings
        return 0

    monkeypatch.setattr(
        aggregator_main, "setup_error_logger", lambda **_: None
    )
    monkeypatch.setattr(aggregator_main, "run_aggregator", _run)
    try:
        aggregator_main.main()
    except SystemExit as exc:
        assert exc.code == 0

    settings = captured["settings"]
    assert settings.session_id == "run-b"
    assert settings.enforce_session_id is True
    assert settings.admit_generated_session_id is True
    assert settings.run_nonce == "abc123"


def test_read_traceml_env_run_nonce_defaults_to_empty(monkeypatch) -> None:
    monkeypatch.delenv("TRACEML_RUN_NONCE", raising=False)
    assert read_traceml_env()["run_nonce"] == ""
