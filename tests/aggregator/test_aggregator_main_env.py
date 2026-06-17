"""Tests for aggregator_main.read_traceml_env dashboard config (TRA-68)."""

from __future__ import annotations

from traceml_ai.aggregator.aggregator_main import read_traceml_env


def test_read_traceml_env_dashboard_defaults(monkeypatch) -> None:
    monkeypatch.delenv("TRACEML_DASHBOARD_PORT", raising=False)
    monkeypatch.delenv("TRACEML_DASHBOARD_AUTO_OPEN", raising=False)
    cfg = read_traceml_env()
    assert cfg["dashboard_port"] == 8765
    assert cfg["dashboard_auto_open"] is True


def test_read_traceml_env_dashboard_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TRACEML_DASHBOARD_PORT", "9100")
    monkeypatch.setenv("TRACEML_DASHBOARD_AUTO_OPEN", "0")
    cfg = read_traceml_env()
    assert cfg["dashboard_port"] == 9100
    assert cfg["dashboard_auto_open"] is False
