"""Isolation for framework integration tests that call ``traceml.init``."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_traceml_init(monkeypatch: pytest.MonkeyPatch):
    """Keep process-global init and patch arming local to each integration."""
    import traceml_ai.sdk.initial as initialization
    from traceml_ai.runtime.arming import _set_tracing_armed

    initialization._INIT_CONFIG = None
    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    monkeypatch.setattr(
        initialization,
        "_start_runtime_for_init",
        lambda **kwargs: None,
    )
    _set_tracing_armed(False)
    yield
    initialization._INIT_CONFIG = None
    _set_tracing_armed(False)
