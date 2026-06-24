"""
Shared pytest test bootstrap.

This keeps the repository runnable with plain `pytest` from the repo root by
ensuring the local `src/` tree is importable without requiring an editable
install first.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_traceml_global_state_between_tests():
    """
    Reset process-global TraceML state after every test.

    1. Optimizer-hook state: ``trace_step`` installs process-global
       optimizer-step hooks (via ``ensure_optimizer_timing_installed``) and
       the installed-flag is never cleared, so a later test calling
       ``wrap_optimizer()`` is refused purely because of test order.
    2. Recording state (#143): a test that leaks
       ``configure_trace_recording(max_steps=N)`` after flushing past step N
       silences ALL telemetry process-wide, which would make unrelated
       emission tests (e.g. the StreamContract conformance gate) report
       streams dark. Upstream tests reset this only inline, which is not
       exception-safe.

    Both resets are best-effort and never fail a test.
    """
    yield
    try:
        from traceml_ai.instrumentation.hooks.optimizer_hooks import (
            reset_optimizer_timing,
        )

        reset_optimizer_timing()
    except Exception:
        pass
    try:
        from traceml_ai.runtime.state import configure_trace_recording

        configure_trace_recording()
    except Exception:
        pass
