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
def _reset_optimizer_timing_between_tests():
    """
    Reset global optimizer-hook state after every test (fixes the TRA-28
    order-dependent ``wrap_optimizer`` poisoning).

    ``trace_step`` installs process-global optimizer-step hooks (via
    ``ensure_optimizer_timing_installed``) and the installed-flag is never
    cleared, so a later test calling ``wrap_optimizer()`` is refused purely
    because of test order. Clearing the hooks + flag on teardown makes the
    suite order-independent. Best-effort; never fails a test.
    """
    yield
    try:
        from traceml_ai.instrumentation.hooks.optimizer_hooks import (
            reset_optimizer_timing,
        )

        reset_optimizer_timing()
    except Exception:
        pass
