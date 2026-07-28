"""
Shared pytest test bootstrap.

This keeps the repository runnable with plain `pytest` from the repo root by
ensuring the local `src/` tree is importable without requiring an editable
install first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _contain_traceml_disabled_env():
    """Keep TRACEML_DISABLED from leaking out of the test that set it.

    `init()` writes this variable straight to os.environ on its fail-open path
    (`sdk/initial.py::_noop_config`), so one unreachable-aggregator call would
    otherwise turn every later test, and every subprocess they spawn, into a
    disabled no-op run.
    """
    original = os.environ.get("TRACEML_DISABLED")
    yield
    if original is None:
        os.environ.pop("TRACEML_DISABLED", None)
    else:
        os.environ["TRACEML_DISABLED"] = original
