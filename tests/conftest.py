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
