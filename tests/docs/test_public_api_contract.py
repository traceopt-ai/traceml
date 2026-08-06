"""Contract tests for the documented ``traceml_ai`` public surface."""

from __future__ import annotations

import re
from pathlib import Path

import traceml_ai


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_API_DOC = ROOT / "docs" / "user_guide" / "public-api.md"


def test_public_api_page_documents_every_export():
    """Keep public exports and their mkdocstrings reference in sync."""
    page = PUBLIC_API_DOC.read_text(encoding="utf-8")
    documented_callables = set(
        re.findall(r"^:::\s+traceml_ai\.api\.([A-Za-z_]\w*)\s*$", page, re.M)
    )
    expected_callables = set(traceml_ai.__all__) - {"__version__"}

    assert documented_callables == expected_callables
    assert "### `traceml.__version__`" in page
