# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Keep published terminal-summary examples aligned with formatter goldens."""

from pathlib import Path

import pytest

from tests.reporting.summary.test_card_golden import GOLDENS

ROOT = Path(__file__).resolve().parents[2]

CANONICAL_REPORTS = (
    "run_input_bound_critical",
    "run_multi_input_straggler",
)
FULL_REPORT_DOCUMENTS = (ROOT / "README.md",)
QUICKSTART = ROOT / "docs" / "user_guide" / "quickstart.md"
AFFECTED_OUTPUT_DOCS = (
    *FULL_REPORT_DOCUMENTS,
    QUICKSTART,
    ROOT / "docs" / "user_guide" / "reading-output.md",
    ROOT / "docs" / "guides" / "pytorch-input-pipeline-bottleneck.md",
    ROOT / "docs" / "user_guide" / "integrations" / "accelerate.md",
    ROOT / "notebooks" / "huggingface_dataloading_bottleneck.ipynb",
    ROOT / "notebooks" / "lightning_dataloading_bottleneck.ipynb",
)
LEGACY_SUMMARY_MARKERS = (
    "TraceML Verdict:",
    "TraceML Run Summary | duration",
    "Section Status",
    "System Evidence",
    "Step Time Evidence",
    "INPUT-BOUND / CRITICAL",
    "INPUT STRAGGLER / CRITICAL",
    "RESIDUAL-HEAVY / CRITICAL",
)


@pytest.mark.parametrize(
    "document", FULL_REPORT_DOCUMENTS, ids=lambda path: path.name
)
@pytest.mark.parametrize("golden_name", CANONICAL_REPORTS)
def test_full_reports_match_formatter_golden(
    document: Path, golden_name: str
) -> None:
    """Require each published full report to match the formatter exactly."""
    assert GOLDENS[golden_name] in document.read_text(encoding="utf-8")


def test_quickstart_excerpt_uses_current_summary_language() -> None:
    """Keep the short quickstart example aligned without requiring a full card."""
    content = QUICKSTART.read_text(encoding="utf-8")
    expected = (
        "Verdict: INPUT-BOUND (CRITICAL)",
        "Why: Input Wait took 64% of Step Time.",
        "Next: Increase workers, prefetch, or storage throughput.",
        "Step Time       200.4 ms  100%",
        "\u251c\u2500 Input Wait   128.0 ms   64%  \u25c0 cause",
        "\u2514\u2500 Residual        3.6 ms    2%",
    )
    missing = [marker for marker in expected if marker not in content]
    assert not missing, f"quickstart summary markers missing: {missing}"


@pytest.mark.parametrize(
    "document",
    AFFECTED_OUTPUT_DOCS,
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_affected_docs_do_not_publish_legacy_summary_markers(
    document: Path,
) -> None:
    """Reject output labels and severity syntax removed by the current card."""
    content = document.read_text(encoding="utf-8")
    found = [marker for marker in LEGACY_SUMMARY_MARKERS if marker in content]
    assert not found, f"legacy terminal-summary markers remain: {found}"
