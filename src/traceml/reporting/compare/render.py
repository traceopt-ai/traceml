"""
Compatibility entry points for TraceML compare rendering.

Formatter implementations live in :mod:`traceml.reporting.compare.formatters`.
This module keeps the historical ``build_compare_text`` function as the stable
call site for the CLI and external users.
"""

from __future__ import annotations

from typing import Any, Dict

from traceml.loggers.error_log import get_error_logger

from .formatters import CompareTextFormatter

_LOGGER = get_error_logger("CompareRender")


def _fallback_compare_text(payload: Dict[str, Any]) -> str:
    """
    Return a minimal compare report when rich text formatting fails.

    Compare rendering should never prevent artifact generation. The structured
    JSON payload remains the source of truth; this fallback gives users enough
    context to find the formatter error log and continue their workflow.
    """
    lhs = payload.get("lhs", {})
    rhs = payload.get("rhs", {})
    verdict = payload.get("verdict", {})
    lhs_label = lhs.get("label", "lhs") if isinstance(lhs, dict) else "lhs"
    rhs_label = rhs.get("label", "rhs") if isinstance(rhs, dict) else "rhs"
    summary = (
        verdict.get("summary", "Compare text unavailable")
        if isinstance(verdict, dict)
        else "Compare text unavailable"
    )

    return "\n".join(
        [
            "TraceML Compare",
            f"- A: {lhs_label}",
            f"- B: {rhs_label}",
            f"- Result: {summary}",
            "- Note: detailed compare text formatting failed; see TraceML error logs.",
        ]
    )


def build_compare_text(payload: Dict[str, Any]) -> str:
    """
    Render a compact terminal-friendly compare report.

    This wrapper is intentionally fail-open: malformed text-only formatting
    should not break compare JSON generation or user training workflows. Any
    formatter failure is logged through TraceML's error logger and a minimal
    fallback report is returned.
    """
    try:
        return CompareTextFormatter().format(payload)
    except Exception as exc:
        _LOGGER.exception("[TraceML] Compare text formatting failed: %s", exc)
        return _fallback_compare_text(payload)


__all__ = [
    "build_compare_text",
]
