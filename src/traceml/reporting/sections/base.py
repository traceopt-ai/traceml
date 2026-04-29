"""
Final-report section contracts.

The canonical protocol definitions live in ``traceml.core.summaries`` so they
can be shared by reporting, tests, and future extension points without pulling
in SQLite, torch, or renderer dependencies. This module provides the reporting
package import path contributors should use when adding sections.
"""

from traceml.core.summaries import SummaryResult, SummarySection

__all__ = [
    "SummaryResult",
    "SummarySection",
]
