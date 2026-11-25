"""
No-op live display driver for TraceML.

This driver is intentionally a no-op for live rendering. It allows TraceML to
run without a live terminal or browser UI. The aggregator independently selects
history persistence and report generation.

Use cases
---------
- batch jobs
- CI runs
- remote sessions where only the final summary matters
- low-noise runs where users only want end-of-run artifacts

Notes
-----
Final summary generation is still owned by the aggregator shutdown path, not by
this driver. This keeps display concerns separate from summary persistence and
run finalization.
"""

from __future__ import annotations

from typing import Any

from traceml_ai.aggregator.display_drivers.base import BaseDisplayDriver
from traceml_ai.runtime.settings import TraceMLSettings


class SummaryDisplayDriver(BaseDisplayDriver):
    """
    No-op display driver used for summary and telemetry-only runs.

    This driver deliberately does not allocate terminal UI or dashboard
    resources. History and summary generation are selected independently by
    the aggregator; telemetry mode enables neither.
    """

    def __init__(
        self,
        logger: Any,
        settings: TraceMLSettings,
    ) -> None:
        super().__init__(logger=logger, settings=settings)

    def start(self) -> None:
        """
        Initialize the summary-only display driver.

        No live UI resources are created in summary mode.
        """
        return None

    def tick(self) -> None:
        """
        Perform one display update cycle.

        Summary mode has no live surface, so ticks are intentionally no-op.
        """
        return None

    def stop(self) -> None:
        """
        Release any display resources.

        Summary mode does not own live resources, so shutdown is a no-op.
        """
        return None
