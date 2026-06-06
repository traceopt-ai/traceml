"""
Telemetry contracts for metric computation and persistence projections.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Generic, Mapping, Optional, Protocol, TypeVar

from traceml_ai.telemetry.envelope import TelemetryEnvelope

MetricResultT = TypeVar("MetricResultT")
TelemetryPayload = Mapping[str, Any]


class MetricComputer(Protocol, Generic[MetricResultT]):
    """Compute a telemetry-derived metric or view model."""

    def compute(self) -> MetricResultT:
        """Compute and return the current metric result."""


class ProjectionWriter(Protocol):
    """
    SQLite projection writer contract.

    Projection writers create query-friendly tables for selected sampler
    payloads while the raw telemetry stream remains available elsewhere.
    """

    def accepts_sampler(self, sampler: Optional[str]) -> bool:
        """Return True when this writer handles the sampler name."""

    def init_schema(self, conn: sqlite3.Connection) -> None:
        """Create any projection tables and indexes."""

    def build_rows(
        self,
        envelope: TelemetryEnvelope,
        recv_ts_ns: int,
    ) -> Dict[str, list[tuple]]:
        """Build table-keyed SQLite rows from one canonical envelope."""

    def insert_rows(
        self,
        conn: sqlite3.Connection,
        rows_by_table: Dict[str, list[tuple]],
    ) -> None:
        """Insert prepared projection rows into SQLite."""


__all__ = [
    "MetricComputer",
    "MetricResultT",
    "ProjectionWriter",
    "TelemetryPayload",
]
