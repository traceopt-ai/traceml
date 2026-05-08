"""Step-memory compare section."""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.compare.model import CompareSection
from traceml.reporting.compare.sections.base import (
    nested_get,
    numeric_metric,
    section_available,
    section_diagnosis,
)


class StepMemoryComparer:
    name = "step_memory"

    def compare(
        self,
        lhs_payload: Dict[str, Any],
        rhs_payload: Dict[str, Any],
    ) -> CompareSection:
        lhs = lhs_payload.get(self.name)
        rhs = rhs_payload.get(self.name)
        return CompareSection(
            name=self.name,
            available=section_available(lhs, rhs),
            diagnosis=section_diagnosis(lhs, rhs),
            metrics={
                "peak_reserved_bytes": numeric_metric(
                    key="peak_reserved_bytes",
                    label="Peak reserved",
                    unit="bytes",
                    lhs=self._peak_reserved(lhs),
                    rhs=self._peak_reserved(rhs),
                    direction="higher_is_worse",
                ),
                "memory_skew_pct": numeric_metric(
                    key="memory_skew_pct",
                    label="Memory skew",
                    unit="percent",
                    lhs=self._primary_value(lhs, "skew_pct"),
                    rhs=self._primary_value(rhs, "skew_pct"),
                    delta_unit="percentage_point",
                    direction="higher_is_worse",
                ),
                "trend_worst_delta_bytes": numeric_metric(
                    key="trend_worst_delta_bytes",
                    label="Memory trend",
                    unit="bytes",
                    lhs=nested_get(
                        self._primary(lhs), "trend", "worst", "delta_bytes"
                    ),
                    rhs=nested_get(
                        self._primary(rhs), "trend", "worst", "delta_bytes"
                    ),
                    direction="higher_is_worse",
                ),
            },
        )

    def _primary(self, section: Any) -> Dict[str, Any]:
        primary = nested_get(section, "global", "primary_metric")
        if isinstance(primary, dict):
            return primary
        primary = nested_get(section, "primary_metric")
        return primary if isinstance(primary, dict) else {}

    def _primary_value(self, section: Any, key: str) -> Any:
        return self._primary(section).get(key)

    def _peak_reserved(self, section: Any) -> Any:
        primary = self._primary(section)
        return primary.get("worst_peak_bytes")


__all__ = ["StepMemoryComparer"]
