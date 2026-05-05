"""Process compare section."""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.compare.model import CompareSection
from traceml.reporting.compare.sections.base import (
    first_present,
    nested_get,
    numeric_metric,
    section_available,
    section_diagnosis,
)


class ProcessComparer:
    name = "process"

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
                "cpu_avg_percent": numeric_metric(
                    key="cpu_avg_percent",
                    label="Process CPU avg",
                    unit="percent",
                    lhs=self._value(lhs, "cpu_avg_percent"),
                    rhs=self._value(rhs, "cpu_avg_percent"),
                    delta_unit="percentage_point",
                    direction="context",
                ),
                "rss_peak_gb": numeric_metric(
                    key="rss_peak_gb",
                    label="Process RSS peak",
                    unit="gb",
                    lhs=self._value(lhs, "rss_peak_gb"),
                    rhs=self._value(rhs, "rss_peak_gb"),
                    direction="context",
                ),
            },
        )

    def _value(self, section: Any, key: str) -> Any:
        if key == "cpu_avg_percent":
            return first_present(
                nested_get(section, "global", "cpu", "avg_percent"),
                nested_get(section, "cpu_avg_percent"),
            )
        if key == "rss_peak_gb":
            return first_present(
                nested_get(section, "global", "ram", "peak_gb"),
                nested_get(section, "ram_peak_gb"),
            )
        return None


__all__ = ["ProcessComparer"]
