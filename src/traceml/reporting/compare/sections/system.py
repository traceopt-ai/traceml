"""System compare section."""

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


class SystemComparer:
    name = "system"

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
                    label="System CPU avg",
                    unit="percent",
                    lhs=self._value(lhs, "cpu_avg_percent"),
                    rhs=self._value(rhs, "cpu_avg_percent"),
                    delta_unit="percentage_point",
                    direction="context",
                ),
                "ram_peak_gb": numeric_metric(
                    key="ram_peak_gb",
                    label="System RAM peak",
                    unit="gb",
                    lhs=self._value(lhs, "ram_peak_gb"),
                    rhs=self._value(rhs, "ram_peak_gb"),
                    direction="context",
                ),
                "gpu_util_avg_percent": numeric_metric(
                    key="gpu_util_avg_percent",
                    label="GPU util avg",
                    unit="percent",
                    lhs=self._value(lhs, "gpu_util_avg_percent"),
                    rhs=self._value(rhs, "gpu_util_avg_percent"),
                    delta_unit="percentage_point",
                    direction="context",
                ),
                "gpu_memory_peak_percent": numeric_metric(
                    key="gpu_memory_peak_percent",
                    label="GPU memory peak",
                    unit="percent",
                    lhs=self._value(lhs, "gpu_memory_peak_percent"),
                    rhs=self._value(rhs, "gpu_memory_peak_percent"),
                    delta_unit="percentage_point",
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
        if key == "ram_peak_gb":
            return first_present(
                nested_get(section, "global", "ram", "peak_gb"),
                nested_get(section, "ram_peak_gb"),
            )
        if key == "gpu_util_avg_percent":
            return first_present(
                nested_get(
                    section, "global", "gpu_rollup", "util_avg_percent"
                ),
                nested_get(section, "gpu_util_avg_percent"),
            )
        if key == "gpu_memory_peak_percent":
            return first_present(
                nested_get(
                    section,
                    "global",
                    "gpu_rollup",
                    "memory_peak_percent",
                ),
                nested_get(section, "gpu_memory_peak_percent"),
            )
        return None


__all__ = ["SystemComparer"]
