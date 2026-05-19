# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""System compare section."""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.compare.model import CompareSection
from traceml.reporting.compare.sections.base import (
    global_average,
    numeric_metric,
    section_available,
    section_diagnosis,
)

BYTES_PER_GB = 1024.0**3


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
                "ram_avg_gb": numeric_metric(
                    key="ram_avg_gb",
                    label="System RAM avg",
                    unit="gb",
                    lhs=self._bytes_to_gb(self._value(lhs, "ram_bytes")),
                    rhs=self._bytes_to_gb(self._value(rhs, "ram_bytes")),
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
                "gpu_memory_avg_percent": numeric_metric(
                    key="gpu_memory_avg_percent",
                    label="GPU memory avg",
                    unit="percent",
                    lhs=self._value(lhs, "gpu_mem_percent"),
                    rhs=self._value(rhs, "gpu_mem_percent"),
                    delta_unit="percentage_point",
                    direction="context",
                ),
            },
        )

    def _value(self, section: Any, key: str) -> Any:
        if key == "cpu_avg_percent":
            key = "cpu_percent"
        if key == "gpu_util_avg_percent":
            key = "gpu_util_percent"
        return global_average(section, key)

    def _bytes_to_gb(self, value: Any) -> Any:
        if value is None:
            return None
        return float(value) / BYTES_PER_GB


__all__ = ["SystemComparer"]
