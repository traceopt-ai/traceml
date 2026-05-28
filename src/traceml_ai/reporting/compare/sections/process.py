# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Process compare section."""

from __future__ import annotations

from typing import Any, Dict

from traceml_ai.reporting.compare.model import CompareSection
from traceml_ai.reporting.compare.sections.base import (
    global_average,
    numeric_metric,
    section_available,
    section_diagnosis,
)

BYTES_PER_GB = 1024.0**3


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
                "rss_avg_gb": numeric_metric(
                    key="rss_avg_gb",
                    label="Process RSS avg",
                    unit="gb",
                    lhs=self._bytes_to_gb(self._value(lhs, "ram_bytes")),
                    rhs=self._bytes_to_gb(self._value(rhs, "ram_bytes")),
                    direction="context",
                ),
            },
        )

    def _value(self, section: Any, key: str) -> Any:
        if key == "cpu_avg_percent":
            key = "cpu_percent"
        return global_average(section, key)

    def _bytes_to_gb(self, value: Any) -> Any:
        if value is None:
            return None
        return float(value) / BYTES_PER_GB


__all__ = ["ProcessComparer"]
