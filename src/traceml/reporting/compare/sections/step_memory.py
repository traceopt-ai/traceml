# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-memory compare section."""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.compare.model import CompareSection
from traceml.reporting.compare.sections.base import (
    as_float,
    global_point_value,
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
                    lhs=self._skew_pct(lhs),
                    rhs=self._skew_pct(rhs),
                    delta_unit="percentage_point",
                    direction="higher_is_worse",
                ),
                "trend_worst_delta_bytes": numeric_metric(
                    key="trend_worst_delta_bytes",
                    label="Memory trend",
                    unit="bytes",
                    lhs=None,
                    rhs=None,
                    direction="higher_is_worse",
                ),
            },
        )

    def _peak_reserved(self, section: Any) -> Any:
        return global_point_value(section, "worst", "peak_reserved_bytes")

    def _skew_pct(self, section: Any) -> Any:
        median = as_float(
            global_point_value(section, "median", "peak_reserved_bytes")
        )
        worst = as_float(
            global_point_value(section, "worst", "peak_reserved_bytes")
        )
        if median is None or worst is None or abs(median) <= 1e-12:
            return None
        return 100.0 * (worst - median) / median


__all__ = ["StepMemoryComparer"]
