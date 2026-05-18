# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-time compare section."""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.compare.model import CompareSection
from traceml.reporting.compare.sections.base import (
    as_float,
    global_average,
    numeric_metric,
    section_available,
    section_diagnosis,
    text_metric,
)


class StepTimeComparer:
    name = "step_time"

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
                "total_step_ms": numeric_metric(
                    key="total_step_ms",
                    label="Total step",
                    unit="ms",
                    lhs=self._value(lhs, "total_step_ms"),
                    rhs=self._value(rhs, "total_step_ms"),
                    direction="higher_is_worse",
                ),
                "model_step_ms": numeric_metric(
                    key="model_step_ms",
                    label="Model step",
                    unit="ms",
                    lhs=self._value(lhs, "model_step_ms"),
                    rhs=self._value(rhs, "model_step_ms"),
                    direction="higher_is_worse",
                ),
                "compute_ms": numeric_metric(
                    key="compute_ms",
                    label="Compute",
                    unit="ms",
                    lhs=self._value(lhs, "compute_ms"),
                    rhs=self._value(rhs, "compute_ms"),
                    direction="higher_is_worse",
                ),
                "wait_ms": numeric_metric(
                    key="wait_ms",
                    label="Wait",
                    unit="ms",
                    lhs=self._value(lhs, "wait_ms"),
                    rhs=self._value(rhs, "wait_ms"),
                    direction="higher_is_worse",
                ),
                "input_ms": numeric_metric(
                    key="input_ms",
                    label="Input",
                    unit="ms",
                    lhs=self._value(lhs, "dataloader_ms"),
                    rhs=self._value(rhs, "dataloader_ms"),
                    direction="higher_is_worse",
                ),
                "wait_share_pct": numeric_metric(
                    key="wait_share_pct",
                    label="Wait share",
                    unit="percent",
                    lhs=self._wait_share_pct(lhs),
                    rhs=self._wait_share_pct(rhs),
                    delta_unit="percentage_point",
                    direction="higher_is_worse",
                ),
                "dominant_phase": text_metric(
                    key="dominant_phase",
                    label="Dominant phase",
                    lhs=self._dominant_phase(lhs),
                    rhs=self._dominant_phase(rhs),
                ),
            },
        )

    def _value(self, section: Any, key: str) -> Any:
        return global_average(section, key)

    def _wait_share_pct(self, section: Any) -> Any:
        explicit = as_float(self._value(section, "wait_share_pct"))
        if explicit is not None:
            return explicit

        model_step_ms = as_float(self._value(section, "model_step_ms"))
        wait_ms = as_float(self._value(section, "wait_ms"))
        if (
            model_step_ms is None
            or wait_ms is None
            or abs(model_step_ms) <= 1e-12
        ):
            return None
        return 100.0 * wait_ms / model_step_ms

    def _dominant_phase(self, section: Any) -> Any:
        phases = {
            "dataloader": as_float(self._value(section, "dataloader_ms")),
            "forward": as_float(self._value(section, "forward_ms")),
            "backward": as_float(self._value(section, "backward_ms")),
            "optimizer": as_float(self._value(section, "optimizer_ms")),
        }
        present = {
            phase: value
            for phase, value in phases.items()
            if value is not None
        }
        if not present:
            return None
        return max(present, key=lambda phase: present[phase])


__all__ = ["StepTimeComparer"]
