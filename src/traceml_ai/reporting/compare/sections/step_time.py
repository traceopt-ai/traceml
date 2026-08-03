# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-time compare section."""

from __future__ import annotations

from typing import Any, Dict, Optional

from traceml_ai.reporting.compare.model import CompareSection
from traceml_ai.reporting.compare.sections.base import (
    as_float,
    global_average,
    global_average_has_key,
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
        lhs_step_clock = self._step_time_clock(lhs_payload, lhs)
        rhs_step_clock = self._step_time_clock(rhs_payload, rhs)
        step_clocks_match = (
            lhs_step_clock is not None
            and rhs_step_clock is not None
            and lhs_step_clock == rhs_step_clock
        )
        lhs_selected_clock = self._selected_clock(lhs)
        rhs_selected_clock = self._selected_clock(rhs)
        selected_clocks_match = (
            lhs_selected_clock is not None
            and rhs_selected_clock is not None
            and lhs_selected_clock == rhs_selected_clock
        )
        notes: tuple[str, ...] = ()
        if not step_clocks_match:
            notes += (
                "Step Time metrics are unavailable because the selected "
                "clocks differ or are missing "
                f"(A: {lhs_step_clock or 'n/a'}, "
                f"B: {rhs_step_clock or 'n/a'}).",
            )
        if not selected_clocks_match:
            notes += (
                "Selected-clock phase metrics are unavailable because "
                f"their clocks differ or are missing "
                f"(A: {lhs_selected_clock or 'n/a'}, "
                f"B: {rhs_selected_clock or 'n/a'}).",
            )

        def selected_value(section: Any, key: str) -> Any:
            return self._value(section, key) if selected_clocks_match else None

        def selected_input(section: Any) -> Any:
            return (
                self._input_value(section) if selected_clocks_match else None
            )

        def selected_dominant_phase(section: Any) -> Any:
            return (
                self._dominant_phase(section)
                if selected_clocks_match
                else None
            )

        return CompareSection(
            name=self.name,
            available=section_available(lhs, rhs),
            diagnosis=section_diagnosis(lhs, rhs),
            metrics={
                "step_time_ms": numeric_metric(
                    key="step_time_ms",
                    label="Step Time",
                    unit="ms",
                    lhs=self._step_time_value(
                        lhs_payload,
                        lhs,
                        clocks_match=step_clocks_match,
                    ),
                    rhs=self._step_time_value(
                        rhs_payload,
                        rhs,
                        clocks_match=step_clocks_match,
                    ),
                    direction="higher_is_worse",
                ),
                "input_ms": numeric_metric(
                    key="input_ms",
                    label="Input",
                    unit="ms",
                    lhs=selected_input(lhs),
                    rhs=selected_input(rhs),
                    direction="higher_is_worse",
                ),
                "h2d_ms": numeric_metric(
                    key="h2d_ms",
                    label="H2D",
                    unit="ms",
                    lhs=selected_value(lhs, "h2d_ms"),
                    rhs=selected_value(rhs, "h2d_ms"),
                    direction="higher_is_worse",
                ),
                "compute_ms": numeric_metric(
                    key="compute_ms",
                    label="Compute",
                    unit="ms",
                    lhs=selected_value(lhs, "compute_ms"),
                    rhs=selected_value(rhs, "compute_ms"),
                    direction="higher_is_worse",
                ),
                "residual_ms": numeric_metric(
                    key="residual_ms",
                    label="Residual",
                    unit="ms",
                    lhs=selected_value(lhs, "residual_ms"),
                    rhs=selected_value(rhs, "residual_ms"),
                    direction="higher_is_worse",
                ),
                "forward_ms": numeric_metric(
                    key="forward_ms",
                    label="Forward",
                    unit="ms",
                    lhs=selected_value(lhs, "forward_ms"),
                    rhs=selected_value(rhs, "forward_ms"),
                    direction="higher_is_worse",
                ),
                "backward_ms": numeric_metric(
                    key="backward_ms",
                    label="Backward",
                    unit="ms",
                    lhs=selected_value(lhs, "backward_ms"),
                    rhs=selected_value(rhs, "backward_ms"),
                    direction="higher_is_worse",
                ),
                "optimizer_ms": numeric_metric(
                    key="optimizer_ms",
                    label="Optimizer",
                    unit="ms",
                    lhs=selected_value(lhs, "optimizer_ms"),
                    rhs=selected_value(rhs, "optimizer_ms"),
                    direction="higher_is_worse",
                ),
                "dominant_phase": text_metric(
                    key="dominant_phase",
                    label="Dominant phase",
                    lhs=selected_dominant_phase(lhs),
                    rhs=selected_dominant_phase(rhs),
                ),
            },
            notes=notes,
        )

    def _value(self, section: Any, key: str) -> Any:
        return global_average(section, key)

    def _input_value(self, section: Any) -> Any:
        """Return selected-clock input wait.

        Falls back to ``dataloader_ms`` only when ``input_wait_ms`` is
        absent entirely (a pre-1.6 payload that never had the key). A
        schema>=1.6 payload always carries the key; a present-but-null
        value there means the signal was genuinely never measured this
        window and must not be silently replaced by a different metric.
        """
        value = self._value(section, "input_wait_ms")
        if value is not None:
            return value
        if global_average_has_key(section, "input_wait_ms"):
            return None
        return self._value(section, "dataloader_ms")

    def _dominant_phase(self, section: Any) -> Any:
        phases = {
            "input": as_float(self._input_value(section)),
            "h2d": as_float(self._value(section, "h2d_ms")),
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

    @staticmethod
    def _schema_version(payload: Dict[str, Any]) -> Optional[float]:
        """Return one numeric input schema version when it is available."""
        try:
            return float(payload.get("schema_version"))
        except (TypeError, ValueError):
            return None

    def _step_time_clock(
        self,
        payload: Dict[str, Any],
        section: Any,
    ) -> Optional[str]:
        """Return the clock represented by this payload's Step Time values."""
        schema = self._schema_version(payload)
        if schema is None or schema < 1.8:
            # Historical outer Step Time was explicitly CPU-clocked.
            return "cpu"
        window = (
            section.get("global", {}).get("window", {})
            if isinstance(section, dict)
            else {}
        )
        clock = str(window.get("diagnosis_clock") or "").lower()
        return clock if clock in {"cpu", "gpu"} else None

    @staticmethod
    def _selected_clock(section: Any) -> Optional[str]:
        """Return the clock behind selected-clock phase values."""
        window = (
            section.get("global", {}).get("window", {})
            if isinstance(section, dict)
            else {}
        )
        clock = str(window.get("diagnosis_clock") or "").lower()
        return clock if clock in {"cpu", "gpu"} else None

    def _step_time_value(
        self,
        payload: Dict[str, Any],
        section: Any,
        *,
        clocks_match: bool,
    ) -> Any:
        """Read outer Step Time through the schema-version compatibility seam."""
        if not clocks_match:
            return None
        schema = self._schema_version(payload)
        key = (
            "step_time_ms"
            if schema is not None and schema >= 1.8
            else "total_step_ms"
        )
        return global_average(section, key)


__all__ = ["StepTimeComparer"]
