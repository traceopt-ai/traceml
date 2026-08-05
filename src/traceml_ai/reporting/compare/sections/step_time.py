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
        comparison_clock, lhs_step_time, rhs_step_time = (
            self._common_step_time_values(
                lhs_payload,
                lhs,
                rhs_payload,
                rhs,
            )
        )
        lhs_selected_clock = self._selected_clock(lhs)
        rhs_selected_clock = self._selected_clock(rhs)
        selected_clocks_match = (
            lhs_selected_clock is not None
            and rhs_selected_clock is not None
            and lhs_selected_clock == rhs_selected_clock
        )
        notes: tuple[str, ...] = ()
        if comparison_clock is None:
            notes += (
                "Step Time metrics are unavailable because the summaries "
                "have no common measured GPU or CPU clock.",
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
                    label=self._step_time_label(comparison_clock),
                    unit="ms",
                    lhs=lhs_step_time,
                    rhs=rhs_step_time,
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
                "dataloader_fetch_cpu_ms": numeric_metric(
                    key="dataloader_fetch_cpu_ms",
                    label="DataLoader Fetch (CPU)",
                    unit="ms",
                    lhs=self._dataloader_fetch_cpu_value(lhs_payload, lhs),
                    rhs=self._dataloader_fetch_cpu_value(rhs_payload, rhs),
                    direction="context",
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
            comparison_clock=comparison_clock,
        )

    def _value(self, section: Any, key: str) -> Any:
        return global_average(section, key)

    def _input_value(self, section: Any) -> Any:
        """Return selected-clock Input Wait without legacy substitution.

        Historical ``dataloader_ms`` is CPU supplemental DataLoader-fetch
        evidence, not selected-clock Input Wait. Compatibility for that field
        is isolated in :meth:`_dataloader_fetch_cpu_value`.
        """
        return self._value(section, "input_wait_ms")

    def _dataloader_fetch_cpu_value(
        self,
        payload: Dict[str, Any],
        section: Any,
    ) -> Any:
        """Read supplemental CPU fetch timing through the versioned seam.

        Pre-1.7 summaries called this CPU-only field ``dataloader_ms``.
        Schema 1.7 publishes the canonical ``dataloader_fetch_cpu_ms`` name.
        This value is comparison context only; it never stands in for Input
        Wait or contributes to a Step Time verdict.
        """
        schema = self._schema_version(payload)
        key = (
            "dataloader_fetch_cpu_ms"
            if schema is not None and schema >= 1.7
            else "dataloader_ms"
        )
        return global_average(section, key)

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

    def _clock_step_time_values(
        self,
        payload: Dict[str, Any],
        section: Any,
    ) -> Dict[str, Any]:
        """Read canonical clocks through the versioned summary seam.

        Schema 1.7 publishes explicit CPU and GPU outer Step Time aggregates.
        Historical summaries only expose ``total_step_ms``; it is compatibility
        evidence for CPU Step Time and is never interpreted as GPU.
        """
        schema = self._schema_version(payload)
        if schema is None or schema < 1.7:
            return {
                "cpu": global_average(section, "total_step_ms"),
                "gpu": None,
            }
        return {
            "cpu": global_average(section, "step_time_cpu_ms"),
            "gpu": global_average(section, "step_time_gpu_ms"),
        }

    def _common_step_time_values(
        self,
        lhs_payload: Dict[str, Any],
        lhs: Any,
        rhs_payload: Dict[str, Any],
        rhs: Any,
    ) -> tuple[Optional[str], Any, Any]:
        """Select one measured common clock without reconstructing Step Time.

        GPU is preferred when both summaries publish it. CPU is the fallback
        when both summaries publish CPU evidence. A numeric zero is measured;
        only absent or null values make a clock unavailable.
        """
        lhs_values = self._clock_step_time_values(lhs_payload, lhs)
        rhs_values = self._clock_step_time_values(rhs_payload, rhs)
        for clock in ("gpu", "cpu"):
            lhs_value = lhs_values[clock]
            rhs_value = rhs_values[clock]
            if (
                as_float(lhs_value) is not None
                and as_float(rhs_value) is not None
            ):
                return clock, lhs_value, rhs_value
        return None, None, None

    @staticmethod
    def _step_time_label(comparison_clock: Optional[str]) -> str:
        """Return the rendered outer Step Time label for the common clock."""
        if comparison_clock in {"cpu", "gpu"}:
            return f"{comparison_clock.upper()} Step Time"
        return "Step Time"

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


__all__ = ["StepTimeComparer"]
