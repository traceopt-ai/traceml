"""Step-time compare section."""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.compare.model import CompareSection
from traceml.reporting.compare.sections.base import (
    first_present,
    nested_get,
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
                "step_avg_ms": numeric_metric(
                    key="step_avg_ms",
                    label="Step avg",
                    unit="ms",
                    lhs=self._value(lhs, "step_avg_ms"),
                    rhs=self._value(rhs, "step_avg_ms"),
                    direction="higher_is_worse",
                ),
                "compute_ms": numeric_metric(
                    key="compute_ms",
                    label="Compute",
                    unit="ms",
                    lhs=self._compute_ms(lhs),
                    rhs=self._compute_ms(rhs),
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
                    lhs=self._split_value(lhs, "split_ms", "dataloader"),
                    rhs=self._split_value(rhs, "split_ms", "dataloader"),
                    direction="higher_is_worse",
                ),
                "wait_share_pct": numeric_metric(
                    key="wait_share_pct",
                    label="Wait share",
                    unit="percent",
                    lhs=self._value(lhs, "wait_share_pct"),
                    rhs=self._value(rhs, "wait_share_pct"),
                    delta_unit="percentage_point",
                    direction="higher_is_worse",
                ),
                "dominant_phase": text_metric(
                    key="dominant_phase",
                    label="Dominant phase",
                    lhs=self._value(lhs, "dominant_phase"),
                    rhs=self._value(rhs, "dominant_phase"),
                ),
            },
        )

    def _primary(self, section: Any) -> Dict[str, Any]:
        primary = nested_get(section, "global", "typical")
        if isinstance(primary, dict):
            return primary
        primary = nested_get(section, "timing_primary")
        return primary if isinstance(primary, dict) else {}

    def _value(self, section: Any, key: str) -> Any:
        primary = self._primary(section)
        if key == "wait_ms":
            explicit = first_present(
                primary.get("wait_ms"),
                nested_get(primary, "split_ms", "wait"),
            )
            if explicit is not None:
                return explicit
            step_avg = primary.get("step_avg_ms")
            wait_share = primary.get("wait_share_pct")
            if step_avg is None or wait_share is None:
                return None
            return float(step_avg) * float(wait_share) / 100.0
        return primary.get(key)

    def _split_value(self, section: Any, split_key: str, phase: str) -> Any:
        primary = self._primary(section)
        split = primary.get(split_key)
        if not isinstance(split, dict):
            split = nested_get(section, f"median_{split_key}")
        if not isinstance(split, dict):
            return None
        return split.get(phase)

    def _compute_ms(self, section: Any) -> Any:
        primary = self._primary(section)
        value = primary.get("compute_ms")
        if value is not None:
            return value
        split = primary.get("split_ms")
        if not isinstance(split, dict):
            split = nested_get(section, "median_split_ms")
        if not isinstance(split, dict):
            return None
        values = [
            split.get("forward"),
            split.get("backward"),
            split.get("optimizer"),
        ]
        if any(value is None for value in values):
            return None
        return sum(float(value) for value in values)


__all__ = ["StepTimeComparer"]
