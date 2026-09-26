# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Declared workload and measurement contract for the guard pilot.

This module validates only facts supplied by the user. Observed runtime facts,
measurement eligibility, and reference/candidate decisions belong to later
stages of the regression pipeline.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

Scalar: TypeAlias = str | int | float | bool

SCHEMA_VERSION = 1
MAX_WORKLOAD_NAME_LENGTH = 128
MAX_PARAMETER_COUNT = 32
MAX_PARAMETER_KEY_LENGTH = 64
MAX_PARAMETER_STRING_LENGTH = 256
MAX_SIGNED_INT = (1 << 63) - 1
MIN_SIGNED_INT = -(1 << 63)

_GUARD_KEYS = frozenset({"schema_version", "workload", "measurement"})
_WORKLOAD_KEYS = frozenset({"name", "parameters"})
_MEASUREMENT_KEYS = frozenset({"start_step", "completed_steps"})


class ContractValidationError(ValueError):
    """Raised when the declared guard contract is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class MeasurementContract:
    """Immutable normalized declaration captured for one TraceML run."""

    schema_version: int
    workload_name: str
    workload_parameters: tuple[tuple[str, Scalar], ...]
    start_step: int
    completed_steps: int

    @property
    def end_step(self) -> int:
        """Return the inclusive final requested TraceML step ID."""
        return self.start_step + self.completed_steps - 1

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical portable representation."""
        return {
            "schema_version": self.schema_version,
            "workload": {
                "name": self.workload_name,
                "parameters": dict(self.workload_parameters),
            },
            "measurement": {
                "start_step": self.start_step,
                "completed_steps": self.completed_steps,
            },
        }


def _mapping(value: Any, path: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{path} must be a mapping")
    return value


def _reject_unknown_keys(
    value: Mapping[Any, Any], allowed: frozenset[str], path: str
) -> None:
    non_string = [key for key in value if not isinstance(key, str)]
    if non_string:
        raise ContractValidationError(f"{path} keys must be strings")
    unknown = sorted(set(value) - allowed)
    if unknown:
        rendered = ", ".join(f"{path}.{key}" for key in unknown)
        raise ContractValidationError(f"unknown field(s): {rendered}")


def _required(value: Mapping[Any, Any], key: str, path: str) -> Any:
    if key not in value:
        raise ContractValidationError(f"{path}.{key} is required")
    return value[key]


def _has_control_characters(value: str) -> bool:
    return any(unicodedata.category(char).startswith("C") for char in value)


def _workload_name(value: Any) -> str:
    path = "guard.workload.name"
    if not isinstance(value, str):
        raise ContractValidationError(f"{path} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ContractValidationError(f"{path} must not be empty")
    if len(normalized) > MAX_WORKLOAD_NAME_LENGTH:
        raise ContractValidationError(
            f"{path} must be at most {MAX_WORKLOAD_NAME_LENGTH} characters"
        )
    if _has_control_characters(normalized):
        raise ContractValidationError(
            f"{path} must not contain control characters"
        )
    return normalized


def _parameter_key(value: Any) -> str:
    path = "guard.workload.parameters"
    if not isinstance(value, str):
        raise ContractValidationError(f"{path} keys must be strings")
    if value != value.strip() or not value:
        raise ContractValidationError(
            f"{path} keys must be nonempty and have no surrounding whitespace"
        )
    if len(value) > MAX_PARAMETER_KEY_LENGTH:
        raise ContractValidationError(
            f"{path} keys must be at most {MAX_PARAMETER_KEY_LENGTH} characters"
        )
    if _has_control_characters(value):
        raise ContractValidationError(
            f"{path} keys must not contain control characters"
        )
    return value


def _parameter_value(value: Any, path: str) -> Scalar:
    # bool is a subclass of int, so accept it before numeric validation.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not MIN_SIGNED_INT <= value <= MAX_SIGNED_INT:
            raise ContractValidationError(
                f"{path} integer must fit in a signed 64-bit value"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{path} must be finite")
        return value
    if isinstance(value, str):
        if not value.strip():
            raise ContractValidationError(f"{path} must not be empty")
        if value != value.strip():
            raise ContractValidationError(
                f"{path} must not have surrounding whitespace"
            )
        if len(value) > MAX_PARAMETER_STRING_LENGTH:
            raise ContractValidationError(
                f"{path} must be at most "
                f"{MAX_PARAMETER_STRING_LENGTH} characters"
            )
        if _has_control_characters(value):
            raise ContractValidationError(
                f"{path} must not contain control characters"
            )
        return value
    raise ContractValidationError(
        f"{path} must be a string, integer, finite float, or Boolean"
    )


def _parameters(value: Any) -> tuple[tuple[str, Scalar], ...]:
    path = "guard.workload.parameters"
    parameters = _mapping(value, path)
    if len(parameters) > MAX_PARAMETER_COUNT:
        raise ContractValidationError(
            f"{path} must contain at most {MAX_PARAMETER_COUNT} entries"
        )

    normalized: list[tuple[str, Scalar]] = []
    for raw_key, raw_value in parameters.items():
        key = _parameter_key(raw_key)
        normalized.append((key, _parameter_value(raw_value, f"{path}.{key}")))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{path} must be an integer")
    if value < minimum:
        raise ContractValidationError(f"{path} must be >= {minimum}")
    if value > MAX_SIGNED_INT:
        raise ContractValidationError(
            f"{path} must fit in a signed 64-bit value"
        )
    return value


def parse_guard_contract(raw: Any) -> MeasurementContract:
    """Validate one raw ``traceml.yaml`` guard block.

    The returned object owns immutable copies of all declared values, so later
    mutations to the YAML-derived mapping cannot change the captured run.
    """
    guard = _mapping(raw, "guard")
    _reject_unknown_keys(guard, _GUARD_KEYS, "guard")

    schema_version = _integer(
        _required(guard, "schema_version", "guard"),
        "guard.schema_version",
        minimum=1,
    )
    if schema_version != SCHEMA_VERSION:
        raise ContractValidationError(
            "guard.schema_version is unsupported; expected 1"
        )

    workload = _mapping(
        _required(guard, "workload", "guard"), "guard.workload"
    )
    _reject_unknown_keys(workload, _WORKLOAD_KEYS, "guard.workload")
    workload_name = _workload_name(
        _required(workload, "name", "guard.workload")
    )
    workload_parameters = _parameters(workload.get("parameters", {}))

    measurement = _mapping(
        _required(guard, "measurement", "guard"), "guard.measurement"
    )
    _reject_unknown_keys(measurement, _MEASUREMENT_KEYS, "guard.measurement")
    start_step = _integer(
        _required(measurement, "start_step", "guard.measurement"),
        "guard.measurement.start_step",
        minimum=1,
    )
    completed_steps = _integer(
        _required(measurement, "completed_steps", "guard.measurement"),
        "guard.measurement.completed_steps",
        minimum=1,
    )
    if completed_steps - 1 > MAX_SIGNED_INT - start_step:
        raise ContractValidationError(
            "guard.measurement requested step range exceeds the supported "
            "signed 64-bit step ID"
        )

    return MeasurementContract(
        schema_version=schema_version,
        workload_name=workload_name,
        workload_parameters=workload_parameters,
        start_step=start_step,
        completed_steps=completed_steps,
    )


__all__ = [
    "ContractValidationError",
    "MeasurementContract",
    "Scalar",
    "parse_guard_contract",
]
