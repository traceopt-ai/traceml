# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Transport-independent records produced by the aggregator export path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

SCHEMA_VERSION = 1


class RecordKind(str, Enum):
    """Stable record kinds in the version-one export contract."""

    STEP_TIMING_WINDOW = "step_timing_window"
    STEP_MEMORY_WINDOW = "step_memory_window"
    PROCESS_WINDOW = "process_window"
    SYSTEM_WINDOW = "system_window"
    RUNTIME_CONTEXT = "runtime_context"


@dataclass(frozen=True)
class ExportRecord:
    """One normalized record moving through the external export path.

    ``timestamp_unix_ns`` is the time the source measurement occurred.
    ``observed_timestamp_unix_ns`` is when the aggregator observed it. Neither
    value describes exporter send time. The window processor combines source
    measurements before records reach OTLP serialization.
    """

    kind: RecordKind
    timestamp_unix_ns: Optional[int]
    observed_timestamp_unix_ns: int
    resource: Mapping[str, Any]
    data: Mapping[str, Any]
    schema_version: int = SCHEMA_VERSION

    @property
    def event_name(self) -> str:
        """Return the versioned OpenTelemetry event name."""
        return f"traceml.{self.kind.value}.v{self.schema_version}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this record."""
        return {
            "schema_version": int(self.schema_version),
            "record_type": self.kind.value,
            "timestamp_unix_ns": self.timestamp_unix_ns,
            "observed_timestamp_unix_ns": int(self.observed_timestamp_unix_ns),
            "resource": dict(self.resource),
            "data": dict(self.data),
        }


__all__ = ["SCHEMA_VERSION", "ExportRecord", "RecordKind"]
