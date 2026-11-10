# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Structured export of aggregator telemetry."""

import re
from typing import Optional, Tuple

from traceml_ai.aggregator.export.base import Exporter, ExporterContext
from traceml_ai.aggregator.export.json_exporter import JsonExporter
from traceml_ai.aggregator.export.records import (
    RECORD_TYPES,
    SCHEMA_VERSION,
    ProcessRecord,
    Record,
    RecordBatch,
    RuntimeEnvironmentRecord,
    StdoutStderrRecord,
    StepMemoryRecord,
    StepTimeRecord,
    SystemGpuRecord,
    SystemRecord,
    records_from_payload,
)
from traceml_ai.aggregator.export.worker import (
    ExporterStartupError,
    ExportStats,
    ExportWorker,
)

# Values the ``exporter`` setting accepts; the launcher validates against
# this before any process starts and the aggregator resolves from it.
EXPORTER_NAMES = ("jsonl",)

_FACTORY_VALUE = re.compile(
    r"^\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*:\s*([A-Za-z_]\w*)\s*$"
)


def parse_exporter_factory(value: str) -> Optional[Tuple[str, str]]:
    """
    Split a ``module:factory`` exporter value into its two names.

    Returns ``(module, attribute)`` when ``value`` is a dotted module path,
    a colon and one identifier, surrounding spaces allowed; else None.
    Imports nothing.
    """
    match = _FACTORY_VALUE.match(str(value))
    if match is None:
        return None
    return match.group(1), match.group(2)


# Names the ``export_streams`` setting accepts. ``system`` selects the host
# record and its GPU records; ``all`` selects every record stream.
EXPORT_STREAM_NAMES = ("step_time", "step_memory", "process", "system", "all")
DEFAULT_EXPORT_STREAMS = "step_time"

_STREAMS_BY_NAME = {
    "step_time": ("step_time",),
    "step_memory": ("step_memory",),
    "process": ("process",),
    "system": ("system", "system_gpu"),
    "all": tuple(cls.STREAM for cls in RECORD_TYPES),
}


def parse_export_streams(value: object) -> Tuple[str, ...]:
    """
    Return the record streams a comma-separated ``export_streams`` value
    selects.

    Names are split on commas, stripped, and must be in EXPORT_STREAM_NAMES.
    The result is the selected ``Record.STREAM`` values in RECORD_TYPES
    order, each once. An empty value or an unknown name raises ValueError
    whose message names the bad value and lists EXPORT_STREAM_NAMES.
    """
    supported = list(EXPORT_STREAM_NAMES)
    names = [name.strip() for name in str(value or "").split(",")]
    names = [name for name in names if name]
    if not names:
        raise ValueError(
            "[TraceML] export_streams must name at least one stream. "
            f"Supported: {supported}"
        )
    selected = set()
    for name in names:
        streams = _STREAMS_BY_NAME.get(name)
        if streams is None:
            raise ValueError(
                f"[TraceML] Unknown export stream: {name!r}. "
                f"Supported: {supported}"
            )
        selected.update(streams)
    return tuple(cls.STREAM for cls in RECORD_TYPES if cls.STREAM in selected)


__all__ = [
    "DEFAULT_EXPORT_STREAMS",
    "EXPORTER_NAMES",
    "EXPORT_STREAM_NAMES",
    "RECORD_TYPES",
    "SCHEMA_VERSION",
    "ExportStats",
    "ExportWorker",
    "ExporterStartupError",
    "Exporter",
    "ExporterContext",
    "JsonExporter",
    "ProcessRecord",
    "Record",
    "RecordBatch",
    "RuntimeEnvironmentRecord",
    "StdoutStderrRecord",
    "StepMemoryRecord",
    "StepTimeRecord",
    "SystemGpuRecord",
    "SystemRecord",
    "parse_export_streams",
    "parse_exporter_factory",
    "records_from_payload",
]
