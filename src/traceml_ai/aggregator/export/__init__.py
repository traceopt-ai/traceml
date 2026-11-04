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
    Record,
    RecordBatch,
    StepTimeRecord,
    records_from_payload,
)
from traceml_ai.aggregator.export.worker import ExportStats, ExportWorker

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


__all__ = [
    "EXPORTER_NAMES",
    "RECORD_TYPES",
    "SCHEMA_VERSION",
    "ExportStats",
    "ExportWorker",
    "Exporter",
    "ExporterContext",
    "JsonExporter",
    "Record",
    "RecordBatch",
    "StepTimeRecord",
    "parse_exporter_factory",
    "records_from_payload",
]
