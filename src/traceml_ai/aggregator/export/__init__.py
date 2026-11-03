# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Structured export of aggregator telemetry."""

from traceml_ai.aggregator.export.base import Exporter
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

__all__ = [
    "EXPORTER_NAMES",
    "RECORD_TYPES",
    "SCHEMA_VERSION",
    "ExportStats",
    "ExportWorker",
    "Exporter",
    "JsonExporter",
    "Record",
    "RecordBatch",
    "StepTimeRecord",
    "records_from_payload",
]
