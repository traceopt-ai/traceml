# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Typed export records, built from the aggregator's projection tables.

A record is the table's row with names: its fields are the table's columns
in DDL order, minus the autoincrement ``id``. Records are built by the same
projection writers' ``build_rows`` that SQLite uses, so the export path and
SQLite cannot disagree about a value. Missing values stay ``None``.

Only step time has a record so far. It carries the decoded ``events`` map
in place of the ``events_json`` text. The other projection tables get
records in a follow-up.

The field contract is documented in
``docs/developer_guide/export-records.md``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Tuple

from traceml_ai.aggregator.sqlite_writers import step_time as step_time_writer
from traceml_ai.telemetry.envelope import normalize_telemetry_envelope

SCHEMA_VERSION = 1

StepEvents = Dict[str, Dict[str, Dict[str, Any]]]


class _Record:
    """Behaviour shared by every export record."""

    STREAM: ClassVar[str]
    TABLE: ClassVar[str]
    SCHEMA_VERSION: ClassVar[int] = SCHEMA_VERSION
    COLUMNS: ClassVar[Tuple[str, ...]]

    @classmethod
    def _decode(cls, row: tuple) -> tuple:
        """Return the row values in field order."""
        return row

    @classmethod
    def from_row(cls, row: tuple) -> "_Record":
        """Build a record from one ``build_rows`` tuple of its table."""
        if len(row) != len(cls.COLUMNS):
            raise ValueError(
                f"{cls.TABLE} row has {len(row)} values, "
                f"expected {len(cls.COLUMNS)}"
            )
        return cls(**dict(zip(cls.COLUMNS, cls._decode(row))))

    def to_dict(self) -> Dict[str, Any]:
        """Return ``stream``, ``schema_version`` and the fields in order."""
        out: Dict[str, Any] = {
            "stream": self.STREAM,
            "schema_version": self.SCHEMA_VERSION,
        }
        out.update(dataclasses.asdict(self))
        return out


def _record(stream: str, table: str):
    """Register a frozen record class for one table."""

    def wrap(cls):
        cls = dataclass(frozen=True)(cls)
        cls.STREAM = stream
        cls.TABLE = table
        cls.COLUMNS = tuple(f.name for f in dataclasses.fields(cls))
        return cls

    return wrap


@_record("step_time", "step_time_samples")
class StepTimeRecord(_Record):
    """One step's timing events for one rank."""

    observed_timestamp_unix_ns: int
    rank: Optional[int]
    global_rank: Optional[int]
    local_rank: Optional[int]
    world_size: Optional[int]
    local_world_size: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]
    timestamp_unix_ns: Optional[int]
    seq: Optional[int]
    step: Optional[int]
    events: StepEvents

    @classmethod
    def _decode(cls, row: tuple) -> tuple:
        """Replace the trailing ``events_json`` text with its decoded map."""
        row = tuple(row)
        text = row[-1]
        events = {} if text is None or text == "" else json.loads(text)
        if not isinstance(events, dict):
            raise ValueError(f"{cls.TABLE} events_json is not an object")
        return row[:-1] + (events,)


RECORD_TYPES = (StepTimeRecord,)

_RECORD_BY_TABLE = {cls.TABLE: cls for cls in RECORD_TYPES}

# The projection writers that have export records; a subset of the SQLite
# writer's projection list.
_WRITERS = (step_time_writer,)


class RecordBatch(NamedTuple):
    """Records built from one payload, and the malformed inputs skipped."""

    records: List[_Record]
    skipped: int


def records_from_payload(
    payload: Any,
    *,
    observed_timestamp_unix_ns: Optional[int] = None,
) -> RecordBatch:
    """
    Build export records from one admitted payload.

    ``payload`` is what the SQLite writer's ``ingest`` receives: one
    envelope dict or a list of them. Each envelope gets one observed time,
    ``observed_timestamp_unix_ns`` when given, else ``time.time_ns()``.
    An item that does not normalise, a writer that raises, and a row that
    does not fit its record are skipped and counted; nothing raises.
    """
    items = payload if isinstance(payload, list) else [payload]
    records: List[_Record] = []
    skipped = 0

    for item in items:
        try:
            envelope = normalize_telemetry_envelope(item)
            if envelope is None:
                skipped += 1
                continue
            observed = (
                observed_timestamp_unix_ns
                if observed_timestamp_unix_ns is not None
                else time.time_ns()
            )
            for writer in _WRITERS:
                if not writer.accepts_sampler(envelope.meta.sampler):
                    continue
                try:
                    rows_by_table = writer.build_rows(
                        envelope=envelope,
                        observed_timestamp_unix_ns=observed,
                    )
                except Exception:
                    skipped += 1
                    continue
                for table, rows in rows_by_table.items():
                    cls = _RECORD_BY_TABLE.get(table)
                    if cls is None:
                        skipped += len(rows)
                        continue
                    for row in rows:
                        try:
                            records.append(cls.from_row(row))
                        except Exception:
                            skipped += 1
        except Exception:
            skipped += 1

    return RecordBatch(records=records, skipped=skipped)
