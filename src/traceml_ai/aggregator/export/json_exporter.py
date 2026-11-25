# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Exporter that writes JSON Lines, one file per selected stream.

Each record becomes one compact JSON object on its own line in
``<directory>/<stream>.jsonl``. ``open`` creates one file per selected
record type, exclusively: an existing file raises, so a reused run folder
cannot mix two runs. ``close`` appends one closing line per file with the
records written and the batches the worker dropped, so a run with no
records still ends with a closing line per selected stream.

``step_time.jsonl`` uses schema version 2, a public shape of the step-time
record: one entry per event with its ``device`` and measurements, and
``traceml:`` in place of the internal ``_traceml_internal:`` prefix (#680).
The record itself, SQLite and the other exporters are unchanged. The other
streams keep ``SCHEMA_VERSION``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

from traceml_ai.aggregator.export.records import (
    RECORD_TYPES,
    SCHEMA_VERSION,
    Record,
    StepEvents,
    StepTimeRecord,
    selected_streams,
)

STEP_TIME_SCHEMA_VERSION = 2
"""Schema version of ``step_time.jsonl`` lines, data and closing alike."""

_INTERNAL_EVENT_PREFIX = "_traceml_internal:"
PUBLIC_EVENT_PREFIX = "traceml:"
"""Reserved prefix of TraceML's built-in events in ``step_time.jsonl``."""

_EXPORTED_EVENT_FIELDS = ("cpu_ms", "gpu_ms", "is_gpu", "n_calls")

_STREAM_SCHEMA_VERSIONS = {StepTimeRecord.STREAM: STEP_TIME_SCHEMA_VERSION}


def _dumps(obj: Dict) -> str:
    """One compact JSON line; NaN and infinity raise ``ValueError``."""
    return json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n"


def _public_event_name(name: str) -> str:
    """``traceml:`` for a built-in event; a custom name is kept as is."""
    if name.startswith(_INTERNAL_EVENT_PREFIX):
        return PUBLIC_EVENT_PREFIX + name[len(_INTERNAL_EVENT_PREFIX) :]
    return name


def public_step_events(events: StepEvents) -> Dict[str, Dict[str, object]]:
    """
    The ``events`` map of a schema-2 ``step_time.jsonl`` line.

    Each event becomes one object: ``device`` and the measurements of its
    first device, copied without recalculation (``None`` stays ``None``,
    ``0`` stays ``0``); ``duration_ms`` is not exported. Version 2 assumes
    one device per event name, rank and step: further device entries are not
    exported. The stored map is sorted by device, so "first" is the first
    device label in that order. An event with no device entry exports
    ``device`` and every measurement as ``None``.

    Raises ``ValueError`` when two events map to the same exported name, for
    example a custom ``traceml:forward_time`` next to the built-in one.
    """
    out: Dict[str, Dict[str, object]] = {}
    for name, by_device in events.items():
        public = _public_event_name(name)
        if public in out:
            raise ValueError(f"Duplicate exported event name: {public}")
        device, stats = next(iter(by_device.items()), (None, {}))
        entry: Dict[str, object] = {"device": device}
        entry.update(
            {field: stats.get(field) for field in _EXPORTED_EVENT_FIELDS}
        )
        out[public] = entry
    return out


def _line_object(record: Record) -> Dict:
    """The JSON object written for ``record``, before the hostname rule."""
    obj = record.to_dict()
    if isinstance(record, StepTimeRecord):
        obj["schema_version"] = STEP_TIME_SCHEMA_VERSION
        obj["events"] = public_step_events(record.events)
    return obj


class JsonExporter:
    """
    Write export records as JSON Lines under one directory.

    ``streams`` names the ``Record.STREAM`` values that get a file; None
    gives every record type one.
    """

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        *,
        include_hostname: bool = False,
        streams: Optional[Iterable[str]] = None,
    ) -> None:
        self._directory = Path(directory)
        self._include_hostname = bool(include_hostname)
        # A string, an empty or an unknown selection raises here.
        self._streams = selected_streams(streams)
        self._files: Dict[str, TextIO] = {}
        self._written: Dict[str, int] = {}
        # Streams where a write or flush to the file raised.
        self._write_failed: Set[str] = set()
        self._closed = False

    def open(self) -> None:
        """
        Create the directory, then one empty file per selected record type.

        Each file is created exclusively: an existing one raises
        ``FileExistsError`` naming its path, and the files this call
        created are closed. Raises ``RuntimeError`` after ``close``.
        """
        if self._closed:
            raise RuntimeError("JsonExporter is closed")
        self._directory.mkdir(parents=True, exist_ok=True)
        created: Dict[str, TextIO] = {}
        try:
            for cls in RECORD_TYPES:
                stream = cls.STREAM
                if self._streams is not None and stream not in self._streams:
                    continue
                if stream in self._files or stream in created:
                    continue
                path = self._directory / f"{stream}.jsonl"
                try:
                    created[stream] = open(path, "x", encoding="utf-8")
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"export file {path} already exists; refusing to "
                        "overwrite or append to an earlier run"
                    ) from exc
        except BaseException:
            for fh in created.values():
                try:
                    fh.close()
                except Exception:
                    pass
            raise
        for stream, fh in created.items():
            self._files[stream] = fh
            self._written[stream] = 0

    def write(self, records: Sequence[Record]) -> None:
        """
        Append one line per record to its stream's file.

        Every record is serialised before any line is written, so a record
        that does not serialise (NaN, infinity, or a duplicate exported
        event name) raises ``ValueError`` and the batch leaves no line. ``hostname`` is dropped unless ``include_hostname``. A
        stream ``open`` did not create raises ``ValueError``; a call after
        ``close`` raises ``RuntimeError``.
        """
        if self._closed:
            raise RuntimeError("JsonExporter is closed")
        lines: List[Tuple[str, str]] = []
        for record in records:
            stream = record.STREAM
            if stream not in self._files:
                raise ValueError(f"no export file for stream {stream!r}")
            obj = _line_object(record)
            if not self._include_hostname:
                obj.pop("hostname", None)
            lines.append((stream, _dumps(obj)))
        for stream, line in lines:
            try:
                self._files[stream].write(line)
            except BaseException:
                self._write_failed.add(stream)
                raise
            self._written[stream] += 1

    def flush(self) -> None:
        """Flush every file; re-raise the first error after trying all."""
        error: Optional[BaseException] = None
        for stream, fh in self._files.items():
            try:
                fh.flush()
            except Exception as exc:
                self._write_failed.add(stream)
                error = error or exc
        if error is not None:
            raise error

    def close(self, *, dropped_batches: int = 0) -> None:
        """
        Append each file's closing line and close it (one-shot).

        ``dropped_batches`` is the worker's count at the moment of close.
        A stream whose write or flush ever raised gets no closing line, so
        a closing line always matches the lines above it. Every file is
        closed regardless; the first error is re-raised.
        """
        if self._closed:
            return
        self._closed = True
        error: Optional[BaseException] = None
        for stream, fh in self._files.items():
            try:
                if stream not in self._write_failed:
                    fh.write(
                        _dumps(
                            {
                                "stream": stream,
                                "closing": True,
                                "schema_version": _STREAM_SCHEMA_VERSIONS.get(
                                    stream, SCHEMA_VERSION
                                ),
                                "written_records": self._written[stream],
                                "dropped_batches": int(dropped_batches),
                            }
                        )
                    )
                    fh.flush()
            except Exception as exc:
                error = error or exc
            finally:
                try:
                    fh.close()
                except Exception as exc:
                    error = error or exc
        if error is not None:
            raise error
