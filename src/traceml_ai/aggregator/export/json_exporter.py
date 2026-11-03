# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Exporter that writes JSON Lines, one file per stream.

Each record becomes one compact JSON object on its own line in
``<directory>/<stream>.jsonl``. ``open`` creates one file per record type,
exclusively: an existing file raises, so a reused run folder cannot mix two
runs. ``close`` appends one closing line per file with the records written
and the batches the worker dropped, so a run with no records still ends
with a closing line per stream.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, TextIO, Tuple, Union

from traceml_ai.aggregator.export.records import (
    RECORD_TYPES,
    SCHEMA_VERSION,
    Record,
)


def _dumps(obj: Dict) -> str:
    """One compact JSON line; NaN and infinity raise ``ValueError``."""
    return json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n"


class JsonExporter:
    """Write export records as JSON Lines under one directory."""

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        *,
        include_hostname: bool = False,
    ) -> None:
        self._directory = Path(directory)
        self._include_hostname = bool(include_hostname)
        self._files: Dict[str, TextIO] = {}
        self._written: Dict[str, int] = {}
        # Streams where a write or flush to the file raised.
        self._write_failed: Set[str] = set()
        self._closed = False

    def open(self) -> None:
        """
        Create the directory, then one empty file per record type.

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
        that does not serialise raises ``ValueError`` and the batch leaves
        no line. ``hostname`` is dropped unless ``include_hostname``. A
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
            obj = record.to_dict()
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
                                "schema_version": SCHEMA_VERSION,
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


__all__ = ["JsonExporter"]
