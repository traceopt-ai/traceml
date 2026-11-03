# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Exporter that writes JSON Lines, one file per stream.

Each record becomes one compact JSON object on its own line in
``<directory>/<stream>.jsonl``. A file is created on the first write for
its stream and never overwritten: an existing file raises, so a reused run
folder cannot mix two runs. ``close`` appends one closing line per file
with the records written and the batches the worker dropped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, TextIO, Union

from traceml_ai.aggregator.export.base import Record
from traceml_ai.aggregator.export.records import SCHEMA_VERSION


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
        self._closed = False

    def open(self) -> None:
        """Create the export directory, parents included."""
        self._directory.mkdir(parents=True, exist_ok=True)

    def write(self, records: Sequence[Record]) -> None:
        """
        Append one line per record to its stream's file.

        ``hostname`` is dropped unless ``include_hostname``. A record that
        does not serialise raises ``ValueError``; lines before it stay
        written.
        """
        for record in records:
            obj = record.to_dict()
            if not self._include_hostname:
                obj.pop("hostname", None)
            line = _dumps(obj)
            stream = record.STREAM
            self._file(stream).write(line)
            self._written[stream] += 1

    def flush(self) -> None:
        """Flush every open file."""
        for fh in self._files.values():
            fh.flush()

    def close(self, *, dropped_batches: int = 0) -> None:
        """Append each file's closing line and close it (idempotent)."""
        if self._closed:
            return
        self._closed = True
        error: Optional[BaseException] = None
        for stream, fh in self._files.items():
            try:
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

    def _file(self, stream: str) -> TextIO:
        """The stream's file, created exclusively on first use."""
        fh = self._files.get(stream)
        if fh is not None:
            return fh
        path = self._directory / f"{stream}.jsonl"
        try:
            fh = open(path, "x", encoding="utf-8")
        except FileExistsError as exc:
            raise FileExistsError(
                f"export file {path} already exists; refusing to overwrite "
                "or append to an earlier run"
            ) from exc
        self._files[stream] = fh
        self._written[stream] = 0
        return fh


__all__ = ["JsonExporter"]
