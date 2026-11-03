# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The interface an export sink implements."""

from __future__ import annotations

from typing import Protocol, Sequence

from traceml_ai.aggregator.export.records import _Record

# The base class of every export record (``StepTimeRecord`` and the ones
# that follow): each has a ``STREAM`` name and a ``to_dict()``.
Record = _Record


class Exporter(Protocol):
    """
    A sink for export records.

    The export worker calls these from one thread at a time: ``open`` once
    before the first write, ``write`` then ``flush`` per tick, and ``close``
    once at the end or after a failed ``write`` or ``flush``. A method may
    raise; the worker logs the failure and stops using the exporter.
    """

    def open(self) -> None:
        """Prepare the sink before the first write."""

    def write(self, records: Sequence[Record]) -> None:
        """Accept one tick's records, in order."""

    def flush(self) -> None:
        """Push accepted records to durable storage."""

    def close(self, *, dropped_batches: int = 0) -> None:
        """
        Release the sink.

        ``dropped_batches`` is the number of payload batches the worker
        never handed to ``write``, so the sink can record the loss.
        """
