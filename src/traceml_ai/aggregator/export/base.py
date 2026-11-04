# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The interface an export sink implements."""

from __future__ import annotations

from typing import Protocol, Sequence

from traceml_ai.aggregator.export.records import Record


class Exporter(Protocol):
    """
    A sink for export records.

    The export worker calls the complete lifecycle from its background
    thread: ``open`` once before the first write, ``write`` then ``flush``
    per tick, and ``close`` once at the end, or after a failed ``write`` or
    ``flush``. Calls never overlap. ``close`` is not called when ``open``
    raised. A method may raise; the worker logs the failure and stops using
    the exporter.
    """

    def open(self) -> None:
        """Prepare the sink before the first write."""

    def write(self, records: Sequence[Record]) -> None:
        """Accept one tick's records, in order."""

    def flush(self) -> None:
        """
        Hand accepted records to the sink's storage layer (for a file, the
        OS buffers); crash durability is the sink's own policy.
        """

    def close(self, *, dropped_batches: int = 0) -> None:
        """
        Release the sink.

        ``dropped_batches`` is the number of payload batches the worker
        never handed to ``write``, so the sink can record the loss.
        """
