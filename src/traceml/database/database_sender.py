# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Any, Optional, Protocol

from traceml.loggers.error_log import get_error_logger
from traceml.runtime.sender import SenderIdentity


class AppendOnlyDatabase(Protocol):
    """Minimal database surface needed by DBIncrementalSender."""

    def all_tables(self) -> dict[str, Any]:
        """Return table-name to rows mapping."""
        ...

    def get_append_count(self, table_name: str) -> int:
        """Return the monotonic append count for a table."""
        ...


class DBIncrementalSender:
    """
    Incremental network sender for database contents.

    This class is responsible for streaming database updates over the network
    in an incremental and low-overhead manner.

    Core assumptions
    ----------------
    - Tables are append-only in logical time.
    - Tables are bounded in memory.
    - Supports a sampling knob:
        * max_rows_per_flush = -1  -> best-effort "send everything since last sent row"
        * max_rows_per_flush = N>0 -> send at most the latest N rows per flush

    Design rationale
    ----------------
    - Sending indices is unsafe with bounded deques because old rows
      may be evicted, shifting indices.
    - Instead, we track progress using a monotonic append counter.

    Payload contract
    ----------------
    This class returns a single payload dict from `collect_payload()`:

    {
        "rank": <int>,          # globally unique runtime rank
        "global_rank": <int>,
        "local_rank": <int>,
        "sampler": <str>,
        "timestamp": <float>,
        "tables": {
            table_name: [row, row, ...]
        }
    }

    Higher runtime layers may batch multiple sampler payload dicts into a
    `list[dict]` before transport. That batching is outside this class.
    """

    def __init__(
        self,
        db: AppendOnlyDatabase,
        sampler_name: str,
        sender: Optional[Any] = None,
        rank: Optional[int] = None,
        max_rows_per_flush: int = -1,
    ) -> None:
        """
        Initialize the incremental sender.

        ``rank`` is usually attached by TelemetryPublisher after sampler
        construction. New runtime code should attach the global rank so
        downstream storage can group multi-node telemetry without collisions.
        """
        self.db = db
        self.sampler_name = str(sampler_name)
        self.sender = sender
        self.rank = rank
        self.identity: Optional[SenderIdentity] = None
        self.max_rows_per_flush = int(max_rows_per_flush)

        self._last_sent_seq: dict[str, int] = {}
        self.logger = get_error_logger("DBIncrementalSender")

    def _payload_rank(self) -> int:
        """
        Return the globally unique rank for outbound payloads.

        A missing rank means the sender was not attached by the runtime
        publisher. Failing fast here prevents malformed telemetry from reaching
        the aggregator with ``rank=None``.
        """
        if self.rank is None:
            raise RuntimeError(
                "DBIncrementalSender requires an attached runtime rank before "
                "collecting payloads."
            )
        return int(self.rank)

    def _payload_identity(self) -> SenderIdentity:
        """
        Return the rank identity for outbound payloads.

        ``identity`` is attached by TelemetryPublisher. The ``rank`` fallback
        keeps direct unit-test and legacy construction paths working, but new
        runtime payloads should always include explicit global/local rank.
        """
        if self.identity is not None:
            return self.identity

        rank = self._payload_rank()
        return SenderIdentity(global_rank=rank, local_rank=rank)

    def collect_payload(self) -> dict | None:
        """
        Collect new rows since the last flush and return them as a ready-to-send
        payload dict. Returns `None` when there is nothing new to send.

        This method advances the internal cursor (`_last_sent_seq`) so that a
        subsequent call to either `collect_payload()` or `flush()` will not
        re-send the same rows.
        """
        tables_payload = {}

        for table_name, rows in self.db.all_tables().items():
            if not rows:
                continue

            total = self.db.get_append_count(table_name)
            last_seq = self._last_sent_seq.get(table_name, 0)
            new_count = total - last_seq

            if new_count <= 0:
                continue

            if self.max_rows_per_flush != -1:
                new_count = min(new_count, self.max_rows_per_flush)

            n = len(rows)
            if new_count >= n:
                new_rows = list(rows)
            else:
                new_rows = [rows[i] for i in range(n - new_count, n)]

            tables_payload[table_name] = new_rows
            self._last_sent_seq[table_name] = total

        if not tables_payload:
            return None
        identity = self._payload_identity()

        return {
            "rank": identity.rank,
            "global_rank": identity.global_rank,
            "local_rank": identity.local_rank,
            "sampler": self.sampler_name,
            "timestamp": time.time(),
            "tables": tables_payload,
        }

    def flush(self) -> None:
        """
        Collect and immediately send incremental updates as a single payload.
        """
        payload = self.collect_payload()
        if payload is None:
            return
        try:
            self.sender.send(payload)
        except Exception as e:
            self.logger.error(
                f"[DBIncrementalSender] sending payload failed with exception {e}"
            )
