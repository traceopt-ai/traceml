# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import struct
from pathlib import Path
from typing import Dict

from traceml_ai.runtime.config import config
from traceml_ai.runtime.identity import resolve_runtime_identity
from traceml_ai.runtime.session import get_session_id, rank_dir_name
from traceml_ai.utils.msgpack_codec import encode as encode_msgpack


def _rank_suffix() -> str:
    """
    Return a rank suffix for filesystem isolation.

    Global rank is unique across all nodes, which keeps local MessagePack
    backups safe when several machines write to a shared log directory.
    """
    identity = resolve_runtime_identity()
    return rank_dir_name(identity.global_rank)


class DatabaseWriter:
    """
    Incrementally writes Database tables to disk in length-prefixed MessagePack format.

    This writer is designed to work with bounded deques where old entries
    may be evicted at any time. Uses monotonic append counters for O(1)
    new-row detection.

    Design principles
    -----------------
    - Append-only MessagePack files (stream-friendly, crash-safe)
    - Delta writes only (no full table rewrites)
    - Safe under deque eviction
    - DDP-safe: each rank writes to its own directory
    - Low overhead: flush frequency is throttled locally

    Directory layout
    ----------------
        <logs_dir>/<session_id>/<rank_suffix>/data/<sampler_name>/<table>.msgpack
    """

    def __init__(self, db, sampler_name: str, flush_every: int = 100):
        """
        Parameters
        ----------
        db : Database
            In-memory database instance to flush.
        sampler_name : str
            Name of the sampler owning this writer.
        flush_every : int, optional
            Flush once every N calls to `flush()` (default: 100).
        """
        self.db = db
        self.flush_every = max(1, int(flush_every))
        self._flush_counter = 0

        session_id = get_session_id()
        rank_dir = _rank_suffix()

        # Pathlib-based log root
        self.logs_dir = (
            Path(config.logs_dir)
            / session_id
            / rank_dir
            / "data"
            / sampler_name
        )

        # Tracks the last written append-count per table.
        # Used for O(1) new-row detection via db.get_append_count().
        self._last_written_seq: Dict[str, int] = {}

    @staticmethod
    def _encode_row(row) -> bytes:
        """
        Encode one row as MessagePack bytes.

        The shared codec helper keeps writer behavior consistent with the CLI
        and transport layers while allowing a safe fallback when ``msgspec`` is
        unavailable in a lightweight environment.
        """
        return encode_msgpack(row)

    def flush(self):
        """
        Flush new records from all tables to disk.

        Notes
        -----
        - This method may be called very frequently (e.g., every training step).
        - Actual disk I/O is throttled via `flush_every`.
        - Only records that were not written in previous flushes are appended.
        """
        if not config.enable_logging:
            return

        self._flush_counter += 1
        if self._flush_counter % self.flush_every != 0:
            return

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        for table_name, rows in self.db.all_tables().items():
            if not rows:
                continue

            total = self.db.get_append_count(table_name)
            last_seq = self._last_written_seq.get(table_name, 0)
            new_count = total - last_seq

            # O(1) fast path: nothing new since last write
            if new_count <= 0:
                continue

            n = len(rows)
            if new_count >= n:
                # First flush or eviction — write entire deque
                new_rows = list(rows)
            else:
                # Slice only the tail (new rows)
                new_rows = [rows[i] for i in range(n - new_count, n)]

            path = self.logs_dir / f"{table_name}.msgpack"
            with open(path, "ab") as f:
                for r in new_rows:
                    payload = self._encode_row(r)
                    f.write(struct.pack("!I", len(payload)))
                    f.write(payload)

            self._last_written_seq[table_name] = total
