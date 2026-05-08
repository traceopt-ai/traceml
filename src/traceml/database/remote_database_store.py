import time
from typing import Dict, Optional

from traceml.loggers.error_log import get_error_logger

from .database import Database


class RemoteDBStore:
    """Bounded in-memory store for telemetry received from workers."""

    def __init__(self, max_rows: int = 500):
        """Initialize the remote database store."""
        if max_rows <= 0:
            raise ValueError(f"max_rows must be > 0, got {max_rows}")
        self.max_rows = int(max_rows)

        # Mapping: rank -> sampler_name -> Database
        self._dbs: Dict[int, Dict[str, Database]] = {}

        # Mapping: rank -> last message arrival time (epoch seconds)
        self._last_seen: Dict[int, float] = {}
        self.logger = get_error_logger("RemoteDBStore")

    def _get_or_create_db(self, rank: int, sampler_name: str) -> Database:
        """Return the bounded Database for a given rank and sampler."""
        if rank not in self._dbs:
            self._dbs[rank] = {}

        if sampler_name not in self._dbs[rank]:
            self._dbs[rank][sampler_name] = Database(
                sampler_name=sampler_name,
                max_rows=self.max_rows,
            )

        return self._dbs[rank][sampler_name]

    def ingest(self, message) -> None:
        """
        Ingest telemetry from a remote worker.

        Accepts two formats:

        **Batch format** (new — from ``TCPClient.send_batch()``):
            A ``list`` of individual payload dicts.  Each item is ingested
            independently, exactly as if it had been sent in a separate message.

        **Single format** (legacy — from ``TCPClient.send()``):
            A single ``dict`` with keys ``rank``, ``sampler``, ``tables``.

        Both formats are fully supported.  The type check on ``message`` is the
        only branching point; all ingestion logic lives in :meth:`_ingest_one`.
        """
        if message is None:
            return
        if isinstance(message, list):
            # Batch envelope: one send_batch() → N logical messages
            for item in message:
                self._ingest_one(item)
        else:
            # Legacy single-dict format
            self._ingest_one(message)

    def _ingest_one(self, message: dict) -> None:
        """
        Ingest a single telemetry payload dict into the store.

        Expected payload format
        -----------------------
            {
              "rank": int,
              "sampler": str,
              "tables": {
                  "table_name": [ {row...}, {row...}, ... ],
                  ...
              }
            }

        Behavior
        --------
        - Invalid or malformed messages are ignored with a warning.
        - Remote databases and tables are created lazily.
        - Rows are appended in arrival order.
        - Memory bounds are enforced by the underlying `Database`.
        """
        if message is None:
            return

        rank = message.get("rank")
        sampler_name = message.get("sampler")

        try:
            rank = int(rank)
        except Exception:
            self.logger.warning(f"Invalid rank in message: {rank}")
            return

        tables = message.get("tables", {})

        # Track last time we received telemetry from this rank
        self._last_seen[rank] = time.time()

        if not tables:
            return

        # Get or create the per-(rank, sampler) database
        db = self._get_or_create_db(
            rank=rank,
            sampler_name=sampler_name,
        )

        # Ingest rows table-by-table
        for table_name, rows in tables.items():
            if not rows:
                continue

            # Ensure table exists before appending
            db.create_or_get_table(table_name)

            for r in rows:
                db.add_record(table_name, r)

    def get_db(self, rank: int, sampler_name: str) -> Optional[Database]:
        """
        Return the remote Database for a given rank and sampler.

        Parameters
        ----------
        rank : int
            Global rank of the worker.
        sampler_name : str
            Sampler name.

        Returns
        -------
        Optional[Database]
            The corresponding database, or None if not present.
        """
        return self._dbs.get(rank, {}).get(sampler_name)

    def ranks(self):
        """
        Return all ranks from which telemetry has been received.

        Returns
        -------
        list[int]
            List of ranks with active remote databases.
        """
        return list(self._dbs.keys())

    def last_seen(self, rank: int) -> float:
        """
        Return the last time telemetry was received from a given rank.

        Parameters
        ----------
        rank : int
            Global rank.

        Returns
        -------
        float
            Epoch timestamp in seconds, or 0.0 if the rank was never seen.
        """
        return self._last_seen.get(rank, 0.0)

    def clear(self):
        """
        Drop all remote telemetry state.

        This removes all remote databases and last-seen timestamps.
        Intended for use between runs or during reset.
        """
        self._dbs.clear()
        self._last_seen.clear()
