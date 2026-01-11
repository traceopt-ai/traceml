import time
from typing import Dict, Iterable, Optional

from .database import Database

class RemoteDBStore:
    """
    Rank-0 only: bounded in-memory databases for remote workers.

    Structure:
        self._dbs[rank][sampler_name] = Database(max_rows=N)

    Notes:
      - Uses the same Database class as local runs
      - Enforces bounded size via Database(max_rows)
    """

    def __init__(self, max_rows: int = 200):
        self.max_rows = int(max_rows)
        self._dbs: Dict[int, Dict[str, Database]] = {}
        self._last_seen: Dict[int, float] = {}

    def _get_or_create_db(self, rank: int, sampler_name: str) -> Database:
        """
        Get or create a bounded Database for (rank, sampler).
        """
        if rank not in self._dbs:
            self._dbs[rank] = {}

        if sampler_name not in self._dbs[rank]:
            self._dbs[rank][sampler_name] = Database(
                sampler_name=sampler_name,
                max_rows=self.max_rows,
            )

        return self._dbs[rank][sampler_name]


    def ingest(
        self,
        message: dict,
    ) -> None:
        """
        Ingest incremental DB rows from a remote worker.

        Args:
            message (dict): payload from DBIncrementalSender.flush()
        """
        if message is None:
            return # nothing to ingest

        rank = message.get("rank")
        sampler_name = message.get("sampler")
        tables = message.get("tables", {})

        print(message)

        self._last_seen[rank] = time.time()

        if not tables:
            return

        db = self._get_or_create_db(
            rank=rank,
            sampler_name=sampler_name,
        )

        for table_name, rows in tables.items():
            if not rows:
                continue

            db.create_or_get_table(table_name)

            for r in rows:
                db.add_record(table_name, r)


    def get_db(self, rank: int, sampler_name: str) -> Optional[Database]:
        """
        Return remote Database for a given rank and sampler.
        """
        return self._dbs.get(rank, {}).get(sampler_name)

    def ranks(self):
        """
        Return all ranks with remote telemetry.
        """
        return list(self._dbs.keys())

    def last_seen(self, rank: int) -> float:
        """
        Last time (epoch seconds) telemetry was received from rank.
        """
        return self._last_seen.get(rank, 0.0)

    def clear(self):
        """
        Drop all remote telemetry (e.g. between runs).
        """
        self._dbs.clear()
        self._last_seen.clear()
