from __future__ import annotations

import time

from traceml.loggers.error_log import get_error_logger


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
        "rank": <int>,
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
        db,
        sampler_name,
        sender=None,
        rank=None,
        max_rows_per_flush: int = -1,
    ):
        """
        Initialize the incremental sender.
        """
        self.db = db
        self.sampler_name = sampler_name
        self.sender = sender
        self.rank = rank
        self.max_rows_per_flush = int(max_rows_per_flush)

        self._last_sent_seq: dict[str, int] = {}
        self.logger = get_error_logger("DBIncrementalSender")

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

        return {
            "rank": self.rank,
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
