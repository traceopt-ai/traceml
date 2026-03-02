import time

from traceml.loggers.error_log import get_error_logger


class DBIncrementalSender:
    """
    Incremental network sender for Database contents.

    This class is responsible for streaming database updates over the network
    in an incremental and low-overhead manner.

    Core assumptions
    ----------------
    - Tables are append-only in logical time (monotonically increasing `step`)
    - Tables are bounded in memory (e.g., backed by deque(maxlen=N))
    - Supports a sampling knob:
        * max_rows_per_flush = -1  -> best-effort "send everything since last sent row"
                                   (bounded by current deque contents)
        * max_rows_per_flush = N>0 -> send at most the latest N rows per flush
                                   (may skip backlog by design)

    Design rationale
    ----------------
    - Sending indices is unsafe with bounded deques because old rows
      may be evicted, shifting indices.
    - Instead, we track progress using a semantic key (`step`) embedded
      in each row.
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

        Parameters
        ----------
        db : Database
            In-memory database instance holding sampler telemetry.
        sampler_name : str
            Logical name of the sampler emitting the data.
        sender : Any
            Transport abstraction with a `.send(payload)` method
            (e.g., socket sender, IPC sender, async queue).
        rank : int
            Distributed rank of the current process (e.g., DDP rank).
        """
        self.db = db
        self.sampler_name = sampler_name
        self.sender = sender
        self.rank = rank
        self.max_rows_per_flush = int(max_rows_per_flush)

        # Tracks the last successfully sent record per table:
        #   table_name -> record (dict object)
        #
        # We rely on object identity to detect
        # new records. Records are **assumed to be immutable** after insertion.
        # This remains safe even with bounded deques, which only drop references
        # without copying or mutating objects.
        self._last_sent_record = {}
        self.logger = get_error_logger("DBIncrementalSender")

    def collect_payload(self) -> "dict | None":
        """
        Collect new rows since the last flush and return them as a ready-to-send
        payload dict.  Returns ``None`` when there is nothing new to send.

        This method advances the internal cursor (``_last_sent_record``) so that
        a subsequent call to either :meth:`collect_payload` or :meth:`flush` will
        not re-send the same rows.

        Payload format
        --------------
        {
            "rank": <int>,
            "sampler": <str>,
            "timestamp": <float>,
            "tables": {
                table_name: [row, row, ...]  # possibly multiple rows per table
            }
        }

        The caller is responsible for actually transmitting the returned dict.
        """
        tables_payload = {}

        # Iterate over all known tables in the database
        for table_name, rows in self.db.all_tables().items():
            if not rows:
                continue

            last_row = self._last_sent_record.get(table_name)

            # Collect new rows since last_row (exclusive), using identity.
            new_rows = []
            found_last = False

            # Walk newest -> oldest
            for r in reversed(rows):
                if last_row is not None and r is last_row:
                    found_last = True
                    break
                new_rows.append(r)

                # If we are in "recent-only" mode, we only care about the latest N rows
                if (
                    self.max_rows_per_flush != -1
                    and len(new_rows) >= self.max_rows_per_flush
                ):
                    break

            if not new_rows:
                continue

            # If the cursor row wasn't found, it likely got evicted; that's fine.
            # Our policy in that case is to send the freshest snapshot available:
            # - N>0 : we already collected up to N newest rows
            # - -1  : send the entire deque (best effort)
            if last_row is not None and not found_last:
                if self.max_rows_per_flush == -1:
                    # send everything currently in the deque
                    new_rows = list(rows)
                # else: keep the collected newest <=N rows (already correct)

            # Restore chronological order (oldest -> newest)
            new_rows.reverse()

            tables_payload[table_name] = new_rows
            self._last_sent_record[table_name] = new_rows[-1]

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
        Collect and immediately send incremental updates (single message).

        This is a backward-compatible wrapper around :meth:`collect_payload`.
        Callers that need batching should use :meth:`collect_payload` directly
        and transmit the payload themselves via ``TCPClient.send_batch()``.
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
