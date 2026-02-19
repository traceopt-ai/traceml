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

        # Tracks the last successfully sent append-count per table.
        # Used for O(1) new-row detection: compare db.get_append_count()
        # against this value to determine how many new rows exist.
        self._last_sent_seq: dict[str, int] = {}
        self.logger = get_error_logger("DBIncrementalSender")

    def flush(self) -> None:
        """
        Flush incremental updates to the sender.

        Uses monotonic append counters for O(1) new-row detection
        instead of scanning the deque.

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
        """
        tables_payload = {}

        for table_name, rows in self.db.all_tables().items():
            if not rows:
                continue

            total = self.db.get_append_count(table_name)
            last_seq = self._last_sent_seq.get(table_name, 0)
            new_count = total - last_seq

            # O(1) fast path: nothing new since last flush
            if new_count <= 0:
                continue

            # Apply "recent-only" cap if configured
            if self.max_rows_per_flush != -1:
                new_count = min(new_count, self.max_rows_per_flush)

            n = len(rows)
            if new_count >= n:
                # First flush or eviction happened — send entire deque
                new_rows = list(rows)
            else:
                # Slice only the tail (new rows), via indexed access
                new_rows = [rows[i] for i in range(n - new_count, n)]

            tables_payload[table_name] = new_rows
            self._last_sent_seq[table_name] = total

        if not tables_payload:
            return

        try:
            self.sender.send(
                {
                    "rank": self.rank,
                    "sampler": self.sampler_name,
                    "timestamp": time.time(),
                    "tables": tables_payload,
                }
            )
        except Exception as e:
            self.logger.error(
                f"[DBIncrementalSender] sending payload failed with exception {e}"
            )
