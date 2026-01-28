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
    - Only the most recent unseen row per table needs to be transmitted

    Design rationale
    ----------------
    - Sending indices is unsafe with bounded deques because old rows
      may be evicted, shifting indices.
    - Instead, we track progress using a semantic key (`step`) embedded
      in each row.
    - At most ONE new row per table is sent per flush call to:
        * bound payload size
        * minimize network overhead
        * avoid bursty traffic during fast training loops
    """

    def __init__(self, db, sampler_name, sender, rank):
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

        # Tracks the last successfully sent record per table:
        #   table_name -> record (dict object)
        #
        # We rely on object identity (not indices or step numbers) to detect
        # new records. Records are **assumed to be immutable** after insertion.
        # This remains safe even with bounded deques, which only drop references
        # without copying or mutating objects.
        self._last_sent_record = {}
        self.logger = get_error_logger("DBIncrementalSender")

    def flush(self):
        """
        Flush incremental updates to the sender.

        Behavior
        --------
        - Iterates over all tables in the database
        - For each table:
            * retrieves the most recent row
            * compares it with the last sent record using object identity
            * if it is a new record, includes it in the payload
        - Sends at most ONE row per table
        - No-op if there is nothing new to send

        Deduplication strategy
        ----------------------
        - Deduplication is based on **object identity**, not indices or step numbers.
        - Records are assumed to be immutable after insertion.
        - This is safe with bounded deques, which drop references but never copy
          or mutate stored objects.

        Payload format
        --------------
        {
            "rank": <int>,
            "sampler": <str>,
            "timestamp": <float>,
            "tables": {
                table_name: [row]
            }
        }

        Notes
        -----
        - `timestamp` reflects wall-clock send time
        - Rows are wrapped in a list to preserve compatibility with
          future batching or multi-row sends
        """
        tables_payload = {}

        # Iterate over all known tables in the database
        for table_name in self.db.all_tables().keys():
            # Fetch the most recent row (O(1))
            row = self.db.get_last_record(table_name)
            if row is None:
                continue

            # Retrieve last record sent for this table, if any
            last_row = self._last_sent_record.get(table_name)

            # Skip if this exact record was already sent
            if last_row is row:
                continue

            # Include the new record in the payload
            tables_payload[table_name] = [row]

            # Update last sent record reference
            self._last_sent_record[table_name] = row

        # Nothing new to send → return early
        if not tables_payload:
            return

        # Send the incremental payload through the configured transport TCP
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
            self.logger.error(f"[DBIncrementalSender] sending payload failed with exception {e}")
