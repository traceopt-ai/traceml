import time


class DBIncrementalSender:
    """
    Incremental network sender for Database contents.

    Assumes tables are append-only in logical time and bounded in memory.
    Sends only the latest unseen row per table to avoid index instability
    with bounded deques.
    """

    def __init__(self, db, sampler_name, sender, rank):
        self.db = db
        self.sampler_name = sampler_name
        self.sender = sender
        self.rank = rank
        self._last_sent_step = {}  # table_name -> last step sent

    def flush(self):
        """
        Send at most ONE new row per table.

        Payload format:
        {
          rank: <int>,
          sampler: <str>,
          timestamp: <float>,
          tables: {
            table_name: [row]
          }
        }
        """
        tables_payload = {}

        for table_name in self.db.all_tables().keys():
            row = self.db.get_last_record(table_name)
            if row is None:
                continue

            step = row.get("step")
            if step is None:
                continue

            last_step = self._last_sent_step.get(table_name)
            if last_step is not None and step <= last_step:
                continue

            tables_payload[table_name] = [row]
            self._last_sent_step[table_name] = step

        if not tables_payload:
            return

        self.sender.send(
            {
                "rank": self.rank,
                "sampler": self.sampler_name,
                "timestamp": time.time(),
                "tables": tables_payload,
            }
        )
