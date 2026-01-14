import time
import traceback


class DBIncrementalSender:
    """
    Incremental network sender for Database contents.
    """

    def __init__(self, db, sampler_name, sender, rank):
        self.db = db
        self.sampler_name = sampler_name
        self.sender = sender
        self.rank = rank
        self._last_sent = {}  # table_name -> index


    def flush(self):
        """
        Send all new rows since last flush.

        Payload format:
        {
          rank: <int>,
          sampler: <str>,
          tables: {
            table_name: [row, row, ...]
          }
        }
        """
        tables_payload = {}

        for table_name, rows in self.db.all_tables().items():
            last = self._last_sent.get(table_name, 0)
            new_rows = rows[last:]
            if not new_rows:
                continue

            tables_payload[table_name] = new_rows
            self._last_sent[table_name] = len(rows)

        if not tables_payload:
            return

        self.sender.send({
            "rank": self.rank,
            "sampler": self.sampler_name,
            "timestamp": time.time(),
            "tables": tables_payload,
        })