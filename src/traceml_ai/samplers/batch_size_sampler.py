"""
BatchSizeSampler

Step-level batch-size sampler for TraceML.

Reads BatchSizeBatch objects from the shared batch-size queue, aggregates
all H2D transfer bytes observed within the same optimizer step using SUM
(handling gradient accumulation naturally), and persists **one record
per step, per rank**.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque

from traceml_ai.samplers.base_sampler import BaseSampler
from traceml_ai.samplers.schema.batch_size_schema import BatchSizeSample
from traceml_ai.samplers.utils import append_queue_nowait_to_deque
from traceml_ai.utils.batch_size import BatchSizeBatch, get_batch_size_queue


class BatchSizeSampler(BaseSampler):
    """
    Sampler for STEP-scoped input batch size in bytes.

    Aggregation
    -----------
    SUM across all H2D events within one step. One row per (step, rank).
    """

    def __init__(self) -> None:
        super().__init__(
            sampler_name="BatchSizeSampler",
            table_name="BatchSizeTable",
        )
        self._pending: Deque[BatchSizeBatch] = deque()
        self.sample_idx = 0

    def _ingest_queue(self) -> None:
        """Drain shared queue into local FIFO buffer."""
        append_queue_nowait_to_deque(get_batch_size_queue(), self._pending)

    def _save_step(self, batch: BatchSizeBatch) -> None:
        """Sum bytes across the step and persist one record."""
        bytes_total = 0
        for evt in batch.events:
            try:
                bytes_total += int(evt.bytes_count)
            except Exception:
                continue

        sample = BatchSizeSample(
            seq=self.sample_idx,
            timestamp=time.time(),
            step=int(batch.step),
            bytes_total=int(bytes_total),
            n_transfers=int(len(batch.events)),
        )
        self._add_record(sample.to_wire())

    def sample(self) -> None:
        """Drain queue -> sum per step -> persist one record per step."""
        self.sample_idx += 1
        try:
            self._ingest_queue()
            while self._pending:
                batch = self._pending.popleft()
                self._save_step(batch)
        except Exception as e:
            self.logger.error(f"[TraceML] BatchSizeSampler error: {e}")
