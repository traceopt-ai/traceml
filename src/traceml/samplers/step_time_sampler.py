"""
StepTimeSampler

Step-level timing sampler for TraceML.

Reads StepTimeBatch objects from the STEP timing queue, resolves GPU timings
asynchronously (without blocking training), aggregates repeated regions within
the same optimizer step (e.g., gradient accumulation), and persists one record
per (step, event_name).

Key semantics
-------------
- Input unit: StepTimeBatch(step, events=[TimeEvent...])
- Correctness: a step is persisted only after *all* GPU events in that step
  have resolved. Later steps are blocked until earlier steps resolve (FIFO).
- Aggregation: durations are SUM'ed across repeated occurrences of the same
  event name within a step; n_calls is counted.
"""

from collections import deque
from queue import Empty
from typing import Deque, Dict, List, Tuple

from traceml.loggers.error_log import get_error_logger
from traceml.samplers.schema.time_schema import TimeEventSample
from traceml.utils.timing import StepTimeBatch, TimeEvent, get_step_time_queue

from .base_sampler import BaseSampler


class StepTimeSampler(BaseSampler):
    """
    Sampler for STEP-scoped timing.

    Design
    ------
    - Drains the shared STEP timing queue (batches) into a local FIFO buffer.
    - Ensures strict step ordering: if the earliest step is unresolved, later
      steps are not written.
    - Resolves GPU timings using non-blocking queries.
    - Aggregates repeated events within the same step (SUM + call count).

    Storage
    -------
    Persists into one DB table per event name (same as legacy TimeSampler),
    but with aggregated per-step rows.

    Notes
    -----
    - CPU events resolve immediately.
    - GPU events resolve asynchronously; sampler never synchronizes the device.
    - Failures are logged and never propagate into training.
    """

    def __init__(self) -> None:
        self.sampler_name = "TimeSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

        # Local FIFO buffer of pending step batches (order preserved).
        self._local_steps: Deque[StepTimeBatch] = deque()

        self.sample_idx = 0

    def _ingest_queue(self) -> None:
        """
        Drain shared STEP queue and append all batches into local FIFO buffer.
        """

        q = get_step_time_queue()
        while True:
            try:
                batch = q.get_nowait()
            except Empty:
                break
            if batch is None:
                continue
            self._local_steps.append(batch)

    @staticmethod
    def _cpu_duration_ms(evt: TimeEvent) -> float:
        return (evt.cpu_end - evt.cpu_start) * 1000.0

    def _step_is_resolved(self, batch: StepTimeBatch) -> bool:
        """
        Return True if *all* events in the batch are resolved.

        CPU events resolve immediately. GPU events resolve only if the recorded
        CUDA end event has completed (non-blocking).
        """
        for evt in batch.events:
            if not evt.try_resolve():
                return False
        return True

    def _aggregate_step(
        self, batch: StepTimeBatch
    ) -> List[Tuple[str, str, bool, float, int, float]]:
        """
        Aggregate all events within a step by (event name, device, is_gpu).

        Returns a list of tuples:
          (name, device, is_gpu, duration_ms_sum, n_calls, timestamp)

        timestamp is the max cpu_end observed for that aggregated group.
        """
        agg: Dict[Tuple[str, str, bool], Dict[str, float]] = {}
        calls: Dict[Tuple[str, str, bool], int] = {}
        ts_max: Dict[Tuple[str, str, bool], float] = {}

        for evt in batch.events:
            is_gpu = evt.gpu_time_ms is not None
            duration_ms = float(evt.gpu_time_ms) if is_gpu else float(self._cpu_duration_ms(evt))
            key = (str(evt.name), str(evt.device), bool(is_gpu))

            agg[key] = {"sum_ms": float(agg.get(key, {}).get("sum_ms", 0.0)) + duration_ms}
            calls[key] = int(calls.get(key, 0) + 1)
            ts_max[key] = float(max(ts_max.get(key, 0.0), float(evt.cpu_end)))

        out: List[Tuple[str, str, bool, float, int, float]] = []
        for (name, device, is_gpu), rec in agg.items():
            out.append((name, device, is_gpu, float(rec["sum_ms"]), int(calls[(name, device, is_gpu)]), float(ts_max[(name, device, is_gpu)])))
        return out

    def _save_aggregates(
        self,
        step: int,
        scope: str,
        aggregates: List[Tuple[str, str, bool, float, int, float]],
    ) -> None:
        """
        Persist aggregated step timings.

        Keeps legacy behavior: one DB table per event name.
        Adds aggregation awareness via `n_calls` encoded in sample_idx stream
        (optionally extend schema later).
        """
        # NOTE: TimeEventSample currently doesn't include n_calls.
        # If we later extend schema, add n_calls here.
        for name, device, is_gpu, duration_ms, _n_calls, ts in aggregates:
            sample = TimeEventSample(
                sample_idx=self.sample_idx,
                timestamp=float(ts),
                step=int(step),
                scope=str(scope),
                device=str(device),
                is_gpu=bool(is_gpu),
                duration_ms=float(duration_ms),
            )
            self.db.add_record(name, sample.to_wire())

    def sample(self) -> None:
        """
        Ingest → resolve earliest step → aggregate → persist (FIFO).
        """
        self.sample_idx += 1
        try:
            self._ingest_queue()

            while self._local_steps:
                batch = self._local_steps[0]

                if not self._step_is_resolved(batch):
                    break

                # Earliest step fully resolved
                self._local_steps.popleft()

                aggregates = self._aggregate_step(batch)
                self._save_aggregates(
                    step=int(batch.step),
                    scope="step",
                    aggregates=aggregates,
                )

        except Exception as e:
            self.logger.error(f"[TraceML] StepTimeSampler error: {e}")