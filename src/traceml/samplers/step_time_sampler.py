"""
StepTimeSampler

Step-level timing sampler for TraceML.

Reads StepTimeBatch objects from the STEP timing queue, resolves GPU timings
asynchronously (without blocking training), aggregates repeated regions within
the same optimizer step (e.g., gradient accumulation), and persists one record
per.

Key semantics
-------------
- Input unit: StepTimeBatch(step, events=[TimeEvent...])
- Correctness: a step is persisted only after *all* GPU events in that step
  have resolved. Later steps are blocked until earlier steps resolve (FIFO).
- Aggregation: durations are SUM'ed across repeated occurrences of the same
  event name within a step; n_calls is counted.
- Storage: one DB table (e.g. "step_time"), one row per step, with a nested
  payload of all aggregated events for that step.
"""

from collections import defaultdict, deque
from queue import Empty
from typing import Any, Deque, Dict, Tuple

from traceml.loggers.error_log import get_error_logger
from traceml.samplers.schema.step_time_schema import StepTimeEventSample
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
    Writes a single DB record per step into a fixed table (default: "step_time").
    Each record contains a nested map of aggregated event stats for the step.

    Notes
    -----
    - CPU events resolve immediately.
    - GPU events resolve asynchronously; sampler never synchronizes the device.
    - Failures are logged and never propagate into training.
    """

    def __init__(self) -> None:
        self.name = "StepTime"
        self.sampler_name = self.name + "Sampler"
        self.table_name = self.name + "Table"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

        # Local FIFO buffer of pending step batches (order preserved).
        self._pending: Deque[StepTimeBatch] = deque()

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
            self._pending.append(batch)

    @staticmethod
    def _cpu_duration_ms(evt: TimeEvent) -> float:
        return (evt.cpu_end - evt.cpu_start) * 1000.0

    def _step_is_resolved(self, batch: StepTimeBatch) -> bool:
        """
        Return True if *all* events in the batch are resolved.

        CPU events resolve immediately. GPU events resolve only if the recorded
        CUDA end event has completed (non-blocking).
        """
        return all(evt.try_resolve() for evt in batch.events)

    def _build_step_payload(
        self, batch: StepTimeBatch
    ) -> Tuple[float, Dict[str, Dict[str, Dict[str, Any]]]]:
        """
        Build a single per-step payload containing all aggregated event timings.

        Returns
        -------
        (timestamp_s, events_payload)

        - timestamp_s: float
            Step timestamp in seconds. Chosen as max cpu_end across all events
            in the step (stable, monotonic-ish for UI ordering).
        - events_payload: dict
            Nested mapping:

            events[event_name][device] = {
                "is_gpu": bool,
                "duration_ms": float,   # SUM across repeats in this step
                "n_calls": int,         # count of repeats in this step
            }

        Aggregation key
        ---------------
        (event_name, device, is_gpu)
        """
        # For each (name, device, is_gpu) accumulate totals.
        sum_ms: Dict[Tuple[str, str, bool], float] = defaultdict(float)
        n_calls: Dict[Tuple[str, str, bool], int] = defaultdict(int)

        ts_max = 0.0

        for evt in batch.events:
            # Update step timestamp (seconds).
            ts_max = float(max(ts_max, float(evt.cpu_end)))

            is_gpu = evt.gpu_time_ms is not None
            duration_ms = (
                float(evt.gpu_time_ms)
                if is_gpu
                else float(self._cpu_duration_ms(evt))
            )

            key = (str(evt.name), str(evt.device), bool(is_gpu))
            sum_ms[key] += duration_ms
            n_calls[key] += 1

        # Convert to nested payload grouped by event_name then device.
        events: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for (name, device, is_gpu), total_ms in sum_ms.items():
            events[name][device] = {
                "is_gpu": bool(is_gpu),
                "duration_ms": float(total_ms),
                "n_calls": int(n_calls[(name, device, is_gpu)]),
            }

        return float(ts_max), dict(events)

    def _save_step(
        self, step: int, timestamp: float, events: Dict[str, Any]
    ) -> None:
        """
        Persist one step-aligned record.

        Storage format is defined by StepTimeEventSample (shared contract).
        The DB layer is expected to serialize nested structures (e.g., JSON).
        """
        sample = StepTimeEventSample(
            seq=self.sample_idx,
            timestamp=float(timestamp),
            step=int(step),
            events=events,
        )
        self.db.add_record(self.table_name, sample.to_wire())

    def sample(self) -> None:
        """
        Drain → resolve earliest step → aggregate → persist (FIFO).

        Guarantees
        ----------
        - If the earliest pending step is not fully resolved, no later steps
          are written (strict FIFO).
        - GPU resolution is non-blocking; training is never synchronized.
        """
        self.sample_idx += 1

        try:
            self._ingest_queue()

            # Write as many fully-resolved steps as possible (in order).
            while self._pending:
                batch = self._pending[0]
                if not self._step_is_resolved(batch):
                    return

                self._pending.popleft()

                ts, events = self._build_step_payload(batch)
                self._save_step(
                    step=int(batch.step), timestamp=ts, events=events
                )

        except Exception as e:
            self.logger.error(f"[TraceML] StepTimeSampler error: {e}")
