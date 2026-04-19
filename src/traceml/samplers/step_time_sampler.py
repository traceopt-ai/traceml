"""
StepTimeSampler

Step-level timing sampler for TraceML.

Reads StepTimeBatch objects from the STEP timing queue, resolves GPU timings
asynchronously (without blocking training), aggregates repeated regions within
the same optimizer step, and persists one record per step.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, Tuple

from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.schema.step_time_schema import StepTimeEventSample
from traceml.samplers.utils import append_queue_nowait_to_deque
from traceml.utils.timing import StepTimeBatch, TimeEvent, get_step_time_queue


class StepTimeSampler(BaseSampler):
    """
    Sampler for STEP-scoped timing.
    """

    def __init__(self) -> None:
        super().__init__(
            sampler_name="StepTimeSampler",
            table_name="StepTimeTable",
        )

        self._pending: Deque[StepTimeBatch] = deque()
        self.sample_idx = 0

    def _ingest_queue(self) -> None:
        """
        Drain shared STEP queue and append all batches into local FIFO buffer.
        """
        append_queue_nowait_to_deque(
            get_step_time_queue(),
            self._pending,
        )

    @staticmethod
    def _cpu_duration_ms(evt: TimeEvent) -> float:
        return (evt.cpu_end - evt.cpu_start) * 1000.0

    def _step_is_resolved(self, batch: StepTimeBatch) -> bool:
        """
        Return True if all events in the batch are resolved.
        """
        return all(evt.try_resolve() for evt in batch.events)

    def _build_step_payload(
        self, batch: StepTimeBatch
    ) -> Tuple[float, Dict[str, Dict[str, Dict[str, Any]]]]:
        """
        Build a single per-step payload containing all aggregated event timings.
        """
        sum_ms: Dict[Tuple[str, str, bool], float] = defaultdict(float)
        n_calls: Dict[Tuple[str, str, bool], int] = defaultdict(int)

        ts_max = 0.0

        for evt in batch.events:
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
        """
        sample = StepTimeEventSample(
            seq=self.sample_idx,
            timestamp=float(timestamp),
            step=int(step),
            events=events,
        )
        self._add_record(sample.to_wire())

    def sample(self) -> None:
        """
        Drain -> resolve earliest step -> aggregate -> persist.
        """
        self.sample_idx += 1

        try:
            self._ingest_queue()

            while self._pending:
                batch = self._pending[0]
                if not self._step_is_resolved(batch):
                    return

                self._pending.popleft()

                ts, events = self._build_step_payload(batch)
                self._save_step(
                    step=int(batch.step),
                    timestamp=ts,
                    events=events,
                )

        except Exception as e:
            self.logger.error(f"[TraceML] StepTimeSampler error: {e}")
