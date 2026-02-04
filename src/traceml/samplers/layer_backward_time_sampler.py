from typing import Dict, List, Tuple
from collections import deque
from queue import Empty

from .base_sampler import BaseSampler
from traceml.loggers.error_log import get_error_logger
from traceml.utils.layer_backward_time_hooks import (
    LayerBackwardTimeStepEvent,
    get_layer_backward_time_queue,
)


class LayerBackwardTimeSampler(BaseSampler):
    """
    Sampler for backward-pass execution time at the layer level.

    This sampler mirrors LayerForwardTimeSampler exactly, but operates
    on backward (gradient) execution.

    Design
    ------
    - One table: `LayerBackwardTimeTable`
    - One row per (model_id, step)
    - Uses a local FIFO buffer to ensure:
        * steps are written in order
        * no step is written until *all* GPU timings are resolved

    Final record format
    -------------------
    {
        "seq": int,
        "model_id": int,
        "step": int,
        "device": str,
        "layers": List[
            (layer_name: str,
             cpu_time_ms: float,
             gpu_time_ms: Optional[float],
             n_calls: int)
        ]
    }

    Notes
    -----
    Multiple backward invocations of the same layer within a step can occur due to:
    - gradient accumulation
    - shared parameters
    - checkpointing / recomputation

    These are aggregated by summation.
    """

    def __init__(self) -> None:
        self.name = "LayerBackwardTime"
        self.sampler_name = self.name + "Sampler"
        self.table_name = self.name + "Table"
        super().__init__(sampler_name=self.sampler_name)

        # Sample transport: send only the most recent row per flush (drops backlog)
        self.sender.max_rows_per_flush = 1

        self.logger = get_error_logger(self.sampler_name)

        # FIFO of LayerBackwardTimeStepEvent
        self._local_buffer: deque[LayerBackwardTimeStepEvent] = deque()

        self.sample_idx = 0


    def _ingest_queue(self) -> None:
        """
        Drain the shared backward-time queue into the local FIFO buffer.

        Queue payload is expected to be LayerBackwardTimeStepEvent.
        """
        q = get_layer_backward_time_queue()

        while True:
            try:
                event = q.get_nowait()
            except Empty:
                break

            if event is None:
                continue

            self._local_buffer.append(event)


    def _step_is_resolved(self, event: LayerBackwardTimeStepEvent) -> bool:
        """
        Check whether *all* backward timing events in the step are resolved.

        FIFO rule:
          If the earliest step is not fully resolved,
          later steps must not be processed.
        """
        for layer_evt in event.layers:
            if not layer_evt.try_resolve():
                return False
        return True


    def _aggregate_step(
            self, event: LayerBackwardTimeStepEvent
    ) -> Dict[str, object]:
        """
        Aggregate per-call backward timing events into per-layer summaries.

        Aggregation is done by summation over:
        - CPU time
        - GPU time
        - number of calls
        """
        agg: Dict[str, Dict[str, float]] = {}

        for evt in event.layers:
            rec = agg.setdefault(
                evt.layer_name,
                {
                    "cpu_ms": 0.0,
                    "gpu_ms": None,
                    "n_calls": 0,
                },
            )

            rec["cpu_ms"] += evt.cpu_duration_ms
            if evt.gpu_duration_ms is not None:
                rec["gpu_ms"] = (rec["gpu_ms"] or 0.0) + evt.gpu_duration_ms
            rec["n_calls"] += 1

        layers: List[Tuple[str, float, float, int]] = []
        for layer_name, rec in agg.items():
            layers.append(
                (
                    layer_name,
                    rec["cpu_ms"],
                    rec["gpu_ms"],
                    int(rec["n_calls"]),
                )
            )

        return {
            "seq": self.sample_idx,
            "model_id": event.model_id,
            "step": event.step,
            "device": event.device,
            "layers": layers,
        }


    def sample(self) -> None:
        """
        Ingest → resolve earliest step → aggregate → persist.

        Stops at the first unresolved step to preserve FIFO semantics.
        """
        try:
            self._ingest_queue()

            while self._local_buffer:
                event = self._local_buffer[0]

                if not self._step_is_resolved(event):
                    break

                # Step fully resolved
                self._local_buffer.popleft()
                self.sample_idx += 1

                record = self._aggregate_step(event)
                self.db.add_record(self.table_name, record)

        except Exception as e:
            self.logger.error(
                f"[TraceML] LayerBackwardTimeSampler error: {e}"
            )