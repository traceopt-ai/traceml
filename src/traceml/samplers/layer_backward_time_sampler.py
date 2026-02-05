import time
from typing import Dict, Optional
from collections import deque
from queue import Empty

from .base_sampler import BaseSampler
from traceml.loggers.error_log import get_error_logger

from traceml.utils.layer_backward_time_hooks import (
    LayerBackwardTimeStepEvent,
    get_layer_backward_time_queue,
)

from traceml.samplers.schema.layer_forward_backward_time import (
    LayerForwardBackwardTimePayload,
    LayerForwardBackwardTimeSample,
)


class LayerBackwardTimeSampler(BaseSampler):
    """
    Sampler for backward-pass execution time at the layer level.

    This sampler records *aggregated per-layer execution time* for each
    fully resolved backward (gradient) step of a model.

    Key characteristics
    -------------------
    - Step-level (not per-call) telemetry
    - Aggregates multiple backward invocations per layer
    - Handles asynchronous GPU timing safely
    - FIFO semantics ensure strict step ordering

    Design
    ------
    - One table: `LayerBackwardTimeTable`
    - One row per fully resolved (model_id, step, device)
    - Forward/backward share the same schema
      (semantic difference expressed by table identity)

    Correctness guarantees
    ----------------------
    - A step is written **only after all layer GPU timings resolve**
    - Later steps are blocked until earlier ones are complete
    - Sampler failures never propagate to training
    """

    def __init__(self) -> None:
        self.name = "LayerBackwardTime"
        self.sampler_name = self.name + "Sampler"
        self.table_name = self.name + "Table"
        super().__init__(sampler_name=self.sampler_name)

        # Transport policy: only the N-most recent rows per flush
        # (prevents UI lag under bursty backward passes)
        self.sender.max_rows_per_flush = 5

        self.logger = get_error_logger(self.sampler_name)

        # FIFO buffer of unresolved backward steps
        self._local_buffer: deque[LayerBackwardTimeStepEvent] = deque()

        self.sample_idx = 0


    def _ingest_queue(self) -> None:
        """
        Drain the shared backward-time queue into the local FIFO buffer.

        Ordering is preserved. This method is non-blocking.
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
          If the earliest step is unresolved,
          later steps MUST NOT be processed.
        """
        for layer_evt in event.layers:
            if not layer_evt.try_resolve():
                return False
        return True


    def _aggregate_step(
        self, event: LayerBackwardTimeStepEvent
    ) -> LayerForwardBackwardTimePayload:
        """
        Aggregate per-call backward timing events into a per-layer payload.

        Aggregation semantics
        ---------------------
        - CPU time: summed across calls
        - GPU time: summed across calls (if available)
        - Call count: total number of invocations

        Multiple backward invocations per layer can occur due to:
        - gradient accumulation
        - shared parameters
        - recomputation / checkpointing
        """
        agg: Dict[str, Dict[str, Optional[float]]] = {}

        for evt in event.layers:
            rec = agg.setdefault(
                evt.layer_name,
                {
                    "cpu_ms": 0.0,
                    "gpu_ms": None,
                    "n_calls": 0,
                },
            )

            rec["cpu_ms"] += float(evt.cpu_duration_ms)

            if evt.gpu_duration_ms is not None:
                rec["gpu_ms"] = (rec["gpu_ms"] or 0.0) + float(evt.gpu_duration_ms)

            rec["n_calls"] += 1

        # Deterministic ordering by layer name
        layer_names = sorted(agg.keys())

        return LayerForwardBackwardTimePayload(
            layer_names=layer_names,
            cpu_time_ms=[agg[k]["cpu_ms"] for k in layer_names],
            gpu_time_ms=[agg[k]["gpu_ms"] for k in layer_names],
            n_calls=[int(agg[k]["n_calls"]) for k in layer_names],
        )



    def sample(self) -> None:
        """
        Ingest → resolve earliest step → aggregate → persist.

        FIFO rule:
          If the earliest step is unresolved,
          later steps MUST NOT be written.
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

                payload = self._aggregate_step(event)

                sample = LayerForwardBackwardTimeSample(
                    sample_idx=self.sample_idx,
                    timestamp=time.time(),
                    model_id=event.model_id,
                    step=event.step,
                    device=event.device,
                    payload=payload,
                )

                self.db.add_record(self.table_name, sample.to_wire())

        except Exception as e:
            # Absolute safety net: sampler must never disrupt training
            self.logger.error(
                f"[TraceML] LayerBackwardTimeSampler error: {e}"
            )
