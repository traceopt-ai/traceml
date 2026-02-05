import time
from typing import Any, List, Tuple

from traceml.samplers.base_sampler import BaseSampler
from traceml.utils.layer_forward_memory_hook import (
    get_layer_forward_memory_queue,
)
from traceml.loggers.error_log import get_error_logger

from traceml.samplers.schema.layer_forward_backward_memory import (
    LayerForwardBackwardMemoryPayload,
    LayerForwardBackwardMemorySample,
)


class LayerForwardMemorySampler(BaseSampler):
    """
    Sampler for forward-pass activation memory at the layer level.

    This sampler drains the shared forward-memory event queue and
    stores each layer's activation memory as a flat record in a
    single database table.

    Design
    ------
    - One table: `layer_forward_memory`
    - One row per forward-pass event
    - Schema is shared with backward memory sampling
      (semantic difference expressed by table identity)

    Expected event payload
    ----------------------
    LayerForwardMemoryEvents with attributes:
      - model_id : int
      - step     : int
      - device   : str
      - layers   : List[(layer_name: str, memory_bytes: float)]

    Notes
    -----
    - This sampler is non-blocking and best-effort
    - Malformed or partial events are safely ignored
    - Serialization is optimized for TCP transport
    """

    def __init__(self) -> None:
        self.name = "LayerForwardMemory"
        self.sampler_name = self.name + "Sampler"
        self.table_name = self.name + "Table"
        super().__init__(sampler_name=self.sampler_name)

        # Transport policy: only the most recent row per flush
        # (drops backlog to avoid UI lag)
        self.sender.max_rows_per_flush = 1

        self.logger = get_error_logger(self.sampler_name)
        self.sample_idx = 0

    def _drain_queue(self) -> None:
        """
        Drain the forward-memory queue and persist all available events.

        This method is intentionally non-blocking and tolerant to
        malformed or partial events.
        """
        queue = get_layer_forward_memory_queue()

        while not queue.empty():
            try:
                event = queue.get_nowait()
            except Exception:
                # Queue state changed unexpectedly
                break

            if event is None:
                continue

            self._save_event(event)


    def _normalize_layers(
            self, layers: List[Tuple[str, float]]
    ) -> LayerForwardBackwardMemoryPayload:
        """
        Normalize raw layer tuples into a deterministic payload.

        Sorting ensures:
        - stable wire representation
        - predictable compute and rendering behavior
        """
        names = [name for name, _ in layers]
        bytes_ = [float(b) for _, b in layers]

        # Enforce deterministic ordering
        order = sorted(range(len(names)), key=lambda i: names[i])
        layer_names = [names[i] for i in order]
        layer_bytes = [bytes_[i] for i in order]

        return LayerForwardBackwardMemoryPayload(
            layer_names=layer_names,
            layer_memory_bytes=layer_bytes,
        )

    def _save_event(self, event: Any) -> None:
        """
        Persist a single forward-memory event to the database.

        Parameters
        ----------
        event : LayerForwardMemoryEvents
            Event produced by forward hooks.
        """
        layers = getattr(event, "layers", None)
        if not layers:
            return

        try:
            payload = self._normalize_layers(layers)

            sample = LayerForwardBackwardMemorySample(
                sample_idx=self.sample_idx,
                timestamp=time.time(),
                model_id=getattr(event, "model_id", None),
                step=getattr(event, "step", None),
                device=getattr(event, "device", None),
                payload=payload,
            )

            self.db.add_record(self.table_name, sample.to_wire())

        except Exception as e:
            # Never let sampler failures propagate
            self.logger.error(
                f"[TraceML] Failed to persist forward layer memory event: {e}"
            )


    def sample(self) -> None:
        """
        Ingest all available forward-memory events from the queue.

        Safe to call frequently; does nothing if the queue is empty.
        """
        self.sample_idx += 1
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(
                f"[TraceML] LayerForwardMemorySampler error: {e}"
            )