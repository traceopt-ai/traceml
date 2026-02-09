import time
from typing import Any, Dict, List, Tuple

from traceml.loggers.error_log import get_error_logger
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.schema.layer_forward_backward_memory import (
    LayerForwardBackwardMemoryPayload,
    LayerForwardBackwardMemorySample,
)
from traceml.utils.hooks.layer_forward_memory_hook import (
    get_layer_forward_memory_queue,
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
    - Activation memory is a *capacity* metric, not additive work.
    - Therefore, per-layer memory is aggregated using **MAX**, not SUM.

    This ensures:
    - no double counting
    - conservative (OOM-safe) reporting
    - alignment with capacity-planning use cases

    Expected event payload
    ----------------------
    LayerForwardMemoryEvents with attributes:
      - model_id : int
      - step     : int
      - device   : str
      - layers   : List[(layer_name: str, memory_bytes: float)]

    Notes
    -----
    - Non-blocking and best-effort
    - Malformed or partial events are safely ignored
    - Serialization is optimized for TCP transport
    """

    def __init__(self) -> None:
        self.name = "LayerForwardMemory"
        self.sampler_name = self.name + "Sampler"
        self.table_name = self.name + "Table"
        super().__init__(sampler_name=self.sampler_name)

        # Transport policy: only the N-most recent row per flush
        # (drops backlog to avoid UI lag)
        self.sender.max_rows_per_flush = 5

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

    def _aggregate_layers_max(
        self, layers: List[Tuple[str, float]]
    ) -> LayerForwardBackwardMemoryPayload:
        """
        Aggregate raw per-call layer memory observations using MAX.

        Rationale
        ---------
        - Memory is a capacity metric.
        - Multiple calls within a step do NOT add memory.
        - We take the maximum observed value per layer.

        The output payload is:
        - deterministic
        - compact
        - safe for transport and downstream aggregation
        """
        agg: Dict[str, float] = {}

        for layer_name, mem in layers:
            mem = float(mem)
            prev = agg.get(layer_name)
            agg[layer_name] = mem if prev is None else max(prev, mem)

        # Deterministic ordering for wire format & hashing
        layer_names = sorted(agg.keys())
        layer_bytes = [agg[name] for name in layer_names]

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
            payload = self._aggregate_layers_max(layers)

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
            # Never let sampler failures propagate into training
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
