import time
from typing import Any, Dict, List, Tuple

from traceml.samplers.base_sampler import BaseSampler
from traceml.utils.hooks.layer_backward_memory_hook import (
    get_layer_backward_queue,
)
from traceml.loggers.error_log import get_error_logger

from traceml.samplers.schema.layer_forward_backward_memory import (
    LayerForwardBackwardMemoryPayload,
    LayerForwardBackwardMemorySample,
)


class LayerBackwardMemorySampler(BaseSampler):
    """
    Sampler for backward-pass (gradient) activation memory at the layer level.

    This sampler drains the shared backward-memory event queue and
    records per-layer gradient memory using the canonical
    `LayerForwardBackwardSample` schema.

    Design
    ------
    - One table: `LayerBackwardMemoryTable`
    - One row per backward-pass event
    - Schema is shared with forward memory sampling
      (semantic difference expressed by table identity)

     Aggregation semantics
    ---------------------
    - Multiple observations of the same layer within a single step
      can occur due to:
        * gradient accumulation
        * shared parameters
        * recomputation / checkpointing
    - Gradient memory is a *capacity* metric, not additive work.
    - Therefore, per-layer memory is aggregated using **MAX**, not SUM.

    Expected event payload
    ----------------------
    LayerBackwardMemoryEvents with attributes:
      - model_id : int
      - step     : int
      - device   : str
      - layers   : List[(layer_name: str, gradient_memory_bytes: float)]

    Notes
    -----
    - Non-blocking and best-effort
    - Malformed or partial events are safely ignored
    - Serialization is optimized for TCP transport
    """

    def __init__(self) -> None:
        self.name = "LayerBackwardMemory"
        self.sampler_name = self.name + "Sampler"
        self.table_name = self.name + "Table"
        super().__init__(sampler_name=self.sampler_name)

        # Transport policy: only keep N most recent rows per flush
        # to prevent UI lag under high-frequency events.
        self.sender.max_rows_per_flush = 5

        self.logger = get_error_logger(self.sampler_name)
        self.sample_idx = 0


    def _drain_queue(self) -> None:
        """
        Drain the backward-memory queue and persist all available events.

        This method is intentionally non-blocking and tolerant to
        malformed or partial events.
        """
        queue = get_layer_backward_queue()

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
        Aggregate raw per-call gradient memory observations using MAX.

        Rationale
        ---------
        - Gradient memory reflects peak tensor residency.
        - Multiple backward calls do NOT add memory.
        - We conservatively track the maximum observed value per layer.

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
        Persist a single backward-memory event to the database.

        Parameters
        ----------
        event : LayerBackwardMemoryEvents
            Event produced by backward hooks.
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
                f"[TraceML] Failed to persist backward layer memory event: {e}"
            )

    def sample(self) -> None:
        """
        Ingest all available backward-memory events from the queue.

        Safe to call frequently; does nothing if the queue is empty.
        """
        self.sample_idx += 1
        try:
            self._drain_queue()
        except Exception as e:
            # Sampler must never disrupt training
            self.logger.error(
                f"[TraceML] LayerBackwardMemorySampler error: {e}"
            )