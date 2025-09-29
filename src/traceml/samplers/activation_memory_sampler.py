from dataclasses import dataclass, field
from collections import defaultdict
from queue import Empty
from typing import Any, Dict, List, Optional

import time

from .base_sampler import BaseSampler
from traceml.utils.activation_hook import get_activation_queue
from traceml.loggers.error_log import get_error_logger, setup_error_logger


@dataclass
class ActivationSnapshot:
    layers: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class DrainResult:
    drained_events: int
    per_device: Dict[str, List[float]]
    per_layer: Dict[str, List[float]]


class ActivationMemorySampler(BaseSampler):
    """
    Drain-all activation-event sampler.

    Each call to `sample()`:
      - Drains the activation queue.
      - Aggregates per-device stats over those new events.
      - Returns a live snapshot dict.
      - If no new events arrive, returns the last snapshot.
    """

    def __init__(
        self,
        max_raw_events: int = 10_000,
        pressure_threshold: float = 0.9,
    ):
        super().__init__()
        setup_error_logger()
        self.logger = get_error_logger("ActivationMemorySampler")

        self.pressure_threshold = float(pressure_threshold)
        self._cumulative_layer_peaks: Dict[str, float] = {}
        self._latest_snapshot: Optional[ActivationSnapshot] = None
        self._ever_seen: bool = False


    def _drain_queue(self) -> DrainResult:
        """
        Drain the activation queue completely and return:
          - number of events drained
          - mapping dev -> list of values in THIS drain cycle
        """
        try:
            q = get_activation_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] ERROR: activation queue unavailable: {e}")
            return DrainResult(0, {}, {})

        drained_events = 0
        batch_per_device: Dict[str, List[float]] = defaultdict(list)
        batch_per_layer: Dict[str, List[float]] = defaultdict(list)
        now = time.time()

        while True:
            try:
                ev = q.get_nowait()
            except Empty:
                break
            except Exception as e:
                self.logger.error(f"[TraceML] WARNING: queue.get_nowait failed: {e}")
                break

            drained_events += 1
            per_dev = getattr(ev, "per_device_activation_memory", None)
            per_layer = getattr(ev, "per_layer", None)

            for dev, mem in per_dev.items():
                batch_per_device[dev].append(float(mem))

            if per_layer:
                layer, dev_dict = next(iter(per_layer.items()))
                for _, mem in (dev_dict or {}).items():
                    batch_per_layer[layer].append(float(mem))

        return DrainResult(drained_events, batch_per_device, batch_per_layer)

    def _build_snapshot(self, result: DrainResult) -> ActivationSnapshot:
        """Construct the live snapshot dict from this drain’s per-device values."""
        layers_out: Dict[str, Dict[str, float]] = {}

        # Per-layer peaks (current + global across time)
        for layer, vals in result.per_layer.items():
            if not vals:
                continue
            curr_peak = float(max(vals))
            prev_global = float(self._cumulative_layer_peaks.get(layer, 0.0))
            global_peak = max(curr_peak, prev_global)
            self._cumulative_layer_peaks[layer] = global_peak

            layers_out[layer] = {
                "current_peak": round(curr_peak, 4),
                "global_peak": round(global_peak, 4),
            }

        return ActivationSnapshot(
            layers=layers_out,  # now: {"layer_name": peak_activation_MB}
        )

    def sample(self) -> Dict[str, Any]:
        """
        Drain the queue and compute a live snapshot.
        If nothing new arrived:
          - If we’ve seen data before: return last snapshot with `stale=True`.
          - If we’ve never seen data: return a guidance note about attaching hooks.
        """
        try:
            result = self._drain_queue()
            if result.drained_events == 0:
                # nothing new then return the last snapshot
                if self._ever_seen and self._latest_snapshot:
                    snap = ActivationSnapshot(
                        layers=dict(self._latest_snapshot.layers),
                    )
                    self._latest_snapshot = snap
                else:
                    # First-time snapshot
                    self._latest_snapshot = ActivationSnapshot(
                        layers={}
                    )
            else:
                self._latest_snapshot = self._build_snapshot(result)
                self._ever_seen = True

            envelope = self.make_snapshot(
                ok=True,
                message="sampled successfully",
                source="activation_memory",
                data=self._latest_snapshot.__dict__,
            )
            return self.snapshot_dict(envelope)

        except Exception as e:
            self.logger.error(f"[TraceML] ActivationMemorySampler.sample() error: {e}")
            envelope = self.make_snapshot(
                ok=False,
                message=f"sampling failed: {e}",
                source="activation_memory",
                data=None,
            )
            return self.snapshot_dict(envelope)

    def get_summary(self) -> Dict[str, Any]:
        pass