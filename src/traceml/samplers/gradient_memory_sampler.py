from dataclasses import dataclass, field
from collections import defaultdict
from queue import Empty
from typing import Any, Dict, List, Optional

from .base_sampler import BaseSampler
from traceml.utils.gradient_hook import get_gradient_queue
from traceml.loggers.error_log import get_error_logger, setup_error_logger


@dataclass
class GradientSnapshot:
    """
    Summary-only snapshot:
      - devices: device-level stats (count/sum/avg/max/min>0/pressure)
      - layers:  {layer_name: {device: mb_total}}
      - params:  {param_name: {device: mb_total}}
    """

    devices: Dict[str, Any] = field(default_factory=dict)  # per device stats
    layers: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )  # per-layer totals


@dataclass
class DrainResult:
    drained_events: int
    per_device: Dict[str, List[float]]
    per_layer: Dict[str, List[float]]
    per_param: Dict[str, Dict[str, List[float]]]


class GradientMemorySampler(BaseSampler):
    """
    Drain-all gradient-event sampler.

    Each call to `sample()`:
      - Drains the gradient queue.
      - Aggregates per-device stats over new events.
      - Aggregates per-layer (module) + per-param totals.
      - Tracks current + global peaks.
      - Returns a live snapshot dict.
    """

    def __init__(
        self,
        max_raw_events: int = 10_000,
        pressure_threshold: float = 0.9,
    ):
        super().__init__()
        setup_error_logger()
        self.logger = get_error_logger("GradientMemorySampler")

        self.pressure_threshold = float(pressure_threshold)

        # layer_name -> global peak MB
        self._cumulative_layer_peaks: Dict[str, float] = {}
        # layer_name -> {param_name -> global peak MB}
        self._cumulative_param_peaks: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._latest_snapshot: Optional[GradientSnapshot] = None
        self._ever_seen: bool = False

    def _process_event(
        self,
        ev,
        batch_per_dev: Dict[str, List[float]],
        batch_per_layer: Dict[str, List[float]],
        batch_per_param: Dict[str, Dict[str, List[float]]],
    ) -> None:
        """Update per-device, per-layer, per-param accumulators from one gradient event."""
        per_dev = getattr(ev, "per_device_memory", {}) or {}
        layer = getattr(ev, "layer_name", "unknown")
        param = getattr(ev, "param_name", None)

        for dev, mem in per_dev.items():
            batch_per_dev[dev].append(float(mem))
            batch_per_layer[layer].append(float(mem))
            if param:
                batch_per_param[layer][param].append(float(mem))

    def _drain_queue(
        self,
    ) -> DrainResult:
        """Drain the gradient queue completely and return per-device, per-layer, per-param stats."""
        try:
            q = get_gradient_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] ERROR: gradient queue unavailable: {e}")
            return DrainResult(0, {}, {}, {})

        drained_events = 0
        batch_per_device: Dict[str, List[float]] = defaultdict(list)
        batch_per_layer: Dict[str, List[float]] = defaultdict(list)
        batch_per_param: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        while True:
            try:
                elem = q.get_nowait()
            except Empty:
                break
            except Exception as e:
                self.logger.error(f"[TraceML] ERROR: unexpected error: {e}")
                break

            drained_events += 1
            self._process_event(
                elem, batch_per_device, batch_per_layer, batch_per_param
            )

        return DrainResult(
            drained_events, batch_per_device, batch_per_layer, batch_per_param
        )

    def _build_layer_stats(
        self,
        per_layer: Dict[str, List[float]],
        per_param: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate per-layer + per-param peaks, updating cumulative peaks."""
        layers_out: Dict[str, Dict[str, Any]] = {}

        for layer, vals in per_layer.items():
            if not vals:
                continue
            curr_peak = float(max(vals))
            prev_global = float(self._cumulative_layer_peaks.get(layer, 0.0))
            global_peak = max(curr_peak, prev_global)
            self._cumulative_layer_peaks[layer] = global_peak

            params_out: Dict[str, Dict[str, float]] = {}
            for pname, pvals in per_param.get(layer, {}).items():
                if not pvals:
                    continue
                p_curr_peak = float(max(pvals))
                prev_p_global = float(
                    self._cumulative_param_peaks[layer].get(pname, 0.0)
                )
                p_global = max(p_curr_peak, prev_p_global)
                self._cumulative_param_peaks[layer][pname] = p_global
                params_out[pname] = {
                    "current_peak": round(p_curr_peak, 4),
                    "global_peak": round(p_global, 4),
                }

            layers_out[layer] = {
                "current_peak": round(curr_peak, 4),
                "global_peak": round(global_peak, 4),
                "params": params_out,
            }

        return layers_out

    def _build_snapshot(self, result: DrainResult) -> GradientSnapshot:
        layers_out = self._build_layer_stats(result.per_layer, result.per_param)

        return GradientSnapshot(
            layers=layers_out,
        )

    def sample(self) -> Dict[str, Any]:
        try:
            result = self._drain_queue()
            if result.drained_events == 0:
                if self._ever_seen and self._latest_snapshot:
                    snap = GradientSnapshot(
                        devices=dict(self._latest_snapshot.devices),
                        layers=dict(self._latest_snapshot.layers),
                    )
                    self._latest_snapshot = snap
                else:
                    self._latest_snapshot = GradientSnapshot(
                        devices={},
                        layers={},
                    )
            else:
                self._latest_snapshot = self._build_snapshot(result)
                self._ever_seen = True

            envelope = self.make_snapshot(
                ok=True,
                message="sampled successfully",
                source="gradient_memory",
                data=self._latest_snapshot.__dict__,
            )
            return self.snapshot_dict(envelope)

        except Exception as e:
            self.logger.error(f"[TraceML] GradientMemorySampler.sample() error: {e}")
            envelope = self.make_snapshot(
                ok=False,
                message=f"sampling failed: {e}",
                source="gradient_memory",
                data=None,
            )
            return self.snapshot_dict(envelope)

    def get_summary(self) -> Dict[str, Any]:
        pass
