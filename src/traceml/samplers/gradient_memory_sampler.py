from dataclasses import dataclass, field
from collections import defaultdict, deque
from queue import Empty
from typing import Any, Deque, Dict, List, Optional, Tuple
import time
import torch
from numpy.distutils.cpuinfo import DarwinCPUInfo

from .base_sampler import BaseSampler
from traceml.utils.gradient_hook import get_gradient_queue
from traceml.loggers.error_log import get_error_logger, setup_error_logger


@dataclass
class _BatchStats:
    count: int = 0
    sum_memory: float = 0.0
    avg_memory: float = 0.0
    max_memory: float = 0.0
    min_nonzero_memory: Optional[float] = None


@dataclass
class GradientSnapshot:
    """
    Summary-only snapshot:
      - devices: device-level stats (count/sum/avg/max/min>0/pressure)
      - layers:  {layer_name: {device: mb_total}}
      - params:  {param_name: {device: mb_total}}
    """
    devices: Dict[str, Any] = field(default_factory=dict)   # per device stats
    layers: Dict[str, Dict[str, float]] = field(default_factory=dict)  # per-layer totals
    overall_avg_memory: float = 0.0
    drained_events: int = 0
    stale: bool = False


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

        self._raw_events: Deque[Dict[str, Any]] = deque(maxlen=int(max_raw_events))
        self._cumulative: Dict[str, Tuple[int, float, float]] = defaultdict(
            lambda: (0, 0.0, 0.0)
        )
        # layer_name -> global peak MB
        self._cumulative_layer_peaks: Dict[str, float] = {}
        # layer_name -> {param_name -> global peak MB}
        self._cumulative_param_peaks: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._latest_snapshot: Optional[GradientSnapshot] = None
        self._ever_seen: bool = False

    def _append_raw_event(
        self, ts: float, per_dev_memory: Dict[str, float], layer: str, param: Optional[str]
    ) -> None:
        self._raw_events.append({
                "ts": float(ts),
                "per_dev_memory": dict(per_dev_memory),
                "layer": layer,
                "param": param,
            }
        )

    def _accumulate_cumulative(self, per_dev_memory: Dict[str, float]) -> None:
        for dev, mem in per_dev_memory.items():
            c_count, c_sum, c_max = self._cumulative[dev]
            m = float(mem)
            self._cumulative[dev] = (c_count + 1, c_sum + m, max(c_max, m))

    @staticmethod
    def _compute_batch_stats(values: List[float]) -> _BatchStats:
        if not values:
            return _BatchStats()
        s = float(sum(values))
        mx = float(max(values))
        nz = [v for v in values if v > 0.0]
        mnz = float(min(nz)) if nz else None
        return _BatchStats(
            count=len(values),
            sum_memory=s,
            avg_memory=s / len(values) if values else 0.0,
            max_memory=mx,
            min_nonzero_memory=mnz,
        )

    def _pressure_flag(self, dev: str, batch_max_memory: float) -> Optional[bool]:
        """True if batch max exceeds threshold of device capacity; None if unknown/not CUDA."""
        if not dev.startswith("cuda"):
            return None
        try:
            idx = int(dev.split(":", 1)[1])
            props = torch.cuda.get_device_properties(idx)
            total_memory = props.total_memory
            return (batch_max_memory / total_memory) >= self.pressure_threshold
        except Exception:
            return None

    def _process_event(
            self,
            ev,
            batch_per_dev: Dict[str, List[float]],
            batch_per_layer: Dict[str, List[float]],
            batch_per_param: Dict[str, Dict[str, List[float]]]
    ) -> None:
        """Update per-device, per-layer, per-param accumulators from one gradient event."""
        ts = getattr(ev, "timestamp", 0.0)
        per_dev = getattr(ev, "per_device_memory", {}) or {}
        layer = getattr(ev, "layer_name", "unknown")
        param = getattr(ev, "param_name", None)

        self._append_raw_event(ts, per_dev, layer, param)
        self._accumulate_cumulative(per_dev)

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
        batch_per_param: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        while True:
            try:
                elem = q.get_nowait()
            except Empty:
                break
            except Exception as e:
                self.logger.error(f"[TraceML] ERROR: unexpected error: {e}")
                break

            drained_events += 1
            self._process_event(elem, batch_per_device, batch_per_layer, batch_per_param)

        return DrainResult(drained_events, batch_per_device, batch_per_layer, batch_per_param)

    def _build_device_stats(
        self,
        per_device: Dict[str, List[float]]
    ) -> Tuple[Dict[str, Any], float, int]:
        """Aggregate device-level stats and return (devices_out, total_avg, n_devices)."""
        devices_out: Dict[str, Any] = {}
        overall_avg, n_devs = 0.0, 0

        for dev, vals in per_device.items():
            stats = self._compute_batch_stats(vals)
            devices_out[dev] = {
                "count": stats.count,
                "sum_memory": round(stats.sum_memory, 4),
                "avg_memory": round(stats.avg_memory, 4),
                "max_memory": round(stats.max_memory, 4),
                "min_nonzero_memory": (
                    round(stats.min_nonzero_memory, 4)
                    if stats.min_nonzero_memory is not None
                    else None
                ),
                "pressure_90pct": self._pressure_flag(dev, stats.max_memory),
            }
            overall_avg += stats.avg_memory
            n_devs += 1

        return devices_out, overall_avg, n_devs


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
                prev_p_global = float(self._cumulative_param_peaks[layer].get(pname, 0.0))
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
        devices_out, overall_avg, n_devs = self._build_device_stats(result.per_device)
        layers_out = self._build_layer_stats(result.per_layer, result.per_param)

        return GradientSnapshot(
            devices=devices_out,
            layers=layers_out,
            overall_avg_memory=round(overall_avg / n_devs, 4) if n_devs else 0.0,
            drained_events=result.drained_events,
            stale=False,
        )

    def sample(self) -> Dict[str, Any]:
        try:
            result = self._drain_queue()
            if result.drained_events == 0:
                if self._ever_seen and self._latest_snapshot:
                    snap = GradientSnapshot(
                        timestamp=self._latest_snapshot.timestamp,
                        devices=dict(self._latest_snapshot.devices),
                        layers=dict(self._latest_snapshot.layers),
                        overall_avg_memory=self._latest_snapshot.overall_avg_memory,
                        drained_events=0,
                        stale=True,
                    )
                    self._latest_snapshot = snap
                else:
                    self._latest_snapshot = GradientSnapshot(
                        timestamp=time.time(),
                        devices={},
                        layers={},
                        overall_avg_memory=0.0,
                        drained_events=0,
                        stale=True,
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
        """
        Summarize all drained data so far.
        Returns cumulative device stats + global peaks for layers/params.
        """
        try:
            per_dev_summary: Dict[str, Any] = {}
            for dev, (c_count, c_sum, c_max) in self._cumulative.items():
                avg = (c_sum / c_count) if c_count else 0.0
                per_dev_summary[dev] = {
                    "cumulative_count": c_count,
                    "cumulative_sum_memory": round(c_sum, 4),
                    "cumulative_avg_memory": round(avg, 4),
                    "cumulative_max_memory": round(c_max, 4),
                }

            return {
                "ever_seen": self._ever_seen,
                "per_device_cumulative": per_dev_summary,
                "layer_global_peaks": dict(self._cumulative_layer_peaks),
                "param_global_peaks": {
                    lname: dict(pmap) for lname, pmap in self._cumulative_param_peaks.items()
                },
                "raw_events_kept": len(self._raw_events),
                "last_snapshot": self._latest_snapshot.__dict__ if self._latest_snapshot else None,
            }

        except Exception as e:
            self.logger.error(f"[TraceML] GradientMemorySampler.get_summary() error: {e}")
            return {
                "error": str(e),
                "ever_seen": self._ever_seen,
                "per_device_cumulative": {},
                "layer_global_peaks": {},
                "param_global_peaks": {},
                "raw_events_kept": 0,
                "last_snapshot": None,
            }
