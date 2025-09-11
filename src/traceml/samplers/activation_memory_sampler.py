from dataclasses import dataclass, field
from collections import defaultdict, deque
from queue import Empty
from typing import Any, Deque, Dict, List, Optional, Tuple

import time
import torch

from .base_sampler import BaseSampler
from traceml.utils.activation_hook import get_activation_queue
from traceml.loggers.error_log import get_error_logger, setup_error_logger


@dataclass
class _BatchStats:
    count: int = 0
    sum_memory: float = 0.0
    avg_memory: float = 0.0
    max_memory: float = 0.0
    min_nonzero_memory: Optional[float] = None


@dataclass
class ActivationSnapshot:
    timestamp: float = 0.0
    devices: Dict[str, Any] = field(default_factory=dict)
    overall_avg_memory: float = 0.0
    drained_events: int = 0
    stale: bool = False
    error: Optional[str] = None


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
        self._raw_events: Deque[Dict[str, Any]] = deque(maxlen=int(max_raw_events))
        self._cumulative: Dict[str, Tuple[int, float, float]] = defaultdict(
            lambda: (0, 0.0, 0.0)
        )
        self._latest_snapshot: Optional[ActivationSnapshot] = None
        self._ever_seen: bool = False

    def _append_raw_event(self, ts: float, per_dev_memory: Dict[str, float]) -> None:
        """Push one raw event into the bounded buffer."""
        self._raw_events.append(
            {"ts": float(ts), "per_dev_memory": dict(per_dev_memory)}
        )

    def _accumulate_cumulative(self, per_dev_memory: Dict[str, float]) -> None:
        """Update cumulative counters for each device."""
        for dev, mem in per_dev_memory.items():
            c_count, c_sum, c_max = self._cumulative[dev]
            mb_f = float(mem)
            self._cumulative[dev] = (c_count + 1, c_sum + mb_f, max(c_max, mb_f))

    @staticmethod
    def _compute_batch_stats(values: List[float]) -> _BatchStats:
        """Compute summary stats."""
        if not values:
            return _BatchStats()
        sum_memory = float(sum(values))
        max_memory = float(max(values))
        non_zero_memory = [v for v in values if v > 0.0]
        min_nonzero_memory = float(min(non_zero_memory)) if non_zero_memory else None
        return _BatchStats(
            count=len(values),
            sum_memory=sum_memory,
            avg_memory=sum_memory / len(values),
            max_memory=max_memory,
            min_nonzero_memory=min_nonzero_memory,
        )

    def _pressure_flag(self, dev: str, batch_max_memory: float) -> Optional[bool]:
        """True if batch max exceeds threshold of device capacity; None if unknown/not CUDA."""
        if not dev.startswith("cuda"):
            return None
        try:
            idx = int(dev.split(":", 1)[1])
            props = torch.cuda.get_device_properties(idx)
            total_memory = props.total_memory / (1024**2)
            return (batch_max_memory / total_memory) >= self.pressure_threshold
        except Exception:
            return None

    def _drain_queue(self) -> Tuple[int, Dict[str, List[float]]]:
        """
        Drain the activation queue completely and return:
          - number of events drained
          - mapping dev -> list of values in THIS drain cycle
        """
        try:
            q = get_activation_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] ERROR: activation queue unavailable: {e}")
            return 0, {}

        drained_events = 0
        batch_per_dev: Dict[str, List[float]] = defaultdict(list)
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
            ts = getattr(ev, "timestamp", now)

            self._append_raw_event(ts, per_dev)
            self._accumulate_cumulative(per_dev)

            for dev, mem in per_dev.items():
                batch_per_dev[dev].append(float(mem))

        return drained_events, batch_per_dev

    def _build_snapshot(
        self, drained_events: int, batch_per_dev: Dict[str, List[float]]
    ) -> ActivationSnapshot:
        """Construct the live snapshot dict from this drain’s per-device values."""
        devices_out: Dict[str, Any] = {}
        overall_avg = 0.0
        n_devs = 0

        for dev, vals in batch_per_dev.items():
            stats = self._compute_batch_stats(vals)
            pressure = self._pressure_flag(dev, stats.max_memory)

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
                "pressure_90pct": pressure,
            }
            overall_avg += stats.avg_memory
            n_devs += 1

        return ActivationSnapshot(
            timestamp=time.time(),
            devices=devices_out,
            overall_avg_memory=round(overall_avg / n_devs, 4) if n_devs else 0.0,
            drained_events=drained_events,
            stale=False,
        )

    def sample(self) -> Dict[str, Any]:
        """
        Drain the queue and compute a live snapshot.
        If nothing new arrived:
          - If we’ve seen data before: return last snapshot with `stale=True`.
          - If we’ve never seen data: return a guidance note about attaching hooks.
        """
        try:
            drained_events, batch_per_dev = self._drain_queue()
            if drained_events == 0:
                # nothing new then return the last snapshot
                if self._ever_seen and self._latest_snapshot:
                    snap = ActivationSnapshot(
                        timestamp=self._latest_snapshot.timestamp,
                        devices=dict(self._latest_snapshot.devices),
                        overall_avg_memory=self._latest_snapshot.overall_avg_memory,
                        drained_events=0,
                        stale=True,
                    )
                    self._latest_snapshot = snap
                else:
                    # First-time snapshot
                    self._latest_snapshot = ActivationSnapshot(
                        timestamp=time.time(),
                        devices={},
                        overall_avg_memory=0.0,
                        drained_events=0,
                        stale=True,
                    )
            else:
                self._latest_snapshot = self._build_snapshot(
                    drained_events, batch_per_dev
                )
                self._ever_seen = True

            ok = self._latest_snapshot.error is None
            msg = (
                "sampled successfully"
                if ok
                else f"sampling completed with error: {self._latest_snapshot.error}"
            )
            envelope = self.make_snapshot(
                ok=ok,
                message=msg,
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
        """
        Summarize all drained data so far using cumulative counters.
        Returns a dict
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
                "raw_events_kept": len(self._raw_events),
                "last_snapshot": self._latest_snapshot.__dict__,
            }

        except Exception as e:
            self.logger.error(
                f"[TraceML] ActivationMemorySampler.get_summary() error: {e}"
            )
            return {
                "error": str(e),
                "ever_seen": self._ever_seen,
                "per_device_cumulative": {},
                "raw_events_kept": len(self._raw_events),
                "last_snapshot": None,
            }
