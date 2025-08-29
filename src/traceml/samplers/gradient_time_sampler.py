import sys
import time
from dataclasses import dataclass
from collections import deque
from queue import Empty
from typing import Any, Deque, Dict, List, Optional, Tuple

from .base_sampler import BaseSampler
from traceml.utils.gradient_time import gradient_time_queue, GradientTimeEvent


@dataclass
class GradientTimeSnapshot:
    timestamp: float = 0.0
    backward_time: float = 0.0
    optimizer_time: float = 0.0
    total_time: float = 0.0
    label: Optional[str] = None
    drained_events: int = 0
    stale: bool = False
    error: Optional[str] = None


class GradientTimeSampler(BaseSampler):
    """
    Sampler that tracks timing of backward/optimizer steps.
    """

    def __init__(self, max_history: int = 10_000):
        super().__init__()
        self._history: Deque[GradientTimeSnapshot] = deque(maxlen=max_history)
        self._latest_snapshot: Optional[GradientTimeSnapshot] = None
        self._ever_seen: bool = False

        # Cumulative counters
        self._cumulative_count: int = 0
        self._cumulative_backward: float = 0.0
        self._cumulative_optimizer: float = 0.0
        self._cumulative_total: float = 0.0

    def _drain_queue(self) -> Tuple[int, List[GradientTimeEvent]]:
        drained = []
        while True:
            try:
                ev = gradient_time_queue.get_nowait()
            except Empty:
                break
            except Exception as e:
                print(
                    f"[TraceML] WARNING: gradient_time_queue.get_nowait failed: {e}",
                    file=sys.stderr,
                )
                break
            drained.append(ev)
        return len(drained), drained

    def _build_snapshot(self, events: List[GradientTimeEvent]) -> GradientTimeSnapshot:
        if not events:
            return GradientTimeSnapshot(
                timestamp=time.time(),
                drained_events=0,
                stale=True,
            )

        avg_backward = sum(e.backward_time for e in events) / len(events)
        avg_optimizer = sum(e.optimizer_time for e in events) / len(events)
        avg_total = sum(e.total_time for e in events) / len(events)
        label = events[-1].label if events else None

        snap = GradientTimeSnapshot(
            timestamp=time.time(),
            backward_time=round(avg_backward, 6),
            optimizer_time=round(avg_optimizer, 6),
            total_time=round(avg_total, 6),
            label=label,
            drained_events=len(events),
            stale=False,
        )
        return snap

    def sample(self) -> Dict[str, Any]:
        try:
            drained_count, events = self._drain_queue()
            if drained_count == 0 and self._ever_seen and self._latest_snapshot:
                # stale snapshot
                snap = GradientTimeSnapshot(
                    timestamp=time.time(),
                    backward_time=self._latest_snapshot.backward_time,
                    optimizer_time=self._latest_snapshot.optimizer_time,
                    total_time=self._latest_snapshot.total_time,
                    label=self._latest_snapshot.label,
                    drained_events=0,
                    stale=True,
                )
                self._latest_snapshot = snap
            else:
                snap = self._build_snapshot(events)
                self._latest_snapshot = snap
                self._ever_seen = True

                # Update cumulative
                self._cumulative_count += drained_count
                self._cumulative_backward += sum(e.backward_time for e in events)
                self._cumulative_optimizer += sum(e.optimizer_time for e in events)
                self._cumulative_total += sum(e.total_time for e in events)

                self._history.append(snap)

            ok = self._latest_snapshot.error is None
            msg = (
                "sampled successfully"
                if ok
                else f"sampling error: {self._latest_snapshot.error}"
            )
            envelope = self.make_snapshot(
                ok=ok,
                message=msg,
                source="gradient_time",
                data=self._latest_snapshot.__dict__,
            )
            return self.snapshot_dict(envelope)

        except Exception as e:
            print(f"[TraceML] GradientTimeSampler.sample() error: {e}", file=sys.stderr)
            snap = GradientTimeSnapshot(timestamp=time.time(), error=str(e))
            envelope = self.make_snapshot(
                ok=False,
                message=f"sampling failed: {e}",
                source="gradient_time",
                data=snap.__dict__,
            )
            return self.snapshot_dict(envelope)

    def get_summary(self) -> Dict[str, Any]:
        try:
            avg_backward = (
                self._cumulative_backward / self._cumulative_count
                if self._cumulative_count
                else 0.0
            )
            avg_optimizer = (
                self._cumulative_optimizer / self._cumulative_count
                if self._cumulative_count
                else 0.0
            )
            avg_total = (
                self._cumulative_total / self._cumulative_count
                if self._cumulative_count
                else 0.0
            )

            return {
                "ever_seen": self._ever_seen,
                "total_steps": self._cumulative_count,
                "avg_backward_time": round(avg_backward, 6),
                "avg_optimizer_time": round(avg_optimizer, 6),
                "avg_total_time": round(avg_total, 6),
                "last_snapshot": self._latest_snapshot,
            }
        except Exception as e:
            print(
                f"[TraceML] GradientTimeSampler.get_summary() error: {e}",
                file=sys.stderr,
            )
            return {
                "error": str(e),
                "ever_seen": self._ever_seen,
                "total_steps": self._cumulative_count,
                "last_snapshot": None,
            }
