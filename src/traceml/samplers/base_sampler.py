from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List, Deque
from collections import deque
import time
from traceml.database.database import Database


@dataclass
class SampleSnapshot:
    """Represents one sampling event stored by the sampler."""

    ts: float = field(default_factory=time.time)
    ok: bool = True
    message: str = ""
    source: str = ""
    data: Optional[dict] = None


class BaseSampler(ABC):
    """
    Abstract base class for samplers that monitor runtime metrics,
    such as CPU usage, tensor allocations, or custom events.

    Samplers may be stateful and are typically polled periodically.
    """

    def __init__(self, max_snapshots: int = 10_000):
        # Optional: Initialize common sampler-level properties or perform global setup
        self._history: Deque[SampleSnapshot] = deque(maxlen=max_snapshots)
        self.db = Database()

    def _store_snapshot(
        self, ok: bool, message: str, source: str, data: Optional[dict] = None
    ):
        """Create and append a snapshot to history."""
        snap = SampleSnapshot(ok=ok, message=message, source=source, data=data)
        with self._lock:
            self._history.append(snap)

    def get_latest_snapshot(self) -> Optional[SampleSnapshot]:
        """Return the most recent snapshot."""
        with self._lock:
            return self._history[-1] if self._history else None

    def get_snapshot_history(self) -> List[SampleSnapshot]:
        """Return all stored snapshots."""
        with self._lock:
            return list(self._history)

    @staticmethod
    def make_snapshot(
        ok: bool, message: str, source: str, data: Optional[dict] = None
    ) -> SampleSnapshot:
        return SampleSnapshot(ok=ok, message=message, source=source, data=data)

    @staticmethod
    def snapshot_dict(snapshot: SampleSnapshot) -> Dict[str, Any]:
        return asdict(snapshot)

    @abstractmethod
    def sample(self) -> Dict[str, Any]:
        """
        Collect the latest data point(s) and return a dict envelope:
        {
          "ok": bool,
          "message": str,
          "ts": float,
          "source": str,
          "data": Optional[dict]   # payload with fields specific to the sampler
        }
        This method should be non-blocking.
        """
        pass

    def get_summary(self) -> Dict[str, Any]:
        """
        Compute and return summary statistics for the collected metrics over the sampling period.

        Returns:
            Dict[str, Any]: Summary statistics. Should return an empty or error
                            dict if no data or calculation fails.
        """
        pass
