from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import time


@dataclass
class SampleSnapshot:
    ok: bool
    message: str
    ts: float = field(default_factory=time.time)
    source: str = ""
    data: Optional[dict] = None


class BaseSampler(ABC):
    """
    Abstract base class for samplers that monitor runtime metrics,
    such as CPU usage, tensor allocations, or custom events.

    Samplers may be stateful and are typically polled periodically.
    """

    def __init__(self):
        # Optional: Initialize common sampler-level properties or perform global setup
        pass

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

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """
        Compute and return summary statistics for the collected metrics over the sampling period.

        Returns:
            Dict[str, Any]: Summary statistics. Should return an empty or error
                            dict if no data or calculation fails.
        """
        pass
