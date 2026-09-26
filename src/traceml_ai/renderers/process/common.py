"""The terminal snapshot contract for process telemetry.

The SQLite reads moved to ``repository.py`` and the dashboard payload to
``dashboard_models.py``; what remains here is the shape the terminal card
consumes.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from traceml_ai.renderers.shared.freshness import RankLiveness


@dataclass(frozen=True)
class ProcessCLISnapshot:
    """Compact terminal snapshot for process telemetry."""

    seq: Optional[int]
    cpu_used: float
    gpu_used: Optional[float]
    gpu_reserved: Optional[float]
    gpu_total: Optional[float]
    gpu_rank: Optional[int]
    gpu_used_imbalance: Optional[float]
    # Every rank's last-seen clock, so the card can name a rank that
    # stopped instead of silently holding the slowest rank's last seq.
    # None when there is no verdict: this tick's read failed and no good
    # verdict is still inside the stale TTL.
    rank_liveness: Optional[Tuple[RankLiveness, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "cpu_used": self.cpu_used,
            "gpu_used": self.gpu_used,
            "gpu_reserved": self.gpu_reserved,
            "gpu_total": self.gpu_total,
            "gpu_rank": self.gpu_rank,
            "gpu_used_imbalance": self.gpu_used_imbalance,
            "rank_liveness": (
                None
                if self.rank_liveness is None
                else [asdict(r) for r in self.rank_liveness]
            ),
        }
