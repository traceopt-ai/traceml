"""The terminal snapshot contract for process telemetry.

The SQLite reads moved to ``repository.py`` and the dashboard payload to
``dashboard_models.py``; what remains here is the shape the terminal card
consumes.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from traceml_ai.renderers.shared.freshness import RankReporting


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
    # Every rank's Process reporting status, so the card can name a rank
    # that stopped sending Process data instead of silently holding the
    # slowest rank's last seq. None when this tick's read failed.
    rank_reporting: Optional[Tuple[RankReporting, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "cpu_used": self.cpu_used,
            "gpu_used": self.gpu_used,
            "gpu_reserved": self.gpu_reserved,
            "gpu_total": self.gpu_total,
            "gpu_rank": self.gpu_rank,
            "gpu_used_imbalance": self.gpu_used_imbalance,
            "rank_reporting": rank_reporting_dicts(self.rank_reporting),
        }


def rank_reporting_dicts(
    reporting: Optional[Tuple[RankReporting, ...]],
) -> Optional[List[Dict[str, Any]]]:
    """The snapshot's ``rank_reporting`` value: plain dicts, or ``None``."""
    if reporting is None:
        return None
    return [asdict(r) for r in reporting]
