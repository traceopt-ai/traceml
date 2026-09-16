"""The terminal snapshot contract for process telemetry.

The SQLite reads moved to ``repository.py`` and the dashboard payload to
``dashboard_models.py``; what remains here is the shape the terminal card
consumes.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "cpu_used": self.cpu_used,
            "gpu_used": self.gpu_used,
            "gpu_reserved": self.gpu_reserved,
            "gpu_total": self.gpu_total,
            "gpu_rank": self.gpu_rank,
            "gpu_used_imbalance": self.gpu_used_imbalance,
        }
