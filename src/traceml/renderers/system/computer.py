"""
Compatibility façade for system telemetry compute.

This wrapper preserves the original high-level calling style while delegating
actual work to the split CLI and dashboard compute classes.

Example
-------
    computer = SystemMetricsComputer(db_path="metrics.db", rank=None)
    cli_payload = computer.compute_cli()
    dash_payload = computer.compute_dashboard(window_n=100)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .cli_compute import SystemCLIComputer
from .dashboard_compute import SystemDashboardComputer


class SystemMetricsComputer:
    """
    Compatibility façade over split CLI and dashboard compute classes.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    rank:
        Optional rank filter.
    stale_ttl_s:
        Maximum age in seconds for stale cached payload reuse.
    """

    def __init__(
        self,
        db_path: str,
        rank: Optional[int] = None,
        stale_ttl_s: Optional[float] = 30.0,
    ) -> None:
        self._cli = SystemCLIComputer(
            db_path=db_path,
            rank=rank,
            stale_ttl_s=stale_ttl_s,
        )
        self._dash = SystemDashboardComputer(
            db_path=db_path,
            rank=rank,
            stale_ttl_s=stale_ttl_s,
        )

    def compute_cli(self) -> Dict[str, Any]:
        """
        Compute the latest CLI system snapshot.
        """
        return self._cli.compute()

    def compute_dashboard(self, window_n: int = 100) -> Dict[str, Any]:
        """
        Compute dashboard rollups and short history series.
        """
        return self._dash.compute(window_n=window_n)
