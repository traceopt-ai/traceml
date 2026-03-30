"""Heuristic recommendation engine.

Accepts pre-parsed manifest dicts (no I/O, no torch imports) and returns a
deduplicated, severity-sorted list of Recommendation objects.
"""

from __future__ import annotations

from typing import Any, Dict, List

from traceml.heuristics._types import Recommendation, Severity  # noqa: F401
from traceml.heuristics.rules.dataloader import check_dataloader
from traceml.heuristics.rules.distributed import check_distributed
from traceml.heuristics.rules.precision import check_precision
from traceml.heuristics.rules.sync_transfer import check_sync_transfer
from traceml.heuristics.rules.train_loop import check_train_loop

_SEVERITY_ORDER = {"crit": 0, "warn": 1, "info": 2}


def build_recommendations(
    code_manifest: Dict[str, Any],
    system_manifest: Dict[str, Any],
) -> List[Recommendation]:
    """Run all rule modules and return deduplicated, severity-sorted recommendations.

    Parameters
    ----------
    code_manifest:
        Dict produced by ``build_code_manifest()``.
    system_manifest:
        Dict produced by ``save_system_manifest()`` in aggregator_main.
    """
    all_recs: List[Recommendation] = []
    for check_fn in (
        check_dataloader,
        check_precision,
        check_distributed,
        check_sync_transfer,
        check_train_loop,
    ):
        try:
            all_recs.extend(check_fn(code_manifest, system_manifest))
        except Exception:
            # Individual rule failures must never block other rules or the run.
            pass

    # Deduplicate by kind (first occurrence wins — most specific check runs first)
    seen: set[str] = set()
    unique: List[Recommendation] = []
    for rec in all_recs:
        if rec.kind not in seen:
            seen.add(rec.kind)
            unique.append(rec)

    unique.sort(key=lambda r: _SEVERITY_ORDER.get(r.severity, 99))
    return unique
