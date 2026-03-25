"""Heuristic recommendation engine for TraceML.

Usage::

    from traceml.heuristics.engine import build_recommendations
    recs = build_recommendations(code_manifest, system_manifest)
"""

from traceml.heuristics._types import Recommendation
from traceml.heuristics.engine import build_recommendations

__all__ = ["Recommendation", "build_recommendations"]
