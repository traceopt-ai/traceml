"""Script Hints renderer — surfaces heuristic recommendations at end of run.

Reads code_manifest.json and system_manifest.json once at startup, runs the
heuristics engine, then prints a plain-text card (matching the +---+ style of
the system/process/step summary cards) after the Live display exits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from traceml.heuristics._types import Recommendation
from traceml.heuristics.engine import build_recommendations

_WIDTH = 78
_INNER = _WIDTH - 4
_SEVERITY_PREFIX = {"crit": "[CRIT]", "warn": "[WARN]", "info": "[INFO]"}


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _border() -> str:
    return "+" + "-" * (_WIDTH - 2) + "+"


def _row(text: str = "") -> str:
    return f"|  {text:<{_INNER}}|"


def _wrap(text: str, indent: int = 0) -> List[str]:
    """Word-wrap text to fit inside the card inner width."""
    available = _INNER - indent
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > available:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".lstrip()
    if current:
        lines.append(current)
    prefix = " " * indent
    return [f"{prefix}{line}" for line in lines]


def _print_hints_card(recs: List[Recommendation]) -> None:
    header = f"TraceML Script Hints | {len(recs)} recommendation{'s' if len(recs) != 1 else ''}"
    lines = [_border(), _row(header), _border(), _row("HINTS"), _row()]

    for i, rec in enumerate(recs):
        prefix = _SEVERITY_PREFIX.get(rec.severity, "[INFO]")
        # Reason line(s)
        reason_lines = _wrap(f"{prefix} {rec.reason}")
        for j, line in enumerate(reason_lines):
            lines.append(_row(line))
        # Action line(s) indented with arrow
        action_lines = _wrap(f"→ {rec.action}", indent=2)
        for line in action_lines:
            lines.append(_row(line))
        if i < len(recs) - 1:
            lines.append(_row())

    lines += [_border()]
    print("\n".join(lines))


def _write_recommendations_json(
    recs: List[Recommendation], dest_path: Path
) -> None:
    payload = [
        {
            "kind": r.kind,
            "severity": r.severity,
            "category": r.category,
            "reason": r.reason,
            "action": r.action,
        }
        for r in recs
    ]
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


class CodeHintsRenderer:
    """Loads manifests, runs heuristics once, prints the Script Hints card.

    Lifecycle: ``start()`` computes recommendations; ``stop()`` prints the
    card (after Live exits) and writes recommendations.json.
    """

    def __init__(self, aggregator_dir: Path) -> None:
        self._aggregator_dir = Path(aggregator_dir)
        self._recs: List[Recommendation] = []

    def start(self) -> None:
        code_manifest_path = os.environ.get("TRACEML_CODE_MANIFEST_PATH", "")
        system_manifest_path = str(
            self._aggregator_dir / "system_manifest.json"
        )
        code_manifest = _load_json(code_manifest_path)
        system_manifest = _load_json(system_manifest_path)
        if code_manifest.get("error"):
            return
        self._recs = build_recommendations(code_manifest, system_manifest)

    def stop(self) -> None:
        """Print the card to stdout after Live exits, then persist JSON."""
        if not self._recs:
            return
        _print_hints_card(self._recs)
        _write_recommendations_json(
            self._recs, self._aggregator_dir / "recommendations.json"
        )
