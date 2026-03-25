"""Script Hints renderer — surfaces heuristic recommendations from code_manifest.json.

Reads code_manifest.json and system_manifest.json at startup, runs the
heuristics engine once, renders a 'Script Hints' Rich panel in the CLI output,
and writes recommendations.json to the aggregator directory at shutdown.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from traceml.heuristics._types import Recommendation
from traceml.heuristics.engine import build_recommendations

_SEVERITY_STYLE = {
    "crit": ("bold red", "✖"),
    "warn": ("bold yellow", "⚠"),
    "info": ("bold blue", "●"),
}


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _render_hints_panel(recs: List[Recommendation], console: Console) -> None:
    if not recs:
        return

    body = Text()
    for rec in recs:
        style, icon = _SEVERITY_STYLE.get(rec.severity, ("", "•"))
        body.append(f"{icon} [{rec.kind}] ", style=style)
        body.append(f"{rec.reason}\n", style="")
        body.append(f"  → {rec.action}\n\n", style="dim")

    console.print(
        Panel(
            body,
            title="[bold]Script Hints[/bold]",
            border_style="blue",
            padding=(0, 1),
        )
    )


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
    """Loads manifests, runs heuristics once, renders the Script Hints panel.

    Lifecycle: call ``start()`` after aggregator startup, ``stop()`` before shutdown.
    """

    def __init__(
        self,
        aggregator_dir: Path,
        console: Console,
    ) -> None:
        self._aggregator_dir = Path(aggregator_dir)
        self._console = console
        self._recs: List[Recommendation] = []

    def start(self) -> None:
        code_manifest_path = os.environ.get("TRACEML_CODE_MANIFEST_PATH", "")
        system_manifest_path = str(
            self._aggregator_dir / "system_manifest.json"
        )

        code_manifest = _load_json(code_manifest_path)
        system_manifest = _load_json(system_manifest_path)

        if code_manifest.get("error"):
            return  # AST parse failed — skip hints silently

        self._recs = build_recommendations(code_manifest, system_manifest)

    def stop(self) -> None:
        """Print the hints panel after the Live display exits and write JSON."""
        if not self._recs:
            return
        # Create a fresh console pointing to stdout — Live is already stopped
        # so we can write directly to the terminal without conflict.
        console = Console()
        _render_hints_panel(self._recs, console)
        _write_recommendations_json(
            self._recs, self._aggregator_dir / "recommendations.json"
        )
