# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Context strip: which run this is, and how much of it is reporting.

The strip sits above every metric pane and carries identity (run name,
script, profile, strategy), a liveness word derived from data age, and a
coverage line in which every count is observed, never configured:
``ranks 3/4 reporting`` when a rank stops, not ``world_size 4``.

Build/update pair like every other section: ``build_context_section`` lays
out the strip from identity that is known at page build (settings + the
launcher's manifest); ``update_context_section`` fills the data-driven parts
from the CONTEXT payload (``renderers/context``) on every tick.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from nicegui import ui

from traceml_ai.renderers.context.common import empty_context

# Sections that only exist on a training run. A watch session has no step
# loop, so showing them there produces permanently dead panels (#355).
STEP_SECTIONS: Tuple[str, ...] = (
    "model_combined",
    "step_memory",
    "model_diagnostics",
)
RESOURCE_SECTIONS: Tuple[str, ...] = ("gpu_gauge", "system", "process")
RUN_SECTIONS: Tuple[str, ...] = (
    "model_combined",
    "gpu_gauge",
    "system",
    "process",
    "step_memory",
    "model_diagnostics",
)

# The facts a context payload carries (renderers/context); anything else
# handed to ``update_context_section`` is ignored rather than misread.
_CONTEXT_KEYS = frozenset(empty_context())

# Data older than this reads "stale"; same bar as the display-loop staleness
# chip (TRA-68), so the two indicators never disagree on what fresh means.
LIVE_THRESHOLD_S = 5.0

# Shape of a launcher-generated session id, e.g. session_20260821_123224_1103e8;
# used when a manifest without identity_source is replayed.
_GENERATED_ID = re.compile(r"^session_\d{8}_\d{6}_[0-9a-f]{6}$")


def sections_for_profile(profile: str) -> Tuple[str, ...]:
    """Sections a page shows for ``profile``: watch gets no step panels."""
    if str(profile or "run") == "watch":
        return RESOURCE_SECTIONS
    return RUN_SECTIONS


def format_elapsed(seconds: float) -> str:
    """``47s`` / ``2m 26s`` / ``3h 12m``."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def live_threshold_s(sampler_interval_s: Optional[float]) -> float:
    """Age below which data counts as live: 5 s, or 2.5 sampler ticks when
    the sampler is slower than that, so a slow sampler never reads stale
    between its own ticks."""
    try:
        interval = float(sampler_interval_s or 0.0)
    except (TypeError, ValueError):
        interval = 0.0
    return max(LIVE_THRESHOLD_S, 2.5 * interval)


def format_liveness(
    age_s: Optional[float], threshold_s: float = LIVE_THRESHOLD_S
) -> Tuple[str, str]:
    """(state word, detail) from the age of the newest sample."""
    if age_s is None:
        return "no data", "waiting for the first sample"
    age = max(0.0, float(age_s))
    if age < threshold_s:
        # No age on the fresh side: with a 2 s sampler it only cycles
        # 0/1/2 and reads as noise. The age matters once data has stopped.
        return "live", ""
    return "stale", f"{age:.0f}s since last data"


def strategy_token(value: Any) -> str:
    """``DDP`` / ``FSDP`` for a recorded strategy; "" when there is none to
    show (``distributed_unknown`` is a non-answer and ``single_process`` is
    the absence of one; neither earns a chip)."""
    text = str(value or "").strip().lower()
    if not text or "unknown" in text or text == "single_process":
        return ""
    return text.upper()


def format_coverage(ctx: Dict[str, Any]) -> str:
    """``ranks 3/4 reporting · 4 GPUs observed · 1 node · 2m 26s``.

    A part is omitted when its fact is absent; nothing is invented. The
    denominator is the configured world size, the numerator is observed.
    """
    parts = []
    expected = int(ctx.get("world_size") or 0)
    reporting = ctx.get("ranks_reporting")
    if reporting is not None and expected:
        parts.append(f"ranks {int(reporting)}/{expected} reporting")
    elif reporting is not None:
        parts.append(f"ranks {int(reporting)} reporting")
    gpus = ctx.get("gpus_observed")
    if gpus is None:
        gpus = ctx.get("gpu_count")
    gpus = int(gpus or 0)
    if gpus:
        parts.append(f"{gpus} GPU{'s' if gpus != 1 else ''} observed")
    nodes = ctx.get("node_count")
    if nodes:
        parts.append(f"{int(nodes)} node{'s' if int(nodes) != 1 else ''}")
    first = ctx.get("first_data_ts")
    last = ctx.get("last_data_ts")
    if first is not None and last is not None and last >= first:
        parts.append(format_elapsed(float(last) - float(first)))
    return " · ".join(parts)


def abbreviate_path(path: str, keep: int = 3) -> str:
    """Last ``keep`` components of a long path, prefixed with an ellipsis.

    The full path stays available (tooltip + click-to-copy in the strip);
    the label only has to identify the file at a glance.
    """
    if not path:
        return ""
    parts = [part for part in Path(path).parts if part not in ("/", "")]
    if len(parts) <= keep:
        return path
    return "…/" + "/".join(parts[-keep:])


def _session_root(
    logs_dir: str, session_id: str, db_path: str
) -> Optional[Path]:
    if logs_dir and session_id:
        return Path(logs_dir) / session_id
    # Replay or manual use: the database sits somewhere under the session
    # root, so look upward for the launcher's manifest.
    if db_path:
        here = Path(db_path).parent
        for _ in range(4):
            if (here / "manifest.json").is_file():
                return here
            if here.parent == here:
                break
            here = here.parent
    return None


def _load_manifest(root: Optional[Path]) -> Dict[str, Any]:
    if root is None:
        return {}
    try:
        with open(root / "manifest.json", "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _artifact_path(
    logs_dir: str, session_id: str, root: Optional[Path]
) -> str:
    if logs_dir and session_id:
        return str(Path(logs_dir) / session_id / "final_summary.json")
    if root is not None:
        target = root / "final_summary.json"
        try:
            rel = os.path.relpath(target)
        except ValueError:  # different drive on Windows
            return str(target)
        # A relative path that climbs out of cwd is unreadable; show the
        # absolute one instead, which is what a user would copy anyway.
        return str(target) if rel.startswith("..") else rel
    return ""


def resolve_run_identity(settings: Any) -> Dict[str, str]:
    """Identity known before any data arrives.

    run name and script come from the launcher's manifest when it exists
    (script as a basename only: a full path leaks directory layout on a
    shared screen), else from settings, else from the session directory.
    """
    profile = str(getattr(settings, "profile", "run") or "run")
    session_id = str(getattr(settings, "session_id", "") or "")
    logs_dir = str(getattr(settings, "logs_dir", "") or "")
    db_path = str(getattr(settings, "db_path", "") or "")
    root = _session_root(logs_dir, session_id, db_path)
    manifest = _load_manifest(root)
    run_block = manifest.get("run")
    run_block = run_block if isinstance(run_block, dict) else {}
    launch = manifest.get("launch")
    launch = launch if isinstance(launch, dict) else {}
    run_name = ""
    for candidate in (
        run_block.get("run_name"),
        manifest.get("session_id"),
        session_id,
        root.name if root is not None else "",
    ):
        text = str(candidate or "").strip()
        if text:
            run_name = text
            break
    script = str(launch.get("script_path") or "").strip()
    script_name = Path(script).name if script else ""
    source = str(run_block.get("identity_source") or "").strip()
    if not source and _GENERATED_ID.match(run_name):
        source = "generated"
    return {
        "profile": profile,
        "run_name": run_name,
        "run_name_source": source,
        "script_name": script_name,
        "artifact_path": _artifact_path(logs_dir, session_id, root),
    }


def run_name_label(identity: Dict[str, str]) -> Tuple[str, bool]:
    """(text, muted). A launcher-generated id is not a name the user chose,
    so the strip says so instead of showing the id as if it were one; the
    id itself stays visible on the artifact-path line."""
    if identity.get("run_name_source") == "generated":
        return "no run name", True
    name = identity.get("run_name") or ""
    if not name:
        return "no run name", True
    return name, False


class StaleProxy:
    """Show the display-loop staleness chip only when there is text."""

    def __init__(self, label) -> None:
        self._label = label

    @property
    def text(self):
        return self._label.text

    @text.setter
    def text(self, value) -> None:
        self._label.text = value or ""
        self._label.style(f"display:{'inline-block' if value else 'none'};")


_MONO = "font-family:var(--mono);"


def build_context_section(
    identity: Dict[str, str],
    register_staleness,
    *,
    show_layers: bool = False,
    sampler_interval_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Lay out the strip; returns the widgets ``update_context_section`` fills."""
    cards: Dict[str, Any] = {
        "live_threshold_s": live_threshold_s(sampler_interval_s)
    }
    with (
        ui.element("div")
        .classes("glass reveal w-full")
        .style("padding:14px 22px;")
    ):
        with (
            ui.row()
            .classes("w-full items-center")
            .style("gap:14px; flex-wrap:wrap;")
        ):
            with ui.row().style("gap:0; align-items:baseline;"):
                ui.label("Trace").classes("wm-trace")
                ui.label("ML").classes("wm-ml")
            ui.label(identity.get("profile") or "run").classes("eyebrow")
            name, muted = run_name_label(identity)
            cards["run_name"] = ui.label(name).style(
                f"{_MONO} font-size:13px; "
                + (
                    "font-style:italic; color:var(--muted);"
                    if muted
                    else "font-weight:600; color:var(--ink);"
                )
            )
            cards["strategy"] = (
                ui.label("").classes("eyebrow").style("display:none;")
            )
            if show_layers:
                ui.link("layers", "/layers").style(
                    f"{_MONO} font-size:12px; color:var(--orange-strong); "
                    "text-decoration:none;"
                )
            ui.element("div").style("flex:1;")
            stale = ui.label("").classes("staleband").style("display:none;")
            register_staleness(StaleProxy(stale))
            with ui.row().classes("items-center").style("gap:7px;"):
                cards["dot"] = ui.element("div").classes("livedot")
                cards["liveness"] = ui.label("waiting for data").style(
                    f"{_MONO} font-size:11px; color:var(--muted); "
                    "font-weight:500;"
                )
        cards["coverage"] = ui.label("").style(
            f"{_MONO} font-size:11px; color:var(--muted); margin-top:6px;"
        )
        script = identity.get("script_name") or ""
        cards["script"] = ui.label(script).style(
            f"{_MONO} font-size:11px; color:var(--ink); margin-top:2px; "
            f"display:{'block' if script else 'none'};"
        )
        artifact = identity.get("artifact_path") or ""
        cards["artifact"] = ui.label(abbreviate_path(artifact)).style(
            f"{_MONO} font-size:10px; color:var(--muted); margin-top:2px; "
            f"cursor:{'copy' if artifact else 'default'}; "
            f"display:{'block' if artifact else 'none'};"
        )
        if artifact:
            cards["artifact"].tooltip(f"{artifact}  (click to copy)")

            def _copy_path(_event=None, _path=artifact) -> None:
                try:
                    ui.run_javascript(
                        "navigator.clipboard.writeText("
                        f"{json.dumps(_path)})"
                    )
                    ui.notify("path copied", type="positive", timeout=1500)
                except Exception:
                    pass  # copy is a convenience; never break the page

            cards["artifact"].on("click", _copy_path)
    return cards


def update_context_section(
    cards: Dict[str, Any],
    payload: Any,
    *,
    now: Optional[float] = None,
) -> None:
    """Fill strategy, coverage, and liveness from the CONTEXT payload."""
    ctx = payload if isinstance(payload, dict) else None
    if not ctx or not (set(ctx) & _CONTEXT_KEYS):
        # Not a context payload (a LayoutError, an empty dict, or another
        # layout's shape): leave the strip as it is rather than blank it.
        return

    strategy = strategy_token(ctx.get("training_strategy"))
    cards["strategy"].text = strategy
    cards["strategy"].style(
        f"display:{'inline-block' if strategy else 'none'};"
    )

    cards["coverage"].text = format_coverage(ctx)

    last = ctx.get("last_data_ts")
    age = None
    if last is not None:
        age = (now if now is not None else time.time()) - float(last)
    word, detail = format_liveness(
        age, cards.get("live_threshold_s", LIVE_THRESHOLD_S)
    )
    live = word == "live"
    color = "#16a34a" if live else "#dc2626"
    cards["liveness"].text = f"{word} · {detail}" if detail else word
    cards["liveness"].style(f"color:{color};")
    cards["dot"].style(
        f"background:{color}; "
        f"animation:{'tml-pulse 2.4s infinite' if live else 'none'};"
    )
