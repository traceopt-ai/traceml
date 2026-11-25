# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Context strip: which run this is, how it executes, and whether data flows.

The strip sits above every metric pane and stays compact: three labelled
rows (the run or session name with the ``Data:`` status, the ``Execution:``
line, and the summary path with a Copy button) and a collapsed Details
section for the script, profile, detected GPUs, observed duration, telemetry
coverage and last sample. Every count is observed, never configured, and a
fact that is missing reads ``unknown`` rather than ``0``.

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
from traceml_ai.renderers.shared.freshness import FreshnessPolicy

from .formatting import format_elapsed

# Sections that only exist on a training run. A watch session has no step
# loop, so showing them there produces permanently dead panels (#355).
STEP_SECTIONS: Tuple[str, ...] = (
    "model_combined",
    "step_memory",
    "model_diagnostics",
)
RESOURCE_SECTIONS: Tuple[str, ...] = ("system", "process")
RUN_SECTIONS: Tuple[str, ...] = (
    "model_combined",
    "system",
    "process",
    "step_memory",
    "model_diagnostics",
)

# The facts a context payload carries (renderers/context); anything else
# handed to ``update_context_section`` is ignored rather than misread.
_CONTEXT_KEYS = frozenset(empty_context())

# Shape of a launcher-generated session id, e.g. session_20260821_123224_1103e8;
# used when a manifest without identity_source is replayed.
_GENERATED_ID = re.compile(r"^session_\d{8}_\d{6}_[0-9a-f]{6}$")


def sections_for_profile(profile: str) -> Tuple[str, ...]:
    """Sections a page shows for ``profile``: watch gets no step panels."""
    if str(profile or "run") == "watch":
        return RESOURCE_SECTIONS
    return RUN_SECTIONS


def live_threshold_s(sampler_interval_s: Optional[float]) -> float:
    """The shared freshness threshold, fed the CONFIGURED cadence.

    One rule -- ``FreshnessPolicy``, ``max(5 s, 3 ticks)`` -- but the strip
    and the System card deliberately feed it different cadences, so their
    thresholds can differ. That is the design, not the drift that the old
    local ``2.5 * interval`` was.

    The strip is run-wide: its age is the newest System or Process
    timestamp anywhere in the run, so there is no single host whose real
    rhythm it could measure. It is judged against the cadence the run was
    configured with.

    The System card describes one node and knows what that node actually
    reported at, so it feeds the same policy its OBSERVED cadence
    (:meth:`SystemDashboardComputer._node_liveness`). A node reporting
    slower than requested is judged against its real rhythm rather than
    against an intention it never met.
    """
    return FreshnessPolicy.from_interval(sampler_interval_s).stale_after_s


def format_liveness(
    age_s: Optional[float], threshold_s: Optional[float] = None
) -> Tuple[str, str]:
    """(state word, detail) from age and a shared-policy threshold."""
    if age_s is None:
        return "no data", "waiting for the first sample"
    age = max(0.0, float(age_s))
    threshold = (
        live_threshold_s(None) if threshold_s is None else float(threshold_s)
    )
    # FreshnessPolicy marks data stale only after the threshold is exceeded.
    if age <= threshold:
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


def data_status(
    age_s: Optional[float], threshold_s: Optional[float] = None
) -> Tuple[str, str]:
    """(``Waiting`` / ``Updating`` / ``Delayed``, detail) for ``Data:``.

    The same freshness rule as :func:`format_liveness`, worded as the state of
    the telemetry, not of the training: fresh samples say nothing about
    whether the run is succeeding or progressing.
    """
    word, detail = format_liveness(age_s, threshold_s)
    if word == "no data":
        return "Waiting", ""
    if word == "live":
        return "Updating", ""
    return "Delayed", detail


def format_execution(ctx: Dict[str, Any]) -> str:
    """``DDP · 4 processes · 1 node`` from the recorded facts.

    The strategy appears only when it was recorded: a process or GPU count
    does not establish one. Missing counts are left out, and with nothing
    known at all the line reads ``unknown``.
    """
    parts = []
    strategy = strategy_token(ctx.get("training_strategy"))
    if strategy:
        parts.append(strategy)
    processes = int(ctx.get("world_size") or 0)
    if processes == 1:
        parts.append("Single process")
    elif processes > 1:
        parts.append(f"{processes} processes")
    nodes = ctx.get("node_count")
    if nodes:
        parts.append(f"{int(nodes)} node{'s' if int(nodes) != 1 else ''}")
    return " · ".join(parts) or "unknown"


def format_last_sample(age_s: Optional[float]) -> str:
    """``1s ago`` from the data age, ``unknown`` before the first sample."""
    if age_s is None:
        return "unknown"
    return f"{format_elapsed(max(0.0, float(age_s)))} ago"


def detail_rows(
    identity: Dict[str, str],
    ctx: Dict[str, Any],
    age_s: Optional[float],
) -> Tuple[Tuple[str, str], ...]:
    """Label/value pairs for the collapsed Details section.

    ``GPUs detected`` is hardware seen on the machines, which can be zero,
    one or several regardless of the process count; it is not the number of
    GPUs the training uses. Before any sample has arrived it is unknown.
    """
    have_data = ctx.get("last_data_ts") is not None
    gpus = ctx.get("gpus_observed")
    if gpus is None:
        gpus = ctx.get("gpu_count")
    gpus_text = str(int(gpus or 0)) if have_data else "unknown"
    first = ctx.get("first_data_ts")
    last = ctx.get("last_data_ts")
    if first is not None and last is not None and last >= first:
        duration = format_elapsed(float(last) - float(first))
    else:
        duration = "unknown"
    reporting = ctx.get("ranks_reporting")
    expected = int(ctx.get("world_size") or 0)
    if reporting is not None and expected:
        coverage = f"{int(reporting)} / {expected} ranks"
    elif reporting is not None:
        coverage = f"{int(reporting)} ranks"
    else:
        coverage = "unknown"
    return (
        ("Script", identity.get("script_name") or "unknown"),
        ("Profile", identity.get("profile") or "run"),
        ("GPUs detected", gpus_text),
        ("Observed duration", duration),
        ("Telemetry coverage", coverage),
        ("Last sample", format_last_sample(age_s)),
    )


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
    so the strip says ``Unnamed run`` instead of showing the id as if it were
    one; the id itself stays visible in the summary path."""
    if identity.get("run_name_source") == "generated":
        return "Unnamed run", True
    name = identity.get("run_name") or ""
    if not name:
        return "Unnamed run", True
    return name, False


def identity_word(profile: str) -> str:
    """``Session`` for a watch session, which has no training run, else ``Run``."""
    return "Session" if str(profile or "run") == "watch" else "Run"


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

# One line, clipped with an ellipsis; the full value stays reachable in a
# tooltip. The strip shares its row with Diagnostics, so nothing may widen it.
_CLIP = (
    "white-space:nowrap; overflow:hidden; text-overflow:ellipsis; "
    "min-width:0; max-width:100%;"
)
_FIELD_LABEL = (
    f"{_MONO} font-size:11px; color:var(--muted); white-space:nowrap;"
)

# Scrolls each summary-path field to its end once, so a long path first shows
# its file name. It runs once after the page is built, before the user can
# select anything, and the path never changes, so refreshes never move it.
_SCROLL_PATH_TO_END = (
    "document.querySelectorAll('.tml-path input').forEach("
    "(el) => { el.scrollLeft = el.scrollWidth; });"
)


def _copy_to_clipboard(path: str) -> None:
    """Copy the complete original path and confirm it briefly."""
    try:
        ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(path)})")
        ui.notify("Copied", type="positive", timeout=1500)
    except Exception:
        pass  # copy is a convenience; never break the page


def build_context_section(
    identity: Dict[str, str],
    register_staleness,
    *,
    show_layers: bool = False,
    sampler_interval_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Lay out the strip; returns the widgets ``update_context_section`` fills."""
    cards: Dict[str, Any] = {
        "live_threshold_s": live_threshold_s(sampler_interval_s),
        "identity": dict(identity),
        "details_open": False,
    }
    with (
        ui.element("div")
        .classes("glass reveal w-full")
        .style("padding:14px 22px; min-width:0;")
    ):
        # Row 1: brand, the labelled run or session name, and Data status.
        with (
            ui.row()
            .classes("w-full items-center")
            .style("gap:10px 14px; flex-wrap:wrap; min-width:0;")
        ):
            with ui.row().style("gap:0; align-items:baseline;"):
                ui.label("Trace").classes("wm-trace")
                ui.label("ML").classes("wm-ml")
            with (
                ui.row()
                .classes("items-baseline no-wrap")
                .style("gap:6px; min-width:0; flex:1 1 160px;")
            ):
                cards["identity_label"] = ui.label(
                    f"{identity_word(identity.get('profile') or 'run')}:"
                ).style(_FIELD_LABEL)
                name, muted = run_name_label(identity)
                cards["run_name"] = ui.label(name).style(
                    f"{_MONO} font-size:13px; {_CLIP}"
                    + (
                        "font-style:italic; color:var(--muted);"
                        if muted
                        else "font-weight:600; color:var(--ink);"
                    )
                )
                if not muted:
                    cards["run_name"].tooltip(name)
                if show_layers:
                    ui.link("layers", "/layers").style(
                        f"{_MONO} font-size:12px; color:var(--orange-strong); "
                        "text-decoration:none; white-space:nowrap;"
                    )
            stale = ui.label("").classes("staleband").style("display:none;")
            register_staleness(StaleProxy(stale))
            with ui.row().classes("items-center no-wrap").style("gap:7px;"):
                ui.label("Data:").style(_FIELD_LABEL)
                cards["dot"] = (
                    ui.element("div")
                    .classes("livedot")
                    .style("background:var(--muted); animation:none;")
                )
                cards["status"] = ui.label("Waiting").style(
                    f"{_MONO} font-size:11px; color:var(--muted); "
                    "font-weight:500; white-space:nowrap;"
                )

        # Row 2: execution facts and the Details toggle.
        with (
            ui.row()
            .classes("w-full items-center no-wrap")
            .style("gap:8px; margin-top:6px; min-width:0;")
        ):
            ui.label("Execution:").style(_FIELD_LABEL)
            # Short and worth reading whole: it wraps on a narrow screen
            # instead of being clipped.
            cards["execution"] = ui.label("unknown").style(
                f"{_MONO} font-size:11px; color:var(--ink); "
                "flex:1; min-width:0; overflow-wrap:anywhere;"
            )
            cards["details_toggle"] = (
                ui.button(
                    "Details ▾",
                    on_click=lambda: set_details_open(
                        cards, not cards["details_open"]
                    ),
                )
                .props("flat dense no-caps color=grey-8 aria-expanded=false")
                .style(
                    f"{_MONO} font-size:11px; color:var(--muted); "
                    "white-space:nowrap; flex:none;"
                )
            )

        # Row 3: the summary destination, read-only, one line, with Copy.
        # The field holds the complete path; only its view is clipped.
        artifact = identity.get("artifact_path") or ""
        cards["artifact_row"] = (
            ui.row()
            .classes("w-full items-center no-wrap")
            .style(
                "gap:8px; margin-top:6px; min-width:0; "
                f"display:{'flex' if artifact else 'none'};"
            )
        )
        with cards["artifact_row"]:
            ui.label("Summary JSON").style(_FIELD_LABEL)
            cards["artifact"] = (
                ui.input(value=artifact)
                .props("readonly dense outlined spellcheck=false")
                .classes("tml-path")
                .style(f"flex:1; min-width:0; {_MONO} font-size:11px;")
            )
            cards["copy"] = (
                ui.button(
                    "Copy", on_click=lambda: _copy_to_clipboard(artifact)
                )
                .props("outline dense no-caps color=grey-8")
                .style(f"{_MONO} font-size:11px;")
            )
        if artifact:
            ui.timer(
                0.3, lambda: ui.run_javascript(_SCROLL_PATH_TO_END), once=True
            )

        # Details: secondary facts, collapsed by default.
        cards["details"] = (
            ui.element("div")
            .classes("w-full")
            .style(
                "display:none; margin-top:8px; grid-template-columns:auto 1fr; "
                "gap:3px 16px; min-width:0;"
            )
        )
        cards["detail_values"] = {}
        with cards["details"]:
            for label, value in detail_rows(identity, {}, None):
                ui.label(label).style(_FIELD_LABEL)
                cards["detail_values"][label] = ui.label(value).style(
                    f"{_MONO} font-size:11px; color:var(--ink); {_CLIP}"
                )
    return cards


def set_details_open(cards: Dict[str, Any], is_open: bool) -> None:
    """Show or hide the Details section and keep its toggle in step."""
    cards["details_open"] = bool(is_open)
    cards["details"].style(f"display:{'grid' if is_open else 'none'};")
    cards["details_toggle"].text = "Details ▴" if is_open else "Details ▾"
    cards["details_toggle"].props(
        f"aria-expanded={'true' if is_open else 'false'}"
    )


# Data status colours: muted while waiting, green while samples are fresh,
# red once they are delayed.
_STATUS_COLOR = {
    "Waiting": "var(--muted)",
    "Updating": "#16a34a",
    "Delayed": "#dc2626",
}


def update_context_section(
    cards: Dict[str, Any],
    payload: Any,
    *,
    now: Optional[float] = None,
) -> None:
    """Fill execution, Data status and Details from the CONTEXT payload."""
    ctx = payload if isinstance(payload, dict) else None
    if not ctx or not (set(ctx) & _CONTEXT_KEYS):
        # Not a context payload (a LayoutError, an empty dict, or another
        # layout's shape): leave the strip as it is rather than blank it.
        return

    cards["execution"].text = format_execution(ctx)

    last = ctx.get("last_data_ts")
    age = None
    if last is not None:
        age = (now if now is not None else time.time()) - float(last)
    word, detail = data_status(age, cards.get("live_threshold_s"))
    color = _STATUS_COLOR[word]
    cards["status"].text = f"{word} · {detail}" if detail else word
    cards["status"].style(f"color:{color};")
    cards["dot"].style(
        f"background:{color}; "
        f"animation:{'tml-pulse 2.4s infinite' if word == 'Updating' else 'none'};"
    )

    for label, value in detail_rows(cards.get("identity") or {}, ctx, age):
        widget = cards.get("detail_values", {}).get(label)
        if widget is not None and widget.text != value:
            widget.text = value
