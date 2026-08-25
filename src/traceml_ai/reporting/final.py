# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Final end-of-run report orchestration."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from traceml_ai.core.summaries import SummaryResult
from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.reporting.analysis_window import (
    AnalysisWindow,
    build_analysis_window_payload,
    resolve_analysis_window,
)
from traceml_ai.reporting.primary_diagnosis import build_primary_diagnosis
from traceml_ai.reporting.schema import empty_section_payload
from traceml_ai.reporting.sections.base import SummarySection
from traceml_ai.reporting.sections.process import ProcessSummarySection
from traceml_ai.reporting.sections.step_memory import StepMemorySummarySection
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection
from traceml_ai.reporting.sections.system import SystemSummarySection
from traceml_ai.reporting.terminal_card.card import (
    build_card_from_payload,
    build_fallback_card,
    card_hints,
    card_to_plain,
    colorize_stored_card,
)
from traceml_ai.sdk.protocol import (
    get_final_summary_html_path,
    get_final_summary_json_path,
    get_final_summary_txt_path,
    utc_now_iso,
)
from traceml_ai.telemetry.retention import DEFAULT_HISTORY_RETENTION_S
from traceml_ai.utils.atomic_io import write_json_atomic, write_text_atomic

# Version 1.8 adds one shared, time-bounded analysis window and explicit
# per-section retention coverage.
SCHEMA_VERSION = 1.8

DEFAULT_PROFILE = "run"


def _log_final_report_error(message: str, exc: Exception) -> None:
    """Log final-report failures without raising from cleanup paths."""
    try:
        get_error_logger("FinalReport").exception("[TraceML] %s", message)
    except Exception:
        pass


@dataclass(frozen=True)
class FunctionSummarySection:
    """Summary section backed by a function."""

    name: str
    builder: Callable[..., Dict[str, Any]]

    def build(self, db_path: str) -> SummaryResult:
        """Build one summary section from the SQLite database path."""
        payload = self.builder(db_path, print_to_stdout=False)
        return SummaryResult(
            section=self.name,
            payload=dict(payload or {}),
            text=str((payload or {}).get("card", "")),
        )


def _safe_int(value: Any, *, allow_zero: bool = True) -> Optional[int]:
    """Return an int when a JSON value is numeric enough for metadata."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < 0 or (parsed == 0 and not allow_zero):
        return None
    return parsed


def _section_metadata(section: Dict[str, Any]) -> Dict[str, Any]:
    """Return a section metadata mapping when present."""
    metadata = section.get("metadata") if isinstance(section, dict) else None
    return dict(metadata) if isinstance(metadata, dict) else {}


def _section_group_rows(section: Dict[str, Any]) -> Dict[str, Any]:
    """Return section group rows when present."""
    groups = section.get("groups") if isinstance(section, dict) else None
    if not isinstance(groups, dict):
        return {}
    rows = groups.get("rows")
    return dict(rows) if isinstance(rows, dict) else {}


def _run_name_from_manifest(session_root: Optional[str]) -> Optional[str]:
    """Load the public run name from the launcher manifest when available."""
    if not session_root:
        return None

    root = Path(session_root).resolve()
    manifest_path = root / "manifest.json"
    manifest: Dict[str, Any] = {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            manifest = loaded
    except Exception:
        manifest = {}

    run = manifest.get("run")
    run_block = run if isinstance(run, dict) else {}
    candidates = (
        run_block.get("run_name"),
        manifest.get("session_id"),
        root.name,
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def _lifecycle_duration_s(session_root: Optional[str]) -> Optional[float]:
    """Return genuine training duration from launcher lifecycle timestamps."""
    if not session_root:
        return None
    try:
        with open(
            Path(session_root).resolve() / "manifest.json",
            "r",
            encoding="utf-8",
        ) as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            return None
        lifecycle = manifest.get("lifecycle")
        if not isinstance(lifecycle, dict):
            return None
        started = datetime.fromisoformat(str(lifecycle["training_started_at"]))
        ended = datetime.fromisoformat(str(lifecycle["training_ended_at"]))
        duration = (ended - started).total_seconds()
        return duration if duration >= 0.0 else None
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def _final_meta_mode(sections: Sequence[Dict[str, Any]]) -> str:
    """Infer run mode from already-built section metadata."""
    saw_single_node = False
    for section in sections:
        mode = str(_section_metadata(section).get("mode") or "")
        if mode == "multi_node":
            return "multi_node"
        if mode == "single_node":
            saw_single_node = True
    return "single_node" if saw_single_node else "no_data"


def _final_meta_world_size(
    sections: Sequence[Dict[str, Any]],
) -> Optional[int]:
    """Infer observed world size from section metadata and rank identities."""
    candidates: List[int] = []
    for section in sections:
        metadata = _section_metadata(section)
        for key in ("global_ranks_seen", "global_ranks_used"):
            value = _safe_int(metadata.get(key), allow_zero=False)
            if value is not None:
                candidates.append(value)

        groups = section.get("groups") if isinstance(section, dict) else None
        group_by = groups.get("by") if isinstance(groups, dict) else None
        rows = _section_group_rows(section)
        if group_by == "global_rank" and rows:
            candidates.append(len(rows))

        for group_row in rows.values():
            if not isinstance(group_row, dict):
                continue
            identity = group_row.get("identity")
            if not isinstance(identity, dict):
                continue
            value = _safe_int(identity.get("world_size"), allow_zero=False)
            if value is not None:
                candidates.append(value)

    return max(candidates) if candidates else None


def _final_meta_nodes_observed(
    system_summary: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
) -> Optional[int]:
    """Infer observed node count from system metadata or row identities."""
    value = _safe_int(
        _section_metadata(system_summary).get("nodes_observed"),
        allow_zero=True,
    )
    if value is not None:
        return value

    node_ranks = set()
    for section in sections:
        for group_row in _section_group_rows(section).values():
            if not isinstance(group_row, dict):
                continue
            identity = group_row.get("identity")
            if not isinstance(identity, dict):
                continue
            node_rank = _safe_int(identity.get("node_rank"), allow_zero=True)
            if node_rank is not None:
                node_ranks.add(node_rank)
    return len(node_ranks) if node_ranks else None


def _final_meta_gpus_observed(
    *,
    system_summary: Dict[str, Any],
    mode: str,
    world_size: Optional[int],
) -> Optional[int]:
    """Infer observed GPU count from system metadata with a safe fallback."""
    value = _safe_int(
        _section_metadata(system_summary).get("gpus_observed"),
        allow_zero=True,
    )
    if value is not None:
        return value
    if mode == "single_node" and world_size is not None:
        return int(world_size)
    return None


def _build_final_meta(
    *,
    session_root: Optional[str],
    system_summary: Dict[str, Any],
    process_summary: Dict[str, Any],
    step_time_summary: Dict[str, Any],
    step_memory_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build run-level identity and observed topology metadata."""
    sections = (
        system_summary,
        process_summary,
        step_time_summary,
        step_memory_summary,
    )
    mode = _final_meta_mode(sections)
    world_size = _final_meta_world_size(sections)
    return {
        "run_name": _run_name_from_manifest(session_root),
        "mode": mode,
        "world_size": world_size,
        "nodes_observed": _final_meta_nodes_observed(
            system_summary,
            sections,
        ),
        "gpus_observed": _final_meta_gpus_observed(
            system_summary=system_summary,
            mode=mode,
            world_size=world_size,
        ),
    }


def _build_final_summary_text(
    payload: Dict[str, Any],
    *,
    profile: str = DEFAULT_PROFILE,
    session_root: Optional[str] = None,
    write_html: bool = False,
) -> str:
    """
    Build the printed end-of-run card from an already-built payload.

    Reporting runs during aggregator shutdown, so a rendering failure must
    degrade to a minimal verdict card instead of raising.
    """
    try:
        return card_to_plain(
            build_card_from_payload(
                payload,
                profile=profile,
                session_root=session_root,
                write_html=write_html,
            )
        )
    except Exception as exc:
        _log_final_report_error("Final summary card failed", exc)

    try:
        return card_to_plain(
            build_fallback_card(
                profile=profile,
                primary_diagnosis=payload.get("primary_diagnosis") or {},
                **card_hints(session_root, write_html=write_html),
            )
        )
    except Exception as exc:
        _log_final_report_error("Final summary fallback card failed", exc)
        return ""


def _stdout_card_text(
    payload: Dict[str, Any],
    *,
    session_root: Optional[str],
    write_html: bool,
) -> str:
    """Return the stdout rendering: ANSI on a terminal, plain otherwise."""
    return colorize_stored_card(
        str(payload.get("text", "")),
        payload,
        session_root=session_root,
        write_html=write_html,
    )


def _fallback_section_result(section: SummarySection) -> SummaryResult:
    """Return a stable section payload when one section fails."""
    name = str(getattr(section, "name", "unknown"))
    index_by = "node_rank" if name == "system" else "global_rank"
    payload = empty_section_payload(section_name=name, index_by=index_by)
    return SummaryResult(
        section=name,
        payload=payload,
        text=str(payload.get("card", "")),
    )


@dataclass(frozen=True)
class FinalReportGenerator:
    """
    Build TraceML's final summary payload from registered sections.

    ``sections`` is ordered because final-summary text and JSON key order are
    intentionally stable for users and compare tooling.
    """

    sections: Sequence[SummarySection]
    analysis_window: Optional[AnalysisWindow] = None

    def generate(
        self,
        db_path: str,
        *,
        session_root: Optional[str] = None,
        profile: str = DEFAULT_PROFILE,
        write_html: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate the structured final summary payload.

        Section failures are isolated and logged. A failed section contributes
        a schema-valid ``NO DATA`` payload so one broken report domain does
        not prevent artifact generation during aggregator shutdown.

        ``profile`` and ``write_html`` are presentation inputs for the printed
        card only; they do not change the JSON contract.
        """
        results: Dict[str, SummaryResult] = {}
        for section in self.sections:
            try:
                result = section.build(db_path)
            except Exception as exc:
                _log_final_report_error(
                    "Final summary section failed: "
                    f"{getattr(section, 'name', section)}",
                    exc,
                )
                result = _fallback_section_result(section)
            results[result.section] = result

        system_summary = dict(
            results.get("system", SummaryResult("system")).payload
        )
        process_summary = dict(
            results.get("process", SummaryResult("process")).payload
        )
        step_time_summary = dict(
            results.get("step_time", SummaryResult("step_time")).payload
        )
        step_memory_summary = dict(
            results.get("step_memory", SummaryResult("step_memory")).payload
        )

        analysis_window_payload = (
            build_analysis_window_payload(db_path, self.analysis_window)
            if self.analysis_window is not None
            else None
        )
        if isinstance(analysis_window_payload, dict):
            observations = analysis_window_payload.get("sections", {})
            for name, section in (
                ("system", system_summary),
                ("process", process_summary),
                ("step_time", step_time_summary),
                ("step_memory", step_memory_summary),
            ):
                metadata = section.get("metadata")
                observation = observations.get(name)
                if isinstance(metadata, dict) and isinstance(
                    observation, dict
                ):
                    metadata["history_coverage"] = observation.get("coverage")

        primary_diagnosis = build_primary_diagnosis(
            system_summary=system_summary,
            process_summary=process_summary,
            step_time_summary=step_time_summary,
            step_memory_summary=step_memory_summary,
        )

        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            # Never promote a bounded telemetry interval to full-run duration.
            "duration_s": _lifecycle_duration_s(session_root),
            "analysis_window": analysis_window_payload,
            "meta": _build_final_meta(
                session_root=session_root,
                system_summary=system_summary,
                process_summary=process_summary,
                step_time_summary=step_time_summary,
                step_memory_summary=step_memory_summary,
            ),
            "primary_diagnosis": primary_diagnosis,
            "system": system_summary,
            "process": process_summary,
            "step_time": step_time_summary,
            "step_memory": step_memory_summary,
            "text": "",
        }
        payload["text"] = _build_final_summary_text(
            payload,
            profile=profile,
            session_root=session_root,
            write_html=write_html,
        )
        return payload


def build_final_report_generator(
    *,
    analysis_window: Optional[AnalysisWindow] = None,
) -> FinalReportGenerator:
    """Build the standard final-report generator."""
    return FinalReportGenerator(
        sections=(
            SystemSummarySection(analysis_window=analysis_window),
            ProcessSummarySection(analysis_window=analysis_window),
            StepTimeSummarySection(analysis_window=analysis_window),
            StepMemorySummarySection(analysis_window=analysis_window),
        ),
        analysis_window=analysis_window,
    )


DEFAULT_FINAL_REPORT_GENERATOR = build_final_report_generator()


def build_summary_payload(
    db_path: str,
    *,
    generator: Optional[FinalReportGenerator] = None,
    session_root: Optional[str] = None,
    history_retention_s: float = DEFAULT_HISTORY_RETENTION_S,
    profile: str = DEFAULT_PROFILE,
    write_html: bool = False,
) -> Dict[str, Any]:
    """
    Build the structured final summary payload for one session database.
    """
    active_generator = generator
    if active_generator is None:
        analysis_window = resolve_analysis_window(
            db_path,
            retention_s=history_retention_s,
        )
        active_generator = build_final_report_generator(
            analysis_window=analysis_window,
        )
    return active_generator.generate(
        db_path,
        session_root=session_root,
        profile=profile,
        write_html=write_html,
    )


def _write_html_artifact(payload: Dict[str, Any], session_root: Path) -> None:
    """
    Render and write the optional HTML report (best-effort).

    Runs after the JSON/TXT artifacts so that a rendering failure can never
    block them. Any failure is logged and reported to stderr, never raised.
    """
    try:
        from traceml_ai.reporting import html as html_report

        html_report.write_html_report(
            payload,
            get_final_summary_html_path(session_root),
        )
    except Exception as exc:  # best-effort: never break the run
        _log_final_report_error("HTML report failed", exc)
        try:
            print(f"[TraceML] HTML report failed: {exc}", file=sys.stderr)
        except Exception:
            pass


def write_summary_artifacts(
    *,
    db_path: str,
    payload: Dict[str, Any],
    session_root: Optional[str] = None,
    write_html: bool = False,
) -> None:
    """
    Write final summary artifacts to disk.

    Artifacts written
    -----------------
    - legacy DB-adjacent artifacts for compatibility
    - canonical session-root artifacts for public API consumers
    - optional self-contained HTML report (``write_html``, best-effort)
    """
    final_text = str(payload.get("text", ""))

    legacy_json_path = Path(str(db_path) + "_summary_card.json").resolve()
    legacy_txt_path = Path(str(db_path) + "_summary_card.txt").resolve()

    write_json_atomic(legacy_json_path, payload)
    write_text_atomic(legacy_txt_path, final_text + "\n")

    if session_root:
        session_root_path = Path(session_root).resolve()
        write_json_atomic(
            get_final_summary_json_path(session_root_path),
            payload,
        )
        write_text_atomic(
            get_final_summary_txt_path(session_root_path),
            final_text + "\n",
        )
        if write_html:
            _write_html_artifact(payload, session_root_path)


def generate_summary(
    db_path: str,
    *,
    session_root: Optional[str] = None,
    print_to_stdout: bool = True,
    history_retention_s: float = DEFAULT_HISTORY_RETENTION_S,
    write_html: bool = False,
    profile: str = DEFAULT_PROFILE,
) -> Dict[str, Any]:
    """
    Generate, write, and optionally print the final end-of-run summary.

    ``payload["text"]`` and the ``.txt`` artifact always hold the plain
    rendering. Severity-scoped ANSI is used only when printing to a terminal.
    """
    payload = build_summary_payload(
        db_path,
        session_root=session_root,
        history_retention_s=history_retention_s,
        profile=profile,
        write_html=write_html,
    )
    write_summary_artifacts(
        db_path=db_path,
        payload=payload,
        session_root=session_root,
        write_html=write_html,
    )

    if print_to_stdout:
        print(
            _stdout_card_text(
                payload,
                session_root=session_root,
                write_html=write_html,
            )
        )

    return payload


__all__ = [
    "DEFAULT_FINAL_REPORT_GENERATOR",
    "FinalReportGenerator",
    "FunctionSummarySection",
    "ProcessSummarySection",
    "StepMemorySummarySection",
    "StepTimeSummarySection",
    "SystemSummarySection",
    "build_final_report_generator",
    "build_summary_payload",
    "generate_summary",
    "write_summary_artifacts",
]
