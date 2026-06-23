# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Run manifest and static code-manifest helpers for the TraceML launcher."""

from __future__ import annotations

import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from traceml_ai.utils.ast_analysis import analyze_script, build_code_manifest
from traceml_ai.utils.atomic_io import write_json_atomic


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def load_json_or_warn(path: Path) -> Dict[str, Any]:
    """Load JSON from disk, returning an empty dict for missing/bad files."""
    path = Path(path).resolve()

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        corrupt_path = path.with_suffix(path.suffix + ".corrupt")
        try:
            corrupt_path.write_text(
                path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        except Exception:
            pass
        print(
            f"[TraceML] WARNING: manifest is malformed and will be rebuilt: {path} ({exc})",
            file=sys.stderr,
        )
        return {}
    except OSError as exc:
        print(
            f"[TraceML] WARNING: manifest could not be read and will be rebuilt: {path} ({exc})",
            file=sys.stderr,
        )
        return {}


def write_code_manifest(
    session_root: Path,
    script_path: str,
) -> Optional[Path]:
    """Write static-analysis details for the launched script.

    Static analysis is useful metadata, not a reason to break a training run.
    If analysis fails, this helper writes a minimal failure manifest when
    possible and otherwise returns ``None``.
    """
    session_root = Path(session_root).resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    manifest_path = session_root / "code_manifest.json"

    try:
        findings = analyze_script(str(Path(script_path).resolve()))
        manifest = build_code_manifest(findings)
        manifest["analysis_status"] = (
            "ok" if not findings.parse_errors else "partial"
        )
        write_json_atomic(manifest_path, manifest)
        return manifest_path
    except Exception as exc:
        fallback: Dict[str, Any] = {
            "schema_version": 1,
            "script_path": str(Path(script_path).resolve()),
            "generated_at": utc_now_iso(),
            "analysis_status": "failed",
            "parse_errors": [f"Static analysis failed: {exc}"],
        }
        try:
            write_json_atomic(manifest_path, fallback)
            return manifest_path
        except Exception:
            return None


def write_run_manifest(
    session_root: Path,
    session_id: str,
    script_path: str,
    profile: str,
    ui_mode: str,
    logs_dir: str,
    aggregator_host: str,
    aggregator_bind_host: str,
    aggregator_port: int,
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    nproc_per_node: int,
    history_enabled: bool,
    summary_window_rows: int,
    finalize_timeout_sec: float,
    status: str,
    launch_cwd: str,
    run: Optional[Dict[str, Any]] = None,
    aggregator_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write or overwrite a launcher run manifest under ``session_root``."""
    session_root = Path(session_root).resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    manifest_path = session_root / "manifest.json"
    run_block: Dict[str, Any] = {
        "run_name": str(session_id),
        "session_id": str(session_id),
    }
    if run:
        run_block.update(run)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "session_id": str(session_id),
        "run": run_block,
        "status": str(status),
        "created_at": utc_now_iso(),
        "host": {"hostname": socket.gethostname()},
        "launch": {
            "script_path": str(Path(script_path).resolve()),
            "profile": str(profile),
            "ui_mode": str(ui_mode),
            "logs_dir": str(Path(logs_dir).resolve()),
            "aggregator_host": str(aggregator_host),
            "aggregator_bind_host": str(aggregator_bind_host),
            "aggregator_port": int(aggregator_port),
            "nnodes": int(nnodes),
            "nproc_per_node": int(nproc_per_node),
            "node_rank": int(node_rank),
            "master_addr": str(master_addr),
            "master_port": int(master_port),
            "history_enabled": bool(history_enabled),
            "summary_window_rows": int(summary_window_rows),
            "finalize_timeout_sec": float(finalize_timeout_sec),
            "launch_cwd": str(Path(launch_cwd).resolve()),
        },
        "paths": {
            "session_root": str(session_root),
            "run_root": str(session_root),
            "aggregator_dir": (
                str(aggregator_dir.resolve()) if aggregator_dir else None
            ),
            "db_path": str(db_path.resolve()) if db_path else None,
        },
        "artifacts": {},
    }

    if extra:
        for key, value in extra.items():
            if key == "artifacts" and isinstance(value, dict):
                manifest.setdefault("artifacts", {}).update(value)
            else:
                manifest[key] = value

    write_json_atomic(manifest_path, manifest)
    return manifest_path


def update_run_manifest(
    manifest_path: Path,
    *,
    status: Optional[str] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Update an existing run manifest in place using an atomic rewrite."""
    manifest_path = Path(manifest_path).resolve()
    manifest = load_json_or_warn(manifest_path)

    if status is not None:
        manifest["status"] = str(status)

    manifest["updated_at"] = utc_now_iso()

    if artifacts:
        manifest.setdefault("artifacts", {}).update(artifacts)

    if extra:
        manifest.update(extra)

    write_json_atomic(manifest_path, manifest)
    return manifest_path


def collect_existing_artifacts(
    db_path: Path,
    session_root: Optional[Path] = None,
) -> Dict[str, str]:
    """Return only launcher artifacts that currently exist on disk."""
    candidates = {
        "db": db_path,
        "summary_card_json": Path(str(db_path) + "_summary_card.json"),
        "summary_card_txt": Path(str(db_path) + "_summary_card.txt"),
        "legacy_summary_card_json": Path(str(db_path) + ".summary_card.json"),
        "legacy_summary_card_txt": Path(str(db_path) + ".summary_card.txt"),
    }
    if session_root is not None:
        candidates["code_manifest"] = (
            Path(session_root).resolve() / "code_manifest.json"
        )

    return {
        name: str(path) for name, path in candidates.items() if path.exists()
    }
