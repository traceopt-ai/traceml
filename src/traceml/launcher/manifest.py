"""Run manifest and static code-manifest helpers for the TraceML launcher."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from traceml.utils.ast_analysis import analyze_script, build_code_manifest


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid partially written manifest files."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as tmp:
            json.dump(payload, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)

        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise


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
    tcp_host: str,
    tcp_port: int,
    nproc_per_node: int,
    history_enabled: bool,
    status: str,
    launch_cwd: str,
    aggregator_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write or overwrite a launcher run manifest under ``session_root``."""
    session_root = Path(session_root).resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    manifest_path = session_root / "manifest.json"
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "session_id": str(session_id),
        "status": str(status),
        "created_at": utc_now_iso(),
        "host": {"hostname": socket.gethostname()},
        "launch": {
            "script_path": str(Path(script_path).resolve()),
            "profile": str(profile),
            "ui_mode": str(ui_mode),
            "logs_dir": str(Path(logs_dir).resolve()),
            "tcp_host": str(tcp_host),
            "tcp_port": int(tcp_port),
            "nproc_per_node": int(nproc_per_node),
            "history_enabled": bool(history_enabled),
            "launch_cwd": str(Path(launch_cwd).resolve()),
        },
        "paths": {
            "session_root": str(session_root),
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
        "summary_card_json": Path(str(db_path) + ".summary_card.json"),
        "summary_card_txt": Path(str(db_path) + ".summary_card.txt"),
    }
    if session_root is not None:
        candidates["code_manifest"] = (
            Path(session_root).resolve() / "code_manifest.json"
        )

    return {
        name: str(path) for name, path in candidates.items() if path.exists()
    }
