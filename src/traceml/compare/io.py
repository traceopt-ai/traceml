"""
I/O helpers for TraceML run comparison.

This module handles:
- strict loading of final summary JSON files
- lightweight payload validation
- stable default labels for compare display/output
- atomic compare artifact writes

It intentionally does not contain comparison logic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

from traceml.final_summary_protocol import write_json_atomic, write_text_atomic

_REQUIRED_TOP_LEVEL_SECTIONS = (
    "system",
    "process",
    "step_time",
    "step_memory",
)

_GENERIC_COMPARE_STEMS = {
    "final_summary",
    "summary",
    "run",
    "output",
}


def load_summary_json(path: str | Path) -> Dict[str, Any]:
    """
    Load one TraceML final summary JSON file strictly.

    Raises
    ------
    RuntimeError
        If the path does not exist, is not a file, or does not contain valid
        JSON.
    """
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise RuntimeError(f"Summary file not found: {resolved}")
    if not resolved.is_file():
        raise RuntimeError(f"Summary path is not a file: {resolved}")

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Summary file is not valid JSON: {resolved} ({exc})"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Summary file could not be read: {resolved} ({exc})"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Summary file must contain a JSON object: {resolved}"
        )

    validate_summary_payload(payload, path=resolved)
    return payload


def validate_summary_payload(
    payload: Dict[str, Any],
    *,
    path: Path,
) -> None:
    """
    Validate the minimum shape required for `traceml compare`.

    This validation is intentionally light:
    - require a dict payload
    - require the known top-level summary sections
    - allow missing optional fields so compare can degrade gracefully
    """
    schema_version = payload.get("schema_version")
    if schema_version is None:
        raise RuntimeError(f"Summary file is missing 'schema_version': {path}")

    for section in _REQUIRED_TOP_LEVEL_SECTIONS:
        if not isinstance(payload.get(section), dict):
            raise RuntimeError(
                f"Summary file is missing required section '{section}': {path}"
            )


def _sanitize_label(value: str) -> str:
    """
    Convert a path-derived label into a filesystem-friendly stem.
    """
    text = str(value).strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "run"


def _stem_is_generic(stem: str) -> bool:
    """
    Return True when a file stem is too generic to be a good default label.
    """
    return stem.lower() in _GENERIC_COMPARE_STEMS


def _default_label_for_path(path: Path) -> str:
    """
    Derive a stable label from one summary path for display and default output.

    Strategy
    --------
    1. Prefer the file stem
    2. If the stem is generic, fall back to the parent directory name
    3. Sanitize for filesystem-safe compare artifact naming
    """
    stem = _sanitize_label(path.stem)
    parent = _sanitize_label(path.parent.name)

    if _stem_is_generic(stem) and parent:
        return parent

    return stem


def derive_compare_labels(
    lhs_path: str | Path,
    rhs_path: str | Path,
) -> Tuple[str, str]:
    """
    Derive stable display/output labels for the two compared summaries.

    Labels prefer the last useful path component and avoid duplicated
    `final_summary`-style labels whenever possible.
    """
    lhs = Path(lhs_path).expanduser().resolve()
    rhs = Path(rhs_path).expanduser().resolve()

    lhs_label = _default_label_for_path(lhs)
    rhs_label = _default_label_for_path(rhs)

    if lhs_label == rhs_label:
        lhs_parent = _sanitize_label(lhs.parent.name)
        rhs_parent = _sanitize_label(rhs.parent.name)

        if lhs_parent:
            lhs_label = f"{lhs_parent}_{lhs_label}"
        if rhs_parent:
            rhs_label = f"{rhs_parent}_{rhs_label}"

        if lhs_label == rhs_label:
            lhs_label = f"lhs_{lhs_label}"
            rhs_label = f"rhs_{rhs_label}"

    return lhs_label, rhs_label


def default_output_base(
    lhs_path: str | Path,
    rhs_path: str | Path,
    output: str | Path | None = None,
) -> Path:
    """
    Resolve the compare output base path.

    Behavior
    --------
    - If `output` is provided and non-empty, treat it as a base path.
      If it ends in `.json` or `.txt`, strip that suffix and use the stem as
      the base.
    - Otherwise generate:
        `<cwd>/compare/<lhs_label>_vs_<rhs_label>`
    """
    if output is not None:
        output_text = str(output).strip()
        if output_text:
            base = Path(output_text).expanduser()
            if base.suffix.lower() in {".json", ".txt"}:
                base = base.with_suffix("")
            return base.resolve()

    lhs_label, rhs_label = derive_compare_labels(lhs_path, rhs_path)
    compare_dir = (Path.cwd() / "compare").resolve()
    return compare_dir / f"{lhs_label}_vs_{rhs_label}"


def write_compare_artifacts(
    *,
    output_base: Path,
    payload: Dict[str, Any],
) -> Tuple[Path, Path]:
    """
    Atomically write compare JSON and text artifacts.

    Returns
    -------
    tuple[Path, Path]
        `(json_path, txt_path)`
    """
    output_base = Path(output_base).expanduser().resolve()
    json_path = output_base.with_suffix(".json")
    txt_path = output_base.with_suffix(".txt")

    write_json_atomic(json_path, payload)
    write_text_atomic(txt_path, str(payload.get("text", "")) + "\n")

    return json_path, txt_path
