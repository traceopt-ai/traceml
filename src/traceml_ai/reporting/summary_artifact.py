# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for reading TraceML summary JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def resolve_summary_artifact_path(path: str | Path) -> Path:
    """Resolve a user-provided summary artifact path."""
    return Path(path).expanduser().resolve()


def load_summary_artifact(path: str | Path) -> Dict[str, Any]:
    """
    Load one TraceML summary JSON artifact as a strict JSON object.

    Raises
    ------
    RuntimeError
        If the path does not exist, is not a file, cannot be read as JSON, or
        does not contain a JSON object.
    """
    resolved = resolve_summary_artifact_path(path)

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

    return payload


def extract_summary_text(
    payload: Dict[str, Any],
    *,
    path: str | Path | None = None,
) -> str:
    """
    Return printable terminal summary text from a loaded summary artifact.

    The top-level ``text`` field is the canonical final-summary terminal
    rendering. A top-level ``card`` fallback supports older/simple summary-card
    JSON artifacts.
    """
    for key in ("text", "card"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.rstrip("\r\n")

    suffix = f": {resolve_summary_artifact_path(path)}" if path else ""
    raise RuntimeError(f"Summary file does not contain printable text{suffix}")


__all__ = [
    "extract_summary_text",
    "load_summary_artifact",
    "resolve_summary_artifact_path",
]
