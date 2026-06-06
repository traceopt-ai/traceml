# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Small stdlib-only helpers for atomic file writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


def write_json_atomic(
    path: Path | str,
    payload: dict[str, Any],
    *,
    sort_keys: bool = False,
) -> None:
    """Write JSON atomically so readers never observe a partial file."""
    _write_atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=sort_keys),
    )


def write_text_atomic(path: Path | str, text: str) -> None:
    """Write text atomically so readers never observe a partial file."""
    _write_atomic_text(path, text)


def _write_atomic_text(path: Path | str, text: str) -> None:
    """Write text through a sibling temporary file, then replace in place."""
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
            tmp.write(text)
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


__all__ = [
    "write_json_atomic",
    "write_text_atomic",
]
