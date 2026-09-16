# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utility helpers for TraceML samplers.

These helpers are intentionally narrow and infrastructure-oriented. They do not
contain sampler-specific aggregation logic.
"""

from __future__ import annotations

from pathlib import Path

from traceml_ai.runtime.session import rank_dir_name


def ensure_session_dir(
    *,
    logs_dir: Path | str,
    session_id: str,
    rank: int | None = None,
) -> Path:
    """
    Return a session directory path and ensure it exists.

    ``rank`` is the global distributed rank. It is used only for process-owned
    files, where local rank would collide across nodes.
    """
    root = Path(logs_dir).resolve() / session_id
    if rank is not None:
        root = root / rank_dir_name(rank)
    root.mkdir(parents=True, exist_ok=True)
    return root
