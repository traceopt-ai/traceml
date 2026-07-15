# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Process-local state for one-shot runtime environment telemetry.

Runtime environment data is rank-scoped context that later travels as sampler
body rows. This module keeps first-publish-wins state separate from transport
metadata and from the detector itself.
"""

from __future__ import annotations

import time
from queue import Empty, Full, Queue
from typing import Any

from traceml_ai.runtime.environment import RuntimeEnvironmentInfo

_RUNTIME_ENVIRONMENT_QUEUE: Queue = Queue(maxsize=10)
_PUBLISHED = False


def publish_runtime_environment_once(info: RuntimeEnvironmentInfo) -> bool:
    """Publish one runtime environment record for this process.

    The first successful publish wins for the lifetime of the process. Later
    calls are ignored because strategy changes are intentionally out of scope
    for this version. Returns True only when a new row was queued.
    """
    global _PUBLISHED

    if _PUBLISHED:
        return False

    row = {
        "seq": 0,
        "ts": time.time(),
        **info.to_record(),
    }
    try:
        _RUNTIME_ENVIRONMENT_QUEUE.put_nowait(row)
    except Full:
        return False

    _PUBLISHED = True
    return True


def pop_runtime_environment_record() -> dict[str, Any] | None:
    """Return one pending runtime environment row, if present."""
    try:
        row = _RUNTIME_ENVIRONMENT_QUEUE.get_nowait()
    except Empty:
        return None
    return row if isinstance(row, dict) else None


def has_runtime_environment_info() -> bool:
    """Return True after this process has published runtime environment info."""
    return bool(_PUBLISHED)


def reset_runtime_environment_state() -> None:
    """Reset process-local runtime environment state for tests."""
    global _PUBLISHED

    while True:
        try:
            _RUNTIME_ENVIRONMENT_QUEUE.get_nowait()
        except Empty:
            break
    _PUBLISHED = False


__all__ = [
    "has_runtime_environment_info",
    "pop_runtime_environment_record",
    "publish_runtime_environment_once",
    "reset_runtime_environment_state",
]
