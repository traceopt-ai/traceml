"""
Shared utility helpers for TraceML samplers.

These helpers are intentionally narrow and infrastructure-oriented. They do not
contain sampler-specific aggregation logic.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from pathlib import Path
from queue import Empty
from typing import Any


def drain_queue_nowait(queue_obj: Any, *, skip_none: bool = True) -> list[Any]:
    """
    Drain a queue without blocking and return all currently available items.

    Notes
    -----
    - Uses `get_nowait()` instead of `queue.empty()` to avoid relying on the
      weaker `empty()` concurrency semantics.
    - Best-effort by design: unexpected queue exceptions stop draining.
    """
    items: list[Any] = []

    while True:
        try:
            item = queue_obj.get_nowait()
        except Empty:
            break
        except Exception:
            break

        if skip_none and item is None:
            continue

        items.append(item)

    return items


def append_queue_nowait_to_deque(
    queue_obj: Any,
    target: deque[Any],
    *,
    skip_none: bool = True,
) -> None:
    """
    Drain a queue and append its current items into a target deque.
    """
    for item in drain_queue_nowait(queue_obj, skip_none=skip_none):
        target.append(item)


def write_json_atomic(
    path: Path | str,
    payload: dict[str, Any],
    *,
    sort_keys: bool = False,
) -> None:
    """
    Atomically write JSON to disk to avoid partial files being observed.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=sort_keys)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, path)


def ensure_session_dir(
    *,
    logs_dir: Path | str,
    session_id: str,
    rank: int | None = None,
) -> Path:
    """
    Return a session directory path and ensure it exists.
    """
    root = Path(logs_dir).resolve() / session_id
    if rank is not None:
        root = root / str(rank)
    root.mkdir(parents=True, exist_ok=True)
    return root
