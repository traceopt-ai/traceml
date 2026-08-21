# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Compute entry point for the run-context payload."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .common import ContextDB, empty_context


class ContextComputer:
    """
    Produce the context payload: one flat dict of observed facts.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    stale_ttl_s:
        How long the last good payload is reused after a read failure;
        None reuses it indefinitely.
    """

    def __init__(
        self, db_path: str, stale_ttl_s: Optional[float] = 30.0
    ) -> None:
        self._db = ContextDB(db_path=db_path)
        self._stale_ttl_s = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )
        self._last_ok: Optional[Dict[str, Any]] = None
        self._last_ok_ts: float = 0.0

    def compute(self) -> Dict[str, Any]:
        """Read the facts; on a transient failure reuse the last good set."""
        try:
            with self._db.connect() as conn:
                facts = self._db.fetch_context(conn)
        except Exception:
            return self._return_stale()
        self._last_ok = facts
        self._last_ok_ts = time.time()
        return facts

    def _return_stale(self) -> Dict[str, Any]:
        if self._last_ok is not None and (
            self._stale_ttl_s is None
            or (time.time() - self._last_ok_ts) <= self._stale_ttl_s
        ):
            return dict(self._last_ok)
        return empty_context()
