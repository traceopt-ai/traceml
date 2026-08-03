"""Shared SQLite lifecycle helpers for database-backed tests."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


@contextmanager
def sqlite_database(
    path: str | Path,
    *initializers: Callable[[sqlite3.Connection], None],
) -> Iterator[sqlite3.Connection]:
    """Open, initialize, commit, and close one temporary test database."""
    conn = sqlite3.connect(path)
    try:
        for initialize in initializers:
            initialize(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()
