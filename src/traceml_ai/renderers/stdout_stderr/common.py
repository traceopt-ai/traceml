"""
Shared SQLite helpers for stdout/stderr rendering.

This module keeps stdout/stderr renderer code small and provides a stable,
short-lived SQLite read path for tailing recent output lines.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class StdoutStderrLine:
    """
    One rendered stdout/stderr line from SQLite history.
    """

    id: int
    rank: Optional[int]
    line: str


class StdoutStderrDB:
    """
    Lightweight SQLite helper for stdout/stderr rendering.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)

    def connect(self) -> sqlite3.Connection:
        """
        Open a short-lived SQLite read connection.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_latest_lines(
        self,
        conn: sqlite3.Connection,
        *,
        rank: int = 0,
        limit: int = 50,
    ) -> List[StdoutStderrLine]:
        """
        Fetch the most recent stdout/stderr lines for one rank.

        Results are returned in ascending display order, even though the inner
        query limits from the newest end first.
        """
        try:
            rows = conn.execute(
                """
                SELECT id, rank, line
                FROM (
                    SELECT id, rank, line
                    FROM stdout_stderr_samples
                    WHERE rank = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC;
                """,
                (int(rank), int(limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            # Table may not exist yet early in a run. Treat as "no data yet".
            return []

        out: List[StdoutStderrLine] = []
        for row in rows:
            try:
                out.append(
                    StdoutStderrLine(
                        id=int(row["id"]),
                        rank=(
                            int(row["rank"])
                            if row["rank"] is not None
                            else None
                        ),
                        line=str(row["line"] or ""),
                    )
                )
            except Exception:
                continue

        return out
