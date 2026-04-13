"""
Runtime context helpers shared by TraceML samplers.

This module centralizes environment-derived session and distributed context so
samplers do not each re-implement the same environment parsing logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    """
    Best-effort integer read from the environment.
    """
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


@dataclass(frozen=True)
class SamplerRuntimeContext:
    """
    Environment-derived TraceML runtime context.
    """

    session_id: str
    logs_dir: Path
    rank: int
    local_rank: int
    world_size: int

    @property
    def session_root(self) -> Path:
        """
        Root directory for the active TraceML session.
        """
        if not self.session_id or not str(self.logs_dir):
            return Path()
        return self.logs_dir / self.session_id

    @property
    def rank_root(self) -> Path:
        """
        Rank-local directory under the active session root.
        """
        return self.session_root / str(self.local_rank)

    @property
    def is_ddp_intended(self) -> bool:
        """
        Return True when the launcher indicates multi-process execution.
        """
        return self.world_size > 1


def resolve_runtime_context() -> SamplerRuntimeContext:
    """
    Resolve the current TraceML runtime context from environment variables.
    """
    logs_dir = os.environ.get("TRACEML_LOGS_DIR", "").strip()
    session_id = os.environ.get("TRACEML_SESSION_ID", "").strip()

    return SamplerRuntimeContext(
        session_id=session_id,
        logs_dir=Path(logs_dir).resolve() if logs_dir else Path(),
        rank=_env_int("RANK", -1),
        local_rank=_env_int("LOCAL_RANK", -1),
        world_size=_env_int("WORLD_SIZE", 1),
    )
