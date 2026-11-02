# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime context helpers shared by TraceML samplers.

This module centralizes environment-derived session and distributed context so
samplers do not each re-implement the same environment parsing logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from traceml_ai.runtime.identity import (
    RuntimeIdentity,
    resolve_runtime_identity,
)


@dataclass(frozen=True)
class SamplerRuntimeContext:
    """
    Environment-derived TraceML runtime context.
    """

    session_id: str
    logs_dir: Path
    identity: RuntimeIdentity

    @property
    def session_root(self) -> Path:
        """
        Root directory for the active TraceML session.
        """
        if not self.session_id or not str(self.logs_dir):
            return Path()
        return self.logs_dir / self.session_id


def resolve_runtime_context() -> SamplerRuntimeContext:
    """
    Resolve the current TraceML runtime context from environment variables.
    """
    logs_dir = os.environ.get("TRACEML_LOGS_DIR", "").strip()
    session_id = os.environ.get("TRACEML_SESSION_ID", "").strip()

    return SamplerRuntimeContext(
        session_id=session_id,
        logs_dir=Path(logs_dir).resolve() if logs_dir else Path(),
        identity=resolve_runtime_identity(),
    )
