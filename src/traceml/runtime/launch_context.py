"""
Helpers for preserving the user's original launch context.

TraceML should instrument user code without changing how that code resolves:

- current working directory
- sibling imports
- script argv

This module centralizes that behavior so the CLI launcher and runtime
executor stay small and consistent.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

TRACEML_LAUNCH_CWD_ENV = "TRACEML_LAUNCH_CWD"


@dataclass(frozen=True)
class LaunchContext:
    """Execution context captured from the user's original CLI invocation.

    Attributes
    ----------
    launch_cwd:
        The directory where the user invoked ``traceml``.
    """

    launch_cwd: str

    @classmethod
    def capture(cls) -> "LaunchContext":
        """Capture the current user-facing launch context.

        This should be called in the main CLI process before spawning any
        subprocesses so child processes can faithfully reproduce the user's
        original execution environment.
        """
        return cls(launch_cwd=str(Path.cwd().resolve()))

    @classmethod
    def from_env(cls) -> "LaunchContext":
        """Load a previously captured launch context from environment vars.

        Falls back to the current process cwd when no explicit launch cwd has
        been provided. This keeps behavior safe for tests and direct executor
        invocation.
        """
        raw = os.environ.get(TRACEML_LAUNCH_CWD_ENV, "").strip()
        if raw:
            return cls(launch_cwd=str(Path(raw).resolve()))
        return cls.capture()

    def to_env(self) -> Dict[str, str]:
        """Serialize this launch context into environment variables."""
        return {TRACEML_LAUNCH_CWD_ENV: self.launch_cwd}


@contextmanager
def script_execution_context(
    *,
    script_path: str,
    script_args: list[str],
    launch_context: LaunchContext,
) -> Iterator[None]:
    """Temporarily apply Python script execution semantics.

    Behavior
    --------
    - ``sys.argv`` looks like a direct script launch
    - ``sys.path[0]`` points at the script directory
    - process cwd matches the user's original launch cwd

    This combination most closely matches normal ``python train.py`` or
    ``torchrun train.py`` behavior from the user's project directory.
    """
    resolved_script_path = str(Path(script_path).resolve())
    script_dir = str(Path(resolved_script_path).parent)

    old_argv = sys.argv[:]
    old_path = sys.path[:]
    old_cwd = os.getcwd()

    try:
        sys.argv = [resolved_script_path, *script_args]

        if sys.path:
            sys.path[0] = script_dir
        else:
            sys.path = [script_dir]

        os.chdir(launch_context.launch_cwd)
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path = old_path
