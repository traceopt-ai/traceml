# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for TraceML's optional torch dependency.

TraceML monitors host and process health without torch, and instruments
training steps with it. The boundary is only honest if "torch is absent"
is detected precisely: a bare ``except ImportError`` around a torch-backed
module would also swallow an unrelated import regression inside it and
report that as a missing torch install.
"""

from __future__ import annotations

TORCH_REQUIRED_HINT = (
    "TraceML step instrumentation requires torch. "
    "Install it with: pip install 'traceml-ai[torch]'"
)


def is_missing_torch(exc: ImportError) -> bool:
    """Return whether ``exc`` reports torch itself as the missing module.

    Any other failing import is a real error and must keep propagating.
    """
    name = getattr(exc, "name", None) or ""
    return name == "torch" or name.startswith("torch.")


def torch_available() -> bool:
    """Return whether torch can be imported in this environment."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        if not is_missing_torch(exc):
            raise
        return False
    return True


__all__ = [
    "TORCH_REQUIRED_HINT",
    "is_missing_torch",
    "torch_available",
]
