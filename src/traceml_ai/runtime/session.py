# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""TraceML session identifiers and session-local path helpers."""

from __future__ import annotations

import datetime
import uuid


def _generate_session_id():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"session_{ts}_{rand}"


def get_session_id():
    _SESSION_ID = _generate_session_id()
    return _SESSION_ID


def rank_dir_name(global_rank: int) -> str:
    """
    Return the process-owned directory name for a distributed rank.

    TraceML uses global rank for filesystem isolation because it is unique
    across all nodes. Local rank is only unique inside one node.
    """
    return f"rank_{max(0, int(global_rank))}"
