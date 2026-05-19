# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from traceml.runtime.identity import resolve_runtime_identity


def _load_torch() -> Optional[Any]:
    """Import torch only when distributed state needs it."""
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def get_ddp_info():
    """
    Detect whether we are running under DistributedDataParallel (DDP).

    Returns:
        is_ddp (bool): True if running with multiple processes.
        local_rank (int): Rank-local GPU index. -1 if not in DDP.
        world_size (int): Number of processes. 1 if not in DDP.
    """
    identity = resolve_runtime_identity(torch_loader=_load_torch)
    is_ddp = identity.is_distributed

    # Preserve the historical API: non-DDP reports local_rank=-1.
    local_rank = identity.local_rank if is_ddp else -1
    return is_ddp, local_rank, identity.world_size
