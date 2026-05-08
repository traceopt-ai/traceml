import os
from typing import Any, Optional


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
    # Defaults for single-GPU / single-process workloads
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Additional safety: PyTorch launcher may set RANK even if LOCAL_RANK missing
    rank = int(os.environ.get("RANK", "-1"))
    torch = _load_torch()
    torch_dist_ready = False
    if torch is not None:
        try:
            torch_dist_ready = bool(
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
        except Exception:
            torch_dist_ready = False

    is_ddp = (
        world_size > 1 or local_rank != -1 or rank != -1 or torch_dist_ready
    )

    # local_rank assignment if missing but DDP detected
    if is_ddp:
        if local_rank == -1:
            # torchrun always sets LOCAL_RANK
            # mp.spawn often sets RANK
            try:
                if torch is None:
                    raise RuntimeError("torch unavailable")
                local_rank = (
                    int(os.environ["RANK"]) % torch.cuda.device_count()
                )
            except Exception:
                local_rank = 0

    return is_ddp, local_rank, world_size
