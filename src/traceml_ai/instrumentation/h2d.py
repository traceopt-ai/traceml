from __future__ import annotations

from typing import Any, Optional

import torch


def device_type(value: Any) -> Optional[str]:
    """Return the device type for a .to(...) argument, if it encodes one."""
    if isinstance(value, torch.device):
        return value.type

    if isinstance(value, torch.Tensor):
        return value.device.type

    if isinstance(value, str):
        try:
            return torch.device(value).type
        except (RuntimeError, TypeError):
            return None

    return None


def is_cuda_target(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """
    Return True when a .to(...) call targets CUDA.

    Handles:
      x.to("cuda")
      x.to(torch.device("cuda"))
      x.to(device="cuda")
      x.to(other_cuda_tensor)
    """
    first = args[0] if args else None

    if device_type(first) == "cuda":
        return True

    if device_type(kwargs.get("device")) == "cuda":
        return True

    return False


def should_time_h2d(
    obj: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> bool:
    """
    Return True when a .to(...) call should be timed as H2D.

    H2D means host/input tensor movement to CUDA inside the training step.
    It intentionally excludes CPU-only moves, dtype-only casts, D2H, D2D, and
    Parameter movement from model.to(...).
    """
    if not is_cuda_target(args, kwargs):
        return False

    if isinstance(obj, torch.nn.Parameter):
        return False

    if isinstance(obj, torch.Tensor):
        return not obj.is_cuda

    # Manual custom batch wrappers: user explicitly wrapped this object and the
    # target is CUDA, but we cannot inspect internal tensor locations safely.
    return True
