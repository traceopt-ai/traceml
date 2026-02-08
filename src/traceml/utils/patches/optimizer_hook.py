import time
import torch
from torch.optim.optimizer import (
    register_optimizer_step_pre_hook,
    register_optimizer_step_post_hook,
)

from traceml.utils.timing import TimeEvent, TimeScope, record_event
from traceml.utils.cuda_event_pool import get_cuda_event


# Per-optimizer in-flight timing state
_OPT_INFLIGHT = {}
_HANDLES = None


def install_optimizer_time_hooks() -> None:
    """
    Install async optimizer step timing.

    - Emits a TimeEvent named `_traceml_internal:optimizer_step`
    - Uses CUDA events without synchronization
    - GPU timing is resolved later by the sampler
    - Safe for AMP, non-AMP, DDP
    - Idempotent
    """
    global _HANDLES
    if _HANDLES is not None:
        return

    def pre_hook(optimizer, args, kwargs):
        cpu_start = time.time()

        gpu_start = gpu_end = None
        device = "cpu"

        if torch.cuda.is_available():
            try:
                device = f"cuda:{torch.cuda.current_device()}"
                gpu_start = get_cuda_event()
                gpu_end = get_cuda_event()
                gpu_start.record()
            except Exception:
                gpu_start = gpu_end = None
                device = "cpu"

        _OPT_INFLIGHT[id(optimizer)] = (
            cpu_start,
            gpu_start,
            gpu_end,
            device,
        )

    def post_hook(optimizer, args, kwargs):
        rec = _OPT_INFLIGHT.pop(id(optimizer), None)
        if rec is None:
            return

        cpu_start, gpu_start, gpu_end, device = rec
        cpu_end = time.time()

        try:
            if gpu_start and gpu_end:
                gpu_end.record()
                evt = TimeEvent(
                    name="_traceml_internal:optimizer_step",
                    device=device,
                    cpu_start=cpu_start,
                    cpu_end=cpu_end,
                    gpu_start=gpu_start,
                    gpu_end=gpu_end,
                    scope=TimeScope.STEP,
                )
            else:
                evt = TimeEvent(
                    name="_traceml_internal:optimizer_step",
                    device="cpu",
                    cpu_start=cpu_start,
                    cpu_end=cpu_end,
                    scope=TimeScope.STEP,
                )

            record_event(evt)

        except Exception:
            # Absolutely nothing here may break training
            pass

    _HANDLES = (
        register_optimizer_step_pre_hook(pre_hook),
        register_optimizer_step_post_hook(post_hook),
    )


def ensure_optimizer_timing_installed() -> None:
    import torch
    if getattr(torch.optim.Optimizer, "_traceml_opt_hooks_installed", False):
        return
    install_optimizer_time_hooks()
    torch.optim.Optimizer._traceml_opt_hooks_installed = True