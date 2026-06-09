"""
DDP gradient-sync timing via ``register_comm_hook``.

What it catches
---------------
Per-bucket NCCL ``all_reduce`` wall-time during ``loss.backward()`` on a
``DistributedDataParallel`` model.  One ``TimeEvent`` per bucket per step,
aggregated per step by ``StepTimeSampler`` via the existing
``(name, device, is_gpu)`` key.

What it does NOT catch
----------------------
- User-issued ``dist.all_reduce()`` calls (no DDP gradient sync).
- FSDP ``reduce_scatter`` / ``all_gather`` (different API surface).
- Collectives outside ``loss.backward()`` (barrier, broadcast, etc.).

Wire name
---------
``_traceml_comm:ddp_grad_sync``

Implementation notes
--------------------
- Hook entry runs on the **compute stream** (autograd thread inside
  ``loss.backward()``).  ``gpu_start.record()`` lands on this stream.
- ``.then()`` callback runs on a **pool stream** (not the NCCL comm
  stream — see ``ivalue_inl.h:1180-1201``).  ``gpu_end.record()`` lands
  there, after the pool stream synchronises with the NCCL end-event.
- ``gpu_start.elapsed_time(gpu_end)`` therefore spans the NCCL kernel
  wall-time (cross-stream, device-global clock).
- ``cpu_start`` / ``cpu_end`` measure kernel-launch overhead only
  (Future resolves at NCCL enqueue, not completion).

Cross-stream sync contract (load-bearing):
    ``reducer.cpp:979-986`` TODO comment: "As long as autograd uses the
    default stream … these operations are implicitly sequenced."
    ``async_op=True`` collectives fork a comm stream synced with the
    current stream at launch.  Our event pair is well-defined.
"""

import sys
import time
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from traceml_ai.utils.cuda_event_pool import get_cuda_event, return_cuda_event
from traceml_ai.utils.timing import TimeEvent, TimeScope, record_event

_WIRE_NAME = "_traceml_comm:ddp_grad_sync"


def _traceml_ddp_comm_hook_factory(
    base_hook: Callable[..., torch.futures.Future[torch.Tensor]],
) -> Callable[..., torch.futures.Future[torch.Tensor]]:
    """
    Return an instrumented comm hook that wraps *base_hook* with timing.

    The returned closure records per-bucket CUDA events around
    ``base_hook(state, bucket)`` and chains a ``.then()`` callback to
    emit a ``TimeEvent`` when the NCCL collective completes.

    Parameters
    ----------
    base_hook:
        The hook that performs the actual collective (e.g.
        ``default_hooks.allreduce_hook``).  Must return a
        ``Future[Tensor]``.
    """

    def instrumented_hook(
        state: Any,
        bucket: dist.GradBucket,
    ) -> torch.futures.Future[torch.Tensor]:
        cpu_start = time.time()

        gpu_start: Optional[torch.cuda.Event] = None
        gpu_end: Optional[torch.cuda.Event] = None
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

        try:
            fut = base_hook(state, bucket)
        except Exception:
            # base_hook raised synchronously, before returning a Future.
            # _on_complete never runs, so the events acquired above would
            # leak from the pool.  Return them, then re-raise unchanged.
            return_cuda_event(gpu_start)
            return_cuda_event(gpu_end)
            raise

        def _on_complete(fut: Any) -> torch.Tensor:
            # .then() passes the Future itself, not the resolved value.
            # base_hook's future already resolves to the reduced bucket
            # tensor (allreduce_hook does the list-extraction internally),
            # so pass it through unchanged. DDP consumes this return value
            # as the bucket gradient -- slicing it (e.g. [0]) feeds DDP a
            # wrong-shaped tensor and silently corrupts gradients.
            result = fut.value()

            try:
                cpu_end = time.time()

                if gpu_start is not None and gpu_end is not None:
                    gpu_end.record()
                    evt = TimeEvent(
                        name=_WIRE_NAME,
                        device=device,
                        cpu_start=cpu_start,
                        cpu_end=cpu_end,
                        gpu_start=gpu_start,
                        gpu_end=gpu_end,
                        scope=TimeScope.STEP,
                    )
                else:
                    evt = TimeEvent(
                        name=_WIRE_NAME,
                        device=device,
                        cpu_start=cpu_start,
                        cpu_end=cpu_end,
                        scope=TimeScope.STEP,
                    )

                record_event(evt)

            except Exception:
                return_cuda_event(gpu_start)
                return_cuda_event(gpu_end)

            return result

        return fut.then(_on_complete)

    return instrumented_hook


def install_ddp_comm_hook(
    ddp_model: torch.nn.parallel.DistributedDataParallel,
    base_hook: Optional[
        Callable[..., torch.futures.Future[torch.Tensor]]
    ] = None,
) -> torch.nn.parallel.DistributedDataParallel:
    """
    Install TraceML timing on a DDP model's gradient-sync hook.

    Parameters
    ----------
    ddp_model:
        A ``DistributedDataParallel``-wrapped model.
    base_hook:
        Optional user-supplied comm hook whose future resolves to the
        reduced bucket tensor (e.g. ``fp16_compress_hook``).  Registered
        with ``state=None``, so ``state``-bearing hooks such as PowerSGD
        are not yet supported.  When ``None``, delegates to PyTorch's
        default ``allreduce_hook``.

    Returns
    -------
    The same ``ddp_model`` instance (in-place, like ``wrap_optimizer``).

    Notes
    -----
    - Idempotent via ``_traceml_ddp_comm_hook_installed`` sentinel.
    - PyTorch allows ONE hook per DDP instance.  If the user already
      registered a hook directly (not through ``wrap_ddp``), our
      ``register_comm_hook`` call raises ``RuntimeError``.  We catch it,
      warn to stderr, and return ``ddp_model`` unchanged (fail-open).
    - Installing ANY comm hook causes PyTorch to lose the fused
      copy+divide optimisation in ``reducer.cpp:383-384`` vs ``:417-421``.
      Cost: one extra tensor scan per bucket per step (sub-microsecond).
    """
    from torch.nn.parallel import DistributedDataParallel

    if not isinstance(ddp_model, DistributedDataParallel):
        raise TypeError(
            "install_ddp_comm_hook() expects a "
            "DistributedDataParallel instance, "
            f"got {type(ddp_model).__name__}."
        )

    if getattr(ddp_model, "_traceml_ddp_comm_hook_installed", False):
        return ddp_model

    if base_hook is None:
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
            allreduce_hook,
        )

        base_hook = allreduce_hook

    if not callable(base_hook):
        raise TypeError(
            "base_hook must be callable, " f"got {type(base_hook).__name__}."
        )

    instrumented = _traceml_ddp_comm_hook_factory(base_hook)

    try:
        ddp_model.register_comm_hook(state=None, hook=instrumented)
    except RuntimeError as exc:
        print(
            f"[TraceML] cannot register DDP comm hook: {exc}",
            file=sys.stderr,
        )
        return ddp_model

    ddp_model._traceml_ddp_comm_hook_installed = True  # type: ignore[attr-defined]
    return ddp_model


def ensure_ddp_comm_hook_installed(model: torch.nn.Module) -> None:
    """
    Auto-install DDP comm-hook timing when *model* is a DDP wrapper.

    No-op for non-DDP models.  Idempotent and best-effort, mirroring
    ``ensure_optimizer_timing_installed``: this is the auto-path entry
    point called from ``trace_step``.  Errors never propagate into the
    user's training loop.
    """
    from torch.nn.parallel import DistributedDataParallel

    if not isinstance(model, DistributedDataParallel):
        return

    # Auto-path (called from the init() forward patch on every DDP forward):
    # instrumentation must never break training, so swallow and report any
    # failure. The explicit wrap_ddp() path calls install_ddp_comm_hook
    # directly and still surfaces errors to the caller.
    try:
        install_ddp_comm_hook(model)
    except Exception as exc:
        print(
            f"[TraceML] DDP comm-hook auto-install failed: {exc}",
            file=sys.stderr,
        )
