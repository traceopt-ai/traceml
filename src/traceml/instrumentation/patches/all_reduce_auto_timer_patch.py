"""TraceML all_reduce communication auto-timer patch (TRA-16 v0).

Patches ``torch.distributed.all_reduce`` so every collective issued inside an
active ``trace_step`` is timed and recorded as the
``_traceml_comm:all_reduce`` event. Outside an active step, the patched
wrapper fast-paths to the original function with one TLS attribute lookup.

What this patch DOES catch
--------------------------
- User-issued ``dist.all_reduce(tensor, ...)`` calls inside ``trace_step``.
  This includes:
  * default process group calls
  * sub-group calls (``dist.new_group([...])``-style)
  * tensor-parallel all-reduces on dedicated TP groups
  * both ``async_op=False`` (default, synchronous) and ``async_op=True``
    (returns a Work handle). For ``async_op=True``, ``cpu_*`` reflects only
    the enqueue cost; ``gpu_*`` (resolved later by the sampler) reflects the
    on-stream kernel wall time.

What this patch does NOT catch
------------------------------
- DDP gradient-sync per-bucket all-reduce. The C++ ``Reducer`` schedules
  these directly through ``c10d::ProcessGroup::allreduce`` and never
  re-enters the Python ``torch.distributed.all_reduce`` symbol. This is a
  load-bearing limitation of the v0 patch site and is documented in the
  TRA-16 design note. v1 plan: integrate via ``register_comm_hook`` on the
  DDP wrapper (per-DDP-instance opt-in, fires from a Python hook for every
  bucket).
- Convenience APIs that bypass the Python symbol (e.g. a hypothetical future
  in-place ``tensor.all_reduce_()``). None exist on PyTorch 2.x today.
- Callers that resolve the function via the internal module path
  ``from torch.distributed.distributed_c10d import all_reduce`` (instead of
  ``import torch.distributed as dist; dist.all_reduce(...)``). Reassigning
  ``torch.distributed.all_reduce`` does NOT update
  ``torch.distributed.distributed_c10d.all_reduce``. The public re-export
  ``dist.all_reduce`` is the dominant call path (including FSDP and standard
  user code) and IS caught.
- Other collectives (``reduce_scatter``, ``all_gather``, ``barrier``). Each
  collective gets its own patch + wire-name in the planned v1 expansion.

Wire name
---------
``_traceml_comm:all_reduce`` -- communication events live under the
``_traceml_comm:`` namespace, distinct from internal-pipeline events like
``_traceml_internal:forward_time``. Future collectives (``reduce_scatter``,
``all_gather``, ...) will share this namespace as siblings.

Implementation notes
--------------------
- The patched callable lives in ``torch.distributed.c10d_logger`` on
  PyTorch 2.x and is re-exported as ``torch.distributed.all_reduce``. The
  re-exported symbol is the public attribute users import; reassigning it
  is the standard family approach.
- The TLS gate is defensive (DDP init uses ``broadcast``, not ``all_reduce``,
  so init traffic is silent even without the gate). We keep the gate for
  family symmetry with forward / backward and to silence any future
  PyTorch caller that fires ``all_reduce`` outside the training step.
- No depth counter: each ``dist.all_reduce`` call is independent; we never
  see nested all_reduce-from-within-all_reduce on a single rank.
- ``use_gpu=True`` -- with NCCL on CUDA, the kernel runs on
  ``torch.cuda.current_stream()`` (the compute stream) for user-issued
  calls and the bracketing CUDA events on that stream resolve correctly via
  the sampler's async ``try_resolve()``. With gloo on CPU (used in tests),
  ``timed_region`` falls back to its no-CUDA branch and only ``cpu_*`` is
  recorded.
"""

from __future__ import annotations

import threading
from typing import Any

import torch.distributed as dist

from traceml.utils.timing import timed_region

_TLS = threading.local()
_ORIG_ALL_REDUCE = dist.all_reduce


def _enabled() -> bool:
    """Return True when the activator is currently open on this thread."""
    return bool(getattr(_TLS, "_traceml_all_reduce_enabled", False))


def _traceml_all_reduce(*args: Any, **kwargs: Any) -> Any:
    """Patched dispatch that times user-issued ``dist.all_reduce`` calls."""
    if not _enabled():
        return _ORIG_ALL_REDUCE(*args, **kwargs)

    with timed_region("_traceml_comm:all_reduce", scope="step", use_gpu=True):
        return _ORIG_ALL_REDUCE(*args, **kwargs)


def patch_all_reduce() -> None:
    """Patch ``torch.distributed.all_reduce`` once. Safe to call multiple
    times.

    The sentinel attribute ``torch.distributed._traceml_all_reduce_patched``
    on the ``torch.distributed`` module ensures repeated calls are no-ops.
    Mirrors the family pattern (``nn.Module._traceml_forward_patched``,
    ``torch._traceml_backward_patched``, ``DataLoader._traceml_patched``).
    """
    if getattr(dist, "_traceml_all_reduce_patched", False):
        return
    dist.all_reduce = _traceml_all_reduce  # type: ignore[assignment]
    dist._traceml_all_reduce_patched = True  # type: ignore[attr-defined]


class all_reduce_auto_timer:
    """Enables all_reduce comm timing during its scope.

    Assumes ``patch_all_reduce()`` has been called once at startup / runtime
    init. ``trace_step`` opens this activator alongside the existing
    forward / backward activators.
    """

    def __enter__(self) -> "all_reduce_auto_timer":
        _TLS._traceml_all_reduce_enabled = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _TLS._traceml_all_reduce_enabled = False
        return False
