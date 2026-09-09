import threading

from torch.utils.data import DataLoader

from traceml_ai.runtime.arming import is_tracing_armed
from traceml_ai.utils.timing import timed_region

_ORIG_DATALOADER_ITER = DataLoader.__iter__

# Thread-local suppression so an integration can keep the fetches of a
# non-training loader (validation, sanity check, test, predict) out of the
# training step's Input Wait. Checked on every fetch, not once per iterator,
# because a loader created before suppression keeps being consumed after it.
_DL_TLS = threading.local()


def _suppressed() -> bool:
    return bool(getattr(_DL_TLS, "_traceml_dl_suppressed", False))


class suppress_dataloader_timing:
    """
    Context manager that pauses DataLoader fetch timing on this thread.

    Nested contexts preserve the outer context's state, mirroring
    ``backward_auto_timer``. Fetches inside the context run untimed; the
    once-per-iterator armed gate is unchanged.
    """

    def __init__(self):
        self._prev = False

    def __enter__(self):
        self._prev = _suppressed()
        _DL_TLS._traceml_dl_suppressed = True
        return self

    def __exit__(self, exc_type, exc, tb):
        _DL_TLS._traceml_dl_suppressed = self._prev
        return False


def _traceml_dataloader_iter(self):
    it = _ORIG_DATALOADER_ITER(self)

    if not is_tracing_armed():
        yield from it
        return

    while True:
        try:
            if _suppressed():
                batch = next(it)
            else:
                with timed_region(
                    name="_traceml_internal:dataloader_next",
                    scope="step",
                    record_gpu_events=True,
                ):
                    batch = next(it)
        except StopIteration:
            break

        yield batch


def patch_dataloader():
    """
    Patch torch.utils.data.DataLoader.__iter__ once.
    Safe to call multiple times.
    """
    if getattr(DataLoader, "_traceml_patched", False):
        return

    DataLoader.__iter__ = _traceml_dataloader_iter
    DataLoader._traceml_patched = True
