import threading
from collections.abc import Callable

from torch.utils.data import DataLoader

from traceml_ai.runtime.arming import is_tracing_armed
from traceml_ai.utils.timing import timed_region

_ORIG_DATALOADER_ITER = DataLoader.__iter__

# Thread-local suppression so an integration can keep the fetches of a
# non-training loader (validation, sanity check, test, predict) out of the
# training step's Input Wait. Checked on every fetch, not once per iterator,
# because a loader created before suppression keeps being consumed after it.
_DL_TLS = threading.local()


def _depth() -> int:
    return getattr(_DL_TLS, "_traceml_dl_depth", 0)


def _suppressed() -> bool:
    return _depth() > 0


def _timing_allowed() -> bool:
    """Return whether the current thread should record this fetch."""
    if _suppressed():
        return False

    enabled = getattr(_DL_TLS, "_traceml_dl_enabled", None)
    if enabled is None:
        return True
    try:
        return bool(enabled())
    except Exception:
        # Instrumentation policy must never interrupt the DataLoader.
        return False


class suppress_dataloader_timing:
    """
    Context manager that pauses DataLoader fetch timing on this thread.

    Refcounted per thread, so contexts may nest or interleave: fetches are
    untimed while any context is open and timing resumes when the last one
    exits, whatever the exit order. Lightning runs every callback's start
    hooks in registration order and the end hooks in the same order, so two
    holders interleave rather than nest. Exiting a context twice is a no-op.
    The once-per-iterator armed gate is unchanged.

    The count is thread-local, like the H2D timer's flag: it covers fetches
    made on the thread that entered the context, which is the thread
    Lightning runs its loops and hooks on. Enter and exit on the same thread.
    """

    def __init__(self):
        self._active = False

    def __enter__(self):
        if not self._active:
            self._active = True
            _DL_TLS._traceml_dl_depth = _depth() + 1
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._active:
            self._active = False
            _DL_TLS._traceml_dl_depth = max(0, _depth() - 1)
        return False


class dataloader_timing_scope:
    """Limit fetch timing to calls for which ``enabled`` returns true.

    Unlike :class:`suppress_dataloader_timing`, this is intended to remain
    active across a framework run. The predicate is evaluated for every
    ``next`` call, which lets integrations distinguish training fetches from
    evaluation prefetches even when the framework creates both kinds of
    iterator internally. Scopes are thread-local, support normal nesting, and
    must be entered and exited on the thread that consumes the DataLoader.

    A predicate failure disables timing for that fetch rather than affecting
    user code.
    """

    def __init__(self, enabled: Callable[[], bool]):
        if not callable(enabled):
            raise TypeError("enabled must be callable")
        self.enabled = enabled
        self._active = False
        self._previous = None

    def __enter__(self):
        if not self._active:
            self._active = True
            self._previous = getattr(_DL_TLS, "_traceml_dl_enabled", None)
            _DL_TLS._traceml_dl_enabled = self.enabled
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._active:
            self._active = False
            _DL_TLS._traceml_dl_enabled = self._previous
            self._previous = None
        return False


def _traceml_dataloader_iter(self):
    it = _ORIG_DATALOADER_ITER(self)

    if not is_tracing_armed():
        yield from it
        return

    while True:
        try:
            if not _timing_allowed():
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
