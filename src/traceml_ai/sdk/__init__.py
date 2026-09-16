from traceml_ai.sdk.initial import (
    TraceMLInitConfig,
    get_init_config,
    init,
    is_initialized,
    start,
)
from traceml_ai.sdk.summary_client import final_summary, summary
from traceml_ai.utils.torch_support import (
    TORCH_REQUIRED_HINT,
    is_missing_torch,
)

try:
    from traceml_ai.sdk.instrumentation import (
        TraceSessionState,
        TraceState,
        get_trace_session_state,
        trace_step,
        trace_time,
    )
    from traceml_ai.sdk.wrappers import (
        wrap_backward,
        wrap_dataloader_fetch,
        wrap_forward,
        wrap_h2d,
        wrap_optimizer,
    )
except ImportError as exc:
    # Torch-free install: monitoring (watch) and post-hoc commands work
    # without torch. Step instrumentation does not; calling a trace API
    # raises a clear error instead of crashing at import time.
    #
    # Only torch's own absence is handled here. Any other failing import
    # inside these modules is a real regression and keeps propagating.
    if not is_missing_torch(exc):
        raise

    def _torch_required(*_args, **_kwargs):
        raise RuntimeError(TORCH_REQUIRED_HINT)

    class _TorchRequiredType:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(TORCH_REQUIRED_HINT)

    TraceSessionState = _TorchRequiredType  # type: ignore[assignment,misc]
    TraceState = _TorchRequiredType  # type: ignore[assignment,misc]
    get_trace_session_state = _torch_required
    trace_step = _torch_required
    trace_time = _torch_required
    wrap_backward = _torch_required
    wrap_dataloader_fetch = _torch_required
    wrap_forward = _torch_required
    wrap_h2d = _torch_required
    wrap_optimizer = _torch_required

__all__ = [
    "TraceMLInitConfig",
    "get_init_config",
    "is_initialized",
    "init",
    "start",
    "TraceSessionState",
    "TraceState",
    "get_trace_session_state",
    "trace_step",
    "trace_time",
    "summary",
    "final_summary",
    "wrap_dataloader_fetch",
    "wrap_forward",
    "wrap_backward",
    "wrap_h2d",
    "wrap_optimizer",
]
