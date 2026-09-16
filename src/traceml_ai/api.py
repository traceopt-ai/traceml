"""Public TraceML module-level API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ContextManager, Dict, Optional

from traceml_ai.sdk.initial import TraceMLInitConfig
from traceml_ai.sdk.summary_client import final_summary as _final_summary
from traceml_ai.sdk.summary_client import summary as _summary

if TYPE_CHECKING:
    import torch.nn as nn


def trace_step(model: nn.Module) -> ContextManager[None]:
    """Return the context manager that defines one training-step boundary.

    Parameters
    ----------
    model:
        The PyTorch model being trained. TraceML uses it to collect step-memory
        measurements and associate automatic phase timing with the step.

    Returns
    -------
    ContextManager[None]
        Use around the work from ``optimizer.zero_grad(...)`` through
        ``optimizer.step()``.

    Notes
    -----
    When tracing is disabled, the context manager is a no-op. Exceptions from
    the training body still propagate normally.
    """
    from traceml_ai.sdk import trace_step as _trace_step

    return _trace_step(model)


def final_summary(
    *,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.1,
    print_text: bool = False,
    rank0_only: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return the full final-summary payload for the active session.

    Parameters
    ----------
    timeout_sec:
        Maximum time to wait for the aggregator to publish the summary.
    poll_interval_sec:
        Delay between checks for the aggregator response.
    print_text:
        Print the matching ``final_summary.txt`` artifact when available.
    rank0_only:
        Return ``None`` on non-primary ranks when true.

    Returns
    -------
    dict or None
        The parsed ``final_summary.json`` payload, or ``None`` for a non-primary
        rank when ``rank0_only=True``.

    Raises
    ------
    RuntimeError
        If session history is unavailable, the aggregator returns an error, or
        the request times out.
    """
    return _final_summary(
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
        print_text=print_text,
        rank0_only=rank0_only,
    )


def summary(
    *,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.1,
    print_text: bool = False,
    rank0_only: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return a compact tracker-friendly summary for the active session.

    Parameters
    ----------
    timeout_sec:
        Maximum time to wait for the aggregator to publish the summary.
    poll_interval_sec:
        Delay between checks for the aggregator response.
    print_text:
        Print the matching ``final_summary.txt`` artifact when available.
    rank0_only:
        Return ``None`` on non-primary ranks when true.

    Returns
    -------
    dict or None
        A flat projection of the final summary for experiment trackers, or
        ``None`` for a non-primary rank when ``rank0_only=True``.

    Raises
    ------
    RuntimeError
        If session history is unavailable, the aggregator returns an error, or
        the request times out.
    """
    return _summary(
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
        print_text=print_text,
        rank0_only=rank0_only,
    )


def wrap_dataloader_fetch(obj: Any) -> Any:
    """Wrap a loader or iterator to time step-scoped batch fetches.

    Parameters
    ----------
    obj:
        A loader implementing ``__iter__`` or an iterator implementing
        ``__next__``.

    Returns
    -------
    Any
        A loader or iterator proxy that records fetch time.

    Raises
    ------
    TypeError
        If ``obj`` is neither a loader nor an iterator.
    RuntimeError
        If a torch ``DataLoader`` is already automatically instrumented.
    """
    from traceml_ai.sdk import wrap_dataloader_fetch as _wrap_dataloader_fetch

    return _wrap_dataloader_fetch(obj)


def wrap_forward(model: nn.Module) -> nn.Module:
    """Wrap one model instance's ``forward(...)`` for step-scoped timing.

    Parameters
    ----------
    model:
        The PyTorch module to instrument.

    Returns
    -------
    torch.nn.Module
        The same model instance after its ``forward`` method is wrapped in
        place.

    Raises
    ------
    TypeError
        If ``model`` is not a PyTorch module with a callable ``forward``.
    RuntimeError
        If automatic forward instrumentation is active or the instance cannot
        be wrapped safely.
    """
    from traceml_ai.sdk import wrap_forward as _wrap_forward

    return _wrap_forward(model)


def wrap_backward(loss: Any) -> Any:
    """Wrap a loss-like object so ``.backward(...)`` records timing.

    Parameters
    ----------
    loss:
        An object with a callable ``backward`` method, normally a loss tensor.

    Returns
    -------
    Any
        A proxy that forwards attributes and times its ``backward`` call.

    Raises
    ------
    TypeError
        If ``loss`` has no callable ``backward`` method.
    RuntimeError
        If automatic backward instrumentation is active.
    """
    from traceml_ai.sdk import wrap_backward as _wrap_backward

    return _wrap_backward(loss)


def wrap_optimizer(optimizer: Any) -> Any:
    """Wrap one optimizer instance's ``.step(...)`` for timing.

    Parameters
    ----------
    optimizer:
        An optimizer with a callable ``step`` method.

    Returns
    -------
    Any
        The same optimizer instance after its ``step`` method is wrapped in
        place.

    Raises
    ------
    TypeError
        If ``optimizer`` has no callable ``step`` method.
    RuntimeError
        If automatic optimizer instrumentation is active or the instance
        cannot be wrapped safely.
    """
    from traceml_ai.sdk import wrap_optimizer as _wrap_optimizer

    return _wrap_optimizer(optimizer)


def wrap_h2d(obj: Any) -> Any:
    """Wrap an object so qualifying CPU-to-CUDA ``.to(...)`` calls are timed.

    Parameters
    ----------
    obj:
        A tensor or batch object with a callable ``.to(...)`` method.

    Returns
    -------
    Any
        A proxy that forwards attributes and returns the transferred object on
        its first ``.to(...)`` call.

    Raises
    ------
    TypeError
        If ``obj`` has no callable ``.to(...)`` method.

    Notes
    -----
    Use this in manual or selective mode. If automatic H2D timing becomes
    active after wrapping, the proxy passes through to avoid double counting.
    """
    from traceml_ai.sdk import wrap_h2d as _wrap_h2d

    return _wrap_h2d(obj)


def init(
    *,
    mode: str = "auto",
    patch_dataloader: Optional[bool] = None,
    patch_forward: Optional[bool] = None,
    patch_backward: Optional[bool] = None,
    patch_h2d: Optional[bool] = None,
    disabled: Optional[bool] = None,
    ui_mode: Optional[str] = None,
    interval: Optional[float] = None,
    logs_dir: Optional[str] = None,
    enable_logging: Optional[bool] = None,
    session_id: Optional[str] = None,
    aggregator_host: Optional[str] = None,
    aggregator_port: Optional[int] = None,
    connect_timeout_sec: float = 10.0,
    connect_retry_interval_sec: float = 0.25,
    on_missing_aggregator: Optional[str] = None,
) -> TraceMLInitConfig:
    """Initialize TraceML and start its runtime for this process.

    Parameters
    ----------
    mode:
        Instrumentation policy: ``"auto"`` (default), ``"manual"``, or
        ``"selective"``. ``"custom"`` is accepted as an alias for selective.
    patch_dataloader, patch_forward, patch_backward, patch_h2d:
        Per-feature automatic-instrumentation settings. They may be supplied
        only with ``mode="selective"``.
    disabled:
        Disable TraceML entirely for this process.
    ui_mode:
        Display mode: ``"summary"``, ``"cli"``, or ``"dashboard"``.
    interval:
        Sampling interval in seconds for runtime telemetry.
    logs_dir:
        Directory for session artifacts.
    enable_logging:
        Enable TraceML file logging.
    session_id:
        Identifier shared by workers that write one run's artifacts.
    aggregator_host, aggregator_port:
        Aggregator endpoint for direct launches. ``traceml run`` configures it
        automatically.
    connect_timeout_sec, connect_retry_interval_sec:
        Bounded connection wait and retry interval for the aggregator.
    on_missing_aggregator:
        Missing-aggregator policy: ``"warn"`` prints one stderr warning and
        continues with tracing disabled; ``"raise"`` fails the call. Resolution
        is explicit argument, then ``TRACEML_ON_MISSING_AGGREGATOR``, then
        ``"warn"``.

    Returns
    -------
    TraceMLInitConfig
        The effective instrumentation configuration. Fail-open initialization
        returns a disabled configuration.

    Raises
    ------
    ValueError
        If mode or selective patch settings are invalid.
    RuntimeError
        If initialization conflicts with an existing configuration, or when
        ``on_missing_aggregator="raise"`` is selected and setup fails.

    """
    from traceml_ai.sdk.initial import init as _init

    return _init(
        mode=mode,
        patch_dataloader=patch_dataloader,
        patch_forward=patch_forward,
        patch_backward=patch_backward,
        patch_h2d=patch_h2d,
        disabled=disabled,
        ui_mode=ui_mode,
        interval=interval,
        logs_dir=logs_dir,
        enable_logging=enable_logging,
        session_id=session_id,
        aggregator_host=aggregator_host,
        aggregator_port=aggregator_port,
        connect_timeout_sec=connect_timeout_sec,
        connect_retry_interval_sec=connect_retry_interval_sec,
        on_missing_aggregator=on_missing_aggregator,
    )


def start(
    *,
    mode: str = "auto",
    patch_dataloader: Optional[bool] = None,
    patch_forward: Optional[bool] = None,
    patch_backward: Optional[bool] = None,
    patch_h2d: Optional[bool] = None,
    disabled: Optional[bool] = None,
    ui_mode: Optional[str] = None,
    interval: Optional[float] = None,
    logs_dir: Optional[str] = None,
    enable_logging: Optional[bool] = None,
    session_id: Optional[str] = None,
    aggregator_host: Optional[str] = None,
    aggregator_port: Optional[int] = None,
    connect_timeout_sec: float = 10.0,
    connect_retry_interval_sec: float = 0.25,
    on_missing_aggregator: Optional[str] = None,
) -> TraceMLInitConfig:
    """Backward-compatible alias for :func:`init`.

    This function accepts the same parameters and has the same missing-
    aggregator behavior as :func:`init`. New code may use either name; prefer
    ``init`` for consistency with framework integration entry points.
    """
    from traceml_ai.sdk.initial import start as _start

    return _start(
        mode=mode,
        patch_dataloader=patch_dataloader,
        patch_forward=patch_forward,
        patch_backward=patch_backward,
        patch_h2d=patch_h2d,
        disabled=disabled,
        ui_mode=ui_mode,
        interval=interval,
        logs_dir=logs_dir,
        enable_logging=enable_logging,
        session_id=session_id,
        aggregator_host=aggregator_host,
        aggregator_port=aggregator_port,
        connect_timeout_sec=connect_timeout_sec,
        connect_retry_interval_sec=connect_retry_interval_sec,
        on_missing_aggregator=on_missing_aggregator,
    )


__all__ = [
    "trace_step",
    "summary",
    "final_summary",
    "start",
    "init",
    "wrap_dataloader_fetch",
    "wrap_forward",
    "wrap_backward",
    "wrap_optimizer",
    "wrap_h2d",
]
