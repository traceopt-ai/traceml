"""Ownership policy for public manual instrumentation wrappers."""

from __future__ import annotations

from typing import Literal

from traceml_ai.sdk import initial
from traceml_ai.sdk.initial import TraceMLInitConfig

InstrumentationFeature = Literal[
    "dataloader_fetch",
    "forward",
    "backward",
    "optimizer",
    "h2d",
]

_PATCH_FLAG_BY_FEATURE: dict[InstrumentationFeature, str] = {
    "dataloader_fetch": "patch_dataloader",
    "forward": "patch_forward",
    "backward": "patch_backward",
    "h2d": "patch_h2d",
}


def _automatically_owned(
    config: TraceMLInitConfig,
    feature: InstrumentationFeature,
) -> bool:
    """Return whether the declared config owns ``feature`` automatically."""
    if config.disabled:
        return False
    if feature == "optimizer":
        # Optimizer hooks are installed lazily by trace_step in auto mode.
        return config.mode == "auto"
    return bool(getattr(config, _PATCH_FLAG_BY_FEATURE[feature]))


def require_manual_wrapper_allowed(
    feature: InstrumentationFeature,
    wrapper_name: str,
    *,
    covered_by_automatic_instrumentation: bool = True,
) -> TraceMLInitConfig:
    """Validate initialization and single ownership for a public wrapper.

    ``covered_by_automatic_instrumentation`` is false only for input sources
    such as Ray Data iterators that the PyTorch DataLoader patch cannot see.
    Such sources may add fetch timing in auto mode without double-counting.
    """
    config = initial.get_init_config()
    if config is None:
        raise RuntimeError(
            f"{wrapper_name}() requires TraceML initialization. Call "
            "traceml.init(...) before creating manual wrappers."
        )

    if covered_by_automatic_instrumentation and _automatically_owned(
        config, feature
    ):
        raise RuntimeError(
            f"{wrapper_name}() cannot instrument {feature!r} because "
            f"TraceML mode={config.mode!r} already owns that measurement. "
            "Use the automatic measurement without this wrapper, or choose "
            "mode='manual' / mode='selective' with that automatic patch "
            "disabled."
        )

    return config


__all__ = ["InstrumentationFeature", "require_manual_wrapper_allowed"]
