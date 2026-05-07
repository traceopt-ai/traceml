"""TraceML initialization and patch policy."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal, Optional

TraceMLInitMode = Literal["auto", "manual", "selective"]


@dataclass(frozen=True)
class TraceMLInitConfig:
    """Effective initialization config returned by `traceml.init()`."""

    mode: TraceMLInitMode
    patch_dataloader: bool
    patch_forward: bool
    patch_backward: bool
    source: str = "user"

    def same_effective_configuration(self, other: "TraceMLInitConfig") -> bool:
        """
        Return True when two init configs result in the same runtime behavior.
        """
        return (
            self.mode == other.mode
            and self.patch_dataloader == other.patch_dataloader
            and self.patch_forward == other.patch_forward
            and self.patch_backward == other.patch_backward
        )


_INIT_LOCK = Lock()
_INIT_CONFIG: Optional[TraceMLInitConfig] = None


def _canonical_mode(mode: str) -> TraceMLInitMode:
    """Normalize a user-provided mode string."""
    text = str(mode or "").strip().lower()
    if text == "custom":
        return "selective"
    if text in {"auto", "manual", "selective"}:
        return text  # type: ignore[return-value]

    raise ValueError(
        "Invalid TraceML init mode "
        f"{mode!r}. Expected one of: 'auto', 'manual', 'selective'. "
        "The alias 'custom' is also accepted and maps to 'selective'."
    )


def _build_config(
    *,
    mode: str,
    patch_dataloader: Optional[bool],
    patch_forward: Optional[bool],
    patch_backward: Optional[bool],
    source: str,
) -> TraceMLInitConfig:
    """Validate user input and return the initialization config."""
    canonical_mode = _canonical_mode(mode)
    override_values = (
        patch_dataloader,
        patch_forward,
        patch_backward,
    )
    has_overrides = any(value is not None for value in override_values)

    if canonical_mode in {"auto", "manual"} and has_overrides:
        raise ValueError(
            "patch_dataloader, patch_forward, and patch_backward may only be "
            "provided when mode='selective'. "
            f"Received overrides with mode={canonical_mode!r}."
        )

    if canonical_mode == "auto":
        return TraceMLInitConfig(
            mode="auto",
            patch_dataloader=True,
            patch_forward=True,
            patch_backward=True,
            source=source,
        )

    if canonical_mode == "manual":
        return TraceMLInitConfig(
            mode="manual",
            patch_dataloader=False,
            patch_forward=False,
            patch_backward=False,
            source=source,
        )

    if not has_overrides:
        raise ValueError(
            "mode='selective' requires at least one explicit patch_* override. "
            "Use mode='manual' for no automatic patches."
        )

    dl = bool(patch_dataloader) if patch_dataloader is not None else False
    fwd = bool(patch_forward) if patch_forward is not None else False
    bwd = bool(patch_backward) if patch_backward is not None else False

    if not any((dl, fwd, bwd)):
        raise ValueError(
            "mode='selective' must enable at least one automatic patch. "
            "Use mode='manual' when you want zero automatic patches."
        )

    return TraceMLInitConfig(
        mode="selective",
        patch_dataloader=dl,
        patch_forward=fwd,
        patch_backward=bwd,
        source=source,
    )


def _apply_requested_patches(config: TraceMLInitConfig) -> None:
    """Apply the patch set requested by the validated config."""
    if not any(
        (
            config.patch_dataloader,
            config.patch_forward,
            config.patch_backward,
        )
    ):
        return

    try:
        if config.patch_dataloader:
            from traceml.instrumentation.patches.dataloader_patch import (
                patch_dataloader,
            )

            patch_dataloader()

        if config.patch_forward:
            from traceml.instrumentation.patches.forward_auto_timer_patch import (
                patch_forward,
            )

            patch_forward()

        if config.patch_backward:
            from traceml.instrumentation.patches.backward_auto_timer_patch import (
                patch_backward,
            )

            patch_backward()
    except Exception as exc:
        raise RuntimeError(
            "TraceML initialization failed while installing automatic "
            f"instrumentation patches for mode={config.mode!r}. "
            "This error is fatal because partial patch installation can lead "
            "to inconsistent tracing behavior. "
            f"Original error: {exc}"
        ) from exc


def get_init_config() -> Optional[TraceMLInitConfig]:
    """
    Return the active TraceML init config, or None when init has not run.
    """
    return _INIT_CONFIG


def is_initialized() -> bool:
    """
    Return True when TraceML init has completed successfully.
    """
    return _INIT_CONFIG is not None


def init(
    *,
    mode: str = "auto",
    patch_dataloader: Optional[bool] = None,
    patch_forward: Optional[bool] = None,
    patch_backward: Optional[bool] = None,
    _source: str = "user",
) -> TraceMLInitConfig:
    """
    Initialize TraceML instrumentation policy for the current Python process.

    Parameters
    ----------
    mode:
        Instrumentation mode. Supported values:
        - 'auto'
        - 'manual'
        - 'selective'
        The alias 'custom' is also accepted and maps to 'selective'.
    patch_dataloader:
        Selective-mode-only override controlling DataLoader fetch patching.
    patch_forward:
        Selective-mode-only override controlling forward timing patching.
    patch_backward:
        Selective-mode-only override controlling backward timing patching.

    Returns
    -------
    TraceMLInitConfig
        The effective initialization config.

    Raises
    ------
    ValueError
        If the init request is invalid.
    RuntimeError
        If init conflicts with an existing config or patch installation fails.

    Notes
    -----
    - Repeating the same effective init configuration is safe and returns the
      existing config.
    - Re-initialization with a different effective configuration is rejected.
      This keeps instrumentation behavior deterministic and easy to reason
      about in production environments.
    """
    requested = _build_config(
        mode=mode,
        patch_dataloader=patch_dataloader,
        patch_forward=patch_forward,
        patch_backward=patch_backward,
        source=_source,
    )

    global _INIT_CONFIG

    with _INIT_LOCK:
        if _INIT_CONFIG is not None:
            if _INIT_CONFIG.same_effective_configuration(requested):
                return _INIT_CONFIG

            raise RuntimeError(
                "TraceML has already been initialized with a different "
                "configuration in this process. "
                f"Existing config: mode={_INIT_CONFIG.mode!r}, "
                f"patch_dataloader={_INIT_CONFIG.patch_dataloader}, "
                f"patch_forward={_INIT_CONFIG.patch_forward}, "
                f"patch_backward={_INIT_CONFIG.patch_backward}, "
                f"source={_INIT_CONFIG.source!r}. "
                f"Requested config: mode={requested.mode!r}, "
                f"patch_dataloader={requested.patch_dataloader}, "
                f"patch_forward={requested.patch_forward}, "
                f"patch_backward={requested.patch_backward}, "
                f"source={requested.source!r}. "
                "Initialize TraceML exactly once per process with the intended "
                "mode at the start of the run."
            )

        _apply_requested_patches(requested)
        _INIT_CONFIG = requested
        return requested


def start(
    *,
    mode: str = "auto",
    patch_dataloader: Optional[bool] = None,
    patch_forward: Optional[bool] = None,
    patch_backward: Optional[bool] = None,
) -> TraceMLInitConfig:
    """
    Alias for `init()` for the current transition period.

    This keeps the already-exposed public surface usable while the TraceML
    is being formalized. Today, `start()` and `init()` are equivalent.
    """
    return init(
        mode=mode,
        patch_dataloader=patch_dataloader,
        patch_forward=patch_forward,
        patch_backward=patch_backward,
        _source="user",
    )


def enable_legacy_decorator_auto_init() -> Optional[TraceMLInitConfig]:
    """
    Preserve historical `traceml.decorators` import behavior.

    Legacy decorator imports previously triggered automatic patching as a module
    import side effect. This function keeps that behavior for the compatibility
    path without affecting the new explicit path.

    If explicit init has already happened, this function does nothing and
    respects the existing config.
    """
    if is_initialized():
        return get_init_config()

    return init(mode="auto", _source="traceml.decorators")


__all__ = [
    "TraceMLInitConfig",
    "TraceMLInitMode",
    "get_init_config",
    "is_initialized",
    "init",
    "start",
    "enable_legacy_decorator_auto_init",
]
