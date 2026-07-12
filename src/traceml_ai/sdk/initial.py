"""TraceML initialization and patch policy."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, Optional

TraceMLInitMode = Literal["auto", "manual", "selective"]


@dataclass(frozen=True)
class TraceMLInitConfig:
    """Effective initialization config returned by `traceml.init()`."""

    mode: TraceMLInitMode
    patch_dataloader: bool
    patch_forward: bool
    patch_backward: bool
    patch_h2d: bool
    source: str = "user"
    disabled: bool = False

    def same_effective_configuration(self, other: "TraceMLInitConfig") -> bool:
        """
        Return True when two init configs result in the same runtime behavior.
        """
        return (
            self.disabled == other.disabled
            and self.mode == other.mode
            and self.patch_dataloader == other.patch_dataloader
            and self.patch_forward == other.patch_forward
            and self.patch_backward == other.patch_backward
            and self.patch_h2d == other.patch_h2d
        )


_INIT_LOCK = Lock()
_INIT_CONFIG: Optional[TraceMLInitConfig] = None

# RuntimeHandle started by init() in this process, and whether the atexit
# shutdown hook has been registered. Tracked separately from _INIT_CONFIG so
# the runtime is stopped exactly once, only when init() actually started it.
_RUNTIME_HANDLE: Any = None
_ATEXIT_REGISTERED: bool = False


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
    patch_h2d: Optional[bool],
    source: str,
) -> TraceMLInitConfig:
    """Validate user input and return the initialization config."""
    canonical_mode = _canonical_mode(mode)
    override_values = (
        patch_dataloader,
        patch_forward,
        patch_backward,
        patch_h2d,
    )
    has_overrides = any(value is not None for value in override_values)

    if canonical_mode in {"auto", "manual"} and has_overrides:
        raise ValueError(
            "patch_dataloader, patch_forward, patch_backward, and patch_h2d "
            "may only be provided when mode='selective'. "
            f"Received overrides with mode={canonical_mode!r}."
        )

    if canonical_mode == "auto":
        return TraceMLInitConfig(
            mode="auto",
            patch_dataloader=True,
            patch_forward=True,
            patch_backward=True,
            patch_h2d=True,
            source=source,
        )

    if canonical_mode == "manual":
        return TraceMLInitConfig(
            mode="manual",
            patch_dataloader=False,
            patch_forward=False,
            patch_backward=False,
            patch_h2d=False,
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
    h2d = bool(patch_h2d) if patch_h2d is not None else False

    if not any((dl, fwd, bwd, h2d)):
        raise ValueError(
            "mode='selective' must enable at least one automatic patch. "
            "Use mode='manual' when you want zero automatic patches."
        )

    return TraceMLInitConfig(
        mode="selective",
        patch_dataloader=dl,
        patch_forward=fwd,
        patch_backward=bwd,
        patch_h2d=h2d,
        source=source,
    )


def _apply_requested_patches(config: TraceMLInitConfig) -> None:
    """Apply the patch set requested by the validated config."""
    if not any(
        (
            config.patch_dataloader,
            config.patch_forward,
            config.patch_backward,
            config.patch_h2d,
        )
    ):
        return

    try:
        if config.patch_dataloader:
            from traceml_ai.instrumentation.patches.dataloader_patch import (
                patch_dataloader,
            )

            patch_dataloader()

        if config.patch_forward:
            from traceml_ai.instrumentation.patches.forward_auto_timer_patch import (
                patch_forward,
            )

            patch_forward()

        if config.patch_backward:
            from traceml_ai.instrumentation.patches.backward_auto_timer_patch import (
                patch_backward,
            )

            patch_backward()

        if config.patch_h2d:
            from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (
                patch_h2d,
            )

            patch_h2d()
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


def _conflict_message(
    existing: TraceMLInitConfig, requested: TraceMLInitConfig
) -> str:
    """Build the error message for an incompatible re-initialization."""
    return (
        "TraceML has already been initialized with a different "
        "configuration in this process. "
        f"Existing config: mode={existing.mode!r}, "
        f"disabled={existing.disabled}, "
        f"patch_dataloader={existing.patch_dataloader}, "
        f"patch_forward={existing.patch_forward}, "
        f"patch_backward={existing.patch_backward}, "
        f"patch_h2d={existing.patch_h2d}, "
        f"source={existing.source!r}. "
        f"Requested config: mode={requested.mode!r}, "
        f"disabled={requested.disabled}, "
        f"patch_dataloader={requested.patch_dataloader}, "
        f"patch_forward={requested.patch_forward}, "
        f"patch_backward={requested.patch_backward}, "
        f"patch_h2d={requested.patch_h2d}, "
        f"source={requested.source!r}. "
        "Initialize TraceML exactly once per process with the intended "
        "mode at the start of the run."
    )


def _env_str(name: str, default: str) -> str:
    """Return a non-empty environment value, otherwise ``default``."""
    import os

    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _resolve_runtime_settings(
    *,
    ui_mode: Optional[str],
    interval: Optional[float],
    logs_dir: Optional[str],
    enable_logging: Optional[bool],
    session_id: Optional[str],
    aggregator_host: Optional[str],
    aggregator_port: Optional[int],
) -> Any:
    """
    Build TraceMLSettings for a user-code runtime (direct ``python``/``torchrun``).

    Runtime/telemetry settings (mode, interval, logs_dir, enable_logging, ...)
    route through the shared config resolver with the same precedence the CLI
    launcher uses: explicit ``traceml.init(...)`` arg > ``TRACEML_*`` env var >
    ``traceml.yaml`` > built-in default. Aggregator host/port come from explicit
    init args, then env, then defaults. Run identity and the aggregator endpoint
    are launch-owned and are not read from ``traceml.yaml``.
    """
    import os
    from pathlib import Path

    from traceml_ai.config.yaml_loader import (
        BUILT_IN_DEFAULTS,
        find_config_file,
        load_yaml_config,
        resolve_config,
    )
    from traceml_ai.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS
    from traceml_ai.runtime.session import get_session_id
    from traceml_ai.runtime.settings import (
        AggregatorTransportSettings,
        TraceMLSettings,
    )

    try:
        config_path = find_config_file(Path.cwd())
        yaml_cfg = (
            load_yaml_config(config_path) if config_path is not None else {}
        )
    except (ValueError, OSError):
        # A broken traceml.yaml must not crash init(); fall back to env/defaults.
        yaml_cfg = {}

    cli_overrides = {
        "mode": ui_mode,
        "interval": interval,
        "enable_logging": enable_logging,
        "logs_dir": logs_dir,
    }
    cfg = resolve_config(
        cli_overrides=cli_overrides,
        parent_env=os.environ,
        yaml_config=yaml_cfg,
        defaults=BUILT_IN_DEFAULTS,
    )

    resolved_session = str(
        session_id or _env_str("TRACEML_SESSION_ID", "") or get_session_id()
    )
    host = str(
        aggregator_host
        if aggregator_host is not None
        else _env_str("TRACEML_AGGREGATOR_HOST", "127.0.0.1")
    )
    port = int(
        aggregator_port
        if aggregator_port is not None
        else int(_env_str("TRACEML_AGGREGATOR_PORT", "29765"))
    )
    raw_max_steps = os.environ.get("TRACEML_TRACE_MAX_STEPS", "")
    trace_max_steps = int(raw_max_steps) if raw_max_steps.strip() else None

    return TraceMLSettings(
        mode=str(cfg["mode"]),
        profile=_env_str("TRACEML_PROFILE", "run"),
        sampler_interval_sec=float(cfg["interval"]),
        enable_logging=bool(cfg["enable_logging"]),
        logs_dir=str(cfg["logs_dir"]),
        num_display_layers=int(cfg["num_display_layers"]),
        remote_max_rows=int(cfg["remote_max_rows"]),
        history_enabled=bool(cfg["history_enabled"]),
        session_id=resolved_session,
        summary_window_rows=int(
            _env_str(
                "TRACEML_SUMMARY_WINDOW_ROWS",
                str(DEFAULT_SUMMARY_WINDOW_ROWS),
            )
        ),
        trace_max_steps=trace_max_steps,
        aggregator=AggregatorTransportSettings(
            connect_host=host, bind_host=host, port=port
        ),
    )


def _start_runtime_for_init(
    *,
    ui_mode: Optional[str],
    interval: Optional[float],
    logs_dir: Optional[str],
    enable_logging: Optional[bool],
    session_id: Optional[str],
    aggregator_host: Optional[str],
    aggregator_port: Optional[int],
    connect_timeout_sec: float,
    connect_retry_interval_sec: float,
) -> None:
    """
    Start the per-process TraceML runtime from user code.

    No-op when a runtime is already active in this process (for example one
    started by the ``traceml run`` executor), which keeps the CLI path working
    even if user code also calls ``traceml.init()``. Otherwise this verifies the
    aggregator is reachable within a bounded window and raises a clear error if
    it is not.
    """
    global _RUNTIME_HANDLE, _ATEXIT_REGISTERED
    import atexit

    from traceml_ai.runtime import lifecycle

    if lifecycle.get_active_runtime_handle() is not None:
        return  # executor/ray already started the runtime in this process

    settings = _resolve_runtime_settings(
        ui_mode=ui_mode,
        interval=interval,
        logs_dir=logs_dir,
        enable_logging=enable_logging,
        session_id=session_id,
        aggregator_host=aggregator_host,
        aggregator_port=aggregator_port,
    )
    host = settings.aggregator.connect_host
    port = settings.aggregator.port

    if not lifecycle.wait_for_aggregator(
        host,
        port,
        timeout_sec=connect_timeout_sec,
        poll_interval_sec=connect_retry_interval_sec,
    ):
        raise RuntimeError(
            f"TraceML could not reach the aggregator at {host}:{port} "
            f"after {connect_timeout_sec:.0f}s. Start it with "
            "`traceml serve` (matching --aggregator-host/--aggregator-port), "
            "or call traceml.init(disabled=True) to run without tracing."
        )

    _RUNTIME_HANDLE = lifecycle.start_runtime(settings, fail_open=False)

    if not _ATEXIT_REGISTERED:
        atexit.register(_stop_runtime_for_init)
        _ATEXIT_REGISTERED = True


def _stop_runtime_for_init() -> None:
    """Best-effort runtime shutdown for the user-code path (atexit)."""
    global _RUNTIME_HANDLE
    handle = _RUNTIME_HANDLE
    _RUNTIME_HANDLE = None
    if handle is not None:
        try:
            handle.stop()
        except Exception:
            pass


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
    _source: str = "user",
) -> TraceMLInitConfig:
    """
    Initialize TraceML for the current Python process and start its runtime.

    This installs the requested instrumentation patches, starts the TraceML
    runtime (background samplers + telemetry) in-process, and verifies that the
    aggregator is reachable over TCP. When TraceML is already running in this
    process (for example under ``traceml run``), runtime startup is skipped and
    only the instrumentation policy is applied.

    Configuration precedence for the runtime settings below is: explicit
    argument here > ``TRACEML_*`` env var > ``traceml.yaml`` > built-in default,
    the same resolver the CLI launcher uses.

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
    disabled:
        When True, TraceML is a complete no-op: no patches are installed and no
        runtime is started. Defaults to the ``TRACEML_DISABLED`` environment
        variable when not set explicitly.
    ui_mode:
        Display/telemetry mode for this run ('cli', 'dashboard', or 'summary').
        Resolved via the shared config resolver.
    interval:
        Sampler interval in seconds. Resolved via the shared config resolver.
    logs_dir:
        Directory for TraceML session logs. Resolved via the shared config
        resolver.
    enable_logging:
        Enable TraceML logging output. Resolved via the shared config resolver.
    session_id:
        Explicit TraceML run/session id. Defaults to ``TRACEML_SESSION_ID`` or a
        generated id. Must match the aggregator's session for shared artifacts.
    aggregator_host:
        Host the runtime connects to for telemetry. Defaults to
        ``TRACEML_AGGREGATOR_HOST`` or ``127.0.0.1``.
    aggregator_port:
        Port the runtime connects to for telemetry. Defaults to
        ``TRACEML_AGGREGATOR_PORT`` or ``29765``.
    connect_timeout_sec:
        Bounded period to wait for the aggregator before failing.
    connect_retry_interval_sec:
        Delay between aggregator connection attempts.

    Returns
    -------
    TraceMLInitConfig
        The effective initialization config.

    Raises
    ------
    ValueError
        If the init request is invalid.
    RuntimeError
        If init conflicts with an existing config, patch installation fails, or
        the aggregator is unreachable within ``connect_timeout_sec``.

    Notes
    -----
    - Repeating the same effective init configuration is safe and returns the
      existing config.
    - Re-initialization with a different effective configuration is rejected.
      This keeps instrumentation behavior deterministic and easy to reason
      about in production environments.
    """
    import os

    global _INIT_CONFIG

    is_disabled = (
        bool(disabled)
        if disabled is not None
        else os.environ.get("TRACEML_DISABLED", "0") == "1"
    )

    if is_disabled:
        os.environ["TRACEML_DISABLED"] = "1"
        disabled_config = TraceMLInitConfig(
            mode="manual",
            patch_dataloader=False,
            patch_forward=False,
            patch_backward=False,
            patch_h2d=False,
            source=_source,
            disabled=True,
        )
        with _INIT_LOCK:
            if _INIT_CONFIG is not None:
                if _INIT_CONFIG.same_effective_configuration(disabled_config):
                    return _INIT_CONFIG
                raise RuntimeError(
                    _conflict_message(_INIT_CONFIG, disabled_config)
                )
            _INIT_CONFIG = disabled_config
            return disabled_config

    requested = _build_config(
        mode=mode,
        patch_dataloader=patch_dataloader,
        patch_forward=patch_forward,
        patch_backward=patch_backward,
        patch_h2d=patch_h2d,
        source=_source,
    )

    with _INIT_LOCK:
        if _INIT_CONFIG is not None:
            if _INIT_CONFIG.same_effective_configuration(requested):
                return _INIT_CONFIG

            raise RuntimeError(_conflict_message(_INIT_CONFIG, requested))

        # Start the runtime first (this also runs the bounded aggregator
        # preflight). Doing so before installing patches keeps torch untouched
        # when the aggregator is missing, and matches the runtime-then-patches
        # order used by the in-process framework integrations.
        _start_runtime_for_init(
            ui_mode=ui_mode,
            interval=interval,
            logs_dir=logs_dir,
            enable_logging=enable_logging,
            session_id=session_id,
            aggregator_host=aggregator_host,
            aggregator_port=aggregator_port,
            connect_timeout_sec=connect_timeout_sec,
            connect_retry_interval_sec=connect_retry_interval_sec,
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
        _source="user",
    )


__all__ = [
    "TraceMLInitConfig",
    "TraceMLInitMode",
    "is_initialized",
    "get_init_config",
    "init",
    "start",
]
