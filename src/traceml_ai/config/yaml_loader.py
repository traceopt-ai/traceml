"""
traceml.yaml config loader.

Precedence: CLI flag > TRACEML_* env var > traceml.yaml > built-in default.

Only imported by the launcher. Child processes (executor, aggregator) receive
already-resolved TRACEML_* env vars and never read this file.

Scope: this loader governs the UI/telemetry settings that have no dedicated
launch-config owner. Distributed and run-identity settings (nproc/nnodes,
node rank, master + aggregator addresses, run name/session id, summary window,
trace step cap) are configured through CLI flags and the typed launch configs
in ``traceml_ai.launcher.launch_config``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Mapping

_log = logging.getLogger(__name__)

CONFIG_FILENAME = "traceml.yaml"

# Max parent dirs to search upward for traceml.yaml.
_MAX_WALK_LEVELS = 10

# Schema: yaml key → (TRACEML_* env var, expected type).
# To add a setting: add here, add to BUILT_IN_DEFAULTS, add default=None in cli.py.
YAML_KEY_SCHEMA: dict[str, tuple[str, type]] = {
    "mode": ("TRACEML_UI_MODE", str),
    "interval": ("TRACEML_INTERVAL", float),
    "enable_logging": ("TRACEML_ENABLE_LOGGING", bool),
    "logs_dir": ("TRACEML_LOGS_DIR", str),
    "history_enabled": ("TRACEML_HISTORY_ENABLED", bool),
}

# Fallback values used when no CLI flag, env var, or yaml entry is present.
BUILT_IN_DEFAULTS: dict[str, Any] = {
    "mode": "summary",
    "interval": 2.0,
    "enable_logging": False,
    "logs_dir": "./logs",
    "history_enabled": True,
}

# Env var strings treated as True for bool fields.
_BOOL_ENV_TRUE = frozenset({"1", "true", "yes"})


def find_config_file(start_dir: Path) -> Path | None:
    """Walk up from *start_dir* looking for traceml.yaml. Returns None if not found."""
    current = start_dir.resolve()
    for _ in range(_MAX_WALK_LEVELS):
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            _log.debug("[TraceML] Found config file: %s", candidate)
            return candidate
        parent = current.parent
        if parent == current:
            break  # filesystem root
        current = parent
    return None


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Parse traceml.yaml. Unknown keys warn-and-skip; type errors raise ValueError."""
    try:
        import yaml  # noqa: PLC0415 — intentional late import
    except ImportError:
        warnings.warn(
            "[TraceML] pyyaml is not installed; traceml.yaml will be ignored. "
            "Install it with: pip install pyyaml",
            stacklevel=2,
        )
        return {}

    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except OSError as exc:
        raise OSError(
            f"[TraceML] Cannot read config file {path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(
            f"[TraceML] {path}: YAML syntax error — {exc}"
        ) from exc

    if raw is None:
        return {}  # empty file

    if not isinstance(raw, dict):
        raise ValueError(
            f"[TraceML] {path}: expected a YAML mapping at the top level, "
            f"got {type(raw).__name__}"
        )

    result: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in YAML_KEY_SCHEMA:
            warnings.warn(
                f"[TraceML] {path}: unknown config key '{key}' — ignored.",
                stacklevel=2,
            )
            continue
        _, expected_type = YAML_KEY_SCHEMA[key]
        result[key] = _validate_and_coerce(key, value, expected_type, path)

    return result


def _validate_and_coerce(
    key: str, value: Any, expected_type: type, path: Path
) -> Any:
    """Coerce value to the expected type or raise ValueError."""
    if expected_type is bool:
        if isinstance(value, bool):
            return value
        # Allow 0/1 as bool.
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise ValueError(
            f"[TraceML] {path}: '{key}' must be a boolean (true/false), "
            f"got {value!r}"
        )

    if expected_type is int:
        # bool is a subclass of int; reject it explicitly.
        if isinstance(value, bool):
            raise ValueError(
                f"[TraceML] {path}: '{key}' must be an integer, got {value!r}"
            )
        if isinstance(value, int):
            return value
        raise ValueError(
            f"[TraceML] {path}: '{key}' must be an integer, got {value!r}"
        )

    if expected_type is float:
        if isinstance(value, bool):
            raise ValueError(
                f"[TraceML] {path}: '{key}' must be a number, got {value!r}"
            )
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(
            f"[TraceML] {path}: '{key}' must be a number, got {value!r}"
        )

    if expected_type is str:
        if isinstance(value, str):
            return value
        raise ValueError(
            f"[TraceML] {path}: '{key}' must be a string, got {value!r}"
        )

    return value


def _coerce_env(key: str, raw_env: str) -> Any:
    """Convert a TRACEML_* env var string to the correct Python type."""
    env_var, expected_type = YAML_KEY_SCHEMA[key]
    if expected_type is bool:
        return raw_env.strip().lower() in _BOOL_ENV_TRUE
    if expected_type is int:
        try:
            return int(raw_env)
        except ValueError:
            raise ValueError(
                f"[TraceML] env var {env_var}={raw_env!r} is not a valid integer."
            ) from None
    if expected_type is float:
        try:
            return float(raw_env)
        except ValueError:
            raise ValueError(
                f"[TraceML] env var {env_var}={raw_env!r} is not a valid number."
            ) from None
    return raw_env  # str


def resolve_config(
    cli_overrides: dict[str, Any],
    parent_env: Mapping[str, str],
    yaml_config: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Return a fully resolved config using CLI > env > yaml > default precedence.

    cli_overrides values of None mean "not set by the user"; any other value wins.
    """
    result: dict[str, Any] = {}
    for key, (env_var, _) in YAML_KEY_SCHEMA.items():
        cli_val = cli_overrides.get(key)
        if cli_val is not None:
            result[key] = cli_val
        elif env_var in parent_env:
            result[key] = _coerce_env(key, parent_env[env_var])
        elif key in yaml_config:
            result[key] = yaml_config[key]
        else:
            result[key] = defaults[key]
    return result
