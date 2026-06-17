"""Tests for traceml_ai.config.yaml_loader."""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import pytest

from traceml_ai.config.yaml_loader import (
    BUILT_IN_DEFAULTS,
    YAML_KEY_SCHEMA,
    find_config_file,
    load_yaml_config,
    resolve_config,
)

# find_config_file


def test_find_config_file_not_present(tmp_path: Path) -> None:
    assert find_config_file(tmp_path) is None


def test_find_config_file_in_start_dir(tmp_path: Path) -> None:
    cfg = tmp_path / "traceml.yaml"
    cfg.write_text("mode: cli\n", encoding="utf-8")
    assert find_config_file(tmp_path) == cfg


def test_find_config_file_walks_up(tmp_path: Path) -> None:
    cfg = tmp_path / "traceml.yaml"
    cfg.write_text("mode: cli\n", encoding="utf-8")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert find_config_file(deep) == cfg


def test_find_config_file_stops_at_nearest_ancestor(tmp_path: Path) -> None:
    """Returns the closest ancestor, not a deeper one."""
    root_cfg = tmp_path / "traceml.yaml"
    root_cfg.write_text("mode: cli\n", encoding="utf-8")
    mid = tmp_path / "sub"
    mid.mkdir()
    mid_cfg = mid / "traceml.yaml"
    mid_cfg.write_text("mode: summary\n", encoding="utf-8")
    deep = mid / "proj"
    deep.mkdir()
    # Starting from deep → should find mid/traceml.yaml first
    assert find_config_file(deep) == mid_cfg


# load_yaml_config


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "traceml.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_load_yaml_config_empty_file(tmp_path: Path) -> None:
    p = _write(tmp_path, "")
    assert load_yaml_config(p) == {}


def test_load_yaml_config_valid(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """\
        mode: summary
        interval: 3.0
        logs_dir: ./runs
        num_display_layers: 8
        history_enabled: true
        enable_logging: false
        remote_max_rows: 100
        """,
    )
    result = load_yaml_config(p)
    assert result["mode"] == "summary"
    assert result["interval"] == 3.0
    assert result["logs_dir"] == "./runs"
    assert result["num_display_layers"] == 8
    assert result["history_enabled"] is True
    assert result["enable_logging"] is False
    assert result["remote_max_rows"] == 100


def test_load_yaml_config_unknown_key_warns(tmp_path: Path) -> None:
    p = _write(tmp_path, "mode: cli\nunknown_key: 42\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = load_yaml_config(p)
    assert "unknown_key" not in result
    assert any("unknown config key" in str(warning.message) for warning in w)


def test_load_yaml_config_type_error_bool_field(tmp_path: Path) -> None:
    p = _write(tmp_path, "enable_logging: not_a_bool\n")
    with pytest.raises(ValueError, match="enable_logging"):
        load_yaml_config(p)


def test_load_yaml_config_type_error_int_field(tmp_path: Path) -> None:
    p = _write(tmp_path, "num_display_layers: abc\n")
    with pytest.raises(ValueError, match="num_display_layers"):
        load_yaml_config(p)


def test_load_yaml_config_type_error_float_field(tmp_path: Path) -> None:
    p = _write(tmp_path, "interval: nope\n")
    with pytest.raises(ValueError, match="interval"):
        load_yaml_config(p)


def test_load_yaml_config_type_error_str_field(tmp_path: Path) -> None:
    p = _write(tmp_path, "mode: 123\n")
    with pytest.raises(ValueError, match="mode"):
        load_yaml_config(p)


def test_load_yaml_config_bool_false_flag(tmp_path: Path) -> None:
    p = _write(tmp_path, "history_enabled: false\n")
    result = load_yaml_config(p)
    assert result["history_enabled"] is False


def test_load_yaml_config_int_as_bool_zero(tmp_path: Path) -> None:
    """YAML sometimes emits 0/1 for bool fields — allow it."""
    p = _write(tmp_path, "enable_logging: 1\n")
    result = load_yaml_config(p)
    assert result["enable_logging"] is True


def test_load_yaml_config_int_as_float(tmp_path: Path) -> None:
    """Integer literal for a float field should be accepted."""
    p = _write(tmp_path, "interval: 5\n")
    result = load_yaml_config(p)
    assert result["interval"] == 5.0
    assert isinstance(result["interval"], float)


def test_load_yaml_config_top_level_not_mapping(tmp_path: Path) -> None:
    p = _write(tmp_path, "- item1\n- item2\n")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        load_yaml_config(p)


def test_load_yaml_config_malformed_yaml(tmp_path: Path) -> None:
    p = _write(tmp_path, "mode: [unclosed\n")
    with pytest.raises(ValueError, match="YAML syntax error"):
        load_yaml_config(p)


# resolve_config  — precedence rules


def _defaults() -> dict:
    return dict(BUILT_IN_DEFAULTS)


def _no_env() -> dict:
    return {}


def _no_yaml() -> dict:
    return {}


def _no_cli() -> dict:
    return {k: None for k in YAML_KEY_SCHEMA}


def test_resolve_config_cli_beats_env_and_yaml(tmp_path: Path) -> None:
    cli = {**_no_cli(), "mode": "dashboard"}
    env = {"TRACEML_UI_MODE": "summary"}
    yaml = {"mode": "cli"}
    result = resolve_config(cli, env, yaml, _defaults())
    assert result["mode"] == "dashboard"


def test_resolve_config_env_beats_yaml(tmp_path: Path) -> None:
    cli = _no_cli()
    env = {"TRACEML_UI_MODE": "summary"}
    yaml = {"mode": "cli"}
    result = resolve_config(cli, env, yaml, _defaults())
    assert result["mode"] == "summary"


def test_resolve_config_yaml_beats_default(tmp_path: Path) -> None:
    cli = _no_cli()
    result = resolve_config(cli, _no_env(), {"mode": "summary"}, _defaults())
    assert result["mode"] == "summary"


def test_resolve_config_default_when_nothing_set() -> None:
    cli = _no_cli()
    result = resolve_config(cli, _no_env(), _no_yaml(), _defaults())
    assert result["mode"] == BUILT_IN_DEFAULTS["mode"]
    assert result["interval"] == BUILT_IN_DEFAULTS["interval"]
    assert result["remote_max_rows"] == BUILT_IN_DEFAULTS["remote_max_rows"]
    assert result["history_enabled"] == BUILT_IN_DEFAULTS["history_enabled"]


def test_resolve_config_none_cli_is_not_an_override() -> None:
    """None must not shadow env/yaml/default values."""
    cli = {**_no_cli(), "mode": None}
    yaml = {"mode": "summary"}
    result = resolve_config(cli, _no_env(), yaml, _defaults())
    assert result["mode"] == "summary"


def test_resolve_config_env_bool_coercion() -> None:
    cli = _no_cli()
    env = {"TRACEML_ENABLE_LOGGING": "1", "TRACEML_HISTORY_ENABLED": "0"}
    result = resolve_config(cli, env, _no_yaml(), _defaults())
    assert result["enable_logging"] is True
    assert result["history_enabled"] is False


def test_resolve_config_env_int_coercion() -> None:
    cli = _no_cli()
    env = {
        "TRACEML_REMOTE_MAX_ROWS": "12345",
        "TRACEML_NUM_DISPLAY_LAYERS": "15",
    }
    result = resolve_config(cli, env, _no_yaml(), _defaults())
    assert result["remote_max_rows"] == 12345
    assert result["num_display_layers"] == 15


def test_resolve_config_env_float_coercion() -> None:
    cli = _no_cli()
    env = {"TRACEML_INTERVAL": "0.5"}
    result = resolve_config(cli, env, _no_yaml(), _defaults())
    assert result["interval"] == 0.5


def test_resolve_config_history_disabled_via_cli() -> None:
    """--no-history maps to history_enabled=False as a CLI override."""
    # Simulate what launch_process does: args.no_history=True → False override
    cli = {**_no_cli(), "history_enabled": False}
    yaml = {"history_enabled": True}
    env = {"TRACEML_HISTORY_ENABLED": "1"}
    result = resolve_config(cli, env, yaml, _defaults())
    assert result["history_enabled"] is False


def test_load_yaml_config_unreadable_file(tmp_path: Path) -> None:
    """A file that exists but cannot be read raises OSError with a clear message."""
    import sys

    if sys.platform == "win32":
        pytest.skip("chmod is not enforced the same way on Windows")

    p = _write(tmp_path, "mode: cli\n")
    p.chmod(0o000)
    try:
        with pytest.raises(OSError, match="Cannot read config file"):
            load_yaml_config(p)
    finally:
        p.chmod(0o644)


def test_coerce_env_malformed_int_raises_value_error() -> None:
    """A malformed TRACEML_* int env var gives a clear error, not a raw Python one."""
    from traceml_ai.config.yaml_loader import _coerce_env

    with pytest.raises(ValueError, match="TRACEML_REMOTE_MAX_ROWS"):
        _coerce_env("remote_max_rows", "not_a_number")


def test_coerce_env_malformed_float_raises_value_error() -> None:
    from traceml_ai.config.yaml_loader import _coerce_env

    with pytest.raises(ValueError, match="TRACEML_INTERVAL"):
        _coerce_env("interval", "bad")


def test_resolve_config_all_keys_present_in_result() -> None:
    """resolve_config must always return a value for every schema key."""
    result = resolve_config(_no_cli(), _no_env(), _no_yaml(), _defaults())
    assert set(result.keys()) == set(YAML_KEY_SCHEMA.keys())


def test_resolve_config_deprecated_traceml_mode_env_var() -> None:
    """TRACEML_MODE (deprecated) must be normalised to TRACEML_UI_MODE before
    resolve_config is called. This test verifies the normalisation logic used
    in launch_process produces the right resolved value."""
    cli = _no_cli()
    # Simulate the normalisation done by launch_process:
    # TRACEML_UI_MODE absent, TRACEML_MODE present → copy to TRACEML_UI_MODE.
    raw_env = {"TRACEML_MODE": "summary"}
    if "TRACEML_UI_MODE" not in raw_env and "TRACEML_MODE" in raw_env:
        normalised_env = {
            **raw_env,
            "TRACEML_UI_MODE": raw_env["TRACEML_MODE"],
        }
    else:
        normalised_env = raw_env
    result = resolve_config(cli, normalised_env, _no_yaml(), _defaults())
    assert result["mode"] == "summary"


def test_resolve_config_ui_mode_takes_priority_over_deprecated_mode() -> None:
    """When both env vars are set, TRACEML_UI_MODE wins (no double-set case)."""
    cli = _no_cli()
    # If TRACEML_UI_MODE is already set, normalisation leaves env unchanged.
    raw_env = {"TRACEML_UI_MODE": "dashboard", "TRACEML_MODE": "summary"}
    if "TRACEML_UI_MODE" not in raw_env and "TRACEML_MODE" in raw_env:
        normalised_env = {
            **raw_env,
            "TRACEML_UI_MODE": raw_env["TRACEML_MODE"],
        }
    else:
        normalised_env = raw_env
    result = resolve_config(cli, normalised_env, _no_yaml(), _defaults())
    assert result["mode"] == "dashboard"


# dashboard_port / dashboard_auto_open  (TRA-68 config keys)


def test_load_yaml_config_dashboard_keys(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """\
        dashboard_port: 9000
        dashboard_auto_open: false
        """,
    )
    result = load_yaml_config(p)
    assert result["dashboard_port"] == 9000
    assert result["dashboard_auto_open"] is False


def test_load_yaml_config_dashboard_port_type_error(tmp_path: Path) -> None:
    p = _write(tmp_path, "dashboard_port: not_a_port\n")
    with pytest.raises(ValueError, match="dashboard_port"):
        load_yaml_config(p)


def test_resolve_config_dashboard_defaults() -> None:
    result = resolve_config(_no_cli(), _no_env(), _no_yaml(), _defaults())
    assert result["dashboard_port"] == 8765
    assert result["dashboard_auto_open"] is True


def test_resolve_config_dashboard_env_coercion() -> None:
    env = {
        "TRACEML_DASHBOARD_PORT": "9000",
        "TRACEML_DASHBOARD_AUTO_OPEN": "0",
    }
    result = resolve_config(_no_cli(), env, _no_yaml(), _defaults())
    assert result["dashboard_port"] == 9000
    assert result["dashboard_auto_open"] is False


def test_resolve_config_dashboard_cli_beats_env_and_yaml() -> None:
    cli = {**_no_cli(), "dashboard_port": 7000}
    env = {"TRACEML_DASHBOARD_PORT": "9000"}
    yaml = {"dashboard_port": 8000}
    result = resolve_config(cli, env, yaml, _defaults())
    assert result["dashboard_port"] == 7000
