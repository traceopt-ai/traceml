import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _StubEncoder:
    def encode(self, payload):
        return b""


class _StubDecoder:
    def decode(self, payload):
        return {}


sys.modules.setdefault(
    "msgspec",
    types.SimpleNamespace(
        msgpack=types.SimpleNamespace(
            Encoder=_StubEncoder,
            Decoder=_StubDecoder,
            encode=lambda payload: b"",
        )
    ),
)

import traceml_ai.runtime.executor as executor  # noqa: E402
from traceml_ai.runtime.executor import (  # noqa: E402
    build_runtime_settings,
    extract_script_args,
    read_traceml_env,
    run_user_script,
)
from traceml_ai.runtime.settings import (
    DEFAULT_FINALIZE_TIMEOUT_SEC,
    TraceMLSettings,
)  # noqa: E402


def test_extract_script_args_uses_separator_when_present(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["executor.py", "--", "--epochs", "2"],
    )

    assert extract_script_args() == ["--epochs", "2"]


def test_extract_script_args_keeps_args_when_torchrun_strips_separator(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["executor.py", "--epochs", "2"],
    )

    assert extract_script_args() == ["--epochs", "2"]


def test_run_user_script_adds_script_dir_to_sys_path(tmp_path, monkeypatch):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    script_dir = workspace_dir / "project"
    script_dir.mkdir()

    (script_dir / "helper_module.py").write_text(
        "VALUE = 'available'\n", encoding="utf-8"
    )
    output_path = workspace_dir / "result.json"
    script_path = script_dir / "train.py"
    script_path.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "from helper_module import VALUE",
                "Path = __import__('pathlib').Path",
                f"Path({str(output_path)!r}).write_text(",
                "    json.dumps({'value': VALUE, 'path0': sys.path[0]}),",
                "    encoding='utf-8',",
                ")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(workspace_dir)

    run_user_script(str(script_path), ["--epochs", "1"])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["value"] == "available"
    assert payload["path0"] == str(script_dir.resolve())


def test_run_user_script_restores_sys_argv_and_sys_path(tmp_path, monkeypatch):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    script_dir = workspace_dir / "project"
    script_dir.mkdir()

    output_path = workspace_dir / "argv.txt"
    script_path = script_dir / "train.py"
    script_path.write_text(
        "\n".join(
            [
                "import sys",
                "Path = __import__('pathlib').Path",
                f"Path({str(output_path)!r}).write_text('|'.join(sys.argv), encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(workspace_dir)
    original_argv = sys.argv[:]
    original_path = sys.path[:]

    run_user_script(str(script_path), ["--epochs", "2"])

    assert output_path.read_text(encoding="utf-8").endswith("--epochs|2")
    assert sys.argv == original_argv
    assert sys.path == original_path


def test_read_traceml_env_parses_trace_max_steps(monkeypatch):
    monkeypatch.setenv("TRACEML_SCRIPT_PATH", "train.py")
    monkeypatch.setenv("TRACEML_TRACE_MAX_STEPS", "123")
    monkeypatch.setenv("TRACEML_FINALIZE_TIMEOUT_SEC", "42.5")
    monkeypatch.setenv("TRACEML_EXPECTED_WORLD_SIZE", "8")

    cfg = read_traceml_env()

    assert cfg["trace_max_steps"] == 123
    assert cfg["finalize_timeout_sec"] == 42.5
    assert cfg["expected_world_size"] == 8


def test_read_traceml_env_defaults_to_two_second_interval(monkeypatch):
    monkeypatch.setenv("TRACEML_SCRIPT_PATH", "train.py")
    monkeypatch.delenv("TRACEML_INTERVAL", raising=False)

    assert read_traceml_env()["interval"] == 2.0


def test_read_traceml_env_defaults_to_summary(monkeypatch):
    monkeypatch.setenv("TRACEML_SCRIPT_PATH", "train.py")
    monkeypatch.delenv("TRACEML_UI_MODE", raising=False)
    monkeypatch.delenv("TRACEML_MODE", raising=False)

    assert read_traceml_env()["mode"] == "summary"


def test_trace_settings_default_to_two_second_cadences():
    settings = TraceMLSettings()

    assert settings.mode == "summary"
    assert settings.sampler_interval_sec == 2.0
    assert settings.render_interval_sec == 2.0


def test_build_runtime_settings_carries_trace_max_steps():
    settings = build_runtime_settings(
        {
            "mode": "summary",
            "profile": "run",
            "interval": 1.0,
            "enable_logging": False,
            "logs_dir": "./logs",
            "session_id": "test",
            "history_retention_s": 1800.0,
            "trace_max_steps": 5,
            "aggregator_host": "127.0.0.1",
            "aggregator_bind_host": "127.0.0.1",
            "aggregator_port": 29765,
        }
    )

    assert settings.trace_max_steps == 5
    assert settings.finalize_timeout_sec == DEFAULT_FINALIZE_TIMEOUT_SEC
    assert settings.expected_world_size == 1


def test_runtime_start_and_stop_failures_use_internal_log(monkeypatch):
    failures = []
    startup_error = RuntimeError("startup failed")
    shutdown_error = RuntimeError("shutdown failed")
    runtime = Mock()
    runtime.stop.side_effect = shutdown_error

    monkeypatch.setattr(
        executor, "build_runtime_settings", lambda _cfg: object()
    )
    monkeypatch.setattr(
        executor,
        "start_runtime_handle",
        Mock(side_effect=startup_error),
    )
    monkeypatch.setattr(
        executor,
        "_log_runtime_exception",
        lambda message, error: failures.append((message, error)),
    )

    assert isinstance(executor.start_runtime({}), executor.NoOpRuntime)
    executor.stop_runtime(runtime)

    assert failures == [
        ("Failed to start TraceMLRuntime", startup_error),
        ("Error during TraceML runtime shutdown", shutdown_error),
    ]


@pytest.mark.parametrize(
    "user_exit",
    [SystemExit(7), KeyboardInterrupt()],
)
def test_execute_with_runtime_preserves_base_exceptions_without_user_log(
    monkeypatch,
    user_exit,
):
    cfg = {"script_path": "train.py"}
    events = []

    monkeypatch.setattr(executor, "read_traceml_env", lambda: cfg)
    monkeypatch.setattr(executor, "start_runtime", lambda value: object())
    monkeypatch.setattr(executor, "extract_script_args", lambda: [])

    def exit_user_script(*_args):
        raise user_exit

    monkeypatch.setattr(executor, "run_user_script", exit_user_script)
    monkeypatch.setattr(
        executor,
        "stop_runtime",
        lambda *_args: events.append("stopped"),
    )

    with pytest.raises(type(user_exit)) as exc_info:
        executor._execute_with_runtime()

    assert exc_info.value is user_exit
    assert events == ["stopped"]


def test_torchelastic_record_wraps_entrypoint_when_configured(monkeypatch):
    calls = []

    def fake_record(entrypoint):
        def wrapped():
            calls.append("record")
            return entrypoint()

        return wrapped

    monkeypatch.setenv("TORCHELASTIC_ERROR_FILE", "error.json")
    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.elastic.multiprocessing.errors",
        types.SimpleNamespace(record=fake_record),
    )

    executor._run_with_torchelastic_record(lambda: calls.append("entrypoint"))

    assert calls == ["record", "entrypoint"]


def test_executor_user_failure_stays_native_and_out_of_internal_logs(
    tmp_path,
):
    script_path = tmp_path / "raise_error.py"
    script_path.write_text(
        "raise RuntimeError('subprocess boom')\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.pop("TORCHELASTIC_ERROR_FILE", None)
    env.update(
        {
            "PYTHONPATH": str(SRC),
            "TRACEML_SCRIPT_PATH": str(script_path),
            "TRACEML_DISABLED": "1",
            "TRACEML_LOGS_DIR": str(tmp_path / "logs"),
            "TRACEML_SESSION_ID": "executor-test",
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "traceml_ai.runtime.executor"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )

    assert result.returncode == 1
    assert "Traceback (most recent call last)" in result.stderr
    assert "RuntimeError: subprocess boom" in result.stderr
    run_root = tmp_path / "logs" / "executor-test"
    assert not (run_root / "torchrun_error.log").exists()
    assert not (run_root / "runtime_error.log").exists()
    assert not list(run_root.rglob("traceml_errors.log"))
