import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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

from traceml_ai.runtime.executor import (
    build_runtime_settings,
    extract_script_args,
    read_traceml_env,
    run_user_script,
    write_user_error_log,
)
from traceml_ai.runtime.settings import DEFAULT_FINALIZE_TIMEOUT_SEC


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
                f"Path = __import__('pathlib').Path",
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
                f"Path = __import__('pathlib').Path",
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


def test_build_runtime_settings_carries_trace_max_steps():
    settings = build_runtime_settings(
        {
            "mode": "summary",
            "profile": "run",
            "interval": 1.0,
            "enable_logging": False,
            "logs_dir": "./logs",
            "session_id": "test",
            "summary_window_rows": 200,
            "trace_max_steps": 5,
            "aggregator_host": "127.0.0.1",
            "aggregator_bind_host": "127.0.0.1",
            "aggregator_port": 29765,
        }
    )

    assert settings.trace_max_steps == 5
    assert settings.finalize_timeout_sec == DEFAULT_FINALIZE_TIMEOUT_SEC
    assert settings.expected_world_size == 1


def test_write_user_error_log_records_error(tmp_path):
    cfg = {"logs_dir": str(tmp_path), "session_id": "session-a"}
    error = RuntimeError("boom")

    write_user_error_log(cfg, "User script failed", error)

    log_text = (tmp_path / "session-a" / "torchrun_error.log").read_text(
        encoding="utf-8"
    )
    assert "User script failed" in log_text
    assert "RuntimeError: boom" in log_text
