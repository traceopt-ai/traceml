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

from traceml.runtime.executor import run_user_script


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
