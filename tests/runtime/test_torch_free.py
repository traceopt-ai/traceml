# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""TraceML must install and monitor without torch.

``pip install traceml-ai`` without the ``[torch]`` extra has to support
``traceml watch`` end to end. Step instrumentation genuinely needs torch,
so it must fail with an actionable message rather than an import crash.

The test environment has torch installed, so each case runs in a
subprocess with torch blocked (``sys.modules["torch"] = None`` makes any
``import torch`` raise ``ModuleNotFoundError`` with ``name == "torch"``).
"""

import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

import traceml_ai
from traceml_ai.launcher.launch_config import (
    TORCH_LAUNCHER_REQUIRED,
    TorchrunLaunchConfig,
)
from traceml_ai.utils.torch_support import is_missing_torch, torch_available

_SRC_ROOT = str(Path(traceml_ai.__file__).resolve().parents[1])

_BLOCK_TORCH = "import sys; sys.modules['torch'] = None; "

_PUBLIC_INSTRUMENTATION_API = (
    "trace_step",
    "wrap_dataloader_fetch",
    "wrap_forward",
    "wrap_backward",
    "wrap_optimizer",
    "wrap_h2d",
)


def _run_torch_free(code: str) -> subprocess.CompletedProcess:
    """Run one snippet in a subprocess that cannot import torch."""
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC_ROOT
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH + code],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def _torch_free_env(tmp_path: Path) -> dict:
    """Return an environment where every child process also lacks torch.

    ``sys.modules`` edits do not survive into spawned processes, and the
    launcher spawns both an aggregator and a training process, so the
    block has to travel through ``PYTHONPATH`` as a sitecustomize module.
    """
    blocker = tmp_path / "torch_blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(
        "import sys\nsys.modules['torch'] = None\n", encoding="utf-8"
    )
    # Drop inherited TRACEML_* settings so the run is not steered by the
    # ambient environment (log location in particular).
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TRACEML_")
    }
    env["PYTHONPATH"] = os.pathsep.join([str(blocker), _SRC_ROOT])
    return env


def test_watch_import_chain_survives_without_torch():
    result = _run_torch_free(
        "import traceml_ai.runtime.sampler_registry; "
        "import traceml_ai.reporting.final; "
        "import traceml_ai.sdk; "
        "import traceml_ai.launcher.commands; "
        "print('IMPORTS_OK')"
    )
    assert result.returncode == 0, result.stderr
    assert "IMPORTS_OK" in result.stdout


def test_watch_profile_does_not_import_torch_backed_samplers():
    """`watch` never needs step timing, so it must not reach torch at all."""
    result = _run_torch_free(
        "import sys; "
        "import traceml_ai.runtime.sampler_registry as reg; "
        "specs = reg.select_sampler_specs("
        "profile='watch', mode='summary', is_ddp=False, local_rank=0); "
        "assert 'traceml_ai.utils.timing' not in sys.modules, "
        "'watch imported the timing module'; "
        "print('KEYS', sorted(s.key for s in specs))"
    )
    assert result.returncode == 0, result.stderr
    assert "step_time" not in result.stdout
    assert "step_memory" not in result.stdout


@pytest.mark.parametrize("name", _PUBLIC_INSTRUMENTATION_API)
def test_public_api_raises_an_actionable_error_without_torch(name):
    """The documented entry points must name the fix, not crash."""
    result = _run_torch_free(
        "import traceml_ai\n"
        f"try:\n"
        f"    traceml_ai.{name}(None)\n"
        "except RuntimeError as exc:\n"
        "    assert 'traceml-ai[torch]' in str(exc), str(exc)\n"
        "    print('ACTIONABLE_OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "ACTIONABLE_OK" in result.stdout


@pytest.mark.parametrize(
    ("module_name", "expected"),
    [
        ("torch", True),
        ("torch.distributed", True),
        ("torchvision", False),
        ("numpy", False),
        (None, False),
    ],
)
def test_only_torch_counts_as_a_missing_torch_install(module_name, expected):
    exc = ImportError("boom")
    exc.name = module_name
    assert is_missing_torch(exc) is expected


def test_torch_available_does_not_mask_an_unrelated_import_error(monkeypatch):
    real_import = builtins.__import__

    def import_with_broken_torch(name, *args, **kwargs):
        if name == "torch":
            exc = ImportError("unrelated dependency is missing")
            exc.name = "unrelated_dependency"
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_broken_torch)

    with pytest.raises(ImportError, match="unrelated dependency"):
        torch_available()


def test_an_unrelated_import_error_is_not_masked():
    """A broken import inside a torch module must still surface.

    Torch is deliberately left importable here: the risk being tested is
    a real regression inside instrumentation being reported as a missing
    torch install and silently replaced by the torch-free fallback.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC_ROOT
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['traceml_ai.utils.timing'] = None\n"
            "try:\n"
            "    import traceml_ai.sdk\n"
            "except ImportError as exc:\n"
            "    assert exc.name == 'traceml_ai.utils.timing', exc.name\n"
            "    print('PROPAGATED_OK')\n"
            "else:\n"
            "    raise AssertionError('the broken import was swallowed')\n",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "PROPAGATED_OK" in result.stdout


def test_single_process_launch_command_falls_back_without_torch():
    result = _run_torch_free(
        "import sys as s; "
        "from traceml_ai.launcher.launch_config import "
        "TorchrunLaunchConfig; "
        "cmd = TorchrunLaunchConfig(nnodes=1, nproc_per_node=1, "
        "node_rank=0, master_addr='127.0.0.1', "
        "master_port=29500).to_command(); "
        "assert cmd == [s.executable], cmd; "
        "print('FALLBACK_OK')"
    )
    assert result.returncode == 0, result.stderr
    assert "FALLBACK_OK" in result.stdout


def test_run_requires_torch_before_launching_processes():
    result = _run_torch_free(
        "import argparse\n"
        "from traceml_ai.launcher.commands import validate_launch_args\n"
        "args = argparse.Namespace(command='run', script='x.py', "
        "nnodes=1, nproc_per_node=1, node_rank=0, "
        "master_addr='127.0.0.1', master_port=29500, "
        "aggregator_host=None, aggregator_bind_host=None, "
        "aggregator_port=29765, disable_traceml=False)\n"
        "try:\n"
        "    validate_launch_args(args)\n"
        "except SystemExit as exc:\n"
        "    assert 'step-aware diagnosis' in str(exc), str(exc)\n"
        "    assert 'traceml-ai[torch]' in str(exc), str(exc)\n"
        "    print('RUN_REJECTED_OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "RUN_REJECTED_OK" in result.stdout


@pytest.mark.parametrize(
    ("nnodes", "nproc_per_node"), [(1, 2), (2, 1), (2, 4)]
)
def test_multi_process_is_rejected_before_anything_starts(
    nnodes, nproc_per_node
):
    """Validation must reject the topology, not a later spawn attempt."""
    result = _run_torch_free(
        "import argparse\n"
        "from traceml_ai.launcher.commands import validate_launch_args\n"
        "args = argparse.Namespace(script='x.py', "
        f"nnodes={nnodes}, nproc_per_node={nproc_per_node}, node_rank=0, "
        "master_addr='127.0.0.1', master_port=29500, run_name='r', "
        "session_id='', mode=None, no_history=False, html_report=False, "
        "summary_window_rows=10000, finalize_timeout_sec=300.0, "
        "trace_max_steps=None, interval=2.0, disable_traceml=False, "
        "aggregator_host=None, aggregator_bind_host=None, "
        "aggregator_port=29765)\n"
        "try:\n"
        "    validate_launch_args(args)\n"
        "except SystemExit as exc:\n"
        "    assert 'traceml-ai[torch]' in str(exc), str(exc)\n"
        "    print('REJECTED_OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "REJECTED_OK" in result.stdout


def test_multi_process_topology_reports_its_requirement():
    cfg = TorchrunLaunchConfig(nnodes=1, nproc_per_node=2)
    assert cfg.requires_torch_launcher() is True
    assert "traceml-ai[torch]" in TORCH_LAUNCHER_REQUIRED

    single = TorchrunLaunchConfig(nnodes=1, nproc_per_node=1)
    assert single.requires_torch_launcher() is False


def test_watch_runs_end_to_end_without_torch(tmp_path):
    """The whole command must work, not merely import.

    This is the case a user actually hits: install without the extra,
    point `traceml watch` at a script, and get a summary card.
    """
    script = tmp_path / "tiny_train.py"
    script.write_text("print('training done')\n", encoding="utf-8")
    logs_dir = tmp_path / "logs"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['traceml', 'watch', "
            f"{str(script)!r}, '--mode=summary', "
            f"'--logs-dir', {str(logs_dir)!r}, "
            "'--run-name', 'torch_free_e2e']; "
            "from traceml_ai.launcher.cli import main; main()",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(tmp_path),
        env=_torch_free_env(tmp_path),
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "training done" in result.stdout
    assert "TraceML Watch Summary" in result.stdout
    assert "Host health" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr

    summary = logs_dir / "torch_free_e2e" / "final_summary.json"
    assert summary.is_file(), "watch wrote no summary artifact"
