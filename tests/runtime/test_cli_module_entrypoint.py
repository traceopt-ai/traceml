# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""``python -m traceml_ai.launcher.cli`` must run the CLI.

Regression test for the missing ``__main__`` guard, which made the module
form import and exit 0 without running anything, silently swallowing the
user's command.
"""

import os
import subprocess
import sys
from pathlib import Path

import traceml_ai

_SRC_ROOT = str(Path(traceml_ai.__file__).resolve().parents[1])


def _run_module(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC_ROOT
    return subprocess.run(
        [sys.executable, "-m", "traceml_ai.launcher.cli", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_module_form_prints_usage():
    result = _run_module("--help")

    assert result.returncode == 0, result.stderr
    assert "usage: " in result.stdout
    for command in ("watch", "run", "compare", "view"):
        assert command in result.stdout


def test_module_form_reports_an_unknown_command():
    result = _run_module("definitely-not-a-command")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
