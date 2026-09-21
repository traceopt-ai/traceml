# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the launcher-owned Hugging Face GPU reproduction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dev.repro.hf_accelerate_h2d_window import (
    _build_launch_command,
    _read_summary,
)


def test_build_launch_command_gives_runtime_sole_telemetry_ownership(
    tmp_path: Path,
) -> None:
    command = _build_launch_command(logs_dir=tmp_path, run_name="hf-check")

    assert command[:4] == [
        sys.executable,
        "-m",
        "traceml_ai.launcher.cli",
        "run",
    ]
    assert command[-2:] == ["--args", "--workload"]
    assert "--logs-dir" in command
    assert "hf-check" in command


def test_read_summary_uses_finalized_public_measurements(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "final_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "step_time": {
                    "metadata": {"training_total_steps": 3},
                    "global": {"average": {"h2d_ms": 0.25}},
                }
            }
        ),
        encoding="utf-8",
    )

    measurement = _read_summary(summary_path)

    assert measurement.h2d_ms == 0.25
    assert measurement.completed_steps == 3


def test_read_summary_preserves_unavailable_measurements(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "final_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "step_time": {
                    "metadata": {"training_total_steps": 3},
                    "global": {"average": {"h2d_ms": None}},
                }
            }
        ),
        encoding="utf-8",
    )

    measurement = _read_summary(summary_path)

    assert measurement.h2d_ms is None
    assert measurement.completed_steps == 3
